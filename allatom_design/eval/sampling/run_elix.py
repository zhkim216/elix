import gc
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.utils.eval_setup_utils import wandb_setup
from allatom_design.eval.utils.folding_utils import (
    evaluate_af3_docking_consistency,
    evaluate_af3_self_consistency,
)
from allatom_design.eval.utils.data_utils import prepare_sample_dict
from allatom_design.eval.utils.misc import print_phase, load_optional_csv
from allatom_design.eval.utils.cfg_utils import resolve_input_cfgs
from allatom_design.eval.utils.design_sequence import iter_design_sequence_per_checkpoint
from allatom_design.eval.utils.design_sequence_two_stage import design_sequence_two_stage


def _evaluate_af3_for_checkpoint(
    *,
    cfg: DictConfig,
    sample_dict_per_ckpt: dict,
    log_dir_per_ckpt: Path,
    ckpt_info: dict,
    csv_suffix: str,
    free_atom_arrays_after_self_consistency: bool,
    input_sample_is_designed: bool,
) -> None:
    ckpt_label = f"step_{ckpt_info['global_step']}_epoch_{ckpt_info['epoch']}"

    if cfg.struct_pred_cfg.evaluate_self_consistency:
        print_phase(f"Phase 3a: AF3 Self-Consistency Evaluation - {ckpt_label}")
        evaluate_af3_self_consistency(
            sample_dict=sample_dict_per_ckpt,
            out_dir=log_dir_per_ckpt,
            struct_pred_cfg=cfg.struct_pred_cfg,
            cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
            preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
            featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
            pocket_cfg=cfg.pocket_cfg,
            no_wandb=cfg.wandb.no_wandb,
            ckpt_info=ckpt_info,
            calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only,
            csv_suffix=csv_suffix,
            input_sample_is_designed=input_sample_is_designed,
            free_atom_arrays_progressively=free_atom_arrays_after_self_consistency,
        )

    if cfg.struct_pred_cfg.evaluate_docking_consistency:
        print_phase(f"Phase 3b: AF3 Docking Consistency Evaluation - {ckpt_label}")
        evaluate_af3_docking_consistency(
            sample_dict=sample_dict_per_ckpt,
            out_dir=log_dir_per_ckpt,
            struct_pred_cfg=cfg.struct_pred_cfg,
            cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
            preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
            featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
            pocket_cfg=cfg.pocket_cfg,
            no_wandb=cfg.wandb.no_wandb,
            ckpt_info=ckpt_info,
            calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only,
            csv_suffix=csv_suffix,
            input_sample_is_designed=input_sample_is_designed,
            free_atom_arrays_progressively=True,
        )


def _two_stage_enabled(cfg: DictConfig) -> bool:
    two_stage_cfg = cfg.get("two_stage", None)
    return two_stage_cfg is not None and two_stage_cfg.get("enabled", False)


def _two_stage_regions(direction: str) -> tuple[str, str, str]:
    if direction == "ps":
        return "pocket", "scaffold", "pocket"
    if direction == "sp":
        return "scaffold", "pocket", "scaffold"
    raise ValueError(f"two_stage.direction must be 'ps' or 'sp', got {direction!r}")


def _stage_cfg(two_stage_cfg: DictConfig, region: str) -> DictConfig:
    cfg = two_stage_cfg.get(region, None)
    if cfg is None:
        raise ValueError(f"two_stage.{region} is required")
    model_cfg = cfg.get("model_cfg", None)
    if model_cfg is None:
        raise ValueError(f"two_stage.{region}.model_cfg is required")
    if cfg.get("sampling_cfg", None) is None:
        raise ValueError(f"two_stage.{region}.sampling_cfg is required")
    if model_cfg.get("model_name", None) is None:
        raise ValueError(f"two_stage.{region}.model_cfg.model_name is required")
    if model_cfg.get("ckpt_path", None) is None:
        if model_cfg.get("denoiser_train_dir", None) is None:
            raise ValueError(
                f"two_stage.{region}.model_cfg.denoiser_train_dir is required "
                f"unless two_stage.{region}.model_cfg.ckpt_path is set"
            )
        if model_cfg.get("ckpt_cfg", None) is None:
            raise ValueError(
                f"two_stage.{region}.model_cfg.ckpt_cfg is required "
                f"unless two_stage.{region}.model_cfg.ckpt_path is set"
            )
    return cfg


def _stage_label(stage_cfg: DictConfig, region: str) -> str:
    label = stage_cfg.get("label", None)
    if label is not None:
        return str(label)
    train_dir = stage_cfg.model_cfg.get("denoiser_train_dir", None)
    if train_dir is None:
        return region
    return Path(str(train_dir)).name


def _copy_stage_design_cfg(cfg: DictConfig, stage_cfg: DictConfig, sample_token_prefix: str) -> DictConfig:
    design_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(design_cfg, False)
    design_cfg.model_cfg = OmegaConf.create(OmegaConf.to_container(stage_cfg.model_cfg, resolve=True))
    design_cfg.sampling_cfg = OmegaConf.create(OmegaConf.to_container(stage_cfg.sampling_cfg, resolve=True))
    if design_cfg.sampling_cfg.get("overrides", None) is None:
        design_cfg.sampling_cfg.overrides = {}
    design_cfg.sampling_cfg.overrides.sample_token_prefix = sample_token_prefix
    return design_cfg


def _pocket_distance_bins(cfg: DictConfig) -> list[tuple[float, float]] | None:
    bins_raw = cfg.pocket_cfg.get("pocket_distance_bins", None)
    return [tuple(b) for b in bins_raw] if bins_raw is not None else None


def _run_single_stage_elix(
    *,
    cfg: DictConfig,
    sample_dict: dict,
    sampling_inputs_df: pd.DataFrame | None,
    pos_constraint_df: pd.DataFrame | None,
    log_dir: Path,
    csv_suffix: str,
) -> None:
    cif_parse_cfg, preprocess_cfg = resolve_input_cfgs(cfg)
    sampling_cfg = cfg.get("sampling_cfg", None)
    guidance_cfg = sampling_cfg.get("guidance", None)

    print_phase("Phase 2: Redesigning sequence")
    ckpt_iter = iter_design_sequence_per_checkpoint(
        seed=cfg.seed,
        input_sample_is_designed=cfg.input_sample_is_designed,
        sample_dict=sample_dict,
        design_cfg=cfg,
        cif_parse_cfg=cif_parse_cfg,
        preprocess_cfg=preprocess_cfg,
        featurizer_cfg=cfg.featurizer_cfg.design,
        cif_save_cfg=cfg.cif_cfg.save,
        sampling_inputs_df=sampling_inputs_df,
        log_dir=log_dir,
        pos_constraint_df=pos_constraint_df,
        protein_only=cfg.get("protein_only", False),
        pocket_distances_for_seq_recovery=cfg.pocket_cfg.pocket_distances_for_seq_recovery,
        pocket_distance_bins=_pocket_distance_bins(cfg),
        pocket_n_min_ligand_atoms_for_seq_recovery=cfg.pocket_cfg.get(
            "n_min_ligand_atoms_for_seq_recovery",
            1,
        ),
        csv_suffix=csv_suffix,
        guidance_cfg=guidance_cfg,
    )

    both_evals = (
        cfg.struct_pred_cfg.evaluate_self_consistency
        and cfg.struct_pred_cfg.evaluate_docking_consistency
    )
    for sample_dict_per_ckpt, log_dir_per_ckpt, ckpt_info in ckpt_iter:
        _evaluate_af3_for_checkpoint(
            cfg=cfg,
            sample_dict_per_ckpt=sample_dict_per_ckpt,
            log_dir_per_ckpt=log_dir_per_ckpt,
            ckpt_info=ckpt_info,
            csv_suffix=csv_suffix,
            free_atom_arrays_after_self_consistency=not both_evals,
            input_sample_is_designed=cfg.input_sample_is_designed,
        )
        del sample_dict_per_ckpt
        gc.collect()


def _run_two_stage_elix(
    *,
    cfg: DictConfig,
    sample_dict: dict,
    sampling_inputs_df: pd.DataFrame | None,
    log_dir: Path,
    csv_suffix: str,
) -> None:
    two_stage_cfg = cfg.two_stage
    if cfg.get("pos_constraint_csv", None) is not None:
        raise ValueError("two_stage mode always runs stage1 as free design; pos_constraint_csv must be null")

    direction = two_stage_cfg.get("direction", None)
    stage1_region, stage2_region, stage2_constraint_type = _two_stage_regions(direction)
    stage1_cfg = _stage_cfg(two_stage_cfg, stage1_region)
    stage2_cfg = _stage_cfg(two_stage_cfg, stage2_region)
    stage1_label = _stage_label(stage1_cfg, stage1_region)
    stage2_label = _stage_label(stage2_cfg, stage2_region)

    stage1_token_prefix = f"twostage_{direction}_{stage1_region[0]}"
    stage2_token_prefix = stage2_region[0]
    stage1_design_cfg = _copy_stage_design_cfg(cfg, stage1_cfg, stage1_token_prefix)
    stage2_design_cfg = _copy_stage_design_cfg(cfg, stage2_cfg, stage2_token_prefix)

    stage1_cif_parse_cfg, stage1_preprocess_cfg = resolve_input_cfgs(cfg)

    print_phase("Phase 2: Two-stage sequence design")
    ckpt_iter = design_sequence_two_stage(
        seed=cfg.seed,
        direction=direction,
        input_sample_is_designed=cfg.input_sample_is_designed,
        sample_dict=sample_dict,
        stage1_design_cfg=stage1_design_cfg,
        stage2_design_cfg=stage2_design_cfg,
        stage1_region=stage1_region,
        stage2_region=stage2_region,
        stage2_constraint_type=stage2_constraint_type,
        stage1_model_label=stage1_label,
        stage2_model_label=stage2_label,
        stage1_cif_parse_cfg=stage1_cif_parse_cfg,
        stage1_preprocess_cfg=stage1_preprocess_cfg,
        stage2_cif_parse_cfg=cfg.cif_cfg.parse.designed_samples,
        stage2_preprocess_cfg=cfg.preprocess_cfg.designed_samples,
        featurizer_cfg=cfg.featurizer_cfg.design,
        cif_save_cfg=cfg.cif_cfg.save,
        stage1_sampling_inputs_df=sampling_inputs_df,
        log_dir=log_dir,
        protein_only=cfg.get("protein_only", False),
        pocket_distances_for_seq_recovery=cfg.pocket_cfg.pocket_distances_for_seq_recovery,
        pocket_distance_bins=_pocket_distance_bins(cfg),
        pocket_n_min_ligand_atoms_for_seq_recovery=cfg.pocket_cfg.get(
            "n_min_ligand_atoms_for_seq_recovery",
            1,
        ),
        csv_suffix=csv_suffix,
        stage1_guidance_cfg=stage1_design_cfg.sampling_cfg.get("guidance", None),
        stage2_guidance_cfg=stage2_design_cfg.sampling_cfg.get("guidance", None),
        pocket_distance=two_stage_cfg.get("pocket_distance", 5.0),
        use_pseudocb_for_pocket_annotation=two_stage_cfg.get(
            "use_pseudocb_for_pocket_annotation",
            False,
        ),
    )

    both_evals = (
        cfg.struct_pred_cfg.evaluate_self_consistency
        and cfg.struct_pred_cfg.evaluate_docking_consistency
    )
    for sample_dict_per_ckpt, log_dir_per_ckpt, ckpt_info, manifest_rows in ckpt_iter:
        for row in manifest_rows:
            row["af3_self_consistency_enabled"] = cfg.struct_pred_cfg.evaluate_self_consistency
            row["af3_docking_consistency_enabled"] = cfg.struct_pred_cfg.evaluate_docking_consistency
        if manifest_rows:
            log_dir_per_ckpt.mkdir(parents=True, exist_ok=True)
            manifest_path = log_dir_per_ckpt / f"twostage_manifest{csv_suffix}.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

        _evaluate_af3_for_checkpoint(
            cfg=cfg,
            sample_dict_per_ckpt=sample_dict_per_ckpt,
            log_dir_per_ckpt=log_dir_per_ckpt,
            ckpt_info=ckpt_info,
            csv_suffix=csv_suffix,
            free_atom_arrays_after_self_consistency=not both_evals,
            input_sample_is_designed=True,
        )
        del sample_dict_per_ckpt
        gc.collect()


def run_elix(cfg: DictConfig) -> None:
    # Setup logging and configuration.
    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Setup logging directory.
    log_dir = Path(
        wandb_setup(
            base_out_dir=cfg.base_out_dir,
            exp_name=cfg.exp_name,
            cfg_dict=cfg_dict,
            **cfg.wandb,
        )
    )

    # Save configuration.
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load sampling inputs and position constraints.
    sampling_inputs_df = load_optional_csv(cfg.sampling_inputs_csv)
    pos_constraint_df = load_optional_csv(cfg.pos_constraint_csv)

    # Prepare input path and ID dictionaries.
    print_phase("Phase 1: Preparing samples")
    sample_dict = prepare_sample_dict(
        cfg=cfg,
        sampling_inputs_df=sampling_inputs_df,
    )

    # Set up array ID and CSV suffix if exists.
    array_id = cfg.pdb_cfg.get("array_id", None)
    csv_suffix = f"_array_{array_id}" if array_id is not None else ""

    if _two_stage_enabled(cfg):
        _run_two_stage_elix(
            cfg=cfg,
            sample_dict=sample_dict,
            sampling_inputs_df=sampling_inputs_df,
            log_dir=log_dir,
            csv_suffix=csv_suffix,
        )
    else:
        _run_single_stage_elix(
            cfg=cfg,
            sample_dict=sample_dict,
            sampling_inputs_df=sampling_inputs_df,
            pos_constraint_df=pos_constraint_df,
            log_dir=log_dir,
            csv_suffix=csv_suffix,
        )

    print_phase(f"All phases complete! Results saved to {log_dir}")


@hydra.main(config_path="../../configs/eval/sampling", config_name="run_elix", version_base="1.3.2")
def main(cfg: DictConfig):
    run_elix(cfg)


if __name__ == "__main__":
    main()
