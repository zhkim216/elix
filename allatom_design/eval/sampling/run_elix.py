import gc
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.run_logging import wandb_setup
from allatom_design.eval.run_logging import print_phase
from allatom_design.eval.sampling.sequence_design.config import resolve_input_cfgs
from allatom_design.eval.sampling.sequence_design.inputs import prepare_sample_dict
from allatom_design.eval.utils.table_io import load_optional_csv
from allatom_design.eval.utils.sampling_inputs import (
    merge_role_pos_constraints,
    normalize_role_sampling_inputs_df,
)
from allatom_design.eval.sampling.sequence_design.evaluation import (
    evaluate_af3_for_checkpoint,
    pocket_distance_bins,
)
from allatom_design.eval.sampling.sequence_design.core import (
    build_sequence_design_run_spec_from_cfg,
    iter_design_sequence_for_run_spec,
)
from allatom_design.eval.sampling.sequence_design.two_stage import (
    build_two_stage_design_context,
    design_sequence_two_stage,
)
from allatom_design.eval.metrics.sequence_recovery import build_sequence_recovery_metric_config


def _two_stage_enabled(cfg: DictConfig) -> bool:
    two_stage_cfg = cfg.get("two_stage", None)
    return two_stage_cfg is not None and two_stage_cfg.get("enabled", False)


def _selectivity_guidance_enabled(cfg: DictConfig) -> bool:
    guidance_cfg = cfg.get("sampling_cfg", {}).get("guidance", None)
    if guidance_cfg is None:
        return False
    return bool(guidance_cfg.get("enabled", False)) and str(guidance_cfg.get("mode", "")) == "selectivity"


def _reject_selectivity_guidance(cfg: DictConfig) -> None:
    if not _selectivity_guidance_enabled(cfg):
        return
    raise NotImplementedError(
        "guidance.mode=selectivity is owned by "
        "allatom_design/eval/sampling/run_selectivity_elix.py; "
        "run_elix.py only supports standard, two-stage, and ensemble sampling"
    )


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
    run_spec = build_sequence_design_run_spec_from_cfg(
        cfg=cfg,
        sample_dict=sample_dict,
        cif_parse_cfg=cif_parse_cfg,
        preprocess_cfg=preprocess_cfg,
        sampling_inputs_df=sampling_inputs_df,
        selectivity_residue_alignment_df=None,
        log_dir=log_dir,
        pos_constraint_df=pos_constraint_df,
        pocket_distance_bins=pocket_distance_bins(cfg),
        csv_suffix=csv_suffix,
    )

    print_phase("Phase 2: Redesigning sequence")
    ckpt_iter = iter_design_sequence_for_run_spec(run_spec)

    both_evals = (
        cfg.struct_pred_cfg.evaluate_self_consistency
        and cfg.struct_pred_cfg.evaluate_docking_consistency
    )
    for sample_dict_per_ckpt, log_dir_per_ckpt, ckpt_info in ckpt_iter:
        evaluate_af3_for_checkpoint(
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
    two_stage_context = build_two_stage_design_context(cfg)
    stage1_cif_parse_cfg, stage1_preprocess_cfg = resolve_input_cfgs(cfg)

    print_phase("Phase 2: Two-stage sequence design")
    ckpt_iter = design_sequence_two_stage(
        seed=cfg.seed,
        direction=two_stage_context.direction,
        input_sample_is_designed=cfg.input_sample_is_designed,
        sample_dict=sample_dict,
        stage1_design_cfg=two_stage_context.stage1_design_cfg,
        stage2_design_cfg=two_stage_context.stage2_design_cfg,
        stage1_region=two_stage_context.stage1_region,
        stage2_region=two_stage_context.stage2_region,
        stage2_constraint_type=two_stage_context.stage2_constraint_type,
        stage1_model_label=two_stage_context.stage1_model_label,
        stage2_model_label=two_stage_context.stage2_model_label,
        stage1_cif_parse_cfg=stage1_cif_parse_cfg,
        stage1_preprocess_cfg=stage1_preprocess_cfg,
        stage2_cif_parse_cfg=cfg.cif_cfg.parse.designed_samples,
        stage2_preprocess_cfg=cfg.preprocess_cfg.designed_samples,
        featurizer_cfg=cfg.featurizer_cfg.design,
        cif_save_cfg=cfg.cif_cfg.save,
        stage1_sampling_inputs_df=sampling_inputs_df,
        log_dir=log_dir,
        protein_only=cfg.get("protein_only", False),
        sequence_recovery_metric_config=build_sequence_recovery_metric_config(
            pocket_cfg=cfg.pocket_cfg,
            input_sample_is_designed=cfg.input_sample_is_designed,
            pocket_distance_bins=pocket_distance_bins(cfg),
        ),
        csv_suffix=csv_suffix,
        stage1_guidance_cfg=two_stage_context.stage1_guidance_cfg,
        stage2_guidance_cfg=two_stage_context.stage2_guidance_cfg,
        pocket_distance=two_stage_context.pocket_distance,
        pocket_annotation_method=two_stage_context.pocket_annotation_method,
        use_calpha_for_pocket_annotation=two_stage_context.use_calpha_for_pocket_annotation,
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

        evaluate_af3_for_checkpoint(
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


def load_sampling_runtime_tables(cfg: DictConfig) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    sampling_inputs_df = load_optional_csv(cfg.get("sampling_inputs_csv", None))
    sampling_inputs_df = normalize_role_sampling_inputs_df(sampling_inputs_df)
    pos_constraint_df = load_optional_csv(cfg.get("pos_constraint_csv", None))
    pos_constraint_df = merge_role_pos_constraints(
        pos_constraint_df=pos_constraint_df,
        sampling_inputs_df=sampling_inputs_df,
    )
    if _two_stage_enabled(cfg) and pos_constraint_df is not None:
        raise ValueError(
            "two_stage mode creates stage2 positional constraints internally; "
            "role-schema fixed_pos_* columns and pos_constraint_csv are not supported "
            "for two-stage stage1 inputs"
        )
    return sampling_inputs_df, pos_constraint_df


def run_elix(cfg: DictConfig) -> None:
    _reject_selectivity_guidance(cfg)

    # Setup logging and configuration.
    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    array_id = cfg.pdb_cfg.get("array_id", None)

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
    config_filename = f"config_array_{array_id}.yaml" if array_id is not None else "config.yaml"
    with open(Path(log_dir, config_filename), "w") as f:
        yaml.safe_dump(cfg_dict, f)

    # Load sampling inputs and position constraints.
    sampling_inputs_df, pos_constraint_df = load_sampling_runtime_tables(cfg)

    # Prepare input path and ID dictionaries.
    print_phase("Phase 1: Preparing samples")
    sample_dict = prepare_sample_dict(
        cfg=cfg,
        sampling_inputs_df=sampling_inputs_df,
    )

    # Set up array ID and CSV suffix if exists.
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
