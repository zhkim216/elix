import gc
from pathlib import Path

import hydra
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


def _evaluate_af3_for_checkpoint(
    *,
    cfg: DictConfig,
    sample_dict_per_ckpt: dict,
    log_dir_per_ckpt: Path,
    ckpt_info: dict,
    csv_suffix: str,
    free_atom_arrays_after_self_consistency: bool,
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
            input_sample_is_designed=cfg.input_sample_is_designed,
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
            input_sample_is_designed=cfg.input_sample_is_designed,
            free_atom_arrays_progressively=True,
        )


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

    # Resolve configuration for input samples.
    cif_parse_cfg, preprocess_cfg = resolve_input_cfgs(cfg)

    # Get sampling and guidance configs.
    sampling_cfg = cfg.get("sampling_cfg", None)
    guidance_cfg = sampling_cfg.get("guidance", None)

    print_phase("Phase 2: Redesigning sequence")

    bins_raw = cfg.pocket_cfg.get("pocket_distance_bins", None)
    pocket_distance_bins = [tuple(b) for b in bins_raw] if bins_raw is not None else None

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
        pocket_distance_bins=pocket_distance_bins,
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
        )
        del sample_dict_per_ckpt
        gc.collect()

    print_phase(f"All phases complete! Results saved to {log_dir}")


@hydra.main(config_path="../../configs/eval/sampling", config_name="run_elix", version_base="1.3.2")
def main(cfg: DictConfig):
    run_elix(cfg)


if __name__ == "__main__":
    main()
