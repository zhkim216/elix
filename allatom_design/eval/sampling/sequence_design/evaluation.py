from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from allatom_design.eval.structure_prediction.af3_evaluation import (
    evaluate_af3_docking_consistency,
    evaluate_af3_self_consistency,
)
from allatom_design.eval.sampling.sequence_design.checkpoints import ckpt_label
from allatom_design.eval.run_logging import print_phase


def pocket_distance_bins(cfg: DictConfig) -> list[tuple[float, float]] | None:
    bins_raw = cfg.pocket_cfg.get("pocket_distance_bins", None)
    return [tuple(b) for b in bins_raw] if bins_raw is not None else None


def evaluate_af3_for_checkpoint(
    *,
    cfg: DictConfig,
    sample_dict_per_ckpt: dict,
    log_dir_per_ckpt: Path,
    ckpt_info: dict,
    csv_suffix: str,
    free_atom_arrays_after_self_consistency: bool,
    input_sample_is_designed: bool,
) -> None:
    label = ckpt_label(ckpt_info)

    if cfg.struct_pred_cfg.evaluate_self_consistency:
        print_phase(f"Phase 3a: AF3 Self-Consistency Evaluation - {label}")
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
        print_phase(f"Phase 3b: AF3 Docking Consistency Evaluation - {label}")
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
