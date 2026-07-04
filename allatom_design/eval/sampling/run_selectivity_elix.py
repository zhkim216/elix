import gc
from pathlib import Path

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.run_logging import wandb_setup
from allatom_design.eval.sampling.sequence_design.evaluation import (
    evaluate_af3_for_checkpoint,
    pocket_distance_bins,
)
from allatom_design.eval.sampling.sequence_design.selectivity_runtime import (
    build_selectivity_sampling_runtime_plan,
)
from allatom_design.eval.sampling.sequence_design.selectivity_inputs import (
    load_selectivity_inputs,
    reject_selectivity_two_stage,
    require_selectivity_guidance_cfg,
    residue_alignment_for_selected_rows,
    selectivity_rows_for_cfg,
    selectivity_sample_dict,
)
from allatom_design.eval.sampling.sequence_design.config import resolve_input_cfgs
from allatom_design.eval.sampling.sequence_design.core import (
    build_sequence_design_run_spec_from_cfg,
    iter_design_sequence_for_run_spec,
)
from allatom_design.eval.run_logging import print_phase
from allatom_design.eval.utils.table_io import load_optional_csv


def _run_selectivity_sequence_design(
    *,
    cfg: DictConfig,
    sample_dict: dict,
    sampling_inputs_df: pd.DataFrame,
    residue_alignment_df: pd.DataFrame,
    pos_constraint_df: pd.DataFrame | None,
    log_dir: Path,
    csv_suffix: str,
) -> None:
    cif_parse_cfg, preprocess_cfg = resolve_input_cfgs(cfg)
    guidance_cfg = require_selectivity_guidance_cfg(cfg)

    def runtime_plan_builder(**kwargs):
        return build_selectivity_sampling_runtime_plan(
            sampling_inputs_df=sampling_inputs_df,
            selectivity_residue_alignment_df=residue_alignment_df,
            pos_constraint_df=pos_constraint_df,
            guidance_cfg=guidance_cfg,
            **kwargs,
        )

    run_spec = build_sequence_design_run_spec_from_cfg(
        cfg=cfg,
        sample_dict=sample_dict,
        cif_parse_cfg=cif_parse_cfg,
        preprocess_cfg=preprocess_cfg,
        sampling_inputs_df=sampling_inputs_df,
        selectivity_residue_alignment_df=residue_alignment_df,
        log_dir=log_dir,
        pos_constraint_df=pos_constraint_df,
        pocket_distance_bins=pocket_distance_bins(cfg),
        csv_suffix=csv_suffix,
        guidance_cfg=guidance_cfg,
    )

    print_phase("Phase 2: Paired selectivity sequence design")
    ckpt_iter = iter_design_sequence_for_run_spec(
        run_spec,
        runtime_plan_builder=runtime_plan_builder,
    )

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


def run_selectivity_elix(cfg: DictConfig) -> None:
    reject_selectivity_two_stage(cfg)
    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    array_id = cfg.pdb_cfg.get("array_id", None)

    log_dir = Path(
        wandb_setup(
            base_out_dir=cfg.base_out_dir,
            exp_name=cfg.exp_name,
            cfg_dict=cfg_dict,
            **cfg.wandb,
        )
    )

    config_filename = f"config_array_{array_id}.yaml" if array_id is not None else "config.yaml"
    with open(Path(log_dir, config_filename), "w") as handle:
        yaml.safe_dump(cfg_dict, handle)

    sampling_inputs_df, residue_alignment_df = load_selectivity_inputs(cfg)
    selected_sampling_inputs_df = selectivity_rows_for_cfg(cfg, sampling_inputs_df)
    selected_pair_ids = selected_sampling_inputs_df["selectivity_pair_id"].astype(str).unique().tolist()
    selected_residue_alignment_df = residue_alignment_for_selected_rows(
        residue_alignment_df,
        selected_sampling_inputs_df,
    )
    pos_constraint_df = load_optional_csv(cfg.pos_constraint_csv)

    print_phase("Phase 1: Preparing paired selectivity samples")
    sample_dict = selectivity_sample_dict(cfg, selected_sampling_inputs_df)
    print(
        "Selected selectivity pairs: "
        f"{len(selected_pair_ids)} pairs, {len(selected_sampling_inputs_df)} endpoint rows"
    )

    csv_suffix = f"_array_{array_id}" if array_id is not None else ""
    _run_selectivity_sequence_design(
        cfg=cfg,
        sample_dict=sample_dict,
        sampling_inputs_df=selected_sampling_inputs_df,
        residue_alignment_df=selected_residue_alignment_df,
        pos_constraint_df=pos_constraint_df,
        log_dir=log_dir,
        csv_suffix=csv_suffix,
    )

    print_phase(f"All phases complete! Results saved to {log_dir}")


@hydra.main(config_path="../../configs/eval/sampling", config_name="guidance/elix_selectivity_guidance", version_base="1.3.2")
def main(cfg: DictConfig):
    run_selectivity_elix(cfg)


if __name__ == "__main__":
    main()
