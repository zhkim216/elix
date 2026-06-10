import copy
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from omegaconf import DictConfig

from allatom_design.eval.utils.cfg_utils import get_stage2_potts_only_cond
from allatom_design.eval.utils.constraint_utils import (
    create_pos_constraint_dict_from_pocket,
    resolve_pocket_annotation_method,
)
from allatom_design.eval.utils.eval_setup_utils import ckpt_label
from allatom_design.eval.utils.design_sequence import iter_design_sequence_per_checkpoint


def _build_stage2_inputs_and_constraints(
    *,
    stage1_sample_dict_per_ckpt: dict,
    stage1_ckpt_info: dict[str, Any],
    stage1_log_dir_per_ckpt: Path,
    stage1_region: str,
    stage2_region: str,
    stage2_constraint_type: str,
    stage1_model_label: str,
    pocket_distance: float,
    pocket_annotation_method: str | None,
    use_calpha_for_pocket_annotation: bool,
) -> tuple[dict, pd.DataFrame, dict[str, dict[str, Any]]]:
    stage2_sample_dict: dict[str, dict[str, Any]] = {}
    constraint_rows = []
    lineage_by_stage1_sample_id: dict[str, dict[str, Any]] = {}

    for original_sample_id, stage1_entry in stage1_sample_dict_per_ckpt.items():
        designed_sample_ids = stage1_entry.get("designed_sample_id", [])
        designed_sample_paths = stage1_entry.get("designed_sample_path", [])
        designed_atom_arrays = stage1_entry.get("designed_sample_atom_array", [])
        designed_sample_seqs = stage1_entry.get("designed_sample_seq", [])

        for sample_idx, stage1_sample_id in enumerate(designed_sample_ids):
            stage1_sample_path = designed_sample_paths[sample_idx]
            stage1_atom_array = designed_atom_arrays[sample_idx]
            stage1_sample_seq = (
                designed_sample_seqs[sample_idx]
                if sample_idx < len(designed_sample_seqs)
                else None
            )

            constraint_row, _ = create_pos_constraint_dict_from_pocket(
                pdb_key=stage1_sample_id,
                atom_array=stage1_atom_array,
                pocket_distance=pocket_distance,
                constraint_type=stage2_constraint_type,
                pocket_annotation_method=pocket_annotation_method,
                use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
                sample_path=stage1_sample_path,
                return_ligand_mpnn_format=False,
            )
            constraint_rows.append(constraint_row)

            stage2_sample_dict[stage1_sample_id] = {
                "input_sample_path": stage1_sample_path,
                "input_sample_id": stage1_sample_id,
            }
            if "pdb_chain_info" in stage1_entry:
                stage2_sample_dict[stage1_sample_id]["pdb_chain_info"] = copy.deepcopy(
                    stage1_entry["pdb_chain_info"]
                )
            lineage_by_stage1_sample_id[stage1_sample_id] = {
                "stage1_region": stage1_region,
                "stage2_region": stage2_region,
                "original_sample_id": original_sample_id,
                "original_sample_path": stage1_entry.get("input_sample_path"),
                "stage1_model_label": stage1_model_label,
                "stage1_step": stage1_ckpt_info["global_step"],
                "stage1_epoch": stage1_ckpt_info["epoch"],
                "stage1_checkpoint_path": stage1_ckpt_info["ckpt_path"],
                "stage1_checkpoint_dir": str(stage1_log_dir_per_ckpt),
                "stage1_sample_id": stage1_sample_id,
                "stage1_sample_path": stage1_sample_path,
                "stage1_sample_seq": stage1_sample_seq,
                "fixed_region": stage2_constraint_type,
                "pocket_distance": pocket_distance,
                "pocket_annotation_method": pocket_annotation_method,
            }

    if len(stage2_sample_dict) == 0:
        raise ValueError("Stage 1 produced no samples for stage 2")

    runtime_constraint_columns = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn"]
    constraint_df = pd.DataFrame(constraint_rows)
    missing_columns = [col for col in runtime_constraint_columns if col not in constraint_df.columns]
    if missing_columns:
        raise ValueError(f"Generated stage2 constraints missing columns: {missing_columns}")
    pos_constraint_df = constraint_df[runtime_constraint_columns]
    if pos_constraint_df["pdb_key"].duplicated().any():
        duplicated = pos_constraint_df[pos_constraint_df["pdb_key"].duplicated()]["pdb_key"].tolist()
        raise ValueError(f"Duplicate stage2 constraint pdb_key values: {duplicated[:5]}")
    if set(stage2_sample_dict.keys()) != set(pos_constraint_df["pdb_key"]):
        raise ValueError("Stage2 sample IDs must exactly match generated constraint pdb_key values")

    return stage2_sample_dict, pos_constraint_df, lineage_by_stage1_sample_id


def _build_twostage_manifest_rows(
    *,
    direction: str,
    lineage_by_stage1_sample_id: dict[str, dict[str, Any]],
    stage2_sample_dict_per_ckpt: dict,
    stage2_ckpt_info: dict[str, Any],
    stage2_log_dir_per_ckpt: Path,
    stage2_model_label: str,
    stage2_constraint_csv: Path,
    stage2_design_cfg: DictConfig,
) -> list[dict[str, Any]]:
    rows = []
    potts_only_cond = get_stage2_potts_only_cond(stage2_design_cfg)

    for stage1_sample_id, stage2_entry in stage2_sample_dict_per_ckpt.items():
        lineage = lineage_by_stage1_sample_id[stage1_sample_id]
        for sample_idx, stage2_sample_id in enumerate(stage2_entry.get("designed_sample_id", [])):
            rows.append({
                **lineage,
                "direction": direction,
                "stage2_model_label": stage2_model_label,
                "stage2_step": stage2_ckpt_info["global_step"],
                "stage2_epoch": stage2_ckpt_info["epoch"],
                "stage2_checkpoint_path": stage2_ckpt_info["ckpt_path"],
                "stage2_checkpoint_dir": str(stage2_log_dir_per_ckpt),
                "stage2_sample_id": stage2_sample_id,
                "stage2_sample_path": stage2_entry["designed_sample_path"][sample_idx],
                "stage2_sample_seq": stage2_entry["designed_sample_seq"][sample_idx],
                "constraint_csv": str(stage2_constraint_csv),
                "potts_only_cond": potts_only_cond,
            })

    return rows


def design_sequence_two_stage(
    *,
    seed: int = 0,
    direction: str,
    input_sample_is_designed: bool = False,
    sample_dict: dict | None = None,
    stage1_design_cfg: DictConfig | None = None,
    stage2_design_cfg: DictConfig | None = None,
    stage1_region: str,
    stage2_region: str,
    stage2_constraint_type: str,
    stage1_model_label: str,
    stage2_model_label: str,
    stage1_cif_parse_cfg: DictConfig | None = None,
    stage1_preprocess_cfg: DictConfig | None = None,
    stage2_cif_parse_cfg: DictConfig | None = None,
    stage2_preprocess_cfg: DictConfig | None = None,
    featurizer_cfg: DictConfig | None = None,
    cif_save_cfg: DictConfig | None = None,
    stage1_sampling_inputs_df: pd.DataFrame | None = None,
    log_dir: Path | None = None,
    protein_only: bool = False,
    pocket_distances_for_seq_recovery: list[float] | None = None,
    pocket_distance_bins: list[tuple[float, float]] | None = None,
    pocket_n_min_ligand_atoms_for_seq_recovery: int = 5,
    csv_suffix: str = "",
    stage1_guidance_cfg: DictConfig | None = None,
    stage2_guidance_cfg: DictConfig | None = None,
    pocket_distance: float = 5.0,
    pocket_annotation_method: str | None = None,
    use_calpha_for_pocket_annotation: bool = False,
) -> Iterator[tuple[dict, Path, dict, list[dict[str, Any]]]]:
    if sample_dict is None:
        raise ValueError("sample_dict must be provided")
    if stage1_design_cfg is None:
        raise ValueError("stage1_design_cfg must be provided")
    if stage2_design_cfg is None:
        raise ValueError("stage2_design_cfg must be provided")
    if log_dir is None:
        raise ValueError("log_dir must be provided")
    pocket_annotation_method = resolve_pocket_annotation_method(
        pocket_annotation_method=pocket_annotation_method,
        use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
    )

    stage1_log_dir = log_dir / direction / f"stage1_{stage1_region}" / stage1_model_label
    stage1_iter = iter_design_sequence_per_checkpoint(
        seed=seed,
        input_sample_is_designed=input_sample_is_designed,
        sample_dict=sample_dict,
        design_cfg=stage1_design_cfg,
        cif_parse_cfg=stage1_cif_parse_cfg,
        preprocess_cfg=stage1_preprocess_cfg,
        featurizer_cfg=featurizer_cfg,
        cif_save_cfg=cif_save_cfg,
        sampling_inputs_df=stage1_sampling_inputs_df,
        log_dir=stage1_log_dir,
        pos_constraint_df=None,
        protein_only=protein_only,
        pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
        pocket_distance_bins=pocket_distance_bins,
        pocket_n_min_ligand_atoms_for_seq_recovery=pocket_n_min_ligand_atoms_for_seq_recovery,
        csv_suffix=csv_suffix,
        guidance_cfg=stage1_guidance_cfg,
    )

    for stage1_sample_dict_per_ckpt, stage1_log_dir_per_ckpt, stage1_ckpt_info in stage1_iter:
        stage2_sample_dict, stage2_pos_constraint_df, lineage_by_stage1_sample_id = (
            _build_stage2_inputs_and_constraints(
                stage1_sample_dict_per_ckpt=stage1_sample_dict_per_ckpt,
                stage1_ckpt_info=stage1_ckpt_info,
                stage1_log_dir_per_ckpt=stage1_log_dir_per_ckpt,
                stage1_region=stage1_region,
                stage2_region=stage2_region,
                stage2_constraint_type=stage2_constraint_type,
                stage1_model_label=stage1_model_label,
                pocket_distance=pocket_distance,
                pocket_annotation_method=pocket_annotation_method,
                use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
            )
        )

        stage1_label = f"{stage1_model_label}_{ckpt_label(stage1_ckpt_info)}"
        stage2_base_log_dir = (
            log_dir
            / direction
            / f"stage2_{stage2_region}"
            / f"from_{stage1_region}_{stage1_label}"
            / stage2_model_label
        )
        stage2_base_log_dir.mkdir(parents=True, exist_ok=True)
        stage2_constraint_csv = (
            stage2_base_log_dir
            / f"pos_constraints_{stage2_constraint_type}_{pocket_distance}A{csv_suffix}.csv"
        )
        stage2_pos_constraint_df.to_csv(stage2_constraint_csv, index=False)

        stage2_iter = iter_design_sequence_per_checkpoint(
            seed=seed,
            input_sample_is_designed=True,
            sample_dict=stage2_sample_dict,
            design_cfg=stage2_design_cfg,
            cif_parse_cfg=stage2_cif_parse_cfg,
            preprocess_cfg=stage2_preprocess_cfg,
            featurizer_cfg=featurizer_cfg,
            cif_save_cfg=cif_save_cfg,
            sampling_inputs_df=None,
            log_dir=stage2_base_log_dir,
            pos_constraint_df=stage2_pos_constraint_df,
            protein_only=protein_only,
            pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
            pocket_distance_bins=pocket_distance_bins,
            pocket_n_min_ligand_atoms_for_seq_recovery=pocket_n_min_ligand_atoms_for_seq_recovery,
            csv_suffix=csv_suffix,
            guidance_cfg=stage2_guidance_cfg,
        )

        for stage2_sample_dict_per_ckpt, stage2_log_dir_per_ckpt, stage2_ckpt_info in stage2_iter:
            manifest_rows = _build_twostage_manifest_rows(
                direction=direction,
                lineage_by_stage1_sample_id=lineage_by_stage1_sample_id,
                stage2_sample_dict_per_ckpt=stage2_sample_dict_per_ckpt,
                stage2_ckpt_info=stage2_ckpt_info,
                stage2_log_dir_per_ckpt=stage2_log_dir_per_ckpt,
                stage2_model_label=stage2_model_label,
                stage2_constraint_csv=stage2_constraint_csv,
                stage2_design_cfg=stage2_design_cfg,
            )
            yield stage2_sample_dict_per_ckpt, stage2_log_dir_per_ckpt, stage2_ckpt_info, manifest_rows
