from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from allatom_design.eval.sampling.sequence_design.inputs import create_sample_dict
from allatom_design.eval.selectivity import (
    default_residue_alignment_csv,
    load_selectivity_pair_dataset,
    normalize_target_ligand_side,
)


def require_selectivity_guidance_cfg(cfg: DictConfig) -> DictConfig:
    sampling_cfg = cfg.get("sampling_cfg", None)
    if sampling_cfg is None or sampling_cfg.get("guidance", None) is None:
        raise ValueError("selectivity run requires sampling_cfg.guidance")
    guidance_cfg = sampling_cfg.guidance
    if not bool(guidance_cfg.get("enabled", False)) or str(guidance_cfg.get("mode", "")) != "selectivity":
        raise ValueError("selectivity run requires sampling_cfg.guidance.mode=selectivity")
    if normalize_target_ligand_side(guidance_cfg.get("target_ligand_side", None)) is None:
        raise ValueError("selectivity run requires explicit sampling_cfg.guidance.target_ligand_side=1 or 2")
    return guidance_cfg


def reject_selectivity_two_stage(cfg: DictConfig) -> None:
    two_stage_cfg = cfg.get("two_stage", None)
    if two_stage_cfg is not None and bool(two_stage_cfg.get("enabled", False)):
        raise NotImplementedError(
            "paired selectivity sequence design does not support two_stage.enabled=true; "
            "run_selectivity_elix.py only supports single-stage paired selectivity sampling"
        )


def alignment_csv_for_cfg(cfg: DictConfig, sampling_inputs_csv: str) -> Path:
    guidance_cfg = require_selectivity_guidance_cfg(cfg)
    configured = guidance_cfg.get("pair_residue_alignment_csv", None)
    if configured is not None:
        return Path(str(configured))
    return default_residue_alignment_csv(sampling_inputs_csv)


def load_selectivity_inputs(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampling_inputs_csv = cfg.get("sampling_inputs_csv", None)
    if sampling_inputs_csv is None:
        raise ValueError("selectivity run requires sampling_inputs_csv")
    alignment_csv = alignment_csv_for_cfg(cfg, str(sampling_inputs_csv))
    sampling_inputs_df, residue_alignment_df = load_selectivity_pair_dataset(
        sampling_inputs_csv=sampling_inputs_csv,
        residue_alignment_csv=alignment_csv,
    )
    print(f"Validated selectivity pair sampling CSV: {sampling_inputs_csv}")
    print(f"Validated selectivity pair residue alignment: {alignment_csv}")
    return sampling_inputs_df, residue_alignment_df


def _pdb_name_stems(pdb_name_list: str | None) -> set[str] | None:
    if pdb_name_list is None:
        return None
    with open(pdb_name_list, "r") as handle:
        stems = {Path(line.strip()).stem for line in handle if line.strip()}
    if not stems:
        raise ValueError(f"pdb_cfg.pdb_name_list is empty: {pdb_name_list}")
    return stems


def _complete_pair_ids(rows: pd.DataFrame) -> list[str]:
    malformed: list[str] = []
    pair_ids: list[str] = []
    for pair_id, group in rows.groupby("selectivity_pair_id", sort=False):
        sides = sorted(group["selectivity_side"].astype(int).tolist())
        if sides != [1, 2]:
            malformed.append(str(pair_id))
        else:
            pair_ids.append(str(pair_id))
    if malformed:
        raise ValueError(
            "selectivity pair shard/filter must keep both sides for each pair; "
            f"bad pair ids: {malformed[:10]}"
        )
    return pair_ids


def select_pair_ids_for_array(
    pair_ids: list[str],
    array_id: int | None,
    num_arrays: int | None,
) -> list[str]:
    if array_id is None:
        return pair_ids
    if num_arrays is None:
        raise ValueError("pdb_cfg.num_arrays is required when pdb_cfg.array_id is set")
    if array_id < 0 or array_id >= num_arrays:
        raise ValueError(f"pdb_cfg.array_id must be in [0, {num_arrays}), got {array_id}")
    chunk_size = math.ceil(len(pair_ids) / num_arrays)
    start_idx = array_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(pair_ids))
    return pair_ids[start_idx:end_idx]


def selectivity_rows_for_cfg(cfg: DictConfig, sampling_inputs_df: pd.DataFrame) -> pd.DataFrame:
    rows = sampling_inputs_df.copy()
    name_stems = _pdb_name_stems(cfg.pdb_cfg.get("pdb_name_list", None))
    if name_stems is not None:
        rows = rows[rows["pdb_key"].astype(str).isin(name_stems)].copy()
        if rows.empty:
            raise ValueError("pdb_cfg.pdb_name_list did not match any selectivity sampling CSV pdb_key values")

    pair_ids = _complete_pair_ids(rows)
    if cfg.debug:
        pair_ids = pair_ids[: max(1, int(cfg.num_debug_samples))]

    selected_pair_ids = select_pair_ids_for_array(
        pair_ids=pair_ids,
        array_id=cfg.pdb_cfg.get("array_id", None),
        num_arrays=cfg.pdb_cfg.get("num_arrays", None),
    )
    if not selected_pair_ids:
        raise ValueError("selectivity pair shard selected no pairs")

    selected = rows[rows["selectivity_pair_id"].astype(str).isin(selected_pair_ids)].copy()
    selected["selectivity_pair_id"] = selected["selectivity_pair_id"].astype(str)
    selected["selectivity_side"] = selected["selectivity_side"].astype(int)
    selected = selected.sort_values(["selectivity_pair_id", "selectivity_side"], kind="stable")
    _complete_pair_ids(selected)
    return selected


def residue_alignment_for_selected_rows(
    residue_alignment_df: pd.DataFrame,
    selected_sampling_inputs_df: pd.DataFrame,
) -> pd.DataFrame:
    selected_pair_ids = selected_sampling_inputs_df["selectivity_pair_id"].astype(str).unique().tolist()
    selected = residue_alignment_df[
        residue_alignment_df["selectivity_pair_id"].astype(str).isin(selected_pair_ids)
    ].copy()
    if selected.empty:
        raise ValueError("selected selectivity pairs have no residue alignment rows")
    return selected


def selectivity_sample_dict(cfg: DictConfig, sampling_inputs_df: pd.DataFrame) -> dict:
    pdb_dir = cfg.pdb_cfg.get("pdb_dir", None)
    if pdb_dir is None:
        raise ValueError("selectivity run requires pdb_cfg.pdb_dir")
    pdb_name_ext = cfg.pdb_cfg.get("pdb_name_ext", ".cif")

    sample_paths: list[str] = []
    sample_ids: list[str] = []
    for row in sampling_inputs_df.itertuples(index=False):
        pdb_key = str(row.pdb_key)
        file_name = str(Path(pdb_key).with_suffix(str(pdb_name_ext)))
        path = Path(str(pdb_dir)) / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing selectivity input CIF for pdb_key={pdb_key!r}: {path}")
        sample_paths.append(str(path))
        sample_ids.append(pdb_key)
    return create_sample_dict(sample_paths=sample_paths, sample_ids=sample_ids)
