from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.selectivity import (
    SELECTIVITY_GUIDANCE_METADATA_KEYS,
    normalize_target_ligand_side,
    pair_residue_alignment_for_pair,
    selectivity_pair_rows_for_batch,
)


POTTS_GUIDANCE_AUX_KEYS = (
    "guidance_mode",
    "guidance_scale",
    "positive_branch_label",
    "negative_branch_label",
    "U_positive",
    "U_negative",
    "U_positive_per_res",
    "U_negative_per_res",
    "U_positive_pocket",
    "U_negative_pocket",
    "U_positive_pocket_per_res",
    "U_negative_pocket_per_res",
    "U_cond",
    "U_uncond",
    "U_cond_per_res",
    "U_uncond_per_res",
    "U_cond_pocket",
    "U_uncond_pocket",
    "U_cond_pocket_per_res",
    "U_uncond_pocket_per_res",
    "N_pocket",
)

POTTS_GUIDANCE_OUTPUT_KEYS = (
    *POTTS_GUIDANCE_AUX_KEYS,
    *SELECTIVITY_GUIDANCE_METADATA_KEYS,
)


def guidance_cfg_to_dict(guidance_cfg: DictConfig | dict | None) -> dict[str, Any] | None:
    if guidance_cfg is None:
        return None
    if isinstance(guidance_cfg, DictConfig):
        return OmegaConf.to_container(guidance_cfg, resolve=True)
    return dict(guidance_cfg)


def guidance_mode(guidance_cfg: DictConfig | dict | None) -> str:
    if guidance_cfg is None:
        return "cond_uncond"
    return str(guidance_cfg.get("mode", "cond_uncond"))


def guidance_is_enabled(cfg: DictConfig | dict | None) -> bool:
    if cfg is None:
        return False
    selectable_cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    direct_enabled = OmegaConf.select(selectable_cfg, "enabled", default=None)
    if direct_enabled is not None:
        return _config_value_as_bool(direct_enabled)
    return _config_value_as_bool(
        OmegaConf.select(selectable_cfg, "sampling_cfg.guidance.enabled", default=False)
    )


def _config_value_as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def build_selectivity_pair_guidance_inputs(
    *,
    batch: dict[str, Any],
    sampling_inputs_df: pd.DataFrame,
    selectivity_residue_alignment_df: pd.DataFrame,
    target_ligand_side: int | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build selectivity pair tensors on the current batch token axis."""
    if sampling_inputs_df is None or selectivity_residue_alignment_df is None:
        raise ValueError(
            "guidance.mode=selectivity requires paired sampling CSV and pair_residue_alignment.csv"
        )
    return build_selectivity_pair_token_map(
        batch=batch,
        sampling_inputs_df=sampling_inputs_df,
        residue_alignment_df=selectivity_residue_alignment_df,
        target_ligand_side=target_ligand_side,
    )


def _residue_token_index_by_key(atom_array) -> dict[tuple[str, str, str], int]:
    atom_names = np.asarray(atom_array.atom_name).astype(str)
    pn_unit_iids = np.asarray(atom_array.pn_unit_iid).astype(str)
    chain_ids = np.asarray(atom_array.chain_id).astype(str)
    res_ids = np.asarray(atom_array.res_id).astype(str)
    token_ids = np.asarray(atom_array.token_id).astype(int)

    mapping: dict[tuple[str, str, str], int] = {}
    ca_indices = np.where(atom_names == "CA")[0]
    for atom_idx in ca_indices:
        key = (pn_unit_iids[atom_idx], chain_ids[atom_idx], res_ids[atom_idx])
        mapping.setdefault(key, int(token_ids[atom_idx]))
    return mapping


def _residue_key(row: pd.Series, side: int) -> tuple[str, str, str]:
    return (
        str(row[f"side{side}_protein_pn_unit_iid"]),
        str(row[f"side{side}_chain_id"]),
        str(row[f"side{side}_res_id"]),
    )


def build_selectivity_pair_token_map(
    *,
    batch: dict[str, Any],
    sampling_inputs_df: pd.DataFrame,
    residue_alignment_df: pd.DataFrame,
    target_ligand_side: int | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build tensor guidance inputs for one selectivity pair batch.

    The returned tensors are indexed in current batch order, while alignment
    positions stay in side1/side2 order from ``pair_residue_alignment.csv``.
    """
    example_ids = [str(example_id) for example_id in batch["example_id"]]
    side_rows = selectivity_pair_rows_for_batch(sampling_inputs_df, example_ids)
    pair_id = str(side_rows[1]["selectivity_pair_id"])
    alignment_rows = pair_residue_alignment_for_pair(residue_alignment_df, pair_id)
    target_ligand_side = normalize_target_ligand_side(target_ligand_side) or 1
    positive_ligand_side = target_ligand_side
    negative_ligand_side = 3 - target_ligand_side

    batch_idx_by_side = {
        int(side_rows[side]["selectivity_side"]): example_ids.index(
            str(side_rows[side]["pdb_key"])
        )
        for side in (1, 2)
    }
    token_maps_by_side = {
        side: _residue_token_index_by_key(batch["atom_array"][batch_idx_by_side[side]])
        for side in (1, 2)
    }

    token_indices_by_side: dict[int, list[int]] = {1: [], 2: []}
    for _, row in alignment_rows.iterrows():
        for side in (1, 2):
            key = _residue_key(row, side)
            token_idx = token_maps_by_side[side].get(key)
            if token_idx is None:
                raise ValueError(
                    "pair_residue_alignment.csv references a residue not found in the "
                    f"batch atom_array token map for pair {pair_id}, side {side}: {key}"
                )
            token_indices_by_side[side].append(token_idx)

    device = batch["token_pad_mask"].device
    common_token_idx = torch.empty((2, len(alignment_rows)), dtype=torch.long, device=device)
    common_token_idx[batch_idx_by_side[1]] = torch.tensor(
        token_indices_by_side[1],
        dtype=torch.long,
        device=device,
    )
    common_token_idx[batch_idx_by_side[2]] = torch.tensor(
        token_indices_by_side[2],
        dtype=torch.long,
        device=device,
    )

    token_pad_mask = batch["token_pad_mask"].bool()
    token_is_prot = batch["token_is_prot_std_aa"].bool()
    common_valid_mask = (
        token_pad_mask[0, common_token_idx[0]]
        & token_pad_mask[1, common_token_idx[1]]
        & token_is_prot[0, common_token_idx[0]]
        & token_is_prot[1, common_token_idx[1]]
    )
    designable_mask = (
        (1 - batch["seq_cond_mask"].float()).bool()[0, common_token_idx[0]]
        & (1 - batch["seq_cond_mask"].float()).bool()[1, common_token_idx[1]]
        & common_valid_mask
    )
    if not bool(common_valid_mask.any()):
        raise ValueError(f"selectivity pair {pair_id} has no valid common protein residues")
    if not bool(designable_mask.any()):
        raise ValueError(f"selectivity pair {pair_id} has no designable common residues")

    positive_batch_idx = batch_idx_by_side[positive_ligand_side]
    negative_batch_idx = batch_idx_by_side[negative_ligand_side]
    guidance_inputs = {
        "selectivity_pair_guidance": True,
        "selectivity_pair_id": pair_id,
        "common_token_idx": common_token_idx,
        "common_valid_mask": common_valid_mask,
        "common_designable_mask": designable_mask,
        "positive_batch_idx": positive_batch_idx,
        "negative_batch_idx": negative_batch_idx,
        "positive_ligand_side": positive_ligand_side,
        "negative_ligand_side": negative_ligand_side,
        "positive_branch_label": f"ligand_{positive_ligand_side}",
        "negative_branch_label": f"ligand_{negative_ligand_side}",
    }

    metadata_by_example_id: dict[str, dict[str, Any]] = {}
    for side in (1, 2):
        row = side_rows[side]
        metadata_by_example_id[str(row["pdb_key"])] = {
            "selectivity_pair_id": pair_id,
            "guidance_target_ligand_side": target_ligand_side,
            "selectivity_side": side,
            "partner_pdb_key": str(row["partner_pdb_key"]),
            "positive_ligand_side": positive_ligand_side,
            "negative_ligand_side": negative_ligand_side,
            "positive_branch_label": f"ligand_{positive_ligand_side}",
            "negative_branch_label": f"ligand_{negative_ligand_side}",
            "common_residue_count": int(common_valid_mask.sum().item()),
            "common_designable_residue_count": int(designable_mask.sum().item()),
        }
    return guidance_inputs, metadata_by_example_id


def _reject_legacy_selectivity_schema(sampling_inputs_df: pd.DataFrame | None) -> None:
    if sampling_inputs_df is None:
        return
    legacy_columns = {
        "native_ligand_side",
        "transformed_ligand_side",
        "ligand_1_pn_unit_iid",
        "ligand_2_pn_unit_iid",
    }
    if legacy_columns & set(sampling_inputs_df.columns):
        raise NotImplementedError(
            "legacy composite dual-ligand selectivity guidance is no longer supported; "
            "use paired selectivity sampling_inputs_csv with pair_residue_alignment.csv"
        )


def inject_guidance_inputs(
    *,
    guidance_cfg: DictConfig | dict | None,
    batch: dict[str, Any],
    sampling_inputs_df: pd.DataFrame | None,
    selectivity_residue_alignment_df: pd.DataFrame | None = None,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    guidance_cfg_dict = guidance_cfg_to_dict(guidance_cfg)
    if guidance_cfg_dict is None:
        return None, {}
    if not guidance_cfg_dict.get("enabled", False):
        return guidance_cfg_dict, {}

    if guidance_mode(guidance_cfg_dict) != "selectivity":
        return guidance_cfg_dict, {}

    _reject_legacy_selectivity_schema(sampling_inputs_df)
    target_ligand_side = normalize_target_ligand_side(
        guidance_cfg_dict.get("target_ligand_side", None)
    )
    selectivity_pair_inputs, metadata_by_example_id = build_selectivity_pair_guidance_inputs(
        batch=batch,
        sampling_inputs_df=sampling_inputs_df,
        selectivity_residue_alignment_df=selectivity_residue_alignment_df,
        target_ligand_side=target_ligand_side,
    )
    guidance_cfg_dict.update(selectivity_pair_inputs)
    return guidance_cfg_dict, metadata_by_example_id


__all__ = [
    "POTTS_GUIDANCE_AUX_KEYS",
    "POTTS_GUIDANCE_OUTPUT_KEYS",
    "build_selectivity_pair_guidance_inputs",
    "build_selectivity_pair_token_map",
    "guidance_cfg_to_dict",
    "guidance_is_enabled",
    "guidance_mode",
    "inject_guidance_inputs",
]
