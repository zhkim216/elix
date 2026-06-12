from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.utils.data_utils import _matched_sampling_input_row
from allatom_design.eval.utils.selectivity import (
    SELECTIVITY_GUIDANCE_METADATA_KEYS,
    normalize_target_ligand_side,
    resolve_selectivity_guidance_branches,
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


def _ligand_atom_mask_for_branch(
    *,
    atom_array,
    atom_cond_mask: torch.Tensor,
    ligand_pn_unit_iid: str,
    example_id: str,
) -> torch.Tensor:
    ligand_mask = torch.zeros_like(atom_cond_mask, dtype=torch.bool)
    atom_iids = np.asarray(atom_array.pn_unit_iid).astype(str)
    selected = atom_iids == str(ligand_pn_unit_iid)
    if not np.any(selected):
        raise ValueError(
            f"selectivity guidance ligand {ligand_pn_unit_iid!r} not found in atom_array for {example_id}"
        )
    ligand_mask[: len(atom_array)] = torch.as_tensor(selected, device=atom_cond_mask.device)
    return ligand_mask


def build_selectivity_guidance_branch_masks(
    *,
    batch: dict[str, Any],
    sampling_inputs_df: pd.DataFrame,
    target_ligand_side: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, Any]]]:
    """Build generic positive/negative branch atom masks for selectivity guidance."""
    if sampling_inputs_df is None:
        raise ValueError("guidance.mode=selectivity requires sampling_inputs_csv")

    positive_mask = torch.zeros_like(batch["atom_cond_mask"])
    negative_mask = torch.zeros_like(batch["atom_cond_mask"])
    metadata_by_example_id: dict[str, dict[str, Any]] = {}

    for batch_idx, example_id_raw in enumerate(batch["example_id"]):
        example_id = str(example_id_raw)
        row = _matched_sampling_input_row(
            sampling_inputs_df,
            pdb_id=example_id.split("_")[0],
            pdb_key=example_id,
        )
        if row is None:
            raise ValueError(
                f"guidance.mode=selectivity could not match example_id={example_id!r} "
                "to sampling_inputs_csv by pdb_key or pdb_id"
            )

        metadata = resolve_selectivity_guidance_branches(
            row,
            target_ligand_side=target_ligand_side,
            example_id=example_id,
        )
        atom_cond_mask = batch["atom_cond_mask"][batch_idx]
        protein_atom_mask = batch["atom_is_protein_chain"][batch_idx].bool()
        positive_ligand_mask = _ligand_atom_mask_for_branch(
            atom_array=batch["atom_array"][batch_idx],
            atom_cond_mask=atom_cond_mask,
            ligand_pn_unit_iid=metadata["positive_ligand_pn_unit_iid"],
            example_id=example_id,
        )
        negative_ligand_mask = _ligand_atom_mask_for_branch(
            atom_array=batch["atom_array"][batch_idx],
            atom_cond_mask=atom_cond_mask,
            ligand_pn_unit_iid=metadata["negative_ligand_pn_unit_iid"],
            example_id=example_id,
        )

        positive_keep = protein_atom_mask | positive_ligand_mask
        negative_keep = protein_atom_mask | negative_ligand_mask
        positive_mask[batch_idx] = atom_cond_mask * positive_keep.to(atom_cond_mask.dtype)
        negative_mask[batch_idx] = atom_cond_mask * negative_keep.to(atom_cond_mask.dtype)
        metadata_by_example_id[example_id] = metadata

    return positive_mask, negative_mask, metadata_by_example_id


def inject_guidance_inputs(
    *,
    guidance_cfg: DictConfig | dict | None,
    batch: dict[str, Any],
    sampling_inputs_df: pd.DataFrame | None,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    guidance_cfg_dict = guidance_cfg_to_dict(guidance_cfg)
    if guidance_cfg_dict is None:
        return None, {}
    if not guidance_cfg_dict.get("enabled", False):
        return guidance_cfg_dict, {}

    if guidance_mode(guidance_cfg_dict) != "selectivity":
        return guidance_cfg_dict, {}

    target_ligand_side = normalize_target_ligand_side(
        guidance_cfg_dict.get("target_ligand_side", None)
    )
    positive_mask, negative_mask, metadata_by_example_id = build_selectivity_guidance_branch_masks(
        batch=batch,
        sampling_inputs_df=sampling_inputs_df,
        target_ligand_side=target_ligand_side,
    )
    guidance_cfg_dict["positive_atom_cond_mask"] = positive_mask
    guidance_cfg_dict["negative_atom_cond_mask"] = negative_mask
    guidance_cfg_dict.setdefault("positive_branch_label", "positive")
    guidance_cfg_dict.setdefault("negative_branch_label", "negative")
    return guidance_cfg_dict, metadata_by_example_id
