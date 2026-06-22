from __future__ import annotations

from typing import Any

import pandas as pd

from allatom_design.eval.sampling_inputs import parse_query_pn_unit_iids
from allatom_design.eval.selectivity.pairs import normalize_target_ligand_side


LEGACY_SELECTIVITY_GUIDANCE_METADATA_KEYS = (
    "selectivity_pair_id",
    "scaffold_side",
    "native_ligand_side",
    "transformed_ligand_side",
    "guidance_target_ligand_side",
    "positive_ligand_side",
    "negative_ligand_side",
    "positive_ligand_pn_unit_iid",
    "negative_ligand_pn_unit_iid",
    "positive_branch_label",
    "negative_branch_label",
    "positive_ligand_role",
    "negative_ligand_role",
)


def _row_has(row: pd.Series, column: str) -> bool:
    return column in row.index


def _required_sampling_value(row: pd.Series, column: str, example_id: str) -> str:
    if not _row_has(row, column):
        raise ValueError(f"selectivity guidance requires column {column!r} for {example_id}")
    value = row[column]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError(f"selectivity guidance column {column!r} is empty for {example_id}")
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        raise ValueError(f"selectivity guidance column {column!r} is empty for {example_id}")
    return text


def _required_ligand_side(row: pd.Series, column: str, example_id: str) -> int:
    raw_side = _required_sampling_value(row, column, example_id)
    try:
        side = int(raw_side)
    except ValueError as exc:
        raise ValueError(f"{column} must be 1 or 2 for {example_id}, got {raw_side!r}") from exc
    if side not in (1, 2):
        raise ValueError(f"{column} must be 1 or 2 for {example_id}, got {side}")
    return side


def _ligand_role(side: int, native_side: int, transformed_side: int, example_id: str) -> str:
    if side == native_side:
        return "native"
    if side == transformed_side:
        return "transformed"
    raise ValueError(
        f"ligand side {side} is neither native_side={native_side} nor "
        f"transformed_side={transformed_side} for {example_id}"
    )


def resolve_selectivity_guidance_branches(
    row: pd.Series,
    *,
    target_ligand_side: int | None,
    example_id: str,
) -> dict[str, Any]:
    """Resolve legacy composite dual-ligand selectivity branch metadata."""
    target_ligand_side = normalize_target_ligand_side(target_ligand_side)
    native_side = _required_ligand_side(row, "native_ligand_side", example_id)
    transformed_side = _required_ligand_side(row, "transformed_ligand_side", example_id)
    if native_side == transformed_side:
        raise ValueError(
            f"native_ligand_side and transformed_ligand_side must differ for {example_id}; "
            f"both are {native_side}"
        )

    if target_ligand_side is None:
        positive_side = native_side
        negative_side = transformed_side
    else:
        positive_side = target_ligand_side
        negative_side = 3 - target_ligand_side

    positive_iid = _required_sampling_value(row, f"ligand_{positive_side}_pn_unit_iid", example_id)
    negative_iid = _required_sampling_value(row, f"ligand_{negative_side}_pn_unit_iid", example_id)
    metadata = {
        "scaffold_side": _required_ligand_side(row, "scaffold_side", example_id),
        "native_ligand_side": native_side,
        "transformed_ligand_side": transformed_side,
        "guidance_target_ligand_side": target_ligand_side,
        "positive_ligand_side": positive_side,
        "negative_ligand_side": negative_side,
        "positive_ligand_pn_unit_iid": positive_iid,
        "negative_ligand_pn_unit_iid": negative_iid,
        "positive_branch_label": f"ligand_{positive_side}",
        "negative_branch_label": f"ligand_{negative_side}",
        "positive_ligand_role": _ligand_role(positive_side, native_side, transformed_side, example_id),
        "negative_ligand_role": _ligand_role(negative_side, native_side, transformed_side, example_id),
    }
    if "selectivity_pair_id" in row.index:
        metadata["selectivity_pair_id"] = str(row["selectivity_pair_id"])
    return metadata


def resolve_selectivity_row(
    *,
    sampling_inputs_df: pd.DataFrame,
    pdb_id: str | None = None,
    pdb_key: str | None = None,
    guidance_direction: int,
) -> dict[str, Any]:
    """Resolve one backbone's context from the legacy composite selectivity CSV."""
    if guidance_direction not in (1, 2):
        raise ValueError(f"guidance_direction must be 1 or 2, got {guidance_direction}")

    required_cols = {
        "pdb_id_1",
        "pdb_id_2",
        "query_pn_unit_iids_1",
        "query_pn_unit_iids_2",
        "ccd_code_1",
        "ccd_code_2",
    }
    missing = required_cols - set(sampling_inputs_df.columns)
    if missing:
        raise ValueError(f"sampling_inputs_df missing columns: {sorted(missing)}")

    if pdb_id is None and pdb_key is None:
        raise ValueError("Either pdb_id or pdb_key must be provided")

    pdb_lc = str(pdb_id).lower() if pdb_id is not None else None
    pdb_key_lc = str(pdb_key).lower() if pdb_key is not None else None
    matches: list[tuple[pd.Series, int]] = []
    for self_pos in (1, 2):
        for _, row in sampling_inputs_df.iterrows():
            row_pdb_id = str(row[f"pdb_id_{self_pos}"]).lower()
            query_iids = parse_query_pn_unit_iids(row[f"query_pn_unit_iids_{self_pos}"])
            row_pdb_key = f"{row[f'pdb_id_{self_pos}']}_{query_iids[0]}_{query_iids[1]}".lower()
            if pdb_key_lc is not None:
                if row_pdb_key == pdb_key_lc:
                    matches.append((row, self_pos))
            elif row_pdb_id == pdb_lc:
                matches.append((row, self_pos))

    if len(matches) > 1:
        raise ValueError(
            f"Endpoint match is ambiguous for pdb_id={pdb_id!r}; pass pdb_key to disambiguate"
        )
    if len(matches) == 1:
        row, self_pos = matches[0]
        other_pos = 3 - self_pos
        out = {
            "pdb_id_self": str(row[f"pdb_id_{self_pos}"]),
            "query_pn_unit_iids_self": parse_query_pn_unit_iids(
                row[f"query_pn_unit_iids_{self_pos}"]
            ),
            "ccd_self": str(row[f"ccd_code_{self_pos}"]),
            "pdb_id_partner": str(row[f"pdb_id_{other_pos}"]),
            "query_pn_unit_iids_partner": parse_query_pn_unit_iids(
                row[f"query_pn_unit_iids_{other_pos}"]
            ),
            "ccd_partner": str(row[f"ccd_code_{other_pos}"]),
            "guidance_target_ccd": str(row[f"ccd_code_{guidance_direction}"]),
            "self_position": self_pos,
        }
        if "pocket_subcluster_id" in sampling_inputs_df.columns:
            out["pocket_subcluster_id"] = int(row["pocket_subcluster_id"])
        return out

    raise ValueError(
        f"pdb_id={pdb_id} pdb_key={pdb_key} not found in paired selectivity columns of "
        f"sampling_inputs_df (rows={len(sampling_inputs_df)})"
    )
