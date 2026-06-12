from __future__ import annotations

from typing import Any

import pandas as pd


SELECTIVITY_GUIDANCE_METADATA_KEYS = (
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


def normalize_target_ligand_side(value: Any) -> int | None:
    """Normalize guidance target side to None, 1, or 2."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"target_ligand_side must be null, 1, or 2; got {value!r}")
    if isinstance(value, int):
        side = value
    elif isinstance(value, float):
        if pd.isna(value):
            return None
        if not value.is_integer():
            raise ValueError(f"target_ligand_side must be null, 1, or 2; got {value!r}")
        side = int(value)
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null", "nan"}:
            return None
        if text not in {"1", "2"}:
            raise ValueError(f"target_ligand_side must be null, 1, or 2; got {value!r}")
        side = int(text)

    if side not in (1, 2):
        raise ValueError(f"target_ligand_side must be null, 1, or 2; got {value!r}")
    return side


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
    """Resolve dual-ligand selectivity positive/negative branch metadata."""
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
