from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


PAIR_RESIDUE_ALIGNMENT_CSV_NAME = "pair_residue_alignment.csv"

SELECTIVITY_PAIR_REQUIRED_COLUMNS = (
    "pdb_key",
    "pdb_id",
    "query_pn_unit_iids",
    "ccd_code",
    "selectivity_pair_id",
    "selectivity_side",
    "partner_pdb_key",
    "partner_ccd_code",
)

PAIR_RESIDUE_ALIGNMENT_REQUIRED_COLUMNS = (
    "selectivity_pair_id",
    "align_pos",
    "side1_pdb_key",
    "side1_protein_pn_unit_iid",
    "side1_chain_id",
    "side1_res_id",
    "side1_res_name",
    "side1_ca_ordinal",
    "side2_pdb_key",
    "side2_protein_pn_unit_iid",
    "side2_chain_id",
    "side2_res_id",
    "side2_res_name",
    "side2_ca_ordinal",
    "alignment_method",
    "alignment_source",
    "warning_flags",
)

SELECTIVITY_PAIR_GUIDANCE_METADATA_KEYS = (
    "selectivity_pair_id",
    "guidance_target_ligand_side",
    "selectivity_side",
    "partner_pdb_key",
    "positive_ligand_side",
    "negative_ligand_side",
    "positive_branch_label",
    "negative_branch_label",
    "common_residue_count",
    "common_designable_residue_count",
)

SELECTIVITY_GUIDANCE_METADATA_KEYS = SELECTIVITY_PAIR_GUIDANCE_METADATA_KEYS


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _nonempty_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip()
    return text != "" and text.lower() != "nan"


def _normalize_side_series(series: pd.Series, *, label: str) -> pd.Series:
    try:
        sides = series.astype(int)
    except ValueError as exc:
        raise ValueError(f"{label} must contain integer selectivity sides 1 and 2") from exc
    bad = sorted(set(sides) - {1, 2})
    if bad:
        raise ValueError(f"{label} contains invalid selectivity_side values: {bad}")
    return sides


def default_residue_alignment_csv(sampling_inputs_csv: str | Path) -> Path:
    return Path(sampling_inputs_csv).parent / PAIR_RESIDUE_ALIGNMENT_CSV_NAME


def load_selectivity_pair_dataset(
    *,
    sampling_inputs_csv: str | Path,
    residue_alignment_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate paired selectivity runtime metadata."""
    sampling_inputs_path = Path(sampling_inputs_csv)
    alignment_path = (
        Path(residue_alignment_csv)
        if residue_alignment_csv is not None
        else default_residue_alignment_csv(sampling_inputs_path)
    )
    sampling_inputs_df = pd.read_csv(sampling_inputs_path, keep_default_na=False)
    residue_alignment_df = pd.read_csv(alignment_path, keep_default_na=False)
    validate_selectivity_pair_dataset(sampling_inputs_df, residue_alignment_df)
    return sampling_inputs_df, residue_alignment_df


def validate_selectivity_pair_dataset(
    sampling_inputs_df: pd.DataFrame,
    residue_alignment_df: pd.DataFrame,
) -> None:
    _require_columns(sampling_inputs_df, SELECTIVITY_PAIR_REQUIRED_COLUMNS, label="selectivity sampling CSV")
    _require_columns(
        residue_alignment_df,
        PAIR_RESIDUE_ALIGNMENT_REQUIRED_COLUMNS,
        label="pair residue alignment CSV",
    )
    if sampling_inputs_df.empty:
        raise ValueError("selectivity sampling CSV has no rows")
    if residue_alignment_df.empty:
        raise ValueError("pair residue alignment CSV has no rows")

    sampling_inputs_df = sampling_inputs_df.copy()
    residue_alignment_df = residue_alignment_df.copy()
    sampling_inputs_df["selectivity_pair_id"] = sampling_inputs_df["selectivity_pair_id"].astype(str)
    residue_alignment_df["selectivity_pair_id"] = residue_alignment_df["selectivity_pair_id"].astype(str)
    sampling_inputs_df["selectivity_side"] = _normalize_side_series(
        sampling_inputs_df["selectivity_side"], label="selectivity sampling CSV"
    )

    duplicated_keys = sampling_inputs_df[sampling_inputs_df.duplicated("pdb_key", keep=False)]
    if not duplicated_keys.empty:
        raise ValueError(
            "selectivity sampling CSV pdb_key values must be unique; duplicates: "
            f"{duplicated_keys['pdb_key'].astype(str).tolist()[:10]}"
        )
    if sampling_inputs_df.duplicated(["selectivity_pair_id", "selectivity_side"]).any():
        raise ValueError("selectivity sampling CSV has duplicate (selectivity_pair_id, selectivity_side) rows")

    side_counts = sampling_inputs_df.groupby("selectivity_pair_id")["selectivity_side"].apply(
        lambda sides: tuple(sorted(sides.tolist()))
    )
    malformed_pairs = side_counts[side_counts != (1, 2)]
    if not malformed_pairs.empty:
        raise ValueError(
            "each selectivity pair must contain exactly sides 1 and 2; "
            f"bad pair ids: {malformed_pairs.index.astype(str).tolist()[:10]}"
        )

    for column in ("pdb_key", "pdb_id", "query_pn_unit_iids", "ccd_code", "partner_pdb_key"):
        empty = sampling_inputs_df[~sampling_inputs_df[column].map(_nonempty_text)]
        if not empty.empty:
            raise ValueError(f"selectivity sampling CSV column {column!r} has empty values")

    pairs = {
        pair_id: group.set_index("selectivity_side")
        for pair_id, group in sampling_inputs_df.groupby("selectivity_pair_id", sort=False)
    }
    for pair_id, group in pairs.items():
        side1_key = str(group.loc[1, "pdb_key"])
        side2_key = str(group.loc[2, "pdb_key"])
        if str(group.loc[1, "partner_pdb_key"]) != side2_key:
            raise ValueError(f"pair {pair_id} side 1 partner_pdb_key does not point to side 2")
        if str(group.loc[2, "partner_pdb_key"]) != side1_key:
            raise ValueError(f"pair {pair_id} side 2 partner_pdb_key does not point to side 1")

    selectivity_pair_ids = set(sampling_inputs_df["selectivity_pair_id"])
    alignment_pair_ids = set(residue_alignment_df["selectivity_pair_id"])
    missing_alignment = sorted(selectivity_pair_ids - alignment_pair_ids)
    extra_alignment = sorted(alignment_pair_ids - selectivity_pair_ids)
    if missing_alignment or extra_alignment:
        raise ValueError(
            "pair residue alignment ids must match selectivity pair ids; "
            f"missing={missing_alignment[:10]} extra={extra_alignment[:10]}"
        )

    if residue_alignment_df.duplicated(["selectivity_pair_id", "align_pos"]).any():
        raise ValueError("pair residue alignment CSV has duplicate (selectivity_pair_id, align_pos) rows")

    sampling_keys = set(sampling_inputs_df["pdb_key"].astype(str))
    alignment_keys = set(residue_alignment_df["side1_pdb_key"].astype(str)) | set(
        residue_alignment_df["side2_pdb_key"].astype(str)
    )
    unknown_keys = sorted(alignment_keys - sampling_keys)
    if unknown_keys:
        raise ValueError(f"pair residue alignment CSV references unknown sampling keys: {unknown_keys[:10]}")

    for pair_id, group in residue_alignment_df.groupby("selectivity_pair_id", sort=False):
        side_rows = pairs[pair_id]
        side1_key = str(side_rows.loc[1, "pdb_key"])
        side2_key = str(side_rows.loc[2, "pdb_key"])
        if set(group["side1_pdb_key"].astype(str)) != {side1_key}:
            raise ValueError(f"pair {pair_id} residue alignment side1_pdb_key does not match selectivity side 1")
        if set(group["side2_pdb_key"].astype(str)) != {side2_key}:
            raise ValueError(f"pair {pair_id} residue alignment side2_pdb_key does not match selectivity side 2")

        align_pos = sorted(group["align_pos"].astype(int).tolist())
        if align_pos != sorted(align_pos) or align_pos[0] != 0 or align_pos[-1] != len(align_pos) - 1:
            raise ValueError(f"pair {pair_id} residue alignment align_pos must be contiguous from 0")

    for column in (
        "side1_protein_pn_unit_iid",
        "side1_chain_id",
        "side1_res_id",
        "side1_res_name",
        "side2_protein_pn_unit_iid",
        "side2_chain_id",
        "side2_res_id",
        "side2_res_name",
    ):
        empty = residue_alignment_df[~residue_alignment_df[column].map(_nonempty_text)]
        if not empty.empty:
            raise ValueError(f"pair residue alignment CSV column {column!r} has empty values")


def pair_residue_alignment_for_pair(
    residue_alignment_df: pd.DataFrame,
    selectivity_pair_id: str | int,
) -> pd.DataFrame:
    pair_id = str(selectivity_pair_id)
    alignment_df = residue_alignment_df.copy()
    alignment_df["selectivity_pair_id"] = alignment_df["selectivity_pair_id"].astype(str)
    subset = alignment_df[alignment_df["selectivity_pair_id"] == pair_id].copy()
    if subset.empty:
        raise ValueError(f"No residue alignment rows for selectivity_pair_id={pair_id!r}")
    return subset.sort_values("align_pos").reset_index(drop=True)


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


def selectivity_pair_rows_for_batch(
    sampling_inputs_df: pd.DataFrame,
    example_ids: list[str],
) -> dict[int, pd.Series]:
    """Return selectivity sampling rows keyed by side for one pair batch."""
    if sampling_inputs_df is None:
        raise ValueError("guidance.mode=selectivity requires paired selectivity sampling_inputs_csv")
    _require_columns(sampling_inputs_df, SELECTIVITY_PAIR_REQUIRED_COLUMNS, label="selectivity sampling CSV")
    if len(example_ids) != 2:
        raise ValueError(f"selectivity pair guidance requires batch size 2, got {len(example_ids)}")

    rows = sampling_inputs_df[
        sampling_inputs_df["pdb_key"].astype(str).isin([str(eid) for eid in example_ids])
    ].copy()
    if len(rows) != 2:
        raise ValueError(
            "selectivity pair guidance could not match both batch example_ids "
            f"to sampling_inputs_csv: {example_ids}"
        )
    pair_ids = {str(value) for value in rows["selectivity_pair_id"]}
    if len(pair_ids) != 1:
        raise ValueError(f"selectivity pair batch spans multiple pair ids: {sorted(pair_ids)}")
    rows["selectivity_side"] = _normalize_side_series(rows["selectivity_side"], label="selectivity sampling CSV")
    side_rows = {int(row["selectivity_side"]): row for _, row in rows.iterrows()}
    if set(side_rows) != {1, 2}:
        raise ValueError(
            "selectivity pair guidance requires exactly sides 1 and 2 in each batch; "
            f"got {sorted(side_rows)} for examples {example_ids}"
        )
    return side_rows
