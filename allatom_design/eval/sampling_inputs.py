from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd
from biotite.structure import AtomArray


def parse_query_pn_unit_iids(raw_value: Any) -> list[str]:
    """Parse query_pn_unit_iids from a CSV/metadata cell."""
    if raw_value is None:
        return []

    if isinstance(raw_value, (float, np.floating)) and np.isnan(raw_value):
        return []

    parsed = raw_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = stripped

    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()

    if isinstance(parsed, (list, tuple, set)):
        return [str(x) for x in parsed if str(x) != ""]

    return [str(parsed)] if str(parsed) != "" else []


def matched_sampling_input_row(
    sampling_inputs_df: pd.DataFrame | None,
    pdb_id: str | None,
    pdb_key: str | None = None,
) -> pd.Series | None:
    if sampling_inputs_df is None:
        return None

    if pdb_key is not None and "pdb_key" in sampling_inputs_df.columns:
        pdb_key_normalized = str(pdb_key).lower()
        matched = sampling_inputs_df[
            sampling_inputs_df["pdb_key"].astype(str).str.lower() == pdb_key_normalized
        ]
        if not matched.empty:
            return matched.iloc[0]

    if pdb_id is not None and "pdb_id" in sampling_inputs_df.columns:
        pdb_id_normalized = str(pdb_id).lower()
        matched = sampling_inputs_df[
            sampling_inputs_df["pdb_id"].astype(str).str.lower() == pdb_id_normalized
        ]
        if not matched.empty:
            return matched.iloc[0]

    return None


def resolve_query_pn_unit_iids_from_sampling_row(row: pd.Series | None) -> list[str]:
    if row is None:
        return []

    for column in ("query_pn_unit_iids", "query_pn_unit_iids_json"):
        if column not in row.index:
            continue
        parsed = parse_query_pn_unit_iids(row[column])
        if len(parsed) > 0:
            return parsed
    return []


def metadata_ccd_code(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (float, np.floating)) and np.isnan(raw_value):
        return None

    ccd_code = str(raw_value).strip()
    if ccd_code == "" or ccd_code.lower() == "nan":
        return None
    return ccd_code


def sampling_ligand_ccd_by_iid(
    sampling_row: pd.Series | None,
    ligand_pn_unit_iids: list[str],
) -> dict[str, str]:
    if sampling_row is None or len(ligand_pn_unit_iids) != 1 or "ccd_code" not in sampling_row.index:
        return {}

    ccd_code = metadata_ccd_code(sampling_row["ccd_code"])
    if ccd_code is None:
        return {}
    return {ligand_pn_unit_iids[0]: ccd_code}


def resolve_query_pn_unit_iids(
    *,
    atom_array: AtomArray,
    sampling_inputs_df: pd.DataFrame | None = None,
    pdb_id: str | None = None,
    pdb_key: str | None = None,
) -> list[str]:
    """Resolve query pn_unit_iids from runtime CSV metadata or atom annotations."""
    parsed = resolve_query_pn_unit_iids_from_sampling_row(
        matched_sampling_input_row(sampling_inputs_df, pdb_id, pdb_key=pdb_key)
    )
    if len(parsed) > 0:
        return parsed

    if "pn_unit_iid" in atom_array.get_annotation_categories():
        return [str(x) for x in np.unique(atom_array.pn_unit_iid).tolist()]

    raise ValueError("pn_unit_iid annotation is required")
