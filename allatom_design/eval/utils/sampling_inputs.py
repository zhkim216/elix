from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd
from biotite.structure import AtomArray


ROLE_SAMPLING_COLUMNS = (
    "pdb_key",
    "binder_pn_unit_iids",
    "context_pn_unit_iids",
    "frame_pn_unit_iids",
    "template_pn_unit_iids",
    "fixed_pos_seq",
    "fixed_pos_scn",
    "fixed_pos_override_seq",
    "pos_restrict_aatype",
)
ROLE_LIST_COLUMNS = (
    "binder_pn_unit_iids",
    "context_pn_unit_iids",
    "frame_pn_unit_iids",
    "template_pn_unit_iids",
)
ROLE_REQUIRED_NONEMPTY_COLUMNS = (
    "pdb_key",
    "binder_pn_unit_iids",
    "frame_pn_unit_iids",
)
ROLE_POS_CONSTRAINT_COLUMNS = (
    "fixed_pos_seq",
    "fixed_pos_scn",
    "fixed_pos_override_seq",
    "pos_restrict_aatype",
)
ROLE_DERIVED_SAMPLE_ID_COLUMN = "_sample_id"
_LEGACY_QUERY_COLUMNS = ("query_pn_unit_iids", "query_pn_unit_iids_json")
_FORBIDDEN_ROLE_INPUT_COLUMNS = (
    "sample_id",
    "pdb_id",
    "protein_binder_pn_unit_iids",
    "protein_template_pn_unit_iids",
)


def _columns(obj: pd.DataFrame | pd.Series) -> set[str]:
    return set(obj.columns if isinstance(obj, pd.DataFrame) else obj.index)


def is_role_sampling_inputs(obj: pd.DataFrame | pd.Series | None) -> bool:
    if obj is None:
        return False
    return set(ROLE_SAMPLING_COLUMNS).issubset(_columns(obj))


def _text_cell(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, (float, np.floating)) and np.isnan(raw_value):
        return ""
    return str(raw_value).strip()


def _deduplicate_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def parse_pn_unit_iids(raw_value: Any) -> list[str]:
    """Parse a list-like pn_unit_iid cell from runtime CSV metadata."""
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
        return [str(x).strip() for x in parsed if str(x).strip() != ""]

    text = str(parsed).strip()
    if text == "":
        return []
    delimiter = ";" if ";" in text else "," if "," in text else None
    if delimiter is None:
        return [text]
    return [part.strip() for part in text.split(delimiter) if part.strip()]


def parse_query_pn_unit_iids(raw_value: Any) -> list[str]:
    """Parse query_pn_unit_iids from a CSV/metadata cell."""
    return parse_pn_unit_iids(raw_value)


def role_query_pn_unit_iids_from_sampling_row(row: pd.Series) -> list[str]:
    """Compatibility projection for legacy code that still consumes query units."""
    return _deduplicate_preserving_order(
        [
            *parse_pn_unit_iids(row["binder_pn_unit_iids"]),
            *parse_pn_unit_iids(row["context_pn_unit_iids"]),
        ]
    )


def role_binder_pn_unit_iids_from_sampling_row(row: pd.Series) -> list[str]:
    return parse_pn_unit_iids(row["binder_pn_unit_iids"])


def role_context_pn_unit_iids_from_sampling_row(row: pd.Series) -> list[str]:
    return parse_pn_unit_iids(row["context_pn_unit_iids"])


def role_frame_pn_unit_iids_from_sampling_row(row: pd.Series) -> list[str]:
    return parse_pn_unit_iids(row["frame_pn_unit_iids"])


def derive_role_sample_id(
    *,
    pdb_key: str,
    binder_pn_unit_iids: list[str],
    context_pn_unit_iids: list[str],
) -> str:
    binder = "-".join(binder_pn_unit_iids)
    context = "-".join(context_pn_unit_iids) if context_pn_unit_iids else "none"
    return f"{pdb_key}_binder_{binder}_context_{context}"


def normalize_role_sampling_inputs_df(
    sampling_inputs_df: pd.DataFrame | None,
    *,
    label: str = "sampling_inputs_csv",
) -> pd.DataFrame | None:
    if sampling_inputs_df is None:
        return None

    columns = set(sampling_inputs_df.columns)
    missing = [column for column in ROLE_SAMPLING_COLUMNS if column not in columns]
    if missing:
        legacy_columns = [column for column in _LEGACY_QUERY_COLUMNS if column in columns]
        if legacy_columns:
            raise ValueError(
                f"{label} uses the legacy query_pn_unit_iids schema "
                f"({legacy_columns}); expected role-schema columns: {list(ROLE_SAMPLING_COLUMNS)}"
            )
        raise ValueError(
            f"{label} is missing required role-schema columns {missing}; "
            f"expected columns: {list(ROLE_SAMPLING_COLUMNS)}"
        )

    forbidden = [column for column in _FORBIDDEN_ROLE_INPUT_COLUMNS if column in columns]
    if forbidden:
        raise ValueError(
            f"{label} contains non-canonical role-schema columns {forbidden}; "
            "derive runtime sample IDs instead of storing sample_id/pdb_id or long role names"
        )

    out = sampling_inputs_df.copy()
    errors: list[str] = []
    sample_ids: list[str] = []
    for column in ROLE_LIST_COLUMNS:
        out[column] = out[column].apply(parse_pn_unit_iids)

    for row_index, row in out.iterrows():
        pdb_key = _text_cell(row["pdb_key"])
        binder_iids = list(row["binder_pn_unit_iids"])
        context_iids = list(row["context_pn_unit_iids"])
        frame_iids = list(row["frame_pn_unit_iids"])
        required_values = {
            "pdb_key": pdb_key,
            "binder_pn_unit_iids": binder_iids,
            "frame_pn_unit_iids": frame_iids,
        }
        for column in ROLE_REQUIRED_NONEMPTY_COLUMNS:
            if not required_values[column]:
                errors.append(f"row {row_index} ({pdb_key}): blank {column}")
        sample_ids.append(
            derive_role_sample_id(
                pdb_key=pdb_key,
                binder_pn_unit_iids=binder_iids,
                context_pn_unit_iids=context_iids,
            )
        )
        out.at[row_index, "pdb_key"] = pdb_key

    if errors:
        raise ValueError(f"Invalid {label} role-schema rows:\n" + "\n".join(errors[:20]))

    out[ROLE_DERIVED_SAMPLE_ID_COLUMN] = sample_ids
    duplicated = out[out[ROLE_DERIVED_SAMPLE_ID_COLUMN].duplicated(keep=False)]
    if not duplicated.empty:
        duplicate_ids = duplicated[ROLE_DERIVED_SAMPLE_ID_COLUMN].astype(str).tolist()
        raise ValueError(
            f"{label} derived sample IDs must be unique; duplicates: {duplicate_ids[:10]}"
        )
    return out


def role_sampling_inputs_to_pos_constraint_df(
    sampling_inputs_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if sampling_inputs_df is None or not is_role_sampling_inputs(sampling_inputs_df):
        return None
    role_df = normalize_role_sampling_inputs_df(sampling_inputs_df)

    rows: list[dict[str, str]] = []
    for _, row in role_df.iterrows():
        constraint_values = {
            column: _text_cell(row[column])
            for column in ROLE_POS_CONSTRAINT_COLUMNS
        }
        if not any(constraint_values.values()):
            continue
        rows.append(
            {
                "pdb_key": str(row[ROLE_DERIVED_SAMPLE_ID_COLUMN]),
                **constraint_values,
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows, columns=["pdb_key", *ROLE_POS_CONSTRAINT_COLUMNS])


def merge_role_pos_constraints(
    *,
    pos_constraint_df: pd.DataFrame | None,
    sampling_inputs_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    role_pos_constraint_df = role_sampling_inputs_to_pos_constraint_df(sampling_inputs_df)
    if role_pos_constraint_df is None:
        if sampling_inputs_df is not None and is_role_sampling_inputs(sampling_inputs_df):
            if pos_constraint_df is not None:
                raise ValueError(
                    "role-schema sampling_inputs_csv owns positional constraints; "
                    "set pos_constraint_csv=null and put fixed_pos_* columns in sampling_inputs_csv"
                )
            return None
        return pos_constraint_df
    if pos_constraint_df is not None:
        raise ValueError(
            "role-schema sampling_inputs_csv owns positional constraints; "
            "set pos_constraint_csv=null and put fixed_pos_* columns in sampling_inputs_csv"
        )
    return role_pos_constraint_df


def matched_sampling_input_row(
    sampling_inputs_df: pd.DataFrame | None,
    pdb_id: str | None,
    pdb_key: str | None = None,
    sample_id: str | None = None,
) -> pd.Series | None:
    if sampling_inputs_df is None:
        return None

    if is_role_sampling_inputs(sampling_inputs_df):
        if sample_id is None:
            raise ValueError(
                "role-schema sampling_inputs row lookup requires a derived sample_id; "
                "pdb_key alone may be duplicated"
            )
        role_df = normalize_role_sampling_inputs_df(sampling_inputs_df)
        matched = role_df[
            role_df[ROLE_DERIVED_SAMPLE_ID_COLUMN].astype(str) == str(sample_id)
        ]
        if matched.empty:
            raise KeyError(f"derived sample_id {sample_id!r} not found in role-schema sampling_inputs")
        return matched.iloc[0]

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


def matched_role_sampling_input_row(
    sampling_inputs_df: pd.DataFrame | None,
    *,
    sample_id: str,
) -> pd.Series | None:
    if sampling_inputs_df is None:
        return None
    if not is_role_sampling_inputs(sampling_inputs_df):
        return None
    return matched_sampling_input_row(
        sampling_inputs_df,
        pdb_id=None,
        sample_id=sample_id,
    )


def resolve_query_pn_unit_iids_from_sampling_row(row: pd.Series | None) -> list[str]:
    if row is None:
        return []

    if is_role_sampling_inputs(row):
        return role_query_pn_unit_iids_from_sampling_row(row)

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
    sample_id: str | None = None,
) -> list[str]:
    """Resolve query pn_unit_iids from runtime CSV metadata or atom annotations."""
    parsed = resolve_query_pn_unit_iids_from_sampling_row(
        matched_sampling_input_row(
            sampling_inputs_df,
            pdb_id,
            pdb_key=pdb_key,
            sample_id=sample_id,
        )
    )
    if len(parsed) > 0:
        return parsed

    if "pn_unit_iid" in atom_array.get_annotation_categories():
        return [str(x) for x in np.unique(atom_array.pn_unit_iid).tolist()]

    raise ValueError("pn_unit_iid annotation is required")
