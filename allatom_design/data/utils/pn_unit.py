"""Pure parsing primitives for pn_unit metadata columns.

cfg-free helpers shared by the SD dataset selectors/interface/metadata modules,
the preprocessing pipelines and legacy code:

- pn_unit_iid component splitting and natural-ordered joining,
- contact / partner-list parsing,
- CCD-code normalization, ligand-key construction and missing-atom fraction.
"""

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig


# --------------------------------------------------------------------------- #
# pn_unit_iid string helpers
# --------------------------------------------------------------------------- #
def split_components(iid):
    if not isinstance(iid, str):
        iid = str(iid)
    if ',' in iid:
        return [tok.strip() for tok in iid.split(',') if tok.strip()]
    return [iid.strip()] if iid else []


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]


def join_sorted(ids):
    ids = list(ids) if ids is not None else []
    if not ids:
        return ""
    return ", ".join(sorted(ids, key=natural_key))


# --------------------------------------------------------------------------- #
# Contact / partner-list parsing
# --------------------------------------------------------------------------- #
def parse_contacts_with_distance(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, list):
        items = val
    else:
        s = val.decode('utf-8', 'ignore') if isinstance(val, (bytes, bytearray)) else str(val)
        s = s.strip()
        if not s:
            return []
        try:
            items = json.loads(s)
        except Exception:
            try:
                items = ast.literal_eval(s)
            except Exception:
                return []
    out = []
    for item in items:
        if isinstance(item, dict) and 'pn_unit_iid' in item and item['pn_unit_iid']:
            md = item.get('min_distance', None)
            try:
                md = float(md) if md is not None else None
            except Exception:
                md = None
            out.append((str(item['pn_unit_iid']), md))
    return out


def parse_partner_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(value, list):
        return []
    return value


def contact_within_cutoff(contact: dict, distance_cutoff: float | None) -> bool:
    if distance_cutoff is None:
        return True
    distance = contact.get("min_distance")
    if distance is None:
        return True
    try:
        return float(distance) <= distance_cutoff
    except (TypeError, ValueError):
        return False


def contact_count(contact: dict) -> int:
    for key in ("num_contacts", "count"):
        value = contact.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def parse_pn_unit_iids_value(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            value = stripped
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    return [str(value)] if str(value) else []


# --------------------------------------------------------------------------- #
# CCD-code normalization and ligand keys
# --------------------------------------------------------------------------- #
def _normalize_ccd_codes(codes) -> set[str]:
    if codes is None:
        return set()
    if isinstance(codes, str):
        codes = [codes]
    return {str(code).strip().upper() for code in codes if str(code).strip()}


def _split_ccd_tokens(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_tokens = value
    else:
        raw_tokens = str(value).replace(";", ",").split(",")
    return [str(token).strip().upper() for token in raw_tokens if str(token).strip()]


def series_has_any_exact_ccd(series, codes, index) -> pd.Series:
    code_set = _normalize_ccd_codes(codes)
    if not code_set:
        return pd.Series(True, index=index)
    if series is None:
        return pd.Series(False, index=index)
    return series.reindex(index).apply(lambda value: bool(code_set.intersection(_split_ccd_tokens(value))))


def normalize_ligand_ccd_key(value):
    codes = _split_ccd_tokens(value)
    if not codes:
        return ("ccd", "unknown")
    return ("ccd", ",".join(sorted(codes)))


def _small_molecule_artifact_list_path(cfg: dict | DictConfig) -> Path | None:
    cfg = cfg or {}
    raw_path = cfg.get("small_molecule_artifact_list_path", None)
    if raw_path in (None, ""):
        return None
    return Path(str(raw_path))


def load_small_molecule_artifact_codes(cfg: dict | DictConfig) -> set[str]:
    path = _small_molecule_artifact_list_path(cfg)
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"Small-molecule artifact list not found: {path}")
    artifact_df = pd.read_csv(path, sep="\t", usecols=["ccd_code"])
    codes = _normalize_ccd_codes(artifact_df["ccd_code"].tolist())
    if not codes:
        raise ValueError(f"Small-molecule artifact list has no CCD codes: {path}")
    return codes


def missing_atom_fraction_from_values(resolved, expected) -> float:
    try:
        expected_float = float(expected)
        if expected_float <= 0:
            return np.nan
        return 1.0 - (float(resolved) / expected_float)
    except (TypeError, ValueError):
        return np.nan
