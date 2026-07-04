"""Sample selection and AF3 prediction resolution for ligand evaluation."""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

logger = logging.getLogger(__name__)


METADATA_LENGTH_COLUMN = "protein_sequence_length_primary"
METADATA_KEY_COLUMNS = ("example_id", "pdb_id", "local_cif_path", "path")
REQUIRED_RETRY_COLUMNS = ("designed_sample_id", "diffusion_idx")

_LENGTH_PATTERN = re.compile(r"(?:^|_)len_?(\d+)(?:_|$)")
_SAMPLE_SUFFIX = re.compile(r"_(?:sample|psample|ssample)\d+$")
_BASELINE_SEQ_SUFFIX = re.compile(r"_\d+$")


@dataclass(frozen=True)
class _MetadataEntry:
    length: int
    row_index: int
    source_column: str
    raw_value: str


@dataclass
class _ProteinLengthMetadata:
    by_key: dict[str, list[_MetadataEntry]]

    def lookup(self, candidates: Iterable[str]) -> tuple[int, str] | None:
        for candidate in candidates:
            key = _canonical_key(candidate)
            entries = self.by_key.get(key)
            if not entries:
                continue
            lengths = {entry.length for entry in entries}
            if len(lengths) > 1:
                rows = ", ".join(
                    f"row={entry.row_index} {entry.source_column}={entry.raw_value!r} "
                    f"length={entry.length}"
                    for entry in entries
                )
                raise ValueError(
                    "Conflicting protein lengths in metadata for key "
                    f"{candidate!r}: {rows}"
                )
            entry = entries[0]
            return entry.length, f"metadata:{entry.source_column}:{candidate}"
        return None


def _canonical_key(value: str) -> str:
    return str(value).strip().lower()


def normalize_sample_id(sample_id: Any, is_baseline: bool = False) -> str:
    """Normalize per-sample ids to the underlying scaffold/design key."""
    if sample_id is None or (isinstance(sample_id, float) and math.isnan(sample_id)):
        return ""
    out = str(sample_id).strip()
    changed = True
    while changed:
        new = _SAMPLE_SUFFIX.sub("", out)
        changed = new != out
        out = new
    if is_baseline:
        out = _BASELINE_SEQ_SUFFIX.sub("", out)
    return out


def parse_protein_length_from_id(sample_id: Any) -> int | None:
    """Parse denovoval-style ``len150`` or ``len_150`` tokens from an id."""
    if sample_id is None or (isinstance(sample_id, float) and math.isnan(sample_id)):
        return None
    match = _LENGTH_PATTERN.search(str(sample_id))
    return int(match.group(1)) if match else None


def _path_stem(value: Any) -> str:
    text = str(value).strip()
    return Path(text).stem if text else ""


def _id_candidates(value: Any, is_baseline: bool) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    raw = str(value).strip()
    if not raw:
        return []
    candidates = [raw]
    stem = _path_stem(raw)
    if stem and stem != raw:
        candidates.append(stem)
    for item in list(candidates):
        normalized = normalize_sample_id(item, is_baseline=False)
        if normalized:
            candidates.append(normalized)
        if is_baseline:
            baseline = normalize_sample_id(item, is_baseline=True)
            if baseline:
                candidates.append(baseline)
    return list(dict.fromkeys(candidates))


def load_protein_length_metadata(
    metadata_csv: str | Path | None,
    *,
    is_baseline: bool = False,
) -> _ProteinLengthMetadata | None:
    """Load optional protein-length metadata and index normalized key variants."""
    if not metadata_csv:
        return None

    metadata_path = Path(metadata_csv)
    df = pd.read_csv(metadata_path)
    if METADATA_LENGTH_COLUMN not in df.columns:
        raise ValueError(
            f"metadata_csv must contain {METADATA_LENGTH_COLUMN!r}: {metadata_path}"
        )

    key_columns = [col for col in METADATA_KEY_COLUMNS if col in df.columns]
    if not key_columns:
        raise ValueError(
            "metadata_csv must contain at least one key column among "
            f"{', '.join(METADATA_KEY_COLUMNS)}: {metadata_path}"
        )

    by_key: dict[str, list[_MetadataEntry]] = {}
    for row_index, row in df.iterrows():
        raw_length = row[METADATA_LENGTH_COLUMN]
        if pd.isna(raw_length):
            continue
        length = int(float(raw_length))
        for source_column in key_columns:
            raw_value = row[source_column]
            if pd.isna(raw_value):
                continue
            for candidate in _id_candidates(raw_value, is_baseline=is_baseline):
                key = _canonical_key(candidate)
                entry = _MetadataEntry(
                    length=length,
                    row_index=int(row_index),
                    source_column=source_column,
                    raw_value=str(raw_value),
                )
                by_key.setdefault(key, []).append(entry)

    return _ProteinLengthMetadata(by_key=by_key)


def _metric_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected diffusion metric dict, got {type(parsed)}")
    return parsed


def load_af3_metrics(
    docking_csv_path: str | Path,
    sc_csv_path: str | Path,
) -> pd.DataFrame:
    """Load AF3 docking and self-consistency metrics as one row per diffusion."""
    dock_df = pd.read_csv(docking_csv_path)
    sc_df = pd.read_csv(sc_csv_path)
    sc_indexed = sc_df.set_index("designed_sample_id")

    records: list[dict[str, Any]] = []
    diffusion_cols = [c for c in dock_df.columns if c.startswith("diffusion_")]
    for _, row in dock_df.iterrows():
        designed_id = row["designed_sample_id"]
        input_id = row.get("input_sample_id", "")
        if designed_id not in sc_indexed.index:
            raise KeyError(f"SC metrics missing designed_sample_id={designed_id!r}")
        sc_row = sc_indexed.loc[designed_id]
        if isinstance(sc_row, pd.DataFrame):
            sc_row = sc_row.iloc[0]

        for col in diffusion_cols:
            diffusion_idx = int(col.rsplit("_", 1)[1])
            records.append({
                "designed_sample_id": designed_id,
                "input_sample_id": input_id,
                "diffusion_idx": diffusion_idx,
                **_metric_dict(row[col]),
                **_metric_dict(sc_row[col]),
            })

    return pd.DataFrame(records)


def find_af3_prediction_path(
    af3_preds_dir: str | Path | None,
    designed_sample_id: str,
    diffusion_idx: int,
    *,
    seed: int = 42,
) -> str | None:
    """Find the AF3 prediction CIF for ``designed_sample_id`` and diffusion."""
    if not af3_preds_dir:
        return None

    seed_dir = f"seed-{seed}_sample-{int(diffusion_idx)}"
    preds_dir = Path(af3_preds_dir)
    designed_sample_id = str(designed_sample_id)

    id_variants = [
        designed_sample_id,
        re.sub(r"_len_(\d+)_", r"_len\1_", designed_sample_id),
        re.sub(r"_len(\d+)_", r"_len_\1_", designed_sample_id),
    ]
    for sample_id in dict.fromkeys(id_variants):
        exact = preds_dir / sample_id / seed_dir
        if exact.is_dir():
            cifs = sorted(exact.glob("*_model_pocket_aligned.cif"))
            if cifs:
                return str(cifs[0])

    prefix = designed_sample_id.split("_")[0]
    matches = sorted(
        preds_dir.glob(f"*{prefix}*/{seed_dir}/*_model_pocket_aligned.cif")
    )
    if matches:
        return str(matches[0])

    logger.warning(
        "AF3 prediction not found: %s diffusion_%s", designed_sample_id, diffusion_idx
    )
    return None


def _resolve_cif_paths(
    df: pd.DataFrame,
    af3_preds_dir: str | Path | None,
    *,
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    if "cif_path" not in out.columns:
        out["cif_path"] = pd.NA

    missing = out["cif_path"].isna() | (out["cif_path"].astype(str).str.len() == 0)
    for idx in out.index[missing]:
        out.at[idx, "cif_path"] = find_af3_prediction_path(
            af3_preds_dir,
            out.at[idx, "designed_sample_id"],
            int(out.at[idx, "diffusion_idx"]),
            seed=seed,
        )
    return out


def _resolve_protein_length(
    row: pd.Series,
    metadata: _ProteinLengthMetadata | None,
    *,
    is_baseline: bool,
) -> tuple[int | None, str]:
    ordered_values = [
        row.get("input_sample_id"),
        row.get("designed_sample_id"),
        _path_stem(row.get("cif_path")),
    ]

    if metadata is not None:
        metadata_candidates: list[str] = []
        for value in ordered_values:
            metadata_candidates.extend(_id_candidates(value, is_baseline=is_baseline))
        match = metadata.lookup(metadata_candidates)
        if match is not None:
            return match

    for source, value in [
        ("input_sample_id", row.get("input_sample_id")),
        ("designed_sample_id", row.get("designed_sample_id")),
        ("cif_path", _path_stem(row.get("cif_path"))),
    ]:
        for candidate in _id_candidates(value, is_baseline=is_baseline):
            length = parse_protein_length_from_id(candidate)
            if length is not None:
                return length, source

    return None, "fallback"


def _protein_quality_thresholds(
    protein_length: int | None,
    protein_quality_cfg: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    short_max_length = int(protein_quality_cfg.get("short_max_length", 250))
    if protein_length is None or protein_length <= short_max_length:
        return "short", protein_quality_cfg.get("short", {})
    return "long", protein_quality_cfg.get("long", {})


def _passes_protein_quality(
    row: pd.Series,
    thresholds: Mapping[str, Any],
) -> bool:
    try:
        ca_plddt_cutoff = float(thresholds["ca_plddt_cutoff"])
        sc_rmsd_cutoff = float(thresholds["sc_rmsd_cutoff"])
    except KeyError as exc:
        raise KeyError(f"Missing protein quality threshold: {exc}") from exc

    avg_ca_plddt = row.get("avg_ca_plddt")
    sc_ca_rmsd = row.get("sc_ca_rmsd")
    if pd.isna(avg_ca_plddt) or pd.isna(sc_ca_rmsd):
        return False
    return float(avg_ca_plddt) >= ca_plddt_cutoff and float(sc_ca_rmsd) <= sc_rmsd_cutoff


def annotate_protein_quality(
    df: pd.DataFrame,
    *,
    protein_quality_cfg: Mapping[str, Any],
    metadata: _ProteinLengthMetadata | None = None,
    is_baseline: bool = False,
) -> pd.DataFrame:
    """Add protein-length and protein-quality provenance columns."""
    rows: list[dict[str, Any]] = []
    enabled = bool(protein_quality_cfg.get("enabled", True))

    for _, row in df.iterrows():
        length, source = _resolve_protein_length(
            row,
            metadata,
            is_baseline=is_baseline,
        )
        tier, thresholds = _protein_quality_thresholds(length, protein_quality_cfg)
        quality_pass = True
        if enabled:
            quality_pass = _passes_protein_quality(row, thresholds)

        rows.append({
            "protein_length": pd.NA if length is None else int(length),
            "protein_quality_source": source,
            "protein_quality_tier": tier,
            "protein_quality_pass": bool(quality_pass),
        })

    annotated = df.copy()
    if rows:
        quality_df = pd.DataFrame(rows, index=annotated.index)
        for col in quality_df.columns:
            annotated[col] = quality_df[col]
    else:
        annotated["protein_length"] = pd.Series(dtype="Int64")
        annotated["protein_quality_source"] = pd.Series(dtype="object")
        annotated["protein_quality_tier"] = pd.Series(dtype="object")
        annotated["protein_quality_pass"] = pd.Series(dtype="bool")
    return annotated


def _default_protein_quality_cfg() -> dict[str, Any]:
    return {
        "enabled": True,
        "metadata_csv": None,
        "short_max_length": 250,
        "short": {"sc_rmsd_cutoff": 2.0, "ca_plddt_cutoff": 80},
        "long": {"sc_rmsd_cutoff": 3.0, "ca_plddt_cutoff": 70},
    }


def _protein_quality_cfg(selection_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = _default_protein_quality_cfg()
    user_cfg = dict(selection_cfg.get("protein_quality", {}) or {})
    for key, value in user_cfg.items():
        if isinstance(value, Mapping) and isinstance(cfg.get(key), Mapping):
            cfg[key] = {**cfg[key], **dict(value)}
        else:
            cfg[key] = value
    return cfg


def _selection_summary(
    df: pd.DataFrame,
    *,
    mode: str,
    flat_rows: int,
    selected_rows_before_debug_or_array: int,
) -> dict[str, Any]:
    fallback_count = 0
    missing_cif_count = 0
    protein_quality_pass_count = 0
    if "protein_quality_source" in df.columns:
        fallback_count = int((df["protein_quality_source"] == "fallback").sum())
    if "cif_path" in df.columns:
        missing_cif_mask = df["cif_path"].isna() | (
            df["cif_path"].fillna("").astype(str).str.len() == 0
        )
        missing_cif_count = int(missing_cif_mask.sum())
    if "protein_quality_pass" in df.columns:
        protein_quality_pass_count = int(df["protein_quality_pass"].fillna(False).sum())
    return {
        "mode": mode,
        "flat_rows": int(flat_rows),
        "selected_rows_before_debug_or_array": int(selected_rows_before_debug_or_array),
        "protein_quality_pass_count": protein_quality_pass_count,
        "protein_quality_fallback_count": fallback_count,
        "protein_quality_conflict_count": 0,
        "missing_cif_path_count": missing_cif_count,
    }


def select_best_diffusion(
    flat_df: pd.DataFrame,
    *,
    selection_cfg: Mapping[str, Any],
    metadata: _ProteinLengthMetadata | None,
    af3_preds_dir: str | Path | None,
) -> pd.DataFrame:
    """Select one resolved diffusion row per normalized input scaffold."""
    if flat_df.empty:
        return flat_df.copy()

    seed = int(selection_cfg.get("seed", 42))
    is_baseline = bool(selection_cfg.get("is_baseline", False))
    protein_quality_cfg = _protein_quality_cfg(selection_cfg)

    df = _resolve_cif_paths(flat_df, af3_preds_dir, seed=seed)
    df = annotate_protein_quality(
        df,
        protein_quality_cfg=protein_quality_cfg,
        metadata=metadata,
        is_baseline=is_baseline,
    )

    if protein_quality_cfg.get("enabled", True):
        df = df[df["protein_quality_pass"].fillna(False)]
        if df.empty:
            logger.warning("No rows survive protein quality filter")
            return df.reset_index(drop=True)

    if "ligand_plddt" not in df.columns:
        raise KeyError("flat_df missing 'ligand_plddt' column required for selection")

    best_diff_idx = df.groupby("designed_sample_id")["ligand_plddt"].idxmax()
    per_design = df.loc[best_diff_idx].copy()
    per_design["_norm_input_id"] = per_design["input_sample_id"].map(
        lambda value: normalize_sample_id(value, is_baseline=is_baseline)
    )
    best_input_idx = per_design.groupby("_norm_input_id")["ligand_plddt"].idxmax()
    per_input = per_design.loc[best_input_idx].drop(columns="_norm_input_id")

    mask = pd.Series(True, index=per_input.index)
    ligand_rmsd_cutoff = selection_cfg.get("ligand_rmsd_cutoff", 2.0)
    ligand_plddt_cutoff = selection_cfg.get("ligand_plddt_cutoff", 80)
    if ligand_rmsd_cutoff is not None and "ligand_rmsd" in per_input.columns:
        mask &= per_input["ligand_rmsd"] <= float(ligand_rmsd_cutoff)
    if ligand_plddt_cutoff is not None:
        mask &= per_input["ligand_plddt"] >= float(ligand_plddt_cutoff)

    return per_input[mask].reset_index(drop=True)


def _load_retry_rows(
    retry_csv: str | Path,
    *,
    selection_cfg: Mapping[str, Any],
    metadata: _ProteinLengthMetadata | None,
    af3_preds_dir: str | Path | None,
) -> pd.DataFrame:
    retry_df = pd.read_csv(retry_csv)
    missing = [col for col in REQUIRED_RETRY_COLUMNS if col not in retry_df.columns]
    if missing:
        raise ValueError("retry_csv missing required column(s): " + ", ".join(missing))
    if "input_sample_id" not in retry_df.columns:
        retry_df["input_sample_id"] = ""

    seed = int(selection_cfg.get("seed", 42))
    is_baseline = bool(selection_cfg.get("is_baseline", False))
    protein_quality_cfg = _protein_quality_cfg(selection_cfg)
    retry_df = _resolve_cif_paths(retry_df, af3_preds_dir, seed=seed)
    retry_df = annotate_protein_quality(
        retry_df,
        protein_quality_cfg=protein_quality_cfg,
        metadata=metadata,
        is_baseline=is_baseline,
    )
    return retry_df.reset_index(drop=True)


def select_samples_for_eval(
    *,
    af3_eval_dir: str | Path,
    selection_cfg: Mapping[str, Any] | None,
    retry_csv: str | Path | None = None,
    af3_preds_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Select resolved ligand-eval rows from AF3 metric CSVs or a retry CSV."""
    selection_cfg = dict(selection_cfg or {})
    protein_quality_cfg = _protein_quality_cfg(selection_cfg)
    selection_cfg["protein_quality"] = protein_quality_cfg
    is_baseline = bool(selection_cfg.get("is_baseline", False))
    metadata = load_protein_length_metadata(
        protein_quality_cfg.get("metadata_csv"),
        is_baseline=is_baseline,
    )

    if af3_preds_dir is None:
        af3_preds_dir = Path(af3_eval_dir) / "af3_ss_preds"

    if retry_csv:
        selected = _load_retry_rows(
            retry_csv,
            selection_cfg=selection_cfg,
            metadata=metadata,
            af3_preds_dir=af3_preds_dir,
        )
        selected.attrs["selection_summary"] = _selection_summary(
            selected,
            mode="retry",
            flat_rows=len(selected),
            selected_rows_before_debug_or_array=len(selected),
        )
        return selected

    af3_eval_dir = Path(af3_eval_dir)
    docking_csv = af3_eval_dir / "all_docking_metrics_per_designed_sample.csv"
    sc_csv = af3_eval_dir / "all_sc_metrics_per_designed_sample.csv"
    flat_df = load_af3_metrics(docking_csv, sc_csv)

    selected = select_best_diffusion(
        flat_df,
        selection_cfg=selection_cfg,
        metadata=metadata,
        af3_preds_dir=af3_preds_dir,
    )
    selected.attrs["selection_summary"] = _selection_summary(
        selected,
        mode="selection",
        flat_rows=len(flat_df),
        selected_rows_before_debug_or_array=len(selected),
    )
    return selected
