"""Output artifact policy for integrated physics-based ligand evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


SCHEMA_VERSION = 1
META_COLUMNS = [
    "designed_sample_id",
    "input_sample_id",
    "diffusion_idx",
    "cif_path",
    "status",
    "error",
]
PHASE_OUTPUTS = {
    "pb_af3": "pb_af3.csv",
    "glide_inplace": "glide_inplace.csv",
    "glide_redock": "glide_redock.csv",
    "pb_mininplace": "pb_mininplace.csv",
    "pb_redocking": "pb_redocking.csv",
}


def resolve_run_output_dir(
    output_dir: str | Path,
    *,
    array_id: int | str | None = None,
) -> Path:
    """Return the concrete output root for normal or SLURM-array execution."""
    if array_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id not in (None, ""):
            array_id = env_id
    root = Path(output_dir)
    if array_id is not None:
        root = root / "arrays" / f"array_{array_id}"
    return root


def _serialise_path(path: Path) -> str:
    return str(path)


def _ensure_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in META_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    leading = [col for col in META_COLUMNS if col in out.columns]
    trailing = [col for col in out.columns if col not in leading]
    return out[leading + trailing]


def _phase_df(results_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    meta = [col for col in META_COLUMNS if col in results_df.columns]
    phase_cols = [col for col in results_df.columns if col.startswith(f"{prefix}_")]
    return results_df[meta + phase_cols].copy()


def _failure_df(results_df: pd.DataFrame) -> pd.DataFrame:
    error_cols = [
        col
        for col in results_df.columns
        if col == "error" or col.endswith("_error")
    ]
    if results_df.empty:
        columns = [col for col in META_COLUMNS if col in results_df.columns]
        columns += [col for col in error_cols if col not in columns]
        return pd.DataFrame(columns=columns)

    status_failed = pd.Series(False, index=results_df.index)
    if "status" in results_df.columns:
        status_failed = results_df["status"].fillna("") != "success"

    has_error = pd.Series(False, index=results_df.index)
    if error_cols:
        has_error = results_df[error_cols].notna().any(axis=1)
        nonempty = results_df[error_cols].fillna("").astype(str).ne("").any(axis=1)
        has_error = has_error & nonempty

    failed = results_df[status_failed | has_error].copy()
    keep = [col for col in META_COLUMNS if col in failed.columns]
    keep += [col for col in error_cols if col not in keep]
    return failed[keep]


def _config_identity(
    *,
    config_path: str | None,
    config_name: str | None,
) -> dict[str, str | None]:
    return {
        "config_path": config_path,
        "config_name": config_name,
    }


def build_manifest(
    *,
    root: Path,
    cfg_dict: Mapping[str, Any],
    config_path: str | None,
    config_name: str | None,
    input_csv_paths: Mapping[str, str | None],
    selected_df: pd.DataFrame,
    results_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    output_paths: Mapping[str, str],
    selection_summary: Mapping[str, Any] | None,
    array_id: int | str | None,
    array_count: int | str | None,
) -> dict[str, Any]:
    """Build the machine-readable run manifest."""
    status_counts = {}
    if "status" in results_df.columns:
        status_counts = {
            str(key): int(value)
            for key, value in results_df["status"].fillna("missing").value_counts().items()
        }
    return {
        "schema_version": SCHEMA_VERSION,
        **_config_identity(config_path=config_path, config_name=config_name),
        "array_id": None if array_id is None else str(array_id),
        "array_count": None if array_count is None else str(array_count),
        "input_csv_paths": dict(input_csv_paths),
        "row_counts": {
            "selection_rows": int(len(selected_df)),
            "result_rows": int(len(results_df)),
            "failure_rows": int(len(failures_df)),
        },
        "status_counts": status_counts,
        "selected_count": int(len(selected_df)),
        "pass_count": int(status_counts.get("success", 0)),
        "failure_count": int(len(failures_df)),
        "selection": {
            "protein_quality_fallback_count": int(
                (selection_summary or {}).get("protein_quality_fallback_count", 0)
            ),
            "protein_quality_conflict_count": int(
                (selection_summary or {}).get("protein_quality_conflict_count", 0)
            ),
            **dict(selection_summary or {}),
        },
        "output_dir": _serialise_path(root),
        "output_paths": dict(output_paths),
        "config": dict(cfg_dict),
    }


def write_ligand_eval_outputs(
    *,
    output_dir: str | Path,
    cfg_dict: Mapping[str, Any],
    selected_df: pd.DataFrame,
    results_df: pd.DataFrame,
    config_path: str | None = None,
    config_name: str | None = None,
    input_csv_paths: Mapping[str, str | None] | None = None,
    selection_summary: Mapping[str, Any] | None = None,
    array_id: int | str | None = None,
    array_count: int | str | None = None,
) -> dict[str, Any]:
    """Write config, selection, results, failures, phase CSVs, and manifest."""
    root = resolve_run_output_dir(output_dir, array_id=array_id)
    root.mkdir(parents=True, exist_ok=True)
    phase_dir = root / "phase_metrics"
    phase_dir.mkdir(parents=True, exist_ok=True)
    (root / "work" / "prep").mkdir(parents=True, exist_ok=True)
    (root / "work" / "mininplace").mkdir(parents=True, exist_ok=True)
    (root / "work" / "redocking").mkdir(parents=True, exist_ok=True)

    selected_df = selected_df.copy()
    results_df = _ensure_meta_columns(results_df)

    config_path_out = root / "config.yaml"
    selection_path = root / "selection.csv"
    results_path = root / "results.csv"
    failures_path = root / "failures.csv"
    manifest_path = root / "manifest.json"

    with config_path_out.open("w") as handle:
        yaml.safe_dump(dict(cfg_dict), handle, sort_keys=False)
    selected_df.to_csv(selection_path, index=False)
    results_df.to_csv(results_path, index=False)

    phase_paths: dict[str, str] = {}
    for prefix, filename in PHASE_OUTPUTS.items():
        path = phase_dir / filename
        _phase_df(results_df, prefix).to_csv(path, index=False)
        phase_paths[prefix] = _serialise_path(path)

    failures_df = _failure_df(results_df)
    failures_df.to_csv(failures_path, index=False)

    output_paths = {
        "config": _serialise_path(config_path_out),
        "selection": _serialise_path(selection_path),
        "results": _serialise_path(results_path),
        "failures": _serialise_path(failures_path),
        "manifest": _serialise_path(manifest_path),
        **{f"phase_{key}": value for key, value in phase_paths.items()},
    }
    manifest = build_manifest(
        root=root,
        cfg_dict=cfg_dict,
        config_path=config_path,
        config_name=config_name,
        input_csv_paths=input_csv_paths or {},
        selected_df=selected_df,
        results_df=results_df,
        failures_df=failures_df,
        output_paths=output_paths,
        selection_summary=selection_summary,
        array_id=array_id,
        array_count=array_count,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
