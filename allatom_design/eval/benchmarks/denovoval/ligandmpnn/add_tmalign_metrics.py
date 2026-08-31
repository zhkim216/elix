"""Backfill TM-align metrics into completed LigandMPNN AF3 result chunks.

The canonical self-consistency CSVs are replaced only after every row in a
chunk succeeds and the enriched frame passes validation. The original chunk is
preserved once as a read-only backup; failed chunks retain a staging CSV and a
diagnostic JSON without modifying the canonical CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.denovoval.ligandmpnn.run_af3_ss import (
    DEFAULT_CONFIG,
    atomic_write_json,
    cfg_path,
    load_reference,
    validate_backmapped_cif_digest,
    validate_design_manifest,
)
from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    sha256_file,
)
from allatom_design.eval.metrics.self_consistency import (
    compute_tmalign_metrics_atomarray,
)
from allatom_design.eval.structure_prediction.AF3.inputs import prepare_af3_prediction


BASE_COLUMNS = (
    "designed_sample_id",
    "source_sample_id",
    "model_name",
    "design_index",
    "ccd_code",
    "diffusion_index",
    "prediction_cif",
    "sc_ca_rmsd",
    "avg_ca_plddt",
    "metric_status",
    "error",
)
BASE_FLOAT_COLUMNS = ("sc_ca_rmsd", "avg_ca_plddt")
BASE_EXACT_COLUMNS = tuple(
    column for column in BASE_COLUMNS if column not in BASE_FLOAT_COLUMNS
)
MANIFEST_COLUMNS = (
    "designed_sample_id",
    "source_sample_id",
    "model_name",
    "design_index",
    "ccd_code",
    "protein_sequence",
    "protein_chain_id",
    "ligand_chain_id",
    "backmapped_cif_path",
    "backmapped_cif_sha256",
    "status",
)
TMALIGN_COLUMNS = (
    "tmalign_score",
    "tmalign_matched_ca_count",
    "tmalign_sample_ca_count",
    "tmalign_pred_ca_count",
    "tmalign_sample_ca_coverage",
    "tmalign_pred_ca_coverage",
    "tmalign_label_span_count",
    "tmalign_excluded_label_positions",
    "tmalign_modified_ca_count",
    "tmaligned_path",
)
TMALIGN_NUMERIC_COLUMNS = TMALIGN_COLUMNS[:-1]
ROW_METADATA_COLUMNS = (
    "source_sample_id",
    "model_name",
    "design_index",
    "ccd_code",
)
CHUNK_PATTERN = re.compile(r"self_consistency_(\d{4})\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--array-id", type=int, default=None)
    parser.add_argument("--num-arrays", type=int, default=85)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--print-plan", action="store_true")
    modes.add_argument("--process-chunk", action="store_true")
    modes.add_argument("--validate-all", action="store_true")
    return parser.parse_args()


def _resolve_array_id(value: int | None) -> int | None:
    if value is not None:
        return value
    env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(env_value) if env_value is not None else None


def _paths(output_dir: Path, array_id: int) -> dict[str, Path]:
    filename = f"self_consistency_{array_id:04d}.csv"
    return {
        "canonical": output_dir / "metrics" / "chunks" / filename,
        "backup": output_dir / "metrics" / "backups" / "pre_tmalign" / filename,
        "staging": output_dir / "metrics" / "tmalign_staging" / filename,
        "diagnostic": (
            output_dir
            / "reports"
            / "tmalign_chunks"
            / f"tmalign_{array_id:04d}.json"
        ),
    }


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_backup(source: Path, backup: Path, expected_sha256: str) -> str:
    backup.parent.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        observed = sha256_file(backup)
        if observed != expected_sha256:
            raise ValueError(
                f"Existing immutable backup digest mismatch: {backup}: "
                f"{observed} != {expected_sha256}"
            )
        return observed

    temporary = backup.with_name(f".{backup.name}.tmp.{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        observed = sha256_file(temporary)
        if observed != expected_sha256:
            raise ValueError(
                f"Temporary backup digest mismatch: {observed} != {expected_sha256}"
            )
        temporary.chmod(0o444)
        try:
            os.link(temporary, backup)
        except FileExistsError:
            observed = sha256_file(backup)
            if observed != expected_sha256:
                raise ValueError(
                    f"Concurrent immutable backup digest mismatch: {backup}"
                )
        return sha256_file(backup)
    finally:
        temporary.unlink(missing_ok=True)


def _load_manifest(cfg: DictConfig) -> tuple[pd.DataFrame, Path]:
    path = cfg_path(cfg, "backmapping.full_manifest_csv")
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(MANIFEST_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame["designed_sample_id"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate designed_sample_id in {path}")
    if not frame["status"].isin(["backmapped", "reused"]).all():
        raise ValueError(f"Design manifest contains failed rows: {path}")
    validate_design_manifest(frame, cfg, smoke=False)
    return frame.reset_index(drop=True), path


def _expected_design_ids(
    manifest: pd.DataFrame,
    *,
    array_id: int,
    num_arrays: int,
) -> list[str]:
    if num_arrays < 1:
        raise ValueError("num_arrays must be positive")
    if not 0 <= array_id < num_arrays:
        raise ValueError(f"array_id {array_id} is outside 0..{num_arrays - 1}")
    chunk_size = max(1, math.ceil(len(manifest) / num_arrays))
    start = min(len(manifest), array_id * chunk_size)
    end = min(len(manifest), start + chunk_size)
    return list(map(str, manifest.iloc[start:end]["designed_sample_id"]))


def _load_chunk(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(BASE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    present_tm = set(TMALIGN_COLUMNS) & set(frame.columns)
    if present_tm and present_tm != set(TMALIGN_COLUMNS):
        raise ValueError(
            f"{path} has a partial TM-align schema: "
            f"present={sorted(present_tm)}, missing={sorted(set(TMALIGN_COLUMNS) - present_tm)}"
        )
    return frame


def _prediction_count(cfg: DictConfig) -> int:
    return int(cfg.af3.inference_config.ss.num_diffusion_samples)


def _validate_chunk_contract(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    array_id: int,
    num_arrays: int,
    predictions_per_design: int,
) -> None:
    expected_ids = _expected_design_ids(
        manifest,
        array_id=array_id,
        num_arrays=num_arrays,
    )
    expected_keys = {
        (designed_sample_id, diffusion_index)
        for designed_sample_id in expected_ids
        for diffusion_index in range(predictions_per_design)
    }
    observed_keys = list(
        zip(
            map(str, frame["designed_sample_id"]),
            map(int, frame["diffusion_index"]),
            strict=True,
        )
    )
    observed_counts = Counter(observed_keys)
    if set(observed_counts) != expected_keys or any(
        count != 1 for count in observed_counts.values()
    ):
        missing = sorted(expected_keys - set(observed_counts))[:20]
        extra = sorted(set(observed_counts) - expected_keys)[:20]
        duplicated = sorted(key for key, count in observed_counts.items() if count != 1)[:20]
        raise ValueError(
            "Chunk design/diffusion matrix mismatch: "
            f"rows={len(frame)} expected={len(expected_keys)}, missing={missing}, "
            f"extra={extra}, duplicated={duplicated}"
        )
    if not frame["metric_status"].astype(str).eq("ok").all():
        raise ValueError("Source self-consistency chunk contains non-ok metric rows")
    if frame["error"].astype(str).str.len().sum():
        raise ValueError("Source self-consistency chunk contains nonempty errors")

    manifest_by_id = manifest.set_index("designed_sample_id", drop=False)
    for row in frame.itertuples(index=False):
        designed_sample_id = str(row.designed_sample_id)
        record = manifest_by_id.loc[designed_sample_id]
        for column in ROW_METADATA_COLUMNS:
            observed = getattr(row, column)
            expected = record[column]
            if column == "design_index":
                observed, expected = int(observed), int(expected)
            else:
                observed, expected = str(observed), str(expected)
            if observed != expected:
                raise ValueError(
                    f"Chunk/manifest mismatch for {designed_sample_id}.{column}: "
                    f"{observed!r} != {expected!r}"
                )


def _validate_prediction_path(
    prediction_cif: str | Path,
    *,
    output_dir: Path,
    designed_sample_id: str,
    diffusion_index: int,
) -> Path:
    path = Path(prediction_cif)
    if not path.is_file():
        raise FileNotFoundError(path)
    prediction_root = (output_dir / "predictions").resolve()
    try:
        relative = path.resolve().relative_to(prediction_root)
    except ValueError as exc:
        raise ValueError(f"Prediction path is outside {prediction_root}: {path}") from exc
    if not relative.parts or relative.parts[0] != designed_sample_id:
        raise ValueError(
            f"Prediction directory does not match {designed_sample_id}: {path}"
        )
    if not re.fullmatch(rf"seed-.+_sample-{diffusion_index}", path.parent.name):
        raise ValueError(
            f"Prediction sample directory does not match diffusion index "
            f"{diffusion_index}: {path.parent.name}"
        )
    if not path.stem.endswith("_model"):
        raise ValueError(f"Prediction CIF is not an AF3 model CIF: {path}")
    return path


def _compute_row_tmalign(
    *,
    prediction_cif: Path,
    reference_atom_array: Any,
    cfg: DictConfig,
) -> dict[str, Any]:
    prediction = prepare_af3_prediction(
        pdb_path=str(prediction_cif),
        cif_parse_cfg=cfg.cif_parse_cfg.af3_predictions,
        preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
        featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
    )
    return compute_tmalign_metrics_atomarray(
        pred_atom_array=prediction["atom_array"],
        sample_atom_array=reference_atom_array,
        pred_sample_path=prediction_cif,
        save_tmaligned=True,
    )


def _expected_tmaligned_path(prediction_cif: str | Path) -> Path:
    prediction_cif = Path(prediction_cif)
    return prediction_cif.with_name(f"{prediction_cif.stem}_tmaligned.cif")


def _validate_enriched_frame(frame: pd.DataFrame) -> None:
    missing = sorted(set(TMALIGN_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Enriched chunk is missing TM-align columns: {missing}")
    if frame[list(TMALIGN_COLUMNS)].isna().any().any():
        raise ValueError("Enriched chunk contains missing TM-align values")
    if frame["tmaligned_path"].astype(str).str.len().eq(0).any():
        raise ValueError("Enriched chunk contains empty tmaligned_path values")

    numeric = frame[list(TMALIGN_NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Enriched chunk contains non-finite TM-align values")
    score = numeric["tmalign_score"].to_numpy(dtype=float)
    if np.any((score < 0.0) | (score > 1.00001)):
        raise ValueError("TM-score is outside [0, 1]")
    matched = numeric["tmalign_matched_ca_count"].to_numpy(dtype=float)
    sample_count = numeric["tmalign_sample_ca_count"].to_numpy(dtype=float)
    pred_count = numeric["tmalign_pred_ca_count"].to_numpy(dtype=float)
    if np.any(matched < 3) or np.any(matched > sample_count) or np.any(matched > pred_count):
        raise ValueError("TM-align CA counts are inconsistent")
    for column in ("tmalign_sample_ca_coverage", "tmalign_pred_ca_coverage"):
        values = numeric[column].to_numpy(dtype=float)
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{column} is outside [0, 1]")
    if np.any(numeric["tmalign_excluded_label_positions"].to_numpy(dtype=float) < 0):
        raise ValueError("tmalign_excluded_label_positions is negative")

    for row in frame.itertuples(index=False):
        path = Path(str(row.tmaligned_path))
        expected = _expected_tmaligned_path(row.prediction_cif)
        if path != expected:
            raise ValueError(f"Unexpected tmaligned_path: {path} != {expected}")
        if not path.is_file():
            raise FileNotFoundError(path)


def _validate_backup_matches_canonical_base(
    backup: pd.DataFrame,
    canonical: pd.DataFrame,
) -> None:
    if set(TMALIGN_COLUMNS) & set(backup.columns):
        raise ValueError("Immutable backup contains TM-align columns")
    if len(backup) != len(canonical):
        raise ValueError("Immutable backup row count differs from canonical chunk")
    if not backup[list(BASE_EXACT_COLUMNS)].equals(
        canonical[list(BASE_EXACT_COLUMNS)]
    ):
        raise ValueError(
            "Immutable backup exact base columns differ from canonical chunk"
        )
    for column in BASE_FLOAT_COLUMNS:
        backup_values = pd.to_numeric(backup[column], errors="raise").to_numpy(
            dtype=float
        )
        canonical_values = pd.to_numeric(
            canonical[column], errors="raise"
        ).to_numpy(dtype=float)
        if not np.isfinite(backup_values).all() or not np.isfinite(
            canonical_values
        ).all():
            raise ValueError(
                f"Immutable backup or canonical {column} contains non-finite values"
            )
        if not np.allclose(
            backup_values,
            canonical_values,
            rtol=0.0,
            atol=1e-12,
        ):
            max_abs_error = float(
                np.nanmax(np.abs(backup_values - canonical_values))
            )
            raise ValueError(
                f"Immutable backup {column} differs from canonical chunk: "
                f"max_abs_error={max_abs_error}"
            )


def _empty_diagnostics() -> dict[str, int]:
    return {
        "errors": 0,
        "missing": 0,
        "skipped": 0,
        "unmapped": 0,
        "unmatched": 0,
        "stale": 0,
        "failed_rows": 0,
    }


def _failure_category(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return "missing"
    if "digest mismatch" in message or "stale" in message:
        return "stale"
    if "mismatch" in message or "match" in message or "correspondence" in message:
        return "unmatched"
    return "errors"


def process_chunk(cfg: DictConfig, *, array_id: int, num_arrays: int) -> dict[str, Any]:
    output_dir = cfg_path(cfg, "af3.full_output_dir")
    paths = _paths(output_dir, array_id)
    manifest, manifest_path = _load_manifest(cfg)
    canonical = paths["canonical"]
    if not canonical.is_file():
        raise FileNotFoundError(canonical)
    source_sha256 = sha256_file(canonical)
    frame = _load_chunk(canonical)
    predictions_per_design = _prediction_count(cfg)
    _validate_chunk_contract(
        frame,
        manifest,
        array_id=array_id,
        num_arrays=num_arrays,
        predictions_per_design=predictions_per_design,
    )

    base_report: dict[str, Any] = {
        "stage": "ligandmpnn-af3-tmalign-backfill",
        "array_id": array_id,
        "num_arrays": num_arrays,
        "canonical_csv": str(canonical),
        "backup_csv": str(paths["backup"]),
        "staging_csv": str(paths["staging"]),
        "manifest_csv": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": source_sha256,
        "row_count": len(frame),
        "predictions_per_design": predictions_per_design,
        "diagnostics": _empty_diagnostics(),
        "error_examples": [],
    }

    if set(TMALIGN_COLUMNS).issubset(frame.columns):
        if not paths["backup"].is_file():
            raise ValueError(
                f"Enriched canonical chunk has no immutable backup: {paths['backup']}"
            )
        backup_sha256 = sha256_file(paths["backup"])
        backup = _load_chunk(paths["backup"])
        _validate_backup_matches_canonical_base(backup, frame)
        _validate_chunk_contract(
            backup,
            manifest,
            array_id=array_id,
            num_arrays=num_arrays,
            predictions_per_design=predictions_per_design,
        )
        if not paths["diagnostic"].is_file():
            raise ValueError(
                f"Enriched canonical chunk has no provenance diagnostic: "
                f"{paths['diagnostic']}"
            )
        previous_diagnostic = json.loads(paths["diagnostic"].read_text())
        if previous_diagnostic.get("source_sha256") != backup_sha256:
            raise ValueError("Existing diagnostic source digest differs from backup")
        _validate_enriched_frame(frame)
        report = {
            **base_report,
            "status": "reused",
            "source_sha256": backup_sha256,
            "backup_sha256": backup_sha256,
            "canonical_sha256": source_sha256,
            "tmaligned_files": len(frame),
        }
        atomic_write_json(paths["diagnostic"], report)
        return report

    enriched = frame.copy()
    for column in TMALIGN_COLUMNS:
        enriched[column] = ""
    manifest_by_id = {
        str(row["designed_sample_id"]): row.to_dict()
        for _, row in manifest.iterrows()
    }
    reference_by_id: dict[str, Any] = {}
    reference_errors: dict[str, Exception] = {}
    failures: list[dict[str, Any]] = []

    for index, row in enriched.iterrows():
        designed_sample_id = str(row["designed_sample_id"])
        diffusion_index = int(row["diffusion_index"])
        try:
            prediction_cif = _validate_prediction_path(
                row["prediction_cif"],
                output_dir=output_dir,
                designed_sample_id=designed_sample_id,
                diffusion_index=diffusion_index,
            )
            if designed_sample_id in reference_errors:
                raise reference_errors[designed_sample_id]
            if designed_sample_id not in reference_by_id:
                record = manifest_by_id[designed_sample_id]
                try:
                    validate_backmapped_cif_digest(record)
                    reference_by_id[designed_sample_id] = load_reference(record, cfg)[
                        "atom_array"
                    ]
                except Exception as exc:
                    reference_errors[designed_sample_id] = exc
                    raise
            metrics = _compute_row_tmalign(
                prediction_cif=prediction_cif,
                reference_atom_array=reference_by_id[designed_sample_id],
                cfg=cfg,
            )
            missing_metrics = sorted(set(TMALIGN_COLUMNS) - set(metrics))
            if missing_metrics:
                raise ValueError(
                    f"TM-only metric helper omitted columns: {missing_metrics}"
                )
            for column in TMALIGN_COLUMNS:
                enriched.at[index, column] = metrics[column]
        except Exception as exc:
            failures.append(
                {
                    "designed_sample_id": designed_sample_id,
                    "diffusion_index": diffusion_index,
                    "prediction_cif": str(row["prediction_cif"]),
                    "category": _failure_category(exc),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    _atomic_write_frame(paths["staging"], enriched)
    diagnostics = _empty_diagnostics()
    for failure in failures:
        diagnostics[failure["category"]] += 1
    diagnostics["failed_rows"] = len(failures)
    report = {
        **base_report,
        "status": "failed" if failures else "ready_to_promote",
        "diagnostics": diagnostics,
        "error_examples": failures[:50],
        "tmaligned_files": int(
            sum(Path(str(path)).is_file() for path in enriched["tmaligned_path"] if path)
        ),
    }
    if failures:
        atomic_write_json(paths["diagnostic"], report)
        raise RuntimeError(
            f"TM-align failed for {len(failures)} rows; canonical CSV was not changed; "
            f"see {paths['diagnostic']}"
        )

    try:
        _validate_enriched_frame(enriched)
        backup_sha256 = _write_immutable_backup(
            canonical,
            paths["backup"],
            source_sha256,
        )
    except Exception as exc:
        category = _failure_category(exc)
        report["status"] = "failed"
        report["diagnostics"][category] += 1
        report["error_examples"] = [
            {
                "designed_sample_id": "chunk",
                "diffusion_index": "",
                "prediction_cif": "",
                "category": category,
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]
        atomic_write_json(paths["diagnostic"], report)
        raise RuntimeError(
            "TM-align chunk validation or backup failed; canonical CSV was not "
            f"changed; see {paths['diagnostic']}"
        ) from exc
    report["backup_sha256"] = backup_sha256
    atomic_write_json(paths["diagnostic"], report)
    os.replace(paths["staging"], canonical)
    report.update(
        {
            "status": "complete",
            "canonical_sha256": sha256_file(canonical),
            "tmaligned_files": len(enriched),
        }
    )
    atomic_write_json(paths["diagnostic"], report)
    return report


def _exact_chunk_paths(output_dir: Path, num_arrays: int) -> list[Path]:
    metrics_dir = output_dir / "metrics" / "chunks"
    expected = [
        metrics_dir / f"self_consistency_{array_id:04d}.csv"
        for array_id in range(num_arrays)
    ]
    observed = sorted(metrics_dir.glob("self_consistency_*.csv"))
    unexpected = [
        path for path in observed if CHUNK_PATTERN.fullmatch(path.name) is None or path not in expected
    ]
    missing = sorted(set(expected) - set(observed))
    if missing or unexpected:
        raise ValueError(
            f"Self-consistency chunk set mismatch: missing={list(map(str, missing[:20]))}, "
            f"unexpected={list(map(str, unexpected[:20]))}"
        )
    return expected


def validate_all(cfg: DictConfig, *, num_arrays: int) -> dict[str, Any]:
    output_dir = cfg_path(cfg, "af3.full_output_dir")
    manifest, manifest_path = _load_manifest(cfg)
    predictions_per_design = _prediction_count(cfg)
    chunk_paths = _exact_chunk_paths(output_dir, num_arrays)
    failures: list[dict[str, str]] = []
    frames: list[pd.DataFrame] = []
    backup_hashes: dict[str, str] = {}

    for array_id, canonical in enumerate(chunk_paths):
        paths = _paths(output_dir, array_id)
        try:
            frame = _load_chunk(canonical)
            _validate_chunk_contract(
                frame,
                manifest,
                array_id=array_id,
                num_arrays=num_arrays,
                predictions_per_design=predictions_per_design,
            )
            _validate_enriched_frame(frame)
            backup = _load_chunk(paths["backup"])
            _validate_backup_matches_canonical_base(backup, frame)
            _validate_chunk_contract(
                backup,
                manifest,
                array_id=array_id,
                num_arrays=num_arrays,
                predictions_per_design=predictions_per_design,
            )
            diagnostic = json.loads(paths["diagnostic"].read_text())
            if diagnostic.get("status") not in {"complete", "reused"}:
                raise ValueError(
                    f"Chunk diagnostic status is {diagnostic.get('status')!r}"
                )
            if sum(map(int, diagnostic.get("diagnostics", {}).values())):
                raise ValueError("Chunk diagnostic contains nonzero failure counts")
            backup_sha256 = sha256_file(paths["backup"])
            if diagnostic.get("source_sha256") != backup_sha256:
                raise ValueError("Chunk diagnostic source digest differs from backup")
            backup_hashes[f"{array_id:04d}"] = backup_sha256
            frames.append(frame)
        except Exception as exc:
            failures.append(
                {
                    "array_id": f"{array_id:04d}",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    expected_rows = len(manifest) * predictions_per_design
    if len(combined) != expected_rows:
        failures.append(
            {
                "array_id": "all",
                "error": f"Combined row count {len(combined)} != {expected_rows}",
            }
        )
    if not combined.empty:
        keys = combined[["designed_sample_id", "diffusion_index"]].astype(
            {"designed_sample_id": str, "diffusion_index": int}
        )
        if keys.duplicated().any():
            failures.append(
                {"array_id": "all", "error": "Duplicate global design/diffusion keys"}
            )
        if combined["designed_sample_id"].astype(str).nunique() != len(manifest):
            failures.append(
                {"array_id": "all", "error": "Global designed-sample coverage mismatch"}
            )

    report_path = output_dir / "reports" / "tmalign_validation.json"
    diagnostics = _empty_diagnostics()
    diagnostics["errors"] = len(failures)
    report = {
        "stage": "ligandmpnn-af3-tmalign-validation",
        "status": "failed" if failures else "complete",
        "manifest_csv": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "num_arrays": num_arrays,
        "design_rows": len(manifest),
        "predictions_per_design": predictions_per_design,
        "expected_rows": expected_rows,
        "observed_rows": len(combined),
        "unique_designed_sample_ids": (
            int(combined["designed_sample_id"].astype(str).nunique())
            if not combined.empty
            else 0
        ),
        "tmalign_score_min": (
            float(pd.to_numeric(combined["tmalign_score"]).min())
            if not combined.empty
            else None
        ),
        "tmalign_score_max": (
            float(pd.to_numeric(combined["tmalign_score"]).max())
            if not combined.empty
            else None
        ),
        "backup_sha256_by_chunk": backup_hashes,
        "diagnostics": diagnostics,
        "error_examples": failures[:50],
    }
    atomic_write_json(report_path, report)
    if failures:
        raise RuntimeError(f"TM-align aggregate validation failed; see {report_path}")
    return report


def print_plan(cfg: DictConfig, *, num_arrays: int) -> dict[str, Any]:
    output_dir = cfg_path(cfg, "af3.full_output_dir")
    manifest, manifest_path = _load_manifest(cfg)
    chunk_paths = _exact_chunk_paths(output_dir, num_arrays)
    predictions_per_design = _prediction_count(cfg)
    report = {
        "stage": "ligandmpnn-af3-tmalign-plan",
        "manifest_csv": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "output_dir": str(output_dir),
        "num_arrays": num_arrays,
        "array": f"0-{num_arrays - 1}",
        "design_rows": len(manifest),
        "predictions_per_design": predictions_per_design,
        "expected_metric_rows": len(manifest) * predictions_per_design,
        "source_chunks": len(chunk_paths),
        "source_rows": sum(len(pd.read_csv(path, usecols=["designed_sample_id"])) for path in chunk_paths),
        "canonical_csv_policy": "atomic replace only after full-chunk success",
        "backup_policy": "one immutable pre-TM-align CSV per chunk",
    }
    return report


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.print_plan:
        report = print_plan(cfg, num_arrays=args.num_arrays)
    elif args.validate_all:
        report = validate_all(cfg, num_arrays=args.num_arrays)
    else:
        array_id = _resolve_array_id(args.array_id)
        if array_id is None:
            raise ValueError("--array-id or SLURM_ARRAY_TASK_ID is required")
        report = process_chunk(cfg, array_id=array_id, num_arrays=args.num_arrays)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
