"""Exact global integrity gates for denovoval ABACUS-T and AF3 artifacts."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.denovoval.ligandmpnn.run_af3_ss import (
    configured_design_indices,
    validate_design_manifest,
)
from allatom_design.eval.benchmarks.sequence_design.structure_bridge import sha256_file


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/abacus-t/default.yaml"
)
SAMPLING_REQUIRED = {
    "source_sample_id",
    "designed_sample_id",
    "model_name",
    "design_index",
    "protein_sequence",
    "sampled_pdb_path",
    "sampled_pdb_sha256",
    "fasta_path",
    "fasta_sha256",
    "iteration",
    "source_staging_manifest_sha256",
    "source_preparation_manifest_sha256",
    "run_fingerprint",
    "status",
}
BACKMAPPING_REQUIRED = {
    "source_sample_id",
    "designed_sample_id",
    "model_name",
    "design_index",
    "protein_sequence",
    "protein_chain_id",
    "ligand_chain_id",
    "ccd_code",
    "backmapped_cif_path",
    "backmapped_cif_sha256",
    "status",
}
SC_FINITE_COLUMNS = (
    "sc_ca_rmsd",
    "avg_ca_plddt",
    "tmalign_score",
    "tmalign_rmsd",
    "tmalign_matched_ca_count",
    "tmalign_sample_ca_count",
    "tmalign_pred_ca_count",
    "tmalign_sample_ca_coverage",
    "tmalign_pred_ca_coverage",
    "tmalign_label_span_count",
    "tmalign_excluded_label_positions",
    "tmalign_modified_ca_count",
)
DOCKING_FINITE_COLUMNS = (
    "ligand_rmsd",
    "binding_site_rmsd",
    "num_bs_residues",
    "ligand_plddt",
    "binding_site_plddt",
    "iptm",
    "interface_min_pae",
)


class Diagnostics:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, str]] = []

    def add(self, category: str, key: str, message: str) -> None:
        self.counts[category] += 1
        if len(self.examples) < 50:
            self.examples.append(
                {"category": category, "key": str(key), "error": str(message)}
            )

    @property
    def failed(self) -> bool:
        return bool(sum(self.counts.values()))

    def payload(self) -> dict[str, int]:
        return {
            "total_errors": int(sum(self.counts.values())),
            **{key: int(value) for key, value in sorted(self.counts.items())},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("sampling", "backmapping", "af3-inputs", "af3-complete")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-arrays", type=int, default=None)
    parser.add_argument("--predictions-per-design", type=int, default=None)
    return parser.parse_args()


def cfg_path(cfg: DictConfig, key: str) -> Path:
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ValueError(f"Missing required config path: {key}")
    return Path(str(value)).expanduser()


def _model_seeds(cfg: DictConfig) -> list[int]:
    seeds = list(map(int, cfg.af3.json_config.model_seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"AF3 model seeds must be nonempty and unique: {seeds}")
    return seeds


def _prediction_sample_dir_names(
    cfg: DictConfig, *, predictions_per_design: int
) -> set[str]:
    seeds = _model_seeds(cfg)
    if predictions_per_design < 1 or predictions_per_design % len(seeds):
        raise ValueError(
            "predictions_per_design must be positive and divisible by the "
            f"number of model seeds: {predictions_per_design}/{len(seeds)}"
        )
    samples_per_seed = predictions_per_design // len(seeds)
    return {
        f"seed-{seed}_sample-{sample_index}"
        for seed in seeds
        for sample_index in range(samples_per_seed)
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _finish(path: Path, report: dict[str, Any], diagnostics: Diagnostics) -> None:
    report["status"] = "failed" if diagnostics.failed else "complete"
    report["diagnostics"] = diagnostics.payload()
    report["error_examples"] = diagnostics.examples
    atomic_write_json(path, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if diagnostics.failed:
        raise RuntimeError(f"Validation failed; see {path}")


def _mode(cfg: DictConfig, smoke: bool) -> tuple[str, Path, pd.DataFrame]:
    mode = "smoke" if smoke else "full"
    staging_path = cfg_path(cfg, f"staging.{mode}_manifest_csv")
    staging = pd.read_csv(staging_path, keep_default_na=False)
    if staging["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate staging sample IDs: {staging_path}")
    if not staging["status"].eq("complete").all():
        raise ValueError(f"Incomplete staging rows: {staging_path}")
    return mode, staging_path, staging


def _read_design_manifest(
    path: Path,
    *,
    diagnostics: Diagnostics,
) -> pd.DataFrame:
    if not path.is_file():
        diagnostics.add("sampling_missing_manifest", path.name, str(path))
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except Exception as exc:
        diagnostics.add(
            "sampling_unreadable_manifest",
            path.name,
            f"{type(exc).__name__}: {exc}",
        )
        return pd.DataFrame()
    missing = sorted(SAMPLING_REQUIRED - set(frame.columns))
    if missing:
        diagnostics.add("sampling_missing_columns", path.name, str(missing))
    frame["_manifest_path"] = str(path)
    return frame


def _expected_design_keys(
    staging: pd.DataFrame, design_indices: Iterable[int]
) -> set[tuple[str, int]]:
    return {
        (str(sample_id), int(design_index))
        for sample_id in staging["sample_id"]
        for design_index in design_indices
    }


def _matrix_diagnostics(
    frame: pd.DataFrame,
    expected: set[tuple[Any, ...]],
    columns: tuple[str, ...],
    diagnostics: Diagnostics,
    *,
    prefix: str,
) -> None:
    missing_columns = sorted(set(columns) - set(frame.columns))
    if missing_columns:
        diagnostics.add(f"{prefix}_missing_key_columns", prefix, str(missing_columns))
        return
    observed = Counter(
        tuple(row)
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None)
    )
    for key in sorted(expected - set(observed), key=str)[:100]:
        diagnostics.add(f"{prefix}_missing_key", str(key), "not present")
    for key in sorted(set(observed) - expected, key=str)[:100]:
        diagnostics.add(f"{prefix}_surplus_key", str(key), "unexpected")
    for key, count in observed.items():
        if count != 1:
            diagnostics.add(f"{prefix}_duplicate_key", str(key), str(count))


def metric_matrix_diagnostics(
    frame: pd.DataFrame,
    *,
    expected_design_ids: Iterable[str],
    predictions_per_design: int,
    finite_columns: Iterable[str],
) -> dict[str, int]:
    """Pure exact-matrix check used by the global validator and unit tests."""
    diagnostics = Diagnostics()
    expected = {
        (str(design_id), diffusion_index)
        for design_id in expected_design_ids
        for diffusion_index in range(predictions_per_design)
    }
    _matrix_diagnostics(
        frame,
        expected,
        ("designed_sample_id", "diffusion_index"),
        diagnostics,
        prefix="metric",
    )
    if "metric_status" not in frame or not frame["metric_status"].eq("ok").all():
        diagnostics.add("metric_error_status", "metric_status", "not all rows are ok")
    if "error" in frame and frame["error"].astype(str).str.len().sum():
        diagnostics.add("metric_nonempty_error", "error", "nonempty errors")
    finite_columns = tuple(finite_columns)
    missing = sorted(set(finite_columns) - set(frame.columns))
    if missing:
        diagnostics.add("metric_missing_columns", "metrics", str(missing))
    elif finite_columns:
        numeric = frame[list(finite_columns)].apply(pd.to_numeric, errors="coerce")
        if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
            diagnostics.add("metric_nonfinite", "metrics", ",".join(finite_columns))
    return diagnostics.payload()


def _validate_digest_rows(
    frame: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    diagnostics: Diagnostics,
    *,
    prefix: str,
) -> None:
    for row in frame.to_dict(orient="records"):
        key = str(row.get("designed_sample_id") or row.get("sample_id") or "unknown")
        for path_column, digest_column in pairs:
            try:
                path = Path(str(row[path_column]))
                if not path.is_file():
                    raise FileNotFoundError(path)
                if sha256_file(path) != str(row[digest_column]):
                    raise ValueError("SHA256 mismatch")
            except Exception as exc:
                diagnostics.add(
                    f"{prefix}_missing_or_stale_artifact",
                    key,
                    f"{path_column}: {type(exc).__name__}: {exc}",
                )


def validate_sampling(cfg: DictConfig, *, smoke: bool) -> None:
    diagnostics = Diagnostics()
    mode, staging_path, staging = _mode(cfg, smoke)
    mode_cfg = cfg.sampling[mode]
    manifest_path = Path(str(mode_cfg.design_manifest_csv)).expanduser()
    sampling = _read_design_manifest(
        manifest_path,
        diagnostics=diagnostics,
    )
    indices = configured_design_indices(cfg)
    expected = _expected_design_keys(staging, indices)
    _matrix_diagnostics(
        sampling,
        expected,
        ("source_sample_id", "design_index"),
        diagnostics,
        prefix="sampling",
    )
    if not sampling.empty:
        if "status" not in sampling or not sampling["status"].isin(
            ["complete", "reused"]
        ).all():
            diagnostics.add("sampling_status_error", "status", "incomplete rows")
        if "model_name" not in sampling or not sampling["model_name"].eq(
            "abacus-t"
        ).all():
            diagnostics.add("sampling_model_error", "model_name", "not abacus-t")
        if "iteration" not in sampling or not pd.to_numeric(
            sampling["iteration"], errors="coerce"
        ).eq(19).all():
            diagnostics.add("sampling_iteration_error", "iteration", "not 19")
        staging_digest = sha256_file(staging_path)
        if "source_staging_manifest_sha256" not in sampling or not sampling[
            "source_staging_manifest_sha256"
        ].eq(staging_digest).all():
            diagnostics.add("sampling_stale_staging", "staging", staging_digest)
        preparation_manifest = cfg_path(cfg, f"preparation.{mode}.manifest_csv")
        preparation_digest = sha256_file(preparation_manifest)
        if "source_preparation_manifest_sha256" not in sampling or not sampling[
            "source_preparation_manifest_sha256"
        ].eq(preparation_digest).all():
            diagnostics.add(
                "sampling_stale_preparation", "preparation", preparation_digest
            )
        _validate_digest_rows(
            sampling,
            (("sampled_pdb_path", "sampled_pdb_sha256"), ("fasta_path", "fasta_sha256")),
            diagnostics,
            prefix="sampling",
        )
    report_path = Path(str(mode_cfg.validation_json))
    _finish(
        report_path,
        {
            "stage": "sampling",
            "mode": mode,
            "artifact_manifest": str(manifest_path),
            "artifact_manifest_present": manifest_path.is_file(),
            "artifact_expected_designs": len(expected),
            "artifact_observed_designs": len(sampling),
        },
        diagnostics,
    )


def _load_backmapping(
    cfg: DictConfig, *, smoke: bool, diagnostics: Diagnostics
) -> tuple[str, pd.DataFrame, pd.DataFrame, Path]:
    mode, _, staging = _mode(cfg, smoke)
    path = cfg_path(cfg, f"backmapping.{mode}.manifest_csv")
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except Exception as exc:
        diagnostics.add(
            "backmapping_unreadable_manifest", path.name, f"{type(exc).__name__}: {exc}"
        )
        return mode, staging, pd.DataFrame(), path
    missing = sorted(BACKMAPPING_REQUIRED - set(frame.columns))
    if missing:
        diagnostics.add("backmapping_missing_columns", path.name, str(missing))
        return mode, staging, frame, path
    expected = _expected_design_keys(staging, configured_design_indices(cfg))
    _matrix_diagnostics(
        frame,
        expected,
        ("source_sample_id", "design_index"),
        diagnostics,
        prefix="backmapping",
    )
    if not frame["status"].isin(["backmapped", "reused"]).all():
        diagnostics.add("backmapping_status_error", "status", "failed rows")
    try:
        validate_design_manifest(frame, cfg, smoke=smoke)
    except Exception as exc:
        diagnostics.add(
            "backmapping_af3_schema_error", path.name, f"{type(exc).__name__}: {exc}"
        )
    _validate_digest_rows(
        frame,
        (("backmapped_cif_path", "backmapped_cif_sha256"),),
        diagnostics,
        prefix="backmapping",
    )
    return mode, staging, frame, path


def validate_backmapping(cfg: DictConfig, *, smoke: bool) -> None:
    diagnostics = Diagnostics()
    mode, staging, frame, path = _load_backmapping(
        cfg, smoke=smoke, diagnostics=diagnostics
    )
    report_path = cfg_path(cfg, f"backmapping.{mode}.root") / "validation_global.json"
    _finish(
        report_path,
        {
            "stage": "backmapping",
            "mode": mode,
            "artifact_expected_designs": len(staging)
            * len(configured_design_indices(cfg)),
            "artifact_observed_designs": len(frame),
            "manifest": str(path),
        },
        diagnostics,
    )


def _expected_chunk_paths(directory: Path, stem: str, count: int) -> set[Path]:
    return {directory / f"{stem}_{index:04d}.csv" for index in range(count)}


def _read_exact_chunks(
    directory: Path,
    stem: str,
    count: int,
    diagnostics: Diagnostics,
    *,
    category: str,
) -> pd.DataFrame:
    expected = _expected_chunk_paths(directory, stem, count)
    observed = set(directory.glob(f"{stem}_*.csv")) if directory.is_dir() else set()
    for path in sorted(expected - observed):
        diagnostics.add(f"{category}_missing_chunk", path.name, str(path))
    for path in sorted(observed - expected):
        diagnostics.add(f"{category}_surplus_chunk", path.name, str(path))
    frames: list[pd.DataFrame] = []
    for path in sorted(expected & observed):
        try:
            frames.append(pd.read_csv(path, keep_default_na=False))
        except Exception as exc:
            diagnostics.add(
                f"{category}_unreadable_chunk",
                path.name,
                f"{type(exc).__name__}: {exc}",
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _validate_status_jsons(
    directory: Path,
    *,
    count: int,
    manifest_sha256: str,
    diagnostics: Diagnostics,
) -> None:
    expected = {
        directory / f"status_{index:04d}.json" for index in range(count)
    }
    observed = set(directory.glob("status_*.json")) if directory.is_dir() else set()
    for path in sorted(expected - observed):
        diagnostics.add("scheduler_missing_status_json", path.name, str(path))
    for path in sorted(observed - expected):
        diagnostics.add("scheduler_surplus_status_json", path.name, str(path))
    for path in sorted(expected & observed):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            array_index = int(path.stem.removeprefix("status_"))
            if payload.get("manifest_sha256") != manifest_sha256:
                raise ValueError("status JSON references a stale design manifest")
            plan = payload.get("plan", {})
            if int(plan.get("num_arrays", -1)) != count:
                raise ValueError("status JSON num_arrays differs")
            if int(plan.get("array_id", -1)) != array_index:
                raise ValueError("status JSON array_id differs from filename")
            nonzero = {
                key: int(value)
                for key, value in payload.get("diagnostics", {}).items()
                if int(value)
            }
            if nonzero:
                raise ValueError(f"status JSON has nonzero diagnostics: {nonzero}")
        except Exception as exc:
            diagnostics.add(
                "scheduler_invalid_status_json",
                path.name,
                f"{type(exc).__name__}: {exc}",
            )


def _validate_af3_inputs(
    cfg: DictConfig,
    *,
    smoke: bool,
    num_arrays: int,
    allowed_statuses: set[str],
    diagnostics: Diagnostics,
) -> tuple[str, pd.DataFrame, Path, pd.DataFrame]:
    mode, _, design, manifest_path = _load_backmapping(
        cfg, smoke=smoke, diagnostics=diagnostics
    )
    output_root = cfg_path(cfg, f"af3.{mode}_output_dir")
    status = _read_exact_chunks(
        output_root / "reports" / "chunks",
        "status",
        num_arrays,
        diagnostics,
        category="scheduler_status",
    )
    if manifest_path.is_file():
        _validate_status_jsons(
            output_root / "reports" / "chunks",
            count=num_arrays,
            manifest_sha256=sha256_file(manifest_path),
            diagnostics=diagnostics,
        )
    expected_ids = set(map(str, design.get("designed_sample_id", [])))
    _matrix_diagnostics(
        status,
        {(sample_id,) for sample_id in expected_ids},
        ("designed_sample_id",),
        diagnostics,
        prefix="scheduler_status",
    )
    if "status" not in status or not status["status"].isin(allowed_statuses).all():
        diagnostics.add(
            "scheduler_invalid_status", "status", f"expected {sorted(allowed_statuses)}"
        )

    input_dir = output_root / "inputs"
    expected_jsons = {input_dir / f"{sample_id}.json" for sample_id in expected_ids}
    observed_jsons = set(input_dir.glob("*.json")) if input_dir.is_dir() else set()
    for path in sorted(expected_jsons - observed_jsons):
        diagnostics.add("af3_input_missing_json", path.stem, str(path))
    for path in sorted(observed_jsons - expected_jsons):
        diagnostics.add("af3_input_surplus_json", path.stem, str(path))
    design_by_id = design.set_index("designed_sample_id", drop=False)
    for path in sorted(expected_jsons & observed_jsons):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            row = design_by_id.loc[path.stem]
            sequences = payload["sequences"]
            if payload["name"] != path.stem:
                raise ValueError("JSON name differs from designed_sample_id")
            if sequences[0] != {
                "protein": {
                    "id": "A",
                    "sequence": str(row["protein_sequence"]),
                    "modifications": [],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates": [],
                }
            }:
                raise ValueError("JSON protein A_1 sequence payload differs")
            if sequences[1] != {
                "ligand": {"id": "L", "ccdCodes": [str(row["ccd_code"])]}
            }:
                raise ValueError("JSON ligand L_1/original CCD payload differs")
            if (
                list(payload["modelSeeds"]) != _model_seeds(cfg)
                or int(payload["version"]) != 2
            ):
                raise ValueError("JSON seed/version differs from pinned config")
        except Exception as exc:
            diagnostics.add(
                "af3_input_invalid_json", path.stem, f"{type(exc).__name__}: {exc}"
            )
    return mode, design, output_root, status


def validate_af3_inputs(cfg: DictConfig, *, smoke: bool, num_arrays: int) -> None:
    diagnostics = Diagnostics()
    mode, design, output_root, status = _validate_af3_inputs(
        cfg,
        smoke=smoke,
        num_arrays=num_arrays,
        allowed_statuses={"input_ready"},
        diagnostics=diagnostics,
    )
    _finish(
        output_root / "reports" / "validation_inputs.json",
        {
            "stage": "af3-inputs",
            "mode": mode,
            "scheduler_expected_chunks": num_arrays,
            "scheduler_status_rows": len(status),
            "artifact_expected_jsons": len(design),
        },
        diagnostics,
    )


def validate_af3_complete(
    cfg: DictConfig,
    *,
    smoke: bool,
    num_arrays: int,
    predictions_per_design: int,
) -> None:
    diagnostics = Diagnostics()
    mode, design, output_root, status = _validate_af3_inputs(
        cfg,
        smoke=smoke,
        num_arrays=num_arrays,
        allowed_statuses={"complete"},
        diagnostics=diagnostics,
    )
    expected_design_ids = list(map(str, design.get("designed_sample_id", [])))
    expected_rows = len(expected_design_ids) * predictions_per_design
    for column, expected in (
        ("n_predictions_expected", predictions_per_design),
        ("n_predictions_found", predictions_per_design),
        ("n_malformed", 0),
        ("n_surplus", 0),
        ("sc_metric_rows", predictions_per_design),
        ("docking_metric_rows", predictions_per_design),
        ("metric_error_rows", 0),
    ):
        if column not in status or not pd.to_numeric(
            status[column], errors="coerce"
        ).eq(expected).all():
            diagnostics.add("prediction_status_count_error", column, str(expected))
    if "input_fingerprint_ok" not in status or not status[
        "input_fingerprint_ok"
    ].astype(str).str.lower().eq("true").all():
        diagnostics.add("prediction_stale_fingerprint", "input_fingerprint_ok", "false")

    prediction_root = output_root / "predictions"
    expected_dirs = {prediction_root / design_id for design_id in expected_design_ids}
    observed_dirs = (
        {path for path in prediction_root.iterdir() if path.is_dir()}
        if prediction_root.is_dir()
        else set()
    )
    for path in sorted(expected_dirs - observed_dirs):
        diagnostics.add("prediction_missing_design_dir", path.name, str(path))
    for path in sorted(observed_dirs - expected_dirs):
        diagnostics.add("prediction_surplus_design_dir", path.name, str(path))
    for directory in sorted(expected_dirs & observed_dirs):
        expected_sample_dirs = {
            directory / name
            for name in _prediction_sample_dir_names(
                cfg, predictions_per_design=predictions_per_design
            )
        }
        observed_sample_dirs = {
            path for path in directory.iterdir() if path.is_dir()
        }
        for path in sorted(expected_sample_dirs - observed_sample_dirs):
            diagnostics.add("prediction_missing_sample_dir", directory.name, str(path))
        for path in sorted(observed_sample_dirs - expected_sample_dirs):
            diagnostics.add("prediction_surplus_sample_dir", directory.name, str(path))
        for sample_dir in sorted(expected_sample_dirs & observed_sample_dirs):
            model_cifs = list(sample_dir.glob("*_model.cif"))
            if len(model_cifs) != 1:
                diagnostics.add(
                    "prediction_model_cif_count",
                    sample_dir.name,
                    f"{len(model_cifs)} != 1",
                )

    metrics_dir = output_root / "metrics" / "chunks"
    sc = _read_exact_chunks(
        metrics_dir,
        "self_consistency",
        num_arrays,
        diagnostics,
        category="sc_metric",
    )
    docking = _read_exact_chunks(
        metrics_dir,
        "docking",
        num_arrays,
        diagnostics,
        category="docking_metric",
    )
    for name, frame, finite_columns in (
        ("sc", sc, SC_FINITE_COLUMNS),
        ("docking", docking, DOCKING_FINITE_COLUMNS),
    ):
        observed = metric_matrix_diagnostics(
            frame,
            expected_design_ids=expected_design_ids,
            predictions_per_design=predictions_per_design,
            finite_columns=finite_columns,
        )
        for category, count in observed.items():
            if category == "total_errors":
                continue
            for _ in range(count):
                diagnostics.add(f"{name}_{category}", name, str(count))
    if "tmalign_score" in sc:
        scores = pd.to_numeric(sc["tmalign_score"], errors="coerce").to_numpy()
        if np.any((scores < 0.0) | (scores > 1.00001)):
            diagnostics.add("sc_tmalign_score_range", "tmalign_score", "outside [0,1]")
    _finish(
        output_root / "reports" / "validation_complete.json",
        {
            "stage": "af3-complete",
            "mode": mode,
            "scheduler_expected_chunks": num_arrays,
            "scheduler_status_rows": len(status),
            "artifact_expected_designs": len(design),
            "prediction_expected_rows": expected_rows,
            "self_consistency_rows": len(sc),
            "docking_rows": len(docking),
        },
        diagnostics,
    )


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.stage == "sampling":
        validate_sampling(cfg, smoke=args.smoke)
    elif args.stage == "backmapping":
        validate_backmapping(cfg, smoke=args.smoke)
    else:
        mode = "smoke" if args.smoke else "full"
        num_arrays = args.num_arrays or int(cfg.af3.arrays[mode])
        if args.stage == "af3-inputs":
            validate_af3_inputs(cfg, smoke=args.smoke, num_arrays=num_arrays)
        else:
            predictions = args.predictions_per_design
            if predictions is None:
                predictions = len(_model_seeds(cfg)) * int(
                    cfg.af3.inference_config.ss.num_diffusion_samples
                )
            validate_af3_complete(
                cfg,
                smoke=args.smoke,
                num_arrays=num_arrays,
                predictions_per_design=predictions,
            )


if __name__ == "__main__":
    main()
