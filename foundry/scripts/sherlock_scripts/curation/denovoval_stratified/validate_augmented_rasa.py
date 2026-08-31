#!/usr/bin/env python3
"""Validate the corrected 2,100-row ligand RASA refresh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from prepare_augmented_rasa_manifest import (
    DEFAULT_RUN_ROOT,
    EXPECTED_BASELINE_ROWS,
    EXPECTED_EXTRA_ROWS,
    EXPECTED_ROWS,
    EXTRA_ORIGIN,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header: {path}")
        return list(reader)


def keyed(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        staged_id = row.get("staged_id", "")
        if not staged_id:
            raise ValueError(f"Blank staged_id in {path}")
        if staged_id in result:
            raise ValueError(f"Duplicate staged_id in {path}: {staged_id}")
        result[staged_id] = row
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    input_path = run_root / "rasa/input_manifest.tsv"
    input_summary_path = run_root / "rasa/input_manifest.json"
    results_root = run_root / "rasa/results"
    result_path = results_root / "ligand_rasa.tsv"
    compute_manifest_path = results_root / "ligand_rasa_manifest.json"
    validation_path = run_root / "rasa/validation.json"
    if validation_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation: {validation_path}")

    input_rows = read_tsv(input_path)
    result_rows = read_tsv(result_path)
    if len(input_rows) != EXPECTED_ROWS or len(result_rows) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} input/results rows, found "
            f"{len(input_rows)}/{len(result_rows)}"
        )
    inputs = keyed(input_rows, input_path)
    results = keyed(result_rows, result_path)
    if set(inputs) != set(results):
        raise ValueError(
            "RASA identity mismatch: "
            f"missing={sorted(set(inputs) - set(results))[:10]} "
            f"extra={sorted(set(results) - set(inputs))[:10]}"
        )

    for staged_id, input_row in inputs.items():
        result = results[staged_id]
        for column in ("condition", "ccd", "length", "source_path", "json_path"):
            if result[column] != input_row[column]:
                raise ValueError(
                    f"RASA metadata mismatch for {staged_id}:{column}: "
                    f"{result[column]!r} != {input_row[column]!r}"
                )
        if result["status"] != "ok":
            raise ValueError(
                f"RASA computation failed for {staged_id}: "
                f"{result['status']} {result.get('detail', '')}"
            )
        if int(result["n_target_residues"]) != 1:
            raise ValueError(
                f"Expected exactly one hetero target residue for {staged_id}, "
                f"found {result['n_target_residues']}"
            )
        if int(result["n_target_heavy_atoms"]) <= 0:
            raise ValueError(f"No target heavy atoms for {staged_id}")
        rasa_value = float(result["rasa_value"])
        if not math.isfinite(rasa_value) or rasa_value < 0.0:
            raise ValueError(f"Invalid RASA value for {staged_id}: {rasa_value}")

    input_summary = json.loads(input_summary_path.read_text())
    compute_manifest = json.loads(compute_manifest_path.read_text())
    if input_summary.get("status") != "complete" or input_summary.get("row_count") != EXPECTED_ROWS:
        raise ValueError("Input manifest summary is incomplete")
    if compute_manifest.get("status") != "complete":
        raise ValueError("RASA compute manifest is incomplete")
    if compute_manifest.get("inputs", {}).get("input_row_count") != EXPECTED_ROWS:
        raise ValueError("RASA compute manifest row count mismatch")
    parameters = compute_manifest.get("parameters", {})
    if parameters.get("probe_radius") != 1.4 or parameters.get("point_number") != 1000:
        raise ValueError(f"Unexpected RASA parameters: {parameters}")

    origin_counts = Counter(row["candidate_origin"] for row in input_rows)
    expected_origin_counts = {
        "baseline": EXPECTED_BASELINE_ROWS,
        EXTRA_ORIGIN: EXPECTED_EXTRA_ROWS,
    }
    if origin_counts != expected_origin_counts:
        raise ValueError(f"Unexpected RASA origin counts: {dict(origin_counts)}")

    payload = {
        "schema_version": 1,
        "status": "complete",
        "input_rows": len(input_rows),
        "result_rows": len(result_rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in result_rows).items())),
        "origin_counts": dict(sorted(origin_counts.items())),
        "target_residue_count": 1,
        "target_semantics": "finite heavy atoms with hetero=True and res_name=CCD",
        "probe_radius": 1.4,
        "point_number": 1000,
        "input_manifest_sha256": sha256_file(input_path),
        "ligand_rasa_sha256": sha256_file(result_path),
    }
    write_json(validation_path, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
