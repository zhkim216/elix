#!/usr/bin/env python3
"""Prepare one production-shaped ss1p0 replacement for a direct-32 canary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PARTIAL_DIFFUSION_DIR = SCRIPT_DIR.parents[1] / "partial_diffusion"
sys.path.insert(0, str(PARTIAL_DIFFUSION_DIR))

from prepare_denovoval_inputs import (  # noqa: E402
    EXPECTED_FOUNDRY_VERSION,
    prepare_source,
    write_csv,
)
from prepare_nativeval_inputs import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


EVAL_ROOT = Path("/scratch/users/zhkim216/datasets/evaluation_datasets")
DEFAULT_VERSION_ROOT = EVAL_ROOT / "denovoval_cross_condition_foldseek_v1"
DEFAULT_REPLACEMENTS = DEFAULT_VERSION_ROOT / "selection/main_ensemble_replacements.tsv"
DEFAULT_MAIN_ROOT = DEFAULT_VERSION_ROOT / "denovoval_re"
DEFAULT_OUTPUT_ROOT = DEFAULT_VERSION_ROOT / "canary/ss1p0_direct32"
CONDITION = "ss1p0_gamma1p0"
PARTIAL_T = 5
NUM_SAMPLES = 32
DIFFUSION_BATCH_SIZE = 32
N_BATCHES = 1
SEED = 7
STEP_SCALE = 1.0
GAMMA_0 = 1.0
NOISE_SCALE = 1.003


def read_replacements(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "condition",
        "pdb_key",
        "length",
        "partial_diffusion_source_index",
        "materialized_cif_sha256",
    }
    if not rows:
        raise ValueError(f"No main replacements are recorded in {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing replacement columns in {path}: {sorted(missing)}")
    return rows


def choose_replacement(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [row for row in rows if row["condition"] == CONDITION]
    if not candidates:
        raise ValueError("No ss1p0_gamma1p0 main replacement requires a direct-32 canary")
    return min(
        candidates,
        key=lambda row: (int(row["length"]) != 300, row["pdb_key"]),
    )


def prepare(args: argparse.Namespace) -> None:
    if version("rc-foundry") != EXPECTED_FOUNDRY_VERSION:
        raise RuntimeError(
            f"Expected rc-foundry {EXPECTED_FOUNDRY_VERSION}, "
            f"got {version('rc-foundry')}"
        )
    replacements = args.replacements.resolve()
    replacement_digest = sha256_file(replacements)
    selected = choose_replacement(read_replacements(replacements))
    output_root = args.output_root.resolve()
    existing_plan = output_root / "canary_plan.json"
    if existing_plan.is_file():
        prior = json.loads(existing_plan.read_text())
        if (
            prior.get("replacement_manifest_sha256") != replacement_digest
            or prior.get("pdb_key") != selected["pdb_key"]
        ):
            raise ValueError("Existing canary plan does not match the current replacement set")

    source_index = int(selected["partial_diffusion_source_index"])
    source_path = (
        args.main_root.resolve()
        / CONDITION
        / "cifs"
        / f"{selected['pdb_key']}.cif"
    )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_digest = sha256_file(source_path)
    if source_digest != selected["materialized_cif_sha256"]:
        raise ValueError(
            f"Materialized replacement hash mismatch for {selected['pdb_key']}"
        )

    records, task_inputs = prepare_source(
        source_index=source_index,
        source_path=source_path,
        output_root=output_root,
        partial_ts=[PARTIAL_T],
        overwrite=True,
    )
    if len(records) != 1 or len(task_inputs) != 1:
        raise ValueError("Canary preparation did not produce exactly one input")
    prepared_csv = output_root / "prepared_inputs_manifest.csv"
    write_csv(prepared_csv, records, overwrite=True)
    atomic_write_text(
        output_root / "canary_input.txt",
        task_inputs[0] + "\n",
        overwrite=True,
    )
    atomic_write_json(
        output_root / "preparation_failures.json",
        {"schema_version": 1, "status": "complete", "failures": []},
        overwrite=True,
    )
    now = datetime.now(timezone.utc).isoformat()
    atomic_write_json(
        output_root / "preparation_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "allow_partial_preparation": False,
            "created_at_utc": now,
            "rc_foundry_version": EXPECTED_FOUNDRY_VERSION,
            "source_dir": str(source_path.parent),
            "output_root": str(output_root),
            "prepared_inputs_manifest": str(prepared_csv),
            "preparation_failures": str(output_root / "preparation_failures.json"),
            "input_index": str(output_root / "canary_input.txt"),
            "partial_t_values": [PARTIAL_T],
            "sequence_modes": ["fixed"],
            "ligand_fixed": True,
            "n_source_cifs": 1,
            "n_source_ccds": 1,
            "n_prepared_sources": 1,
            "n_prepared_inputs": 1,
            "n_expected_inputs_without_failures": 1,
            "n_array_tasks": 1,
            "diagnostics": {
                "preparation_failures": 0,
                "missing": 0,
                "duplicate_sample_ids": 0,
            },
        },
        overwrite=True,
    )
    plan = {
        "schema_version": 1,
        "status": "prepared",
        "prepared_at_utc": now,
        "selection_policy": "prefer_length_300_then_pdb_key",
        "condition": CONDITION,
        "pdb_key": selected["pdb_key"],
        "length": int(selected["length"]),
        "partial_diffusion_source_index": source_index,
        "source_cif": str(source_path),
        "source_cif_sha256": source_digest,
        "replacement_manifest": str(replacements),
        "replacement_manifest_sha256": replacement_digest,
        "input_json": task_inputs[0],
        "partial_t": PARTIAL_T,
        "num_samples": NUM_SAMPLES,
        "diffusion_batch_size": DIFFUSION_BATCH_SIZE,
        "n_batches": N_BATCHES,
        "seed": SEED,
        "step_scale": STEP_SCALE,
        "gamma_0": GAMMA_0,
        "noise_scale": NOISE_SCALE,
    }
    atomic_write_json(existing_plan, plan, overwrite=True)
    print(json.dumps(plan, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacements", type=Path, default=DEFAULT_REPLACEMENTS)
    parser.add_argument("--main-root", type=Path, default=DEFAULT_MAIN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
