#!/usr/bin/env python3
"""Stage the five approved denovoval extra-generation tasks.

The four infeasible ``(CCD, length)`` groups need candidates only from the
sampler conditions that are short on disjoint Foldseek clusters.  Inputs are
copied into the versioned curation root so the new generation does not depend
on, or write into, either original raw corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


EVAL_ROOT = Path("/scratch/users/zhkim216/datasets/evaluation_datasets")
DEFAULT_VERSION_ROOT = EVAL_ROOT / "denovoval_cross_condition_foldseek_v1"
DEFAULT_ARCHIVED_P0_SOURCE_ROOT = (
    DEFAULT_VERSION_ROOT
    / "extra_generation/source_archive/denovoval_stratified/ss1p0_gamma1p0/source_data"
)
DEFAULT_P6_SOURCE_ROOT = EVAL_ROOT / "curation/ver2/outputs/denovoval/step2/source_data"

# task_id order is part of the Slurm array contract.
TARGETS = (
    ("ss1p6_gamma0p6", "G2P", 300, 1.5, 0.6),
    ("ss1p6_gamma0p6", "GLC", 300, 1.5, 0.6),
    ("ss1p0_gamma1p0", "GLU", 150, 1.0, 1.0),
    ("ss1p0_gamma1p0", "GLU", 300, 1.0, 1.0),
    ("ss1p6_gamma0p6", "GLU", 300, 1.5, 0.6),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-root", type=Path, default=DEFAULT_VERSION_ROOT)
    parser.add_argument(
        "--archived-p0-source-root",
        type=Path,
        default=DEFAULT_ARCHIVED_P0_SOURCE_ROOT,
    )
    parser.add_argument("--p6-source-root", type=Path, default=DEFAULT_P6_SOURCE_ROOT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def stage_target(
    *,
    condition: str,
    ccd: str,
    length: int,
    source_root: Path,
    inputs_root: Path,
) -> tuple[Path, Path, str, str]:
    group = f"{ccd}_len{length}"
    source_cif = source_root / "inputs" / f"{ccd}.cif"
    source_json = source_root / "rfd3_inputs" / f"{group}.json"
    if not source_cif.is_file():
        raise FileNotFoundError(source_cif)
    if not source_json.is_file():
        raise FileNotFoundError(source_json)

    payload = json.loads(source_json.read_text())
    if set(payload) != {group}:
        raise ValueError(f"Unexpected RFD3 spec keys in {source_json}: {sorted(payload)}")
    spec = payload[group]
    expected_length = f"{length}-{length}"
    if spec.get("length") != expected_length:
        raise ValueError(
            f"Unexpected length in {source_json}: {spec.get('length')!r} != {expected_length!r}"
        )
    if spec.get("ligand") != ccd:
        raise ValueError(f"Unexpected ligand in {source_json}: {spec.get('ligand')!r}")
    if spec.get("select_buried") != {ccd: "ALL"}:
        raise ValueError(f"Unexpected select_buried in {source_json}")
    if spec.get("select_fixed_atoms") != {ccd: ""}:
        raise ValueError(f"Unexpected select_fixed_atoms in {source_json}")

    condition_root = inputs_root / condition
    staged_cif = condition_root / "inputs" / f"{ccd}.cif"
    staged_json = condition_root / "rfd3_inputs" / f"{group}.json"
    staged_cif.parent.mkdir(parents=True, exist_ok=True)
    staged_json.parent.mkdir(parents=True, exist_ok=True)
    if staged_json.exists():
        raise FileExistsError(f"Refusing to overwrite staged target: {staged_json}")
    if staged_cif.exists():
        if sha256_file(staged_cif) != sha256_file(source_cif):
            raise FileExistsError(f"Staged CIF differs from source: {staged_cif}")
    else:
        shutil.copy2(source_cif, staged_cif)
    spec["input"] = str(staged_cif.resolve())
    write_json(staged_json, payload)
    return (
        staged_cif,
        staged_json,
        sha256_file(source_cif),
        sha256_file(source_json),
    )


def main() -> None:
    args = parse_args()
    version_root = args.version_root.resolve()
    extra_root = version_root / "extra_generation"
    inputs_root = extra_root / "inputs"
    task_manifest = extra_root / "task_manifest.tsv"
    provenance_path = extra_root / "input_manifest.json"
    if task_manifest.exists() or provenance_path.exists() or inputs_root.exists():
        raise FileExistsError(
            "Targeted generation inputs already exist; refusing to overwrite: "
            f"{inputs_root}, {task_manifest}, {provenance_path}"
        )
    extra_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for task_id, (condition, ccd, length, step_scale, gamma_0) in enumerate(TARGETS):
        source_root = (
            args.archived_p0_source_root.resolve()
            if condition == "ss1p0_gamma1p0"
            else args.p6_source_root.resolve()
        )
        staged_cif, staged_json, source_cif_sha256, source_json_sha256 = stage_target(
            condition=condition,
            ccd=ccd,
            length=length,
            source_root=source_root,
            inputs_root=inputs_root,
        )
        group = f"{ccd}_len{length}"
        rows.append(
            {
                "task_id": task_id,
                "condition": condition,
                "group": group,
                "ccd": ccd,
                "length": length,
                "input_json": str(staged_json.resolve()),
                "step_scale": step_scale,
                "gamma_0": gamma_0,
            }
        )
        provenance_rows.append(
            {
                **rows[-1],
                "source_root": str(source_root),
                "staged_cif": str(staged_cif.resolve()),
                "staged_cif_sha256": sha256_file(staged_cif),
                "staged_json_sha256": sha256_file(staged_json),
                "source_cif_sha256": source_cif_sha256,
                "source_json_sha256": source_json_sha256,
            }
        )

    fieldnames = tuple(rows[0])
    temporary = task_manifest.with_name(f".{task_manifest.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, task_manifest)
    write_json(
        provenance_path,
        {
            "schema_version": 1,
            "task_count": len(rows),
            "targets": provenance_rows,
            "sampler_contract": {
                "ss1p0_gamma1p0": {"step_scale": 1.0, "gamma_0": 1.0},
                "ss1p6_gamma0p6": {
                    "step_scale": 1.5,
                    "gamma_0": 0.6,
                    "note": "Preserves the original raw producer's effective RFD3 defaults.",
                },
                "noise_scale": 1.003,
                "cfg_scale": 2.0,
                "cfg_features": ["ref_atomwise_rasa"],
            },
        },
    )
    print(f"Wrote {task_manifest}")
    print(f"Wrote {provenance_path}")


if __name__ == "__main__":
    main()
