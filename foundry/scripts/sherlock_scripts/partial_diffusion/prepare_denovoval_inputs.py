#!/usr/bin/env python3
"""Prepare denovoval CIFs for fixed-sequence, fixed-ligand partial diffusion."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
from rfd3.transforms.util_transforms import assign_types_

from prepare_nativeval_inputs import (
    EXPECTED_FOUNDRY_VERSION,
    annotation_as_str,
    atomic_copy,
    atomic_write_json,
    atomic_write_text,
    context_identity,
    load_structure,
    residue_ids_for_chain,
    selection_from_residue_ids,
    sha256_file,
    validate_spec,
)


DEFAULT_SOURCE_DIR = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/cifs"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/ensembles"
)
DEFAULT_SMOKE_SAMPLE_ID = "A1ISN_len300_9"
EXPECTED_CIF_COUNT = 3400
EXPECTED_CCD_COUNT = 154
BINDER_CHAIN = "A"
CONTEXT_CHAIN = "L"
SOURCE_NAME = re.compile(
    r"^(?P<ccd>[A-Za-z0-9]+)_len(?P<length>150|300)_(?P<sample_index>[0-9]+)$"
)


def condition_name(partial_t: int) -> str:
    return f"partialt{partial_t}"


def parse_source_name(path: Path) -> dict[str, Any]:
    match = SOURCE_NAME.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unexpected denovoval CIF name: {path.name}")
    return {
        "sample_id": path.stem,
        "ccd_code": match.group("ccd"),
        "length": int(match.group("length")),
        "sample_index": int(match.group("sample_index")),
    }


def source_sort_key(path: Path) -> tuple[str, int, int]:
    parsed = parse_source_name(path)
    return parsed["ccd_code"], parsed["length"], parsed["sample_index"]


def write_csv(path: Path, records: list[dict[str, Any]], *, overwrite: bool) -> None:
    if not records:
        raise ValueError("No prepared records to write")
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    atomic_write_text(path, buffer.getvalue(), overwrite=overwrite)


def prepare_source(
    *,
    source_index: int,
    source_path: Path,
    output_root: Path,
    partial_ts: list[int],
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    identity = parse_source_name(source_path)
    sample_id = identity["sample_id"]
    source_sha256 = sha256_file(source_path)
    original_path = output_root / "original_samples" / f"{sample_id}.cif"
    if not original_path.exists() or sha256_file(original_path) != source_sha256:
        atomic_copy(source_path, original_path, overwrite=overwrite)
    if sha256_file(original_path) != source_sha256:
        raise ValueError(f"Staged original digest mismatch: {sample_id}")

    structure = load_structure(original_path)
    chain_ids = set(structure.chain_id.astype(str))
    if chain_ids != {BINDER_CHAIN, CONTEXT_CHAIN}:
        raise ValueError(f"Expected chains A/L for {sample_id}, got {sorted(chain_ids)}")
    binder_residue_ids = residue_ids_for_chain(structure, BINDER_CHAIN)
    binder_selector = selection_from_residue_ids(BINDER_CHAIN, binder_residue_ids)
    binder_ca_count = int(
        np.sum(
            (structure.chain_id.astype(str) == BINDER_CHAIN)
            & (structure.atom_name.astype(str) == "CA")
        )
    )
    if binder_ca_count != identity["length"]:
        raise ValueError(
            f"Protein CA count mismatch for {sample_id}: "
            f"{binder_ca_count} != {identity['length']}"
        )
    context_res_id, context_ccd, context_atom_count = context_identity(
        structure, CONTEXT_CHAIN
    )
    if context_ccd != identity["ccd_code"]:
        raise ValueError(
            f"Ligand CCD mismatch for {sample_id}: {context_ccd} != {identity['ccd_code']}"
        )
    context_selector = f"{CONTEXT_CHAIN}{context_res_id}"
    typed_structure = assign_types_(structure.copy())
    context_mask = typed_structure.chain_id.astype(str) == CONTEXT_CHAIN
    if bool(
        np.any(
            typed_structure.is_protein[context_mask]
            | typed_structure.is_dna[context_mask]
            | typed_structure.is_rna[context_mask]
        )
    ):
        raise ValueError(f"RFD3 classified ligand {sample_id}:{context_selector} as polymer")

    specs: list[tuple[int, dict[str, Any], Path]] = []
    for partial_t in partial_ts:
        condition = condition_name(partial_t)
        input_json = output_root / condition / "inputs" / f"{sample_id}.json"
        spec = {
            "input": str(original_path),
            "ligand": context_selector,
            "partial_t": float(partial_t),
            "select_fixed_atoms": {context_selector: "ALL"},
            "select_unfixed_sequence": False,
            "extra": {
                "schema_version": 1,
                "sample_id": sample_id,
                "source_index": source_index,
                "ccd_code": identity["ccd_code"],
                "length": identity["length"],
                "sample_index": identity["sample_index"],
                "condition": condition,
                "sequence_unfixed": False,
                "ligand_fixed": True,
                "binder_chain_id": BINDER_CHAIN,
                "context_chain_id": CONTEXT_CHAIN,
                "context_res_id": context_res_id,
                "context_ccd": context_ccd,
                "rfd3_context_ccd": context_ccd,
                "ccd_alias": None,
                "original_cif": str(original_path),
                "original_cif_sha256": source_sha256,
                "rfd3_input_cif_sha256": source_sha256,
            },
        }
        specs.append((partial_t, spec, input_json))

    validate_spec(
        specs[0][1],
        atom_array_input=structure,
        binder_chain=BINDER_CHAIN,
        context_chain=CONTEXT_CHAIN,
        sequence_unfixed=False,
        expected_context_atoms=context_atom_count,
    )

    records: list[dict[str, Any]] = []
    task_inputs: list[str] = []
    for partial_t, spec, input_json in specs:
        atomic_write_json(input_json, {"partial_diffusion": spec}, overwrite=overwrite)
        input_path = str(input_json.resolve())
        task_inputs.append(input_path)
        records.append(
            {
                "source_index": source_index,
                "sample_id": sample_id,
                "ccd_code": identity["ccd_code"],
                "length": identity["length"],
                "sample_index": identity["sample_index"],
                "condition": condition_name(partial_t),
                "partial_t": partial_t,
                "sequence_mode": "fixed",
                "sequence_unfixed": False,
                "ligand_fixed": True,
                "binder_chain_id": BINDER_CHAIN,
                "context_chain_id": CONTEXT_CHAIN,
                "context_res_id": context_res_id,
                "context_ccd": context_ccd,
                "rfd3_context_ccd": context_ccd,
                "ccd_alias": "",
                "binder_selector": binder_selector,
                "context_selector": context_selector,
                "protein_residue_count": len(binder_residue_ids),
                "protein_ca_count": binder_ca_count,
                "context_atom_count": context_atom_count,
                "source_cif": str(source_path),
                "source_cif_sha256": source_sha256,
                "original_cif": str(original_path),
                "original_cif_sha256": source_sha256,
                "rfd3_input_cif": str(original_path),
                "rfd3_input_cif_sha256": source_sha256,
                "input_json": input_path,
                "input_json_sha256": sha256_file(input_json),
                "consumer_validation_ok": True,
            }
        )
    return records, task_inputs


def prepare(args: argparse.Namespace) -> None:
    package_version = version("rc-foundry")
    if package_version != EXPECTED_FOUNDRY_VERSION:
        raise RuntimeError(
            f"Expected rc-foundry {EXPECTED_FOUNDRY_VERSION}, got {package_version}"
        )
    partial_ts = sorted(set(args.partial_t))
    if not partial_ts or any(value <= 0 for value in partial_ts):
        raise ValueError("--partial-t values must be positive integers")

    source_dir = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    source_paths = sorted(source_dir.glob("*.cif"), key=source_sort_key)
    if len(source_paths) != args.expected_cif_count:
        raise ValueError(
            f"Expected {args.expected_cif_count} source CIFs, found {len(source_paths)}"
        )
    identities = [parse_source_name(path) for path in source_paths]
    sample_ids = [item["sample_id"] for item in identities]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate denovoval sample IDs")
    source_ccds = sorted({item["ccd_code"] for item in identities})
    if len(source_ccds) != args.expected_ccd_count:
        raise ValueError(
            f"Expected {args.expected_ccd_count} CCD codes, found {len(source_ccds)}"
        )
    if args.smoke_sample_id not in set(sample_ids):
        raise ValueError(f"Smoke sample is absent: {args.smoke_sample_id}")

    manifest_csv = output_root / "prepared_inputs_manifest.csv"
    manifest_json = output_root / "preparation_manifest.json"
    failures_json = output_root / "preparation_failures.json"
    input_index = output_root / "input_index.txt"
    array_index = output_root / "array_index.txt"
    smoke_array_index = output_root / "smoke_array_index.txt"
    managed = [
        manifest_csv,
        manifest_json,
        failures_json,
        input_index,
        array_index,
        smoke_array_index,
    ]
    existing = [path for path in managed if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Preparation outputs exist; use --overwrite: " + ", ".join(map(str, existing))
        )

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    task_inputs: dict[str, list[str]] = {ccd: [] for ccd in source_ccds}
    smoke_inputs: list[str] = []
    for source_index, source_path in enumerate(source_paths):
        identity = parse_source_name(source_path)
        try:
            source_records, source_inputs = prepare_source(
                source_index=source_index,
                source_path=source_path,
                output_root=output_root,
                partial_ts=partial_ts,
                overwrite=args.overwrite,
            )
        except Exception as error:
            failures.append(
                {
                    "source_index": source_index,
                    "sample_id": identity["sample_id"],
                    "ccd_code": identity["ccd_code"],
                    "length": identity["length"],
                    "source_cif": str(source_path),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            print(f"Preparation failed; continuing: {source_path}: {error}")
            continue
        records.extend(source_records)
        task_inputs[identity["ccd_code"]].extend(source_inputs)
        if identity["sample_id"] == args.smoke_sample_id:
            smoke_inputs.extend(source_inputs)

    if not records:
        raise ValueError("Every denovoval input failed preparation")
    if len(smoke_inputs) != len(partial_ts):
        raise ValueError(f"Smoke sample did not prepare all conditions: {args.smoke_sample_id}")

    write_csv(manifest_csv, records, overwrite=args.overwrite)
    atomic_write_text(
        input_index,
        "\n".join(record["input_json"] for record in records) + "\n",
        overwrite=args.overwrite,
    )
    tasks_dir = output_root / "tasks"
    array_lines: list[str] = []
    for ccd in source_ccds:
        task_path = tasks_dir / f"{ccd}.txt"
        contents = "\n".join(task_inputs[ccd])
        atomic_write_text(
            task_path,
            contents + ("\n" if contents else ""),
            overwrite=args.overwrite,
        )
        array_lines.append(str(task_path.resolve()))
    atomic_write_text(array_index, "\n".join(array_lines) + "\n", overwrite=args.overwrite)
    smoke_task = tasks_dir / f"smoke_{args.smoke_sample_id}.txt"
    atomic_write_text(
        smoke_task,
        "\n".join(smoke_inputs) + "\n",
        overwrite=args.overwrite,
    )
    atomic_write_text(
        smoke_array_index,
        str(smoke_task.resolve()) + "\n",
        overwrite=args.overwrite,
    )
    failure_payload = {
        "schema_version": 1,
        "status": "complete" if not failures else "complete_with_failures",
        "failures": failures,
    }
    atomic_write_json(failures_json, failure_payload, overwrite=args.overwrite)

    expected_inputs = len(source_paths) * len(partial_ts)
    summary = {
        "schema_version": 1,
        "status": "complete" if not failures else "complete_with_failures",
        "allow_partial_preparation": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rc_foundry_version": package_version,
        "source_dir": str(source_dir),
        "output_root": str(output_root),
        "prepared_inputs_manifest": str(manifest_csv),
        "preparation_failures": str(failures_json),
        "input_index": str(input_index),
        "array_index": str(array_index),
        "smoke_array_index": str(smoke_array_index),
        "smoke_sample_id": args.smoke_sample_id,
        "partial_t_values": partial_ts,
        "sequence_modes": ["fixed"],
        "ligand_fixed": True,
        "n_source_cifs": len(source_paths),
        "n_source_ccds": len(source_ccds),
        "n_prepared_sources": len({record["sample_id"] for record in records}),
        "n_prepared_inputs": len(records),
        "n_expected_inputs_without_failures": expected_inputs,
        "n_array_tasks": len(array_lines),
        "diagnostics": {
            "preparation_failures": len(failures),
            "missing": 0,
            "duplicate_sample_ids": 0,
        },
    }
    atomic_write_json(manifest_json, summary, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--partial-t", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--smoke-sample-id", default=DEFAULT_SMOKE_SAMPLE_ID)
    parser.add_argument("--expected-cif-count", type=int, default=EXPECTED_CIF_COUNT)
    parser.add_argument("--expected-ccd-count", type=int, default=EXPECTED_CCD_COUNT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
