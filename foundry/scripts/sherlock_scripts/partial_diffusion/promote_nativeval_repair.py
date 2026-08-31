#!/usr/bin/env python3
"""Promote a complete nativeval subset repair into the canonical ensemble tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_METADATA_FILES = (
    "preparation_manifest.json",
    "prepared_inputs_manifest.csv",
    "input_index.txt",
    "array_index.txt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def atomic_write_csv(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def replace_path_prefix(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [replace_path_prefix(item, old, new) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_path_prefix(item, old, new)
            for key, item in value.items()
        }
    return value


def move(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def sample_manifest(root: Path, condition: str, role_id: str) -> Path:
    return root / condition / "samples" / role_id / "manifest.json"


def input_json(root: Path, condition: str, role_id: str) -> Path:
    return root / condition / "inputs" / f"{role_id}.json"


def validate_repair(
    repair_root: Path,
    *,
    expected_samples: int,
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    _, rows = load_csv(repair_root / "prepared_inputs_manifest.csv")
    if not rows:
        raise ValueError("Repair prepared-input manifest is empty")
    roles = sorted({row["role_sample_id"] for row in rows})
    conditions = sorted({row["condition"] for row in rows})
    expected_keys = {(role_id, condition) for role_id in roles for condition in conditions}
    actual_keys = {(row["role_sample_id"], row["condition"]) for row in rows}
    if actual_keys != expected_keys:
        raise ValueError("Repair manifest is not a complete role-by-condition product")

    for role_id, condition in sorted(expected_keys):
        manifest_path = sample_manifest(repair_root, condition, role_id)
        manifest = load_json(manifest_path)
        sample_indices = sorted(int(item["sample_index"]) for item in manifest["samples"])
        if (
            manifest.get("status") != "complete"
            or manifest.get("num_samples_generated") != expected_samples
            or sample_indices != list(range(expected_samples))
        ):
            raise ValueError(
                f"Incomplete repair manifest for {condition}/{role_id}: "
                f"status={manifest.get('status')!r}, "
                f"generated={manifest.get('num_samples_generated')!r}"
            )
        for sample in manifest["samples"]:
            output_cif = Path(sample["output_cif"])
            if not output_cif.is_file():
                raise FileNotFoundError(output_cif)
    return roles, conditions, rows


def promote(args: argparse.Namespace) -> dict[str, Any]:
    canonical_root = args.canonical_root.resolve()
    repair_root = args.repair_root.resolve()
    legacy_root = args.legacy_root.resolve()
    if repair_root.parent != canonical_root:
        raise ValueError("Repair root must be a direct hidden child of the canonical root")
    if legacy_root.exists():
        raise FileExistsError(f"Legacy destination already exists: {legacy_root}")

    roles, conditions, repair_rows = validate_repair(
        repair_root,
        expected_samples=args.expected_samples,
    )
    canonical_fields, canonical_rows = load_csv(
        canonical_root / "prepared_inputs_manifest.csv"
    )
    repair_fields, _ = load_csv(repair_root / "prepared_inputs_manifest.csv")
    if canonical_fields != repair_fields:
        raise ValueError("Canonical and repair prepared-input schemas differ")

    replacement_keys = {
        (row["role_sample_id"], row["condition"]) for row in repair_rows
    }
    canonical_keys = {
        (row["role_sample_id"], row["condition"]) for row in canonical_rows
    }
    missing_keys = sorted(replacement_keys - canonical_keys)
    if missing_keys:
        raise KeyError(f"Canonical manifest lacks repair keys: {missing_keys}")

    required_paths = {
        *(canonical_root / name for name in ROOT_METADATA_FILES),
        *(canonical_root / "original_samples" / f"{role_id}.cif" for role_id in roles),
        *(repair_root / "original_samples" / f"{role_id}.cif" for role_id in roles),
    }
    for role_id, condition in sorted(replacement_keys):
        required_paths.update(
            {
                input_json(canonical_root, condition, role_id),
                sample_manifest(canonical_root, condition, role_id).parent,
                input_json(repair_root, condition, role_id),
                sample_manifest(repair_root, condition, role_id).parent,
            }
        )
    missing_paths = sorted(str(path) for path in required_paths if not path.exists())
    if missing_paths:
        raise FileNotFoundError(f"Promotion inputs are missing: {missing_paths}")

    summary = {
        "status": "ready" if not args.apply else "complete",
        "canonical_root": str(canonical_root),
        "repair_root": str(repair_root),
        "legacy_root": str(legacy_root),
        "roles": roles,
        "conditions": conditions,
        "expected_samples_per_role_condition": args.expected_samples,
        "n_replaced_role_conditions": len(replacement_keys),
    }
    if not args.apply:
        return summary

    legacy_root.mkdir(parents=True)
    metadata_snapshot = legacy_root / "root_metadata_before_promotion"
    metadata_snapshot.mkdir()
    for name in ROOT_METADATA_FILES:
        shutil.copy2(canonical_root / name, metadata_snapshot / name)

    for role_id in roles:
        move(
            canonical_root / "original_samples" / f"{role_id}.cif",
            legacy_root / "original_samples" / f"{role_id}.cif",
        )

    for role_id, condition in sorted(replacement_keys):
        move(
            input_json(canonical_root, condition, role_id),
            input_json(legacy_root, condition, role_id),
        )
        move(
            sample_manifest(canonical_root, condition, role_id).parent,
            sample_manifest(legacy_root, condition, role_id).parent,
        )

    for role_id in roles:
        move(
            repair_root / "original_samples" / f"{role_id}.cif",
            canonical_root / "original_samples" / f"{role_id}.cif",
        )

    for role_id, condition in sorted(replacement_keys):
        move(
            input_json(repair_root, condition, role_id),
            input_json(canonical_root, condition, role_id),
        )
        move(
            sample_manifest(repair_root, condition, role_id).parent,
            sample_manifest(canonical_root, condition, role_id).parent,
        )

    canonical_prefix = str(canonical_root)
    repair_prefix = str(repair_root)
    replacement_rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in repair_rows:
        role_id = row["role_sample_id"]
        condition = row["condition"]
        key = (role_id, condition)
        canonical_input = input_json(canonical_root, condition, role_id)
        payload = replace_path_prefix(load_json(canonical_input), repair_prefix, canonical_prefix)
        atomic_write_json(canonical_input, payload)
        input_sha256 = sha256_file(canonical_input)

        canonical_manifest = sample_manifest(canonical_root, condition, role_id)
        manifest = replace_path_prefix(
            load_json(canonical_manifest), repair_prefix, canonical_prefix
        )
        manifest["input_json_sha256"] = input_sha256
        manifest["prepared_manifest_row"]["input_json_sha256"] = input_sha256
        atomic_write_json(canonical_manifest, manifest)

        rewritten_row = replace_path_prefix(dict(row), repair_prefix, canonical_prefix)
        rewritten_row["input_json_sha256"] = input_sha256
        replacement_rows[key] = rewritten_row

    # The six t=10 ensembles remain unchanged. Point their producer provenance
    # at the archived original instead of the newly promoted t=2/5 source.
    for role_id in roles:
        old_original = canonical_root / "original_samples" / f"{role_id}.cif"
        archived_original = legacy_root / "original_samples" / f"{role_id}.cif"
        condition = "partialt10"
        key = (role_id, condition)
        if key not in canonical_keys:
            continue
        t10_input = input_json(canonical_root, condition, role_id)
        payload = replace_path_prefix(
            load_json(t10_input), str(old_original), str(archived_original)
        )
        atomic_write_json(t10_input, payload)
        input_sha256 = sha256_file(t10_input)

        t10_manifest_path = sample_manifest(canonical_root, condition, role_id)
        manifest = replace_path_prefix(
            load_json(t10_manifest_path), str(old_original), str(archived_original)
        )
        manifest["input_json_sha256"] = input_sha256
        manifest["prepared_manifest_row"]["input_json_sha256"] = input_sha256
        atomic_write_json(t10_manifest_path, manifest)

        original_row = next(
            row
            for row in canonical_rows
            if (row["role_sample_id"], row["condition"]) == key
        )
        rewritten_row = replace_path_prefix(
            dict(original_row), str(old_original), str(archived_original)
        )
        rewritten_row["input_json_sha256"] = input_sha256
        replacement_rows[key] = rewritten_row

    merged_rows = [
        replacement_rows.get((row["role_sample_id"], row["condition"]), row)
        for row in canonical_rows
    ]
    atomic_write_csv(
        canonical_root / "prepared_inputs_manifest.csv",
        fieldnames=canonical_fields,
        rows=merged_rows,
    )

    preparation_manifest_path = canonical_root / "preparation_manifest.json"
    preparation_manifest = load_json(preparation_manifest_path)
    repair_staging_manifest = Path(
        load_json(repair_root / "preparation_manifest.json")["staging_manifest"]
    )
    preparation_manifest.setdefault("semantic_repairs", []).append(
        {
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
            "semantic_contract": "elix_native_stage1_structure_v1",
            "roles": roles,
            "conditions": conditions,
            "staging_manifest": str(repair_staging_manifest),
            "staging_manifest_sha256": sha256_file(repair_staging_manifest),
            "legacy_root": str(legacy_root),
        }
    )
    atomic_write_json(preparation_manifest_path, preparation_manifest)

    summary["promoted_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(legacy_root / "promotion_report.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=32)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(promote(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
