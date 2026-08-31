#!/usr/bin/env python3
"""Reconcile denovoval partial-diffusion manifests without revalidating CIF contents."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/ensembles"
)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def validate_complete_manifest(
    *,
    path: Path,
    record: dict[str, str],
    expected_num_samples: int,
) -> tuple[bool, str]:
    try:
        manifest = load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return False, f"manifest unreadable: {error}"
    identity = str(manifest.get("sample_id") or manifest.get("role_sample_id") or "")
    if identity != record["sample_id"] or manifest.get("condition") != record["condition"]:
        return False, "manifest identity mismatch"
    if manifest.get("status") != "complete":
        return False, f"manifest status={manifest.get('status')!r}"
    if (
        int(manifest.get("num_samples_requested", -1)) != expected_num_samples
        or int(manifest.get("num_samples_generated", -1)) != expected_num_samples
    ):
        return False, "manifest sample count mismatch"
    diagnostics = manifest.get("diagnostics", {})
    if any(int(value) != 0 for value in diagnostics.values()):
        return False, f"nonzero diagnostics: {diagnostics}"
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_num_samples:
        return False, "manifest samples list mismatch"
    if [item.get("sample_index") for item in samples] != list(range(expected_num_samples)):
        return False, "sample indices mismatch"
    missing_paths = [
        item.get("output_cif")
        for item in samples
        if not Path(str(item.get("output_cif", ""))).is_file()
    ]
    if missing_paths:
        return False, f"missing {len(missing_paths)} declared output CIF(s)"
    if not Path(str(manifest.get("original_cif", ""))).is_file():
        return False, "missing per-condition original CIF"
    return True, ""


def audit(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    preparation_path = output_root / "preparation_manifest.json"
    prepared_csv = output_root / "prepared_inputs_manifest.csv"
    preparation_failures_path = output_root / "preparation_failures.json"
    for path in (preparation_path, prepared_csv, preparation_failures_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    preparation = load_json(preparation_path)
    preparation_failures = load_json(preparation_failures_path).get("failures", [])
    with prepared_csv.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    keys = [(row["condition"], row["sample_id"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Prepared manifest contains duplicate condition/sample_id rows")

    attempts_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    unreadable_attempts: list[str] = []
    for path in output_root.glob("failure_attempts/*/*/*.json"):
        try:
            attempt = load_json(path)
        except (OSError, json.JSONDecodeError):
            unreadable_attempts.append(str(path))
            continue
        key = (str(attempt.get("condition", "")), str(attempt.get("sample_id", "")))
        attempt["failure_path"] = str(path)
        attempts_by_key[key].append(attempt)

    complete: list[dict[str, str]] = []
    current_failed: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    retry_inputs: list[str] = []
    complete_keys: set[tuple[str, str]] = set()
    for record in records:
        key = (record["condition"], record["sample_id"])
        manifest_path = (
            output_root / record["condition"] / "samples" / record["sample_id"] / "manifest.json"
        )
        if manifest_path.is_file():
            valid, reason = validate_complete_manifest(
                path=manifest_path,
                record=record,
                expected_num_samples=args.expected_num_samples,
            )
            if valid:
                complete.append(record)
                complete_keys.add(key)
            else:
                invalid.append({**record, "reason": reason, "manifest": str(manifest_path)})
                retry_inputs.append(record["input_json"])
            continue
        attempts = sorted(
            attempts_by_key.get(key, []), key=lambda item: str(item.get("failed_at_utc", ""))
        )
        if attempts:
            latest = attempts[-1]
            current_failed.append(
                {
                    **record,
                    "attempt_count": len(attempts),
                    "error_type": latest.get("error_type", ""),
                    "error_message": latest.get("error_message", ""),
                    "failure_path": latest.get("failure_path", ""),
                }
            )
        else:
            missing.append(record)
        retry_inputs.append(record["input_json"])

    resolved_attempts = [
        attempt
        for key, attempts in attempts_by_key.items()
        if key in complete_keys
        for attempt in attempts
    ]
    expected_input_conditions = int(preparation["n_expected_inputs_without_failures"])
    partial_t_count = len(preparation["partial_t_values"])
    accounted_preparation_conditions = len(records) + len(preparation_failures) * partial_t_count
    accounted_prepared_inputs = len(complete) + len(current_failed) + len(missing) + len(invalid)
    reconciliation_errors: list[str] = []
    if accounted_preparation_conditions != expected_input_conditions:
        reconciliation_errors.append(
            "prepared rows plus preparation failures do not match expected input-conditions: "
            f"{accounted_preparation_conditions} != {expected_input_conditions}"
        )
    if accounted_prepared_inputs != len(records):
        reconciliation_errors.append(
            "complete, failed, missing, and invalid statuses do not reconcile to prepared rows: "
            f"{accounted_prepared_inputs} != {len(records)}"
        )
    blocking = bool(missing or invalid or unreadable_attempts or reconciliation_errors)
    has_failures = bool(preparation_failures or current_failed)
    status = "incomplete" if blocking else ("complete_with_failures" if has_failures else "complete")
    summary = {
        "schema_version": 1,
        "status": status,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "preparation_manifest": str(preparation_path),
        "n_source_cifs": int(preparation["n_source_cifs"]),
        "n_expected_input_conditions": expected_input_conditions,
        "n_accounted_preparation_conditions": accounted_preparation_conditions,
        "n_prepared_inputs": len(records),
        "n_accounted_prepared_inputs": accounted_prepared_inputs,
        "n_complete_inputs": len(complete),
        "n_complete_sample_cifs": len(complete) * args.expected_num_samples,
        "n_current_generation_failures": len(current_failed),
        "n_preparation_failures": len(preparation_failures),
        "n_missing_inputs": len(missing),
        "n_invalid_manifests": len(invalid),
        "n_resolved_failure_attempts": len(resolved_attempts),
        "n_unreadable_failure_attempts": len(unreadable_attempts),
        "reconciliation_errors": reconciliation_errors,
        "expected_num_samples_per_input": args.expected_num_samples,
        "validation_scope": [
            "manifest identity and declared sample count",
            "zero manifest diagnostics",
            "declared output file existence",
        ],
        "intentionally_not_revalidated": [
            "CIF content parsing",
            "output CIF SHA256",
            "alignment metrics already checked during generation",
        ],
    }
    summary_path = output_root / "generation_summary.json"
    retry_path = output_root / "retry_index.txt"
    failures_md = output_root / "FAILURES.md"
    atomic_write(summary_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_write(retry_path, "\n".join(sorted(set(retry_inputs))) + ("\n" if retry_inputs else ""))

    lines = [
        "# Denovoval partial-diffusion failures",
        "",
        f"Generated: {summary['audited_at_utc']}",
        "",
        "## Summary",
        "",
        f"- Status: `{status}`",
        f"- Prepared input-conditions: {len(records)}",
        f"- Complete input-conditions: {len(complete)}",
        f"- Current preparation failures: {len(preparation_failures)}",
        f"- Current generation failures: {len(current_failed)}",
        f"- Missing inputs: {len(missing)}",
        f"- Invalid manifests: {len(invalid)}",
        f"- Reconciliation errors: {len(reconciliation_errors)}",
        f"- Resolved failure attempts: {len(resolved_attempts)}",
        "",
        "Final audit intentionally does not reparse or rehash generated CIFs.",
        "",
        "## Current preparation failures",
        "",
    ]
    if preparation_failures:
        lines.extend(["| Sample | CCD | Error |", "|---|---|---|"])
        for failure in preparation_failures:
            lines.append(
                f"| {markdown_cell(failure.get('sample_id'))} | "
                f"{markdown_cell(failure.get('ccd_code'))} | "
                f"{markdown_cell(failure.get('error_message'))} |"
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Current generation failures", ""])
    if current_failed:
        lines.extend(["| Sample | Condition | Attempts | Error |", "|---|---|---:|---|"])
        for failure in current_failed:
            lines.append(
                f"| {markdown_cell(failure['sample_id'])} | {markdown_cell(failure['condition'])} | "
                f"{failure['attempt_count']} | {markdown_cell(failure['error_message'])} |"
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Missing or invalid outputs", ""])
    problems = [
        (row["sample_id"], row["condition"], "missing manifest") for row in missing
    ] + [
        (row["sample_id"], row["condition"], row["reason"]) for row in invalid
    ] + [
        ("-", "-", reason) for reason in reconciliation_errors
    ]
    if problems:
        lines.extend(["| Sample | Condition | Problem |", "|---|---|---|"])
        for sample_id, condition, reason in problems:
            lines.append(
                f"| {markdown_cell(sample_id)} | {markdown_cell(condition)} | "
                f"{markdown_cell(reason)} |"
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Resolved failures", ""])
    if resolved_attempts:
        lines.extend(["| Sample | Condition | Previous error |", "|---|---|---|"])
        for attempt in sorted(
            resolved_attempts,
            key=lambda item: (str(item.get("sample_id")), str(item.get("condition"))),
        ):
            lines.append(
                f"| {markdown_cell(attempt.get('sample_id'))} | "
                f"{markdown_cell(attempt.get('condition'))} | "
                f"{markdown_cell(attempt.get('error_message'))} |"
            )
    else:
        lines.append("None.")
    atomic_write(failures_md, "\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if blocking else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-num-samples", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(audit(parse_args()))
