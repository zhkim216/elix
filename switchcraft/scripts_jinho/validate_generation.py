#!/usr/bin/env python3
"""Validate SwitchCraft design directories against the frozen run manifest."""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


DEFAULT_MANIFEST = Path(
    "/scratch/users/zhkim216/experiment_result_analysis/benchmarks/"
    "state_switching/denovoval/manifests/manifest.json"
)
DESIGN_RE = re.compile(r"^design([0-9]+)$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Validate only this case ID; may be supplied more than once.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-details", type=int, default=100)
    return parser.parse_args()


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def expected_files(case):
    names = set()
    for state in range(case["num_states"]):
        names.add("state{}.pkl".format(state))
        for sample in range(5):
            stem = "state{}_sample{}".format(state, sample)
            names.add(stem + ".pdb")
            names.add(stem + ".cif")
    for motif in case.get("motifs", []):
        names.add(motif + "_spec.pkl")
    return names


def append_detail(target, item, max_details):
    if len(target) < max_details:
        target.append(item)


def validate_case(case, max_details):
    design_root = Path(case["design_root"])
    requested = case["num_designs"]
    expected = expected_files(case)
    by_index = {}
    unexpected_root_entries = []
    unexpected_root_entry_count = 0

    if design_root.is_dir():
        for entry in design_root.iterdir():
            match = DESIGN_RE.match(entry.name)
            if entry.is_dir() and match:
                by_index.setdefault(int(match.group(1)), []).append(entry)
            else:
                unexpected_root_entry_count += 1
                append_detail(unexpected_root_entries, entry.name, max_details)

    valid_indices = {index for index in by_index if 0 <= index < requested}
    missing_indices = []
    for index in range(requested):
        if index not in valid_indices:
            append_detail(missing_indices, index, max_details)

    duplicate_indices = []
    for index, paths in sorted(by_index.items()):
        if len(paths) > 1:
            append_detail(
                duplicate_indices,
                {"index": index, "paths": [str(path) for path in paths]},
                max_details,
            )

    unexpected_indices = []
    for index, paths in sorted(by_index.items()):
        if index < 0 or index >= requested:
            append_detail(
                unexpected_indices,
                {"index": index, "paths": [str(path) for path in paths]},
                max_details,
            )

    incomplete_details = []
    unexpected_file_details = []
    complete = 0
    incomplete = 0
    unexpected_file_count = 0
    for index in sorted(valid_indices):
        design_dir = sorted(by_index[index], key=lambda path: path.name)[0]
        present = {entry.name for entry in design_dir.iterdir() if entry.is_file()}
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing:
            incomplete += 1
            append_detail(
                incomplete_details,
                {"index": index, "missing_files": missing},
                max_details,
            )
        else:
            complete += 1
        if extra:
            unexpected_file_count += len(extra)
            append_detail(
                unexpected_file_details,
                {"index": index, "unexpected_files": extra},
                max_details,
            )

    missing_count = requested - len(valid_indices)
    duplicate_count = sum(max(0, len(paths) - 1) for paths in by_index.values())
    unexpected_index_count = sum(
        len(paths) for index, paths in by_index.items() if index < 0 or index >= requested
    )
    return {
        "case_id": case["case_id"],
        "family": case["family"],
        "design_root": str(design_root),
        "requested": requested,
        "present_in_range": len(valid_indices),
        "complete": complete,
        "incomplete": incomplete,
        "missing": missing_count,
        "duplicate_index_paths": duplicate_count,
        "unexpected_index_paths": unexpected_index_count,
        "unexpected_files": unexpected_file_count,
        "unexpected_root_entries_observed": unexpected_root_entry_count,
        "details": {
            "missing_indices": missing_indices,
            "incomplete_designs": incomplete_details,
            "duplicate_indices": duplicate_indices,
            "unexpected_indices": unexpected_indices,
            "unexpected_files": unexpected_file_details,
            "unexpected_root_entries": unexpected_root_entries,
            "max_details_per_category": max_details,
        },
    }


def main():
    args = parse_args()
    if args.max_details < 0:
        raise ValueError("--max-details must be nonnegative")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    if args.case_id:
        requested_ids = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in requested_ids]
        found_ids = {case["case_id"] for case in cases}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError("Unknown case IDs: {}".format(", ".join(missing_ids)))

    case_reports = [validate_case(case, args.max_details) for case in cases]
    totals = {
        key: sum(report[key] for report in case_reports)
        for key in (
            "requested",
            "present_in_range",
            "complete",
            "incomplete",
            "missing",
            "duplicate_index_paths",
            "unexpected_index_paths",
            "unexpected_files",
            "unexpected_root_entries_observed",
        )
    }
    blocking_incomplete = any(
        totals[key] != 0
        for key in (
            "incomplete",
            "missing",
            "duplicate_index_paths",
            "unexpected_index_paths",
        )
    )
    report = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "selected_case_count": len(cases),
        "complete": not blocking_incomplete,
        "totals": totals,
        "cases": case_reports,
    }
    report_path = args.report
    if report_path is None:
        report_path = Path(manifest["output_root"]) / "manifests" / "completeness_report.json"
    atomic_json(report_path.resolve(), report)
    print(json.dumps({"report": str(report_path), "complete": not blocking_incomplete, "totals": totals}, indent=2, sort_keys=True))
    raise SystemExit(0 if not blocking_incomplete else 1)


if __name__ == "__main__":
    main()
