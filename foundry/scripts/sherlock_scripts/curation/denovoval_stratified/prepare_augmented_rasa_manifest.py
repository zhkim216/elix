#!/usr/bin/env python3
"""Prepare the exact RASA refresh manifest for augmented joint curation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


EVAL_ROOT = Path("/scratch/users/zhkim216/datasets/evaluation_datasets")
VERSION_ROOT = EVAL_ROOT / "denovoval_cross_condition_foldseek_v1"
DEFAULT_RUN_ROOT = VERSION_ROOT / "runs/seed8_n100_hetero_rasa"
DEFAULT_P0_AUDIT = (
    VERSION_ROOT
    / "work/ss1p0_source/curation_steps/step1/selection/candidate_audit.tsv"
)
DEFAULT_P0_RAW_ROOT = VERSION_ROOT / "work/ss1p0_source/raw"
DEFAULT_P6_AUDIT = EVAL_ROOT / "curation/ver2/outputs/denovoval/step6/candidate_audit.tsv"
DEFAULT_P6_RAW_ROOT = (
    EVAL_ROOT / "curation/ver2/outputs/denovoval/step2/generated_samples/full"
)
DEFAULT_TASK_MANIFEST = VERSION_ROOT / "extra_generation/task_manifest.tsv"
DEFAULT_EXTRA_ROOT = VERSION_ROOT / "extra_generation/generated_samples/seed8_n100"

CONDITIONS = ("ss1p0_gamma1p0", "ss1p6_gamma0p6")
BASELINE_CCDS = ("GLU", "MET")
LENGTHS = (150, 300)
BASELINE_PER_GROUP = {
    "ss1p0_gamma1p0": 100,
    "ss1p6_gamma0p6": 300,
}
EXTRA_ORIGIN = "seed8_n100"
EXTRA_COUNTS = {
    ("ss1p6_gamma0p6", "G2P_len300"): 100,
    ("ss1p6_gamma0p6", "GLC_len300"): 100,
    ("ss1p0_gamma1p0", "GLU_len150"): 100,
    ("ss1p0_gamma1p0", "GLU_len300"): 100,
    ("ss1p6_gamma0p6", "GLU_len300"): 100,
}
EXPECTED_BASELINE_ROWS = 1600
EXPECTED_EXTRA_ROWS = 500
EXPECTED_ROWS = EXPECTED_BASELINE_ROWS + EXPECTED_EXTRA_ROWS
FIELDNAMES = (
    "staged_id",
    "condition",
    "group",
    "ccd",
    "length",
    "source_path",
    "json_path",
    "candidate_staged_id",
    "candidate_origin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--p0-audit", type=Path, default=DEFAULT_P0_AUDIT)
    parser.add_argument("--p0-raw-root", type=Path, default=DEFAULT_P0_RAW_ROOT)
    parser.add_argument("--p6-audit", type=Path, default=DEFAULT_P6_AUDIT)
    parser.add_argument("--p6-raw-root", type=Path, default=DEFAULT_P6_RAW_ROOT)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--extra-root", type=Path, default=DEFAULT_EXTRA_ROOT)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header: {path}")
        return list(reader)


def require_columns(rows: list[dict[str, str]], required: set[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows in {path}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def cif_stem(path: Path) -> str:
    if not path.name.endswith(".cif.gz"):
        raise ValueError(f"Expected .cif.gz path: {path}")
    return path.name[: -len(".cif.gz")]


def rasa_id(condition: str, origin: str, raw_staged_id: str) -> str:
    return f"{condition}__{origin}__{raw_staged_id}"


def baseline_rows(
    *, condition: str, audit_path: Path, raw_root: Path
) -> list[dict[str, Any]]:
    audit_rows = read_tsv(audit_path)
    require_columns(audit_rows, {"staged_id", "ccd", "length"}, audit_path)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for audit_row in audit_rows:
        ccd = audit_row["ccd"].strip().upper()
        length = int(audit_row["length"])
        if ccd not in BASELINE_CCDS or length not in LENGTHS:
            continue
        group = f"{ccd}_len{length}"
        candidate_id = audit_row["staged_id"]
        source_path = (raw_root / group / f"{candidate_id}.cif.gz").resolve()
        json_path = (raw_root / group / f"{candidate_id}.json").resolve()
        if not source_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"Missing baseline pair: {source_path}, {json_path}")
        selected.append(
            {
                "staged_id": rasa_id(condition, "baseline", candidate_id),
                "condition": condition,
                "group": group,
                "ccd": ccd,
                "length": length,
                "source_path": str(source_path),
                "json_path": str(json_path),
                "candidate_staged_id": candidate_id,
                "candidate_origin": "baseline",
            }
        )
        counts[group] += 1

    expected_groups = {f"{ccd}_len{length}" for ccd in BASELINE_CCDS for length in LENGTHS}
    expected_count = BASELINE_PER_GROUP[condition]
    invalid = {
        group: counts[group]
        for group in sorted(expected_groups)
        if counts[group] != expected_count
    }
    if invalid:
        raise ValueError(f"Unexpected baseline counts for {condition}: {invalid}")
    return selected


def extra_rows(task_manifest: Path, extra_root: Path) -> list[dict[str, Any]]:
    tasks = read_tsv(task_manifest)
    require_columns(tasks, {"condition", "group", "ccd", "length"}, task_manifest)
    task_keys = {(row["condition"], row["group"]) for row in tasks}
    if task_keys != set(EXTRA_COUNTS) or len(tasks) != len(EXTRA_COUNTS):
        raise ValueError(
            "Extra task identity mismatch: "
            f"missing={sorted(set(EXTRA_COUNTS) - task_keys)} "
            f"extra={sorted(task_keys - set(EXTRA_COUNTS))}"
        )

    selected: list[dict[str, Any]] = []
    for task in tasks:
        condition = task["condition"]
        group = task["group"]
        ccd = task["ccd"].strip().upper()
        length = int(task["length"])
        if group != f"{ccd}_len{length}":
            raise ValueError(f"Extra task group mismatch: {task}")
        group_root = extra_root / condition / group
        cif_paths = sorted(group_root.glob("*.cif.gz"))
        expected_count = EXTRA_COUNTS[(condition, group)]
        if len(cif_paths) != expected_count:
            raise ValueError(
                f"Expected {expected_count} extra CIFs in {group_root}, found {len(cif_paths)}"
            )
        for source_path in cif_paths:
            raw_staged_id = cif_stem(source_path)
            json_path = source_path.with_name(f"{raw_staged_id}.json")
            if not json_path.is_file():
                raise FileNotFoundError(json_path)
            candidate_id = f"{EXTRA_ORIGIN}__{raw_staged_id}"
            selected.append(
                {
                    "staged_id": rasa_id(condition, EXTRA_ORIGIN, raw_staged_id),
                    "condition": condition,
                    "group": group,
                    "ccd": ccd,
                    "length": length,
                    "source_path": str(source_path.resolve()),
                    "json_path": str(json_path.resolve()),
                    "candidate_staged_id": candidate_id,
                    "candidate_origin": EXTRA_ORIGIN,
                }
            )
    return selected


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    manifest_path = args.run_root.resolve() / "rasa/input_manifest.tsv"
    summary_path = args.run_root.resolve() / "rasa/input_manifest.json"
    if manifest_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite RASA input artifacts: {manifest_path}, {summary_path}"
        )

    rows: list[dict[str, Any]] = []
    rows.extend(
        baseline_rows(
            condition=CONDITIONS[0], audit_path=args.p0_audit, raw_root=args.p0_raw_root
        )
    )
    rows.extend(
        baseline_rows(
            condition=CONDITIONS[1], audit_path=args.p6_audit, raw_root=args.p6_raw_root
        )
    )
    rows.extend(extra_rows(args.task_manifest, args.extra_root))
    rows.sort(
        key=lambda row: (
            CONDITIONS.index(str(row["condition"])),
            str(row["group"]),
            str(row["candidate_origin"]),
            str(row["candidate_staged_id"]),
        )
    )
    ids = [str(row["staged_id"]) for row in rows]
    candidate_ids = [
        (str(row["condition"]), str(row["candidate_staged_id"])) for row in rows
    ]
    if len(ids) != len(set(ids)) or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("RASA or candidate identities are duplicated")
    origin_counts = Counter(str(row["candidate_origin"]) for row in rows)
    if origin_counts != {"baseline": EXPECTED_BASELINE_ROWS, EXTRA_ORIGIN: EXPECTED_EXTRA_ROWS}:
        raise ValueError(f"Unexpected origin counts: {dict(origin_counts)}")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")

    write_tsv(manifest_path, rows)
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "status": "complete",
            "row_count": len(rows),
            "origin_counts": dict(sorted(origin_counts.items())),
            "condition_counts": dict(
                sorted(Counter(str(row["condition"]) for row in rows).items())
            ),
            "group_condition_counts": {
                f"{condition}:{group}": count
                for (condition, group), count in sorted(
                    Counter(
                        (str(row["condition"]), str(row["group"])) for row in rows
                    ).items()
                )
            },
            "target_semantics": "finite heavy atoms with hetero=True and res_name=CCD",
        },
    )
    print(f"Wrote {manifest_path} ({len(rows)} rows)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
