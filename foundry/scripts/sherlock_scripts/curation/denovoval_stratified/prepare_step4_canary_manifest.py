#!/usr/bin/env python3
"""Create a small Step4 manifest spanning requested CCD/length groups."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Groups formatted as CCD:length, for example 12C:150.",
    )
    parser.add_argument("--per-group", type=int, default=2)
    return parser.parse_args()


def parse_groups(values: list[str]) -> set[tuple[str, int]]:
    groups: set[tuple[str, int]] = set()
    for value in values:
        ccd, separator, length_text = value.rpartition(":")
        if not separator or not ccd:
            raise ValueError(f"Invalid group {value!r}; expected CCD:length")
        groups.add((ccd.upper(), int(length_text)))
    return groups


def main() -> int:
    args = parse_args()
    if args.per_group <= 0:
        raise ValueError("--per-group must be positive")
    if args.output_manifest.exists():
        raise FileExistsError(args.output_manifest)

    groups = parse_groups(args.groups)
    counts: Counter[tuple[str, int]] = Counter()
    selected: list[dict[str, str]] = []
    with args.input_manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {args.input_manifest}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            group = (row["ccd"].upper(), int(row["length"]))
            if group not in groups or counts[group] >= args.per_group:
                continue
            selected.append(row)
            counts[group] += 1

    short = {
        f"{ccd}:{length}": counts[(ccd, length)]
        for ccd, length in sorted(groups)
        if counts[(ccd, length)] != args.per_group
    }
    if short:
        raise ValueError(f"Could not fill canary groups: {short}")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(selected)

    report = {
        "status": "complete",
        "input_manifest": str(args.input_manifest),
        "output_manifest": str(args.output_manifest),
        "per_group": args.per_group,
        "row_count": len(selected),
        "groups": {
            f"{ccd}:{length}": counts[(ccd, length)]
            for ccd, length in sorted(groups)
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
