#!/usr/bin/env python3
"""Copy the 304 unchanged Foldseek group results into the augmented run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from prepare_augmented_rasa_manifest import DEFAULT_RUN_ROOT, VERSION_ROOT


SOURCE_GROUPS_ROOT = VERSION_ROOT / "foldseek/groups"
AFFECTED_GROUPS = frozenset({"G2P_len300", "GLC_len300", "GLU_len150", "GLU_len300"})
EXPECTED_GROUPS = 308
EXPECTED_REUSED_GROUPS = EXPECTED_GROUPS - len(AFFECTED_GROUPS)
EXPECTED_CANDIDATES_PER_GROUP = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--source-groups-root", type=Path, default=SOURCE_GROUPS_ROOT)
    return parser.parse_args()


def row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


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
    source_root = args.source_groups_root.resolve()
    target_root = args.run_root.resolve() / "foldseek/groups"
    summary_path = args.run_root.resolve() / "foldseek/reused_groups.json"
    if target_root.exists() or target_root.is_symlink() or summary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Foldseek cache: {target_root}, {summary_path}"
        )
    source_groups = sorted(path for path in source_root.iterdir() if path.is_dir())
    if len(source_groups) != EXPECTED_GROUPS:
        raise ValueError(f"Expected {EXPECTED_GROUPS} source groups, found {len(source_groups)}")
    source_names = {path.name for path in source_groups}
    if not AFFECTED_GROUPS.issubset(source_names):
        raise ValueError(f"Missing affected source groups: {sorted(AFFECTED_GROUPS - source_names)}")

    reused = [path for path in source_groups if path.name not in AFFECTED_GROUPS]
    if len(reused) != EXPECTED_REUSED_GROUPS:
        raise ValueError(f"Expected {EXPECTED_REUSED_GROUPS} reusable groups, found {len(reused)}")
    validated: list[dict[str, Any]] = []
    for source_group in reused:
        manifest_path = source_group / "manifest.json"
        assignments_path = source_group / "cluster_assignments.tsv"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("status") != "complete"
            or manifest.get("group") != source_group.name
            or int(manifest.get("candidate_count", -1)) != EXPECTED_CANDIDATES_PER_GROUP
            or row_count(assignments_path) != EXPECTED_CANDIDATES_PER_GROUP
        ):
            raise ValueError(f"Incomplete reusable Foldseek group: {source_group}")
        validated.append(
            {
                "group": source_group.name,
                "manifest_sha256": sha256_file(manifest_path),
                "assignments_sha256": sha256_file(assignments_path),
            }
        )

    target_root.mkdir(parents=True)
    for source_group in reused:
        shutil.copytree(source_group, target_root / source_group.name)
    write_json(
        summary_path,
        {
            "schema_version": 1,
            "status": "complete",
            "source_groups_root": str(source_root),
            "target_groups_root": str(target_root),
            "source_group_count": len(source_groups),
            "reused_group_count": len(reused),
            "affected_groups_excluded": sorted(AFFECTED_GROUPS),
            "groups": validated,
        },
    )
    print(f"Copied {len(reused)} unchanged Foldseek groups to {target_root}")


if __name__ == "__main__":
    main()
