#!/usr/bin/env python3
"""Scan, merge, and gather CSV artifacts from Sherlock eval output trees.

This utility recognizes two shard conventions:
- ``{stem}_array_{N}.csv`` files in one directory.
- ``array_{N}/.../*.csv`` directory shards under one parent directory.

Parameter-sweep directories such as ``scale_*`` are never merged implicitly;
they are gathered with their relative paths preserved.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import tarfile
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SUFFIX_SHARD_RE = re.compile(r"^(.+)_array_(\d+)\.csv$")
ARRAY_DIR_RE = re.compile(r"^array_(\d+)$")


def natural_key(value: str | Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value))]


def is_backup_csv(path: Path) -> bool:
    return path.name.endswith(".bak.csv")


def suffix_shard_match(path: Path) -> tuple[str, int] | None:
    match = SUFFIX_SHARD_RE.match(path.name)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def array_dir_for_csv(path: Path, root: Path) -> tuple[Path, int] | None:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return None

    current = root
    for part in relative_parts[:-1]:
        current = current / part
        match = ARRAY_DIR_RE.match(part)
        if match:
            return current, int(match.group(1))
    return None


def iter_csvs(src_dir: Path) -> list[Path]:
    return sorted(
        [path for path in src_dir.rglob("*.csv") if path.is_file() and not is_backup_csv(path)],
        key=natural_key,
    )


def discover_suffix_groups(src_dirs: list[Path]) -> dict[tuple[Path, str], list[tuple[int, Path]]]:
    groups: dict[tuple[Path, str], list[tuple[int, Path]]] = defaultdict(list)
    seen: set[Path] = set()
    for src_dir in src_dirs:
        if not src_dir.is_dir():
            continue
        for csv_path in iter_csvs(src_dir):
            resolved = csv_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            match = suffix_shard_match(csv_path)
            if match is None:
                continue
            base_name, shard_id = match
            groups[(csv_path.parent, base_name)].append((shard_id, csv_path))
    return groups


def discover_array_dir_groups(src_dirs: list[Path]) -> dict[tuple[Path, Path], list[tuple[int, Path]]]:
    groups: dict[tuple[Path, Path], list[tuple[int, Path]]] = defaultdict(list)
    seen: set[Path] = set()
    for src_dir in src_dirs:
        if not src_dir.is_dir():
            continue
        root = src_dir.resolve()
        for csv_path in iter_csvs(root):
            resolved = csv_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            array_info = array_dir_for_csv(csv_path, root)
            if array_info is None:
                continue
            array_dir, shard_id = array_info
            groups[(array_dir.parent, csv_path.relative_to(array_dir))].append((shard_id, csv_path))
    return groups


def is_under_array_dir(path: Path, root: Path) -> bool:
    return array_dir_for_csv(path, root) is not None


def discover_gather_csvs(src_dirs: list[Path], *, include_array_shards: bool) -> list[tuple[Path, Path]]:
    found: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    root_names: dict[str, int] = {}

    for src_dir in src_dirs:
        if not src_dir.is_dir():
            print(f"  [SKIP] Not a directory: {src_dir}")
            continue
        root = src_dir.resolve()
        root_count = root_names.get(root.name, 0) + 1
        root_names[root.name] = root_count
        archive_root = root.name if root_count == 1 else f"{root.name}_{root_count}"

        for csv_path in iter_csvs(root):
            if not include_array_shards:
                if suffix_shard_match(csv_path) is not None:
                    continue
                if is_under_array_dir(csv_path, root):
                    continue
            resolved = csv_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            arcname = Path(archive_root) / csv_path.relative_to(root)
            found.append((csv_path, arcname))
    return sorted(found, key=lambda item: natural_key(item[1]))


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("missing CSV header")
        return list(reader.fieldnames), list(reader)


def merge_fieldnames(fieldname_lists: list[list[str]]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for fieldnames in fieldname_lists:
        for fieldname in fieldnames:
            if fieldname in seen:
                continue
            merged.append(fieldname)
            seen.add(fieldname)
    return merged


def format_stat(path: Path) -> str:
    st = path.stat()
    mtime = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    return f"size={st.st_size}B, mtime={mtime}"


def backup_existing(out_path: Path) -> Path | None:
    if not out_path.exists():
        return None
    backup_path = out_path.with_name(out_path.stem + ".bak.csv")
    shutil.copy2(out_path, backup_path)
    return backup_path


def validate_shard_ids(
    label: str,
    shard_ids: list[int],
    *,
    allow_gaps: bool,
) -> bool:
    ok = True
    if len(set(shard_ids)) != len(shard_ids):
        duplicates = sorted({sid for sid in shard_ids if shard_ids.count(sid) > 1})
        msg = f"duplicate shard ids for {label}: {duplicates}"
        if allow_gaps:
            print(f"    WARN: {msg} (allowed via --allow-gaps)")
        else:
            print(f"    ERROR: {msg}")
            ok = False

    expected = set(range(max(shard_ids) + 1))
    missing = sorted(expected - set(shard_ids))
    if missing:
        msg = f"missing shard ids for {label}: {missing} (have {len(shard_ids)} of {max(shard_ids) + 1})"
        if allow_gaps:
            print(f"    WARN: {msg} (allowed via --allow-gaps)")
        else:
            print(f"    ERROR: {msg}")
            ok = False
    return ok


def merge_csv_group(
    *,
    label: str,
    entries: list[tuple[int, Path]],
    out_path: Path,
    force: bool,
    allow_gaps: bool,
    allow_broken: bool,
) -> bool:
    entries = sorted(entries, key=lambda item: item[0])
    shard_ids = [shard_id for shard_id, _ in entries]
    csv_paths = [path for _, path in entries]

    print(f"  Merging {len(csv_paths)} files for {label}:")
    for csv_path in csv_paths:
        print(f"    - {csv_path}")

    ok = validate_shard_ids(label, shard_ids, allow_gaps=allow_gaps)
    rows: list[dict[str, str]] = []
    fieldname_lists: list[list[str]] = []
    failed: list[str] = []
    for csv_path in csv_paths:
        try:
            fieldnames, csv_rows = read_csv_rows(csv_path)
        except Exception as exc:
            failed.append(f"{csv_path}: {exc}")
            continue
        fieldname_lists.append(fieldnames)
        rows.extend(csv_rows)

    if failed:
        for failure in failed:
            print(f"    WARN: failed to read shard {failure}")
        if not allow_broken:
            print(f"    ERROR: {len(failed)} shard(s) failed to read for {label}")
            ok = False

    if not rows and not fieldname_lists:
        print(f"    WARN: no readable shards for {label}, skipping write")
        return False if not allow_broken else ok

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            _, old_rows = read_csv_rows(out_path)
            old_row_count = len(old_rows)
        except Exception:
            old_row_count = -1
        print(
            f"    NOTE: {out_path} already exists "
            f"({format_stat(out_path)}, rows={old_row_count}); new merged rows={len(rows)}"
        )
        if force:
            print("    NOTE: --force set, overwriting without backup")
        else:
            backup = backup_existing(out_path)
            print(f"    NOTE: backed up existing file to {backup}")

    fieldnames = merge_fieldnames(fieldname_lists)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"    -> Saved merged CSV: {out_path} ({len(rows)} rows)")
    return ok


def merge_artifacts(
    src_dirs: list[Path],
    *,
    force: bool = False,
    allow_gaps: bool = False,
    allow_broken: bool = False,
) -> bool:
    ok = True
    suffix_groups = discover_suffix_groups(src_dirs)
    array_dir_groups = discover_array_dir_groups(src_dirs)

    print(f"Found {len(suffix_groups)} suffix shard group(s)")
    for (parent_dir, base_name), entries in sorted(suffix_groups.items(), key=lambda item: natural_key(item[0][0] / item[0][1])):
        out_path = parent_dir / f"{base_name}.csv"
        ok &= merge_csv_group(
            label=str(out_path),
            entries=entries,
            out_path=out_path,
            force=force,
            allow_gaps=allow_gaps,
            allow_broken=allow_broken,
        )

    print(f"Found {len(array_dir_groups)} array-directory shard group(s)")
    for (parent_dir, rel_path), entries in sorted(array_dir_groups.items(), key=lambda item: natural_key(item[0][0] / item[0][1])):
        out_path = parent_dir / "merged" / rel_path
        ok &= merge_csv_group(
            label=str(out_path),
            entries=entries,
            out_path=out_path,
            force=force,
            allow_gaps=allow_gaps,
            allow_broken=allow_broken,
        )

    if not suffix_groups and not array_dir_groups:
        print("No array shard CSV groups found")
    return ok


def gather_artifacts(
    src_dirs: list[Path],
    output_tar: Path,
    *,
    include_array_shards: bool = False,
) -> int:
    csvs = discover_gather_csvs(src_dirs, include_array_shards=include_array_shards)
    output_tar = output_tar.expanduser().resolve()
    output_tar.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for csv_path, arcname in csvs:
            dest = tmp / arcname
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, dest)
        with tarfile.open(output_tar, "w:gz") as tar:
            for item in sorted([path for path in tmp.rglob("*") if path.is_file()], key=natural_key):
                tar.add(item, arcname=item.relative_to(tmp))

    size_mb = output_tar.stat().st_size / (1024 * 1024)
    print(f"Done: {len(csvs)} CSV files -> {output_tar} ({size_mb:.1f} MB)")
    return len(csvs)


def scan_artifacts(src_dirs: list[Path], *, include_array_shards: bool = False) -> None:
    suffix_groups = discover_suffix_groups(src_dirs)
    array_dir_groups = discover_array_dir_groups(src_dirs)
    gather_csvs = discover_gather_csvs(src_dirs, include_array_shards=include_array_shards)
    print(f"suffix_shard_groups={len(suffix_groups)}")
    for (parent_dir, base_name), entries in sorted(suffix_groups.items(), key=lambda item: natural_key(item[0][0] / item[0][1])):
        shard_ids = [shard_id for shard_id, _ in sorted(entries)]
        print(f"  suffix: {parent_dir / base_name}.csv shards={shard_ids}")
    print(f"array_directory_shard_groups={len(array_dir_groups)}")
    for (parent_dir, rel_path), entries in sorted(array_dir_groups.items(), key=lambda item: natural_key(item[0][0] / item[0][1])):
        shard_ids = [shard_id for shard_id, _ in sorted(entries)]
        print(f"  array-dir: {parent_dir / 'merged' / rel_path} shards={shard_ids}")
    print(f"gather_csvs={len(gather_csvs)}")
    for csv_path, arcname in gather_csvs[:20]:
        print(f"  gather: {csv_path} -> {arcname}")
    if len(gather_csvs) > 20:
        print(f"  ... {len(gather_csvs) - 20} more")


def add_common_src_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("src_dirs", type=Path, nargs="+", help="Experiment output directories")


def add_merge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--force", action="store_true", help="Overwrite merged CSVs without .bak.csv backup")
    parser.add_argument("--allow-gaps", action="store_true", help="Warn instead of failing on missing shard IDs")
    parser.add_argument("--allow-broken", action="store_true", help="Warn instead of failing on unreadable shards")


def add_gather_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-tar", type=Path, required=True, help="Output .tar.gz path")
    parser.add_argument(
        "--include-array-shards",
        "--array-jobs",
        action="store_true",
        help="Include raw *_array_N.csv and array_N/** CSV shards in the tarball",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    scan_parser = subparsers.add_parser("scan", help="Print discovered shard and gather artifacts")
    scan_parser.add_argument("--include-array-shards", "--array-jobs", action="store_true")
    add_common_src_args(scan_parser)

    merge_parser = subparsers.add_parser("merge", help="Merge array shard CSVs")
    add_merge_args(merge_parser)
    add_common_src_args(merge_parser)

    gather_parser = subparsers.add_parser("gather", help="Gather CSVs into a tar.gz")
    add_gather_args(gather_parser)
    add_common_src_args(gather_parser)

    collect_parser = subparsers.add_parser("collect", help="Merge shard CSVs, then gather CSVs")
    add_merge_args(collect_parser)
    add_gather_args(collect_parser)
    add_common_src_args(collect_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src_dirs = [path.expanduser() for path in args.src_dirs]

    if args.action == "scan":
        scan_artifacts(src_dirs, include_array_shards=args.include_array_shards)
        return 0
    if args.action == "merge":
        return 0 if merge_artifacts(
            src_dirs,
            force=args.force,
            allow_gaps=args.allow_gaps,
            allow_broken=args.allow_broken,
        ) else 1
    if args.action == "gather":
        gather_artifacts(
            src_dirs,
            args.output_tar,
            include_array_shards=args.include_array_shards,
        )
        return 0
    if args.action == "collect":
        ok = merge_artifacts(
            src_dirs,
            force=args.force,
            allow_gaps=args.allow_gaps,
            allow_broken=args.allow_broken,
        )
        gather_artifacts(
            src_dirs,
            args.output_tar,
            include_array_shards=args.include_array_shards,
        )
        return 0 if ok else 1
    raise AssertionError(f"Unhandled action: {args.action}")


if __name__ == "__main__":
    sys.exit(main())
