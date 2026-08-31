#!/usr/bin/env python3
"""Reshard selected denovoval CCD task lists into bounded one-hour chunks."""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/ensembles"
)
EXPECTED_CONDITIONS = ("partialt2", "partialt5")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_original_task_ids(values: List[str]) -> List[int]:
    task_ids: List[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                task_ids.append(int(token))
    if not task_ids:
        raise ValueError("At least one original task ID is required")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Duplicate original task IDs")
    return sorted(task_ids)


def condition_from_input(path: Path) -> str:
    if path.parent.name != "inputs":
        raise ValueError(f"Unexpected input path layout: {path}")
    return path.parent.parent.name


def read_source_pairs(task_path: Path, *, ccd_code: str) -> List[Tuple[Path, Path]]:
    paths = [Path(line) for line in task_path.read_text().splitlines() if line]
    if not paths or len(paths) % 2:
        raise ValueError(f"Expected nonempty t2/t5 pairs in {task_path}")
    pairs: List[Tuple[Path, Path]] = []
    for offset in range(0, len(paths), 2):
        pair = paths[offset : offset + 2]
        if tuple(condition_from_input(path) for path in pair) != EXPECTED_CONDITIONS:
            raise ValueError(f"Expected ordered t2/t5 pair in {task_path}: {pair}")
        if pair[0].stem != pair[1].stem:
            raise ValueError(f"Mismatched sample IDs in {task_path}: {pair}")
        if not pair[0].stem.startswith(f"{ccd_code}_"):
            raise ValueError(f"Sample does not match CCD {ccd_code}: {pair[0].stem}")
        missing = [str(path) for path in pair if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing prepared input(s): {missing}")
        pairs.append((pair[0].resolve(), pair[1].resolve()))
    return pairs


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = args.output_root.resolve()
    source_array_index = (args.source_array_index or output_root / "array_index.txt").resolve()
    prepared_manifest = output_root / "prepared_inputs_manifest.csv"
    for required in (source_array_index, prepared_manifest):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_task_paths = [
        Path(line).resolve() for line in source_array_index.read_text().splitlines() if line
    ]
    original_task_ids = parse_original_task_ids(args.original_task_ids)
    invalid_ids = [task_id for task_id in original_task_ids if task_id >= len(source_task_paths)]
    if invalid_ids or min(original_task_ids) < 0:
        raise ValueError(f"Original task IDs outside array index: {invalid_ids}")
    if args.source_cifs_per_task <= 0:
        raise ValueError("--source-cifs-per-task must be positive")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temporary_dir = output_dir.with_name(f".{output_dir.name}.tmp.{os.getpid()}")
    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)

    chunks: List[Dict[str, Any]] = []
    all_inputs: List[str] = []
    all_sample_ids: List[str] = []
    try:
        tasks_dir = temporary_dir / "tasks"
        tasks_dir.mkdir(parents=True)
        array_lines: List[str] = []
        for original_task_id in original_task_ids:
            source_task_path = source_task_paths[original_task_id]
            if not source_task_path.is_file():
                raise FileNotFoundError(source_task_path)
            ccd_code = source_task_path.stem
            pairs = read_source_pairs(source_task_path, ccd_code=ccd_code)
            for chunk_index, start in enumerate(range(0, len(pairs), args.source_cifs_per_task)):
                chunk_pairs = pairs[start : start + args.source_cifs_per_task]
                chunk_inputs = [str(path) for pair in chunk_pairs for path in pair]
                sample_ids = [pair[0].stem for pair in chunk_pairs]
                task_name = (
                    f"{ccd_code}_orig{original_task_id:03d}_chunk{chunk_index:02d}.txt"
                )
                task_path = tasks_dir / task_name
                write_text(task_path, "\n".join(chunk_inputs) + "\n")
                published_task_path = output_dir / "tasks" / task_name
                array_lines.append(str(published_task_path))
                all_inputs.extend(chunk_inputs)
                all_sample_ids.extend(sample_ids)
                chunks.append(
                    {
                        "new_array_task_id": len(chunks),
                        "original_array_task_id": original_task_id,
                        "ccd_code": ccd_code,
                        "chunk_index": chunk_index,
                        "task_file": str(published_task_path),
                        "n_source_cifs": len(chunk_pairs),
                        "n_input_conditions": len(chunk_inputs),
                        "sample_ids": sample_ids,
                    }
                )

        if len(all_inputs) != len(set(all_inputs)):
            raise ValueError("Resharded input conditions are duplicated")
        if len(all_sample_ids) != len(set(all_sample_ids)):
            raise ValueError("Resharded source CIFs are duplicated")
        if any(chunk["n_input_conditions"] > args.source_cifs_per_task * 2 for chunk in chunks):
            raise ValueError("A generated task exceeds the requested chunk size")

        array_index = output_dir / "array_index.txt"
        manifest_path = output_dir / "reshard_manifest.json"
        write_text(temporary_dir / "array_index.txt", "\n".join(array_lines) + "\n")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "source_array_index": str(source_array_index),
            "source_array_index_sha256": sha256_file(source_array_index),
            "prepared_inputs_manifest": str(prepared_manifest),
            "prepared_inputs_manifest_sha256": sha256_file(prepared_manifest),
            "original_array_task_ids": original_task_ids,
            "source_cifs_per_task": args.source_cifs_per_task,
            "input_conditions_per_full_task": args.source_cifs_per_task * 2,
            "n_original_array_tasks": len(original_task_ids),
            "n_source_cifs": len(all_sample_ids),
            "n_input_conditions": len(all_inputs),
            "n_new_array_tasks": len(chunks),
            "array_index": str(array_index),
            "diagnostics": {
                "duplicate_input_conditions": 0,
                "duplicate_source_cifs": 0,
                "missing_prepared_inputs": 0,
                "malformed_condition_pairs": 0,
                "mixed_ccd_tasks": 0,
            },
            "chunks": chunks,
        }
        write_text(
            temporary_dir / "reshard_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-array-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-task-ids", nargs="+", required=True)
    parser.add_argument("--source-cifs-per-task", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
