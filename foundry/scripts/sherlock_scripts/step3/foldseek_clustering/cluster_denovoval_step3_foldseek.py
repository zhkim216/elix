#!/usr/bin/env python3
"""Run Foldseek clustering over denovoval step2 generated CIFs.

The script dynamically discovers condition directories under the step2
generated_samples root, stages selected ``.cif.gz`` files as symlinks, runs
``foldseek easy-cluster``, and writes source/cluster manifests under step3.
Empty condition directories are reported as diagnostics so the same command can
be rerun after those conditions are filled.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/"
    "denovoval/step2/generated_samples/full"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/"
    "denovoval/step3/foldseek_clustering"
)
DEFAULT_FOLDSEEK_BIN = Path("/oak/stanford/groups/possu/jinho/software/foldseek/bin/foldseek")
CONDITION_RE = re.compile(r"^(?P<ccd>.+)_len(?P<length>[0-9]+)$")


@dataclass(frozen=True)
class SampleRecord:
    staged_id: str
    source_path: Path
    json_path: Path | None
    condition: str
    ccd: str
    length: int
    source_size: int
    source_mtime: float


@dataclass(frozen=True)
class ConditionRecord:
    condition: str
    ccd: str
    length: int
    condition_dir: Path
    discovered_cif_count: int
    discovered_json_count: int
    selected_cif_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--foldseek-bin", type=Path, default=DEFAULT_FOLDSEEK_BIN)
    parser.add_argument("--cluster-c", type=float, default=0.8)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Pass --gpu 1 to Foldseek easy-cluster.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Optional condition names such as NOS_len150. Defaults to all conditions.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=None,
        help="Optional lengths to include.",
    )
    parser.add_argument(
        "--ccds",
        nargs="+",
        default=None,
        help="Optional CCD codes to include.",
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=None,
        help="Keep only the first N matching condition directories after sorting.",
    )
    parser.add_argument(
        "--max-structures-per-condition",
        type=int,
        default=None,
        help="Keep only the first N CIFs per matching condition after sorting.",
    )
    parser.add_argument(
        "--max-total-structures",
        type=int,
        default=None,
        help="Stop after staging this many total structures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output root before running.",
    )
    parser.add_argument(
        "--keep-existing-output",
        action="store_true",
        help="Allow writing into an existing output root without deleting it.",
    )
    parser.add_argument(
        "--allow-incomplete-clustering",
        action="store_true",
        help="Do not fail if selected staged structures are missing from the cluster TSV.",
    )
    parser.add_argument(
        "--skip-foldseek",
        action="store_true",
        help="Only discover inputs and write the staged symlink manifest.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def strip_cif_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".cif.gz"):
        return name[: -len(".cif.gz")]
    if name.endswith(".cif"):
        return name[: -len(".cif")]
    return path.stem


def parse_condition(condition: str) -> tuple[str, int] | None:
    match = CONDITION_RE.match(condition)
    if match is None:
        return None
    return match.group("ccd"), int(match.group("length"))


def condition_to_dict(row: ConditionRecord) -> dict[str, Any]:
    return {
        "condition": row.condition,
        "ccd": row.ccd,
        "length": row.length,
        "condition_dir": str(row.condition_dir),
        "discovered_cif_count": row.discovered_cif_count,
        "discovered_json_count": row.discovered_json_count,
        "selected_cif_count": row.selected_cif_count,
    }


def require_positive(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided")


def selected_condition_dirs(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"input root not found: {args.input_root}")

    requested_conditions = set(args.conditions or [])
    requested_lengths = set(args.lengths or [])
    requested_ccds = set(args.ccds or [])
    invalid_conditions: list[str] = []
    dirs: list[Path] = []
    for condition_dir in sorted(path for path in args.input_root.iterdir() if path.is_dir()):
        parsed = parse_condition(condition_dir.name)
        if parsed is None:
            invalid_conditions.append(condition_dir.name)
            continue
        ccd, length = parsed
        if requested_conditions and condition_dir.name not in requested_conditions:
            continue
        if requested_lengths and length not in requested_lengths:
            continue
        if requested_ccds and ccd not in requested_ccds:
            continue
        dirs.append(condition_dir)

    if requested_conditions:
        found = {path.name for path in dirs}
        missing = sorted(requested_conditions - found)
        if missing:
            raise FileNotFoundError(f"requested conditions not found or filtered out: {missing}")

    if args.max_conditions is not None:
        dirs = dirs[: args.max_conditions]
    return dirs, invalid_conditions


def discover_samples(args: argparse.Namespace) -> tuple[list[SampleRecord], list[ConditionRecord], dict[str, Any]]:
    require_positive("--max-conditions", args.max_conditions)
    require_positive("--max-structures-per-condition", args.max_structures_per_condition)
    require_positive("--max-total-structures", args.max_total_structures)

    condition_dirs, invalid_conditions = selected_condition_dirs(args)
    samples: list[SampleRecord] = []
    conditions: list[ConditionRecord] = []
    duplicate_stage_ids: list[dict[str, str]] = []
    missing_json: list[str] = []
    json_without_cif: list[str] = []
    stage_ids: set[str] = set()

    for condition_dir in condition_dirs:
        parsed = parse_condition(condition_dir.name)
        if parsed is None:
            continue
        ccd, length = parsed
        cif_paths = sorted(condition_dir.glob("*.cif.gz"))
        json_paths = sorted(condition_dir.glob("*.json"))
        json_by_stem = {path.stem: path for path in json_paths}
        cif_stems = {strip_cif_suffix(path) for path in cif_paths}
        for json_path in json_paths:
            if json_path.stem not in cif_stems:
                json_without_cif.append(str(json_path))

        selected_cifs = cif_paths
        if args.max_structures_per_condition is not None:
            selected_cifs = selected_cifs[: args.max_structures_per_condition]

        if args.max_total_structures is not None:
            remaining = args.max_total_structures - len(samples)
            if remaining <= 0:
                selected_cifs = []
            else:
                selected_cifs = selected_cifs[:remaining]

        for cif_path in selected_cifs:
            stem = strip_cif_suffix(cif_path)
            staged_id = stem
            if staged_id in stage_ids:
                staged_id = f"{condition_dir.name}__{stem}"
            if staged_id in stage_ids:
                duplicate_stage_ids.append({"condition": condition_dir.name, "source": str(cif_path)})
                continue
            stage_ids.add(staged_id)

            json_path = json_by_stem.get(stem)
            if json_path is None:
                missing_json.append(str(cif_path))
            stat = cif_path.stat()
            samples.append(
                SampleRecord(
                    staged_id=staged_id,
                    source_path=cif_path,
                    json_path=json_path,
                    condition=condition_dir.name,
                    ccd=ccd,
                    length=length,
                    source_size=stat.st_size,
                    source_mtime=stat.st_mtime,
                )
            )

        conditions.append(
            ConditionRecord(
                condition=condition_dir.name,
                ccd=ccd,
                length=length,
                condition_dir=condition_dir,
                discovered_cif_count=len(cif_paths),
                discovered_json_count=len(json_paths),
                selected_cif_count=len(selected_cifs),
            )
        )

        if args.max_total_structures is not None and len(samples) >= args.max_total_structures:
            break

    diagnostics = {
        "invalid_condition_dirs": invalid_conditions,
        "empty_conditions": [
            condition_to_dict(row) for row in conditions if row.discovered_cif_count == 0
        ],
        "missing_json_for_selected_cif": missing_json,
        "json_without_cif": json_without_cif[:100],
        "json_without_cif_count": len(json_without_cif),
        "duplicate_stage_ids": duplicate_stage_ids,
    }
    return samples, conditions, diagnostics


def prepare_output_root(output_root: Path, *, overwrite: bool, keep_existing: bool) -> None:
    if overwrite and keep_existing:
        raise ValueError("--overwrite and --keep-existing-output are mutually exclusive")
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        elif not keep_existing:
            raise FileExistsError(
                f"output root already exists: {output_root}; pass --overwrite or --keep-existing-output"
            )
    output_root.mkdir(parents=True, exist_ok=True)


def write_tsv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join("" if item is None else str(item) for item in row) + "\n")


def stage_inputs(samples: list[SampleRecord], input_dir: Path) -> None:
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    for sample in samples:
        target = input_dir / f"{sample.staged_id}.cif.gz"
        target.symlink_to(sample.source_path)


def write_input_reports(
    output_root: Path,
    samples: list[SampleRecord],
    conditions: list[ConditionRecord],
) -> None:
    write_tsv(
        output_root / "input_manifest.tsv",
        [
            "staged_id",
            "condition",
            "ccd",
            "length",
            "source_path",
            "json_path",
            "source_size",
            "source_mtime",
        ],
        [
            [
                sample.staged_id,
                sample.condition,
                sample.ccd,
                sample.length,
                sample.source_path,
                sample.json_path,
                sample.source_size,
                sample.source_mtime,
            ]
            for sample in samples
        ],
    )
    write_tsv(
        output_root / "condition_manifest.tsv",
        [
            "condition",
            "ccd",
            "length",
            "condition_dir",
            "discovered_cif_count",
            "discovered_json_count",
            "selected_cif_count",
        ],
        [
            [
                row.condition,
                row.ccd,
                row.length,
                row.condition_dir,
                row.discovered_cif_count,
                row.discovered_json_count,
                row.selected_cif_count,
            ]
            for row in conditions
        ],
    )


def run_command(cmd: list[str], log_path: Path) -> None:
    print("[cmd]", " ".join(cmd), flush=True)
    with log_path.open("w") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def foldseek_version(foldseek_bin: Path) -> str:
    try:
        result = subprocess.run(
            [str(foldseek_bin), "version"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return f"unavailable: {exc}"
    return result.stdout.strip()


def run_foldseek(args: argparse.Namespace, output_root: Path) -> Path:
    foldseek_dir = output_root / "foldseek"
    foldseek_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_root / "work" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prefix = foldseek_dir / "clu"
    cmd = [
        str(args.foldseek_bin),
        "easy-cluster",
        str(output_root / "work" / "input"),
        str(prefix),
        str(tmp_dir),
        "-c",
        str(args.cluster_c),
        "--threads",
        str(args.threads),
    ]
    if args.gpu:
        cmd.extend(["--gpu", "1"])
    run_command(cmd, output_root / "foldseek_easy_cluster.log")
    cluster_tsv = foldseek_dir / "clu_cluster.tsv"
    if not cluster_tsv.is_file():
        raise FileNotFoundError(f"Foldseek did not write expected cluster TSV: {cluster_tsv}")
    return cluster_tsv


def resolve_foldseek_id(raw_id: str, by_staged_id: dict[str, SampleRecord]) -> tuple[str | None, SampleRecord | None]:
    candidates = [raw_id]
    for ext in (".cif.gz", ".cif", ".gz"):
        if raw_id.endswith(ext):
            candidates.append(raw_id[: -len(ext)])
            break
    for candidate in list(candidates):
        if "_" in candidate:
            candidates.append(candidate.rsplit("_", 1)[0])
    for candidate in candidates:
        sample = by_staged_id.get(candidate)
        if sample is not None:
            return candidate, sample
    return None, None


def parse_clusters(
    cluster_tsv: Path,
    samples: list[SampleRecord],
    output_root: Path,
) -> dict[str, Any]:
    by_staged_id = {sample.staged_id: sample for sample in samples}
    clusters: "OrderedDict[str, list[str]]" = OrderedDict()
    cluster_rows: list[list[Any]] = []
    unmapped_rows: list[dict[str, str]] = []
    member_assignments: Counter[str] = Counter()

    with cluster_tsv.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.rstrip("\n")
            if not raw:
                continue
            parts = raw.split("\t")
            if len(parts) < 2:
                unmapped_rows.append({"line": str(line_no), "raw": raw, "reason": "expected_two_columns"})
                continue
            raw_rep, raw_member = parts[0], parts[1]
            rep_id, rep = resolve_foldseek_id(raw_rep, by_staged_id)
            member_id, member = resolve_foldseek_id(raw_member, by_staged_id)
            if rep_id is None or rep is None or member_id is None or member is None:
                unmapped_rows.append(
                    {
                        "line": str(line_no),
                        "raw_representative": raw_rep,
                        "raw_member": raw_member,
                        "reason": "id_not_found_in_staged_manifest",
                    }
                )
                continue
            clusters.setdefault(rep_id, [])
            clusters[rep_id].append(member_id)
            member_assignments[member_id] += 1
            cluster_rows.append(
                [
                    rep_id,
                    member_id,
                    raw_rep,
                    raw_member,
                    rep.condition,
                    member.condition,
                    rep.ccd,
                    member.ccd,
                    rep.length,
                    member.length,
                    rep.source_path,
                    member.source_path,
                ]
            )

    write_tsv(
        output_root / "foldseek_clusters.tsv",
        [
            "representative_id",
            "member_id",
            "raw_representative_id",
            "raw_member_id",
            "representative_condition",
            "member_condition",
            "representative_ccd",
            "member_ccd",
            "representative_length",
            "member_length",
            "representative_source_path",
            "member_source_path",
        ],
        cluster_rows,
    )

    summary_rows: list[list[Any]] = []
    for rank, (rep_id, members) in enumerate(clusters.items(), start=1):
        rep = by_staged_id[rep_id]
        summary_rows.append(
            [
                rank,
                rep_id,
                len(members),
                rep.condition,
                rep.ccd,
                rep.length,
                rep.source_path,
            ]
        )
    write_tsv(
        output_root / "cluster_summary.tsv",
        [
            "cluster_rank",
            "representative_id",
            "cluster_size",
            "representative_condition",
            "representative_ccd",
            "representative_length",
            "representative_source_path",
        ],
        summary_rows,
    )

    duplicate_members = sorted(member_id for member_id, count in member_assignments.items() if count > 1)
    clustered_members = set(member_assignments)
    expected_members = set(by_staged_id)
    missing_members = sorted(expected_members - clustered_members)
    return {
        "raw_cluster_rows": sum(len(members) for members in clusters.values()),
        "cluster_count": len(clusters),
        "clustered_member_count": len(clustered_members),
        "unmapped_cluster_rows": unmapped_rows,
        "unmapped_cluster_row_count": len(unmapped_rows),
        "missing_cluster_member_ids": missing_members,
        "missing_cluster_member_count": len(missing_members),
        "duplicate_cluster_member_ids": duplicate_members,
        "duplicate_cluster_member_count": len(duplicate_members),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def make_manifest(
    *,
    args: argparse.Namespace,
    samples: list[SampleRecord],
    conditions: list[ConditionRecord],
    diagnostics: dict[str, Any],
    cluster_report: dict[str, Any] | None,
    status: str,
) -> dict[str, Any]:
    return {
        "created_at": utc_now(),
        "status": status,
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "foldseek_bin": str(args.foldseek_bin),
        "foldseek_version": foldseek_version(args.foldseek_bin),
        "foldseek_cluster_c": args.cluster_c,
        "foldseek_gpu": args.gpu,
        "threads": args.threads,
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "filters": {
            "conditions": args.conditions,
            "lengths": args.lengths,
            "ccds": args.ccds,
            "max_conditions": args.max_conditions,
            "max_structures_per_condition": args.max_structures_per_condition,
            "max_total_structures": args.max_total_structures,
        },
        "counts": {
            "condition_dirs": len(conditions),
            "selected_structures": len(samples),
            "discovered_cifs_in_selected_conditions": sum(row.discovered_cif_count for row in conditions),
            "empty_conditions": len([row for row in conditions if row.discovered_cif_count == 0]),
        },
        "diagnostics": diagnostics,
        "cluster_report": cluster_report,
    }


def main() -> int:
    args = parse_args()
    if not args.foldseek_bin.is_file():
        raise FileNotFoundError(f"foldseek binary not found: {args.foldseek_bin}")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    samples, conditions, diagnostics = discover_samples(args)
    if not samples:
        raise RuntimeError("no CIF files selected for clustering")

    prepare_output_root(
        args.output_root,
        overwrite=args.overwrite,
        keep_existing=args.keep_existing_output,
    )
    write_input_reports(args.output_root, samples, conditions)
    stage_inputs(samples, args.output_root / "work" / "input")

    cluster_report: dict[str, Any] | None = None
    status = "staged_only" if args.skip_foldseek else "complete"
    if not args.skip_foldseek:
        cluster_tsv = run_foldseek(args, args.output_root)
        cluster_report = parse_clusters(cluster_tsv, samples, args.output_root)
        bad_cluster = (
            cluster_report["unmapped_cluster_row_count"] > 0
            or cluster_report["missing_cluster_member_count"] > 0
            or cluster_report["duplicate_cluster_member_count"] > 0
        )
        if bad_cluster and not args.allow_incomplete_clustering:
            status = "failed_validation"
        elif bad_cluster:
            status = "complete_with_cluster_diagnostics"

    manifest = make_manifest(
        args=args,
        samples=samples,
        conditions=conditions,
        diagnostics=diagnostics,
        cluster_report=cluster_report,
        status=status,
    )
    write_json(args.output_root / "manifest.json", manifest)
    write_json(
        args.output_root / "validation_report.json",
        {
            "status": status,
            "diagnostics": diagnostics,
            "cluster_report": cluster_report,
        },
    )

    print(
        json.dumps(
            {
                "status": status,
                "output_root": str(args.output_root),
                "conditions": len(conditions),
                "selected_structures": len(samples),
                "cluster_report": cluster_report,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if status == "failed_validation" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
