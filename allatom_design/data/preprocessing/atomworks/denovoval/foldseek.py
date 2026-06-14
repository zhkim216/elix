"""Foldseek staging and cluster parsing for denovoval source records."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from allatom_design.data.preprocessing.atomworks.denovoval.records import (
    SourceRecord,
    strip_cif_gz,
)


@dataclass(frozen=True)
class FoldseekRunResult:
    command: list[str]
    cluster_tsv: Path
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str


def clean_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Directory exists; set overwrite=true to replace: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def clean_foldseek_outputs(cluster_prefix: Path, tmp_dir: Path, *, overwrite: bool) -> None:
    cluster_prefix.parent.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Foldseek tmp dir exists; set overwrite=true to replace: {tmp_dir}")
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for path in cluster_prefix.parent.glob(f"{cluster_prefix.name}*"):
        if path == tmp_dir or tmp_dir in path.parents:
            continue
        if not overwrite:
            raise FileExistsError(f"Foldseek output exists; set overwrite=true to replace: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def staged_filename(record: SourceRecord) -> str:
    return f"{record.kind}__{record.sample_id}.cif.gz"


def stage_foldseek_inputs(
    records: list[SourceRecord],
    input_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, str]:
    """Symlink source CIFs into a clean Foldseek input dir.

    Returns a lookup from staged Foldseek base IDs to source ``sample_id``.
    """
    clean_directory(input_dir, overwrite=overwrite)
    id_lookup: dict[str, str] = {}
    for record in records:
        filename = staged_filename(record)
        staged_path = input_dir / filename
        if staged_path.exists() or staged_path.is_symlink():
            staged_path.unlink()
        staged_path.symlink_to(record.source_path)

        staged_stem = strip_cif_gz(staged_path)
        if staged_stem in id_lookup:
            raise RuntimeError(f"Duplicate staged Foldseek ID: {staged_stem}")
        id_lookup[staged_stem] = record.sample_id
        id_lookup[filename] = record.sample_id
    return id_lookup


def build_foldseek_easy_cluster_command(
    *,
    foldseek_bin: str,
    input_dir: Path,
    cluster_prefix: Path,
    tmp_dir: Path,
    threads: int,
    coverage: float,
    use_gpu: bool,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        foldseek_bin,
        "easy-cluster",
        str(input_dir),
        str(cluster_prefix),
        str(tmp_dir),
        "-c",
        str(coverage),
        "--threads",
        str(threads),
        "--gpu",
        "1" if use_gpu else "0",
    ]
    if extra_args:
        command.extend(str(arg) for arg in extra_args)
    return command


def run_foldseek_easy_cluster(
    *,
    foldseek_bin: str,
    input_dir: Path,
    cluster_prefix: Path,
    tmp_dir: Path,
    threads: int,
    coverage: float,
    use_gpu: bool,
    overwrite: bool,
    extra_args: list[str] | None = None,
) -> FoldseekRunResult:
    clean_foldseek_outputs(cluster_prefix, tmp_dir, overwrite=overwrite)
    command = build_foldseek_easy_cluster_command(
        foldseek_bin=foldseek_bin,
        input_dir=input_dir,
        cluster_prefix=cluster_prefix,
        tmp_dir=tmp_dir,
        threads=threads,
        coverage=coverage,
        use_gpu=use_gpu,
        extra_args=extra_args,
    )
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    cluster_tsv = cluster_prefix.with_name(f"{cluster_prefix.name}_cluster.tsv")
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Foldseek did not produce cluster TSV: {cluster_tsv}")
    return FoldseekRunResult(
        command=command,
        cluster_tsv=cluster_tsv,
        elapsed_seconds=round(time.perf_counter() - start, 3),
        stdout_tail=completed.stdout[-4000:],
        stderr_tail=completed.stderr[-4000:],
    )


def parse_cluster_tsv(cluster_tsv: Path) -> tuple[dict[str, str], dict[str, int], list[dict[str, str]]]:
    """Parse ``*_cluster.tsv`` into member->representative and cluster sizes."""
    representative_by_member: dict[str, str] = {}
    cluster_sizes: dict[str, int] = {}
    conflicts: list[dict[str, str]] = []
    with cluster_tsv.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                raise ValueError(
                    f"Malformed Foldseek cluster row {line_number} in {cluster_tsv}: "
                    "expected at least two tab-separated fields."
                )
            representative, member = fields[0], fields[1]
            previous = representative_by_member.get(member)
            if previous is not None and previous != representative:
                conflicts.append(
                    {
                        "member": member,
                        "first_representative": previous,
                        "second_representative": representative,
                    }
                )
                representative = min(previous, representative)
            representative_by_member[member] = representative
            cluster_sizes[representative] = cluster_sizes.get(representative, 0) + 1
    return representative_by_member, cluster_sizes, conflicts


def resolve_foldseek_id(foldseek_id: str, id_lookup: dict[str, str]) -> str | None:
    """Resolve a Foldseek TSV ID back to the source ``sample_id`` when possible."""
    candidates = [str(foldseek_id).strip()]
    raw = candidates[0]
    for suffix in (".cif.gz", ".cif", ".gz"):
        if raw.endswith(suffix):
            candidates.append(raw[: -len(suffix)])
            break

    base = candidates[-1]
    while "_" in base:
        base = base.rsplit("_", 1)[0]
        candidates.append(base)

    for candidate in candidates:
        if candidate in id_lookup:
            return id_lookup[candidate]
    return None


def assign_foldseek_clusters(
    *,
    records: list[SourceRecord],
    representative_by_member: dict[str, str],
    cluster_sizes: dict[str, int],
    id_lookup: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    """Build per-sample Foldseek metadata for source records."""
    foldseek_by_sample: dict[str, dict[str, Any]] = {}
    unresolved: list[dict[str, str]] = []
    for member_id, representative_id in representative_by_member.items():
        sample_id = resolve_foldseek_id(member_id, id_lookup)
        representative_sample_id = resolve_foldseek_id(representative_id, id_lookup)
        if sample_id is None:
            unresolved.append({"role": "member", "foldseek_id": member_id})
            continue
        if representative_sample_id is None:
            unresolved.append({"role": "representative", "foldseek_id": representative_id})
            representative_sample_id = representative_id

        foldseek_by_sample[sample_id] = {
            "foldseek_member_id": member_id,
            "foldseek_representative_id": representative_id,
            "foldseek_cluster": representative_sample_id,
            "foldseek_cluster_size": cluster_sizes.get(representative_id, 0),
        }

    for record in records:
        if record.sample_id not in foldseek_by_sample:
            unresolved.append({"role": "record", "foldseek_id": record.sample_id})
    return foldseek_by_sample, unresolved
