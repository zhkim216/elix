"""Index released ABACUS-T chunk outputs into one validated mode manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    protein_sequence,
    read_structure,
    sha256_file,
)


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/abacus-t/default.yaml"
)
MODES = ("smoke", "per_ccd", "full")
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
PREPARATION_REQUIRED = {
    "sample_id",
    "mode",
    "chunk_index",
    "manifest_row_index",
    "abacust_ligand_basename",
    "protein_length",
    "source_staging_manifest_sha256",
    "run_fingerprint",
    "protein_sequence_sha256",
    "status",
}
STAGING_REQUIRED = {"sample_id", "status"}
MANIFEST_COLUMNS = (
    "source_sample_id",
    "designed_sample_id",
    "model_name",
    "design_index",
    "protein_sequence",
    "sampled_pdb_path",
    "sampled_pdb_sha256",
    "fasta_path",
    "fasta_sha256",
    "iteration",
    "source_staging_manifest_sha256",
    "source_preparation_manifest_sha256",
    "run_fingerprint",
    "status",
)


class Diagnostics:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, str]] = []

    def add(self, category: str, key: str, error: object) -> None:
        self.counts[category] += 1
        if len(self.examples) < 50:
            self.examples.append(
                {"category": category, "key": str(key), "error": str(error)}
            )

    @property
    def failed(self) -> bool:
        return bool(sum(self.counts.values()))

    def payload(self) -> dict[str, int]:
        return {
            "total_errors": int(sum(self.counts.values())),
            **{key: int(value) for key, value in sorted(self.counts.items())},
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=MODES, required=True)
    return parser.parse_args(argv)


def cfg_path(cfg: DictConfig, key: str) -> Path:
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ValueError(f"Missing required config path: {key}")
    return Path(str(value)).expanduser().resolve()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts)))
                header = line[1:].strip()
                sequence_parts = []
                if not header:
                    raise ValueError(f"Empty FASTA header at {path}:{line_number}")
            else:
                if header is None:
                    raise ValueError(
                        f"FASTA sequence precedes its header at {path}:{line_number}"
                    )
                sequence_parts.append(line.upper())
    if header is not None:
        records.append((header, "".join(sequence_parts)))
    if not records:
        raise ValueError(f"No FASTA records found: {path}")
    return records


def parse_design_fasta(
    path: Path,
    *,
    sample_id: str,
    design_indices: Iterable[int],
    iteration: int,
    protein_length: int,
) -> dict[str, str]:
    by_id: dict[str, str] = {}
    for header, sequence in read_fasta(path):
        record_id = header.split(";", 1)[0].strip()
        if record_id in by_id:
            raise ValueError(f"Duplicate FASTA record ID {record_id!r}: {path}")
        if len(sequence) != protein_length:
            raise ValueError(
                f"FASTA length differs for {record_id}: "
                f"{len(sequence)} != {protein_length}"
            )
        invalid = sorted(set(sequence) - CANONICAL_AA)
        if invalid:
            raise ValueError(f"Noncanonical FASTA residues for {record_id}: {invalid}")
        by_id[record_id] = sequence

    expected_ids = {sample_id} | {
        f"{sample_id}_design_{int(index)}_{iteration}" for index in design_indices
    }
    observed_ids = set(by_id)
    if observed_ids != expected_ids:
        raise ValueError(
            "FASTA design matrix differs: "
            f"missing={sorted(expected_ids - observed_ids)}, "
            f"extra={sorted(observed_ids - expected_ids)}"
        )
    return by_id


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def _staging_manifest_path(cfg: DictConfig, mode: str) -> Path:
    key = "staging.full_manifest_csv" if mode == "full" else f"staging.{mode}_manifest_csv"
    return cfg_path(cfg, key)


def _output_leaf_name(cfg: DictConfig) -> str:
    defaults = cfg.sampling.defaults
    return (
        f"T_{defaults.temperature}_R_{defaults.iter_num}_"
        f"{defaults.suffix}_{defaults.augment_eps}"
    )


def _check_exact_names(
    *,
    expected: set[str],
    observed: set[str],
    category: str,
    key: str,
    diagnostics: Diagnostics,
) -> None:
    for name in sorted(expected - observed):
        diagnostics.add(f"missing_{category}", key, name)
    for name in sorted(observed - expected):
        diagnostics.add(f"surplus_{category}", key, name)


def _validate_preparation(
    cfg: DictConfig,
    mode: str,
    diagnostics: Diagnostics,
) -> tuple[pd.DataFrame, Path, str, dict[str, Any]]:
    manifest_path = cfg_path(cfg, f"preparation.{mode}.manifest_csv")
    status_path = cfg_path(cfg, f"preparation.{mode}.status_json")
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    status = _load_json(status_path)
    _require_columns(frame, PREPARATION_REQUIRED, manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    mode_cfg = cfg.sampling[mode]
    expected_count = int(mode_cfg.input_count)
    chunk_count = int(mode_cfg.chunk_count)
    inputs_per_chunk = int(mode_cfg.inputs_per_chunk)

    if status.get("state") != "complete" or status.get("mode") != mode:
        diagnostics.add("preparation_status", mode, status)
    if status.get("preparation_manifest_sha256") != manifest_sha256:
        diagnostics.add("stale_preparation_manifest", mode, manifest_sha256)
    if int(status.get("sample_count", -1)) != expected_count:
        diagnostics.add("preparation_status_count", mode, status.get("sample_count"))
    if len(frame) != expected_count:
        diagnostics.add("preparation_row_count", mode, len(frame))
    if frame["sample_id"].duplicated().any():
        diagnostics.add("duplicate_preparation_sample", mode, "duplicate sample_id")
    if not frame["status"].eq("complete").all():
        diagnostics.add("incomplete_preparation_row", mode, "status != complete")
    if not frame["mode"].eq(mode).all():
        diagnostics.add("preparation_mode", mode, "row mode mismatch")
    observed_chunks = Counter(pd.to_numeric(frame["chunk_index"], errors="coerce"))
    expected_chunks = {index: inputs_per_chunk for index in range(chunk_count)}
    if dict(observed_chunks) != expected_chunks:
        diagnostics.add(
            "preparation_chunk_matrix", mode, f"{dict(observed_chunks)} != {expected_chunks}"
        )
    return frame, manifest_path, manifest_sha256, status


def _validate_staging(
    cfg: DictConfig,
    mode: str,
    preparation: pd.DataFrame,
    preparation_status: dict[str, Any],
    diagnostics: Diagnostics,
) -> tuple[Path, str]:
    path = _staging_manifest_path(cfg, mode)
    frame = pd.read_csv(path, keep_default_na=False)
    _require_columns(frame, STAGING_REQUIRED, path)
    digest = sha256_file(path)
    if frame["sample_id"].duplicated().any():
        diagnostics.add("duplicate_staging_sample", mode, "duplicate sample_id")
    if not frame["status"].eq("complete").all():
        diagnostics.add("incomplete_staging_row", mode, "status != complete")
    _check_exact_names(
        expected=set(map(str, preparation["sample_id"])),
        observed=set(map(str, frame["sample_id"])),
        category="staging_sample",
        key=mode,
        diagnostics=diagnostics,
    )
    if preparation_status.get("source_staging_manifest_sha256") != digest:
        diagnostics.add("stale_staging_status", mode, digest)
    if not preparation["source_staging_manifest_sha256"].eq(digest).all():
        diagnostics.add("stale_staging_preparation_row", mode, digest)
    return path, digest


def _manifest_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _artifact_fingerprint(
    cfg: DictConfig,
    *,
    mode: str,
    preparation_manifest_sha256: str,
    staging_manifest_sha256: str,
    rows: list[dict[str, Any]],
) -> str:
    defaults = cfg.sampling.defaults
    payload = {
        "mode": mode,
        "preparation_manifest_sha256": preparation_manifest_sha256,
        "staging_manifest_sha256": staging_manifest_sha256,
        "inference": OmegaConf.to_container(defaults, resolve=True),
        "artifacts": [
            {
                "designed_sample_id": row["designed_sample_id"],
                "sampled_pdb_sha256": row["sampled_pdb_sha256"],
                "fasta_sha256": row["fasta_sha256"],
            }
            for row in rows
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _index_outputs(
    cfg: DictConfig,
    *,
    mode: str,
    preparation: pd.DataFrame,
    preparation_manifest_sha256: str,
    staging_manifest_sha256: str,
    diagnostics: Diagnostics,
) -> list[dict[str, Any]]:
    mode_cfg = cfg.sampling[mode]
    chunks_root = cfg_path(cfg, f"sampling.{mode}.chunks_root")
    chunk_count = int(mode_cfg.chunk_count)
    design_indices = [int(value) for value in cfg.sequence_design.design_indices]
    iteration = int(cfg.sampling.defaults.final_iteration)
    protein_chain = str(cfg.sampling.defaults.protein_chain)
    leaf_name = _output_leaf_name(cfg)
    expected_chunk_names = {f"chunk_{index:04d}" for index in range(chunk_count)}
    observed_chunk_names = (
        {path.name for path in chunks_root.iterdir() if path.is_dir()}
        if chunks_root.is_dir()
        else set()
    )
    _check_exact_names(
        expected=expected_chunk_names,
        observed=observed_chunk_names,
        category="sampling_chunk",
        key=mode,
        diagnostics=diagnostics,
    )

    rows: list[dict[str, Any]] = []
    ordered = preparation.sort_values("manifest_row_index", kind="stable")
    for chunk_index in range(chunk_count):
        chunk_name = f"chunk_{chunk_index:04d}"
        chunk_root = chunks_root / chunk_name
        leaf = chunk_root / leaf_name
        if not leaf.is_dir():
            diagnostics.add("missing_sampling_leaf", chunk_name, leaf)
            continue
        observed_leaves = {path.name for path in chunk_root.iterdir() if path.is_dir()}
        _check_exact_names(
            expected={leaf_name},
            observed=observed_leaves,
            category="sampling_leaf",
            key=chunk_name,
            diagnostics=diagnostics,
        )
        chunk_rows = ordered[
            pd.to_numeric(ordered["chunk_index"], errors="coerce").eq(chunk_index)
        ]
        expected_samples = set(map(str, chunk_rows["sample_id"]))
        observed_samples = {path.name for path in leaf.iterdir() if path.is_dir()}
        _check_exact_names(
            expected=expected_samples,
            observed=observed_samples,
            category="sampling_sample",
            key=chunk_name,
            diagnostics=diagnostics,
        )
        for prepared in chunk_rows.itertuples(index=False):
            sample_id = str(prepared.sample_id)
            sample_dir = leaf / sample_id
            if not sample_dir.is_dir():
                continue
            fasta_path = sample_dir / f"{sample_id}_design.fa"
            ligand_name = f"{prepared.abacust_ligand_basename}.sdf"
            expected_files = {
                f"{sample_id}.pdb",
                f"{sample_id}_design.fa",
                ligand_name,
                *{
                    f"{sample_id}_design_{index}_{iteration}.pdb"
                    for index in design_indices
                },
            }
            observed_files = {
                path.name for path in sample_dir.iterdir() if path.is_file()
            }
            _check_exact_names(
                expected=expected_files,
                observed=observed_files,
                category="sampling_file",
                key=sample_id,
                diagnostics=diagnostics,
            )
            if not expected_files.issubset(observed_files):
                continue
            try:
                fasta = parse_design_fasta(
                    fasta_path,
                    sample_id=sample_id,
                    design_indices=design_indices,
                    iteration=iteration,
                    protein_length=int(prepared.protein_length),
                )
                if sha256_text(fasta[sample_id]) != str(
                    prepared.protein_sequence_sha256
                ):
                    raise ValueError("native FASTA sequence differs from preparation")
                fasta_sha256 = sha256_file(fasta_path)
                for design_index in design_indices:
                    designed_id = f"{sample_id}_design_{design_index}_{iteration}"
                    sampled_pdb = sample_dir / f"{designed_id}.pdb"
                    structure = read_structure(sampled_pdb)
                    chains = set(map(str, structure.chain_id))
                    if chains != {protein_chain}:
                        raise ValueError(
                            f"sampled PDB chains differ for {designed_id}: {chains}"
                        )
                    if np.any(structure.hetero):
                        raise ValueError(f"sampled PDB contains HETATM: {designed_id}")
                    observed_sequence = protein_sequence(structure, protein_chain)
                    if observed_sequence != fasta[designed_id]:
                        raise ValueError(
                            f"FASTA/PDB sequence mismatch for {designed_id}"
                        )
                    rows.append(
                        {
                            "source_sample_id": sample_id,
                            "designed_sample_id": designed_id,
                            "model_name": "abacus-t",
                            "design_index": design_index,
                            "protein_sequence": observed_sequence,
                            "sampled_pdb_path": str(sampled_pdb.resolve()),
                            "sampled_pdb_sha256": sha256_file(sampled_pdb),
                            "fasta_path": str(fasta_path.resolve()),
                            "fasta_sha256": fasta_sha256,
                            "iteration": iteration,
                            "source_staging_manifest_sha256": staging_manifest_sha256,
                            "source_preparation_manifest_sha256": (
                                preparation_manifest_sha256
                            ),
                            "run_fingerprint": "",
                            "status": "complete",
                        }
                    )
            except Exception as exc:
                diagnostics.add("invalid_sampling_sample", sample_id, exc)

    expected_designs = len(preparation) * len(design_indices)
    if len(rows) != expected_designs:
        diagnostics.add("sampling_design_count", mode, f"{len(rows)} != {expected_designs}")
    designed_ids = [str(row["designed_sample_id"]) for row in rows]
    duplicates = sorted(
        designed_id
        for designed_id, count in Counter(designed_ids).items()
        if count != 1
    )
    if duplicates:
        diagnostics.add("duplicate_designed_sample", mode, duplicates[:20])
    if not diagnostics.failed:
        fingerprint = _artifact_fingerprint(
            cfg,
            mode=mode,
            preparation_manifest_sha256=preparation_manifest_sha256,
            staging_manifest_sha256=staging_manifest_sha256,
            rows=rows,
        )
        for row in rows:
            row["run_fingerprint"] = fingerprint
    return rows


def run(cfg: DictConfig, *, mode: str) -> dict[str, Any]:
    diagnostics = Diagnostics()
    mode_cfg = cfg.sampling[mode]
    manifest_path = cfg_path(cfg, f"sampling.{mode}.design_manifest_csv")
    report_path = cfg_path(cfg, f"sampling.{mode}.index_validation_json")
    preparation_path = cfg_path(cfg, f"preparation.{mode}.manifest_csv")
    staging_path = _staging_manifest_path(cfg, mode)
    preparation_sha256 = ""
    staging_sha256 = ""
    rows: list[dict[str, Any]] = []
    try:
        preparation, preparation_path, preparation_sha256, preparation_status = (
            _validate_preparation(cfg, mode, diagnostics)
        )
        staging_path, staging_sha256 = _validate_staging(
            cfg, mode, preparation, preparation_status, diagnostics
        )
        rows = _index_outputs(
            cfg,
            mode=mode,
            preparation=preparation,
            preparation_manifest_sha256=preparation_sha256,
            staging_manifest_sha256=staging_sha256,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        diagnostics.add("index_input_error", mode, f"{type(exc).__name__}: {exc}")

    publication_status = "not_published"
    if not diagnostics.failed:
        payload = _manifest_csv_bytes(rows)
        stale_temps = sorted(manifest_path.parent.glob(f".{manifest_path.name}.tmp.*"))
        if stale_temps:
            diagnostics.add("stale_manifest_temp", mode, stale_temps)
        elif manifest_path.is_file():
            if manifest_path.read_bytes() == payload:
                publication_status = "reused"
            else:
                diagnostics.add(
                    "incompatible_existing_manifest",
                    mode,
                    manifest_path,
                )
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = manifest_path.with_name(
                f".{manifest_path.name}.tmp.{os.getpid()}"
            )
            temporary.write_bytes(payload)
            os.replace(temporary, manifest_path)
            publication_status = "published"

    report = {
        "stage": "sampling-index",
        "mode": mode,
        "status": "failed" if diagnostics.failed else "complete",
        "publication_status": publication_status,
        "preparation_manifest": str(preparation_path),
        "preparation_manifest_sha256": preparation_sha256,
        "staging_manifest": str(staging_path),
        "staging_manifest_sha256": staging_sha256,
        "sampling_output_leaf_name": _output_leaf_name(cfg),
        "expected_chunks": int(mode_cfg.chunk_count),
        "expected_samples": int(mode_cfg.input_count),
        "observed_manifest_rows": len(rows),
        "expected_manifest_rows": int(mode_cfg.input_count)
        * len(cfg.sequence_design.design_indices),
        "design_manifest": str(manifest_path),
        "design_manifest_sha256": (
            sha256_file(manifest_path)
            if not diagnostics.failed and manifest_path.is_file()
            else ""
        ),
        "run_fingerprint": rows[0]["run_fingerprint"] if rows else "",
        "diagnostics": diagnostics.payload(),
        "error_examples": diagnostics.examples,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if diagnostics.failed:
        raise RuntimeError(f"Sampling index validation failed; see {report_path}")
    return report


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    run(cfg, mode=args.mode)


if __name__ == "__main__":
    main()
