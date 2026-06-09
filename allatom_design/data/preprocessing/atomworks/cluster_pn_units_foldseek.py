#!/usr/bin/env python3
"""Cluster AtomWorks PN units with Foldseek for proteins and hash IDs otherwise."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import atomworks.enums as aw_enums
import atomworks.ml.preprocessing.constants as aw_const
import numpy as np
import pandas as pd
from atomworks.ml.utils.misc import hash_sequence
from biotite.structure.io import pdbx
from tqdm import tqdm


logger = logging.getLogger(__name__)

PROTEIN_COLUMN = "q_pn_unit_is_protein"
PEPTIDE_COLUMN = "q_pn_unit_is_peptide"
CLUSTER_ID_COLUMN = "q_pn_unit_cluster_id"

PDB_ID_COLUMN = "pdb_id"
REL_PATH_COLUMN = "rel_path"
IID_COLUMN = "q_pn_unit_iid"
TYPE_COLUMN = "q_pn_unit_type"
SEQUENCE_COLUMN = "q_pn_unit_processed_entity_canonical_sequence"
NON_POLYMER_COLUMN = "q_pn_unit_non_polymer_res_names"

DNA_RNA_TYPES = {
    aw_enums.ChainType.DNA.value,
    aw_enums.ChainType.RNA.value,
}
NON_POLYMER_TYPES = {
    aw_enums.ChainType.BRANCHED.value,
    aw_enums.ChainType.MACROLIDE.value,
    aw_enums.ChainType.NON_POLYMER.value,
    aw_enums.ChainType.WATER.value,
}


@dataclass(frozen=True)
class StagedStructure:
    pdb_id: str
    rel_path: str
    source_path: Path
    staged_path: Path


@dataclass(frozen=True)
class MissingStructure:
    pdb_id: str
    rel_path: str
    tried_paths: tuple[str, ...]


def resolve_structure_path(
    rel_path: str,
    pdb_mirror_root: Path,
) -> tuple[Path | None, tuple[str, ...]]:
    """Resolve a metadata rel_path under the local PDB mirror."""
    raw_rel_path = Path(str(rel_path))
    candidates = [pdb_mirror_root / raw_rel_path]
    rel_path_str = str(raw_rel_path)
    if rel_path_str.endswith(".cif.gz"):
        candidates.append(pdb_mirror_root / rel_path_str[: -len(".gz")])
    elif rel_path_str.endswith(".cif"):
        candidates.append(pdb_mirror_root / f"{rel_path_str}.gz")

    tried: list[str] = []
    for candidate in candidates:
        tried.append(str(candidate))
        if candidate.exists():
            return candidate, tuple(tried)
    return None, tuple(tried)


def _staged_filename(pdb_id: str, source_path: Path) -> str:
    suffix = "".join(source_path.suffixes)
    return f"{str(pdb_id).lower()}{suffix}"


def stage_unique_structure_files(
    metadata_df: pd.DataFrame,
    pdb_mirror_root: Path,
    input_dir: Path,
) -> tuple[list[StagedStructure], list[MissingStructure]]:
    """Symlink unique protein-row structures into a Foldseek input directory."""
    input_dir.mkdir(parents=True, exist_ok=True)
    protein_df = metadata_df.loc[metadata_df[PROTEIN_COLUMN]]
    structure_rows = (
        protein_df[[PDB_ID_COLUMN, REL_PATH_COLUMN]]
        .drop_duplicates()
        .sort_values([PDB_ID_COLUMN, REL_PATH_COLUMN])
    )

    staged: list[StagedStructure] = []
    missing: list[MissingStructure] = []
    for row in tqdm(
        structure_rows.itertuples(index=False),
        total=len(structure_rows),
        desc="Staging structures",
    ):
        pdb_id = str(getattr(row, PDB_ID_COLUMN))
        rel_path = str(getattr(row, REL_PATH_COLUMN))
        source_path, tried_paths = resolve_structure_path(rel_path, pdb_mirror_root)
        if source_path is None:
            missing.append(
                MissingStructure(
                    pdb_id=pdb_id,
                    rel_path=rel_path,
                    tried_paths=tried_paths,
                )
            )
            continue

        staged_path = input_dir / _staged_filename(pdb_id, source_path)
        if staged_path.exists() or staged_path.is_symlink():
            staged_path.unlink()
        staged_path.symlink_to(source_path)
        staged.append(
            StagedStructure(
                pdb_id=pdb_id,
                rel_path=rel_path,
                source_path=source_path,
                staged_path=staged_path,
            )
        )
    return staged, missing


def get_foldseek_version() -> str:
    completed = subprocess.run(
        ["foldseek", "version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or completed.stderr.strip()


def build_foldseek_easy_cluster_command(
    input_dir: Path,
    cluster_prefix: Path,
    tmp_dir: Path,
    threads: int,
    use_gpu: bool = False,
) -> list[str]:
    command = [
        "foldseek",
        "easy-cluster",
        str(input_dir),
        str(cluster_prefix),
        str(tmp_dir),
        "--threads",
        str(threads),
    ]
    if use_gpu:
        command.extend(["--gpu", "1"])
    return command


def run_foldseek_easy_cluster(
    input_dir: Path,
    cluster_prefix: Path,
    tmp_dir: Path,
    threads: int,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """Run Foldseek default easy-cluster on a directory of staged structures."""
    command = build_foldseek_easy_cluster_command(
        input_dir,
        cluster_prefix,
        tmp_dir,
        threads=threads,
        use_gpu=use_gpu,
    )
    logger.info("Running Foldseek: %s", " ".join(command))
    t0 = time.perf_counter()
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def parse_foldseek_cluster_tsv(
    cluster_tsv: Path,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Parse Foldseek ``*_cluster.tsv`` into member -> representative labels."""
    representative_by_member: dict[str, str] = {}
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
                    f"expected at least 2 tab-separated columns."
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
    return representative_by_member, conflicts


def compact_representative_labels(labels: pd.Series) -> dict[str, int]:
    """Assign deterministic 0-based integer IDs to non-null cluster labels."""
    unique_labels = sorted({str(label) for label in labels.dropna().tolist()})
    return {label: idx for idx, label in enumerate(unique_labels)}


def parse_foldseek_id(foldseek_id: str) -> tuple[str, str] | None:
    """Parse Foldseek IDs such as ``9c0h_B`` and ``6c00_MODEL_2_A``."""
    raw = str(foldseek_id).strip()
    if "_" not in raw:
        return None

    parts = raw.split("_")
    pdb_id = parts[0]
    for suffix in (".cif.gz", ".cif", ".pdb.gz", ".pdb"):
        if pdb_id.endswith(suffix):
            pdb_id = pdb_id[: -len(suffix)]
            break

    chain_id = parts[-1]
    if not pdb_id or not chain_id:
        return None
    return pdb_id.lower(), chain_id


def normalise_foldseek_pdb_id(foldseek_id: str) -> str:
    raw = str(foldseek_id).strip()
    pdb_id = raw.split("_", 1)[0]
    for suffix in (".cif.gz", ".cif", ".pdb.gz", ".pdb"):
        if pdb_id.endswith(suffix):
            pdb_id = pdb_id[: -len(suffix)]
            break
    return pdb_id.lower()


def chain_prefix_from_q_pn_unit_iid(iid: str) -> str | None:
    iid_str = str(iid)
    if "_" not in iid_str:
        return None
    prefix = iid_str.rsplit("_", 1)[0]
    return prefix or None


def build_single_protein_chain_lookup(metadata_df: pd.DataFrame) -> dict[str, str]:
    """Return pdb_id -> only protein chain prefix for unambiguous entries."""
    chains_by_pdb: dict[str, set[str]] = {}
    for row in metadata_df.loc[metadata_df[PROTEIN_COLUMN]].itertuples(index=False):
        chain_id = chain_prefix_from_q_pn_unit_iid(getattr(row, IID_COLUMN))
        if chain_id is None:
            continue
        pdb_id = str(getattr(row, PDB_ID_COLUMN)).lower()
        chains_by_pdb.setdefault(pdb_id, set()).add(chain_id)
    return {
        pdb_id: next(iter(chain_ids))
        for pdb_id, chain_ids in chains_by_pdb.items()
        if len(chain_ids) == 1
    }


def _normalise_cif_id(raw_value: Any) -> str | None:
    value = str(raw_value).strip()
    if value in {"", ".", "?"}:
        return None
    return value


def _read_cif_file(path: Path) -> pdbx.CIFFile:
    if "".join(path.suffixes).endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return pdbx.CIFFile.read(handle)
    return pdbx.CIFFile.read(path)


def _extract_auth_label_chain_pairs(source_path: Path) -> list[tuple[str, str]]:
    cif_file = _read_cif_file(source_path)
    atom_site = cif_file.block.get("atom_site")
    if atom_site is None:
        raise KeyError("CIF block is missing atom_site category")
    if "label_asym_id" not in atom_site:
        raise KeyError("atom_site category is missing label_asym_id")

    label_values = atom_site["label_asym_id"].as_array()
    auth_values = (
        atom_site["auth_asym_id"].as_array()
        if "auth_asym_id" in atom_site
        else label_values
    )
    if len(label_values) != len(auth_values):
        raise ValueError(
            f"label_asym_id/auth_asym_id length mismatch in {source_path}: "
            f"{len(label_values)} vs {len(auth_values)}"
        )

    pairs: set[tuple[str, str]] = set()
    for label_raw, auth_raw in zip(label_values, auth_values, strict=True):
        label_chain = _normalise_cif_id(label_raw)
        if label_chain is None:
            continue
        auth_chain = _normalise_cif_id(auth_raw) or label_chain
        pairs.add((auth_chain, label_chain))
    return sorted(pairs)


def build_auth_to_label_chain_lookup(
    metadata_df: pd.DataFrame,
    pdb_mirror_root: Path,
) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    """Build ``(pdb_id, auth_asym_id) -> label_asym_id`` from source mmCIF files."""
    protein_df = metadata_df.loc[metadata_df[PROTEIN_COLUMN]]
    structure_rows = (
        protein_df[[PDB_ID_COLUMN, REL_PATH_COLUMN]]
        .drop_duplicates()
        .sort_values([PDB_ID_COLUMN, REL_PATH_COLUMN])
    )

    labels_by_auth: dict[tuple[str, str], set[str]] = {}
    missing_structures: list[dict[str, Any]] = []
    read_errors: list[dict[str, str]] = []
    structures_read = 0
    for row in tqdm(
        structure_rows.itertuples(index=False),
        total=len(structure_rows),
        desc="Reading auth/label chain mappings",
    ):
        pdb_id = str(getattr(row, PDB_ID_COLUMN)).lower()
        rel_path = str(getattr(row, REL_PATH_COLUMN))
        source_path, tried_paths = resolve_structure_path(rel_path, pdb_mirror_root)
        if source_path is None:
            missing_structures.append(
                {
                    "pdb_id": pdb_id,
                    "rel_path": rel_path,
                    "tried_paths": list(tried_paths),
                }
            )
            continue
        try:
            auth_label_pairs = _extract_auth_label_chain_pairs(source_path)
        except Exception as exc:
            read_errors.append(
                {
                    "pdb_id": pdb_id,
                    "rel_path": rel_path,
                    "source_path": str(source_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        structures_read += 1
        for auth_chain, label_chain in auth_label_pairs:
            labels_by_auth.setdefault((pdb_id, auth_chain), set()).add(label_chain)

    ambiguous_mappings: list[dict[str, Any]] = []
    auth_to_label: dict[tuple[str, str], str] = {}
    for (pdb_id, auth_chain), label_chains in sorted(labels_by_auth.items()):
        if len(label_chains) == 1:
            auth_to_label[(pdb_id, auth_chain)] = next(iter(label_chains))
            continue
        ambiguous_mappings.append(
            {
                "pdb_id": pdb_id,
                "auth_asym_id": auth_chain,
                "label_asym_ids": sorted(label_chains),
            }
        )

    diagnostics = {
        "structure_count": int(len(structure_rows)),
        "structures_read": int(structures_read),
        "mapping_count": int(len(auth_to_label)),
        "missing_structure_count": len(missing_structures),
        "missing_structures": missing_structures,
        "read_error_count": len(read_errors),
        "read_errors": read_errors,
        "ambiguous_auth_chain_count": len(ambiguous_mappings),
        "ambiguous_auth_chain_mappings": ambiguous_mappings,
    }
    return auth_to_label, diagnostics


def build_chain_representative_map(
    representative_by_member: dict[str, str],
    single_chain_by_pdb: dict[str, str] | None = None,
    auth_to_label_chain: dict[tuple[str, str], str] | None = None,
    ambiguous_auth_label_chains: set[tuple[str, str]] | None = None,
) -> tuple[
    dict[tuple[str, str], str],
    list[dict[str, Any]],
    list[str],
    list[dict[str, str]],
]:
    """Map Foldseek chain labels to AtomWorks ``(pdb_id, label_chain)`` keys."""
    single_chain_by_pdb = single_chain_by_pdb or {}
    ambiguous_auth_label_chains = ambiguous_auth_label_chains or set()
    pdbs_with_auth_label_mapping = {
        pdb_id for pdb_id, _auth_chain in (auth_to_label_chain or {}).keys()
    }
    pdbs_with_auth_label_mapping.update(
        pdb_id for pdb_id, _auth_chain in ambiguous_auth_label_chains
    )
    representatives_by_chain: dict[tuple[str, str], set[str]] = {}
    unparsable_ids: list[str] = []
    auth_label_mapping_misses: list[dict[str, str]] = []
    for member, representative in representative_by_member.items():
        parsed = parse_foldseek_id(member)
        if parsed is None:
            pdb_id = normalise_foldseek_pdb_id(member)
            chain_id = single_chain_by_pdb.get(pdb_id)
            if chain_id is None:
                unparsable_ids.append(member)
                continue
            parsed = (pdb_id, chain_id)
        elif auth_to_label_chain is not None:
            pdb_id, auth_chain = parsed
            label_chain = auth_to_label_chain.get((pdb_id, auth_chain))
            if label_chain is None:
                if (pdb_id, auth_chain) in ambiguous_auth_label_chains:
                    reason = "ambiguous_auth_chain_label_mapping"
                elif pdb_id not in pdbs_with_auth_label_mapping:
                    reason = "no_pdb_auth_label_mapping"
                else:
                    reason = "no_auth_chain_label_mapping"
                auth_label_mapping_misses.append(
                    {
                        "member": member,
                        "pdb_id": pdb_id,
                        "auth_asym_id": auth_chain,
                        "reason": reason,
                    }
                )
                continue
            parsed = (pdb_id, label_chain)
        representatives_by_chain.setdefault(parsed, set()).add(representative)

    chain_to_representative: dict[tuple[str, str], str] = {}
    conflicts: list[dict[str, Any]] = []
    for key, representatives in sorted(representatives_by_chain.items()):
        chosen = sorted(representatives)[0]
        chain_to_representative[key] = chosen
        if len(representatives) > 1:
            conflicts.append(
                {
                    "pdb_id": key[0],
                    "chain_id": key[1],
                    "representatives": sorted(representatives),
                    "chosen_representative": chosen,
                }
            )
    return (
        chain_to_representative,
        conflicts,
        sorted(unparsable_ids),
        sorted(auth_label_mapping_misses, key=lambda item: item["member"]),
    )


def sorted_non_polymer_res_names(raw_value: Any) -> str | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    if not tokens:
        return None
    return ",".join(sorted(tokens))


def _hash_label(value: Any) -> str | None:
    if not isinstance(value, str) or value == "":
        return None
    return f"hash:{hash_sequence(value)}"


def add_pn_unit_type_flags(metadata_df: pd.DataFrame) -> pd.DataFrame:
    out_df = metadata_df.copy()
    is_polypeptide_l = out_df[TYPE_COLUMN] == aw_enums.ChainType.POLYPEPTIDE_L.value
    sequence_length = out_df[SEQUENCE_COLUMN].fillna("").astype(str).str.len()
    out_df[PROTEIN_COLUMN] = (
        is_polypeptide_l & (sequence_length >= aw_const.PEPTIDE_MAX_RESIDUES)
    )
    out_df[PEPTIDE_COLUMN] = (
        is_polypeptide_l & (sequence_length < aw_const.PEPTIDE_MAX_RESIDUES)
    )
    return out_df


def assign_foldseek_and_hash_clusters(
    metadata_df: pd.DataFrame,
    chain_to_representative: dict[tuple[str, str], str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign final compact int32 cluster IDs to all supported PN-unit rows."""
    out_df = add_pn_unit_type_flags(metadata_df)
    labels = pd.Series([None] * len(out_df), index=out_df.index, dtype=object)

    protein_mask = out_df[PROTEIN_COLUMN].astype(bool)
    unmapped_protein_rows: list[dict[str, Any]] = []
    for idx, row in out_df.loc[protein_mask].iterrows():
        chain_id = chain_prefix_from_q_pn_unit_iid(row[IID_COLUMN])
        key = (str(row[PDB_ID_COLUMN]).lower(), chain_id) if chain_id else None
        representative = chain_to_representative.get(key) if key else None
        if representative is None:
            unmapped_protein_rows.append(
                {
                    "index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                    "pdb_id": str(row[PDB_ID_COLUMN]),
                    "q_pn_unit_iid": str(row[IID_COLUMN]),
                }
            )
            continue
        labels.at[idx] = f"foldseek:{representative}"

    peptide_mask = out_df[PEPTIDE_COLUMN].astype(bool)
    labels.loc[peptide_mask] = out_df.loc[peptide_mask, SEQUENCE_COLUMN].map(_hash_label)

    nucleic_acid_mask = out_df[TYPE_COLUMN].isin(DNA_RNA_TYPES)
    labels.loc[nucleic_acid_mask] = out_df.loc[
        nucleic_acid_mask, SEQUENCE_COLUMN
    ].map(_hash_label)

    non_polymer_mask = out_df[TYPE_COLUMN].isin(NON_POLYMER_TYPES)
    labels.loc[non_polymer_mask] = out_df.loc[
        non_polymer_mask, NON_POLYMER_COLUMN
    ].map(sorted_non_polymer_res_names).map(_hash_label)

    label_to_cluster_id = compact_representative_labels(labels)
    cluster_ids = labels.map(label_to_cluster_id).fillna(-1).astype(np.int32)
    out_df[CLUSTER_ID_COLUMN] = cluster_ids

    diagnostics = {
        "num_clusters": len(label_to_cluster_id),
        "unmapped_protein_rows": unmapped_protein_rows,
        "label_to_cluster_id": label_to_cluster_id,
    }
    return out_df, diagnostics


def _require_columns(metadata_df: pd.DataFrame) -> None:
    required = [
        PDB_ID_COLUMN,
        REL_PATH_COLUMN,
        IID_COLUMN,
        TYPE_COLUMN,
        SEQUENCE_COLUMN,
        NON_POLYMER_COLUMN,
    ]
    missing = [column for column in required if column not in metadata_df.columns]
    if missing:
        raise KeyError(f"Input parquet is missing required columns: {missing}")


def _limit_to_first_pdb_ids(metadata_df: pd.DataFrame, max_pdb_ids: int) -> pd.DataFrame:
    if max_pdb_ids <= 0:
        return metadata_df
    keep_ids = metadata_df[PDB_ID_COLUMN].drop_duplicates().head(max_pdb_ids)
    return metadata_df[metadata_df[PDB_ID_COLUMN].isin(keep_ids)].copy()


def _prepare_work_dir(work_dir: Path, overwrite: bool) -> dict[str, Path]:
    input_dir = work_dir / "foldseek_input"
    cluster_dir = work_dir / "foldseek_clusters"
    tmp_dir = work_dir / "tmp"
    cluster_prefix = cluster_dir / "pn_units"
    if overwrite:
        for path in (input_dir, cluster_dir, tmp_dir):
            if path.exists():
                shutil.rmtree(path)
    input_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return {
        "input_dir": input_dir,
        "cluster_dir": cluster_dir,
        "tmp_dir": tmp_dir,
        "cluster_prefix": cluster_prefix,
    }


def _write_report(manifest: dict[str, Any], report_path: Path) -> None:
    row_counts = manifest["row_counts"]
    lines = [
        "# Foldseek PN-Unit Clustering Report",
        "",
        f"- input_parquet: {manifest['input_parquet']}",
        f"- output_parquet: {manifest['output_parquet']}",
        f"- work_dir: {manifest['work_dir']}",
        f"- foldseek_version: {manifest['foldseek_version']}",
        f"- foldseek_command: {' '.join(manifest['foldseek_command'] or [])}",
        f"- input_rows: {row_counts['input_rows']}",
        f"- output_rows: {row_counts['output_rows']}",
        f"- protein_rows: {row_counts['protein_rows']}",
        f"- peptide_rows: {row_counts['peptide_rows']}",
        f"- non_protein_rows: {row_counts['non_protein_rows']}",
        f"- num_clusters: {manifest['num_clusters']}",
        f"- missing_structure_paths: {manifest['missing_structure_count']}",
        f"- auth_label_mapping_read_errors: {manifest['auth_label_mapping']['read_error_count']}",
        f"- auth_label_mapping_ambiguous_auth_chains: {manifest['auth_label_mapping']['ambiguous_auth_chain_count']}",
        f"- foldseek_auth_label_mapping_misses: {manifest['foldseek_auth_label_mapping_miss_count']}",
        f"- unmapped_protein_rows: {manifest['unmapped_protein_row_count']}",
        f"- foldseek_id_mapping_conflicts: {manifest['foldseek_id_mapping_conflict_count']}",
        f"- foldseek_tsv_conflicts: {manifest['foldseek_tsv_conflict_count']}",
    ]
    report_path.write_text("\n".join(lines) + "\n")


def _serialise_dataclass_list(values: list[Any]) -> list[dict[str, Any]]:
    serialised: list[dict[str, Any]] = []
    for value in values:
        raw = value.__dict__
        serialised.append(
            {key: str(item) if isinstance(item, Path) else item for key, item in raw.items()}
        )
    return serialised


def build_manifest(
    args: argparse.Namespace,
    foldseek_version: str,
    foldseek_result: dict[str, Any] | None,
    row_counts: dict[str, int],
    staged: list[StagedStructure],
    missing: list[MissingStructure],
    tsv_conflicts: list[dict[str, str]],
    mapping_conflicts: list[dict[str, Any]],
    unparsable_ids: list[str],
    auth_label_mapping_diagnostics: dict[str, Any],
    auth_label_mapping_misses: list[dict[str, str]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    unmapped = diagnostics["unmapped_protein_rows"]
    return {
        "input_parquet": str(args.input_parquet),
        "output_parquet": str(args.output_parquet),
        "work_dir": str(args.work_dir),
        "pdb_mirror_root": str(args.pdb_mirror_root),
        "max_pdb_ids": int(args.max_pdb_ids),
        "threads": int(args.threads),
        "foldseek_gpu": bool(args.foldseek_gpu),
        "foldseek_version": foldseek_version,
        "foldseek_command": None if foldseek_result is None else foldseek_result["command"],
        "foldseek_result": foldseek_result,
        "row_counts": row_counts,
        "num_clusters": int(diagnostics["num_clusters"]),
        "staged_structure_count": len(staged),
        "staged_structures": _serialise_dataclass_list(staged),
        "missing_structure_count": len(missing),
        "missing_structure_paths": _serialise_dataclass_list(missing),
        "auth_label_mapping": auth_label_mapping_diagnostics,
        "foldseek_auth_label_mapping_miss_count": len(auth_label_mapping_misses),
        "foldseek_auth_label_mapping_misses": auth_label_mapping_misses,
        "unmapped_protein_row_count": len(unmapped),
        "unmapped_protein_rows": unmapped,
        "foldseek_tsv_conflict_count": len(tsv_conflicts),
        "foldseek_tsv_conflicts": tsv_conflicts,
        "foldseek_id_mapping_conflict_count": len(mapping_conflicts),
        "foldseek_id_mapping_conflicts": mapping_conflicts,
        "unparsable_foldseek_ids": unparsable_ids,
    }


def run(args: argparse.Namespace) -> None:
    if args.output_parquet.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output parquet already exists: {args.output_parquet}. "
            "Pass --overwrite to replace it."
        )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    work_paths = _prepare_work_dir(args.work_dir, overwrite=args.overwrite)

    logger.info("Reading input parquet: %s", args.input_parquet)
    input_df = pd.read_parquet(args.input_parquet)
    _require_columns(input_df)
    input_rows_before_limit = len(input_df)
    input_df = _limit_to_first_pdb_ids(input_df, args.max_pdb_ids)
    logger.info(
        "Loaded %s rows (%s before max-pdb-id limit).",
        f"{len(input_df):,}",
        f"{input_rows_before_limit:,}",
    )

    typed_df = add_pn_unit_type_flags(input_df)
    foldseek_version = get_foldseek_version()
    staged, missing = stage_unique_structure_files(
        typed_df,
        args.pdb_mirror_root,
        work_paths["input_dir"],
    )

    foldseek_result: dict[str, Any] | None = None
    representative_by_member: dict[str, str] = {}
    tsv_conflicts: list[dict[str, str]] = []
    if staged:
        foldseek_result = run_foldseek_easy_cluster(
            work_paths["input_dir"],
            work_paths["cluster_prefix"],
            work_paths["tmp_dir"],
            args.threads,
            use_gpu=args.foldseek_gpu,
        )
        cluster_tsv = work_paths["cluster_prefix"].with_name(
            f"{work_paths['cluster_prefix'].name}_cluster.tsv"
        )
        representative_by_member, tsv_conflicts = parse_foldseek_cluster_tsv(cluster_tsv)
    else:
        logger.warning("No structures staged for Foldseek; protein rows will be unmapped.")

    single_chain_by_pdb = build_single_protein_chain_lookup(typed_df)
    auth_to_label_chain, auth_label_mapping_diagnostics = (
        build_auth_to_label_chain_lookup(typed_df, args.pdb_mirror_root)
    )
    ambiguous_auth_label_chains = {
        (str(item["pdb_id"]).lower(), str(item["auth_asym_id"]))
        for item in auth_label_mapping_diagnostics["ambiguous_auth_chain_mappings"]
    }
    (
        chain_to_representative,
        mapping_conflicts,
        unparsable_ids,
        auth_label_mapping_misses,
    ) = (
        build_chain_representative_map(
            representative_by_member,
            single_chain_by_pdb=single_chain_by_pdb,
            auth_to_label_chain=auth_to_label_chain,
            ambiguous_auth_label_chains=ambiguous_auth_label_chains,
        )
    )
    out_df, diagnostics = assign_foldseek_and_hash_clusters(
        typed_df,
        chain_to_representative,
    )
    out_df[CLUSTER_ID_COLUMN] = out_df[CLUSTER_ID_COLUMN].astype(np.int32)

    if len(out_df) != len(input_df):
        raise RuntimeError(
            f"Output row count mismatch: {len(out_df)} vs input {len(input_df)}."
        )
    if out_df[CLUSTER_ID_COLUMN].dtype != np.int32:
        raise RuntimeError(f"{CLUSTER_ID_COLUMN} is not np.int32.")

    logger.info("Writing output parquet: %s", args.output_parquet)
    out_df.to_parquet(args.output_parquet)

    row_counts = {
        "input_rows": int(len(input_df)),
        "input_rows_before_max_pdb_ids": int(input_rows_before_limit),
        "output_rows": int(len(out_df)),
        "protein_rows": int(out_df[PROTEIN_COLUMN].sum()),
        "peptide_rows": int(out_df[PEPTIDE_COLUMN].sum()),
        "non_protein_rows": int((~out_df[PROTEIN_COLUMN].astype(bool)).sum()),
    }
    manifest = build_manifest(
        args=args,
        foldseek_version=foldseek_version,
        foldseek_result=foldseek_result,
        row_counts=row_counts,
        staged=staged,
        missing=missing,
        tsv_conflicts=tsv_conflicts,
        mapping_conflicts=mapping_conflicts,
        unparsable_ids=unparsable_ids,
        auth_label_mapping_diagnostics=auth_label_mapping_diagnostics,
        auth_label_mapping_misses=auth_label_mapping_misses,
        diagnostics=diagnostics,
    )
    manifest_path = args.output_parquet.with_suffix(".manifest.json")
    report_path = args.output_parquet.with_suffix(".report.md")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _write_report(manifest, report_path)
    logger.info("Wrote manifest: %s", manifest_path)
    logger.info("Wrote report: %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster base AtomWorks PN units by Foldseek default easy-cluster for "
            "protein chains, with legacy hash singleton clusters for peptides, "
            "nucleic acids, and non-polymers."
        )
    )
    parser.add_argument("--input-parquet", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--pdb-mirror-root", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--foldseek-gpu",
        action="store_true",
        help="Pass --gpu 1 to Foldseek easy-cluster. CPU/default mode is unchanged.",
    )
    parser.add_argument(
        "--max-pdb-ids",
        type=int,
        default=0,
        help="Smoke/debug mode: keep only the first N distinct pdb_ids (0 = no limit).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output files and clear this entrypoint's work subdirectories.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run(parse_args())


if __name__ == "__main__":
    main()
