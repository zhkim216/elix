#!/usr/bin/env python3
"""Jointly curate denovoval sampler panels within each ``(CCD, length)``.

The two sampler conditions are clustered together, but independently for each
``(CCD, length)`` group.  Selection is fail-closed and lexicographically keeps
existing main sources, then the existing subset source, then the original
curation order.  RASA is a strict eligibility filter, not a ranking signal.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


EVAL_ROOT = Path("/scratch/users/zhkim216/datasets/evaluation_datasets")
DEFAULT_OUTPUT_ROOT = EVAL_ROOT / "denovoval_cross_condition_foldseek_v1"
DEFAULT_SS1P0_SOURCE_ROOT = DEFAULT_OUTPUT_ROOT / "work/ss1p0_source"
DEFAULT_SS1P0_AUDIT = (
    DEFAULT_SS1P0_SOURCE_ROOT
    / "curation_steps/step1/selection/candidate_audit.tsv"
)
DEFAULT_SS1P6_AUDIT = (
    EVAL_ROOT / "curation/ver2/outputs/denovoval/step6/candidate_audit.tsv"
)
DEFAULT_SS1P6_RAW_ROOT = (
    EVAL_ROOT
    / "curation/ver2/outputs/denovoval/step2/generated_samples/full"
)
DEFAULT_CURRENT_MAIN_ROOT = EVAL_ROOT / "denovoval_re"
DEFAULT_CURRENT_SUBSET_ROOT = EVAL_ROOT / "denovoval_subset"
DEFAULT_FOLDSEEK_BIN = Path(
    "/oak/stanford/groups/possu/jinho/software/foldseek/bin/foldseek"
)
EXPECTED_FOLDSEEK_VERSION = "718d42176d2f67d36a60866fedfb881f8d5a7ebf"
EXPECTED_FOLDSEEK_SHA256 = (
    "870b802d0fd715f43d599652c7fcf48c421aa0eb1df28862d32b9ae04b0e88fa"
)
FOLDSEEK_CLUSTER_C = 0.8
RASA_CUTOFF = 0.2
CONDITIONS = ("ss1p0_gamma1p0", "ss1p6_gamma0p6")
COMBINED_CONDITION_ORDER = ("ss1p6_gamma0p6", "ss1p0_gamma1p0")
MAIN_SLOTS = tuple(range(5))
SUBSET_SLOT = 5
SLOTS_PER_CONDITION = 6
EXPECTED_GROUPS = 308
EXPECTED_RAW_PER_GROUP = {
    "ss1p0_gamma1p0": 100,
    "ss1p6_gamma0p6": 300,
}
CORRECTED_BASELINE_CCDS = frozenset({"GLU", "MET"})
EXTRA_ORIGIN = "seed8_n100"
EXPECTED_EXTRA_PER_GROUP = {
    ("ss1p6_gamma0p6", "G2P_len300"): 100,
    ("ss1p6_gamma0p6", "GLC_len300"): 100,
    ("ss1p0_gamma1p0", "GLU_len150"): 100,
    ("ss1p0_gamma1p0", "GLU_len300"): 100,
    ("ss1p6_gamma0p6", "GLU_len300"): 100,
}
ROLE_COLUMNS = (
    "pdb_key",
    "binder_pn_unit_iids",
    "context_pn_unit_iids",
    "frame_pn_unit_iids",
    "template_pn_unit_iids",
    "fixed_pos_seq",
    "fixed_pos_scn",
    "fixed_pos_override_seq",
    "pos_restrict_aatype",
)
SOURCE_INDEX_COLUMN = "partial_diffusion_source_index"
SAMPLER_COLUMN = "partial_diffusion_sampler"


@dataclass(frozen=True)
class Candidate:
    condition: str
    group: str
    ccd: str
    length: int
    staged_id: str
    joint_id: str
    source_path: str
    category: str
    ligand_class: str
    status: str
    rasa_value: float
    eligible: bool
    audit_selection_rank: int | None
    continuation_rank: int | None
    current_panel: str | None
    current_slot: int | None
    candidate_origin: str = "baseline"
    preference_order: int = 0


@dataclass(frozen=True)
class CurrentSlot:
    condition: str
    group: str
    slot: int
    panel: str
    pdb_key: str
    staged_id: str
    source_index: int | None
    cif_path: str


@dataclass(frozen=True)
class SelectedSlot:
    condition: str
    group: str
    ccd: str
    length: int
    panel: str
    slot: int
    pdb_key: str
    runtime_pdb_key: str
    source_index: int | None
    source_staged_id: str
    source_cif_path: str
    source_cif_sha256: str
    materialized_cif_sha256: str
    rasa_value: float
    foldseek_cluster_id: str
    current_source_staged_id: str
    changed: bool
    preference_order: int
    selection_reason: str


@dataclass(frozen=True)
class SearchState:
    score: tuple[int, int, int]
    chosen: tuple[Candidate, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_table(path: Path, *, delimiter: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Missing table header: {path}")
        return list(reader)


def table_header(path: Path, *, delimiter: str) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        return next(reader)


def write_table(
    path: Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
    *,
    delimiter: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter=delimiter,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_columns(rows: list[dict[str, str]], required: set[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows in {path}")
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def parse_optional_int(raw: str | None) -> int | None:
    value = str(raw or "").strip()
    if not value:
        return None
    try:
        numeric = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Expected an integer-valued field, got {value!r}") from error
    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        raise ValueError(f"Expected an integer-valued field, got {value!r}")
    return int(numeric)


def parse_key(key: str) -> tuple[str, int]:
    group, separator, raw_slot = key.rpartition("_")
    if not separator or not group or not raw_slot.isdigit():
        raise ValueError(f"Unexpected denovoval key: {key!r}")
    return group, int(raw_slot)


def parse_group(group: str) -> tuple[str, int]:
    ccd, separator, raw_length = group.rpartition("_len")
    if not separator or not ccd or not raw_length.isdigit():
        raise ValueError(f"Unexpected denovoval group: {group!r}")
    return ccd, int(raw_length)


def safe_group(group: str) -> str:
    if group in {"", ".", ".."} or "/" in group or "\x00" in group:
        raise ValueError(f"Unsafe group name: {group!r}")
    parse_group(group)
    return group


def unique_rows(
    rows: Iterable[dict[str, str]], columns: Sequence[str], label: str
) -> dict[tuple[str, ...], dict[str, str]]:
    result: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[column] for column in columns)
        if key in result:
            raise ValueError(f"Duplicate {columns}={key} in {label}")
        result[key] = row
    return result


def current_state(
    main_root: Path, subset_root: Path
) -> tuple[
    dict[tuple[str, str, int], CurrentSlot],
    dict[tuple[str, str], int],
    dict[tuple[str, str], int],
    list[str],
    dict[str, list[dict[str, str]]],
]:
    rasa_path = main_root / "selected_ligand_rasa.tsv"
    rasa_rows = read_table(rasa_path, delimiter="\t")
    require_columns(
        rasa_rows,
        {"condition", "pdb_key", "source_staged_id"},
        rasa_path,
    )
    rasa_by_key = unique_rows(rasa_rows, ("condition", "pdb_key"), str(rasa_path))

    slots: dict[tuple[str, str, int], CurrentSlot] = {}
    main_source_indices: dict[tuple[str, str], int] = {}
    role_rows_by_condition: dict[str, list[dict[str, str]]] = {}
    group_order: list[str] = []
    for condition in CONDITIONS:
        csv_path = main_root / f"denovoval_{condition}_sampling_inputs.csv"
        rows = read_table(csv_path, delimiter=",")
        require_columns(rows, set(ROLE_COLUMNS) | {SOURCE_INDEX_COLUMN}, csv_path)
        if len(rows) != EXPECTED_GROUPS * len(MAIN_SLOTS):
            raise ValueError(f"Expected 1540 rows in {csv_path}, found {len(rows)}")
        role_rows_by_condition[condition] = rows
        seen_indices: set[int] = set()
        for row in rows:
            pdb_key = row["pdb_key"]
            group, slot = parse_key(pdb_key)
            if slot not in MAIN_SLOTS:
                raise ValueError(f"Main slot is outside 0..4: {condition}:{pdb_key}")
            source_index = int(row[SOURCE_INDEX_COLUMN])
            if source_index < 0 or source_index in seen_indices:
                raise ValueError(f"Invalid or duplicate source index in {csv_path}: {source_index}")
            seen_indices.add(source_index)
            source = rasa_by_key.get((condition, pdb_key))
            if source is None:
                raise KeyError(f"Missing source identity for {condition}:{pdb_key}")
            identity = (condition, group, slot)
            slots[identity] = CurrentSlot(
                condition=condition,
                group=group,
                slot=slot,
                panel="main",
                pdb_key=pdb_key,
                staged_id=source["source_staged_id"],
                source_index=source_index,
                cif_path=str(main_root / condition / "cifs" / f"{pdb_key}.cif"),
            )
            main_source_indices[(condition, pdb_key)] = source_index
            if condition == CONDITIONS[0] and slot == MAIN_SLOTS[0]:
                group_order.append(group)

    if len(group_order) != EXPECTED_GROUPS or len(set(group_order)) != EXPECTED_GROUPS:
        raise ValueError("Current main group order is incomplete or duplicated")
    expected_slot_keys = {
        (condition, group, slot)
        for condition in CONDITIONS
        for group in group_order
        for slot in MAIN_SLOTS
    }
    if set(slots) != expected_slot_keys:
        raise ValueError(
            "Current main slot identity mismatch: "
            f"missing={len(expected_slot_keys - set(slots))} "
            f"extra={len(set(slots) - expected_slot_keys)}"
        )

    subset_selection = subset_root / "selection.tsv"
    subset_rows = read_table(subset_selection, delimiter="\t")
    require_columns(
        subset_rows,
        {
            "condition",
            "pdb_key",
            "output_index",
            "source_staged_id",
            "continuation_rank",
        },
        subset_selection,
    )
    continuation_ranks: dict[tuple[str, str], int] = {}
    for row in subset_rows:
        condition = row["condition"]
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown subset condition: {condition}")
        group, slot = parse_key(row["pdb_key"])
        output_index = int(row["output_index"])
        if slot != output_index:
            raise ValueError(f"Subset key/index mismatch: {row['pdb_key']}")
        rank = int(row["continuation_rank"])
        source_identity = (condition, row["source_staged_id"])
        previous_rank = continuation_ranks.get(source_identity)
        if previous_rank is None or rank < previous_rank:
            continuation_ranks[source_identity] = rank
        if output_index != SUBSET_SLOT:
            continue
        identity = (condition, group, SUBSET_SLOT)
        if identity in slots:
            raise ValueError(f"Duplicate current subset slot: {identity}")
        combined_key = f"{group}_{condition}_{SUBSET_SLOT}"
        combined_cif = subset_root / "cifs" / f"{combined_key}.cif"
        slots[identity] = CurrentSlot(
            condition=condition,
            group=group,
            slot=SUBSET_SLOT,
            panel="subset",
            pdb_key=row["pdb_key"],
            staged_id=row["source_staged_id"],
            source_index=None,
            cif_path=str(combined_cif),
        )

    expected_all_slots = {
        (condition, group, slot)
        for condition in CONDITIONS
        for group in group_order
        for slot in range(SLOTS_PER_CONDITION)
    }
    if set(slots) != expected_all_slots:
        raise ValueError(
            "Current main/subset slot identity mismatch: "
            f"missing={len(expected_all_slots - set(slots))} "
            f"extra={len(set(slots) - expected_all_slots)}"
        )
    return (
        slots,
        main_source_indices,
        continuation_ranks,
        group_order,
        role_rows_by_condition,
    )


def candidate_preference_key(candidate: Candidate) -> tuple[Any, ...]:
    if candidate.current_panel == "main":
        return (0, int(candidate.current_slot or 0), candidate.staged_id)
    if candidate.current_panel == "subset":
        return (1, 0, candidate.staged_id)
    ranks = [
        rank
        for rank in (candidate.continuation_rank, candidate.audit_selection_rank)
        if rank is not None
    ]
    if ranks:
        return (2, min(ranks), candidate.staged_id)
    if candidate.candidate_origin == "baseline":
        return (3, candidate.staged_id)
    return (4, candidate.candidate_origin, candidate.staged_id)


def load_rasa_overrides(
    manifest_path: Path | None, result_path: Path | None
) -> dict[tuple[str, str], dict[str, Any]]:
    if manifest_path is None and result_path is None:
        return {}
    if manifest_path is None or result_path is None:
        raise ValueError(
            "--rasa-override-manifest and --rasa-override-results must be provided together"
        )
    manifest_rows = read_table(manifest_path, delimiter="\t")
    result_rows = read_table(result_path, delimiter="\t")
    require_columns(
        manifest_rows,
        {
            "staged_id",
            "condition",
            "group",
            "ccd",
            "length",
            "source_path",
            "json_path",
            "candidate_staged_id",
            "candidate_origin",
        },
        manifest_path,
    )
    require_columns(
        result_rows,
        {
            "staged_id",
            "condition",
            "ccd",
            "length",
            "category",
            "ligand_class",
            "source_path",
            "json_path",
            "status",
            "detail",
            "n_target_heavy_atoms",
            "n_target_residues",
            "rasa_value",
        },
        result_path,
    )
    manifest_by_id = unique_rows(manifest_rows, ("staged_id",), str(manifest_path))
    result_by_id = unique_rows(result_rows, ("staged_id",), str(result_path))
    if set(manifest_by_id) != set(result_by_id):
        raise ValueError(
            "RASA override identity mismatch: "
            f"missing={list(set(manifest_by_id) - set(result_by_id))[:10]} "
            f"extra={list(set(result_by_id) - set(manifest_by_id))[:10]}"
        )

    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    for (rasa_id,), manifest_row in manifest_by_id.items():
        result = result_by_id[(rasa_id,)]
        condition = manifest_row["condition"]
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown RASA override condition: {condition}")
        ccd = manifest_row["ccd"].strip().upper()
        length = int(manifest_row["length"])
        group = f"{ccd}_len{length}"
        if manifest_row["group"] != group:
            raise ValueError(f"RASA override group mismatch for {rasa_id}")
        origin = manifest_row["candidate_origin"]
        if origin not in {"baseline", EXTRA_ORIGIN}:
            raise ValueError(f"Unknown candidate origin for {rasa_id}: {origin}")
        candidate_staged_id = manifest_row["candidate_staged_id"]
        if not candidate_staged_id:
            raise ValueError(f"Blank candidate_staged_id for {rasa_id}")
        for column in ("condition", "ccd", "length", "source_path", "json_path"):
            if result[column] != manifest_row[column]:
                raise ValueError(f"RASA override metadata mismatch for {rasa_id}:{column}")
        if result["status"] != "ok":
            raise ValueError(
                f"RASA override failed for {rasa_id}: "
                f"{result['status']} {result['detail']}"
            )
        rasa_value = float(result["rasa_value"])
        if not math.isfinite(rasa_value):
            raise ValueError(f"Non-finite RASA override for {rasa_id}")
        if int(result["n_target_heavy_atoms"]) <= 0 or int(result["n_target_residues"]) != 1:
            raise ValueError(f"Invalid hetero target identity for {rasa_id}")
        source_path = Path(manifest_row["source_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        identity = (condition, candidate_staged_id)
        if identity in overrides:
            raise ValueError(f"Duplicate RASA candidate identity: {identity}")
        overrides[identity] = {
            "condition": condition,
            "group": group,
            "ccd": ccd,
            "length": length,
            "candidate_staged_id": candidate_staged_id,
            "candidate_origin": origin,
            "source_path": str(source_path),
            "category": result["category"],
            "ligand_class": result["ligand_class"],
            "status": result["status"],
            "rasa_value": rasa_value,
        }
    return overrides


def load_candidates(
    *,
    ss1p0_audit: Path,
    ss1p0_raw_root: Path,
    ss1p6_audit: Path,
    ss1p6_raw_root: Path,
    current_slots: dict[tuple[str, str, int], CurrentSlot],
    continuation_ranks: dict[tuple[str, str], int],
    group_order: list[str],
    rasa_override_manifest: Path | None = None,
    rasa_override_results: Path | None = None,
) -> list[Candidate]:
    rasa_overrides = load_rasa_overrides(
        rasa_override_manifest, rasa_override_results
    )
    current_by_source: dict[tuple[str, str], CurrentSlot] = {}
    for slot in current_slots.values():
        identity = (slot.condition, slot.staged_id)
        if identity in current_by_source:
            raise ValueError(f"Current source appears in multiple slots: {identity}")
        current_by_source[identity] = slot

    group_set = set(group_order)
    all_candidates: list[Candidate] = []
    for condition, audit_path, raw_root in (
        ("ss1p0_gamma1p0", ss1p0_audit, ss1p0_raw_root),
        ("ss1p6_gamma0p6", ss1p6_audit, ss1p6_raw_root),
    ):
        rows = read_table(audit_path, delimiter="\t")
        require_columns(
            rows,
            {
                "staged_id",
                "ccd",
                "length",
                "category",
                "ligand_class",
                "status",
                "rasa_value",
                "selection_rank",
            },
            audit_path,
        )
        seen: set[str] = set()
        by_group: Counter[str] = Counter()
        for row in rows:
            group = f"{row['ccd']}_len{int(row['length'])}"
            if group not in group_set:
                continue
            staged_id = row["staged_id"]
            if staged_id in seen:
                raise ValueError(f"Duplicate staged_id in {audit_path}: {staged_id}")
            seen.add(staged_id)
            source_path = raw_root / group / f"{staged_id}.cif.gz"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            try:
                rasa_value = float(row["rasa_value"])
            except ValueError:
                rasa_value = math.nan
            status = row["status"]
            override = rasa_overrides.get((condition, staged_id))
            if override is not None:
                if override["candidate_origin"] != "baseline":
                    raise ValueError(
                        f"Baseline candidate has non-baseline RASA origin: {condition}:{staged_id}"
                    )
                if (
                    override["group"] != group
                    or Path(override["source_path"]).resolve() != source_path.resolve()
                ):
                    raise ValueError(
                        f"Baseline RASA override source mismatch: {condition}:{staged_id}"
                    )
                status = override["status"]
                rasa_value = override["rasa_value"]
            eligible = status == "ok" and math.isfinite(rasa_value) and rasa_value < RASA_CUTOFF
            current = current_by_source.get((condition, staged_id))
            candidate = Candidate(
                condition=condition,
                group=group,
                ccd=row["ccd"],
                length=int(row["length"]),
                staged_id=staged_id,
                joint_id=f"{condition}__{staged_id}",
                source_path=str(source_path),
                category=row["category"],
                ligand_class=row["ligand_class"],
                status=status,
                rasa_value=rasa_value,
                eligible=eligible,
                audit_selection_rank=parse_optional_int(row["selection_rank"]),
                continuation_rank=continuation_ranks.get((condition, staged_id)),
                current_panel=current.panel if current else None,
                current_slot=current.slot if current else None,
                candidate_origin="baseline",
            )
            all_candidates.append(candidate)
            by_group[group] += 1
        expected_per_group = EXPECTED_RAW_PER_GROUP[condition]
        invalid_counts = {
            group: by_group[group]
            for group in group_order
            if by_group[group] != expected_per_group
        }
        if invalid_counts:
            examples = list(invalid_counts.items())[:10]
            raise ValueError(
                f"Unexpected raw candidate counts for {condition}: {examples}"
            )

    if rasa_overrides:
        expected_baseline_overrides = {
            (candidate.condition, candidate.staged_id)
            for candidate in all_candidates
            if candidate.ccd in CORRECTED_BASELINE_CCDS
        }
        actual_baseline_overrides = {
            identity
            for identity, override in rasa_overrides.items()
            if override["candidate_origin"] == "baseline"
        }
        if actual_baseline_overrides != expected_baseline_overrides:
            raise ValueError(
                "Corrected baseline RASA coverage mismatch: "
                f"missing={len(expected_baseline_overrides - actual_baseline_overrides)} "
                f"extra={len(actual_baseline_overrides - expected_baseline_overrides)}"
            )

        extra_counts: Counter[tuple[str, str]] = Counter()
        for identity, override in rasa_overrides.items():
            if override["candidate_origin"] == "baseline":
                continue
            condition, staged_id = identity
            group = override["group"]
            current = current_by_source.get(identity)
            if current is not None:
                raise ValueError(f"Extra candidate collides with a current source: {identity}")
            all_candidates.append(
                Candidate(
                    condition=condition,
                    group=group,
                    ccd=override["ccd"],
                    length=override["length"],
                    staged_id=staged_id,
                    joint_id=f"{condition}__{staged_id}",
                    source_path=override["source_path"],
                    category=override["category"],
                    ligand_class=override["ligand_class"],
                    status=override["status"],
                    rasa_value=override["rasa_value"],
                    eligible=override["rasa_value"] < RASA_CUTOFF,
                    audit_selection_rank=None,
                    continuation_rank=None,
                    current_panel=None,
                    current_slot=None,
                    candidate_origin=EXTRA_ORIGIN,
                )
            )
            extra_counts[(condition, group)] += 1
        if extra_counts != EXPECTED_EXTRA_PER_GROUP:
            raise ValueError(
                f"Unexpected extra candidate counts: {dict(sorted(extra_counts.items()))}"
            )

    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in all_candidates:
        grouped[(candidate.condition, candidate.group)].append(candidate)
    ordered: list[Candidate] = []
    for group in group_order:
        for condition in CONDITIONS:
            candidates = sorted(
                grouped[(condition, group)], key=candidate_preference_key
            )
            for preference_order, candidate in enumerate(candidates, start=1):
                ordered.append(
                    Candidate(**{**asdict(candidate), "preference_order": preference_order})
                )
    expected_counts = {
        (condition, group): EXPECTED_RAW_PER_GROUP[condition]
        + (EXPECTED_EXTRA_PER_GROUP.get((condition, group), 0) if rasa_overrides else 0)
        for condition in CONDITIONS
        for group in group_order
    }
    actual_counts = Counter((candidate.condition, candidate.group) for candidate in ordered)
    invalid_counts = {
        identity: actual_counts[identity]
        for identity, expected_count in expected_counts.items()
        if actual_counts[identity] != expected_count
    }
    if invalid_counts:
        raise ValueError(f"Augmented candidate count mismatch: {list(invalid_counts.items())[:10]}")
    expected_total = sum(expected_counts.values())
    if len(ordered) != expected_total:
        raise ValueError(f"Expected {expected_total} candidates, found {len(ordered)}")
    current_source_identities = {
        (slot.condition, slot.staged_id) for slot in current_slots.values()
    }
    candidate_identities = {(row.condition, row.staged_id) for row in ordered}
    missing_current = sorted(current_source_identities - candidate_identities)
    if missing_current:
        raise ValueError(f"Current sources are absent from raw candidates: {missing_current[:10]}")
    return ordered


def candidate_manifest_rows(candidates: list[Candidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = asdict(candidate)
        row["rasa_value"] = "" if not math.isfinite(candidate.rasa_value) else candidate.rasa_value
        row["eligible"] = str(candidate.eligible)
        row["audit_selection_rank"] = candidate.audit_selection_rank or ""
        row["continuation_rank"] = candidate.continuation_rank or ""
        row["current_panel"] = candidate.current_panel or ""
        row["current_slot"] = (
            "" if candidate.current_slot is None else candidate.current_slot
        )
        rows.append(row)
    return rows


def foldseek_version(foldseek_bin: Path) -> str:
    result = subprocess.run(
        [str(foldseek_bin), "version"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def validate_foldseek(foldseek_bin: Path) -> dict[str, str]:
    if not foldseek_bin.is_file():
        raise FileNotFoundError(foldseek_bin)
    version = foldseek_version(foldseek_bin)
    digest = sha256_file(foldseek_bin)
    if version != EXPECTED_FOLDSEEK_VERSION:
        raise ValueError(
            f"Foldseek version mismatch: expected {EXPECTED_FOLDSEEK_VERSION}, got {version}"
        )
    if digest != EXPECTED_FOLDSEEK_SHA256:
        raise ValueError(
            f"Foldseek SHA256 mismatch: expected {EXPECTED_FOLDSEEK_SHA256}, got {digest}"
        )
    return {"version": version, "sha256": digest}


def group_candidate_fingerprint(candidates: list[Candidate]) -> str:
    records: list[str] = []
    for candidate in sorted(candidates, key=lambda row: row.joint_id):
        path = Path(candidate.source_path)
        stat = path.stat()
        records.append(
            f"{candidate.joint_id}\t{path}\t{stat.st_size}\t{stat.st_mtime_ns}\n"
        )
    return sha256_text("".join(records))


def resolve_foldseek_id(raw_id: str, known: dict[str, Candidate]) -> str | None:
    name = Path(raw_id).name
    candidates = [name]
    for suffix in (".cif.gz", ".cif", ".gz"):
        if name.endswith(suffix):
            candidates.append(name[: -len(suffix)])
            break
    for candidate in list(candidates):
        if "_" in candidate:
            candidates.append(candidate.rsplit("_", 1)[0])
    for candidate in candidates:
        if candidate in known:
            return candidate
    return None


def read_complete_group(
    group_dir: Path, expected_fingerprint: str, expected_count: int
) -> bool:
    manifest_path = group_dir / "manifest.json"
    assignments_path = group_dir / "cluster_assignments.tsv"
    if not manifest_path.is_file() or not assignments_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "complete"
        or manifest.get("candidate_fingerprint") != expected_fingerprint
        or int(manifest.get("candidate_count", -1)) != expected_count
    ):
        return False
    rows = read_table(assignments_path, delimiter="\t")
    return len(rows) == expected_count


def cluster_one_group(
    *,
    group: str,
    candidates: list[Candidate],
    groups_root: Path,
    work_root: Path,
    foldseek_bin: Path,
    foldseek_info: dict[str, str],
    threads: int,
) -> tuple[str, str]:
    safe_group(group)
    group_dir = groups_root / group
    fingerprint = group_candidate_fingerprint(candidates)
    if read_complete_group(group_dir, fingerprint, len(candidates)):
        return group, "skipped"
    if group_dir.exists() or group_dir.is_symlink():
        raise FileExistsError(f"Incomplete or stale group output: {group_dir}")

    stage = Path(tempfile.mkdtemp(prefix=f".{group}.", dir=work_root))
    try:
        input_dir = stage / "input"
        input_dir.mkdir()
        known = {candidate.joint_id: candidate for candidate in candidates}
        if len(known) != len(candidates):
            raise ValueError(f"Duplicate joint IDs in {group}")
        for candidate in sorted(candidates, key=lambda row: row.joint_id):
            os.symlink(candidate.source_path, input_dir / f"{candidate.joint_id}.cif.gz")
        foldseek_dir = stage / "foldseek"
        foldseek_dir.mkdir()
        tmp_dir = stage / "tmp"
        tmp_dir.mkdir()
        prefix = foldseek_dir / "clu"
        command = [
            str(foldseek_bin),
            "easy-cluster",
            str(input_dir),
            str(prefix),
            str(tmp_dir),
            "-c",
            str(FOLDSEEK_CLUSTER_C),
            "--threads",
            str(threads),
        ]
        log_path = stage / "foldseek.log"
        with log_path.open("w") as log:
            log.write("[cmd] " + " ".join(command) + "\n")
            log.flush()
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        cluster_path = foldseek_dir / "clu_cluster.tsv"
        if not cluster_path.is_file():
            raise FileNotFoundError(cluster_path)

        assignments: dict[str, tuple[str, str]] = {}
        unmapped: list[str] = []
        with cluster_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    unmapped.append(f"line {line_number}: expected two fields")
                    continue
                raw_representative, raw_member = fields[:2]
                representative = resolve_foldseek_id(raw_representative, known)
                member = resolve_foldseek_id(raw_member, known)
                if representative is None or member is None:
                    unmapped.append(
                        f"line {line_number}: {raw_representative!r} {raw_member!r}"
                    )
                    continue
                if member in assignments:
                    raise ValueError(f"Duplicate Foldseek member assignment in {group}: {member}")
                assignments[member] = (representative, raw_member)
        if unmapped:
            raise ValueError(f"Unmapped Foldseek IDs in {group}: {unmapped[:10]}")
        missing = sorted(set(known) - set(assignments))
        extra = sorted(set(assignments) - set(known))
        if missing or extra:
            raise ValueError(
                f"Foldseek membership mismatch in {group}: missing={missing[:10]} extra={extra[:10]}"
            )

        assignment_rows: list[dict[str, Any]] = []
        for joint_id in sorted(assignments):
            candidate = known[joint_id]
            representative, raw_member = assignments[joint_id]
            assignment_rows.append(
                {
                    "condition": candidate.condition,
                    "ccd": candidate.ccd,
                    "length": candidate.length,
                    "group": group,
                    "staged_id": candidate.staged_id,
                    "joint_id": joint_id,
                    "foldseek_cluster_id": representative,
                    "raw_member_id": raw_member,
                }
            )
        write_table(
            stage / "cluster_assignments.tsv",
            assignment_rows,
            (
                "condition",
                "ccd",
                "length",
                "group",
                "staged_id",
                "joint_id",
                "foldseek_cluster_id",
                "raw_member_id",
            ),
            delimiter="\t",
        )
        shutil.copy2(cluster_path, stage / "foldseek_cluster.tsv")
        write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "status": "complete",
                "created_at_utc": utc_now(),
                "group": group,
                "candidate_count": len(candidates),
                "cluster_count": len({value[0] for value in assignments.values()}),
                "candidate_fingerprint": fingerprint,
                "foldseek_bin": str(foldseek_bin),
                "foldseek_version": foldseek_info["version"],
                "foldseek_sha256": foldseek_info["sha256"],
                "foldseek_cluster_c": FOLDSEEK_CLUSTER_C,
                "threads": threads,
                "command": command,
            },
        )
        shutil.rmtree(input_dir)
        shutil.rmtree(foldseek_dir)
        shutil.rmtree(tmp_dir)
        groups_root.mkdir(parents=True, exist_ok=True)
        os.replace(stage, group_dir)
        return group, "clustered"
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def cluster_groups(
    *,
    candidates: list[Candidate],
    group_order: list[str],
    output_root: Path,
    foldseek_bin: Path,
    workers: int,
    threads_per_foldseek: int,
    requested_groups: set[str] | None,
) -> dict[str, int]:
    if workers <= 0 or threads_per_foldseek <= 0:
        raise ValueError("workers and threads-per-foldseek must be positive")
    foldseek_info = validate_foldseek(foldseek_bin)
    groups_root = output_root / "foldseek/groups"
    work_root = output_root / "foldseek/work"
    groups_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    by_group: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.group].append(candidate)
    selected_groups = [
        group
        for group in group_order
        if requested_groups is None or group in requested_groups
    ]
    if requested_groups is not None:
        missing = sorted(requested_groups - set(selected_groups))
        if missing:
            raise KeyError(f"Requested groups are absent: {missing}")
    counts: Counter[str] = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_group = {
            executor.submit(
                cluster_one_group,
                group=group,
                candidates=by_group[group],
                groups_root=groups_root,
                work_root=work_root,
                foldseek_bin=foldseek_bin,
                foldseek_info=foldseek_info,
                threads=threads_per_foldseek,
            ): group
            for group in selected_groups
        }
        completed = 0
        for future in concurrent.futures.as_completed(future_to_group):
            group, status = future.result()
            counts[status] += 1
            completed += 1
            if completed % 20 == 0 or completed == len(selected_groups):
                print(
                    f"FOLDSEEK groups={completed}/{len(selected_groups)} "
                    f"clustered={counts['clustered']} skipped={counts['skipped']}",
                    flush=True,
                )
    return dict(counts)


def state_is_better(candidate: SearchState, incumbent: SearchState | None) -> bool:
    if incumbent is None:
        return True
    if candidate.score != incumbent.score:
        return candidate.score > incumbent.score
    candidate_ids = tuple(sorted(row.joint_id for row in candidate.chosen))
    incumbent_ids = tuple(sorted(row.joint_id for row in incumbent.chosen))
    return candidate_ids < incumbent_ids


def edge_candidate_key(candidate: Candidate) -> tuple[Any, ...]:
    return (
        -(1 if candidate.current_panel == "main" else 0),
        -(1 if candidate.current_panel == "subset" else 0),
        candidate.preference_order,
        candidate.joint_id,
    )


def choose_group_candidates(
    candidates: list[Candidate], assignments: dict[str, str]
) -> tuple[list[Candidate] | None, dict[str, Any]]:
    eligible = [candidate for candidate in candidates if candidate.eligible]
    by_cluster_condition: dict[str, dict[str, list[Candidate]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate in eligible:
        cluster = assignments[candidate.joint_id]
        by_cluster_condition[cluster][candidate.condition].append(candidate)

    edges: dict[str, dict[str, Candidate]] = {}
    for cluster, by_condition in by_cluster_condition.items():
        edges[cluster] = {
            condition: sorted(rows, key=edge_candidate_key)[0]
            for condition, rows in by_condition.items()
        }

    states: dict[tuple[int, int], SearchState] = {
        (0, 0): SearchState(score=(0, 0, 0), chosen=())
    }
    for cluster in sorted(edges):
        next_states = dict(states)
        for (count0, count1), state in states.items():
            for condition_index, condition in enumerate(CONDITIONS):
                candidate = edges[cluster].get(condition)
                if candidate is None:
                    continue
                next_count0 = count0 + (1 if condition_index == 0 else 0)
                next_count1 = count1 + (1 if condition_index == 1 else 0)
                if next_count0 > SLOTS_PER_CONDITION or next_count1 > SLOTS_PER_CONDITION:
                    continue
                added_score = (
                    1 if candidate.current_panel == "main" else 0,
                    1 if candidate.current_panel == "subset" else 0,
                    -candidate.preference_order,
                )
                proposed = SearchState(
                    score=tuple(
                        left + right
                        for left, right in zip(state.score, added_score, strict=True)
                    ),
                    chosen=state.chosen + (candidate,),
                )
                key = (next_count0, next_count1)
                if state_is_better(proposed, next_states.get(key)):
                    next_states[key] = proposed
        states = next_states
    selected = states.get((SLOTS_PER_CONDITION, SLOTS_PER_CONDITION))
    diagnostics = {
        "eligible_candidates": len(eligible),
        "eligible_by_condition": {
            condition: sum(row.condition == condition for row in eligible)
            for condition in CONDITIONS
        },
        "eligible_unique_clusters_by_condition": {
            condition: len(
                {
                    assignments[row.joint_id]
                    for row in eligible
                    if row.condition == condition
                }
            )
            for condition in CONDITIONS
        },
        "eligible_joint_unique_clusters": len(by_cluster_condition),
        "common_eligible_clusters": sum(
            set(by_condition) == set(CONDITIONS)
            for by_condition in by_cluster_condition.values()
        ),
    }
    return (list(selected.chosen) if selected else None), diagnostics


def assign_slots(
    *,
    chosen: list[Candidate],
    condition: str,
    group: str,
    current_slots: dict[tuple[str, str, int], CurrentSlot],
    assignments: dict[str, str],
) -> list[tuple[int, Candidate, CurrentSlot, str]]:
    condition_rows = [row for row in chosen if row.condition == condition]
    if len(condition_rows) != SLOTS_PER_CONDITION:
        raise ValueError(f"Selection count mismatch for {condition}:{group}")
    assigned: dict[int, Candidate] = {}
    remaining: list[Candidate] = []
    for candidate in condition_rows:
        if candidate.current_panel is not None:
            assert candidate.current_slot is not None
            if candidate.current_slot in assigned:
                raise ValueError(f"Multiple selected sources retain one slot in {condition}:{group}")
            assigned[candidate.current_slot] = candidate
        else:
            remaining.append(candidate)
    remaining.sort(key=lambda row: (row.preference_order, row.joint_id))
    empty_main = [slot for slot in MAIN_SLOTS if slot not in assigned]
    empty_subset = [SUBSET_SLOT] if SUBSET_SLOT not in assigned else []
    for slot, candidate in zip(empty_main + empty_subset, remaining, strict=True):
        assigned[slot] = candidate
    if set(assigned) != set(range(SLOTS_PER_CONDITION)):
        raise ValueError(f"Incomplete slot assignment for {condition}:{group}")

    rows: list[tuple[int, Candidate, CurrentSlot, str]] = []
    for slot in range(SLOTS_PER_CONDITION):
        candidate = assigned[slot]
        current = current_slots[(condition, group, slot)]
        if candidate.staged_id == current.staged_id:
            reason = "retained_current_main" if slot in MAIN_SLOTS else "retained_current_subset"
        elif slot in MAIN_SLOTS:
            reason = "replacement_main_minimum_churn"
        else:
            reason = "replacement_subset_minimum_churn"
        rows.append((slot, candidate, current, reason))
    clusters = [assignments[candidate.joint_id] for _, candidate, _, _ in rows]
    if len(clusters) != len(set(clusters)):
        raise ValueError(f"Within-condition cluster duplication for {condition}:{group}")
    return rows


def read_group_assignments(group_dir: Path) -> dict[str, str]:
    rows = read_table(group_dir / "cluster_assignments.tsv", delimiter="\t")
    require_columns(rows, {"joint_id", "foldseek_cluster_id"}, group_dir / "cluster_assignments.tsv")
    result: dict[str, str] = {}
    for row in rows:
        if row["joint_id"] in result:
            raise ValueError(f"Duplicate assignment in {group_dir}: {row['joint_id']}")
        result[row["joint_id"]] = row["foldseek_cluster_id"]
    return result


def validate_selection_canary(
    *,
    candidates: list[Candidate],
    groups: Sequence[str],
    current_slots: dict[tuple[str, str, int], CurrentSlot],
    output_root: Path,
) -> dict[str, Any]:
    by_group: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.group].append(candidate)
    selected_rows: list[dict[str, Any]] = []
    diagnostics_by_group: dict[str, Any] = {}
    for group in groups:
        assignments = read_group_assignments(output_root / "foldseek/groups" / group)
        expected_joint_ids = {row.joint_id for row in by_group[group]}
        if set(assignments) != expected_joint_ids:
            raise ValueError(f"Canary assignment identity mismatch for {group}")
        chosen, diagnostics = choose_group_candidates(by_group[group], assignments)
        if chosen is None:
            raise RuntimeError(f"Selection canary is infeasible for {group}: {diagnostics}")
        selected_clusters = [assignments[row.joint_id] for row in chosen]
        if len(selected_clusters) != 12 or len(set(selected_clusters)) != 12:
            raise ValueError(f"Selection canary cluster uniqueness failed for {group}")
        diagnostics_by_group[group] = diagnostics
        for condition in CONDITIONS:
            for slot, candidate, current, reason in assign_slots(
                chosen=chosen,
                condition=condition,
                group=group,
                current_slots=current_slots,
                assignments=assignments,
            ):
                selected_rows.append(
                    {
                        "group": group,
                        "condition": condition,
                        "slot": slot,
                        "current_source_staged_id": current.staged_id,
                        "selected_source_staged_id": candidate.staged_id,
                        "changed": str(candidate.staged_id != current.staged_id),
                        "rasa_value": candidate.rasa_value,
                        "foldseek_cluster_id": assignments[candidate.joint_id],
                        "selection_reason": reason,
                    }
                )
    write_table(
        output_root / "selection_canary.tsv",
        selected_rows,
        (
            "group",
            "condition",
            "slot",
            "current_source_staged_id",
            "selected_source_staged_id",
            "changed",
            "rasa_value",
            "foldseek_cluster_id",
            "selection_reason",
        ),
        delimiter="\t",
    )
    return {
        "status": "complete",
        "groups": list(groups),
        "selected_slots": len(selected_rows),
        "diagnostics": diagnostics_by_group,
    }


def select_slots(
    *,
    candidates: list[Candidate],
    group_order: list[str],
    current_slots: dict[tuple[str, str, int], CurrentSlot],
    output_root: Path,
) -> list[SelectedSlot]:
    by_group: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.group].append(candidate)
    infeasible: list[dict[str, Any]] = []
    pending: list[tuple[str, list[Candidate], dict[str, str]]] = []
    for group in group_order:
        assignments = read_group_assignments(output_root / "foldseek/groups" / group)
        expected_joint_ids = {row.joint_id for row in by_group[group]}
        if set(assignments) != expected_joint_ids:
            raise ValueError(
                f"Cluster assignment set mismatch for {group}: "
                f"missing={len(expected_joint_ids - set(assignments))} "
                f"extra={len(set(assignments) - expected_joint_ids)}"
            )
        chosen, diagnostics = choose_group_candidates(by_group[group], assignments)
        if chosen is None:
            infeasible.append({"group": group, **diagnostics})
        else:
            pending.append((group, chosen, assignments))

    selection_root = output_root / "selection"
    if infeasible:
        write_table(
            selection_root / "infeasible_groups.tsv",
            infeasible,
            (
                "group",
                "eligible_candidates",
                "eligible_by_condition",
                "eligible_unique_clusters_by_condition",
                "eligible_joint_unique_clusters",
                "common_eligible_clusters",
            ),
            delimiter="\t",
        )
        write_json(
            selection_root / "status.json",
            {
                "schema_version": 1,
                "status": "infeasible",
                "created_at_utc": utc_now(),
                "rasa_operator": "<",
                "rasa_cutoff": RASA_CUTOFF,
                "infeasible_group_count": len(infeasible),
            },
        )
        raise RuntimeError(
            f"Selection is infeasible for {len(infeasible)} groups; no dataset was materialized"
        )

    selected_slots: list[SelectedSlot] = []
    all_source_paths: set[Path] = set()
    slot_rows: list[tuple[int, Candidate, CurrentSlot, str, str]] = []
    for group, chosen, assignments in pending:
        group_clusters: list[str] = []
        for condition in CONDITIONS:
            rows = assign_slots(
                chosen=chosen,
                condition=condition,
                group=group,
                current_slots=current_slots,
                assignments=assignments,
            )
            for slot, candidate, current, reason in rows:
                cluster = assignments[candidate.joint_id]
                group_clusters.append(cluster)
                slot_rows.append((slot, candidate, current, reason, cluster))
                all_source_paths.add(Path(candidate.source_path))
        if len(group_clusters) != 12 or len(set(group_clusters)) != 12:
            raise ValueError(f"Joint cluster uniqueness failed for {group}")

    source_hashes = {path: sha256_file(path) for path in sorted(all_source_paths)}
    subset_source_indices = {
        (condition, group): source_index
        for source_index, (group, condition) in enumerate(
            (group, condition)
            for group in group_order
            for condition in COMBINED_CONDITION_ORDER
        )
    }
    for group in group_order:
        for condition in CONDITIONS:
            group_condition_rows = [
                row
                for row in slot_rows
                if row[1].group == group and row[1].condition == condition
            ]
            group_condition_rows.sort(key=lambda row: row[0])
            for slot, candidate, current, reason, cluster in group_condition_rows:
                panel = "main" if slot in MAIN_SLOTS else "subset"
                pdb_key = f"{group}_{slot}"
                runtime_pdb_key = (
                    pdb_key if panel == "main" else f"{group}_{condition}_{slot}"
                )
                source_index = (
                    current.source_index
                    if panel == "main"
                    else subset_source_indices[(condition, group)]
                )
                changed = candidate.staged_id != current.staged_id
                materialized_cif_sha256 = (
                    sha256_text(reidentified_cif_text(Path(candidate.source_path), runtime_pdb_key))
                    if changed
                    else sha256_file(Path(current.cif_path))
                )
                selected_slots.append(
                    SelectedSlot(
                        condition=condition,
                        group=group,
                        ccd=candidate.ccd,
                        length=candidate.length,
                        panel=panel,
                        slot=slot,
                        pdb_key=pdb_key,
                        runtime_pdb_key=runtime_pdb_key,
                        source_index=source_index,
                        source_staged_id=candidate.staged_id,
                        source_cif_path=candidate.source_path,
                        source_cif_sha256=source_hashes[Path(candidate.source_path)],
                        materialized_cif_sha256=materialized_cif_sha256,
                        rasa_value=candidate.rasa_value,
                        foldseek_cluster_id=cluster,
                        current_source_staged_id=current.staged_id,
                        changed=changed,
                        preference_order=candidate.preference_order,
                        selection_reason=reason,
                    )
                )
    if len(subset_source_indices) != EXPECTED_GROUPS * len(CONDITIONS):
        raise ValueError(
            f"Subset source-index count mismatch: {len(subset_source_indices)}"
        )
    return selected_slots


def selected_slot_rows(slots: list[SelectedSlot]) -> list[dict[str, Any]]:
    rows = []
    for slot in slots:
        row = asdict(slot)
        row["changed"] = str(slot.changed)
        row["source_index"] = "" if slot.source_index is None else slot.source_index
        rows.append(row)
    return rows


def read_cif_text(path: Path) -> str:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt") as handle:
            return handle.read()
    return path.read_text()


def reidentified_cif_text(source: Path, new_key: str) -> str:
    lines = read_cif_text(source).splitlines(keepends=True)
    if not lines or not lines[0].startswith("data_"):
        raise ValueError(f"Missing data_ header in {source}")
    first_newline = "\r\n" if lines[0].endswith("\r\n") else "\n"
    lines[0] = f"data_{new_key}{first_newline}"
    entry_indices = [
        index for index, line in enumerate(lines) if line.startswith("_entry.id")
    ]
    if len(entry_indices) != 1:
        raise ValueError(f"Expected one _entry.id in {source}, found {len(entry_indices)}")
    index = entry_indices[0]
    newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
    lines[index] = f"_entry.id   {new_key}{newline}"
    return "".join(lines)


def copy_or_reidentify_cif(
    *, source: Path, current: Path, destination: Path, new_key: str, changed: bool
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not changed:
        if not current.is_file():
            raise FileNotFoundError(current)
        shutil.copy2(current, destination)
        return
    destination.write_text(reidentified_cif_text(source, new_key))


def materialize_datasets(
    *,
    slots: list[SelectedSlot],
    current_slots: dict[tuple[str, str, int], CurrentSlot],
    group_order: list[str],
    role_rows_by_condition: dict[str, list[dict[str, str]]],
    output_root: Path,
) -> None:
    main_target = output_root / "denovoval_re"
    subset_target = output_root / "denovoval_subset"
    for target in (main_target, subset_target):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite dataset root: {target}")
    stage = Path(tempfile.mkdtemp(prefix=".datasets.", dir=output_root))
    stage_main = stage / "denovoval_re"
    stage_subset = stage / "denovoval_subset"
    try:
        by_identity = {
            (slot.condition, slot.group, slot.slot): slot for slot in slots
        }
        selection_provenance = str(output_root / "selection/selected_slots.tsv")
        main_rasa_rows: list[dict[str, Any]] = []
        for condition in CONDITIONS:
            current_role_by_key = unique_rows(
                role_rows_by_condition[condition],
                ("pdb_key",),
                f"current role rows for {condition}",
            )
            main_rows: list[dict[str, str]] = []
            for group in group_order:
                for slot_index in MAIN_SLOTS:
                    slot = by_identity[(condition, group, slot_index)]
                    current = current_slots[(condition, group, slot_index)]
                    if slot.pdb_key != current.pdb_key:
                        raise ValueError(f"Main slot identity drift: {condition}:{group}:{slot_index}")
                    role_row = dict(current_role_by_key[(slot.pdb_key,)])
                    if int(role_row[SOURCE_INDEX_COLUMN]) != slot.source_index:
                        raise ValueError(f"Main source-index drift: {condition}:{slot.pdb_key}")
                    main_rows.append(role_row)
                    main_rasa_rows.append(
                        {
                            "condition": condition,
                            "pdb_key": slot.pdb_key,
                            "rasa_value": slot.rasa_value,
                            "source_staged_id": slot.source_staged_id,
                            "selection_provenance": selection_provenance,
                        }
                    )
                    copy_or_reidentify_cif(
                        source=Path(slot.source_cif_path),
                        current=Path(current.cif_path),
                        destination=(
                            stage_main / condition / "cifs" / f"{slot.pdb_key}.cif"
                        ),
                        new_key=slot.pdb_key,
                        changed=slot.changed,
                    )
            csv_name = f"denovoval_{condition}_sampling_inputs.csv"
            write_table(
                stage_main / csv_name,
                main_rows,
                tuple(ROLE_COLUMNS) + (SOURCE_INDEX_COLUMN,),
                delimiter=",",
            )
            (stage_main / f"denovoval_{condition}.txt").write_text(
                "".join(f"{row['pdb_key']}\n" for row in main_rows)
            )
        write_table(
            stage_main / "selected_ligand_rasa.tsv",
            main_rasa_rows,
            (
                "condition",
                "pdb_key",
                "rasa_value",
                "source_staged_id",
                "selection_provenance",
            ),
            delimiter="\t",
        )

        subset_rows: list[dict[str, Any]] = []
        subset_role_rows_by_condition: dict[str, list[dict[str, str]]] = {
            condition: [] for condition in CONDITIONS
        }
        subset_rasa_rows: list[dict[str, Any]] = []
        subset_slots = [slot for slot in slots if slot.panel == "subset"]
        subset_by_identity = {
            (slot.condition, slot.group): slot for slot in subset_slots
        }
        current_role_by_condition = {
            condition: unique_rows(
                role_rows_by_condition[condition],
                ("pdb_key",),
                f"current role rows for {condition}",
            )
            for condition in CONDITIONS
        }
        for group in group_order:
            for condition in COMBINED_CONDITION_ORDER:
                slot = subset_by_identity[(condition, group)]
                current = current_slots[(condition, group, SUBSET_SLOT)]
                current_main_row = current_role_by_condition[condition][
                    (f"{group}_0",)
                ]
                role_row = {
                    column: (
                        slot.runtime_pdb_key
                        if column == "pdb_key"
                        else current_main_row[column]
                    )
                    for column in ROLE_COLUMNS
                }
                role_row[SOURCE_INDEX_COLUMN] = slot.source_index
                role_row[SAMPLER_COLUMN] = condition
                subset_rows.append(role_row)
                subset_role_rows_by_condition[condition].append(
                    {
                        column: (
                            slot.pdb_key if column == "pdb_key" else role_row[column]
                        )
                        for column in ROLE_COLUMNS
                    }
                )
                subset_rasa_rows.append(
                    {
                        "condition": condition,
                        "pdb_key": slot.pdb_key,
                        "rasa_value": slot.rasa_value,
                        "source_staged_id": slot.source_staged_id,
                        "selection_provenance": selection_provenance,
                    }
                )
                copy_or_reidentify_cif(
                    source=Path(slot.source_cif_path),
                    current=Path(current.cif_path),
                    destination=stage_subset / "cifs" / f"{slot.runtime_pdb_key}.cif",
                    new_key=slot.runtime_pdb_key,
                    changed=slot.changed,
                )
        write_table(
            stage_subset / "sampling_inputs_denovoval_subset_ensemble.csv",
            subset_rows,
            tuple(ROLE_COLUMNS) + (SOURCE_INDEX_COLUMN, SAMPLER_COLUMN),
            delimiter=",",
        )
        (stage_subset / "denovoval_subset.txt").write_text(
            "".join(f"{row['pdb_key']}\n" for row in subset_rows)
        )
        for condition in CONDITIONS:
            condition_rows = subset_role_rows_by_condition[condition]
            write_table(
                stage_subset / f"denovoval_{condition}_sampling_inputs.csv",
                condition_rows,
                ROLE_COLUMNS,
                delimiter=",",
            )
            (stage_subset / f"denovoval_{condition}.txt").write_text(
                "".join(f"{row['pdb_key']}\n" for row in condition_rows)
            )
        write_table(
            stage_subset / "selected_ligand_rasa.tsv",
            subset_rasa_rows,
            (
                "condition",
                "pdb_key",
                "rasa_value",
                "source_staged_id",
                "selection_provenance",
            ),
            delimiter="\t",
        )

        selection_fields = tuple(selected_slot_rows(slots)[0])
        write_table(
            stage_main / "selection.tsv",
            selected_slot_rows([slot for slot in slots if slot.panel == "main"]),
            selection_fields,
            delimiter="\t",
        )
        write_table(
            stage_subset / "selection.tsv",
            selected_slot_rows(subset_slots),
            selection_fields,
            delimiter="\t",
        )

        validate_materialized(
            main_root=stage_main,
            subset_root=stage_subset,
            slots=slots,
            current_slots=current_slots,
            group_order=group_order,
        )
        os.replace(stage_main, main_target)
        os.replace(stage_subset, subset_target)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        if stage.exists():
            stage.rmdir()


def validate_materialized(
    *,
    main_root: Path,
    subset_root: Path,
    slots: list[SelectedSlot],
    current_slots: dict[tuple[str, str, int], CurrentSlot],
    group_order: list[str],
) -> None:
    if any(not math.isfinite(slot.rasa_value) or slot.rasa_value >= RASA_CUTOFF for slot in slots):
        raise ValueError("A materialized slot violates strict RASA < 0.2")
    for condition in CONDITIONS:
        csv_path = main_root / f"denovoval_{condition}_sampling_inputs.csv"
        rows = read_table(csv_path, delimiter=",")
        keys = [row["pdb_key"] for row in rows]
        expected = [f"{group}_{slot}" for group in group_order for slot in MAIN_SLOTS]
        if keys != expected or len(set(keys)) != len(expected):
            raise ValueError(f"Main runtime key/order mismatch in {csv_path}")
        cifs = sorted(path.stem for path in (main_root / condition / "cifs").glob("*.cif"))
        if sorted(keys) != cifs:
            raise ValueError(f"Main CSV/CIF mismatch for {condition}")
        current_indices = {
            slot.pdb_key: slot.source_index
            for slot in current_slots.values()
            if slot.panel == "main" and slot.condition == condition
        }
        actual_indices = {row["pdb_key"]: int(row[SOURCE_INDEX_COLUMN]) for row in rows}
        if actual_indices != current_indices:
            raise ValueError(f"Sparse main source indices changed for {condition}")

    main_rasa_path = main_root / "selected_ligand_rasa.tsv"
    main_rasa_rows = read_table(main_rasa_path, delimiter="\t")
    expected_main_sources = {
        (slot.condition, slot.pdb_key): slot.source_staged_id
        for slot in slots
        if slot.panel == "main"
    }
    actual_main_sources = {
        (row["condition"], row["pdb_key"]): row["source_staged_id"]
        for row in main_rasa_rows
    }
    if actual_main_sources != expected_main_sources:
        raise ValueError("Main selected_ligand_rasa.tsv identity mismatch")

    subset_csv = subset_root / "sampling_inputs_denovoval_subset_ensemble.csv"
    subset_rows = read_table(subset_csv, delimiter=",")
    expected_subset_count = EXPECTED_GROUPS * len(CONDITIONS)
    if len(subset_rows) != expected_subset_count:
        raise ValueError(f"Expected {expected_subset_count} subset rows")
    subset_keys = [row["pdb_key"] for row in subset_rows]
    subset_indices = [int(row[SOURCE_INDEX_COLUMN]) for row in subset_rows]
    if len(set(subset_keys)) != expected_subset_count:
        raise ValueError("Subset runtime keys are not unique")
    if subset_indices != list(range(expected_subset_count)):
        raise ValueError("Subset source indices are not dense in runtime row order")
    subset_cifs = sorted(path.stem for path in (subset_root / "cifs").glob("*.cif"))
    if sorted(subset_keys) != subset_cifs:
        raise ValueError("Subset CSV/CIF identity mismatch")
    if Counter(row[SAMPLER_COLUMN] for row in subset_rows) != {
        condition: EXPECTED_GROUPS for condition in CONDITIONS
    }:
        raise ValueError("Subset sampler counts are incorrect")
    for condition in CONDITIONS:
        condition_csv = subset_root / f"denovoval_{condition}_sampling_inputs.csv"
        condition_rows = read_table(condition_csv, delimiter=",")
        expected_keys = [f"{group}_{SUBSET_SLOT}" for group in group_order]
        if [row["pdb_key"] for row in condition_rows] != expected_keys:
            raise ValueError(f"Subset condition key/order mismatch in {condition_csv}")
    subset_rasa_rows = read_table(subset_root / "selected_ligand_rasa.tsv", delimiter="\t")
    expected_subset_sources = {
        (slot.condition, slot.pdb_key): slot.source_staged_id
        for slot in slots
        if slot.panel == "subset"
    }
    actual_subset_sources = {
        (row["condition"], row["pdb_key"]): row["source_staged_id"]
        for row in subset_rasa_rows
    }
    if actual_subset_sources != expected_subset_sources:
        raise ValueError("Subset selected_ligand_rasa.tsv identity mismatch")

    for slot in slots:
        if slot.panel == "main":
            installed = main_root / slot.condition / "cifs" / f"{slot.pdb_key}.cif"
        else:
            installed = subset_root / "cifs" / f"{slot.runtime_pdb_key}.cif"
        if sha256_file(installed) != slot.materialized_cif_sha256:
            raise ValueError(f"Materialized CIF hash mismatch: {installed}")
        if slot.changed:
            continue
        current = current_slots[(slot.condition, slot.group, slot.slot)]
        if sha256_file(installed) != sha256_file(Path(current.cif_path)):
            raise ValueError(f"Unchanged CIF is not byte-identical: {installed}")


def write_selection_outputs(
    *, slots: list[SelectedSlot], output_root: Path
) -> dict[str, Any]:
    selection_root = output_root / "selection"
    selection_root.mkdir(parents=True, exist_ok=True)
    rows = selected_slot_rows(slots)
    fields = tuple(rows[0])
    write_table(selection_root / "selected_slots.tsv", rows, fields, delimiter="\t")
    replacements = [slot for slot in slots if slot.panel == "main" and slot.changed]
    ss1p0_replacement_count = sum(
        slot.condition == "ss1p0_gamma1p0" for slot in replacements
    )
    ss1p0_generation_input_conditions = ss1p0_replacement_count * 5 + EXPECTED_GROUPS
    replacement_rows = [
        {
            "condition": slot.condition,
            "ccd": slot.ccd,
            "length": slot.length,
            "pdb_key": slot.pdb_key,
            SOURCE_INDEX_COLUMN: slot.source_index,
            "old_source_staged_id": slot.current_source_staged_id,
            "new_source_staged_id": slot.source_staged_id,
            "source_cif_path": slot.source_cif_path,
            "source_cif_sha256": slot.source_cif_sha256,
            "materialized_cif_sha256": slot.materialized_cif_sha256,
            "partial_t_values": "1,2,3,4,5",
            "num_samples_per_partial_t": 32,
        }
        for slot in replacements
    ]
    write_table(
        selection_root / "main_ensemble_replacements.tsv",
        replacement_rows,
        (
            "condition",
            "ccd",
            "length",
            "pdb_key",
            SOURCE_INDEX_COLUMN,
            "old_source_staged_id",
            "new_source_staged_id",
            "source_cif_path",
            "source_cif_sha256",
            "materialized_cif_sha256",
            "partial_t_values",
            "num_samples_per_partial_t",
        ),
        delimiter="\t",
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": utc_now(),
        "selection_contract": {
            "group_keys": ["ccd", "length"],
            "sampler_conditions": list(CONDITIONS),
            "main_per_condition_group": len(MAIN_SLOTS),
            "subset_per_condition_group": 1,
            "joint_unique_foldseek_cluster_per_group": True,
            "main_subset_source_disjoint": True,
            "rasa_operator": "<",
            "rasa_cutoff": RASA_CUTOFF,
            "rasa_is_ranking_signal": False,
            "objective": [
                "maximize retained main sources",
                "maximize retained subset sources",
                "minimize original curation order",
            ],
        },
        "counts": {
            "groups": EXPECTED_GROUPS,
            "selected_main": sum(slot.panel == "main" for slot in slots),
            "selected_subset": sum(slot.panel == "subset" for slot in slots),
            "main_replacements": len(replacements),
            "main_replacements_by_condition": dict(
                Counter(slot.condition for slot in replacements)
            ),
            "subset_replacements": sum(
                slot.panel == "subset" and slot.changed for slot in slots
            ),
            "subset_total_partialt2_generated_cifs": EXPECTED_GROUPS
            * len(CONDITIONS)
            * 32,
            "main_replacement_generated_cifs": len(replacements) * 5 * 32,
            "ss1p0_generation_input_conditions": ss1p0_generation_input_conditions,
            "ss1p0_direct32_generated_cifs": ss1p0_generation_input_conditions * 32,
            "ss1p0_fallback128_generated_cifs": ss1p0_generation_input_conditions * 128,
            "ss1p0_fallback128_retained_cifs": ss1p0_generation_input_conditions * 32,
            "ss1p0_fallback128_discarded_cifs": ss1p0_generation_input_conditions * 96,
        },
    }
    write_json(selection_root / "status.json", summary)
    return summary


def run(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (
        current_slots,
        _,
        continuation_ranks,
        group_order,
        role_rows_by_condition,
    ) = current_state(args.current_main_root, args.current_subset_root)
    candidates = load_candidates(
        ss1p0_audit=args.ss1p0_audit,
        ss1p0_raw_root=args.ss1p0_raw_root,
        ss1p6_audit=args.ss1p6_audit,
        ss1p6_raw_root=args.ss1p6_raw_root,
        current_slots=current_slots,
        continuation_ranks=continuation_ranks,
        group_order=group_order,
        rasa_override_manifest=args.rasa_override_manifest,
        rasa_override_results=args.rasa_override_results,
    )
    candidate_fields = tuple(candidate_manifest_rows(candidates)[0])
    write_table(
        output_root / "candidate_manifest.tsv",
        candidate_manifest_rows(candidates),
        candidate_fields,
        delimiter="\t",
    )
    requested_groups = set(args.groups) if args.groups else None
    cluster_counts = cluster_groups(
        candidates=candidates,
        group_order=group_order,
        output_root=output_root,
        foldseek_bin=args.foldseek_bin,
        workers=args.workers,
        threads_per_foldseek=args.threads_per_foldseek,
        requested_groups=requested_groups,
    )
    if requested_groups is not None or args.cluster_only:
        selection_canary = (
            validate_selection_canary(
                candidates=candidates,
                groups=[group for group in group_order if group in requested_groups],
                current_slots=current_slots,
                output_root=output_root,
            )
            if requested_groups is not None
            else None
        )
        write_json(
            output_root / "cluster_status.json",
            {
                "schema_version": 1,
                "status": "complete",
                "created_at_utc": utc_now(),
                "groups": sorted(requested_groups) if requested_groups else group_order,
                "counts": cluster_counts,
                "selection_canary": selection_canary,
            },
        )
        print(f"CLUSTER_ONLY {cluster_counts}", flush=True)
        return
    slots = select_slots(
        candidates=candidates,
        group_order=group_order,
        current_slots=current_slots,
        output_root=output_root,
    )
    summary = write_selection_outputs(slots=slots, output_root=output_root)
    materialize_datasets(
        slots=slots,
        current_slots=current_slots,
        group_order=group_order,
        role_rows_by_condition=role_rows_by_condition,
        output_root=output_root,
    )
    write_json(
        output_root / "curation_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "created_at_utc": utc_now(),
            "output_root": str(output_root),
            "foldseek_bin": str(args.foldseek_bin.resolve()),
            "foldseek_version": EXPECTED_FOLDSEEK_VERSION,
            "foldseek_sha256": EXPECTED_FOLDSEEK_SHA256,
            "foldseek_cluster_c": FOLDSEEK_CLUSTER_C,
            "candidate_manifest": str(output_root / "candidate_manifest.tsv"),
            "candidate_manifest_sha256": sha256_file(output_root / "candidate_manifest.tsv"),
            "rasa_refresh": (
                {
                    "input_manifest": str(args.rasa_override_manifest.resolve()),
                    "input_manifest_sha256": sha256_file(args.rasa_override_manifest),
                    "results": str(args.rasa_override_results.resolve()),
                    "results_sha256": sha256_file(args.rasa_override_results),
                    "target_semantics": "finite heavy atoms with hetero=True and res_name=CCD",
                }
                if args.rasa_override_manifest is not None
                and args.rasa_override_results is not None
                else None
            ),
            "selection_status": summary,
            "datasets": {
                "main": str(output_root / "denovoval_re"),
                "subset": str(output_root / "denovoval_subset"),
            },
            "rfd3_submission_performed": False,
            "af3_sbatch_updated": False,
        },
    )
    print(json.dumps(summary["counts"], sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ss1p0-audit", type=Path, default=DEFAULT_SS1P0_AUDIT)
    parser.add_argument(
        "--ss1p0-raw-root", type=Path, default=DEFAULT_SS1P0_SOURCE_ROOT / "raw"
    )
    parser.add_argument("--ss1p6-audit", type=Path, default=DEFAULT_SS1P6_AUDIT)
    parser.add_argument("--ss1p6-raw-root", type=Path, default=DEFAULT_SS1P6_RAW_ROOT)
    parser.add_argument(
        "--current-main-root", type=Path, default=DEFAULT_CURRENT_MAIN_ROOT
    )
    parser.add_argument(
        "--current-subset-root", type=Path, default=DEFAULT_CURRENT_SUBSET_ROOT
    )
    parser.add_argument(
        "--rasa-override-manifest",
        type=Path,
        default=None,
        help="Input manifest mapping corrected RASA rows to baseline or extra candidates.",
    )
    parser.add_argument(
        "--rasa-override-results",
        type=Path,
        default=None,
        help="Corrected ligand_rasa.tsv paired with --rasa-override-manifest.",
    )
    parser.add_argument("--foldseek-bin", type=Path, default=DEFAULT_FOLDSEEK_BIN)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-foldseek", type=int, default=4)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Cluster only these exact CCD_lenL groups; implies cluster-only.",
    )
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="Stop after clustering without selection or materialization.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
