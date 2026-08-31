"""Validated loading and deterministic selection for cached design bundles."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from allatom_design.eval.sampling.sequence_design.outputs import (
    load_sample_dict_bundle,
    sample_bundle_paths,
)


DESIGNED_INDEXED_KEYS = (
    "designed_sample_id",
    "designed_sample_atom_array",
    "designed_sample_seq",
    "designed_sample_path",
    "designed_sample_path_for_af3_tc",
    "caliby_raw_output_path",
)


def load_validated_bundle(
    *, source_dir: Path, csv_suffix: str = ""
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a sample bundle through its completion-marker contract."""
    _, marker_path = sample_bundle_paths(
        log_dir_per_ckpt=source_dir,
        csv_suffix=csv_suffix,
    )
    if not marker_path.is_file():
        raise FileNotFoundError(f"Missing sample completion marker: {marker_path}")
    with marker_path.open() as handle:
        marker = json.load(handle)
    ckpt_info = marker.get("ckpt_info")
    if not isinstance(ckpt_info, dict):
        raise ValueError(f"Missing ckpt_info in sample marker: {marker_path}")
    bundle = load_sample_dict_bundle(
        log_dir_per_ckpt=source_dir,
        ckpt_info=ckpt_info,
        csv_suffix=csv_suffix,
    )
    return bundle, marker


def validate_bundle_shape(
    sample_dict: dict[str, dict[str, Any]],
    *,
    expected_input_count: int,
    expected_sequences_per_input: int,
) -> None:
    """Validate cached-design cardinality and per-design field alignment."""
    if len(sample_dict) != expected_input_count:
        raise ValueError(
            f"Expected {expected_input_count} input samples, found {len(sample_dict)}"
        )
    invalid: list[str] = []
    designed_ids: list[str] = []
    for input_sample_id, entry in sample_dict.items():
        ids = entry.get("designed_sample_id")
        atom_arrays = entry.get("designed_sample_atom_array")
        if (
            not isinstance(ids, list)
            or len(ids) != expected_sequences_per_input
            or not isinstance(atom_arrays, list)
            or len(atom_arrays) != expected_sequences_per_input
        ):
            invalid.append(str(input_sample_id))
            continue
        for key in DESIGNED_INDEXED_KEYS:
            if key not in entry:
                continue
            value = entry[key]
            if not isinstance(value, list) or len(value) != len(ids):
                invalid.append(f"{input_sample_id}:{key}")
        designed_ids.extend(str(value) for value in ids)
    if invalid:
        raise ValueError(
            "Bundle entries do not match the expected sequence count; first inputs: "
            f"{invalid[:10]}"
        )
    if len(designed_ids) != len(set(designed_ids)):
        raise ValueError("designed_sample_id values are not unique within the bundle")


def input_ids_for_chunk(
    input_sample_ids: list[str], *, chunk_index: int, num_chunks: int
) -> list[str]:
    """Return one stable, contiguous chunk of sorted input IDs."""
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(
            f"chunk_index must be in [0, {num_chunks}), got {chunk_index}"
        )
    ordered_ids = sorted(str(value) for value in input_sample_ids)
    chunk_size = math.ceil(len(ordered_ids) / num_chunks) if ordered_ids else 0
    start = chunk_index * chunk_size
    return ordered_ids[start : min(start + chunk_size, len(ordered_ids))]


def _slice_entry_designs(
    entry: dict[str, Any], *, indices: list[int], input_sample_id: str
) -> dict[str, Any]:
    selected = copy.deepcopy(entry)
    n_available = len(entry["designed_sample_id"])
    for key in DESIGNED_INDEXED_KEYS:
        if key not in selected:
            continue
        value = selected[key]
        if not isinstance(value, list) or len(value) != n_available:
            raise ValueError(
                f"Per-design field {key!r} is not aligned for {input_sample_id}"
            )
        selected[key] = [value[index] for index in indices]
    return selected


def subset_bundle(
    sample_dict: dict[str, dict[str, Any]],
    *,
    selected_input_ids: list[str],
    max_designed_sequences: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Copy selected inputs and optionally retain only the first N designs."""
    selected = {
        input_sample_id: copy.deepcopy(sample_dict[input_sample_id])
        for input_sample_id in selected_input_ids
    }
    if max_designed_sequences is None:
        return selected
    if max_designed_sequences <= 0:
        raise ValueError(
            "max_designed_sequences must be positive when provided, got "
            f"{max_designed_sequences}"
        )

    remaining = max_designed_sequences
    limited: dict[str, dict[str, Any]] = {}
    for input_sample_id, entry in selected.items():
        if remaining <= 0:
            break
        keep = min(remaining, len(entry["designed_sample_id"]))
        limited[input_sample_id] = _slice_entry_designs(
            entry,
            indices=list(range(keep)),
            input_sample_id=input_sample_id,
        )
        remaining -= keep
    if remaining > 0:
        raise ValueError(
            f"Requested {max_designed_sequences} designs, but only "
            f"{max_designed_sequences - remaining} were available"
        )
    return limited


def select_designed_samples(
    sample_dict: dict[str, dict[str, Any]],
    *,
    designed_sample_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Select exact designed sample IDs in the requested order."""
    requested = list(dict.fromkeys(map(str, designed_sample_ids)))
    if len(requested) != len(designed_sample_ids):
        raise ValueError("designed_sample_ids contains duplicates")

    locations: dict[str, tuple[str, int]] = {}
    for input_sample_id, entry in sample_dict.items():
        for index, designed_sample_id in enumerate(entry["designed_sample_id"]):
            locations[str(designed_sample_id)] = (str(input_sample_id), index)
    missing = [sample_id for sample_id in requested if sample_id not in locations]
    if missing:
        raise ValueError(f"Requested designed sample IDs are absent: {missing}")

    indices_by_input: dict[str, list[int]] = {}
    input_order: list[str] = []
    for sample_id in requested:
        input_sample_id, index = locations[sample_id]
        if input_sample_id not in indices_by_input:
            input_order.append(input_sample_id)
            indices_by_input[input_sample_id] = []
        indices_by_input[input_sample_id].append(index)

    return {
        input_sample_id: _slice_entry_designs(
            sample_dict[input_sample_id],
            indices=indices_by_input[input_sample_id],
            input_sample_id=input_sample_id,
        )
        for input_sample_id in input_order
    }


def designed_sample_ids(sample_dict: dict[str, dict[str, Any]]) -> list[str]:
    """Return designed sample IDs in bundle iteration order."""
    return [
        str(designed_sample_id)
        for entry in sample_dict.values()
        for designed_sample_id in entry["designed_sample_id"]
    ]
