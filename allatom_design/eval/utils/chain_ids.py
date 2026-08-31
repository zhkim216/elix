from __future__ import annotations

import string
from collections.abc import Iterable, Mapping

import numpy as np
from biotite.structure import AtomArray


SINGLE_CHARACTER_CHAIN_ID_CANDIDATES = (
    string.ascii_uppercase + string.ascii_lowercase + string.digits
)


def allocate_single_character_chain_ids(
    source_chain_ids: Iterable[str],
) -> dict[str, str]:
    """Allocate deterministic, collision-free one-character chain IDs."""
    source_chain_ids = list(dict.fromkeys(map(str, source_chain_ids)))
    if not source_chain_ids:
        raise ValueError("Cannot allocate chain IDs for an empty structure")

    kept = {
        chain_id
        for chain_id in source_chain_ids
        if len(chain_id) == 1
        and chain_id in SINGLE_CHARACTER_CHAIN_ID_CANDIDATES
    }
    used = set(kept)
    mapping: dict[str, str] = {}
    for chain_id in source_chain_ids:
        if chain_id in kept:
            mapping[chain_id] = chain_id
            continue
        replacement = next(
            (
                candidate
                for candidate in SINGLE_CHARACTER_CHAIN_ID_CANDIDATES
                if candidate not in used
            ),
            None,
        )
        if replacement is None:
            raise ValueError(
                f"Exhausted single-character chain IDs for {source_chain_ids}"
            )
        mapping[chain_id] = replacement
        used.add(replacement)
    return mapping


def _replace_string_annotation(
    structure: AtomArray,
    annotation: str,
    values: Iterable[str],
) -> None:
    values = [str(value) for value in values]
    width = max(1, *(len(value) for value in values))
    if annotation in structure.get_annotation_categories():
        structure.del_annotation(annotation)
    structure.set_annotation(annotation, np.asarray(values, dtype=f"U{width}"))


def remap_atom_array_chain_identity(
    source_structure: AtomArray,
    chain_id_mapping: Mapping[str, str],
    *,
    transformation_id: str = "1",
) -> AtomArray:
    """Return a copy with consistent staged chain and PN-unit annotations."""
    mapping = {
        str(source_chain_id): str(staged_chain_id)
        for source_chain_id, staged_chain_id in chain_id_mapping.items()
    }
    if not mapping:
        raise ValueError("chain_id_mapping must not be empty")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"Chain mapping is not one-to-one: {mapping}")
    invalid_targets = sorted(
        staged_chain_id
        for staged_chain_id in mapping.values()
        if len(staged_chain_id) != 1
        or staged_chain_id not in SINGLE_CHARACTER_CHAIN_ID_CANDIDATES
    )
    if invalid_targets:
        raise ValueError(
            f"Staged chain IDs must be single-character IDs: {invalid_targets}"
        )

    source_chain_values = source_structure.chain_id.astype(str)
    missing = sorted(set(source_chain_values) - set(mapping))
    if missing:
        raise ValueError(f"Chain mapping does not cover source chains: {missing}")

    staged_chain_values = [
        mapping[source_chain_id] for source_chain_id in source_chain_values
    ]
    staged_pn_unit_iids = [
        f"{staged_chain_id}_{transformation_id}"
        for staged_chain_id in staged_chain_values
    ]

    structure = source_structure.copy()
    for annotation in ("chain_id", "auth_asym_id", "chain_iid", "pn_unit_id"):
        _replace_string_annotation(structure, annotation, staged_chain_values)
    _replace_string_annotation(structure, "pn_unit_iid", staged_pn_unit_iids)
    _replace_string_annotation(
        structure,
        "transformation_id",
        [transformation_id] * len(structure),
    )
    return structure


__all__ = [
    "SINGLE_CHARACTER_CHAIN_ID_CANDIDATES",
    "allocate_single_character_chain_ids",
    "remap_atom_array_chain_identity",
]
