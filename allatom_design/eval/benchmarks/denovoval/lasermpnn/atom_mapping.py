"""Atom-identity sidecars for stock-NISE PDB transport and CIF backmapping."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from biotite.structure import AtomArray


ATOM_MAPPING_COLUMNS = (
    "sample_id",
    "transport_atom_index",
    "transport_atom_serial",
    "transport_atom_name",
    "transport_element",
    "transport_chain_id",
    "transport_res_id",
    "transport_res_name",
    "atom_role",
    "source_atom_index",
    "source_atom_serial",
    "source_atom_name",
    "source_element",
    "source_chain_id",
    "source_res_id",
    "source_res_name",
    "parent_transport_atom_index",
    "parent_source_atom_name",
)


def _validate_rows(rows: list[dict[str, Any]], path: Path | None = None) -> None:
    location = f" in {path}" if path is not None else ""
    if not rows:
        raise ValueError(f"Atom mapping is empty{location}")
    missing = sorted(set(ATOM_MAPPING_COLUMNS) - set(rows[0]))
    if missing:
        raise ValueError(f"Atom mapping is missing columns {missing}{location}")
    indices = [int(row["transport_atom_index"]) for row in rows]
    if indices != list(range(1, len(rows) + 1)):
        raise ValueError(f"Transport atom indices are not exactly 1..N{location}")
    serials = [int(row["transport_atom_serial"]) for row in rows]
    if len(serials) != len(set(serials)):
        raise ValueError(f"Transport atom serials are not unique{location}")
    heavy = [row for row in rows if row["atom_role"] == "source_heavy"]
    hydrogens = [row for row in rows if row["atom_role"] == "generated_hydrogen"]
    if len(heavy) + len(hydrogens) != len(rows):
        raise ValueError(f"Atom mapping has an unknown atom role{location}")
    for field in (
        "sample_id",
        "transport_chain_id",
        "transport_res_id",
        "transport_res_name",
        "source_chain_id",
        "source_res_id",
        "source_res_name",
    ):
        values = {str(row[field]) for row in rows}
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"Atom mapping has inconsistent {field}{location}")
    source_names = [str(row["source_atom_name"]) for row in heavy]
    if not all(source_names) or len(source_names) != len(set(source_names)):
        raise ValueError(f"Source heavy atom names are blank or duplicated{location}")
    heavy_indices = {int(row["transport_atom_index"]) for row in heavy}
    for row in hydrogens:
        if int(row["parent_transport_atom_index"]) not in heavy_indices:
            raise ValueError(f"Generated H has no mapped heavy parent{location}")
        if not str(row["parent_source_atom_name"]):
            raise ValueError(f"Generated H has no source parent name{location}")
        parent_names = {
            str(heavy_row["source_atom_name"])
            for heavy_row in heavy
            if int(heavy_row["transport_atom_index"])
            == int(row["parent_transport_atom_index"])
        }
        if parent_names != {str(row["parent_source_atom_name"])}:
            raise ValueError(f"Generated H parent identity is inconsistent{location}")


def write_atom_mapping(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ATOM_MAPPING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_atom_mapping(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(ATOM_MAPPING_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing atom-mapping columns {missing}")
        rows = list(reader)
    _validate_rows(rows, path)
    return rows


def _replace_string_annotation(
    structure: AtomArray, annotation: str, values: Iterable[str]
) -> None:
    values = [str(value) for value in values]
    width = max(1, *(len(value) for value in values))
    if annotation in structure.get_annotation_categories():
        structure.del_annotation(annotation)
    structure.set_annotation(annotation, np.asarray(values, dtype=f"U{width}"))


def restore_transport_ligand_metadata(
    sampled_structure: AtomArray,
    mapping_rows: list[dict[str, str]],
    *,
    transport_chain_id: str,
    canonical_chain_id: str,
    pdb_ccd_code: str,
) -> AtomArray:
    """Restore canonical heavy identity while retaining generated hydrogens."""
    _validate_rows(mapping_rows)
    structure = sampled_structure.copy()
    ligand_indices = np.flatnonzero(
        structure.chain_id.astype(str) == transport_chain_id
    )
    if len(ligand_indices) != len(mapping_rows):
        raise ValueError(
            "Sampled transport ligand atom count differs from atom mapping: "
            f"{len(ligand_indices)} != {len(mapping_rows)}"
        )
    source_chain_ids = {str(row["source_chain_id"]) for row in mapping_rows}
    source_res_ids = {int(row["source_res_id"]) for row in mapping_rows}
    if source_chain_ids != {canonical_chain_id} or len(source_res_ids) != 1:
        raise ValueError("Atom mapping does not identify one canonical ligand residue")

    atom_names = structure.atom_name.astype(str).tolist()
    chain_ids = structure.chain_id.astype(str).tolist()
    res_names = structure.res_name.astype(str).tolist()
    res_ids = np.asarray(structure.res_id, dtype=int).copy()
    used_names = {
        str(row["source_atom_name"])
        for row in mapping_rows
        if row["atom_role"] == "source_heavy"
    }
    hydrogen_counter = 1
    for atom_index, row in zip(ligand_indices, mapping_rows, strict=True):
        observed_name = str(structure.atom_name[atom_index])
        observed_element = str(structure.element[atom_index]).upper()
        observed_res_name = str(structure.res_name[atom_index])
        observed_res_id = int(structure.res_id[atom_index])
        if str(row["transport_chain_id"]) != transport_chain_id:
            raise ValueError("Atom mapping transport chain differs from the requested chain")
        if observed_res_name != str(row["transport_res_name"]):
            raise ValueError(
                f"Transport residue name mismatch: {observed_res_name} != "
                f"{row['transport_res_name']}"
            )
        if observed_res_id != int(row["transport_res_id"]):
            raise ValueError(
                f"Transport residue ID mismatch: {observed_res_id} != "
                f"{row['transport_res_id']}"
            )
        if observed_name != str(row["transport_atom_name"]):
            raise ValueError(
                f"Transport atom name mismatch at index {row['transport_atom_index']}: "
                f"{observed_name} != {row['transport_atom_name']}"
            )
        if observed_element != str(row["transport_element"]).upper():
            raise ValueError(
                f"Transport element mismatch at index {row['transport_atom_index']}: "
                f"{observed_element} != {row['transport_element']}"
            )
        if row["atom_role"] == "source_heavy":
            restored_name = str(row["source_atom_name"])
        else:
            while True:
                restored_name = f"H{hydrogen_counter:03d}"
                hydrogen_counter += 1
                if restored_name not in used_names:
                    break
            used_names.add(restored_name)
        atom_names[atom_index] = restored_name
        chain_ids[atom_index] = canonical_chain_id
        res_names[atom_index] = pdb_ccd_code
        res_ids[atom_index] = next(iter(source_res_ids))

    _replace_string_annotation(structure, "atom_name", atom_names)
    _replace_string_annotation(structure, "chain_id", chain_ids)
    _replace_string_annotation(structure, "res_name", res_names)
    structure.res_id = res_ids
    return structure
