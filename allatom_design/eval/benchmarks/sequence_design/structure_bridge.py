from __future__ import annotations

import csv
import hashlib
import os
import string
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack, get_residue_starts
from biotite.structure.io import pdb as pdbio
from biotite.structure.io import pdbx
from omegaconf import OmegaConf

from atomworks.constants import STANDARD_AA
from atomworks.enums import ChainType
from atomworks.io.utils.io_utils import get_structure, read_any
from atomworks.io.utils.sequence import get_1_from_3_letter_code

from allatom_design.utils.sample_io_utils import save_cif_file


DEFAULT_EXTRA_FIELDS = [
    "atom_id",
    "auth_asym_id",
    "auth_seq_id",
    "occupancy",
    "b_factor",
    "charge",
]
BASE36 = string.digits + string.ascii_uppercase
STANDARD_AA_CODES = frozenset(map(str, STANDARD_AA))


@dataclass(frozen=True)
class CcdAlias:
    original_ccd_code: str
    pdb_ccd_code: str
    reason: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _alias_candidates() -> Iterable[str]:
    for prefix in "XYZQ":
        for first in BASE36:
            for second in BASE36:
                yield f"{prefix}{first}{second}"


def allocate_ccd_aliases(ccd_codes: Iterable[str]) -> list[CcdAlias]:
    """Allocate deterministic PDB-safe aliases for long and amino-acid ligands."""
    normalized = sorted({str(code).strip().upper() for code in ccd_codes})
    if any(not code for code in normalized):
        raise ValueError("CCD codes must be non-empty")
    needs_alias = [
        code for code in normalized if len(code) > 3 or code in STANDARD_AA_CODES
    ]
    reserved = set(normalized) | set(STANDARD_AA_CODES)
    candidates = (candidate for candidate in _alias_candidates() if candidate not in reserved)
    aliases: list[CcdAlias] = []
    for code in needs_alias:
        try:
            alias = next(candidates)
        except StopIteration as exc:
            raise ValueError("Exhausted collision-free three-character CCD aliases") from exc
        reserved.add(alias)
        reason = "pdb_res_name_too_long" if len(code) > 3 else "amino_acid_ligand"
        aliases.append(CcdAlias(code, alias, reason))
    return aliases


def write_ccd_aliases(path: str | Path, aliases: Iterable[CcdAlias]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_ccd_code", "pdb_ccd_code", "reason"],
        )
        writer.writeheader()
        for alias in aliases:
            writer.writerow(asdict(alias))
    os.replace(tmp_path, path)


def load_ccd_aliases(path: str | Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"original_ccd_code", "pdb_ccd_code"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        mapping = {
            str(row["original_ccd_code"]): str(row["pdb_ccd_code"])
            for row in reader
        }
    if len(mapping) != len(set(mapping.values())):
        raise ValueError(f"CCD alias mapping is not one-to-one: {path}")
    return mapping


def read_structure(path: str | Path) -> AtomArray:
    """Read one structure through AtomWorks without synthesizing a bond graph."""
    file_obj = read_any(path)
    extra_fields = DEFAULT_EXTRA_FIELDS
    if isinstance(file_obj, pdbio.PDBFile):
        # PDB has no label/auth split; Biotite rejects mmCIF-only field names.
        extra_fields = [
            field
            for field in DEFAULT_EXTRA_FIELDS
            if field not in {"auth_asym_id", "auth_seq_id"}
        ]
    structure = get_structure(
        file_obj,
        extra_fields=extra_fields,
        include_bonds=False,
        model=1,
        altloc="first",
    )
    if isinstance(structure, AtomArrayStack):
        if structure.stack_depth() != 1:
            raise ValueError(f"Expected one model in {path}, got {structure.stack_depth()}")
        structure = structure[0]
    return structure


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


def prepare_pdb_structure(
    source_structure: AtomArray,
    *,
    ligand_chain_id: str,
    original_ccd_code: str,
    pdb_ccd_code: str,
) -> AtomArray:
    structure = source_structure.copy()
    ligand_mask = structure.chain_id.astype(str) == ligand_chain_id
    if not np.any(ligand_mask):
        raise ValueError(f"No ligand atoms found in chain {ligand_chain_id!r}")
    observed = sorted(set(map(str, structure.res_name[ligand_mask])))
    if observed != [original_ccd_code]:
        raise ValueError(
            f"Ligand chain {ligand_chain_id!r} has CCD codes {observed}, "
            f"expected {[original_ccd_code]}"
        )
    if not np.all(structure.hetero[ligand_mask]):
        raise ValueError(f"Ligand chain {ligand_chain_id!r} contains non-HETATM atoms")
    res_names = structure.res_name.astype(str).tolist()
    for index in np.flatnonzero(ligand_mask):
        res_names[index] = pdb_ccd_code
    _replace_string_annotation(structure, "res_name", res_names)
    return structure


def write_pdb(path: str | Path, structure: AtomArray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    pdb_file = pdbio.PDBFile()
    pdb_file.set_structure(structure)
    pdb_file.write(tmp_path)
    os.replace(tmp_path, path)


def _annotation_or_default(
    structure: AtomArray,
    annotation: str,
    default: float,
) -> np.ndarray:
    if annotation in structure.get_annotation_categories():
        return np.asarray(getattr(structure, annotation), dtype=float)
    return np.full(len(structure), default, dtype=float)


def validate_pdb_roundtrip(expected: AtomArray, observed: AtomArray) -> dict[str, object]:
    errors: list[str] = []
    if len(expected) != len(observed):
        errors.append(f"atom_count:{len(expected)}!={len(observed)}")
    n_compare = min(len(expected), len(observed))
    categorical = ["chain_id", "res_id", "res_name", "atom_name", "element", "hetero"]
    for annotation in ("ins_code", "atom_id"):
        if (
            annotation in expected.get_annotation_categories()
            and annotation in observed.get_annotation_categories()
        ):
            categorical.append(annotation)
    mismatch_counts: dict[str, int] = {}
    for annotation in categorical:
        left = np.asarray(getattr(expected, annotation))[:n_compare]
        right = np.asarray(getattr(observed, annotation))[:n_compare]
        mismatches = int(np.sum(left.astype(str) != right.astype(str)))
        mismatch_counts[annotation] = mismatches
        if mismatches:
            errors.append(f"{annotation}_mismatches:{mismatches}")

    coordinate_max_abs_error = (
        float(np.max(np.abs(expected.coord[:n_compare] - observed.coord[:n_compare])))
        if n_compare
        else 0.0
    )
    if coordinate_max_abs_error > 0.00051:
        errors.append(f"coordinate_max_abs_error:{coordinate_max_abs_error:.8g}")

    numeric_max_abs_error: dict[str, float] = {}
    for annotation, default, tolerance in (
        ("occupancy", 1.0, 0.0051),
        ("b_factor", 0.0, 0.0051),
        ("charge", 0.0, 0.0),
    ):
        left = _annotation_or_default(expected, annotation, default)[:n_compare]
        right = _annotation_or_default(observed, annotation, default)[:n_compare]
        left_finite = np.isfinite(left)
        right_finite = np.isfinite(right)
        finite = left_finite & right_finite
        finite_mismatches = int(np.sum(left_finite != right_finite))
        if finite_mismatches:
            errors.append(f"{annotation}_finite_mismatches:{finite_mismatches}")
        max_error = float(np.max(np.abs(left[finite] - right[finite]))) if np.any(finite) else 0.0
        numeric_max_abs_error[annotation] = max_error
        if max_error > tolerance:
            errors.append(f"{annotation}_max_abs_error:{max_error:.8g}")

    return {
        "ok": not errors,
        "errors": errors,
        "n_expected_atoms": len(expected),
        "n_observed_atoms": len(observed),
        "categorical_mismatch_counts": mismatch_counts,
        "coordinate_max_abs_error": coordinate_max_abs_error,
        "numeric_max_abs_error": numeric_max_abs_error,
    }


def restore_ligand_ccd_and_roles(
    sampled_structure: AtomArray,
    *,
    protein_chain_id: str,
    ligand_chain_id: str,
    pdb_ccd_code: str,
    original_ccd_code: str,
) -> AtomArray:
    structure = sampled_structure.copy()
    protein_mask = structure.chain_id.astype(str) == protein_chain_id
    ligand_mask = structure.chain_id.astype(str) == ligand_chain_id
    unexpected_chains = sorted(
        set(map(str, structure.chain_id)) - {protein_chain_id, ligand_chain_id}
    )
    if unexpected_chains:
        raise ValueError(f"Unexpected chains in sampled PDB: {unexpected_chains}")
    if not np.any(protein_mask) or not np.any(ligand_mask):
        raise ValueError(
            f"Expected protein {protein_chain_id!r} and ligand {ligand_chain_id!r} chains"
        )
    observed_ligand_codes = sorted(set(map(str, structure.res_name[ligand_mask])))
    if observed_ligand_codes != [pdb_ccd_code]:
        raise ValueError(
            f"Sampled ligand CCD codes {observed_ligand_codes} do not match "
            f"expected alias {[pdb_ccd_code]}"
        )
    if np.any(structure.hetero[protein_mask]) or not np.all(structure.hetero[ligand_mask]):
        raise ValueError("Sampled PDB ATOM/HETATM roles do not match protein/ligand chains")

    res_names = structure.res_name.astype(str).tolist()
    for index in np.flatnonzero(ligand_mask):
        res_names[index] = original_ccd_code
    _replace_string_annotation(structure, "res_name", res_names)

    structure.set_annotation("is_polymer", protein_mask.astype(bool))
    structure.set_annotation(
        "chain_entity",
        np.where(protein_mask, 0, 1).astype(np.int32),
    )
    structure.set_annotation(
        "chain_type",
        np.where(
            protein_mask,
            int(ChainType.POLYPEPTIDE_L),
            int(ChainType.NON_POLYMER),
        ).astype(np.int8),
    )
    _replace_string_annotation(structure, "chain_iid", structure.chain_id.astype(str))
    return structure


def write_semantic_cif(path: str | Path, structure: AtomArray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp.{os.getpid()}.cif")
    cfg = OmegaConf.create(
        {
            "file_type": "cif",
            "id": path.stem,
            "date": "1959-01-07",
            "include_entity_poly": True,
            "include_entity_nonpoly": True,
            "include_nan_coords": False,
            "include_bonds": False,
            "extra_fields": [],
            "exclude_field_keys": ["token_id", "is_ligand_pocket"],
        }
    )
    try:
        save_cif_file(structure, tmp_path, cif_save_cfg=cfg)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def validate_semantic_cif_entities(
    path: str | Path,
    *,
    protein_chain_id: str,
    ligand_chain_id: str,
    original_ccd_code: str,
    pdb_ccd_code: str,
) -> dict[str, object]:
    """Validate the raw mmCIF entity relationships used by AtomWorks and AF3.

    Entity identifiers are intentionally compared relationally: AtomWorks may
    serialize them as either zero- or one-based values, but every category must
    agree on the same mapping.
    """
    path = Path(path)
    block = pdbx.CIFFile.read(str(path)).block
    required_categories = {
        "atom_site",
        "struct_asym",
        "entity",
        "entity_poly",
        "pdbx_entity_nonpoly",
    }
    missing_categories = sorted(required_categories - set(block.keys()))
    if missing_categories:
        raise ValueError(
            f"Semantic CIF is missing required categories {missing_categories}: {path}"
        )

    def column(category: str, field: str) -> np.ndarray:
        if field not in block[category]:
            raise ValueError(f"Semantic CIF is missing _{category}.{field}: {path}")
        return np.asarray(block[category][field].as_array(str), dtype=str)

    atom_chain_ids = column("atom_site", "label_asym_id")
    atom_entity_ids = column("atom_site", "label_entity_id")
    atom_comp_ids = column("atom_site", "label_comp_id")
    expected_chains = {protein_chain_id, ligand_chain_id}
    observed_chains = set(atom_chain_ids)
    if observed_chains != expected_chains:
        raise ValueError(
            f"Semantic CIF chains {sorted(observed_chains)} != {sorted(expected_chains)}"
        )

    atom_chain_to_entity: dict[str, str] = {}
    for chain_id in expected_chains:
        entity_ids = set(atom_entity_ids[atom_chain_ids == chain_id])
        if len(entity_ids) != 1:
            raise ValueError(
                f"Chain {chain_id!r} maps to atom_site entities {sorted(entity_ids)}"
            )
        atom_chain_to_entity[chain_id] = next(iter(entity_ids))

    struct_chain_ids = column("struct_asym", "id")
    struct_entity_ids = column("struct_asym", "entity_id")
    struct_chain_to_entity = dict(zip(struct_chain_ids, struct_entity_ids, strict=True))
    if struct_chain_to_entity != atom_chain_to_entity:
        raise ValueError(
            "atom_site/struct_asym entity maps differ: "
            f"{atom_chain_to_entity} != {struct_chain_to_entity}"
        )

    entity_ids = column("entity", "id")
    entity_types = column("entity", "type")
    entity_type_by_id = dict(zip(entity_ids, entity_types, strict=True))
    if set(entity_type_by_id) != set(atom_chain_to_entity.values()):
        raise ValueError(
            "struct_asym/entity identifiers differ: "
            f"{sorted(set(atom_chain_to_entity.values()))} != "
            f"{sorted(entity_type_by_id)}"
        )

    protein_entity = atom_chain_to_entity[protein_chain_id]
    ligand_entity = atom_chain_to_entity[ligand_chain_id]
    if entity_type_by_id[protein_entity].lower() != "polymer":
        raise ValueError(f"Protein entity {protein_entity!r} is not polymer")
    if entity_type_by_id[ligand_entity].lower() != "non-polymer":
        raise ValueError(f"Ligand entity {ligand_entity!r} is not non-polymer")

    polymer_entities = set(column("entity_poly", "entity_id"))
    if protein_entity not in polymer_entities or ligand_entity in polymer_entities:
        raise ValueError(
            f"entity_poly membership is invalid: protein={protein_entity}, "
            f"ligand={ligand_entity}, observed={sorted(polymer_entities)}"
        )

    nonpoly_entities = column("pdbx_entity_nonpoly", "entity_id")
    nonpoly_codes = column("pdbx_entity_nonpoly", "comp_id")
    nonpoly_by_entity = dict(zip(nonpoly_entities, nonpoly_codes, strict=True))
    if nonpoly_by_entity.get(ligand_entity) != original_ccd_code:
        raise ValueError(
            f"Ligand nonpoly CCD {nonpoly_by_entity.get(ligand_entity)!r} != "
            f"{original_ccd_code!r}"
        )

    ligand_comp_ids = set(atom_comp_ids[atom_chain_ids == ligand_chain_id])
    if ligand_comp_ids != {original_ccd_code}:
        raise ValueError(
            f"Ligand atom_site CCDs {sorted(ligand_comp_ids)} != {[original_ccd_code]}"
        )
    if pdb_ccd_code != original_ccd_code and pdb_ccd_code in set(atom_comp_ids):
        raise ValueError(f"PDB CCD alias {pdb_ccd_code!r} leaked into {path}")

    return {
        "ok": True,
        "protein_entity_id": protein_entity,
        "ligand_entity_id": ligand_entity,
        "ligand_ccd_code": original_ccd_code,
    }


def protein_sequence(structure: AtomArray, chain_id: str) -> str:
    chain = structure[structure.chain_id.astype(str) == chain_id]
    if len(chain) == 0:
        raise ValueError(f"No atoms found for protein chain {chain_id!r}")
    starts = get_residue_starts(chain)
    letters = [
        get_1_from_3_letter_code(
            str(code),
            ChainType.POLYPEPTIDE_L,
            use_closest_canonical=True,
        )
        for code in chain.res_name[starts]
    ]
    sequence = "".join(letters)
    if len(sequence) != len(starts) or any(len(letter) != 1 for letter in letters):
        raise ValueError(f"Could not serialize protein sequence for chain {chain_id!r}")
    return sequence


def unique_ligand_ccd(structure: AtomArray, chain_id: str) -> str:
    codes = sorted(
        set(map(str, structure.res_name[structure.chain_id.astype(str) == chain_id]))
    )
    if len(codes) != 1:
        raise ValueError(f"Expected one ligand CCD in chain {chain_id!r}, got {codes}")
    return codes[0]
