"""CCD-indexed chemistry helpers for the denovoval LASErMPNN bridge.

The PDB format is only a transport format. Canonical heavy identity and staging
bond topology come from the pinned wwPDB component mirror; protonated charge and
hydrogen topology come from the selected stock-NISE or explicit AtomWorks route.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import prody as pr
from biotite.structure.io.pdbx import CIFFile
from rdkit import Chem


BASE36 = string.digits + string.ascii_uppercase
SMILES_PRECEDENCE = (
    ("SMILES_CANONICAL", "OpenEye OEToolkits"),
)
METAL_ELEMENTS = frozenset(
    {
        "LI", "NA", "K", "RB", "CS", "FR", "BE", "MG", "CA", "SR", "BA", "RA",
        "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "Y", "ZR",
        "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "HF", "TA", "W", "RE",
        "OS", "IR", "PT", "AU", "HG", "AL", "GA", "IN", "TL", "SN", "PB", "BI",
    }
)


@dataclass(frozen=True)
class ComponentBond:
    atom_id_1: str
    atom_id_2: str
    value_order: str


@dataclass(frozen=True)
class ComponentDefinition:
    ccd_code: str
    path: Path
    atom_ids: tuple[str, ...]
    elements: tuple[str, ...]
    charges: tuple[int, ...]
    bonds: tuple[ComponentBond, ...]
    smiles: str | None
    smiles_type: str | None
    smiles_program: str | None
    smiles_program_version: str | None

    @property
    def heavy_atom_ids(self) -> tuple[str, ...]:
        return tuple(
            atom_id
            for atom_id, element in zip(self.atom_ids, self.elements, strict=True)
            if element.upper() != "H"
        )

    @property
    def heavy_elements(self) -> tuple[str, ...]:
        return tuple(element for element in self.elements if element.upper() != "H")

    @property
    def heavy_bonds(self) -> tuple[ComponentBond, ...]:
        heavy = set(self.heavy_atom_ids)
        return tuple(
            bond
            for bond in self.bonds
            if bond.atom_id_1 in heavy and bond.atom_id_2 in heavy
        )

    @property
    def explicit_h_parent_counts(self) -> dict[str, int]:
        element_by_id = dict(zip(self.atom_ids, self.elements, strict=True))
        counts = {atom_id: 0 for atom_id in self.heavy_atom_ids}
        for bond in self.bonds:
            left = element_by_id[bond.atom_id_1].upper()
            right = element_by_id[bond.atom_id_2].upper()
            if left == "H" and right != "H":
                counts[bond.atom_id_2] += 1
            elif right == "H" and left != "H":
                counts[bond.atom_id_1] += 1
        return counts


@dataclass(frozen=True)
class CcdAlias:
    original_ccd_code: str
    pdb_ccd_code: str
    reason: str


def component_path(mirror: Path, ccd_code: str) -> Path:
    normalized = ccd_code.strip().upper()
    path = mirror / normalized[0] / normalized / f"{normalized}.cif"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _column(category: object, name: str) -> list[str]:
    return [str(value) for value in category[name].as_array()]


def _select_smiles(
    category: object | None,
) -> tuple[str | None, str | None, str | None, str | None]:
    if category is None:
        return None, None, None, None
    types = _column(category, "type")
    programs = _column(category, "program")
    program_versions = _column(category, "program_version")
    descriptors = _column(category, "descriptor")
    for wanted_type, wanted_program in SMILES_PRECEDENCE:
        for descriptor_type, program, program_version, descriptor in zip(
            types, programs, program_versions, descriptors, strict=True
        ):
            if descriptor_type != wanted_type:
                continue
            if wanted_program is not None and program != wanted_program:
                continue
            if Chem.MolFromSmiles(descriptor) is not None:
                return descriptor, descriptor_type, program, program_version
    return None, None, None, None


@lru_cache(maxsize=None)
def load_component(mirror: Path, ccd_code: str) -> ComponentDefinition:
    path = component_path(mirror, ccd_code)
    cif = CIFFile.read(path)
    block = cif[next(iter(cif.keys()))]
    atoms = block["chem_comp_atom"]
    atom_ids = tuple(_column(atoms, "atom_id"))
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError(f"Duplicate CCD atom IDs in {path}")
    elements = tuple(value.upper() for value in _column(atoms, "type_symbol"))
    charges = tuple(int(value) for value in _column(atoms, "charge"))
    bonds: list[ComponentBond] = []
    if "chem_comp_bond" in block:
        category = block["chem_comp_bond"]
        for left, right, order in zip(
            _column(category, "atom_id_1"),
            _column(category, "atom_id_2"),
            _column(category, "value_order"),
            strict=True,
        ):
            if left not in atom_ids or right not in atom_ids:
                raise ValueError(f"Unknown CCD bond endpoint in {path}: {left}-{right}")
            bonds.append(ComponentBond(left, right, order))
    descriptor = block.get("pdbx_chem_comp_descriptor")
    smiles, smiles_type, smiles_program, smiles_program_version = _select_smiles(
        descriptor
    )
    return ComponentDefinition(
        ccd_code=ccd_code,
        path=path,
        atom_ids=atom_ids,
        elements=elements,
        charges=charges,
        bonds=tuple(bonds),
        smiles=smiles,
        smiles_type=smiles_type,
        smiles_program=smiles_program,
        smiles_program_version=smiles_program_version,
    )


def mirror_ccd_codes(mirror: Path) -> set[str]:
    return {path.stem.upper() for path in mirror.glob("*/*/*.cif")}


def _alias_candidates() -> Iterator[str]:
    for prefix in "ZYXQ":
        for first in BASE36:
            for second in BASE36:
                yield f"{prefix}{first}{second}"


def allocate_aliases(
    ccd_codes: Iterable[str], *, reserved_codes: Iterable[str], amino_acids: Iterable[str]
) -> list[CcdAlias]:
    normalized = sorted({str(code).strip().upper() for code in ccd_codes})
    amino_acids = {str(code).strip().upper() for code in amino_acids}
    needs_alias = [code for code in normalized if len(code) > 3 or code in amino_acids]
    reserved = (
        {str(code).strip().upper() for code in reserved_codes}
        | set(normalized)
        | amino_acids
        | {"CAP", "HOH", "WAT"}
    )
    candidates = (candidate for candidate in _alias_candidates() if candidate not in reserved)
    output: list[CcdAlias] = []
    for code in needs_alias:
        alias = next(candidates)
        reserved.add(alias)
        reason = "pdb_res_name_too_long" if len(code) > 3 else "amino_acid_ligand"
        output.append(CcdAlias(code, alias, reason))
    return output


def conect_lines(edges: Iterable[tuple[int, int]]) -> list[str]:
    neighbours: dict[int, set[int]] = {}
    for left, right in edges:
        if left == right:
            raise ValueError(f"Self bond at atom serial {left}")
        neighbours.setdefault(left, set()).add(right)
        neighbours.setdefault(right, set()).add(left)
    lines: list[str] = []
    for serial in sorted(neighbours):
        bonded = sorted(neighbours[serial])
        for start in range(0, len(bonded), 4):
            lines.append(
                "CONECT" + f"{serial:5d}" + "".join(
                    f"{other:5d}" for other in bonded[start : start + 4]
                )
            )
    return lines


def component_edges(
    component: ComponentDefinition,
    *,
    atom_names: Iterable[str],
    serials: Iterable[int],
    include_hydrogen: bool,
) -> list[tuple[int, int]]:
    serial_by_name = dict(zip(map(str, atom_names), map(int, serials), strict=True))
    element_by_id = dict(zip(component.atom_ids, component.elements, strict=True))
    output: list[tuple[int, int]] = []
    for bond in component.bonds:
        if not include_hydrogen and (
            element_by_id[bond.atom_id_1] == "H" or element_by_id[bond.atom_id_2] == "H"
        ):
            continue
        if bond.atom_id_1 in serial_by_name and bond.atom_id_2 in serial_by_name:
            output.append((serial_by_name[bond.atom_id_1], serial_by_name[bond.atom_id_2]))
    return output


def parse_conect_edges(path: Path) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("CONECT"):
            continue
        fields = [line[index : index + 5].strip() for index in range(6, len(line), 5)]
        values = [int(value) for value in fields if value]
        if not values:
            continue
        for other in values[1:]:
            edges.add(tuple(sorted((values[0], other))))
    return edges


def rigid_alignment(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError(f"Cannot align coordinate arrays {mobile.shape} and {target.shape}")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd((mobile - mobile_center).T @ (target - target_center))
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation, target_center - mobile_center @ rotation


def atomworks_fallback_molecule(
    component: ComponentDefinition, target_ligand: pr.Atomic, ccd_mirror_path: Path
) -> Chem.Mol:
    """Build one of the configured stock-NISE exceptions through AtomWorks."""
    from atomworks.io.tools.rdkit import ccd_code_to_rdkit

    atom_array, mol = ccd_code_to_rdkit(
        component.ccd_code,
        ccd_mirror_path=ccd_mirror_path,
        hydrogen_policy="infer",
        return_atom_array=True,
    )
    names = list(map(str, atom_array.atom_name))
    heavy_indices = [index for index, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() != "H"]
    heavy_names = [names[index] for index in heavy_indices]
    target_names = list(map(str, target_ligand.getNames()))
    if set(heavy_names) != set(target_names):
        raise ValueError(
            f"Fallback heavy atom names differ for {component.ccd_code}: "
            f"{sorted(heavy_names)} != {sorted(target_names)}"
        )
    conformer = mol.GetConformer()
    source = np.asarray(conformer.GetPositions(), dtype=float)
    target_by_name = dict(zip(target_names, target_ligand.getCoords(), strict=True))
    mobile_heavy = source[heavy_indices]
    target_heavy = np.asarray([target_by_name[name] for name in heavy_names])
    rotation, translation = rigid_alignment(mobile_heavy, target_heavy)
    transformed = source @ rotation + translation
    heavy_target_displacements = {
        atom_index: target_by_name[name] - transformed[atom_index]
        for atom_index, name in zip(heavy_indices, heavy_names, strict=True)
    }
    # A global rigid alignment is not sufficient when the CCD conformer and the
    # target have different torsions.  Move each inferred H with its bonded
    # heavy parent before snapping the heavy atoms themselves.  This preserves
    # the local CCD H-parent bond vector while retaining every target heavy
    # coordinate exactly.
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":
            continue
        heavy_neighbours = [
            neighbour.GetIdx()
            for neighbour in atom.GetNeighbors()
            if neighbour.GetSymbol() != "H"
        ]
        if not heavy_neighbours:
            raise ValueError(
                f"Fallback hydrogen {atom.GetIdx()} has no heavy parent for "
                f"{component.ccd_code}"
            )
        displacement = np.mean(
            [heavy_target_displacements[index] for index in heavy_neighbours], axis=0
        )
        transformed[atom.GetIdx()] += displacement
    for atom_index, name in zip(heavy_indices, heavy_names, strict=True):
        transformed[atom_index] = target_by_name[name]
    for index, coord in enumerate(transformed):
        conformer.SetAtomPosition(index, coord.tolist())
    for atom_index, name in zip(heavy_indices, heavy_names, strict=True):
        mol.GetAtomWithIdx(atom_index).SetProp("_CanonicalHeavyName", name)
    return mol


def molecule_heavy_name_order(mol: Chem.Mol, source_names: list[str]) -> list[str]:
    named = [
        atom.GetProp("_CanonicalHeavyName") if atom.HasProp("_CanonicalHeavyName") else ""
        for atom in mol.GetAtoms()
        if atom.GetSymbol() != "H"
    ]
    return named if all(named) else source_names
