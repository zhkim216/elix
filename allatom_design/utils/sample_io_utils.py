from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io import pdbx
from biotite.structure.io.pdbx import convert as pdbx_convert
from omegaconf import DictConfig, OmegaConf

from atomworks.io.parser import parse as aw_parse
from atomworks.io.utils.io_utils import to_cif_file

###########################################################
# Functions for loading examples
###########################################################

def load_example_with_parse(
    pdb_path: str,
    cif_parse_cfg: DictConfig | None = None,
) -> dict[str, Any]:
    """
    Load an example dictionary from a structure file using atomworks parse.
    """
    if cif_parse_cfg is None:
        cif_parse_cfg = {
            "add_missing_atoms": True,
            "remove_waters": True,
            "remove_ccds": [],
            "fix_ligands_at_symmetry_centers": True,
            "fix_arginines": True,
            "convert_mse_to_met": True,
            "hydrogen_policy": "remove",
            "extra_fields": "all",
        }
    else:
        cif_parse_cfg = OmegaConf.to_container(cif_parse_cfg, resolve=True)

    transformation_id = "1"
    cif_parse_cfg["build_assembly"] = [transformation_id]
    input_data = aw_parse(pdb_path, **cif_parse_cfg)
    atom_array = input_data["assemblies"][transformation_id][0]
    
    # Fix annotation types for atom_array loaded from CIF.
    atom_array = fix_cif_annotation_types_atom_array(atom_array)
    
    chain_info = input_data["chain_info"]

    return {"example_id": Path(pdb_path).stem, "atom_array": atom_array, "chain_info": chain_info}

def fix_cif_annotation_types_atom_array(atom_array: AtomArray) -> AtomArray:
    """
    Fix annotation types for atom_array loaded from CIF.
    CIF format stores values as strings, so convert back to expected numeric/bool types where possible.
    """
    bool_annotations = [
        "atomize",
        "is_polymer",
        "is_aromatic",
        "is_covalent_modification",
        "is_backbone_atom",
        "hetero",
        "is_leaving_atom",
        "is_n_terminal_atom",
        "is_c_terminal_atom",
    ]
    
    for ann in bool_annotations:
        if ann in atom_array.get_annotation_categories():
            val = getattr(atom_array, ann)
            if val.dtype.kind in ("U", "S", "O"):
                new_val = val == "True"
                atom_array.del_annotation(ann)
                atom_array.set_annotation(ann, new_val)

    int_annotations = [
        "chain_type",
        "atomic_number",
        "within_chain_res_idx",
        "within_poly_res_idx",
        "chain_entity",
        "molecule_entity",
        "pn_unit_entity",
        "token_id",
        "transformation_id",
        "pdbx_PDB_model_num",
        "label_entity_id",
        "label_seq_id",
        "auth_seq_id",
        "molecule_id",
        "molecule_iid",
        "charge",
        "pdbx_formal_charge",
    ]
    for ann in int_annotations:
        if ann in atom_array.get_annotation_categories():
            val = getattr(atom_array, ann)
            if val.dtype.kind in ("U", "S", "O"):                
                new_val = np.array([int(v) if str(v).lstrip("-").isdigit() else 0 for v in val])                
                atom_array.del_annotation(ann)
                atom_array.set_annotation(ann, new_val)
    float_annotations = ["B_iso_or_equiv", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "b_factor"]
    for ann in float_annotations:
        if ann in atom_array.get_annotation_categories():
            val = getattr(atom_array, ann)
            if val.dtype.kind in ("U", "S", "O"):
                new_val = np.array([float(v) if v not in ("?", ".", "") else np.nan for v in val])
                atom_array.del_annotation(ann)
                atom_array.set_annotation(ann, new_val)

    return atom_array

### Functions for saving cif files ###

def save_cif_file(
    atom_array: AtomArray,
    cif_path: str | Path,
    cif_save_cfg: dict[str, Any] | None = None,
) -> None:    
    """
    Save an atom array to a cif file.
    """
    if cif_save_cfg is None:
        cif_save_cfg = {
            "file_type": "cif",
            "date": "1959-01-07",
            "include_entity_poly": True,
            "include_entity_nonpoly": True,
            "include_nan_coords": False,
            "include_bonds": True,
            "extra_fields": [],
            "exclude_field_keys": ["token_id", "is_ligand_pocket"],
            "extra_categories": {
                "pdbx_audit_revision_history": {
                    "ordinal": [1],
                    "revision_id": [1],
                    "revision_date": ["1959-01-07"],
                    "major_revision": [1],
                    "minor_revision": [0],
                    "revision_description": ["Dummy date for template-conditioning AF3"],
                },
            },
        }
    else:
        cif_save_cfg = OmegaConf.to_container(cif_save_cfg, resolve=True)
    
    # Ensure b_factor annotation exists in atom array for AF3 template conditioning
    # AF3 requires _atom_site.B_iso_or_equiv in template CIF files
    if "b_factor" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("b_factor", np.zeros(len(atom_array)))
    
    try:
        to_cif_file(atom_array, cif_path, **cif_save_cfg)
    except AttributeError as exc:
        if cif_save_cfg.get("include_bonds", True) and "convert_bond_type" in str(exc):
            retry_cfg = dict(cif_save_cfg)
            retry_cfg["include_bonds"] = False
            print(
                f"Warning: failed to write bonds for {cif_path}; "
                "retrying with include_bonds=False"
            )
            to_cif_file(atom_array, cif_path, **retry_cfg)
        else:
            raise
    fix_cif_formal_charge_format(cif_path)
    return cif_path


def _default_pdb_to_cif_parse_cfg() -> dict[str, Any]:
    return {
        "file_type": "pdb",
        "add_missing_atoms": False,
        "remove_waters": True,
        "remove_ccds": [],
        "fix_ligands_at_symmetry_centers": True,
        "fix_arginines": True,
        "fix_formal_charges": False,
        "convert_mse_to_met": True,
        "hydrogen_policy": "remove",
        "build_assembly": ["1"],
        "extra_fields": "all",
    }


def _default_pdb_to_cif_save_cfg(pdb_path: str | Path) -> dict[str, Any]:
    return {
        "file_type": "cif",
        "id": Path(pdb_path).stem,
        "date": "1959-01-07",
        "include_entity_poly": True,
        "include_entity_nonpoly": True,
        "include_nan_coords": False,
        "include_bonds": True,
        "extra_fields": [],
        "exclude_field_keys": ["token_id", "is_ligand_pocket"],
    }


def _get_primary_assembly(input_data: dict[str, Any], parse_cfg: dict[str, Any]) -> AtomArray:
    build_assembly = parse_cfg.get("build_assembly", ["1"])
    if isinstance(build_assembly, str) and build_assembly not in {"all", "first"}:
        assembly_id = build_assembly
    elif isinstance(build_assembly, (list, tuple)) and len(build_assembly) == 1:
        assembly_id = str(build_assembly[0])
    else:
        assembly_id = sorted(input_data["assemblies"].keys())[0]
    return input_data["assemblies"][assembly_id][0]


def _atom_array_for_written_cif(
    atom_array: AtomArray,
    *,
    include_nan_coords: bool,
) -> AtomArray:
    written_atom_array = atom_array.copy()
    if include_nan_coords:
        return written_atom_array
    if len(written_atom_array) == 0:
        return written_atom_array
    finite_coord_mask = np.all(np.isfinite(written_atom_array.coord), axis=-1)
    return written_atom_array[finite_coord_mask]


def _append_intra_residue_bonds_to_cif(cif_path: str | Path, atom_array: AtomArray) -> int:
    if atom_array.bonds is None or atom_array.bonds.get_bond_count() == 0:
        return 0

    cif_path = Path(cif_path)
    cif_file = pdbx.CIFFile.read(str(cif_path))
    block = cif_file.block
    chem_comp_bond = pdbx_convert._set_intra_residue_bonds(atom_array, block["atom_site"])
    if chem_comp_bond is None:
        return 0

    block["chem_comp_bond"] = chem_comp_bond
    with open(cif_path, "w") as f:
        cif_file.write(f)
    return len(chem_comp_bond["comp_id"])


def _to_cif_file_preserving_intra_residue_bonds(
    atom_array: AtomArray,
    cif_path: str | Path,
    *,
    cif_save_cfg: dict[str, Any],
) -> dict[str, Any]:
    cif_save_cfg = dict(cif_save_cfg)
    include_bonds = bool(cif_save_cfg.get("include_bonds", True))
    include_nan_coords = bool(cif_save_cfg.get("include_nan_coords", True))

    atom_array_to_write = atom_array.copy()
    if "b_factor" not in atom_array_to_write.get_annotation_categories():
        atom_array_to_write.set_annotation("b_factor", np.zeros(len(atom_array_to_write)))

    writer_cfg = dict(cif_save_cfg)
    if include_bonds:
        # Keep atom_site writing independent of Biotite/AtomWorks bond-writer
        # compatibility, then add chem_comp_bond from the prepared BondList.
        writer_cfg["include_bonds"] = False

    written_path = Path(to_cif_file(atom_array_to_write, cif_path, **writer_cfg))
    chem_comp_bond_rows = 0
    if include_bonds:
        bond_atom_array = _atom_array_for_written_cif(
            atom_array_to_write,
            include_nan_coords=include_nan_coords,
        )
        chem_comp_bond_rows = _append_intra_residue_bonds_to_cif(written_path, bond_atom_array)

    fix_cif_formal_charge_format(written_path)
    return {
        "cif_path": str(written_path),
        "chem_comp_bond_rows": chem_comp_bond_rows,
        "include_bonds": include_bonds,
    }


def _bond_counts(atom_array: AtomArray) -> dict[str, int]:
    bond_array = (
        atom_array.bonds.as_array()
        if atom_array.bonds is not None
        else np.empty((0, 3), dtype=int)
    )
    return {
        "atom_count": int(len(atom_array)),
        "bond_count": int(len(bond_array)),
        "ligand_bond_count": int(len(_ligand_intra_residue_bond_signature(atom_array))),
    }


def _ligand_atom_mask(atom_array: AtomArray) -> np.ndarray:
    annotations = atom_array.get_annotation_categories()
    if "is_polymer" in annotations:
        return ~atom_array.is_polymer.astype(bool)
    return atom_array.hetero.astype(bool)


def _normalize_element_symbol(element: Any) -> str:
    element_text = str(element).strip()
    if not element_text:
        return ""
    return element_text[0].upper() + element_text[1:].lower()


def _coerce_ligand_sdf_paths(
    ligand_sdf_paths: str | Path | Sequence[str | Path] | None,
) -> list[Path]:
    if ligand_sdf_paths is None:
        return []
    if isinstance(ligand_sdf_paths, (str, Path)):
        return [Path(ligand_sdf_paths)]
    return [Path(path) for path in ligand_sdf_paths]


def _ligand_residue_groups(atom_array: AtomArray) -> list[np.ndarray]:
    ligand_mask = _ligand_atom_mask(atom_array)
    groups_by_key: dict[tuple[str, str, str, str], list[int]] = {}
    annotations = atom_array.get_annotation_categories()
    has_ins_code = "ins_code" in annotations

    for atom_index in np.flatnonzero(ligand_mask):
        atom_index = int(atom_index)
        ins_code = str(atom_array.ins_code[atom_index]) if has_ins_code else ""
        key = (
            str(atom_array.chain_id[atom_index]),
            str(atom_array.res_id[atom_index]),
            ins_code,
            str(atom_array.res_name[atom_index]),
        )
        groups_by_key.setdefault(key, []).append(atom_index)

    return [np.array(indices, dtype=int) for indices in groups_by_key.values()]


def _describe_ligand_group(atom_array: AtomArray, group: np.ndarray) -> str:
    first_atom = int(group[0])
    return (
        f"{atom_array.res_name[first_atom]} "
        f"chain={atom_array.chain_id[first_atom]} "
        f"res_id={atom_array.res_id[first_atom]} "
        f"atoms={len(group)}"
    )


def _ligand_group_elements(atom_array: AtomArray, group: np.ndarray) -> list[str]:
    return [
        _normalize_element_symbol(atom_array.element[int(atom_index)])
        for atom_index in group
    ]


def _rdkit_bond_type_to_biotite(bond_type: Any) -> int:
    from rdkit import Chem

    bond_type_map = {
        Chem.BondType.SINGLE: struc.bonds.BondType.SINGLE,
        Chem.BondType.DOUBLE: struc.bonds.BondType.DOUBLE,
        Chem.BondType.TRIPLE: struc.bonds.BondType.TRIPLE,
        Chem.BondType.QUADRUPLE: struc.bonds.BondType.QUADRUPLE,
        Chem.BondType.AROMATIC: struc.bonds.BondType.AROMATIC,
    }
    # chem_comp_bond has order/aromatic fields, not coordination semantics.
    if hasattr(Chem.BondType, "DATIVE"):
        bond_type_map[Chem.BondType.DATIVE] = struc.bonds.BondType.SINGLE
    if hasattr(Chem.BondType, "DATIVEL"):
        bond_type_map[Chem.BondType.DATIVEL] = struc.bonds.BondType.SINGLE
    if hasattr(Chem.BondType, "DATIVER"):
        bond_type_map[Chem.BondType.DATIVER] = struc.bonds.BondType.SINGLE

    return int(bond_type_map.get(bond_type, struc.bonds.BondType.ANY))


def _sdf_heavy_atom_bond_orders(sdf_path: Path) -> tuple[list[str], list[tuple[int, int, int]]]:
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    mol = next((candidate for candidate in supplier if candidate is not None), None)
    if mol is None:
        raise ValueError(f"Could not read ligand SDF: {sdf_path}")

    heavy_index_by_mol_index: dict[int, int] = {}
    heavy_elements: list[str] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        heavy_index_by_mol_index[atom.GetIdx()] = len(heavy_elements)
        heavy_elements.append(_normalize_element_symbol(atom.GetSymbol()))

    heavy_bonds: list[tuple[int, int, int]] = []
    for bond in mol.GetBonds():
        begin_index = bond.GetBeginAtomIdx()
        end_index = bond.GetEndAtomIdx()
        if begin_index not in heavy_index_by_mol_index or end_index not in heavy_index_by_mol_index:
            continue
        heavy_bonds.append(
            (
                heavy_index_by_mol_index[begin_index],
                heavy_index_by_mol_index[end_index],
                _rdkit_bond_type_to_biotite(bond.GetBondType()),
            )
        )

    return heavy_elements, heavy_bonds


def _apply_ligand_sdf_bond_orders(
    atom_array: AtomArray,
    ligand_sdf_paths: str | Path | Sequence[str | Path] | None,
) -> dict[str, Any]:
    sdf_paths = _coerce_ligand_sdf_paths(ligand_sdf_paths)
    if not sdf_paths:
        return {
            "ligand_sdf_count": 0,
            "ligand_sdf_paths": [],
            "sdf_bond_order_applied": False,
            "sdf_ligand_bond_count": 0,
            "sdf_ligand_atom_count": 0,
            "sdf_ligand_mappings": [],
        }

    ligand_groups = _ligand_residue_groups(atom_array)
    if len(sdf_paths) > len(ligand_groups):
        raise ValueError(
            "Ligand SDF count exceeds parsed ligand residue count: "
            f"{len(sdf_paths)} SDF(s), {len(ligand_groups)} ligand residue(s)"
        )

    unmatched_groups = list(enumerate(ligand_groups))
    mapped_group_by_atom: dict[int, int] = {}
    new_ligand_bonds: list[tuple[int, int, int]] = []
    mapping_reports: list[dict[str, Any]] = []

    for sdf_path in sdf_paths:
        if not sdf_path.exists():
            raise FileNotFoundError(f"Ligand SDF does not exist: {sdf_path}")

        sdf_elements, sdf_bonds = _sdf_heavy_atom_bond_orders(sdf_path)
        matched_group_position: int | None = None
        matched_group_index: int | None = None
        matched_group: np.ndarray | None = None
        for candidate_position, (candidate_group_index, candidate_group) in enumerate(
            unmatched_groups
        ):
            if _ligand_group_elements(atom_array, candidate_group) == sdf_elements:
                matched_group_position = candidate_position
                matched_group_index = candidate_group_index
                matched_group = candidate_group
                break

        if matched_group is None or matched_group_index is None or matched_group_position is None:
            available_groups = "; ".join(
                _describe_ligand_group(atom_array, group) for _, group in unmatched_groups
            )
            raise ValueError(
                f"Could not map SDF heavy atom order to any parsed ligand residue for {sdf_path}; "
                f"SDF heavy atom count={len(sdf_elements)}, available groups=[{available_groups}]"
            )

        group = matched_group
        group_index = matched_group_index
        unmatched_groups.pop(matched_group_position)

        for atom_index in group:
            mapped_group_by_atom[int(atom_index)] = group_index
        for sdf_i, sdf_j, bond_type in sdf_bonds:
            atom_i = int(group[sdf_i])
            atom_j = int(group[sdf_j])
            new_ligand_bonds.append((atom_i, atom_j, int(bond_type)))

        mapping_reports.append(
            {
                "sdf_path": str(sdf_path),
                "chain_id": str(atom_array.chain_id[int(group[0])]),
                "res_id": str(atom_array.res_id[int(group[0])]),
                "res_name": str(atom_array.res_name[int(group[0])]),
                "heavy_atom_count": len(sdf_elements),
                "heavy_bond_count": len(sdf_bonds),
            }
        )

    existing_bonds = (
        atom_array.bonds.as_array()
        if atom_array.bonds is not None
        else np.empty((0, 3), dtype=int)
    )
    kept_bonds: list[tuple[int, int, int]] = []
    for atom_i, atom_j, bond_type in existing_bonds:
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        group_i = mapped_group_by_atom.get(atom_i)
        group_j = mapped_group_by_atom.get(atom_j)
        if group_i is not None and group_i == group_j:
            continue
        kept_bonds.append((atom_i, atom_j, int(bond_type)))

    updated_bonds = np.array(kept_bonds + new_ligand_bonds, dtype=int)
    if updated_bonds.size == 0:
        updated_bonds = np.empty((0, 3), dtype=int)
    atom_array.bonds = struc.BondList(len(atom_array), updated_bonds)

    return {
        "ligand_sdf_count": len(sdf_paths),
        "ligand_sdf_paths": [str(path) for path in sdf_paths],
        "sdf_bond_order_applied": True,
        "sdf_ligand_bond_count": len(new_ligand_bonds),
        "sdf_ligand_atom_count": sum(report["heavy_atom_count"] for report in mapping_reports),
        "unmapped_ligand_group_count": len(unmatched_groups),
        "sdf_ligand_mappings": mapping_reports,
    }


def _ligand_intra_residue_bond_signature(atom_array: AtomArray) -> Counter[tuple[Any, ...]]:
    bond_array = (
        atom_array.bonds.as_array()
        if atom_array.bonds is not None
        else np.empty((0, 3), dtype=int)
    )
    ligand_mask = _ligand_atom_mask(atom_array)
    signature: Counter[tuple[Any, ...]] = Counter()
    for atom_i, atom_j, bond_type in bond_array:
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if not (ligand_mask[atom_i] and ligand_mask[atom_j]):
            continue
        same_residue = (
            atom_array.chain_id[atom_i] == atom_array.chain_id[atom_j]
            and atom_array.res_id[atom_i] == atom_array.res_id[atom_j]
            and atom_array.res_name[atom_i] == atom_array.res_name[atom_j]
        )
        if not same_residue:
            continue
        atom_names = tuple(
            sorted((str(atom_array.atom_name[atom_i]), str(atom_array.atom_name[atom_j])))
        )
        signature[
            (
                str(atom_array.chain_id[atom_i]),
                int(atom_array.res_id[atom_i]),
                str(atom_array.res_name[atom_i]),
                atom_names,
                int(bond_type),
            )
        ] += 1
    return signature


def pdb_to_cif(
    pdb_path: str | Path,
    cif_path: str | Path,
    *,
    overwrite: bool = False,
    ligand_sdf_paths: str | Path | Sequence[str | Path] | None = None,
    parse_cfg: dict[str, Any] | DictConfig | None = None,
    save_cfg: dict[str, Any] | DictConfig | None = None,
    validate_roundtrip: bool = False,
) -> dict[str, Any]:
    """
    Convert a PDB file to an AtomWorks-compatible CIF while preserving ligand bonds.

    If ``ligand_sdf_paths`` is given, each SDF is mapped in order to the parsed
    nonpolymer ligand residues and its heavy-atom bond graph replaces the PDB
    ligand intra-residue graph before writing. This is the preferred path for
    generated PDBs whose CONECT records preserve connectivity but not order.
    """
    pdb_path = Path(pdb_path)
    cif_path = Path(cif_path)
    if cif_path.exists() and not overwrite:
        raise FileExistsError(f"CIF already exists: {cif_path}")

    parse_kwargs = _default_pdb_to_cif_parse_cfg()
    if parse_cfg is not None:
        parse_kwargs.update(
            OmegaConf.to_container(parse_cfg, resolve=True)
            if isinstance(parse_cfg, DictConfig)
            else parse_cfg
        )

    save_kwargs = _default_pdb_to_cif_save_cfg(pdb_path)
    if save_cfg is not None:
        save_kwargs.update(
            OmegaConf.to_container(save_cfg, resolve=True)
            if isinstance(save_cfg, DictConfig)
            else save_cfg
        )

    input_data = aw_parse(pdb_path, **parse_kwargs)
    atom_array = _get_primary_assembly(input_data, parse_kwargs)
    sdf_bond_info = _apply_ligand_sdf_bond_orders(atom_array, ligand_sdf_paths)
    atom_counts = _bond_counts(atom_array)

    cif_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cif_path.with_name(f".{cif_path.stem}.tmp.{os.getpid()}.cif")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        write_info = _to_cif_file_preserving_intra_residue_bonds(
            atom_array,
            tmp_path,
            cif_save_cfg=save_kwargs,
        )
        tmp_written_path = Path(write_info["cif_path"])

        roundtrip_counts: dict[str, Any] = {}
        if validate_roundtrip:
            roundtrip_cfg = dict(parse_kwargs)
            roundtrip_cfg["file_type"] = "cif"
            roundtrip_data = aw_parse(tmp_written_path, **roundtrip_cfg)
            roundtrip_atom_array = _get_primary_assembly(roundtrip_data, roundtrip_cfg)
            roundtrip_counts = {
                f"roundtrip_{key}": value
                for key, value in _bond_counts(roundtrip_atom_array).items()
            }
            ligand_bonds_match = (
                _ligand_intra_residue_bond_signature(atom_array)
                == _ligand_intra_residue_bond_signature(roundtrip_atom_array)
            )
            roundtrip_counts["roundtrip_ligand_bonds_match"] = bool(ligand_bonds_match)
            if not ligand_bonds_match:
                raise ValueError(f"Ligand bond graph changed during roundtrip: {pdb_path}")

        os.replace(tmp_written_path, cif_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return {
        "pdb_path": str(pdb_path),
        "cif_path": str(cif_path),
        "chem_comp_bond_rows": int(write_info["chem_comp_bond_rows"]),
        **sdf_bond_info,
        **atom_counts,
        **roundtrip_counts,
    }


def fix_cif_formal_charge_format(cif_path: str | Path) -> None:
    """
    Fix pdbx_formal_charge format in CIF files for OpenStructure compatibility.
    Convert +N -> N while preserving negatives.
    """
    cif_path = Path(cif_path)
    if not cif_path.exists():
        return

    with open(cif_path, "r") as f:
        content = f.read()

    fixed_content = re.sub(r"(\s)\+(\d+)(\s)", r"\1\2\3", content)

    if content != fixed_content:
        with open(cif_path, "w") as f:
            f.write(fixed_content)
