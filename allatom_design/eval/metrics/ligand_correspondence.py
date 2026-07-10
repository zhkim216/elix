from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolAlign
from rdkit.Geometry import Point3D

from atomworks.io.tools.rdkit import atom_array_to_rdkit

from allatom_design.data.const import METAL_ELEMENTS
from allatom_design.eval.chemical_components import normalize_ccd_code


def _metal_atom_key(atom_array: AtomArray, idx: int) -> tuple[str, str, str, str]:
    return (
        str(atom_array.pn_unit_iid[idx]),
        str(atom_array.res_name[idx]),
        str(atom_array.atom_name[idx]),
        normalize_ccd_code(atom_array.element[idx]),
    )


def matched_metal_atom_masks(
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    metal_pn_unit_iids: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[str, str, str, str]] | None, str | None]:
    sample_indices = np.where(
        np.isin(sample_atom_array.pn_unit_iid, metal_pn_unit_iids)
        & np.isin(
            [normalize_ccd_code(element) for element in sample_atom_array.element],
            list(METAL_ELEMENTS),
        )
    )[0]
    pred_indices = np.where(
        np.isin(pred_atom_array.pn_unit_iid, metal_pn_unit_iids)
        & np.isin(
            [normalize_ccd_code(element) for element in pred_atom_array.element],
            list(METAL_ELEMENTS),
        )
    )[0]

    sample_by_key = {}
    for idx in sample_indices:
        key = _metal_atom_key(sample_atom_array, idx)
        if key in sample_by_key:
            return None, None, None, f"Ambiguous reference metal atom key: {key}"
        sample_by_key[key] = idx

    pred_by_key = {}
    for idx in pred_indices:
        key = _metal_atom_key(pred_atom_array, idx)
        if key in pred_by_key:
            return None, None, None, f"Ambiguous predicted metal atom key: {key}"
        pred_by_key[key] = idx

    if not sample_by_key:
        return None, None, None, "No reference metal atoms found"
    if set(sample_by_key) != set(pred_by_key):
        if (
            len(metal_pn_unit_iids) == 1
            and len(sample_indices) == 1
            and len(pred_indices) == 1
        ):
            sample_idx = int(sample_indices[0])
            pred_idx = int(pred_indices[0])
            sample_pn_unit_iid = str(sample_atom_array.pn_unit_iid[sample_idx])
            pred_pn_unit_iid = str(pred_atom_array.pn_unit_iid[pred_idx])
            sample_element = normalize_ccd_code(sample_atom_array.element[sample_idx])
            pred_element = normalize_ccd_code(pred_atom_array.element[pred_idx])
            if (
                sample_pn_unit_iid == pred_pn_unit_iid
                and sample_pn_unit_iid == str(metal_pn_unit_iids[0])
                and sample_element == pred_element
            ):
                sample_mask = np.zeros(len(sample_atom_array), dtype=bool)
                pred_mask = np.zeros(len(pred_atom_array), dtype=bool)
                sample_mask[sample_idx] = True
                pred_mask[pred_idx] = True
                return (
                    sample_mask,
                    pred_mask,
                    [
                        (
                            sample_pn_unit_iid,
                            str(sample_atom_array.res_name[sample_idx]),
                            str(sample_atom_array.atom_name[sample_idx]),
                            sample_element,
                        )
                    ],
                    None,
                )
        return None, None, None, "Reference and predicted metal atom keys do not match"

    matched_keys = list(sample_by_key.keys())
    sample_mask = np.zeros(len(sample_atom_array), dtype=bool)
    pred_mask = np.zeros(len(pred_atom_array), dtype=bool)
    sample_mask[[sample_by_key[key] for key in matched_keys]] = True
    pred_mask[[pred_by_key[key] for key in matched_keys]] = True
    return sample_mask, pred_mask, matched_keys, None


def rdkit_mol_from_ligand_atom_array(ligand_atom_array: AtomArray):
    try:
        return atom_array_to_rdkit(ligand_atom_array, sanitize=True)
    except Exception:
        return atom_array_to_rdkit(ligand_atom_array, sanitize=False)


def _rdkit_mol_from_smiles_with_atom_array_coords(
    smiles: str | None,
    ligand_atom_array: AtomArray,
):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    if mol.GetNumAtoms() != len(ligand_atom_array):
        return None

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for idx, coord in enumerate(ligand_atom_array.coord):
        if np.isnan(coord).any():
            return None
        conformer.SetAtomPosition(
            idx,
            Point3D(float(coord[0]), float(coord[1]), float(coord[2])),
        )
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    return mol


def mcs_core_rmsd_no_ligand_alignment(
    *,
    reference_ligand_atom_array: AtomArray,
    pred_ligand_atom_array: AtomArray,
    pred_ligand_smiles: str | None = None,
) -> tuple[dict[str, float | int | str | None] | None, str | None]:
    try:
        reference_mol = rdkit_mol_from_ligand_atom_array(reference_ligand_atom_array)
        pred_mol = (
            _rdkit_mol_from_smiles_with_atom_array_coords(
                pred_ligand_smiles,
                pred_ligand_atom_array,
            )
            or rdkit_mol_from_ligand_atom_array(pred_ligand_atom_array)
        )
    except Exception as exc:
        return None, f"Failed to build RDKit ligand molecules for MCS RMSD: {exc}"

    if reference_mol is None or pred_mol is None:
        return None, "Failed to build RDKit ligand molecules for MCS RMSD"
    if reference_mol.GetNumAtoms() == 0 or pred_mol.GetNumAtoms() == 0:
        return None, "No ligand atoms available for MCS RMSD"

    mcs = rdFMCS.FindMCS(
        [reference_mol, pred_mol],
        completeRingsOnly=True,
        ringMatchesRingOnly=True,
        timeout=10,
    )
    if mcs.canceled:
        return None, "MCS search timed out"
    if mcs.numAtoms == 0 or not mcs.smartsString:
        return None, "No common ligand MCS atoms"

    mcs_query = Chem.MolFromSmarts(mcs.smartsString)
    if mcs_query is None:
        return None, "Failed to parse MCS SMARTS"
    reference_matches = reference_mol.GetSubstructMatches(
        mcs_query, uniquify=True, maxMatches=1024
    )
    pred_matches = pred_mol.GetSubstructMatches(
        mcs_query, uniquify=True, maxMatches=1024
    )
    if not reference_matches or not pred_matches:
        return None, "MCS SMARTS did not match both ligand molecules"

    atom_maps = [
        [
            (int(pred_idx), int(reference_idx))
            for pred_idx, reference_idx in zip(pred_match, reference_match)
        ]
        for reference_match in reference_matches
        for pred_match in pred_matches
    ]
    if not atom_maps:
        return None, "No MCS atom maps found for RMSD"
    try:
        best_rmsd = rdMolAlign.CalcRMS(pred_mol, reference_mol, map=atom_maps)
    except Exception as exc:
        return None, f"Failed to calculate MCS RMSD using RDKit CalcRMS: {exc}"

    return {
        "ligand_rmsd": float(best_rmsd),
        "ligand_rmsd_mode": "smiles_mcs_core",
        "mcs_num_atoms": int(mcs.numAtoms),
        "mcs_pred_coverage": float(mcs.numAtoms / max(pred_mol.GetNumAtoms(), 1)),
        "mcs_reference_coverage": float(
            mcs.numAtoms / max(reference_mol.GetNumAtoms(), 1)
        ),
    }, None
