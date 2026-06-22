from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.utils.geometry import align_atom_arrays

from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets


def calculate_ligand_rmsd_with_binding_site_superposition(
    pred_example: dict[str, Any] = None,
    sample_example: dict[str, Any] = None,
    receptor_pn_unit_iids: list = ["A_1"],
    ligand_pn_unit_iids: list = ["C_1"],
    pocket_distance: float = 8.0,
    save_aligned: bool = True,
    sample_path: str | Path = None,
    pred_path: str | Path = None,
) -> dict[str, float]:
    """
    Calculate ligand RMSD after superimposing structures based on binding site residues.
    Uses Atomworks framework for loading and processing.

    Parameters
    ----------
    ref_cif_path : Path
        Path to reference CIF file.
    pred_cif_path : Path
        Path to predicted CIF file.
    receptor_chain : str
        Chain ID for receptor.
    ligand_chain : str
        Chain ID for ligand.
    binding_site_radius : float
        Radius for defining binding site residues.
    save_aligned : bool
        If True, save the pocket-aligned predicted structure to the same directory
        with "_pocket_aligned" suffix.
    cif_parser_args: DictConfig = None,
        Additional keyword arguments to pass to atomworks.io.parser.parse.
        Useful for controlling hydrogen_policy, add_missing_atoms, etc.

    Returns
    -------
    dict
        Dictionary with RMSD values and other metrics.
    """

    sample_array = sample_example['atom_array']
    pred_array = pred_example['atom_array']

    print(f"pocket_distance: {pocket_distance}")
    # Annotate ligand pockets (binding site residues)
    sample_array = annotate_ligand_pockets(atom_array=sample_array,
                                           pocket_distance=pocket_distance,
                                           receptor_pn_unit_iids=receptor_pn_unit_iids,
                                           ligand_pn_unit_iids=ligand_pn_unit_iids)
    pred_array = annotate_ligand_pockets(atom_array=pred_array,
                                         pocket_distance=pocket_distance,
                                         receptor_pn_unit_iids=receptor_pn_unit_iids,
                                         ligand_pn_unit_iids=ligand_pn_unit_iids)

    # Get binding site CA atoms for superposition
    # Use sequential residue index (order in chain) instead of res_id for matching
    # because res_id may differ between structures (ref vs AF3 prediction)
    sample_receptor_mask = np.isin(sample_array.pn_unit_iid, receptor_pn_unit_iids)
    pred_receptor_mask = np.isin(pred_array.pn_unit_iid, receptor_pn_unit_iids)

    # Get all CA atoms from receptor chain
    sample_ca_mask = sample_receptor_mask & (sample_array.atom_name == "CA") & (sample_array.res_name != "UNK")

    # Delete UNK residues from pred_atom_array, it's from the sample sequence for the gaps between the actual residues.
    # Designed sequence don't output UNK residues, so we can safely delete them.
    pred_ca_mask = pred_receptor_mask & (pred_array.atom_name == "CA") & (pred_array.res_name != "UNK")

    sample_ca = sample_array[sample_ca_mask]
    pred_ca = pred_array[pred_ca_mask]

    assert len(sample_ca) == len(pred_ca), "Number of CA atoms in sample and pred must match"

    if len(sample_ca) == 0 or len(pred_ca) == 0:
        return {"error": "No CA atoms found", "ligand_rmsd": None}

    # Get binding site mask for CA atoms
    sample_bs_ca_mask = sample_array.is_ligand_pocket[sample_ca_mask]

    # Get binding site CA atoms by sequential index
    sample_bs_sorted = sample_ca[sample_bs_ca_mask]
    pred_bs_sorted = pred_ca[sample_bs_ca_mask]  # Use ref's BS mask for both

    assert (sample_bs_sorted.res_name == pred_bs_sorted.res_name).all(), "amino acid residues in sample and pred binding site must match"

    num_bs_residues = np.sum(sample_bs_ca_mask)

    if len(sample_bs_sorted) == 0:
        return {"error": "No binding site CA atoms found", "ligand_rmsd": None}

    # Align pred onto ref using binding site CA atoms
    # align_atom_arrays: aligns mbl_sele to tgt_sele, applies transform to mbl_full
    pred_aligned, bs_rmsd = align_atom_arrays(
        mbl_sele=pred_bs_sorted,  # pred binding site (to be aligned)
        tgt_sele=sample_bs_sorted,   # ref binding site (target)
        mbl_full=pred_array       # full pred structure (to be transformed)
    )

    # Get ligand atoms
    sample_lig_mask = np.isin(sample_array.pn_unit_iid, ligand_pn_unit_iids) & (sample_array.element != "H")
    pred_lig_mask = np.isin(pred_aligned.pn_unit_iid, ligand_pn_unit_iids) & (pred_aligned.element != "H")

    sample_lig = sample_array[sample_lig_mask]
    pred_lig = pred_aligned[pred_lig_mask]

    if len(sample_lig) == 0 or len(pred_lig) == 0:
        return {"error": "No ligand atoms found", "ligand_rmsd": None}

    # Match ligand atoms by name
    sample_atom_names = sample_lig.atom_name
    pred_atom_names = pred_lig.atom_name
    common_atom_names = np.intersect1d(sample_atom_names, pred_atom_names)

    if len(common_atom_names) == 0:
        return {"error": "No common ligand atoms", "ligand_rmsd": None}

    # Calculate symmetry-corrected RMSD using RDKit
    ligand_rmsd = None
    try:
        # Convert ligand atom arrays to RDKit molecules
        sample_lig_full = sample_array[np.isin(sample_array.pn_unit_iid, ligand_pn_unit_iids)]
        pred_lig_full = pred_aligned[np.isin(pred_aligned.pn_unit_iid, ligand_pn_unit_iids)]

        # Use atom_array_to_rdkit with sanitize fallback
        try:
            sample_mol = atom_array_to_rdkit(sample_lig_full, sanitize=True)
            print("Sample ligand sanitization successful")
        except Exception:
            sample_mol = atom_array_to_rdkit(sample_lig_full, sanitize=False)
            print("Sample ligand sanitization failed, not using sanitization fallback")

        try:
            pred_mol = atom_array_to_rdkit(pred_lig_full, sanitize=True)
            print("Pred ligand sanitization successful")
        except Exception:
            pred_mol = atom_array_to_rdkit(pred_lig_full, sanitize=False)
            print("Pred ligand sanitization failed, not using sanitization fallback")

        if sample_mol and pred_mol:
            # Remove hydrogens for RMSD calculation
            sample_mol = Chem.RemoveHs(sample_mol)
            pred_mol = Chem.RemoveHs(pred_mol)

            # Try substructure match first
            match = sample_mol.GetSubstructMatch(pred_mol)
            if match:
                ligand_rmsd = rdMolAlign.CalcRMS(sample_mol, pred_mol)
                print(f"Substructure match found, symmetry-corrected RMSD: {ligand_rmsd:.4f} Å")
            else:
                ligand_rmsd = AllChem.GetBestRMS(sample_mol, pred_mol)
                print(f"No substructure match found, using GetBestRMS: {ligand_rmsd:.4f} Å")

    except Exception:
        return {"error": "Failed to calculate ligand RMSD using RDKit", "ligand_rmsd": None}

    # Calculate best ligand RMSD
    # Save pocket-aligned structure if requested
    aligned_path = None
    if save_aligned:
        # Create output path with "_pocket_aligned" suffix
        aligned_path = Path(pred_path).parent / f"{Path(pred_path).stem}_pocket_aligned.cif"
        try:
            to_cif_file(
                pred_aligned,
                aligned_path,
                include_entity_poly=True,
                include_entity_nonpoly=True,
                include_nan_coords=False,
                include_bonds=True,
            )
        except Exception as e:
            print(f"Warning: Failed to save aligned structure: {e}")
            aligned_path = None

    #! return pred_array and masks for pLDDT extraction
    # Create ligand and binding site masks for the aligned pred structure
    pred_ligand_mask = np.isin(pred_aligned.pn_unit_iid, ligand_pn_unit_iids)
    pred_binding_site_mask = (pred_aligned.is_ligand_pocket == True) & (pred_aligned.res_name != "UNK")

    return {
        "ligand_rmsd": ligand_rmsd,
        "binding_site_rmsd": bs_rmsd,
        "num_bs_residues": int(num_bs_residues),
        "num_matched_atoms": len(common_atom_names),
        "aligned_path": str(aligned_path) if aligned_path else None,
        "aligned_pred_array": pred_aligned,
        "pred_ligand_mask": pred_ligand_mask,
        "pred_binding_site_mask": pred_binding_site_mask,
    }
