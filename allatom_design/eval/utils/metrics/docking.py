import re
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import rdMolAlign

import atomworks.enums as aw_enums
from atomworks.constants import METAL_ELEMENTS
from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from atomworks.ml.utils.geometry import align_atom_arrays

from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets, annotate_ligand_pockets_calpha
from allatom_design.utils.sample_io_utils import save_cif_file

from allatom_design.eval.utils.metrics.af3_confidence import extract_af3_confidence_metrics


def _docking_metric_error(message: str, ligand_ccd_code: str | None = None) -> dict[str, float | int | str | None]:
    return {
        "error": message,
        "ligand_rmsd": None,
        "binding_site_rmsd": None,
        "num_bs_residues": 0,
        "ligand_plddt": None,
        "binding_site_plddt": None,
        "iptm": None,
        "interface_min_pae": None,
        "ligand_ccd_code": ligand_ccd_code,
    }


def _normalise_element(element: object) -> str:
    return str(element).strip().upper()


def _join_unique(values: list[str]) -> str | None:
    unique_values = []
    for value in values:
        if value and value not in unique_values:
            unique_values.append(value)
    return ";".join(unique_values) if unique_values else None


def _ligand_ccd_code_for_iids(
    atom_array: AtomArray,
    ligand_pn_unit_iids: list[str],
    ligand_ccd_codes: list[str] | None = None,
) -> str | None:
    if ligand_ccd_codes is not None and len(ligand_ccd_codes) == len(ligand_pn_unit_iids):
        return _join_unique([str(code) for code in ligand_ccd_codes])

    codes = []
    for ligand_pn_unit_iid in ligand_pn_unit_iids:
        mask = atom_array.pn_unit_iid == ligand_pn_unit_iid
        if np.any(mask):
            codes.append(str(atom_array.res_name[mask][0]))
    return _join_unique(codes)


def _selected_metal_pn_unit_iids(atom_array: AtomArray, ligand_pn_unit_iids: list[str]) -> list[str]:
    metal_pn_unit_iids = []
    for ligand_pn_unit_iid in ligand_pn_unit_iids:
        mask = (atom_array.pn_unit_iid == ligand_pn_unit_iid) & (atom_array.element != "H")
        if not np.any(mask):
            continue
        ligand_atoms = atom_array[mask]
        ligand_elements = [_normalise_element(element) for element in ligand_atoms.element]
        if len(ligand_elements) == 1 and ligand_elements[0] in METAL_ELEMENTS:
            metal_pn_unit_iids.append(str(ligand_pn_unit_iid))
    return metal_pn_unit_iids


def _spread_atom_mask_by_residue(atom_array: AtomArray, atom_mask: np.ndarray) -> np.ndarray:
    residue_mask = np.zeros(len(atom_array), dtype=bool)
    for idx in np.where(atom_mask)[0]:
        same_residue = (
            (atom_array.pn_unit_iid == atom_array.pn_unit_iid[idx])
            & (atom_array.res_id == atom_array.res_id[idx])
        )
        if hasattr(atom_array, "ins_code"):
            same_residue &= atom_array.ins_code == atom_array.ins_code[idx]
        residue_mask |= same_residue
    return residue_mask


def _joint_resolved_receptor_ca_masks(
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    receptor_pn_unit_iids: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    sample_receptor_mask = np.isin(sample_atom_array.pn_unit_iid, receptor_pn_unit_iids)
    pred_receptor_mask = np.isin(pred_atom_array.pn_unit_iid, receptor_pn_unit_iids)

    sample_ca_mask_initial = sample_receptor_mask & (sample_atom_array.atom_name == "CA") & (sample_atom_array.res_name != "UNK")
    pred_ca_mask_initial = pred_receptor_mask & (pred_atom_array.atom_name == "CA") & (pred_atom_array.res_name != "UNK")

    if sample_ca_mask_initial.sum() != pred_ca_mask_initial.sum():
        return None, None, "Number of CA atoms in sample and pred must match"

    if sample_ca_mask_initial.sum() == 0:
        return None, None, "No CA atoms found"

    sample_ca_resolved_mask = ~np.isnan(sample_atom_array[sample_ca_mask_initial].coord[:, 0])
    pred_ca_resolved_mask = ~np.isnan(pred_atom_array[pred_ca_mask_initial].coord[:, 0])
    ca_resolved_mask = sample_ca_resolved_mask & pred_ca_resolved_mask

    sample_ca_indices = np.where(sample_ca_mask_initial)[0]
    sample_ca_mask = np.zeros(len(sample_atom_array), dtype=bool)
    sample_ca_mask[sample_ca_indices[ca_resolved_mask]] = True

    pred_ca_indices = np.where(pred_ca_mask_initial)[0]
    pred_ca_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_ca_mask[pred_ca_indices[ca_resolved_mask]] = True

    if sample_ca_mask.sum() == 0:
        return None, None, "No resolved CA atoms found"

    return sample_ca_mask, pred_ca_mask, None


def _metal_atom_key(atom_array: AtomArray, idx: int) -> tuple[str, str, str, str]:
    return (
        str(atom_array.pn_unit_iid[idx]),
        str(atom_array.res_name[idx]),
        str(atom_array.atom_name[idx]),
        _normalise_element(atom_array.element[idx]),
    )


def _matched_metal_atom_masks(
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    metal_pn_unit_iids: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[str, str, str, str]] | None, str | None]:
    sample_indices = np.where(
        np.isin(sample_atom_array.pn_unit_iid, metal_pn_unit_iids)
        & np.isin([_normalise_element(element) for element in sample_atom_array.element], list(METAL_ELEMENTS))
    )[0]
    pred_indices = np.where(
        np.isin(pred_atom_array.pn_unit_iid, metal_pn_unit_iids)
        & np.isin([_normalise_element(element) for element in pred_atom_array.element], list(METAL_ELEMENTS))
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
            sample_element = _normalise_element(sample_atom_array.element[sample_idx])
            pred_element = _normalise_element(pred_atom_array.element[pred_idx])
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


def _extract_confidence_metrics(
    *,
    pred_sample_path: str | Path,
    pred_aligned_atom_array: AtomArray,
    ligand_mask: np.ndarray,
    binding_site_mask: np.ndarray,
) -> dict[str, float | None]:
    confidence_dir = str(Path(pred_sample_path).parent)
    stem = str(Path(pred_sample_path).stem)
    full_confidence_file_path = f"{confidence_dir}/{re.sub(r'_model$', '_confidences', stem)}.json"
    summary_confidence_file_path = f"{confidence_dir}/{re.sub(r'_model$', '_summary_confidences', stem)}.json"

    return {
        "ligand_plddt": extract_af3_confidence_metrics(
            confidence_file_path=full_confidence_file_path,
            atom_array=pred_aligned_atom_array,
            mask=ligand_mask,
            metrics_to_extract="atom_plddts",
            return_mean=True,
        ),
        "binding_site_plddt": extract_af3_confidence_metrics(
            confidence_file_path=full_confidence_file_path,
            atom_array=pred_aligned_atom_array,
            mask=binding_site_mask,
            metrics_to_extract="atom_plddts",
            return_mean=True,
        ),
        "iptm": extract_af3_confidence_metrics(
            confidence_file_path=summary_confidence_file_path,
            atom_array=pred_aligned_atom_array,
            metrics_to_extract="iptm",
            return_mean=True,
        ),
        "interface_min_pae": extract_af3_confidence_metrics(
            confidence_file_path=summary_confidence_file_path,
            atom_array=pred_aligned_atom_array,
            metrics_to_extract="interface_min_pae",
            return_mean=True,
        ),
    }


def _maybe_save_aligned(
    pred_aligned_atom_array: AtomArray,
    pred_sample_path: str | Path,
    save_aligned: bool,
) -> None:
    if not save_aligned:
        return
    out_file = Path(pred_sample_path).parent / f"{Path(pred_sample_path).stem}_pocket_aligned.cif"
    try:
        save_cif_file(pred_aligned_atom_array, out_file)
    except Exception as exc:
        print(f"Warning: Failed to save aligned structure: {exc}")


def _compute_metal_docking_metrics_atomarray(
    *,
    pred_atom_array: AtomArray,
    sample_atom_array: AtomArray,
    pred_sample_path: str | Path,
    pocket_distance_for_docking_metrics: float,
    receptor_pn_unit_iids: list[str],
    metal_pn_unit_iids: list[str],
    ligand_ccd_code: str | None,
    save_aligned: bool = True,
) -> dict[str, float | int | str | None]:
    sample_metal_mask, _, _, error = _matched_metal_atom_masks(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        metal_pn_unit_iids=metal_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(error, ligand_ccd_code=ligand_ccd_code)

    sample_receptor_heavy_mask = (
        np.isin(sample_atom_array.pn_unit_iid, receptor_pn_unit_iids)
        & (sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
        & (sample_atom_array.element != "H")
        & ~np.isnan(sample_atom_array.coord).any(axis=1)
    )
    sample_metal_coords = sample_atom_array[sample_metal_mask].coord
    valid_sample_metal_coords = sample_metal_coords[~np.isnan(sample_metal_coords).any(axis=1)]
    if len(valid_sample_metal_coords) == 0:
        return _docking_metric_error("No resolved reference metal coordinates found", ligand_ccd_code=ligand_ccd_code)

    receptor_coords = sample_atom_array.coord[sample_receptor_heavy_mask]
    if len(receptor_coords) == 0:
        return _docking_metric_error("No receptor heavy atoms found", ligand_ccd_code=ligand_ccd_code)

    distances = np.linalg.norm(
        receptor_coords[:, None, :] - valid_sample_metal_coords[None, :, :],
        axis=2,
    )
    near_metal_receptor_atoms = np.any(distances <= pocket_distance_for_docking_metrics, axis=1)

    sample_binding_site_atom_mask = np.zeros(len(sample_atom_array), dtype=bool)
    sample_receptor_indices = np.where(sample_receptor_heavy_mask)[0]
    sample_binding_site_atom_mask[sample_receptor_indices[near_metal_receptor_atoms]] = True
    sample_binding_site_residue_mask = _spread_atom_mask_by_residue(
        sample_atom_array,
        sample_binding_site_atom_mask,
    ) & np.isin(sample_atom_array.pn_unit_iid, receptor_pn_unit_iids)

    sample_ca_mask, pred_ca_mask, error = _joint_resolved_receptor_ca_masks(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(error, ligand_ccd_code=ligand_ccd_code)

    sample_ca = sample_atom_array[sample_ca_mask]
    pred_ca = pred_atom_array[pred_ca_mask]
    sample_binding_site_ca_mask = sample_binding_site_residue_mask[sample_ca_mask]

    sample_binding_site_ca = sample_ca[sample_binding_site_ca_mask]
    pred_binding_site_ca = pred_ca[sample_binding_site_ca_mask]

    if len(sample_binding_site_ca) == 0:
        return _docking_metric_error("No binding site CA atoms found", ligand_ccd_code=ligand_ccd_code)
    if not (sample_binding_site_ca.res_name == pred_binding_site_ca.res_name).all():
        return _docking_metric_error(
            "Amino acid residues in sample and pred binding site must match",
            ligand_ccd_code=ligand_ccd_code,
        )

    pred_aligned_atom_array, bs_rmsd = align_atom_arrays(
        mbl_sele=pred_binding_site_ca,
        tgt_sele=sample_binding_site_ca,
        mbl_full=pred_atom_array,
    )

    pred_binding_site_ca_full_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_ca_indices = np.where(pred_ca_mask)[0]
    pred_binding_site_ca_full_mask[pred_ca_indices[sample_binding_site_ca_mask]] = True
    pred_binding_site_mask = _spread_atom_mask_by_residue(
        pred_aligned_atom_array,
        pred_binding_site_ca_full_mask,
    ) & np.isin(pred_aligned_atom_array.pn_unit_iid, receptor_pn_unit_iids)

    sample_metal_mask, pred_metal_mask, _, error = _matched_metal_atom_masks(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_aligned_atom_array,
        metal_pn_unit_iids=metal_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(error, ligand_ccd_code=ligand_ccd_code)

    sample_coords = sample_atom_array[sample_metal_mask].coord
    pred_coords = pred_aligned_atom_array[pred_metal_mask].coord
    valid_pair_mask = ~np.isnan(sample_coords).any(axis=1) & ~np.isnan(pred_coords).any(axis=1)
    if not np.any(valid_pair_mask):
        return _docking_metric_error("No resolved matched metal atom pairs found", ligand_ccd_code=ligand_ccd_code)

    coord_delta = pred_coords[valid_pair_mask] - sample_coords[valid_pair_mask]
    ligand_rmsd = float(np.sqrt(np.mean(np.sum(coord_delta * coord_delta, axis=1))))

    confidence = _extract_confidence_metrics(
        pred_sample_path=pred_sample_path,
        pred_aligned_atom_array=pred_aligned_atom_array,
        ligand_mask=pred_metal_mask,
        binding_site_mask=pred_binding_site_mask,
    )
    _maybe_save_aligned(pred_aligned_atom_array, pred_sample_path, save_aligned)

    return {
        "ligand_rmsd": ligand_rmsd,
        "binding_site_rmsd": float(bs_rmsd),
        "num_bs_residues": int(sample_binding_site_ca_mask.sum()),
        **confidence,
        "ligand_ccd_code": ligand_ccd_code,
    }


def _compute_small_molecule_docking_metrics_atomarray(
    *,
    pred_atom_array: AtomArray,
    sample_atom_array: AtomArray,
    pred_sample_path: str | Path,
    pocket_distance_for_docking_metrics: float,
    receptor_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
    ligand_ccd_code: str | None,
    save_aligned: bool = True,
    ref_sample_is_designed: bool = True,
) -> dict[str, float | int | str | None]:
    # Annotate ligand pockets (binding site residues)
    if ref_sample_is_designed:
        sample_atom_array = annotate_ligand_pockets_calpha(atom_array=sample_atom_array,
                                                           pocket_distance=pocket_distance_for_docking_metrics,
                                                           annotation_name="is_ligand_pocket_for_metrics",
                                                           receptor_pn_unit_iids=receptor_pn_unit_iids,
                                                           ligand_pn_unit_iids=ligand_pn_unit_iids)
    else:
        sample_atom_array = annotate_ligand_pockets(atom_array=sample_atom_array,
                                            pocket_distance=pocket_distance_for_docking_metrics,
                                            annotation_name="is_ligand_pocket_for_metrics",
                                            receptor_pn_unit_iids=receptor_pn_unit_iids,
                                            ligand_pn_unit_iids=ligand_pn_unit_iids)

    # Apply and spread residue-wise to get pocket mask
    sample_atom_array_pocket_mask = apply_and_spread_residue_wise(sample_atom_array, sample_atom_array.get_annotation("is_ligand_pocket_for_metrics"), function=np.any)
    sample_atom_array.set_annotation("is_ligand_pocket_for_metrics", sample_atom_array_pocket_mask)

    if ref_sample_is_designed:
        pred_atom_array = annotate_ligand_pockets_calpha(atom_array=pred_atom_array,
                                                           pocket_distance=pocket_distance_for_docking_metrics,
                                                           annotation_name="is_ligand_pocket_for_metrics",
                                                           receptor_pn_unit_iids=receptor_pn_unit_iids,
                                                           ligand_pn_unit_iids=ligand_pn_unit_iids)
    else:
        pred_atom_array = annotate_ligand_pockets(atom_array=pred_atom_array,
                                            pocket_distance=pocket_distance_for_docking_metrics,
                                            annotation_name="is_ligand_pocket_for_metrics",
                                            receptor_pn_unit_iids=receptor_pn_unit_iids,
                                            ligand_pn_unit_iids=ligand_pn_unit_iids)

    # Apply and spread residue-wise to get pocket mask
    pred_atom_array_pocket_mask = apply_and_spread_residue_wise(pred_atom_array, pred_atom_array.get_annotation("is_ligand_pocket_for_metrics"), function=np.any)
    pred_atom_array.set_annotation("is_ligand_pocket_for_metrics", pred_atom_array_pocket_mask)

    # Get binding site CA atoms for superposition
    # Use sequential residue index (order in chain) instead of res_id for matching
    # because res_id may differ between structures (ref vs AF3 prediction)
    sample_ca_mask, pred_ca_mask, error = _joint_resolved_receptor_ca_masks(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(error, ligand_ccd_code=ligand_ccd_code)

    sample_ca = sample_atom_array[sample_ca_mask]
    pred_ca = pred_atom_array[pred_ca_mask]

    # Get binding site mask for CA atoms
    sample_bs_ca_mask = sample_atom_array.is_ligand_pocket_for_metrics[sample_ca_mask]

    # Get binding site CA atoms by sequential index
    sample_bs_sorted = sample_ca[sample_bs_ca_mask]
    pred_bs_sorted = pred_ca[sample_bs_ca_mask]  # Use sample's binding site mask for pred

    # check if the binding site residues in sample and pred match
    assert (sample_bs_sorted.res_name == pred_bs_sorted.res_name).all(), "amino acid residues in sample and pred binding site must match"

    num_bs_residues = np.sum(sample_bs_ca_mask)

    if len(sample_bs_sorted) == 0:
        return _docking_metric_error("No binding site CA atoms found", ligand_ccd_code=ligand_ccd_code)

    # Align pred onto ref using binding site CA atoms
    # align_atom_arrays: aligns mbl_sele to tgt_sele, applies transform to mbl_full
    pred_aligned_atom_array, bs_rmsd = align_atom_arrays(
        mbl_sele=pred_bs_sorted,  # pred binding site (to be aligned)
        tgt_sele=sample_bs_sorted,   # ref binding site (target)
        mbl_full=pred_atom_array       # full pred structure (to be transformed)
    )

    # Prepare masks for ligand and binding site
    sample_ligand_mask = np.isin(sample_atom_array.pn_unit_iid, ligand_pn_unit_iids) & (sample_atom_array.element != "H")
    pred_ligand_mask = np.isin(pred_aligned_atom_array.pn_unit_iid, ligand_pn_unit_iids) & (pred_aligned_atom_array.element != "H")
    pred_binding_site_mask = (pred_aligned_atom_array.is_ligand_pocket_for_metrics == True) & (pred_aligned_atom_array.res_name != "UNK")

    # Get ligand atom arrays from sample and pred
    sample_ligand_atom_array = sample_atom_array[sample_ligand_mask]
    pred_ligand_atom_array = pred_aligned_atom_array[pred_ligand_mask]

    if len(sample_ligand_atom_array) == 0 or len(pred_ligand_atom_array) == 0:
        return _docking_metric_error("No ligand atoms found", ligand_ccd_code=ligand_ccd_code)

    # Match ligand atoms by name
    sample_ligand_atom_names = sample_ligand_atom_array.atom_name
    pred_ligand_atom_names = pred_ligand_atom_array.atom_name
    common_atom_names = np.intersect1d(sample_ligand_atom_names, pred_ligand_atom_names)

    if len(common_atom_names) == 0:
        return _docking_metric_error("No common ligand atoms", ligand_ccd_code=ligand_ccd_code)

    # Calculate symmetry-corrected RMSD using RDKit
    ligand_rmsd = None
    try:
        try:
            sample_mol = atom_array_to_rdkit(sample_ligand_atom_array, sanitize=True)
        except Exception:
            sample_mol = atom_array_to_rdkit(sample_ligand_atom_array, sanitize=False)
        try:
            pred_mol = atom_array_to_rdkit(pred_ligand_atom_array, sanitize=True)
        except Exception:
            pred_mol = atom_array_to_rdkit(pred_ligand_atom_array, sanitize=False)

        if sample_mol and pred_mol:
            # Remove hydrogens for RMSD calculation
            sample_mol = Chem.RemoveHs(sample_mol)
            pred_mol = Chem.RemoveHs(pred_mol)

            try:
                # Use CalcRMS instead of GetBestRMS to compute symmetry-aware RMSD
                # WITHOUT additional alignment (in-place calculation)
                # This is what we want for docking poses after binding site superposition
                ligand_rmsd = rdMolAlign.CalcRMS(sample_mol, pred_mol)
                print(f"using CalcRMS (no alignment, symmetry-aware): {ligand_rmsd:.4f} Å")
            except:
                print(f"CalcRMS failed using (sample_mol, pred_mol), sample_mol: {sample_mol.GetNumHeavyAtoms()}, pred_mol: {pred_mol.GetNumHeavyAtoms()}")
                print(f"This is i) because the number of heavy atoms of sample_mol can be modified because of atomworks preprocessing")
                print(f"This is ii) or the ligand structure of AF3 prediction is wrong, e.g.) RI2 in 5yft")
                print(f"In this case, sample_mol can be not a substructure of pred_mol, thus giving CalcRMS error")
                print(f"So trying (pred_mol, sample_mol) instead")
                try:
                    ligand_rmsd = rdMolAlign.CalcRMS(pred_mol, sample_mol)
                    print(f"using CalcRMS (no alignment, symmetry-aware): {ligand_rmsd:.4f} Å")
                except Exception as e:
                    print(f"Both directions failed, cannot compute RMSD")
                    print(f"Error: {e}")

    except Exception as e:
        print(f"Failed to calculate ligand RMSD: {e}")
        return _docking_metric_error("Failed to calculate ligand RMSD using RDKit", ligand_ccd_code=ligand_ccd_code)


    # Calculate AF3 confidence metrics using the aligned pred structure
    confidence = _extract_confidence_metrics(
        pred_sample_path=pred_sample_path,
        pred_aligned_atom_array=pred_aligned_atom_array,
        ligand_mask=pred_ligand_mask,
        binding_site_mask=pred_binding_site_mask,
    )
    _maybe_save_aligned(pred_aligned_atom_array, pred_sample_path, save_aligned)

    return {
        "ligand_rmsd": ligand_rmsd,
        "binding_site_rmsd": float(bs_rmsd),
        "num_bs_residues": int(num_bs_residues),
        **confidence,
        "ligand_ccd_code": ligand_ccd_code,
    }


def compute_docking_metrics_atomarray(*, pred_atom_array: AtomArray,
                                       sample_atom_array: AtomArray,
                                       pred_sample_path: str = None,
                                       pocket_distance_for_docking_metrics: float = 6.0,
                                       receptor_pn_unit_iids: list = ["A_1"],
                                       ligand_pn_unit_iids: list = ["C_1"],
                                       ligand_ccd_codes: list[str] | None = None,
                                       save_aligned: bool = True,
                                       ref_sample_is_designed: bool = True,
                                       ) -> dict[str, float]:
    """
    Compute docking metrics between a designed structure and its predicted structure, using atom array.

    Dispatches to the metal-specific or small-molecule-specific implementation based on whether
    the selected ligand pn_unit(s) resolve to single metal atoms.
    """
    ligand_ccd_code = _ligand_ccd_code_for_iids(sample_atom_array, ligand_pn_unit_iids, ligand_ccd_codes)
    metal_pn_unit_iids = _selected_metal_pn_unit_iids(sample_atom_array, ligand_pn_unit_iids)
    if metal_pn_unit_iids:
        metal_ligand_ccd_code = _ligand_ccd_code_for_iids(
            sample_atom_array,
            metal_pn_unit_iids,
            [
                str(code)
                for pn_unit_iid, code in zip(ligand_pn_unit_iids, ligand_ccd_codes or [])
                if str(pn_unit_iid) in set(metal_pn_unit_iids)
            ] or None,
        )
        return _compute_metal_docking_metrics_atomarray(
            pred_atom_array=pred_atom_array,
            sample_atom_array=sample_atom_array,
            pred_sample_path=pred_sample_path,
            pocket_distance_for_docking_metrics=pocket_distance_for_docking_metrics,
            receptor_pn_unit_iids=receptor_pn_unit_iids,
            metal_pn_unit_iids=metal_pn_unit_iids,
            ligand_ccd_code=metal_ligand_ccd_code,
            save_aligned=save_aligned,
        )

    return _compute_small_molecule_docking_metrics_atomarray(
        pred_atom_array=pred_atom_array,
        sample_atom_array=sample_atom_array,
        pred_sample_path=pred_sample_path,
        pocket_distance_for_docking_metrics=pocket_distance_for_docking_metrics,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
        ligand_ccd_code=ligand_ccd_code,
        save_aligned=save_aligned,
        ref_sample_is_designed=ref_sample_is_designed,
    )
