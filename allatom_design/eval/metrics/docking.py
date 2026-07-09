import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolAlign
from rdkit.Geometry import Point3D

from atomworks.io.tools.rdkit import atom_array_to_rdkit
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from atomworks.ml.utils.geometry import align_atom_arrays

from allatom_design.data.const import METAL_ELEMENTS
from allatom_design.data.transform.custom_transforms import (
    count_metal_coordinating_protein_residues,
)

from allatom_design.eval.metrics.af3_confidence import extract_af3_confidence_metrics
from allatom_design.eval.utils.pocket_constraints import (
    annotate_ligand_pocket,
    resolve_pocket_annotation_method,
)
from allatom_design.eval.chemical_components import normalize_ccd_code


def _docking_metric_error(
    message: str,
    ligand_ccd_code: str | None = None,
    metric_metadata: dict[str, float | int | str | None] | None = None,
    extra_metric_defaults: dict[str, float | int | str | None] | None = None,
) -> dict[str, float | int | str | None]:
    result = {
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
    if extra_metric_defaults:
        result.update(extra_metric_defaults)
    if metric_metadata:
        result.update(metric_metadata)
        if ligand_ccd_code is not None:
            result["ligand_ccd_code"] = ligand_ccd_code
    return result


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


def _metadata_for_ligand_metric(
    *,
    ligand_pn_unit_iids: list[str],
    reference_ligand_pn_unit_iids: list[str] | None,
    ligand_smiles: list[str | None] | None,
    ligand_rmsd_mode: str | None = None,
) -> dict[str, str | None]:
    nonempty_smiles = [
        str(smiles).strip()
        for smiles in ligand_smiles or []
        if str(smiles or "").strip() and str(smiles).strip().lower() != "nan"
    ]
    ligand_input_type = "smiles" if nonempty_smiles else "ccd_code"
    metadata = {
        "ligand_input_type": ligand_input_type,
        "ligand_pn_unit_iids": _join_unique([str(iid) for iid in ligand_pn_unit_iids]),
        "reference_ligand_pn_unit_iids": _join_unique(
            [str(iid) for iid in reference_ligand_pn_unit_iids or []]
        ),
    }
    if ligand_rmsd_mode is not None:
        metadata["ligand_rmsd_mode"] = ligand_rmsd_mode
    return metadata


def _nonempty_ligand_smiles(ligand_smiles: list[str | None] | None) -> list[str]:
    return [
        str(smiles).strip()
        for smiles in ligand_smiles or []
        if str(smiles or "").strip() and str(smiles).strip().lower() != "nan"
    ]


def _selected_metal_pn_unit_iids(atom_array: AtomArray, ligand_pn_unit_iids: list[str]) -> list[str]:
    metal_pn_unit_iids = []
    for ligand_pn_unit_iid in ligand_pn_unit_iids:
        mask = (atom_array.pn_unit_iid == ligand_pn_unit_iid) & (atom_array.element != "H")
        if not np.any(mask):
            continue
        ligand_atoms = atom_array[mask]
        ligand_elements = [normalize_ccd_code(element) for element in ligand_atoms.element]
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


def _receptor_ca_mask(atom_array: AtomArray, receptor_pn_unit_iids: list[str]) -> np.ndarray:
    receptor_mask = np.isin(atom_array.pn_unit_iid, receptor_pn_unit_iids)
    return receptor_mask & (atom_array.atom_name == "CA") & (atom_array.res_name != "UNK")


def _ca_by_pn_unit_res_offset(
    atom_array: AtomArray,
    ca_mask: np.ndarray,
) -> dict[tuple[str, int], int]:
    ca_indices = np.where(ca_mask)[0]
    pn_unit_min_res_id = {}
    for idx in ca_indices:
        pn_unit_iid = str(atom_array.pn_unit_iid[idx])
        res_id = int(atom_array.res_id[idx])
        pn_unit_min_res_id[pn_unit_iid] = min(
            res_id,
            pn_unit_min_res_id.get(pn_unit_iid, res_id),
        )
    return {
        (
            str(atom_array.pn_unit_iid[idx]),
            int(atom_array.res_id[idx]) - pn_unit_min_res_id[str(atom_array.pn_unit_iid[idx])],
        ): idx
        for idx in ca_indices
    }


def _joint_resolved_reference_sample_pred_ca_indices(
    *,
    reference_atom_array: AtomArray,
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    receptor_pn_unit_iids: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str | None]:
    reference_ca_initial = _receptor_ca_mask(reference_atom_array, receptor_pn_unit_iids)
    sample_ca_initial = _receptor_ca_mask(sample_atom_array, receptor_pn_unit_iids)
    pred_ca_initial = _receptor_ca_mask(pred_atom_array, receptor_pn_unit_iids)

    if reference_ca_initial.sum() == 0:
        return None, None, None, "No reference CA atoms found"
    if sample_ca_initial.sum() == 0 or pred_ca_initial.sum() == 0:
        return None, None, None, "No CA atoms found"

    reference_ca_by_key = _ca_by_pn_unit_res_offset(reference_atom_array, reference_ca_initial)
    sample_ca_by_key = _ca_by_pn_unit_res_offset(sample_atom_array, sample_ca_initial)
    pred_ca_by_key = _ca_by_pn_unit_res_offset(pred_atom_array, pred_ca_initial)

    common_keys = [
        key for key in reference_ca_by_key
        if key in sample_ca_by_key and key in pred_ca_by_key
    ]
    if len(common_keys) == 0:
        return None, None, None, "No common receptor CA residue offsets found"

    reference_ca_indices = np.array([reference_ca_by_key[key] for key in common_keys])
    sample_ca_indices = np.array([sample_ca_by_key[key] for key in common_keys])
    pred_ca_indices = np.array([pred_ca_by_key[key] for key in common_keys])

    reference_ca_resolved = ~np.isnan(reference_atom_array[reference_ca_indices].coord[:, 0])
    sample_ca_resolved = ~np.isnan(sample_atom_array[sample_ca_indices].coord[:, 0])
    pred_ca_resolved = ~np.isnan(pred_atom_array[pred_ca_indices].coord[:, 0])
    ca_resolved = reference_ca_resolved & sample_ca_resolved & pred_ca_resolved

    if not np.any(ca_resolved):
        return None, None, None, "No resolved CA atoms found"

    return (
        reference_ca_indices[ca_resolved],
        sample_ca_indices[ca_resolved],
        pred_ca_indices[ca_resolved],
        None,
    )


def _metal_atom_key(atom_array: AtomArray, idx: int) -> tuple[str, str, str, str]:
    return (
        str(atom_array.pn_unit_iid[idx]),
        str(atom_array.res_name[idx]),
        str(atom_array.atom_name[idx]),
        normalize_ccd_code(atom_array.element[idx]),
    )


def _matched_metal_atom_masks(
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    metal_pn_unit_iids: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[str, str, str, str]] | None, str | None]:
    sample_indices = np.where(
        np.isin(sample_atom_array.pn_unit_iid, metal_pn_unit_iids)
        & np.isin([normalize_ccd_code(element) for element in sample_atom_array.element], list(METAL_ELEMENTS))
    )[0]
    pred_indices = np.where(
        np.isin(pred_atom_array.pn_unit_iid, metal_pn_unit_iids)
        & np.isin([normalize_ccd_code(element) for element in pred_atom_array.element], list(METAL_ELEMENTS))
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


def _annotate_reference_ligand_pocket(
    *,
    reference_atom_array: AtomArray,
    pocket_distance: float,
    annotation_name: str,
    receptor_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
    method: str,
    n_min_ligand_atoms: int = 5,
) -> AtomArray:
    try:
        resolved_method = resolve_pocket_annotation_method(method)
    except ValueError as exc:
        raise ValueError(
            "reference_pocket_annotation_method must be one of "
            "'all_atom', 'calpha', or 'pseudocb', got "
            f"{method!r}"
        ) from exc
    annotated = annotate_ligand_pocket(
        atom_array=reference_atom_array,
        pocket_distance=pocket_distance,
        n_min_ligand_atoms=n_min_ligand_atoms,
        annotation_name=annotation_name,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
        pocket_annotation_method=resolved_method,
    )

    pocket_mask = apply_and_spread_residue_wise(
        annotated,
        annotated.get_annotation(annotation_name),
        function=np.any,
    )
    annotated.set_annotation(annotation_name, pocket_mask)
    return annotated


def _align_prediction_on_global_receptor_ca(
    *,
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    sample_ca_indices: np.ndarray,
    pred_ca_indices: np.ndarray,
) -> tuple[AtomArray | None, str | None]:
    sample_ca = sample_atom_array[sample_ca_indices]
    pred_ca = pred_atom_array[pred_ca_indices]
    if not (sample_ca.res_name == pred_ca.res_name).all():
        return None, "Amino acid residues in sample and pred receptor CA atoms must match"
    pred_aligned_atom_array, _ = align_atom_arrays(
        mbl_sele=pred_ca,
        tgt_sele=sample_ca,
        mbl_full=pred_atom_array,
    )
    return pred_aligned_atom_array, None


def _binding_site_ca_rmsd_after_global_alignment(
    *,
    sample_atom_array: AtomArray,
    pred_aligned_atom_array: AtomArray,
    sample_ca_indices: np.ndarray,
    pred_ca_indices: np.ndarray,
    reference_binding_site_ca_mask: np.ndarray,
) -> float:
    sample_bs_ca = sample_atom_array[sample_ca_indices][reference_binding_site_ca_mask]
    pred_bs_ca = pred_aligned_atom_array[pred_ca_indices][reference_binding_site_ca_mask]
    coord_delta = pred_bs_ca.coord - sample_bs_ca.coord
    return float(np.sqrt(np.mean(np.sum(coord_delta * coord_delta, axis=1))))


def _pred_binding_site_mask_from_reference(
    *,
    reference_atom_array: AtomArray,
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    pocket_distance: float,
    receptor_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
    pocket_annotation_method: str,
    n_min_ligand_atoms: int,
) -> tuple[np.ndarray | None, int, str | None]:
    annotation_name = f"is_ligand_pocket_for_plddt_{float(pocket_distance)}"
    reference_atom_array = _annotate_reference_ligand_pocket(
        reference_atom_array=reference_atom_array,
        pocket_distance=pocket_distance,
        annotation_name=annotation_name,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
        method=pocket_annotation_method,
        n_min_ligand_atoms=n_min_ligand_atoms,
    )
    reference_binding_site_residue_mask = (
        reference_atom_array.get_annotation(annotation_name)
        & np.isin(reference_atom_array.pn_unit_iid, receptor_pn_unit_iids)
    )
    reference_ca_indices, _, pred_ca_indices, error = _joint_resolved_reference_sample_pred_ca_indices(
        reference_atom_array=reference_atom_array,
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
    )
    if error:
        return None, 0, error
    reference_binding_site_ca_mask = reference_binding_site_residue_mask[reference_ca_indices]
    n_residues = int(reference_binding_site_ca_mask.sum())
    pred_binding_site_ca_full_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_binding_site_ca_full_mask[pred_ca_indices[reference_binding_site_ca_mask]] = True
    pred_binding_site_mask = (
        _spread_atom_mask_by_residue(pred_atom_array, pred_binding_site_ca_full_mask)
        & np.isin(pred_atom_array.pn_unit_iid, receptor_pn_unit_iids)
        & (pred_atom_array.res_name != "UNK")
    )
    return pred_binding_site_mask, n_residues, None


def _binding_site_plddt_metric_names_for_distance(distance: float) -> tuple[str, str]:
    suffix = str(float(distance))
    return f"binding_site_plddt_{suffix}", f"binding_site_n_residues_{suffix}"


def _binding_site_plddt_metric_names_for_bin(lo: float, hi: float) -> tuple[str, str]:
    suffix = f"{float(lo)}_to_{float(hi)}"
    return f"binding_site_plddt_bin_{suffix}", f"binding_site_n_residues_bin_{suffix}"


def _configured_binding_site_plddt_defaults(
    *,
    pocket_distances: Sequence[float] | None,
    pocket_distance_bins: Sequence[tuple[float, float]] | None,
) -> dict[str, float | int | None]:
    defaults: dict[str, float | int | None] = {}
    for distance in pocket_distances or []:
        metric_key, n_key = _binding_site_plddt_metric_names_for_distance(float(distance))
        defaults[metric_key] = None
        defaults[n_key] = 0
    for lo, hi in pocket_distance_bins or []:
        metric_key, n_key = _binding_site_plddt_metric_names_for_bin(float(lo), float(hi))
        defaults[metric_key] = None
        defaults[n_key] = 0
    return defaults


def _extract_mean_plddt_for_mask(
    *,
    pred_sample_path: str | Path,
    pred_atom_array: AtomArray,
    mask: np.ndarray,
    n_residues: int,
) -> float | None:
    if n_residues == 0:
        return float("nan")
    confidence_dir = str(Path(pred_sample_path).parent)
    stem = str(Path(pred_sample_path).stem)
    full_confidence_file_path = f"{confidence_dir}/{re.sub(r'_model$', '_confidences', stem)}.json"
    try:
        return extract_af3_confidence_metrics(
            confidence_file_path=full_confidence_file_path,
            atom_array=pred_atom_array,
            mask=mask,
            metrics_to_extract="atom_plddts",
            return_mean=True,
        )
    except Exception:
        return None


def _extract_configured_binding_site_plddt_metrics(
    *,
    pred_sample_path: str | Path,
    pred_atom_array: AtomArray,
    sample_atom_array: AtomArray,
    reference_atom_array: AtomArray,
    receptor_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
    pocket_annotation_method: str,
    pocket_distances: Sequence[float] | None,
    pocket_distance_bins: Sequence[tuple[float, float]] | None,
    n_min_ligand_atoms: int,
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    edge_to_mask_and_count: dict[float, tuple[np.ndarray | None, int, str | None]] = {}

    def _get_edge(distance: float) -> tuple[np.ndarray | None, int, str | None]:
        distance = float(distance)
        if distance not in edge_to_mask_and_count:
            edge_to_mask_and_count[distance] = _pred_binding_site_mask_from_reference(
                reference_atom_array=reference_atom_array,
                sample_atom_array=sample_atom_array,
                pred_atom_array=pred_atom_array,
                pocket_distance=distance,
                receptor_pn_unit_iids=receptor_pn_unit_iids,
                ligand_pn_unit_iids=ligand_pn_unit_iids,
                pocket_annotation_method=pocket_annotation_method,
                n_min_ligand_atoms=n_min_ligand_atoms,
            )
        return edge_to_mask_and_count[distance]

    for distance in pocket_distances or []:
        metric_key, n_key = _binding_site_plddt_metric_names_for_distance(float(distance))
        mask, n_residues, error = _get_edge(float(distance))
        metrics[n_key] = n_residues
        metrics[metric_key] = (
            None
            if error or mask is None
            else _extract_mean_plddt_for_mask(
                pred_sample_path=pred_sample_path,
                pred_atom_array=pred_atom_array,
                mask=mask,
                n_residues=n_residues,
            )
        )

    for lo, hi in pocket_distance_bins or []:
        lo_f, hi_f = float(lo), float(hi)
        metric_key, n_key = _binding_site_plddt_metric_names_for_bin(lo_f, hi_f)
        hi_mask, _, hi_error = _get_edge(hi_f)
        if hi_error or hi_mask is None:
            metrics[n_key] = 0
            metrics[metric_key] = None
            continue
        if lo_f == 0.0:
            bin_mask = hi_mask
        else:
            lo_mask, _, lo_error = _get_edge(lo_f)
            if lo_error or lo_mask is None:
                metrics[n_key] = 0
                metrics[metric_key] = None
                continue
            bin_mask = hi_mask & ~lo_mask
        bin_ca_count = int(np.sum(bin_mask & (pred_atom_array.atom_name == "CA")))
        metrics[n_key] = bin_ca_count
        metrics[metric_key] = _extract_mean_plddt_for_mask(
            pred_sample_path=pred_sample_path,
            pred_atom_array=pred_atom_array,
            mask=bin_mask,
            n_residues=bin_ca_count,
        )

    return metrics


def _heavy_ligand_mask(atom_array: AtomArray, ligand_pn_unit_iids: list[str]) -> np.ndarray:
    return np.isin(atom_array.pn_unit_iid, ligand_pn_unit_iids) & (atom_array.element != "H")


def _rdkit_mol_from_ligand_atom_array(ligand_atom_array: AtomArray):
    try:
        mol = atom_array_to_rdkit(ligand_atom_array, sanitize=True)
    except Exception:
        mol = atom_array_to_rdkit(ligand_atom_array, sanitize=False)
    return mol


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


def _mcs_core_rmsd_no_ligand_alignment(
    *,
    reference_ligand_atom_array: AtomArray,
    pred_ligand_atom_array: AtomArray,
    pred_ligand_smiles: str | None = None,
) -> tuple[dict[str, float | int | str | None] | None, str | None]:
    try:
        reference_mol = _rdkit_mol_from_ligand_atom_array(reference_ligand_atom_array)
        pred_mol = (
            _rdkit_mol_from_smiles_with_atom_array_coords(
                pred_ligand_smiles,
                pred_ligand_atom_array,
            )
            or _rdkit_mol_from_ligand_atom_array(pred_ligand_atom_array)
        )
    except Exception as exc:
        return None, f"Failed to build RDKit ligand molecules for MCS RMSD: {exc}"

    if reference_mol is None or pred_mol is None:
        return None, "Failed to build RDKit ligand molecules for MCS RMSD"
    if reference_mol.GetNumAtoms() == 0 or pred_mol.GetNumAtoms() == 0:
        return None, "No ligand atoms available for MCS RMSD"

    # RDKit defaults both ring flags to False. Keep strict ring matching here
    # so cyclic ligand cores are not matched to partial/open-chain fragments.
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
        mcs_query,
        uniquify=True,
        maxMatches=1024,
    )
    pred_matches = pred_mol.GetSubstructMatches(
        mcs_query,
        uniquify=True,
        maxMatches=1024,
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
        best_rmsd = rdMolAlign.CalcRMS(
            pred_mol,
            reference_mol,
            map=atom_maps,
        )
    except Exception as exc:
        return None, f"Failed to calculate MCS RMSD using RDKit CalcRMS: {exc}"

    return {
        "ligand_rmsd": float(best_rmsd),
        "ligand_rmsd_mode": "smiles_mcs_core",
        "mcs_num_atoms": int(mcs.numAtoms),
        "mcs_pred_coverage": float(mcs.numAtoms / max(pred_mol.GetNumAtoms(), 1)),
        "mcs_reference_coverage": float(mcs.numAtoms / max(reference_mol.GetNumAtoms(), 1)),
    }, None


def _compute_metal_docking_metrics_atomarray(
    *,
    pred_atom_array: AtomArray,
    sample_atom_array: AtomArray,
    reference_atom_array: AtomArray,
    pred_sample_path: str | Path,
    pocket_distance_for_docking_metrics: float,
    receptor_pn_unit_iids: list[str],
    metal_pn_unit_iids: list[str],
    ligand_ccd_code: str | None,
    save_aligned: bool = True,
    reference_pocket_annotation_method: str = "all_atom",
    binding_site_plddt_distances: Sequence[float] | None = None,
    binding_site_plddt_bins: Sequence[tuple[float, float]] | None = None,
    binding_site_plddt_reference_atom_array: AtomArray | None = None,
    binding_site_plddt_reference_pocket_annotation_method: str | None = None,
    binding_site_plddt_reference_ligand_pn_unit_iids: list[str] | None = None,
    binding_site_plddt_n_min_ligand_atoms: int = 1,
    metric_metadata: dict[str, float | int | str | None] | None = None,
) -> dict[str, float | int | str | None]:
    extra_metric_defaults = _configured_binding_site_plddt_defaults(
        pocket_distances=binding_site_plddt_distances,
        pocket_distance_bins=binding_site_plddt_bins,
    )
    _, _, _, error = _matched_metal_atom_masks(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        metal_pn_unit_iids=metal_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(
            error,
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    normalized_reference_elements = np.array([
        normalize_ccd_code(element) for element in reference_atom_array.element
    ])
    reference_metal_mask = (
        np.isin(reference_atom_array.pn_unit_iid, metal_pn_unit_iids)
        & np.isin(normalized_reference_elements, list(METAL_ELEMENTS))
        & ~np.isnan(reference_atom_array.coord).any(axis=1)
    )
    reference_metal_coords = reference_atom_array[reference_metal_mask].coord
    if len(reference_metal_coords) == 0:
        return _docking_metric_error(
            "No resolved reference metal coordinates found",
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    reference_atom_array = _annotate_reference_ligand_pocket(
        reference_atom_array=reference_atom_array,
        pocket_distance=pocket_distance_for_docking_metrics,
        annotation_name="is_ligand_pocket_for_metrics",
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=metal_pn_unit_iids,
        method=reference_pocket_annotation_method,
        n_min_ligand_atoms=1,
    )
    reference_binding_site_residue_mask = (
        reference_atom_array.is_ligand_pocket_for_metrics
        & np.isin(reference_atom_array.pn_unit_iid, receptor_pn_unit_iids)
    )

    reference_ca_indices, sample_ca_indices, pred_ca_indices, error = _joint_resolved_reference_sample_pred_ca_indices(
        reference_atom_array=reference_atom_array,
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(
            error,
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    sample_ca = sample_atom_array[sample_ca_indices]
    reference_binding_site_ca_mask = reference_binding_site_residue_mask[reference_ca_indices]

    sample_binding_site_ca = sample_ca[reference_binding_site_ca_mask]

    if len(sample_binding_site_ca) == 0:
        return _docking_metric_error(
            "No binding site CA atoms found",
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )
    pred_aligned_atom_array, error = _align_prediction_on_global_receptor_ca(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        sample_ca_indices=sample_ca_indices,
        pred_ca_indices=pred_ca_indices,
    )
    if error:
        return _docking_metric_error(
            error,
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    bs_rmsd = _binding_site_ca_rmsd_after_global_alignment(
        sample_atom_array=sample_atom_array,
        pred_aligned_atom_array=pred_aligned_atom_array,
        sample_ca_indices=sample_ca_indices,
        pred_ca_indices=pred_ca_indices,
        reference_binding_site_ca_mask=reference_binding_site_ca_mask,
    )

    pred_binding_site_ca_full_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_binding_site_ca_full_mask[pred_ca_indices[reference_binding_site_ca_mask]] = True
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
        return _docking_metric_error(
            error,
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    sample_coords = sample_atom_array[sample_metal_mask].coord
    pred_coords = pred_aligned_atom_array[pred_metal_mask].coord
    valid_pair_mask = ~np.isnan(sample_coords).any(axis=1) & ~np.isnan(pred_coords).any(axis=1)
    if not np.any(valid_pair_mask):
        return _docking_metric_error(
            "No resolved matched metal atom pairs found",
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    coord_delta = pred_coords[valid_pair_mask] - sample_coords[valid_pair_mask]
    ligand_rmsd = float(np.sqrt(np.mean(np.sum(coord_delta * coord_delta, axis=1))))

    confidence = _extract_confidence_metrics(
        pred_sample_path=pred_sample_path,
        pred_aligned_atom_array=pred_aligned_atom_array,
        ligand_mask=pred_metal_mask,
        binding_site_mask=pred_binding_site_mask,
    )
    configured_plddt = _extract_configured_binding_site_plddt_metrics(
        pred_sample_path=pred_sample_path,
        pred_atom_array=pred_aligned_atom_array,
        sample_atom_array=sample_atom_array,
        reference_atom_array=(
            binding_site_plddt_reference_atom_array
            if binding_site_plddt_reference_atom_array is not None
            else reference_atom_array
        ),
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=(
            binding_site_plddt_reference_ligand_pn_unit_iids or metal_pn_unit_iids
        ),
        pocket_annotation_method=(
            binding_site_plddt_reference_pocket_annotation_method
            or reference_pocket_annotation_method
        ),
        pocket_distances=binding_site_plddt_distances,
        pocket_distance_bins=binding_site_plddt_bins,
        n_min_ligand_atoms=binding_site_plddt_n_min_ligand_atoms,
    )

    return {
        "ligand_rmsd": ligand_rmsd,
        "ligand_rmsd_mode": "metal",
        "binding_site_rmsd": float(bs_rmsd),
        "num_bs_residues": int(reference_binding_site_ca_mask.sum()),
        **confidence,
        **configured_plddt,
        "ligand_ccd_code": ligand_ccd_code,
        **(metric_metadata or {}),
    }


def _compute_small_molecule_docking_metrics_atomarray(
    *,
    pred_atom_array: AtomArray,
    sample_atom_array: AtomArray,
    reference_atom_array: AtomArray,
    pred_sample_path: str | Path,
    pocket_distance_for_docking_metrics: float,
    receptor_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
    ligand_ccd_code: str | None,
    save_aligned: bool = True,
    reference_pocket_annotation_method: str = "calpha",
    ligand_smiles: list[str | None] | None = None,
    reference_ligand_pn_unit_iids: list[str] | None = None,
    binding_site_plddt_distances: Sequence[float] | None = None,
    binding_site_plddt_bins: Sequence[tuple[float, float]] | None = None,
    binding_site_plddt_reference_atom_array: AtomArray | None = None,
    binding_site_plddt_reference_pocket_annotation_method: str | None = None,
    binding_site_plddt_reference_ligand_pn_unit_iids: list[str] | None = None,
    binding_site_plddt_n_min_ligand_atoms: int = 1,
    metric_metadata: dict[str, float | int | str | None] | None = None,
) -> dict[str, float | int | str | None]:
    extra_metric_defaults = _configured_binding_site_plddt_defaults(
        pocket_distances=binding_site_plddt_distances,
        pocket_distance_bins=binding_site_plddt_bins,
    )
    reference_ligand_pn_unit_iids = reference_ligand_pn_unit_iids or ligand_pn_unit_iids
    has_smiles_ligand = bool(_nonempty_ligand_smiles(ligand_smiles))
    if has_smiles_ligand and (
        len(ligand_pn_unit_iids) != 1 or len(reference_ligand_pn_unit_iids) != 1
    ):
        return _docking_metric_error(
            "SMILES docking metrics require exactly one predicted ligand iid and one reference ligand iid",
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    reference_atom_array = _annotate_reference_ligand_pocket(
        reference_atom_array=reference_atom_array,
        pocket_distance=pocket_distance_for_docking_metrics,
        annotation_name="is_ligand_pocket_for_metrics",
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=reference_ligand_pn_unit_iids,
        method=reference_pocket_annotation_method,
        n_min_ligand_atoms=binding_site_plddt_n_min_ligand_atoms,
    )

    # Get binding site CA atoms for superposition
    # Use sequential residue index (order in chain) instead of res_id for matching
    # because res_id may differ between structures (ref vs AF3 prediction)
    reference_ca_indices, sample_ca_indices, pred_ca_indices, error = _joint_resolved_reference_sample_pred_ca_indices(
        reference_atom_array=reference_atom_array,
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
    )
    if error:
        return _docking_metric_error(
            error,
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    sample_ca = sample_atom_array[sample_ca_indices]
    pred_ca = pred_atom_array[pred_ca_indices]

    # Get binding site mask for CA atoms
    reference_bs_ca_mask = reference_atom_array.is_ligand_pocket_for_metrics[reference_ca_indices]

    sample_bs_sorted = sample_ca[reference_bs_ca_mask]
    num_bs_residues = np.sum(reference_bs_ca_mask)

    if len(sample_bs_sorted) == 0:
        return _docking_metric_error(
            "No binding site CA atoms found",
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    pred_aligned_atom_array, error = _align_prediction_on_global_receptor_ca(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        sample_ca_indices=sample_ca_indices,
        pred_ca_indices=pred_ca_indices,
    )
    if error:
        return _docking_metric_error(
            error,
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    bs_rmsd = _binding_site_ca_rmsd_after_global_alignment(
        sample_atom_array=sample_atom_array,
        pred_aligned_atom_array=pred_aligned_atom_array,
        sample_ca_indices=sample_ca_indices,
        pred_ca_indices=pred_ca_indices,
        reference_binding_site_ca_mask=reference_bs_ca_mask,
    )

    # Prepare masks for ligand and binding site
    sample_ligand_mask = _heavy_ligand_mask(sample_atom_array, ligand_pn_unit_iids)
    reference_ligand_mask = _heavy_ligand_mask(reference_atom_array, reference_ligand_pn_unit_iids)
    pred_ligand_mask = _heavy_ligand_mask(pred_aligned_atom_array, ligand_pn_unit_iids)
    pred_binding_site_ca_full_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_binding_site_ca_full_mask[pred_ca_indices[reference_bs_ca_mask]] = True
    pred_binding_site_mask = (
        _spread_atom_mask_by_residue(pred_aligned_atom_array, pred_binding_site_ca_full_mask)
        & (pred_aligned_atom_array.res_name != "UNK")
    )

    # Get ligand atom arrays from sample and pred
    sample_ligand_atom_array = sample_atom_array[sample_ligand_mask]
    reference_ligand_atom_array = reference_atom_array[reference_ligand_mask]
    pred_ligand_atom_array = pred_aligned_atom_array[pred_ligand_mask]

    if len(pred_ligand_atom_array) == 0:
        return _docking_metric_error(
            "No predicted ligand atoms found",
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )

    ligand_rmsd_metadata: dict[str, float | int | str | None] = {}
    if has_smiles_ligand:
        if len(reference_ligand_atom_array) == 0:
            return _docking_metric_error(
                "No explicit reference ligand atoms found for SMILES metric",
                ligand_ccd_code=ligand_ccd_code,
                metric_metadata=metric_metadata,
                extra_metric_defaults=extra_metric_defaults,
            )
        reference_ligand_ccd_code = _ligand_ccd_code_for_iids(
            reference_atom_array,
            reference_ligand_pn_unit_iids,
        )
        mcs_metrics, error = _mcs_core_rmsd_no_ligand_alignment(
            reference_ligand_atom_array=reference_ligand_atom_array,
            pred_ligand_atom_array=pred_ligand_atom_array,
            pred_ligand_smiles=_nonempty_ligand_smiles(ligand_smiles)[0],
        )
        if error:
            return _docking_metric_error(
                error,
                ligand_ccd_code=ligand_ccd_code,
                metric_metadata=metric_metadata,
                extra_metric_defaults=extra_metric_defaults,
            )
        ligand_rmsd = mcs_metrics.pop("ligand_rmsd")
        ligand_rmsd_metadata = {
            **mcs_metrics,
            "reference_ligand_ccd_code": reference_ligand_ccd_code,
        }
    else:
        if len(sample_ligand_atom_array) == 0:
            return _docking_metric_error(
                "No ligand atoms found",
                ligand_ccd_code=ligand_ccd_code,
                metric_metadata=metric_metadata,
                extra_metric_defaults=extra_metric_defaults,
            )

        # Match ligand atoms by name
        sample_ligand_atom_names = sample_ligand_atom_array.atom_name
        pred_ligand_atom_names = pred_ligand_atom_array.atom_name
        common_atom_names = np.intersect1d(sample_ligand_atom_names, pred_ligand_atom_names)

        if len(common_atom_names) == 0:
            return _docking_metric_error(
                "No common ligand atoms",
                ligand_ccd_code=ligand_ccd_code,
                metric_metadata=metric_metadata,
                extra_metric_defaults=extra_metric_defaults,
            )

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
                    # This is what we want for docking poses after receptor superposition
                    ligand_rmsd = rdMolAlign.CalcRMS(sample_mol, pred_mol)
                    print(f"using CalcRMS (no alignment, symmetry-aware): {ligand_rmsd:.4f} Å")
                    ligand_rmsd_metadata["ligand_rmsd_mode"] = "ccd_exact"
                except:
                    print(f"CalcRMS failed using (sample_mol, pred_mol), sample_mol: {sample_mol.GetNumHeavyAtoms()}, pred_mol: {pred_mol.GetNumHeavyAtoms()}")
                    print(f"This is i) because the number of heavy atoms of sample_mol can be modified because of atomworks preprocessing")
                    print(f"This is ii) or the ligand structure of AF3 prediction is wrong, e.g.) RI2 in 5yft")
                    print(f"In this case, sample_mol can be not a substructure of pred_mol, thus giving CalcRMS error")
                    print(f"So trying (pred_mol, sample_mol) instead")
                    try:
                        ligand_rmsd = rdMolAlign.CalcRMS(pred_mol, sample_mol)
                        print(f"using CalcRMS (no alignment, symmetry-aware): {ligand_rmsd:.4f} Å")
                        ligand_rmsd_metadata["ligand_rmsd_mode"] = "ccd_exact"
                    except Exception as e:
                        print(f"Both directions failed, cannot compute RMSD")
                        print(f"Error: {e}")

        except Exception as e:
            print(f"Failed to calculate ligand RMSD: {e}")
            return _docking_metric_error(
                "Failed to calculate ligand RMSD using RDKit",
                ligand_ccd_code=ligand_ccd_code,
                metric_metadata=metric_metadata,
                extra_metric_defaults=extra_metric_defaults,
            )


    # Calculate AF3 confidence metrics using the aligned pred structure
    confidence = _extract_confidence_metrics(
        pred_sample_path=pred_sample_path,
        pred_aligned_atom_array=pred_aligned_atom_array,
        ligand_mask=pred_ligand_mask,
        binding_site_mask=pred_binding_site_mask,
    )
    configured_plddt = _extract_configured_binding_site_plddt_metrics(
        pred_sample_path=pred_sample_path,
        pred_atom_array=pred_aligned_atom_array,
        sample_atom_array=sample_atom_array,
        reference_atom_array=(
            binding_site_plddt_reference_atom_array
            if binding_site_plddt_reference_atom_array is not None
            else reference_atom_array
        ),
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=(
            binding_site_plddt_reference_ligand_pn_unit_iids or reference_ligand_pn_unit_iids
        ),
        pocket_annotation_method=(
            binding_site_plddt_reference_pocket_annotation_method
            or reference_pocket_annotation_method
        ),
        pocket_distances=binding_site_plddt_distances,
        pocket_distance_bins=binding_site_plddt_bins,
        n_min_ligand_atoms=binding_site_plddt_n_min_ligand_atoms,
    )

    return {
        "ligand_rmsd": ligand_rmsd,
        "binding_site_rmsd": float(bs_rmsd),
        "num_bs_residues": int(num_bs_residues),
        **confidence,
        **configured_plddt,
        "ligand_ccd_code": ligand_ccd_code,
        **(metric_metadata or {}),
        **ligand_rmsd_metadata,
    }


def compute_docking_metrics_atomarray(*, pred_atom_array: AtomArray,
                                       sample_atom_array: AtomArray,
                                       reference_atom_array: AtomArray | None = None,
                                       pred_sample_path: str = None,
                                       pocket_distance_for_docking_metrics: float = 6.0,
                                       receptor_pn_unit_iids: list = ["A_1"],
                                       ligand_pn_unit_iids: list = ["C_1"],
                                       ligand_ccd_codes: list[str] | None = None,
                                       ligand_smiles: list[str | None] | None = None,
                                       reference_ligand_pn_unit_iids: list[str] | None = None,
                                       save_aligned: bool = True,
                                       ref_sample_is_designed: bool = True,
                                       reference_pocket_annotation_method: str | None = None,
                                       metal_pn_unit_iids: list[str] | None = None,
                                       binding_site_plddt_distances: Sequence[float] | None = None,
                                       binding_site_plddt_bins: Sequence[tuple[float, float]] | None = None,
                                       binding_site_plddt_reference_atom_array: AtomArray | None = None,
                                       binding_site_plddt_reference_pocket_annotation_method: str | None = None,
                                       binding_site_plddt_reference_ligand_pn_unit_iids: list[str] | None = None,
                                       binding_site_plddt_n_min_ligand_atoms: int = 1,
                                       ) -> dict[str, float]:
    """
    Compute docking metrics between a designed structure and its predicted structure, using atom array.

    Dispatches to the metal-specific or small-molecule-specific implementation based on whether
    the selected ligand pn_unit(s) resolve to single metal atoms.

    Args:
        reference_atom_array: Structure used to annotate the binding-site residues before
            projecting corresponding CA residues into sample and prediction. If None, uses
            sample_atom_array.
        reference_pocket_annotation_method: Pocket annotation method for reference_atom_array.
            If None, designed references default to "calpha" for both metals and
            small molecules; non-designed references default to "all_atom".
        ligand_smiles: Optional SMILES strings aligned to ligand_pn_unit_iids. Non-empty
            values use explicit-reference-ligand MCS-core RMSD instead of CCD exact RMSD.
        reference_ligand_pn_unit_iids: Optional explicit ligand pn_unit_iids in
            reference_atom_array. Defaults to ligand_pn_unit_iids.
        ref_sample_is_designed: Selects the default reference pocket annotation
            method when reference_pocket_annotation_method is None.
        metal_pn_unit_iids: Optional predicted metal pn_unit_iids for the direct
            predicted metal donor-residue count. If None, predicted single-atom
            metal pn_unit_iids are auto-detected.
    """
    if reference_atom_array is None:
        reference_atom_array = sample_atom_array
    reference_ligand_pn_unit_iids = reference_ligand_pn_unit_iids or ligand_pn_unit_iids
    ligand_ccd_code = _ligand_ccd_code_for_iids(sample_atom_array, ligand_pn_unit_iids, ligand_ccd_codes)
    metric_metadata = _metadata_for_ligand_metric(
        ligand_pn_unit_iids=ligand_pn_unit_iids,
        reference_ligand_pn_unit_iids=reference_ligand_pn_unit_iids,
        ligand_smiles=ligand_smiles,
    )
    metal_donor_residue_metadata = {
        "metal_num_predicted_coordinating_protein_residues": count_metal_coordinating_protein_residues(
            atom_array=pred_atom_array,
            receptor_pn_unit_iids=receptor_pn_unit_iids,
            metal_pn_unit_iids=metal_pn_unit_iids,
        )
    }
    metric_metadata.update(metal_donor_residue_metadata)
    extra_metric_defaults = _configured_binding_site_plddt_defaults(
        pocket_distances=binding_site_plddt_distances,
        pocket_distance_bins=binding_site_plddt_bins,
    )
    if ligand_smiles is not None and len(ligand_smiles) != len(ligand_pn_unit_iids):
        return _docking_metric_error(
            "ligand_smiles must have the same length as ligand_pn_unit_iids",
            ligand_ccd_code=ligand_ccd_code,
            metric_metadata=metric_metadata,
            extra_metric_defaults=extra_metric_defaults,
        )
    selected_sample_metal_pn_unit_iids = _selected_metal_pn_unit_iids(sample_atom_array, ligand_pn_unit_iids)
    if reference_pocket_annotation_method is None:
        reference_pocket_annotation_method = "calpha" if ref_sample_is_designed else "all_atom"
    metal_reference_pocket_annotation_method = reference_pocket_annotation_method
    small_molecule_reference_pocket_annotation_method = reference_pocket_annotation_method

    if selected_sample_metal_pn_unit_iids:
        metal_ligand_ccd_code = _ligand_ccd_code_for_iids(
            sample_atom_array,
            selected_sample_metal_pn_unit_iids,
            [
                str(code)
                for pn_unit_iid, code in zip(ligand_pn_unit_iids, ligand_ccd_codes or [])
                if str(pn_unit_iid) in set(selected_sample_metal_pn_unit_iids)
            ] or None,
        )
        metal_metadata = _metadata_for_ligand_metric(
            ligand_pn_unit_iids=selected_sample_metal_pn_unit_iids,
            reference_ligand_pn_unit_iids=selected_sample_metal_pn_unit_iids,
            ligand_smiles=None,
        )
        metal_metadata.update(metal_donor_residue_metadata)
        return _compute_metal_docking_metrics_atomarray(
            pred_atom_array=pred_atom_array,
            sample_atom_array=sample_atom_array,
            reference_atom_array=reference_atom_array,
            pred_sample_path=pred_sample_path,
            pocket_distance_for_docking_metrics=pocket_distance_for_docking_metrics,
            receptor_pn_unit_iids=receptor_pn_unit_iids,
            metal_pn_unit_iids=selected_sample_metal_pn_unit_iids,
            ligand_ccd_code=metal_ligand_ccd_code,
            save_aligned=save_aligned,
            reference_pocket_annotation_method=metal_reference_pocket_annotation_method,
            binding_site_plddt_distances=binding_site_plddt_distances,
            binding_site_plddt_bins=binding_site_plddt_bins,
            binding_site_plddt_reference_atom_array=binding_site_plddt_reference_atom_array,
            binding_site_plddt_reference_pocket_annotation_method=(
                binding_site_plddt_reference_pocket_annotation_method
            ),
            binding_site_plddt_reference_ligand_pn_unit_iids=(
                binding_site_plddt_reference_ligand_pn_unit_iids
            ),
            binding_site_plddt_n_min_ligand_atoms=binding_site_plddt_n_min_ligand_atoms,
            metric_metadata=metal_metadata,
        )

    return _compute_small_molecule_docking_metrics_atomarray(
        pred_atom_array=pred_atom_array,
        sample_atom_array=sample_atom_array,
        reference_atom_array=reference_atom_array,
        pred_sample_path=pred_sample_path,
        pocket_distance_for_docking_metrics=pocket_distance_for_docking_metrics,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
        ligand_ccd_code=ligand_ccd_code,
        save_aligned=save_aligned,
        reference_pocket_annotation_method=small_molecule_reference_pocket_annotation_method,
        ligand_smiles=ligand_smiles,
        reference_ligand_pn_unit_iids=reference_ligand_pn_unit_iids,
        binding_site_plddt_distances=binding_site_plddt_distances,
        binding_site_plddt_bins=binding_site_plddt_bins,
        binding_site_plddt_reference_atom_array=binding_site_plddt_reference_atom_array,
        binding_site_plddt_reference_pocket_annotation_method=(
            binding_site_plddt_reference_pocket_annotation_method
        ),
        binding_site_plddt_reference_ligand_pn_unit_iids=(
            binding_site_plddt_reference_ligand_pn_unit_iids
        ),
        binding_site_plddt_n_min_ligand_atoms=binding_site_plddt_n_min_ligand_atoms,
        metric_metadata=metric_metadata,
    )
