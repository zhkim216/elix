from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import atomworks.enums as aw_enums
import numpy as np
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from atomworks.ml.utils.geometry import align_atom_arrays
from biotite.structure import AtomArray
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from allatom_design.data.const import METAL_ELEMENTS
from allatom_design.eval.chemical_components import normalize_ccd_code
from allatom_design.eval.metrics.af3_confidence import extract_af3_confidence_metrics
from allatom_design.eval.metrics.ligand_correspondence import (
    matched_metal_atom_masks as _matched_metal_atom_masks,
    mcs_core_rmsd_no_ligand_alignment as _mcs_core_rmsd_no_ligand_alignment,
    rdkit_mol_from_ligand_atom_array as _rdkit_mol_from_ligand_atom_array,
)
from allatom_design.eval.metrics.protein_correspondence import (
    CaMatch as _CaMatch,
    ResidueKey,
    ThreeWayCaMatch as _ThreeWayCaMatch,
    ca_by_pn_unit_res_id,
    match_reference_designed_predicted_ca,
    match_resolved_ca,
)
from allatom_design.eval.utils.pocket_constraints import annotate_ligand_pocket
from allatom_design.eval.utils.sampling_inputs import normalize_pn_unit_roles
from allatom_design.utils.sample_io_utils import save_cif_file


NATIVE_REFERENCE_POCKET_DISTANCE = 10.0
DESIGNED_REFERENCE_POCKET_DISTANCE = 12.0


ROLE_METRIC_ID_COLUMNS = (
    "input_sample_id",
    "designed_sample_id",
    "prediction_id",
)
ROLE_METRIC_VALUE_COLUMNS = (
    "metric_family",
    "operation_scope",
    "operation_id",
    "status",
    "error",
    "alignment_pn_unit_iids",
    "measurement_pn_unit_iids",
    "target_pn_unit_iid",
    "target_kind",
    "target_res_name",
    "rmsd_mode",
    "ca_rmsd",
    "ca_plddt",
    "ligand_rmsd",
    "ligand_plddt",
    "matched_atom_count",
    "reference_atom_count",
    "prediction_atom_count",
    "reference_match_coverage",
    "prediction_match_coverage",
    "alignment_mode",
    "pocket_annotation_method",
    "pocket_distance",
    "aligned_path",
    "frame_ca_rmsd",
    "frame_matched_ca_count",
    "frame_reference_ca_count",
    "frame_designed_ca_count",
    "frame_prediction_ca_count",
    "frame_reference_match_coverage",
    "frame_designed_match_coverage",
    "frame_prediction_match_coverage",
)
ROLE_METRIC_COLUMNS = (*ROLE_METRIC_ID_COLUMNS, *ROLE_METRIC_VALUE_COLUMNS)


@dataclass(frozen=True)
class RoleMetricOperation:
    metric_family: str
    operation_scope: str
    operation_id: str
    alignment_pn_unit_iids: tuple[str, ...]
    measurement_pn_unit_iids: tuple[str, ...]
    target_pn_unit_iid: str | None = None


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _iid_label(pn_unit_iids: Sequence[str]) -> str:
    return "+".join(str(pn_unit_iid) for pn_unit_iid in pn_unit_iids)


def build_role_metric_plan(
    pn_unit_roles: Mapping[str, Any],
) -> list[RoleMetricOperation]:
    """Derive the default metric operations from the four role lists."""
    roles = normalize_pn_unit_roles(pn_unit_roles)
    binder = roles["binder_pn_unit_iids"]
    context = roles["context_pn_unit_iids"]
    frame = roles["frame_pn_unit_iids"]
    frame_label = _iid_label(frame)

    operations: list[RoleMetricOperation] = []
    for pn_unit_iid in binder:
        operations.append(
            RoleMetricOperation(
                metric_family="intrinsic_sc",
                operation_scope="monomer",
                operation_id=f"intrinsic_sc:{pn_unit_iid}",
                alignment_pn_unit_iids=(pn_unit_iid,),
                measurement_pn_unit_iids=(pn_unit_iid,),
                target_pn_unit_iid=pn_unit_iid,
            )
        )

    if len(binder) > 1:
        binder_tuple = tuple(binder)
        binder_label = _iid_label(binder)
        operations.append(
            RoleMetricOperation(
                metric_family="binder_complex_sc",
                operation_scope="union",
                operation_id=f"binder_complex_sc:{binder_label}",
                alignment_pn_unit_iids=binder_tuple,
                measurement_pn_unit_iids=binder_tuple,
            )
        )
        for pn_unit_iid in binder:
            operations.append(
                RoleMetricOperation(
                    metric_family="binder_complex_sc",
                    operation_scope="member_residual",
                    operation_id=f"binder_complex_sc:{binder_label}->{pn_unit_iid}",
                    alignment_pn_unit_iids=binder_tuple,
                    measurement_pn_unit_iids=(pn_unit_iid,),
                    target_pn_unit_iid=pn_unit_iid,
                )
            )

    frame_tuple = tuple(frame)
    operations.append(
        RoleMetricOperation(
            metric_family="frame_fit",
            operation_scope="union",
            operation_id=f"frame_fit:{frame_label}",
            alignment_pn_unit_iids=frame_tuple,
            measurement_pn_unit_iids=frame_tuple,
        )
    )
    if len(frame) > 1:
        for pn_unit_iid in frame:
            operations.append(
                RoleMetricOperation(
                    metric_family="frame_fit",
                    operation_scope="member_residual",
                    operation_id=f"frame_fit:{frame_label}->{pn_unit_iid}",
                    alignment_pn_unit_iids=frame_tuple,
                    measurement_pn_unit_iids=(pn_unit_iid,),
                    target_pn_unit_iid=pn_unit_iid,
                )
            )

    placement_targets = [
        pn_unit_iid
        for pn_unit_iid in [*binder, *context]
        if pn_unit_iid not in set(frame)
    ]
    if placement_targets:
        for pn_unit_iid in placement_targets:
            operations.append(
                RoleMetricOperation(
                    metric_family="frame_placement",
                    operation_scope="target",
                    operation_id=f"frame_placement:{frame_label}->{pn_unit_iid}",
                    alignment_pn_unit_iids=frame_tuple,
                    measurement_pn_unit_iids=(pn_unit_iid,),
                    target_pn_unit_iid=pn_unit_iid,
                )
            )
    else:
        operations.append(
            RoleMetricOperation(
                metric_family="frame_placement",
                operation_scope="no_targets",
                operation_id=f"frame_placement:{frame_label}->none",
                alignment_pn_unit_iids=frame_tuple,
                measurement_pn_unit_iids=(),
            )
        )
    return operations


def _empty_row(operation: RoleMetricOperation) -> dict[str, Any]:
    row = {column: None for column in ROLE_METRIC_VALUE_COLUMNS}
    row.update(
        {
            "metric_family": operation.metric_family,
            "operation_scope": operation.operation_scope,
            "operation_id": operation.operation_id,
            "status": "pending",
            "error": "",
            "alignment_pn_unit_iids": ";".join(operation.alignment_pn_unit_iids),
            "measurement_pn_unit_iids": ";".join(operation.measurement_pn_unit_iids),
            "target_pn_unit_iid": operation.target_pn_unit_iid,
        }
    )
    return row


def _error_row(operation: RoleMetricOperation, error: str) -> dict[str, Any]:
    row = _empty_row(operation)
    row.update(status="error", error=str(error))
    return row


def _match_ca(
    *,
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    pn_unit_iids: Sequence[str],
) -> _CaMatch:
    return match_resolved_ca(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        pn_unit_iids=pn_unit_iids,
    )


def _as_float(value: Any) -> float:
    try:
        return float(value.item())
    except AttributeError:
        return float(value)


def _validate_frame_geometry(
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    match: _CaMatch,
) -> None:
    if match.matched_count < 3:
        raise ValueError(
            "Frame alignment requires at least 3 matched resolved CA atoms; "
            f"found {match.matched_count}"
        )
    for label, coordinates in (
        ("sample", sample_atom_array.coord[match.sample_indices]),
        ("prediction", pred_atom_array.coord[match.pred_indices]),
    ):
        centered = coordinates - coordinates.mean(axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered, tol=1e-6) < 2:
            raise ValueError(
                "Frame alignment requires non-collinear matched CA atoms; "
                f"{label} frame is collinear"
            )


def _validate_three_way_frame_geometry(
    reference_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    match: _ThreeWayCaMatch,
) -> None:
    if match.matched_count < 3:
        raise ValueError(
            "Pocket alignment requires at least 3 common resolved CA residues; "
            f"found {match.matched_count}"
        )
    for label, coordinates in (
        ("reference", reference_atom_array.coord[match.reference_indices]),
        ("prediction", pred_atom_array.coord[match.pred_indices]),
    ):
        centered = coordinates - coordinates.mean(axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered, tol=1e-6) < 2:
            raise ValueError(
                "Pocket alignment requires non-collinear common CA residues; "
                f"{label} frame is collinear"
            )


def _align_on_match(
    *,
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    match: _CaMatch,
) -> tuple[AtomArray, float]:
    aligned_pred, rmsd = align_atom_arrays(
        mbl_sele=pred_atom_array[match.pred_indices],
        tgt_sele=sample_atom_array[match.sample_indices],
        mbl_full=pred_atom_array,
    )
    return aligned_pred, _as_float(rmsd)


def _align_prediction_to_reference(
    *,
    reference_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    match: _ThreeWayCaMatch,
) -> tuple[AtomArray, float]:
    _validate_three_way_frame_geometry(reference_atom_array, pred_atom_array, match)
    aligned_pred, rmsd = align_atom_arrays(
        mbl_sele=pred_atom_array[match.pred_indices],
        tgt_sele=reference_atom_array[match.reference_indices],
        mbl_full=pred_atom_array,
    )
    return aligned_pred, _as_float(rmsd)


def _rmsd_after_alignment(
    *,
    sample_atom_array: AtomArray,
    aligned_pred_atom_array: AtomArray,
    match: _CaMatch,
) -> float:
    delta = (
        aligned_pred_atom_array.coord[match.pred_indices]
        - sample_atom_array.coord[match.sample_indices]
    )
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _confidence_file_path(pred_sample_path: str | Path) -> Path:
    path = Path(pred_sample_path)
    name = re.sub(r"_model$", "_confidences", path.stem) + ".json"
    return path.parent / name


def _mean_plddt(
    *,
    pred_sample_path: str | Path | None,
    pred_atom_array: AtomArray,
    mask: np.ndarray,
) -> float | None:
    if pred_sample_path is None:
        raise ValueError("pred_sample_path is required to extract AF3 pLDDT")
    return extract_af3_confidence_metrics(
        confidence_file_path=str(_confidence_file_path(pred_sample_path)),
        atom_array=pred_atom_array,
        mask=mask,
        metrics_to_extract="atom_plddts",
        return_mean=True,
    )


def _plddt_or_error(
    *,
    pred_sample_path: str | Path | None,
    pred_atom_array: AtomArray,
    mask: np.ndarray,
) -> tuple[float | None, str]:
    try:
        return _mean_plddt(
            pred_sample_path=pred_sample_path,
            pred_atom_array=pred_atom_array,
            mask=mask,
        ), ""
    except Exception as exc:
        return None, f"pLDDT extraction failed: {exc}"


def _pn_unit_kind(atom_array: AtomArray, pn_unit_iid: str) -> str:
    mask = atom_array.pn_unit_iid == pn_unit_iid
    if not np.any(mask):
        raise ValueError(f"PN unit {pn_unit_iid!r} is missing")
    chain_types = atom_array.chain_type[mask]
    if np.all(chain_types == aw_enums.ChainType.POLYPEPTIDE_L):
        return "protein"
    if np.all(np.isin(chain_types, list(aw_enums.ChainTypeInfo.NON_POLYMERS))):
        heavy_mask = mask & (atom_array.element != "H") & ~np.isnan(atom_array.coord).any(axis=1)
        elements = [normalize_ccd_code(element) for element in atom_array.element[heavy_mask]]
        if len(elements) == 1 and elements[0] in METAL_ELEMENTS:
            return "metal"
        return "small_molecule"
    raise ValueError(f"PN unit {pn_unit_iid!r} has unsupported or mixed chain types")


def _validate_protein_units(atom_array: AtomArray, pn_unit_iids: Sequence[str], role: str) -> None:
    nonprotein = []
    for pn_unit_iid in pn_unit_iids:
        try:
            kind = _pn_unit_kind(atom_array, pn_unit_iid)
        except ValueError as exc:
            raise ValueError(f"Invalid {role} PN unit: {exc}") from exc
        if kind != "protein":
            nonprotein.append(f"{pn_unit_iid}({kind})")
    if nonprotein:
        raise ValueError(f"{role} PN units must be proteins; got {nonprotein}")


def _match_metadata(match: _CaMatch) -> dict[str, float | int]:
    return {
        "matched_atom_count": match.matched_count,
        "reference_atom_count": match.sample_count,
        "prediction_atom_count": match.pred_count,
        "reference_match_coverage": match.sample_coverage,
        "prediction_match_coverage": match.pred_coverage,
    }


def _frame_metadata(match: _CaMatch, frame_ca_rmsd: float) -> dict[str, float | int]:
    return {
        "frame_ca_rmsd": frame_ca_rmsd,
        "frame_matched_ca_count": match.matched_count,
        "frame_reference_ca_count": match.sample_count,
        "frame_designed_ca_count": match.sample_count,
        "frame_prediction_ca_count": match.pred_count,
        "frame_reference_match_coverage": match.sample_coverage,
        "frame_designed_match_coverage": match.sample_coverage,
        "frame_prediction_match_coverage": match.pred_coverage,
    }


def _three_way_frame_metadata(
    match: _ThreeWayCaMatch,
    frame_ca_rmsd: float,
) -> dict[str, float | int]:
    return {
        "frame_ca_rmsd": frame_ca_rmsd,
        "frame_matched_ca_count": match.matched_count,
        "frame_reference_ca_count": match.reference_count,
        "frame_designed_ca_count": match.designed_count,
        "frame_prediction_ca_count": match.pred_count,
        "frame_reference_match_coverage": match.reference_coverage,
        "frame_designed_match_coverage": match.designed_coverage,
        "frame_prediction_match_coverage": match.pred_coverage,
    }


def _reference_template_residue_keys(
    reference_atom_array: AtomArray,
    template_pn_unit_iids: Sequence[str],
) -> list[ResidueKey]:
    reference_by_key, _ = ca_by_pn_unit_res_id(
        reference_atom_array,
        template_pn_unit_iids,
    )
    return list(reference_by_key)


def _reference_pocket_residue_keys(
    *,
    reference_atom_array: AtomArray,
    receptor_pn_unit_iids: Sequence[str],
    ligand_pn_unit_iid: str,
    pocket_distance: float,
    pocket_annotation_method: str,
) -> list[ResidueKey]:
    annotated = annotate_ligand_pocket(
        atom_array=reference_atom_array.copy(),
        pocket_distance=pocket_distance,
        n_min_ligand_atoms=1,
        annotation_name="is_role_ligand_pocket",
        receptor_pn_unit_iids=list(receptor_pn_unit_iids),
        ligand_pn_unit_iids=[ligand_pn_unit_iid],
        pocket_annotation_method=pocket_annotation_method,
    )
    pocket_mask = apply_and_spread_residue_wise(
        annotated,
        annotated.get_annotation("is_role_ligand_pocket"),
        function=np.any,
    )
    pocket_ca_indices = np.where(
        pocket_mask
        & np.isin(annotated.pn_unit_iid, list(receptor_pn_unit_iids))
        & (annotated.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
        & (annotated.atom_name == "CA")
    )[0]
    return list(
        dict.fromkeys(
            (
                str(annotated.pn_unit_iid[index]),
                int(annotated.res_id[index]),
            )
            for index in pocket_ca_indices
        )
    )


def _ligand_alignment_residue_keys(
    *,
    reference_atom_array: AtomArray,
    pn_unit_roles: Mapping[str, Sequence[str]],
    ligand_pn_unit_iid: str,
    reference_is_designed: bool,
) -> tuple[list[ResidueKey], list[str], str, str | None, float | None]:
    template_iids = [str(iid) for iid in pn_unit_roles["template_pn_unit_iids"]]
    if template_iids:
        return (
            _reference_template_residue_keys(reference_atom_array, template_iids),
            template_iids,
            "template_full_ca",
            None,
            None,
        )

    query_iids = _ordered_unique(
        [
            *pn_unit_roles["binder_pn_unit_iids"],
            *pn_unit_roles["context_pn_unit_iids"],
        ]
    )
    receptor_iids = [
        iid
        for iid in query_iids
        if iid != ligand_pn_unit_iid
        and _pn_unit_kind(reference_atom_array, iid) == "protein"
    ]
    if not receptor_iids:
        raise ValueError(f"No selected protein PN units are available around ligand {ligand_pn_unit_iid}")
    pocket_distance = (
        DESIGNED_REFERENCE_POCKET_DISTANCE
        if reference_is_designed
        else NATIVE_REFERENCE_POCKET_DISTANCE
    )
    pocket_annotation_method = "calpha" if reference_is_designed else "all_atom"
    return (
        _reference_pocket_residue_keys(
            reference_atom_array=reference_atom_array,
            receptor_pn_unit_iids=receptor_iids,
            ligand_pn_unit_iid=ligand_pn_unit_iid,
            pocket_distance=pocket_distance,
            pocket_annotation_method=pocket_annotation_method,
        ),
        receptor_iids,
        f"{pocket_annotation_method}_pocket_ca",
        pocket_annotation_method,
        pocket_distance,
    )


def _pocket_aligned_path(
    pred_sample_path: str | Path,
    ligand_pn_unit_iid: str,
) -> Path:
    path = Path(pred_sample_path)
    safe_iid = re.sub(r"[^A-Za-z0-9_.-]+", "_", ligand_pn_unit_iid)
    return path.with_name(f"{path.stem}_{safe_iid}_pocket_aligned.cif")


def _set_status_from_plddt_error(row: dict[str, Any], plddt_error: str) -> None:
    if plddt_error:
        row.update(status="partial", error=plddt_error)
    else:
        row.update(status="ok", error="")


def _compute_intrinsic_row(
    *,
    operation: RoleMetricOperation,
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    pred_sample_path: str | Path | None,
) -> dict[str, Any]:
    try:
        _validate_protein_units(sample_atom_array, operation.measurement_pn_unit_iids, "binder")
        _validate_protein_units(pred_atom_array, operation.measurement_pn_unit_iids, "predicted binder")
        match = _match_ca(
            sample_atom_array=sample_atom_array,
            pred_atom_array=pred_atom_array,
            pn_unit_iids=operation.measurement_pn_unit_iids,
        )
        _, ca_rmsd = _align_on_match(
            sample_atom_array=sample_atom_array,
            pred_atom_array=pred_atom_array,
            match=match,
        )
        ca_mask = np.zeros(len(pred_atom_array), dtype=bool)
        ca_mask[match.pred_indices] = True
        ca_plddt, plddt_error = _plddt_or_error(
            pred_sample_path=pred_sample_path,
            pred_atom_array=pred_atom_array,
            mask=ca_mask,
        )
    except Exception as exc:
        return _error_row(operation, str(exc))

    row = _empty_row(operation)
    row.update(
        target_kind="protein",
        rmsd_mode="self_aligned_ca",
        ca_rmsd=ca_rmsd,
        ca_plddt=ca_plddt,
        **_match_metadata(match),
    )
    _set_status_from_plddt_error(row, plddt_error)
    return row


def _compute_small_molecule_rmsd(
    *,
    sample_ligand: AtomArray,
    aligned_pred_ligand: AtomArray,
    ligand_smiles: str | None,
) -> tuple[float, str, int, dict[str, Any]]:
    if ligand_smiles:
        metrics, error = _mcs_core_rmsd_no_ligand_alignment(
            reference_ligand_atom_array=sample_ligand,
            pred_ligand_atom_array=aligned_pred_ligand,
            pred_ligand_smiles=ligand_smiles,
        )
        if error or metrics is None:
            raise ValueError(error or "MCS ligand RMSD failed")
        metrics = dict(metrics)
        rmsd = float(metrics.pop("ligand_rmsd"))
        mode = str(metrics.pop("ligand_rmsd_mode"))
        matched_count = int(metrics.get("mcs_num_atoms", 0))
        return rmsd, mode, matched_count, {}

    sample_mol = Chem.RemoveHs(_rdkit_mol_from_ligand_atom_array(sample_ligand))
    pred_mol = Chem.RemoveHs(_rdkit_mol_from_ligand_atom_array(aligned_pred_ligand))
    try:
        rmsd = rdMolAlign.CalcRMS(sample_mol, pred_mol)
    except Exception:
        rmsd = rdMolAlign.CalcRMS(pred_mol, sample_mol)
    return float(rmsd), "ccd_exact", int(min(sample_mol.GetNumAtoms(), pred_mol.GetNumAtoms())), {}


def _compute_ligand_placement_row(
    *,
    operation: RoleMetricOperation,
    reference_atom_array: AtomArray,
    designed_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    pn_unit_roles: Mapping[str, Sequence[str]],
    pred_sample_path: str | Path | None,
    target_kind: str,
    ligand_smiles: str | None,
    reference_is_designed: bool,
    save_aligned: bool,
) -> dict[str, Any]:
    target = str(operation.target_pn_unit_iid)
    reference_heavy_mask = (
        (reference_atom_array.pn_unit_iid == target)
        & (reference_atom_array.element != "H")
        & ~np.isnan(reference_atom_array.coord).any(axis=1)
    )
    pred_heavy_mask = (
        (pred_atom_array.pn_unit_iid == target)
        & (pred_atom_array.element != "H")
        & ~np.isnan(pred_atom_array.coord).any(axis=1)
    )
    reference_count = int(reference_heavy_mask.sum())
    pred_count = int(pred_heavy_mask.sum())
    if reference_count == 0 or pred_count == 0:
        return _error_row(
            operation,
            "No resolved heavy atoms for target "
            f"{target}: reference={reference_count}, prediction={pred_count}",
        )

    row = _empty_row(operation)
    frame_metadata: dict[str, Any] = {}
    try:
        (
            residue_keys,
            alignment_iids,
            alignment_mode,
            pocket_annotation_method,
            pocket_distance,
        ) = _ligand_alignment_residue_keys(
            reference_atom_array=reference_atom_array,
            pn_unit_roles=pn_unit_roles,
            ligand_pn_unit_iid=target,
            reference_is_designed=reference_is_designed,
        )
        match = match_reference_designed_predicted_ca(
            reference_atom_array=reference_atom_array,
            designed_atom_array=designed_atom_array,
            pred_atom_array=pred_atom_array,
            residue_keys=residue_keys,
        )
        aligned_pred_atom_array, frame_ca_rmsd = _align_prediction_to_reference(
            reference_atom_array=reference_atom_array,
            pred_atom_array=pred_atom_array,
            match=match,
        )
        frame_metadata = _three_way_frame_metadata(match, frame_ca_rmsd)

        if target_kind == "metal":
            reference_mask, pred_mask, _, error = _matched_metal_atom_masks(
                sample_atom_array=reference_atom_array,
                pred_atom_array=aligned_pred_atom_array,
                metal_pn_unit_iids=[target],
            )
            if error or reference_mask is None or pred_mask is None:
                raise ValueError(error or f"Failed to match metal target {target}")
            delta = (
                aligned_pred_atom_array.coord[pred_mask]
                - reference_atom_array.coord[reference_mask]
            )
            ligand_rmsd = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
            rmsd_mode = "metal"
            matched_count = int(reference_mask.sum())
            metric_metadata: dict[str, Any] = {}
            confidence_mask = pred_mask
        else:
            reference_ligand = reference_atom_array[reference_heavy_mask]
            pred_ligand = aligned_pred_atom_array[pred_heavy_mask]
            ligand_rmsd, rmsd_mode, matched_count, metric_metadata = (
                _compute_small_molecule_rmsd(
                    sample_ligand=reference_ligand,
                    aligned_pred_ligand=pred_ligand,
                    ligand_smiles=ligand_smiles,
                )
            )
            confidence_mask = pred_heavy_mask

        aligned_path = None
        if save_aligned and pred_sample_path is not None:
            aligned_path = _pocket_aligned_path(pred_sample_path, target)
            save_cif_file(aligned_pred_atom_array, aligned_path)
        ligand_plddt, plddt_error = _plddt_or_error(
            pred_sample_path=pred_sample_path,
            pred_atom_array=pred_atom_array,
            mask=confidence_mask,
        )
    except Exception as exc:
        return _error_row(operation, str(exc)) | dict(frame_metadata)

    row.update(
        target_kind=target_kind,
        target_res_name=";".join(
            _ordered_unique(
                str(res_name)
                for res_name in reference_atom_array.res_name[reference_heavy_mask]
            )
        ),
        alignment_pn_unit_iids=";".join(alignment_iids),
        alignment_mode=alignment_mode,
        pocket_annotation_method=pocket_annotation_method,
        pocket_distance=pocket_distance,
        aligned_path=str(aligned_path) if aligned_path is not None else None,
        rmsd_mode=rmsd_mode,
        ligand_rmsd=ligand_rmsd,
        ligand_plddt=ligand_plddt,
        matched_atom_count=matched_count,
        reference_atom_count=reference_count,
        prediction_atom_count=pred_count,
        reference_match_coverage=float(matched_count / reference_count),
        prediction_match_coverage=float(matched_count / pred_count),
        **metric_metadata,
        **frame_metadata,
    )
    _set_status_from_plddt_error(row, plddt_error)
    return row


def compute_role_aware_metrics_atomarray(
    *,
    pred_atom_array: AtomArray,
    sample_atom_array: AtomArray,
    pn_unit_roles: Mapping[str, Any],
    reference_atom_array: AtomArray | None = None,
    reference_is_designed: bool = False,
    pred_sample_path: str | Path | None = None,
    ligand_smiles_by_iid: Mapping[str, str | None] | None = None,
    save_aligned: bool = True,
) -> list[dict[str, Any]]:
    """Compute role-derived protein and per-ligand placement metrics.

    Protein metrics use designed/predicted exact-residue CA correspondence.
    Every non-protein target gets an independent reference-defined alignment:
    full template-chain CA when a template role exists, otherwise a 10 A native
    all-atom pocket or 12 A designed-reference CA pocket.
    """
    roles = normalize_pn_unit_roles(pn_unit_roles)
    operations = build_role_metric_plan(roles)
    if reference_atom_array is None:
        reference_atom_array = sample_atom_array
    rows: list[dict[str, Any]] = []

    intrinsic_operations = [op for op in operations if op.metric_family == "intrinsic_sc"]
    for operation in intrinsic_operations:
        rows.append(
            _compute_intrinsic_row(
                operation=operation,
                sample_atom_array=sample_atom_array,
                pred_atom_array=pred_atom_array,
                pred_sample_path=pred_sample_path,
            )
        )

    complex_operations = [op for op in operations if op.metric_family == "binder_complex_sc"]
    if complex_operations:
        union_operation = next(op for op in complex_operations if op.operation_scope == "union")
        try:
            _validate_protein_units(sample_atom_array, union_operation.alignment_pn_unit_iids, "binder")
            _validate_protein_units(pred_atom_array, union_operation.alignment_pn_unit_iids, "predicted binder")
            complex_match = _match_ca(
                sample_atom_array=sample_atom_array,
                pred_atom_array=pred_atom_array,
                pn_unit_iids=union_operation.alignment_pn_unit_iids,
            )
            complex_aligned, complex_rmsd = _align_on_match(
                sample_atom_array=sample_atom_array,
                pred_atom_array=pred_atom_array,
                match=complex_match,
            )
            complex_ca_mask = np.zeros(len(pred_atom_array), dtype=bool)
            complex_ca_mask[complex_match.pred_indices] = True
            complex_plddt, complex_plddt_error = _plddt_or_error(
                pred_sample_path=pred_sample_path,
                pred_atom_array=pred_atom_array,
                mask=complex_ca_mask,
            )
        except Exception as exc:
            rows.extend(_error_row(operation, str(exc)) for operation in complex_operations)
        else:
            union_row = _empty_row(union_operation)
            union_row.update(
                target_kind="protein_union",
                rmsd_mode="self_aligned_ca_union",
                ca_rmsd=complex_rmsd,
                ca_plddt=complex_plddt,
                **_match_metadata(complex_match),
            )
            _set_status_from_plddt_error(union_row, complex_plddt_error)
            rows.append(union_row)
            for operation in complex_operations:
                if operation.operation_scope != "member_residual":
                    continue
                try:
                    member_match = _match_ca(
                        sample_atom_array=sample_atom_array,
                        pred_atom_array=pred_atom_array,
                        pn_unit_iids=operation.measurement_pn_unit_iids,
                    )
                    member_rmsd = _rmsd_after_alignment(
                        sample_atom_array=sample_atom_array,
                        aligned_pred_atom_array=complex_aligned,
                        match=member_match,
                    )
                    member_mask = np.zeros(len(pred_atom_array), dtype=bool)
                    member_mask[member_match.pred_indices] = True
                    member_plddt, member_plddt_error = _plddt_or_error(
                        pred_sample_path=pred_sample_path,
                        pred_atom_array=pred_atom_array,
                        mask=member_mask,
                    )
                except Exception as exc:
                    rows.append(_error_row(operation, str(exc)))
                    continue
                member_row = _empty_row(operation)
                member_row.update(
                    target_kind="protein",
                    rmsd_mode="binder_union_aligned_ca_residual",
                    ca_rmsd=member_rmsd,
                    ca_plddt=member_plddt,
                    **_match_metadata(member_match),
                )
                _set_status_from_plddt_error(member_row, member_plddt_error)
                rows.append(member_row)

    frame_operations = [op for op in operations if op.metric_family == "frame_fit"]
    placement_operations = [op for op in operations if op.metric_family == "frame_placement"]
    frame_union_operation = next(op for op in frame_operations if op.operation_scope == "union")
    frame_error = ""
    frame_info: dict[str, Any] = {}
    frame_aligned: AtomArray | None = None
    try:
        _validate_protein_units(sample_atom_array, frame_union_operation.alignment_pn_unit_iids, "frame")
        _validate_protein_units(pred_atom_array, frame_union_operation.alignment_pn_unit_iids, "predicted frame")
        frame_match = _match_ca(
            sample_atom_array=sample_atom_array,
            pred_atom_array=pred_atom_array,
            pn_unit_iids=frame_union_operation.alignment_pn_unit_iids,
        )
        _validate_frame_geometry(sample_atom_array, pred_atom_array, frame_match)
        frame_aligned, frame_rmsd = _align_on_match(
            sample_atom_array=sample_atom_array,
            pred_atom_array=pred_atom_array,
            match=frame_match,
        )
        frame_info = _frame_metadata(frame_match, frame_rmsd)
        frame_mask = np.zeros(len(pred_atom_array), dtype=bool)
        frame_mask[frame_match.pred_indices] = True
        frame_plddt, frame_plddt_error = _plddt_or_error(
            pred_sample_path=pred_sample_path,
            pred_atom_array=pred_atom_array,
            mask=frame_mask,
        )
    except Exception as exc:
        frame_error = str(exc)
        rows.extend(_error_row(operation, frame_error) for operation in frame_operations)
    else:
        frame_union_row = _empty_row(frame_union_operation)
        frame_union_row.update(
            target_kind="protein_union",
            rmsd_mode="frame_union_aligned_ca",
            ca_rmsd=frame_rmsd,
            ca_plddt=frame_plddt,
            **_match_metadata(frame_match),
            **frame_info,
        )
        _set_status_from_plddt_error(frame_union_row, frame_plddt_error)
        rows.append(frame_union_row)

        for operation in frame_operations:
            if operation.operation_scope != "member_residual":
                continue
            try:
                member_match = _match_ca(
                    sample_atom_array=sample_atom_array,
                    pred_atom_array=pred_atom_array,
                    pn_unit_iids=operation.measurement_pn_unit_iids,
                )
                member_rmsd = _rmsd_after_alignment(
                    sample_atom_array=sample_atom_array,
                    aligned_pred_atom_array=frame_aligned,
                    match=member_match,
                )
                member_mask = np.zeros(len(pred_atom_array), dtype=bool)
                member_mask[member_match.pred_indices] = True
                member_plddt, member_plddt_error = _plddt_or_error(
                    pred_sample_path=pred_sample_path,
                    pred_atom_array=pred_atom_array,
                    mask=member_mask,
                )
            except Exception as exc:
                rows.append(_error_row(operation, str(exc)) | frame_info)
                continue
            member_row = _empty_row(operation)
            member_row.update(
                target_kind="protein",
                rmsd_mode="frame_union_aligned_ca_residual",
                ca_rmsd=member_rmsd,
                ca_plddt=member_plddt,
                **_match_metadata(member_match),
                **frame_info,
            )
            _set_status_from_plddt_error(member_row, member_plddt_error)
            rows.append(member_row)

    for operation in placement_operations:
        if operation.operation_scope == "no_targets":
            row = _empty_row(operation)
            row.update(status="not_applicable", error="no_non_frame_targets", **frame_info)
            rows.append(row)
            continue
        target = str(operation.target_pn_unit_iid)
        try:
            reference_kind = _pn_unit_kind(reference_atom_array, target)
            sample_kind = _pn_unit_kind(sample_atom_array, target)
            pred_kind = _pn_unit_kind(pred_atom_array, target)
            if reference_kind != sample_kind or sample_kind != pred_kind:
                raise ValueError(
                    f"Target {target} kind differs across reference ({reference_kind}), "
                    f"designed ({sample_kind}), and prediction ({pred_kind})"
                )
        except Exception as exc:
            rows.append(_error_row(operation, str(exc)) | frame_info)
            continue

        if sample_kind == "protein":
            if frame_aligned is None:
                rows.append(
                    _error_row(operation, f"Frame alignment failed: {frame_error}")
                )
                continue
            try:
                target_match = _match_ca(
                    sample_atom_array=sample_atom_array,
                    pred_atom_array=pred_atom_array,
                    pn_unit_iids=[target],
                )
                target_rmsd = _rmsd_after_alignment(
                    sample_atom_array=sample_atom_array,
                    aligned_pred_atom_array=frame_aligned,
                    match=target_match,
                )
                target_mask = np.zeros(len(pred_atom_array), dtype=bool)
                target_mask[target_match.pred_indices] = True
                target_plddt, target_plddt_error = _plddt_or_error(
                    pred_sample_path=pred_sample_path,
                    pred_atom_array=pred_atom_array,
                    mask=target_mask,
                )
            except Exception as exc:
                rows.append(_error_row(operation, str(exc)) | frame_info)
                continue
            row = _empty_row(operation)
            row.update(
                target_kind="protein",
                rmsd_mode="frame_aligned_ca_position",
                ca_rmsd=target_rmsd,
                ca_plddt=target_plddt,
                **_match_metadata(target_match),
                **frame_info,
            )
            _set_status_from_plddt_error(row, target_plddt_error)
            rows.append(row)
            continue

        rows.append(
            _compute_ligand_placement_row(
                operation=operation,
                reference_atom_array=reference_atom_array,
                designed_atom_array=sample_atom_array,
                pred_atom_array=pred_atom_array,
                pn_unit_roles=roles,
                pred_sample_path=pred_sample_path,
                target_kind=sample_kind,
                ligand_smiles=(ligand_smiles_by_iid or {}).get(target),
                reference_is_designed=reference_is_designed,
                save_aligned=save_aligned,
            )
        )
    return rows
