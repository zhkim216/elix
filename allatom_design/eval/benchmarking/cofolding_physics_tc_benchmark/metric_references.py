from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from rdkit import Chem

import atomworks.enums as aw_enums
from atomworks.enums import ChainType
from atomworks.io.tools.rdkit import atom_array_from_rdkit
from atomworks.io.utils.sequence import get_1_from_3_letter_code

from allatom_design.data.const import METAL_ELEMENTS
from allatom_design.eval.metrics.docking import compute_docking_metrics_atomarray
from allatom_design.eval.metrics.self_consistency import (
    compute_self_consistency_metrics_atomarray,
)
from allatom_design.eval.chemical_components import normalize_ccd_code
from allatom_design.utils.sample_io_utils import load_example_with_parse


AA_ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


@dataclass(frozen=True)
class MetricTarget:
    pred_ligand_id: str
    reference_ligand_id: str
    ligand_smiles: str | None
    ligand_ccd_codes: tuple[str, ...]
    reason: str


def load_atom_array(path: Path, *, af3_prediction: bool = False):
    configs = [
        {
            "add_missing_atoms": True,
            "remove_waters": not af3_prediction,
            "remove_ccds": [],
            "fix_ligands_at_symmetry_centers": not af3_prediction,
            "fix_arginines": not af3_prediction,
            "convert_mse_to_met": True,
            "hydrogen_policy": "remove",
            "extra_fields": None if af3_prediction else "all",
        },
        {
            "add_missing_atoms": False,
            "remove_waters": not af3_prediction,
            "remove_ccds": [],
            "fix_ligands_at_symmetry_centers": False,
            "fix_arginines": False,
            "convert_mse_to_met": True,
            "hydrogen_policy": "remove",
            "extra_fields": None,
        },
    ]
    last_error = None
    for config in configs:
        try:
            return load_example_with_parse(str(path), OmegaConf.create(config))["atom_array"]
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not parse {path}: {last_error}")


def ligand_atom_array_from_sdf(path: Path, *, ligand_id: str, res_name: str = "REF"):
    supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    mol = supplier[0] if len(supplier) else None
    if mol is None:
        raise ValueError(f"Could not read ligand SDF without sanitization: {path}")
    atom_array = atom_array_from_rdkit(mol)
    atom_array.chain_id[:] = ligand_id
    atom_array.res_id[:] = 1
    atom_array.ins_code[:] = ""
    atom_array.res_name[:] = res_name
    atom_array.hetero[:] = True
    if "pn_unit_iid" not in atom_array.get_annotation_categories():
        atom_array.set_annotation("pn_unit_iid", np.array([f"{ligand_id}_1"] * len(atom_array)))
    else:
        atom_array.pn_unit_iid[:] = f"{ligand_id}_1"
    if "chain_type" not in atom_array.get_annotation_categories():
        atom_array.set_annotation(
            "chain_type",
            np.full(len(atom_array), int(aw_enums.ChainType.NON_POLYMER)),
        )
    else:
        atom_array.chain_type[:] = int(aw_enums.ChainType.NON_POLYMER)
    return atom_array


def reference_atom_array_from_metadata(metadata: dict[str, Any]):
    source_structure = Path(metadata["source_structure"])
    reference_atom_array = load_atom_array(source_structure, af3_prediction=False)
    ligand_sources = [Path(path) for path in metadata.get("ligand_sources", []) if str(path)]
    if not ligand_sources:
        return reference_atom_array

    ligands = metadata.get("ligands", [])
    for idx, ligand_source in enumerate(ligand_sources):
        ligand = ligands[min(idx, len(ligands) - 1)] if ligands else {"id": "L"}
        ligand_id = str(ligand.get("id") or "L")
        ligand_atoms = ligand_atom_array_from_sdf(
            ligand_source,
            ligand_id=ligand_id,
            res_name=f"REF{idx + 1}",
        )
        reference_atom_array = reference_atom_array + ligand_atoms
    return reference_atom_array


def pn_unit_iids_for_chain(atom_array, chain_id: str) -> list[str]:
    chain_id = str(chain_id)
    if hasattr(atom_array, "pn_unit_iid"):
        unique_iids = [str(iid) for iid in np.unique(np.asarray(atom_array.pn_unit_iid).astype(str))]
        exact = [iid for iid in unique_iids if iid == chain_id]
        if exact:
            return exact
        matching = [iid for iid in unique_iids if iid.startswith(f"{chain_id}_")]
        if matching:
            return matching

    iids: list[str] = []
    for idx in range(len(atom_array)):
        atom_chain_id = str(atom_array.chain_id[idx])
        pn_unit_iid = str(atom_array.pn_unit_iid[idx]) if hasattr(atom_array, "pn_unit_iid") else ""
        if atom_chain_id == chain_id and pn_unit_iid and pn_unit_iid not in iids:
            iids.append(pn_unit_iid)
    return iids or [f"{chain_id}_1"]


def protein_pn_unit_iids_from_metadata(metadata: dict[str, Any], atom_array) -> list[str]:
    protein_iids: list[str] = []
    for protein in metadata.get("protein_chains", []):
        chain_id = str(protein.get("id", ""))
        if not chain_id:
            continue
        protein_iids.extend(pn_unit_iids_for_chain(atom_array, chain_id))
    return list(dict.fromkeys(protein_iids))


def global_aligned_index_pairs(query_sequence: str, template_sequence: str) -> list[tuple[int, int]]:
    match_score = 2
    mismatch_score = 0
    gap_score = -2
    n_query = len(query_sequence)
    n_template = len(template_sequence)
    scores = np.zeros((n_query + 1, n_template + 1), dtype=np.int32)
    traceback = np.zeros((n_query + 1, n_template + 1), dtype=np.int8)
    for query_idx in range(1, n_query + 1):
        scores[query_idx, 0] = query_idx * gap_score
        traceback[query_idx, 0] = 1
    for template_idx in range(1, n_template + 1):
        scores[0, template_idx] = template_idx * gap_score
        traceback[0, template_idx] = 2
    for query_idx in range(1, n_query + 1):
        for template_idx in range(1, n_template + 1):
            diagonal = scores[query_idx - 1, template_idx - 1] + (
                match_score
                if query_sequence[query_idx - 1] == template_sequence[template_idx - 1]
                else mismatch_score
            )
            up = scores[query_idx - 1, template_idx] + gap_score
            left = scores[query_idx, template_idx - 1] + gap_score
            best = max(diagonal, up, left)
            scores[query_idx, template_idx] = best
            traceback[query_idx, template_idx] = 0 if best == diagonal else (1 if best == up else 2)
    pairs: list[tuple[int, int]] = []
    query_idx = n_query
    template_idx = n_template
    while query_idx > 0 or template_idx > 0:
        step = traceback[query_idx, template_idx]
        if query_idx > 0 and template_idx > 0 and step == 0:
            pairs.append((query_idx - 1, template_idx - 1))
            query_idx -= 1
            template_idx -= 1
        elif query_idx > 0 and (template_idx == 0 or step == 1):
            query_idx -= 1
        else:
            template_idx -= 1
    pairs.reverse()
    return pairs


def chain_ca_residue_keys_and_sequence(atom_array, chain_id: str) -> tuple[list[tuple[str, str, str]], str]:
    seen = set()
    residue_keys: list[tuple[str, str, str]] = []
    letters: list[str] = []
    for idx in range(len(atom_array)):
        atom_chain_id = str(atom_array.chain_id[idx])
        pn_unit_iid = str(atom_array.pn_unit_iid[idx]) if hasattr(atom_array, "pn_unit_iid") else ""
        if atom_chain_id != chain_id and not pn_unit_iid.startswith(f"{chain_id}_"):
            continue
        if str(atom_array.atom_name[idx]) != "CA":
            continue
        residue_key = (
            atom_chain_id,
            str(atom_array.res_id[idx]),
            str(atom_array.ins_code[idx]) if hasattr(atom_array, "ins_code") else "",
        )
        if residue_key in seen:
            continue
        seen.add(residue_key)
        residue_keys.append(residue_key)
        letters.append(
            get_1_from_3_letter_code(
                str(atom_array.res_name[idx]),
                chain_type=ChainType.POLYPEPTIDE_L,
                use_closest_canonical=True,
            )
        )
    return residue_keys, "".join(letters)


def sample_atom_array_with_metadata_sequence(atom_array, metadata: dict[str, Any]):
    out = atom_array.copy()
    for protein in metadata.get("protein_chains", []):
        chain_id = str(protein.get("id", ""))
        query_sequence = str(protein.get("sequence", ""))
        if not chain_id or not query_sequence:
            continue
        residue_keys, template_sequence = chain_ca_residue_keys_and_sequence(out, chain_id)
        if not residue_keys or not template_sequence:
            continue
        for query_idx, template_idx in global_aligned_index_pairs(query_sequence, template_sequence):
            new_res_name = AA_ONE_TO_THREE.get(query_sequence[query_idx])
            if new_res_name is None:
                continue
            atom_chain_id, res_id, ins_code = residue_keys[template_idx]
            mask = (
                (np.asarray(out.chain_id).astype(str) == atom_chain_id)
                & (np.asarray(out.res_id).astype(str) == res_id)
                & ~np.asarray(out.hetero).astype(bool)
            )
            if hasattr(out, "ins_code"):
                mask &= np.asarray(out.ins_code).astype(str) == ins_code
            out.res_name[mask] = new_res_name

    for mutation in metadata.get("mutations", []):
        match = re.fullmatch(r"([^:]+):([A-Z])(-?\d+)([A-Z])", str(mutation))
        if match is None:
            continue
        chain_id, _, res_id, new_residue = match.groups()
        new_res_name = AA_ONE_TO_THREE.get(new_residue)
        if new_res_name is None:
            continue
        mask = (
            (np.asarray(out.chain_id).astype(str) == chain_id)
            & (np.asarray(out.res_id).astype(int) == int(res_id))
            & ~np.asarray(out.hetero).astype(bool)
        )
        out.res_name[mask] = new_res_name
    return out


def sample_atom_array_with_metric_ligand_iid(
    atom_array,
    *,
    reference_ligand_iids: list[str],
    pred_ligand_iids: list[str],
    pred_chain_id: str,
):
    if len(reference_ligand_iids) != 1 or len(pred_ligand_iids) != 1:
        return atom_array
    reference_iid = reference_ligand_iids[0]
    pred_iid = pred_ligand_iids[0]
    reference_mask = np.asarray(atom_array.pn_unit_iid).astype(str) == reference_iid
    collision_mask = (np.asarray(atom_array.pn_unit_iid).astype(str) == pred_iid) & ~reference_mask
    out = atom_array[~collision_mask].copy()
    target_mask = np.asarray(out.pn_unit_iid).astype(str) == reference_iid
    out.chain_id[target_mask] = pred_chain_id
    out.pn_unit_iid[target_mask] = pred_iid
    return out


def ligand_by_id(metadata: dict[str, Any], ligand_id: str) -> dict[str, Any] | None:
    for ligand in metadata.get("ligands", []):
        if str(ligand.get("id", "")) == ligand_id:
            return ligand
    return None


def is_metal_ligand(ligand: dict[str, Any]) -> bool:
    codes = [normalize_ccd_code(str(code)) for code in ligand.get("ccd_codes", [])]
    return bool(codes) and all(code in METAL_ELEMENTS for code in codes)


def metric_targets_for_metadata(metadata: dict[str, Any]) -> list[MetricTarget]:
    pdb_id = str(metadata.get("pdb_id", "")).upper()
    challenge_group = str(metadata.get("challenge_group", ""))
    base_system_id = str(metadata.get("base_system_id", ""))
    ligands = metadata.get("ligands", [])

    explicit_rules: list[tuple[str, str, str]] = []
    if pdb_id == "1B38":
        explicit_rules.append(("B", "C", "cdk2_atp_source_chain"))
    elif challenge_group == "atp_charge_modification" or "atp_modification" in base_system_id:
        explicit_rules.append(("B", "C", "atp_modification_source_atp_chain"))
    elif pdb_id == "2VWH" and challenge_group == "ligand_methylation":
        explicit_rules.append(("D", "C", "gdh_glucose_source_chain"))
    elif challenge_group == "casf2016_af3_screen":
        ligand_id = str(ligands[0].get("id", "L")) if ligands else "L"
        explicit_rules.append((ligand_id, ligand_id, "casf_ligand_sdf"))

    targets: list[MetricTarget] = []
    for pred_ligand_id, reference_ligand_id, reason in explicit_rules:
        ligand = ligand_by_id(metadata, pred_ligand_id)
        if ligand is None:
            continue
        targets.append(
            MetricTarget(
                pred_ligand_id=pred_ligand_id,
                reference_ligand_id=reference_ligand_id,
                ligand_smiles=str(ligand.get("smiles") or "") or None,
                ligand_ccd_codes=tuple(str(code) for code in ligand.get("ccd_codes", [])),
                reason=reason,
            )
        )
    if targets:
        return targets

    for ligand in ligands:
        if is_metal_ligand(ligand):
            continue
        ligand_id = str(ligand.get("id", ""))
        if not ligand_id:
            continue
        targets.append(
            MetricTarget(
                pred_ligand_id=ligand_id,
                reference_ligand_id=ligand_id,
                ligand_smiles=str(ligand.get("smiles") or "") or None,
                ligand_ccd_codes=tuple(str(code) for code in ligand.get("ccd_codes", [])),
                reason="default_non_metal_ligand",
            )
        )
    return targets


def metric_rows_for_prediction(
    *,
    pred_cif: Path,
    metadata: dict[str, Any],
    pocket_distance_for_docking_metrics: float,
) -> list[dict[str, Any]]:
    pred_atom_array = load_atom_array(pred_cif, af3_prediction=True)
    reference_atom_array = reference_atom_array_from_metadata(metadata)
    sample_base_atom_array = sample_atom_array_with_metadata_sequence(reference_atom_array, metadata)
    receptor_iids = protein_pn_unit_iids_from_metadata(metadata, sample_base_atom_array)

    rows: list[dict[str, Any]] = []
    for target in metric_targets_for_metadata(metadata):
        row: dict[str, Any] = {
            "pred_ligand_id": target.pred_ligand_id,
            "reference_ligand_id": target.reference_ligand_id,
            "metric_target_reason": target.reason,
            "pred_cif": str(pred_cif),
        }
        try:
            pred_iids = pn_unit_iids_for_chain(pred_atom_array, target.pred_ligand_id)
            reference_iids = pn_unit_iids_for_chain(reference_atom_array, target.reference_ligand_id)
            sample_atom_array = sample_atom_array_with_metric_ligand_iid(
                sample_base_atom_array,
                reference_ligand_iids=reference_iids,
                pred_ligand_iids=pred_iids,
                pred_chain_id=target.pred_ligand_id,
            )
            metrics = compute_docking_metrics_atomarray(
                pred_atom_array=pred_atom_array,
                sample_atom_array=sample_atom_array,
                reference_atom_array=reference_atom_array,
                pred_sample_path=str(pred_cif),
                pocket_distance_for_docking_metrics=pocket_distance_for_docking_metrics,
                receptor_pn_unit_iids=receptor_iids,
                ligand_pn_unit_iids=pred_iids,
                ligand_ccd_codes=list(target.ligand_ccd_codes) or None,
                ligand_smiles=[target.ligand_smiles] if target.ligand_smiles else None,
                reference_ligand_pn_unit_iids=reference_iids,
                save_aligned=False,
                ref_sample_is_designed=False,
            )
        except Exception as exc:
            metrics = {
                "error": f"{type(exc).__name__}: {exc}",
                "ligand_rmsd": None,
                "binding_site_rmsd": None,
                "ligand_plddt": None,
                "binding_site_plddt": None,
                "iptm": None,
                "interface_min_pae": None,
            }
        row.update(metrics)
        row["metric_status"] = "ok" if not row.get("error") else "error"
        rows.append(row)
    return rows


def self_consistency_row_for_prediction(
    *,
    pred_cif: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pred_cif": str(pred_cif),
    }
    try:
        pred_atom_array = load_atom_array(pred_cif, af3_prediction=True)
        reference_atom_array = reference_atom_array_from_metadata(metadata)
        sample_atom_array = sample_atom_array_with_metadata_sequence(reference_atom_array, metadata)
        metrics = compute_self_consistency_metrics_atomarray(
            pred_atom_array=pred_atom_array,
            sample_atom_array=sample_atom_array,
            pred_sample_path=pred_cif,
            save_aligned=False,
        )
    except Exception as exc:
        metrics = {
            "error": f"{type(exc).__name__}: {exc}",
            "sc_ca_rmsd": None,
            "avg_ca_plddt": None,
        }
    row.update(metrics)
    row["metric_status"] = "ok" if not row.get("error") else "error"
    return row
