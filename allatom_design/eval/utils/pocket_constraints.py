from __future__ import annotations

import numpy as np
import atomworks.enums as aw_enums
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from biotite.structure import AtomArray, get_residue_starts

from allatom_design.data.transform.custom_transforms import (
    annotate_ligand_pockets,
    annotate_ligand_pockets_calpha,
    annotate_ligand_pockets_pseudocb,
)

POCKET_ANNOTATION_METHODS = {"all_atom", "calpha", "pseudocb"}
POCKET_ANNOTATION_METHOD_ALIASES = {
    "all_atom": "all_atom",
    "allatom": "all_atom",
    "atom": "all_atom",
    "calpha": "calpha",
    "c_alpha": "calpha",
    "ca": "calpha",
    "pseudocb": "pseudocb",
    "pseudo_cb": "pseudocb",
    "pseudo_cbeta": "pseudocb",
}


def resolve_pocket_annotation_method(
    pocket_annotation_method: str | None = None,
    use_calpha_for_pocket_annotation: bool = False,
) -> str:
    if pocket_annotation_method is None:
        return "calpha" if use_calpha_for_pocket_annotation else "all_atom"

    method = str(pocket_annotation_method).replace("-", "_").lower()
    resolved_method = POCKET_ANNOTATION_METHOD_ALIASES.get(method)
    if resolved_method is None:
        valid = ", ".join(sorted(POCKET_ANNOTATION_METHODS))
        raise ValueError(
            f"pocket_annotation_method must be one of {valid}; got {pocket_annotation_method!r}"
        )
    return resolved_method


def annotate_ligand_pocket(
    *,
    atom_array: AtomArray,
    pocket_distance: float = 5.0,
    n_min_ligand_atoms: int = 5,
    annotation_name: str = "is_ligand_pocket",
    receptor_pn_unit_iids: list[str] | None = None,
    ligand_pn_unit_iids: list[str] | None = None,
    pocket_annotation_method: str | None = None,
    use_calpha_for_pocket_annotation: bool = False,
) -> AtomArray:
    """Annotate ligand pockets using the canonical eval method names."""
    resolved_method = resolve_pocket_annotation_method(
        pocket_annotation_method=pocket_annotation_method,
        use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
    )
    common_kwargs = {
        "atom_array": atom_array,
        "pocket_distance": pocket_distance,
        "n_min_ligand_atoms": n_min_ligand_atoms,
        "receptor_pn_unit_iids": receptor_pn_unit_iids,
        "ligand_pn_unit_iids": ligand_pn_unit_iids,
        "annotation_name": annotation_name,
    }
    if resolved_method == "calpha":
        return annotate_ligand_pockets_calpha(**common_kwargs)
    if resolved_method == "pseudocb":
        return annotate_ligand_pockets_pseudocb(**common_kwargs)
    return annotate_ligand_pockets(**common_kwargs)


def _indices_to_pos_string(chain_ids: np.ndarray, res_ids: np.ndarray) -> str:
    chain_to_res = {}
    for chain_id, res_id in zip(chain_ids, res_ids):
        chain_to_res.setdefault(chain_id, []).append(res_id)

    pos_parts = []
    for chain_id in sorted(chain_to_res.keys()):
        res_list = sorted(set(chain_to_res[chain_id]))
        if not res_list:
            continue

        ranges = []
        start = end = res_list[0]
        for res_id in res_list[1:]:
            if res_id == end + 1:
                end = res_id
            else:
                ranges.append((start, end))
                start = end = res_id
        ranges.append((start, end))

        for start, end in ranges:
            pos_parts.append(
                f"{chain_id}{start}" if start == end else f"{chain_id}{start}-{end}"
            )

    return ",".join(pos_parts)


def create_pos_constraint_dict_from_pocket(
    pdb_key: str,
    atom_array: AtomArray,
    pocket_distance: float = 5.0,
    constraint_type: str = "pocket",
    receptor_pn_unit_iids: list[str] | None = None,
    ligand_pn_unit_iids: list[str] | None = None,
    pocket_annotation_method: str | None = None,
    use_calpha_for_pocket_annotation: bool = False,
    sample_path: str | None = None,
    return_ligand_mpnn_format: bool = False,
) -> tuple[dict, dict]:
    """Create runtime positional-constraint rows from ligand-pocket annotation."""
    resolved_method = resolve_pocket_annotation_method(
        pocket_annotation_method=pocket_annotation_method,
        use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
    )
    annotated_atom_array = annotate_ligand_pocket(
        atom_array=atom_array,
        pocket_distance=pocket_distance,
        n_min_ligand_atoms=1,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
        annotation_name="is_ligand_pocket",
        pocket_annotation_method=resolved_method,
    )

    residue_wise_pocket_mask = apply_and_spread_residue_wise(
        annotated_atom_array,
        annotated_atom_array.get_annotation("is_ligand_pocket"),
        function=np.any,
    )
    protein_mask = annotated_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
    if receptor_pn_unit_iids:
        if "pn_unit_iid" not in annotated_atom_array.get_annotation_categories():
            raise ValueError("pn_unit_iid annotation is required for receptor-scoped constraints")
        receptor_iid_set = {str(pn_unit_iid) for pn_unit_iid in receptor_pn_unit_iids}
        receptor_mask = np.isin(
            annotated_atom_array.get_annotation("pn_unit_iid").astype(str),
            list(receptor_iid_set),
        )
        protein_mask = protein_mask & receptor_mask

    if constraint_type == "pocket":
        constrained_mask = protein_mask & residue_wise_pocket_mask
    elif constraint_type == "scaffold":
        constrained_mask = protein_mask & ~residue_wise_pocket_mask
    else:
        raise ValueError(f"Invalid constraint type: {constraint_type}")

    constrained_atom_array = annotated_atom_array[constrained_mask]
    if len(constrained_atom_array) == 0:
        return {
            "pdb_key": pdb_key,
            "fixed_pos_seq": "",
            "fixed_pos_scn": np.nan,
            "pocket_distance": pocket_distance,
            "constraint_type": constraint_type,
            "num_constrained_residues": 0,
        }, {}

    res_starts = get_residue_starts(constrained_atom_array)
    chain_ids = constrained_atom_array.chain_id[res_starts]
    res_ids = constrained_atom_array.res_id[res_starts]

    result = {
        "pdb_key": pdb_key,
        "fixed_pos_seq": _indices_to_pos_string(chain_ids, res_ids),
        "fixed_pos_scn": np.nan,
        "pocket_distance": pocket_distance,
        "constraint_type": constraint_type,
        "num_constrained_residues": len(res_starts),
    }

    results_for_ligand_mpnn = {}
    if return_ligand_mpnn_format:
        results_for_ligand_mpnn["pdb_path"] = sample_path if sample_path else ""
        fixed_residues_list = [f"{cid}{rid}" for cid, rid in zip(chain_ids, res_ids)]
        results_for_ligand_mpnn["fixed_residues"] = " ".join(fixed_residues_list)
        protein_chain_ids = list(
            {pn_unit_iid.split("_")[0] for pn_unit_iid in receptor_pn_unit_iids or []}
        )
        ligand_chain_ids = list(
            {pn_unit_iid.split("_")[0] for pn_unit_iid in ligand_pn_unit_iids or []}
        )
        results_for_ligand_mpnn["chains"] = ",".join(protein_chain_ids + ligand_chain_ids)

    return result, results_for_ligand_mpnn


__all__ = [
    "POCKET_ANNOTATION_METHODS",
    "POCKET_ANNOTATION_METHOD_ALIASES",
    "annotate_ligand_pocket",
    "create_pos_constraint_dict_from_pocket",
    "resolve_pocket_annotation_method",
]
