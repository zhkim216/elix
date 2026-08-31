import re
from pathlib import Path

import numpy as np
import torch
from biotite.structure import AtomArray

import atomworks.enums as aw_enums
from atomworks.ml.utils.geometry import align_atom_arrays

from allatom_design.utils.sample_io_utils import save_cif_file

from allatom_design.eval.metrics.af3_confidence import extract_af3_confidence_metrics
from allatom_design.eval.metrics.protein_correspondence import match_resolved_ca


def compute_self_consistency_metrics_atomarray(*, pred_atom_array: AtomArray,
                                                sample_atom_array: AtomArray,
                                                pred_sample_path: str = None,
                                                save_aligned: bool = True,
                                                ) -> dict[str, float]:
    """
    Compute self-consistency metrics between a designed structure and its predicted structure, using atom array.

    Uses atomworks align_atom_arrays to handle structures with different atom sets
    (e.g., sample with backbone only vs pred with full sidechain atoms).
    """
    metrics = {}

    sample_protein_iids = [
        str(pn_unit_iid)
        for pn_unit_iid in dict.fromkeys(sample_atom_array.pn_unit_iid.tolist())
        if np.any(
            (sample_atom_array.pn_unit_iid == pn_unit_iid)
            & (sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
        )
    ]
    pred_protein_iids = set(
        str(pn_unit_iid)
        for pn_unit_iid in pred_atom_array.pn_unit_iid[
            pred_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
        ]
    )
    protein_pn_unit_iids = [
        pn_unit_iid for pn_unit_iid in sample_protein_iids if pn_unit_iid in pred_protein_iids
    ]
    if not protein_pn_unit_iids:
        raise ValueError("No common protein PN units found between sample and prediction")
    ca_match = match_resolved_ca(
        sample_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        pn_unit_iids=protein_pn_unit_iids,
    )

    pred_ca_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_ca_mask[ca_match.pred_indices] = True

    # Preserve the explicit key-paired order. Independent boolean masks would
    # sort each array by its own atom order and silently scramble correspondence
    # when AF3 emits chains in a different order from the designed sample.
    sample_ca = sample_atom_array[ca_match.sample_indices]
    pred_ca = pred_atom_array[ca_match.pred_indices]

    # Align pred CA to sample CA using atomworks align_atom_arrays
    # This aligns pred_ca to sample_ca and applies the transformation to the full pred_atom_array
    aligned_pred_atom_array, ca_rmsd = align_atom_arrays(
        mbl_sele=pred_ca,           # CA atoms from pred to align
        tgt_sele=sample_ca,         # CA atoms from sample as target
        mbl_full=pred_atom_array    # Full pred structure to transform
    )

    # Write aligned coords to mmcif
    if save_aligned:
        out_file = f"{Path(pred_sample_path).parent}/{Path(pred_sample_path).stem}_ca_aligned.cif"
        save_cif_file(aligned_pred_atom_array, out_file)

    # Create CA atom mask for pLDDT extraction (matching aligned structure)
    ca_atom_mask = torch.tensor(pred_ca_mask, dtype=torch.bool)

    # Compute metrics.
    # for metric in ["sc_ca_rmsd", "avg_ca_plddt", "tmalign_score"]:
    for metric in ["sc_ca_rmsd", "avg_ca_plddt"]:
        if metric == "sc_ca_rmsd":
            # CA RMSD computed via align_atom_arrays (already a float)
            if not isinstance(ca_rmsd, float):
                ca_rmsd = float(ca_rmsd.item()) if hasattr(ca_rmsd, "item") else float(ca_rmsd)

            metrics[metric] = ca_rmsd

        elif metric == "avg_ca_plddt":
            # Compute average pLDDT across all CA atoms.
            confidence_dir = str(pred_sample_path.parent)
            confidence_file_name = re.sub(r'_model$', '_confidences', str(pred_sample_path.stem)) + '.json'

            avg_ca_plddt = extract_af3_confidence_metrics(confidence_file_path=f"{confidence_dir}/{confidence_file_name}",
                                                        atom_array=pred_atom_array,
                                                        mask=ca_atom_mask,
                                                        metrics_to_extract="atom_plddts",
                                                        return_mean=True)
            metrics[metric] = avg_ca_plddt

        # elif metric == "tmalign_score":
        #     # Compute TM-score using TM-align.
        #     tmalign_score, _ = _compute_tmalign_score(pred_pdb, design_pdb)
        #     metrics[metric] = tmalign_score

    return metrics
