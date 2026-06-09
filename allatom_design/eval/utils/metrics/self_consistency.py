import re
from pathlib import Path

import numpy as np
import torch
from biotite.structure import AtomArray

import atomworks.enums as aw_enums
from atomworks.ml.utils.geometry import align_atom_arrays

from allatom_design.utils.sample_io_utils import save_cif_file

from allatom_design.eval.utils.metrics.af3_confidence import extract_af3_confidence_metrics


def _ca_mask_by_chain_res_offset(atom_array: AtomArray, ca_mask: np.ndarray) -> dict[tuple[str, int], int]:
    ca_indices = np.where(ca_mask)[0]
    chain_min_res_id = {}
    for idx in ca_indices:
        chain_id = str(atom_array.chain_id[idx])
        res_id = int(atom_array.res_id[idx])
        chain_min_res_id[chain_id] = min(res_id, chain_min_res_id.get(chain_id, res_id))

    return {
        (str(atom_array.chain_id[idx]), int(atom_array.res_id[idx]) - chain_min_res_id[str(atom_array.chain_id[idx])]): idx
        for idx in ca_indices
    }


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

    # Build initial CA masks (without NaN filtering) to identify matching residue positions
    sample_ca_mask_initial = (sample_atom_array.atom_name == "CA") & (sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
    pred_ca_mask_initial = (pred_atom_array.atom_name == "CA") & (pred_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L) & (pred_atom_array.res_name != "UNK")

    sample_ca_by_key = _ca_mask_by_chain_res_offset(sample_atom_array, sample_ca_mask_initial)
    pred_ca_by_key = _ca_mask_by_chain_res_offset(pred_atom_array, pred_ca_mask_initial)
    common_keys = [key for key in sample_ca_by_key if key in pred_ca_by_key]
    if len(common_keys) == 0:
        raise ValueError("No common protein CA residue offsets found between sample and pred")

    sample_ca_indices_initial = np.array([sample_ca_by_key[key] for key in common_keys])
    pred_ca_indices_initial = np.array([pred_ca_by_key[key] for key in common_keys])

    # Compute joint resolved mask: exclude positions where EITHER array has NaN coordinates
    # (NaN occurs in native structures for unresolved residues; AF3 predictions never have NaN)
    sample_ca_resolved_mask = ~np.isnan(sample_atom_array[sample_ca_indices_initial].coord[:, 0])
    pred_ca_resolved_mask = ~np.isnan(pred_atom_array[pred_ca_indices_initial].coord[:, 0])
    ca_resolved_mask = sample_ca_resolved_mask & pred_ca_resolved_mask

    # Apply joint resolved mask back to full atom_array-level masks
    sample_ca_mask = np.zeros(len(sample_atom_array), dtype=bool)
    sample_ca_mask[sample_ca_indices_initial[ca_resolved_mask]] = True

    pred_ca_mask = np.zeros(len(pred_atom_array), dtype=bool)
    pred_ca_mask[pred_ca_indices_initial[ca_resolved_mask]] = True

    sample_ca = sample_atom_array[sample_ca_mask]
    pred_ca = pred_atom_array[pred_ca_mask]

    assert (sample_ca.res_name == pred_ca.res_name).all(), "Sample and pred CA residues must match"

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
            if type(ca_rmsd) != float:
                try:
                    ca_rmsd = ca_rmsd.item()
                except:
                    ca_rmsd = float(ca_rmsd)

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
