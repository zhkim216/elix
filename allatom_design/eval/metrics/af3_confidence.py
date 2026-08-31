import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from biotite.structure import AtomArray
from torchtyping import TensorType


@lru_cache(maxsize=16)
def _load_confidence_json_cached(
    confidence_file_path: str,
    mtime_ns: int,
    size: int,
) -> dict:
    del mtime_ns, size
    with open(confidence_file_path, "r") as f:
        return json.load(f)


def _load_confidence_json(confidence_file_path: str) -> dict:
    path = Path(confidence_file_path)
    stat = path.stat()
    return _load_confidence_json_cached(str(path), stat.st_mtime_ns, stat.st_size)


def extract_af3_confidence_metrics(confidence_file_path: str = None,
                                    atom_array: AtomArray = None,
                                    mask: TensorType["n", bool] = None,
                                    metrics_to_extract: str = "atom_plddts",
                                    return_mean: bool = True):
    """
    Extract confidence metrics from an AF3 confidence file.

    Note: aw_parse adds unresolved residues with NaN coordinates, so atom_array may have
    more atoms than the confidence file. We filter to only valid (non-NaN) coordinates.
    """
    confidence_data = _load_confidence_json(confidence_file_path)

    if metrics_to_extract == "atom_plddts":
        metric = torch.tensor(confidence_data["atom_plddts"], dtype=torch.float32)

        # Filter out NaN coordinate atoms: aw_parse with add_missing_atoms=True
        # adds unresolved atoms with NaN coordinates that don't exist in AF3 output.
        valid_coords_mask = ~np.isnan(atom_array.coord).any(axis=1)
        num_valid_atoms = int(valid_coords_mask.sum())

        assert len(metric) == num_valid_atoms, (
            f"Number of pLDDTs ({len(metric)}) != number of valid (non-NaN) atoms ({num_valid_atoms}). "
            f"Total atoms in atom_array: {len(atom_array)}, NaN atoms: {len(atom_array) - num_valid_atoms}"
        )

        # Filter mask to only valid (non-NaN) atoms so it aligns with metric
        if isinstance(mask, np.ndarray):
            mask_torch = torch.tensor(mask[valid_coords_mask], dtype=torch.bool)
        else:
            mask_torch = mask[torch.tensor(valid_coords_mask, dtype=torch.bool)].bool()

        # Apply mask to pLDDTs
        metric = metric[mask_torch]

        if return_mean:
            if len(metric) == 0:
                return None
            return metric.mean().item()
        else:
            return metric
    elif metrics_to_extract == "iptm":
        metric = confidence_data["iptm"]

    elif metrics_to_extract == "interface_min_pae":
        try:
            pae_01 = confidence_data["chain_pair_pae_min"][0][1]
            pae_10 = confidence_data["chain_pair_pae_min"][1][0]
            metric = min(pae_01, pae_10)
        except:
            print(f"Warning: Failed to extract interface_min_pae from confidence file: {confidence_file_path}")
            metric = None

    else:
        raise ValueError(f"Invalid metric to extract: {metrics_to_extract}")

    return metric
