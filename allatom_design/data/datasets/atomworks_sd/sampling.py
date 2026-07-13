"""Sampling-weight dispatch and validation for the SD dataset."""

import numpy as np
import pandas as pd

from allatom_design.data.datasets.atomworks_sd.sampling_schemes.interface_fraction import (
    _add_interface_fraction_sampling_weights,
)
from allatom_design.data.datasets.atomworks_sd.sampling_schemes.ligand_grouped import (
    _add_ligand_grouped_protein_equalized_weights,
)
from allatom_design.data.datasets.atomworks_sd.sampling_schemes.fixed_k import (
    _add_fixed_k_sampling_weights,
)

SAMPLING_FIXED_K = "fixed_k"
SAMPLING_INTERFACE_FRACTION = "interface_fraction"
SAMPLING_LIGAND_GROUPED_PROTEIN_EQUALIZED = "ligand_grouped_protein_equalized"
VALID_SAMPLING_SCHEMES = (
    SAMPLING_FIXED_K,
    SAMPLING_INTERFACE_FRACTION,
    SAMPLING_LIGAND_GROUPED_PROTEIN_EQUALIZED,
)


def _canonicalize_sampling_scheme(sampling_scheme: str) -> str:
    if sampling_scheme not in VALID_SAMPLING_SCHEMES:
        raise ValueError(
            f"Unknown `clustering.sampling_scheme`: {sampling_scheme!r}. "
            f"Expected one of {VALID_SAMPLING_SCHEMES}."
        )
    return sampling_scheme


def add_sampling_weights(
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    alphas_interface: dict[str, float],
    cluster_col: str = "q_pn_unit_cluster_id",
    fixed_k: float | None = None,
    single_protein_context_weight: float = 1.0,
    multi_protein_context_weight: float = 1.0,
    clustering_cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch sampling weights while preserving the stable dataset API."""

    clustering_cfg = clustering_cfg or {}
    sampling_scheme = _canonicalize_sampling_scheme(
        str(clustering_cfg.get("sampling_scheme", SAMPLING_FIXED_K))
    )
    if sampling_scheme == SAMPLING_FIXED_K:
        if fixed_k is None:
            raise ValueError("`fixed_k` is required for the fixed-K sampling scheme.")
        return _add_fixed_k_sampling_weights(
            monomer_df=monomer_df,
            interface_df=interface_df,
            alphas_interface=alphas_interface,
            cluster_col=cluster_col,
            fixed_k=fixed_k,
            single_protein_context_weight=single_protein_context_weight,
            multi_protein_context_weight=multi_protein_context_weight,
        )
    if sampling_scheme == SAMPLING_INTERFACE_FRACTION:
        interface_grouping_scheme = str(
            clustering_cfg.get("interface_grouping_scheme", "per_center")
        )
        if interface_grouping_scheme != "per_center":
            raise ValueError(
                "`clustering.sampling_scheme='interface_fraction'` requires "
                "`clustering.interface_grouping_scheme='per_center'`; "
                f"got {interface_grouping_scheme!r}."
            )
        return _add_interface_fraction_sampling_weights(
            monomer_df=monomer_df,
            interface_df=interface_df,
            alphas_interface=alphas_interface,
            cluster_col=cluster_col,
            single_protein_context_weight=single_protein_context_weight,
            cfg=clustering_cfg.get("interface_fraction_sampling_weights"),
        )
    return _add_ligand_grouped_protein_equalized_weights(
        monomer_df=monomer_df,
        interface_df=interface_df,
        cluster_col=cluster_col,
        cfg=clustering_cfg.get("ligand_grouped_sampling_weights", {}),
    )


def validate_sampling_weights(
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
) -> None:
    weights = np.concatenate(
        [
            monomer_df["sampling_weight"].to_numpy(dtype=float),
            interface_df["sampling_weight"].to_numpy(dtype=float),
        ]
    )
    if len(weights) == 0:
        raise ValueError("Train dataset has no rows after filtering.")
    if not np.isfinite(weights).all():
        raise ValueError("Sampling weights contain non-finite values.")
    if (weights < 0).any():
        raise ValueError("Sampling weights contain negative values.")
    if weights.sum() <= 0:
        raise ValueError("Sampling weights have zero total mass.")

    if len(interface_df) > 0 and interface_df["sampling_weight"].sum() <= 0:
        raise ValueError("Interface rows exist but have zero total sampling mass.")
    if len(monomer_df) > 0 and monomer_df["sampling_weight"].sum() <= 0:
        raise ValueError("Monomer rows exist but have zero total sampling mass.")
