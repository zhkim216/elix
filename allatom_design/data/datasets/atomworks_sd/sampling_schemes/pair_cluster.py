"""Uniform monomer rows with alpha-balanced interface pair clusters."""

import logging

import pandas as pd

from allatom_design.data.datasets.atomworks_sd.sampling_schemes._per_center import (
    _add_pair_cluster_columns,
    _resolve_interface_type_alphas,
)

logger = logging.getLogger("allatom_design.data.datasets.atomworks_sd.sampling")


def _add_pair_cluster_balanced_sampling_weights(
    *,
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    alphas_interface: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balance interface pair clusters without balancing protein clusters."""

    monomer_df = monomer_df.copy()
    interface_df = interface_df.copy()
    if "protein_cluster_multiset" not in interface_df.columns:
        raise ValueError("interface_df must contain `protein_cluster_multiset`.")

    alpha_by_interface_type = _resolve_interface_type_alphas(alphas_interface)
    _add_pair_cluster_columns(interface_df, alpha_by_interface_type)

    monomer_df["sampling_weight"] = 1.0
    interface_df["sampling_weight"] = (
        interface_df["alpha"].astype(float)
        / interface_df["pair_cluster_size"].astype(float)
    )

    logger.info(
        "Pair-cluster-balanced sampling weights: monomer_rows=%d, "
        "interface_rows=%d, pair_clusters=%d, alphas=%s",
        len(monomer_df),
        len(interface_df),
        interface_df["pair_cluster"].nunique(dropna=False),
        alpha_by_interface_type,
    )
    return monomer_df, interface_df
