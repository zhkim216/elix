"""Fixed-K capped per-center sampling weights."""

import logging

import numpy as np
import pandas as pd

from allatom_design.data.datasets.atomworks_sd.sampling_schemes._per_center import (
    _add_pair_cluster_columns,
    _protein_clusters,
    _resolve_interface_type_alphas,
)

logger = logging.getLogger("allatom_design.data.datasets.atomworks_sd.sampling")


def _add_fixed_k_sampling_weights(
    *,
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    alphas_interface: dict[str, float],
    cluster_col: str,
    fixed_k: float,
    single_protein_context_weight: float,
    multi_protein_context_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monomer_df = monomer_df.copy()
    interface_df = interface_df.copy()
    fixed_k = float(fixed_k)
    single_protein_context_weight = float(single_protein_context_weight)
    multi_protein_context_weight = float(multi_protein_context_weight)

    if not np.isfinite(fixed_k) or fixed_k <= 0:
        raise ValueError(f"`fixed_k` must be finite and positive; got {fixed_k!r}.")
    if "protein_cluster_multiset" not in interface_df.columns:
        raise ValueError("interface_df must contain `protein_cluster_multiset`.")

    alpha_by_interface_type = _resolve_interface_type_alphas(alphas_interface)

    def _context_weight(row):
        n_prot = row.get("n_prot", len(_protein_clusters(row)))
        if pd.isna(n_prot):
            n_prot = len(_protein_clusters(row))
        if int(n_prot) >= 2:
            return multi_protein_context_weight
        return single_protein_context_weight

    _add_pair_cluster_columns(interface_df, alpha_by_interface_type)
    if interface_df.empty:
        interface_df["context_weight"] = []
        interface_df["sampling_weight"] = []
        interface_contrib = {}
    else:
        interface_df["context_weight"] = interface_df.apply(
            _context_weight,
            axis=1,
        )
        interface_df["sampling_weight"] = (
            interface_df["alpha"]
            * interface_df["context_weight"]
            / interface_df["pair_cluster_size"]
        )
        interface_contrib = _compute_interface_contrib(interface_df)

        scaling = {
            cluster_id: fixed_k / contrib
            for cluster_id, contrib in interface_contrib.items()
            if contrib > fixed_k and contrib > 0
        }
        if scaling:
            interface_df["sampling_weight"] = interface_df.apply(
                lambda row: row["sampling_weight"]
                * min(
                    [
                        scaling[cluster]
                        for cluster in _protein_clusters(row)
                        if cluster in scaling
                    ]
                    or [1.0]
                ),
                axis=1,
            )
            interface_contrib = _compute_interface_contrib(interface_df)

    monomer_counts = monomer_df[cluster_col].value_counts().to_dict()

    def _monomer_weight(row):
        cluster_id = row[cluster_col]
        target = fixed_k - interface_contrib.get(cluster_id, 0.0)
        return max(target, 0.0) / monomer_counts.get(cluster_id, 1)

    monomer_df["sampling_weight"] = monomer_df.apply(_monomer_weight, axis=1)
    logger.info(
        "SD sampling weights: monomer_rows=%d, interface_rows=%d, K=%.4f, "
        "alpha_metal=%.4f, alpha_small_molecule=%.4f, alpha_nuc_lig=%.4f, "
        "alpha_peptide=%.4f, alpha_protein_protein=%.4f, "
        "single_protein_context_weight=%.4f, multi_protein_context_weight=%.4f",
        len(monomer_df),
        len(interface_df),
        fixed_k,
        alpha_by_interface_type["bmm_protein"],
        alpha_by_interface_type["bmsm_protein"],
        alpha_by_interface_type["nuc_lig_protein"],
        alpha_by_interface_type["peptide_protein"],
        alpha_by_interface_type["protein_protein"],
        single_protein_context_weight,
        multi_protein_context_weight,
    )
    return monomer_df, interface_df


def _compute_interface_contrib(interface_df: pd.DataFrame) -> dict:
    contrib = {}
    for _, row in interface_df.iterrows():
        weight = float(row["sampling_weight"])
        for cluster_id in _protein_clusters(row):
            contrib[cluster_id] = contrib.get(cluster_id, 0.0) + weight
    return contrib
