"""Sampling-weight computation and validation for the SD dataset.

Computes per-row sampling weights for protein monomer chains and interface rows
using a cluster-balanced scheme, and validates the resulting weights.
"""

import logging

import numpy as np
import pandas as pd

from allatom_design.data.utils.pn_unit import normalize_ligand_ccd_key

logger = logging.getLogger(__name__)


def add_sampling_weights(
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    alphas_interface: dict[str, float],
    cluster_col: str = "q_pn_unit_cluster_id",
    k_percentile: float = 100.0,
    single_protein_context_weight: float = 1.0,
    multi_protein_context_weight: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monomer_df = monomer_df.copy()
    interface_df = interface_df.copy()
    single_protein_context_weight = float(single_protein_context_weight)
    multi_protein_context_weight = float(multi_protein_context_weight)

    if "protein_cluster_multiset" not in interface_df.columns:
        raise ValueError("interface_df must contain `protein_cluster_multiset`.")

    alpha_by_interface_type = {
        "bmm_protein": float(alphas_interface.get("alpha_protein_metal", 0.0)),
        "bmsm_protein": float(alphas_interface.get("alpha_protein_small_molecule", 0.0)),
        "nuc_lig_protein": float(alphas_interface.get("alpha_protein_nuc_lig", 0.0)),
        "peptide_protein": float(alphas_interface.get("alpha_protein_peptide", 0.0)),
        "protein_protein": float(alphas_interface.get("alpha_protein_protein", 0.0)),
    }

    def _protein_clusters(row):
        clusters = row.get("protein_cluster_multiset", ())
        if isinstance(clusters, float) and pd.isna(clusters):
            return []
        return list(clusters)

    def _sort_key(value):
        return repr(value)

    def _ligand_key(row):
        if row.get("interface_type") == "protein_protein":
            return ("protein_interface", "none")
        return row.get("ligand_ccd_key", normalize_ligand_ccd_key(row.get("q_pn_unit_non_polymer_res_names")))

    def _alpha(row):
        return alpha_by_interface_type.get(row.get("interface_type"), 0.0)

    def _context_weight(row):
        n_prot = row.get("n_prot", len(_protein_clusters(row)))
        if pd.isna(n_prot):
            n_prot = len(_protein_clusters(row))
        return multi_protein_context_weight if int(n_prot) >= 2 else single_protein_context_weight

    if interface_df.empty:
        interface_df["pair_cluster"] = []
        interface_df["pair_cluster_size"] = []
        interface_df["alpha"] = []
        interface_df["context_weight"] = []
        interface_df["sampling_weight"] = []
        interface_contrib = {}
        k_value = 1.0
    else:
        unknown_types = sorted(
            t for t in interface_df["interface_type"].dropna().unique()
            if t not in alpha_by_interface_type
        )
        if unknown_types:
            logger.warning(
                "Unknown interface_type values get alpha=0: %s",
                unknown_types,
            )
        interface_df["pair_cluster"] = interface_df.apply(
            lambda row: (
                _ligand_key(row),
                tuple(sorted((("seq", c) for c in _protein_clusters(row)), key=_sort_key)),
            ),
            axis=1,
        )
        pair_cluster_sizes = interface_df["pair_cluster"].value_counts()
        interface_df["pair_cluster_size"] = interface_df["pair_cluster"].map(pair_cluster_sizes)
        interface_df["alpha"] = interface_df.apply(_alpha, axis=1)
        interface_df["context_weight"] = interface_df.apply(_context_weight, axis=1)
        interface_df["sampling_weight"] = (
            interface_df["alpha"] * interface_df["context_weight"] / interface_df["pair_cluster_size"]
        )
        interface_contrib = _compute_interface_contrib(interface_df, _protein_clusters)

        if interface_contrib and max(interface_contrib.values()) > 0:
            k_value = float(np.percentile(list(interface_contrib.values()), k_percentile))
        else:
            k_value = 1.0

        scaling = {
            cluster_id: k_value / contrib
            for cluster_id, contrib in interface_contrib.items()
            if contrib > k_value and contrib > 0
        }
        if scaling:
            interface_df["sampling_weight"] = interface_df.apply(
                lambda row: row["sampling_weight"]
                * min([scaling[c] for c in _protein_clusters(row) if c in scaling] or [1.0]),
                axis=1,
            )
            interface_contrib = _compute_interface_contrib(interface_df, _protein_clusters)

    monomer_counts = monomer_df[cluster_col].value_counts().to_dict()

    def _monomer_weight(row):
        cluster_id = row[cluster_col]
        target = k_value - interface_contrib.get(cluster_id, 0.0)
        return max(target, 0.0) / monomer_counts.get(cluster_id, 1)

    monomer_df["sampling_weight"] = monomer_df.apply(_monomer_weight, axis=1)
    logger.info(
        "SD sampling weights: monomer_rows=%d, interface_rows=%d, K=%.4f, "
        "alpha_metal=%.4f, alpha_small_molecule=%.4f, alpha_nuc_lig=%.4f, "
        "alpha_peptide=%.4f, alpha_protein_protein=%.4f, "
        "single_protein_context_weight=%.4f, multi_protein_context_weight=%.4f",
        len(monomer_df),
        len(interface_df),
        k_value,
        alpha_by_interface_type["bmm_protein"],
        alpha_by_interface_type["bmsm_protein"],
        alpha_by_interface_type["nuc_lig_protein"],
        alpha_by_interface_type["peptide_protein"],
        alpha_by_interface_type["protein_protein"],
        single_protein_context_weight,
        multi_protein_context_weight,
    )
    return monomer_df, interface_df


def _compute_interface_contrib(interface_df: pd.DataFrame, protein_clusters_fn) -> dict:
    contrib = {}
    for _, row in interface_df.iterrows():
        weight = float(row["sampling_weight"])
        for cluster_id in protein_clusters_fn(row):
            contrib[cluster_id] = contrib.get(cluster_id, 0.0) + weight
    return contrib


def validate_sampling_weights(monomer_df: pd.DataFrame, interface_df: pd.DataFrame) -> None:
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
