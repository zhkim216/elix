"""Shared row annotation for per-center sampling schemes."""

import logging

import pandas as pd

from allatom_design.data.utils.pn_unit import normalize_ligand_ccd_key

logger = logging.getLogger("allatom_design.data.datasets.atomworks_sd.sampling")

_ALPHA_KEY_BY_INTERFACE_TYPE = {
    "bmm_protein": "alpha_protein_metal",
    "bmsm_protein": "alpha_protein_small_molecule",
    "nuc_lig_protein": "alpha_protein_nuc_lig",
    "peptide_protein": "alpha_protein_peptide",
    "protein_protein": "alpha_protein_protein",
}


def _resolve_interface_type_alphas(
    alphas_interface: dict[str, float],
) -> dict[str, float]:
    return {
        interface_type: float(alphas_interface.get(alpha_key, 0.0))
        for interface_type, alpha_key in _ALPHA_KEY_BY_INTERFACE_TYPE.items()
    }


def _protein_clusters(row: pd.Series) -> list:
    clusters = row.get("protein_cluster_multiset", ())
    if isinstance(clusters, float) and pd.isna(clusters):
        return []
    return list(clusters)


def _ligand_key(row: pd.Series):
    if row.get("interface_type") == "protein_protein":
        return ("protein_interface", "none")
    return row.get(
        "ligand_ccd_key",
        normalize_ligand_ccd_key(row.get("q_pn_unit_non_polymer_res_names")),
    )


def _add_pair_cluster_columns(
    interface_df: pd.DataFrame,
    alpha_by_interface_type: dict[str, float],
) -> None:
    if interface_df.empty:
        interface_df["pair_cluster"] = []
        interface_df["pair_cluster_size"] = []
        interface_df["alpha"] = []
        return

    unknown_types = sorted(
        interface_type
        for interface_type in interface_df["interface_type"].dropna().unique()
        if interface_type not in alpha_by_interface_type
    )
    if unknown_types:
        logger.warning(
            "Unknown interface_type values get alpha=0: %s",
            unknown_types,
        )
    interface_df["pair_cluster"] = interface_df.apply(
        lambda row: (
            _ligand_key(row),
            tuple(
                sorted(
                    (("seq", cluster) for cluster in _protein_clusters(row)),
                    key=repr,
                )
            ),
        ),
        axis=1,
    )
    pair_cluster_sizes = interface_df["pair_cluster"].value_counts()
    interface_df["pair_cluster_size"] = interface_df["pair_cluster"].map(
        pair_cluster_sizes
    )
    interface_df["alpha"] = interface_df.apply(
        lambda row: alpha_by_interface_type.get(row.get("interface_type"), 0.0),
        axis=1,
    )
