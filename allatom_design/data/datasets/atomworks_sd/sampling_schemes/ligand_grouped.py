"""Ligand-grouped component weighting with protein-mass equalization."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("allatom_design.data.datasets.atomworks_sd.sampling")


def _add_ligand_grouped_protein_equalized_weights(
    *,
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    cluster_col: str,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply component-weighted base weights, then equalize protein clusters.

    The ligand-grouped training profile filters to ``n_prot == 1`` before sampling.
    That makes each row attributable to exactly one protein sequence cluster,
    so normalizing the total base mass per protein cluster is unambiguous while
    preserving the monomer/interface ratio induced by beta and alpha values.
    """

    monomer_df = monomer_df.copy()
    interface_df = interface_df.copy()
    required_alpha_keys = (
        "alpha_protein",
        "alpha_small_molecule",
        "alpha_metal",
        "alpha_peptide",
        "alpha_nuc_ligand",
        "alpha_nuc_polymer",
    )
    missing = [key for key in required_alpha_keys if key not in cfg]
    if missing:
        raise KeyError(
            "Ligand-grouped sampling config is missing keys: "
            f"{missing}"
        )
    beta_monomer = float(cfg.get("beta_monomer", 1.0))
    beta_interface = float(cfg.get("beta_interface", 1.0))
    if beta_monomer < 0 or beta_interface < 0:
        raise ValueError("Ligand-grouped beta weights must be non-negative.")
    alphas = {key: float(cfg[key]) for key in required_alpha_keys}
    if any(value < 0 for value in alphas.values()):
        raise ValueError("Ligand-grouped alpha weights must be non-negative.")

    if cluster_col not in monomer_df.columns:
        raise KeyError(
            f"monomer_df is missing protein cluster column {cluster_col!r}."
        )
    if "interface_cluster_key" not in interface_df.columns and not interface_df.empty:
        raise KeyError(
            "Ligand-grouped interface rows require `interface_cluster_key`."
        )

    monomer_df["sampling_cluster"] = monomer_df[cluster_col].apply(
        lambda cluster: ("monomer", cluster)
    )
    monomer_cluster_sizes = monomer_df["sampling_cluster"].value_counts()
    monomer_df["sampling_cluster_size"] = monomer_df["sampling_cluster"].map(
        monomer_cluster_sizes
    )
    monomer_df["base_sampling_weight"] = (
        beta_monomer
        * alphas["alpha_protein"]
        * monomer_df["n_prot"].astype(float)
        / monomer_df["sampling_cluster_size"].astype(float)
    )

    if interface_df.empty:
        interface_df["sampling_cluster"] = []
        interface_df["sampling_cluster_size"] = []
        interface_df["pair_cluster"] = []
        interface_df["pair_cluster_size"] = []
        interface_df["alpha"] = []
        interface_df["base_sampling_weight"] = []
    else:
        protein_cluster_counts = interface_df["protein_cluster_multiset"].apply(
            lambda clusters: len(tuple(clusters))
        )
        if not (protein_cluster_counts == 1).all():
            bad = interface_df.loc[
                protein_cluster_counts != 1,
                ["example_id", "protein_cluster_multiset"],
            ]
            raise ValueError(
                "Ligand-grouped protein equalization requires rows "
                "with exactly one protein cluster; "
                f"examples={bad.head(10).to_dict('records')}"
            )
        interface_df["sampling_cluster"] = interface_df["interface_cluster_key"]
        interface_cluster_sizes = interface_df["sampling_cluster"].value_counts()
        interface_df["sampling_cluster_size"] = interface_df[
            "sampling_cluster"
        ].map(interface_cluster_sizes)
        interface_df["pair_cluster"] = interface_df["sampling_cluster"]
        interface_df["pair_cluster_size"] = interface_df["sampling_cluster_size"]
        interface_df["alpha"] = (
            alphas["alpha_protein"] * interface_df["n_prot"].astype(float)
            + alphas["alpha_small_molecule"]
            * interface_df["n_small_molecule"].astype(float)
            + alphas["alpha_metal"] * interface_df["n_metal"].astype(float)
            + alphas["alpha_peptide"] * interface_df["n_peptide"].astype(float)
            + alphas["alpha_nuc_ligand"]
            * interface_df["n_nuc_ligand"].astype(float)
            + alphas["alpha_nuc_polymer"]
            * interface_df["n_nuc_polymer"].astype(float)
        )
        interface_df["base_sampling_weight"] = (
            beta_interface
            * interface_df["alpha"]
            / interface_df["sampling_cluster_size"].astype(float)
        )

    protein_mass: dict[object, float] = {}
    for cluster_id, weight in zip(
        monomer_df[cluster_col],
        monomer_df["base_sampling_weight"],
    ):
        protein_mass[cluster_id] = protein_mass.get(cluster_id, 0.0) + float(weight)
    for _, row in interface_df.iterrows():
        cluster_id = tuple(row["protein_cluster_multiset"])[0]
        protein_mass[cluster_id] = protein_mass.get(cluster_id, 0.0) + float(
            row["base_sampling_weight"]
        )

    bad_mass = {
        cluster_id: mass
        for cluster_id, mass in protein_mass.items()
        if not np.isfinite(mass) or mass <= 0
    }
    if bad_mass:
        raise ValueError(
            "Protein-cluster base sampling mass must be finite and positive; "
            f"bad={bad_mass}"
        )

    monomer_df["sampling_weight"] = [
        float(weight) / protein_mass[cluster_id]
        for cluster_id, weight in zip(
            monomer_df[cluster_col],
            monomer_df["base_sampling_weight"],
        )
    ]
    interface_df["sampling_weight"] = [
        float(row.base_sampling_weight)
        / protein_mass[tuple(row.protein_cluster_multiset)[0]]
        for row in interface_df.itertuples(index=False)
    ]

    logger.info(
        "Ligand-grouped sampling weights: monomer_rows=%d, interface_rows=%d, "
        "protein_clusters=%d, beta_monomer=%.4f, beta_interface=%.4f, alphas=%s",
        len(monomer_df),
        len(interface_df),
        len(protein_mass),
        beta_monomer,
        beta_interface,
        alphas,
    )
    return monomer_df, interface_df
