"""Metadata preprocessing for the SD dataset.

Pure functions that turn raw metadata parquet files into the train/val indices:
phase splitting, modality columns, external-evidence flags, filtering and the
parsing of monomer / interface / val rows.

These were previously ``AtomworksSDDataset`` methods; they now take the data
``cfg`` explicitly so the dataset orchestration stays thin and the steps are
independently testable.
"""

import logging

import atomworks.enums as aw_enums
import numpy as np
import pandas as pd
from atomworks.ml.datasets.parsers import GenericDFParser
from atomworks.ml.example_id import generate_example_id
from atomworks.ml.utils.io import read_parquet_with_metadata
from omegaconf import DictConfig

from allatom_design.data.utils.nucleic_acid_groups import (
    DEFAULT_NUCLEIC_ACID_GROUP_DISTANCE_CUTOFF,
    DEFAULT_NUCLEIC_ACID_LIGAND_MAX_RESIDUES,
    add_nucleic_acid_group_columns,
)
from allatom_design.data.utils.pn_unit import (
    load_small_molecule_artifact_codes,
    parse_pn_unit_iids_value,
    series_has_any_exact_ccd,
)

from allatom_design.data.datasets.atomworks_sd.interface import build_interface_df
from allatom_design.data.datasets.atomworks_sd.selectors import (
    metal_center_mask,
    nucleic_acid_ligand_center_mask,
    peptide_center_mask,
    small_molecule_center_mask,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Generic column helpers
# --------------------------------------------------------------------------- #
def ensure_example_id_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "example_id" not in out.columns and out.index.name == "example_id":
        out = out.reset_index()
    if "example_id" not in out.columns:
        raise KeyError("metadata dataframe is missing required column `example_id`.")
    return out


def resolve_cluster_columns(cfg: dict | DictConfig | None) -> tuple[str, str]:
    """Return (active_cluster_col, val_exclusion_cluster_col) from cfg.

    ``cluster_id_col`` selects the clustering used for cluster-balanced sampling
    weights and ``protein_cluster_multiset``. ``val_exclusion_cluster_id_col``
    selects the (sequence-identity) clustering used to remove train/val overlap.
    """
    cfg = cfg or {}
    active_col = cfg.get("cluster_id_col", "q_pn_unit_cluster_id")
    val_col = cfg.get("val_exclusion_cluster_id_col", "q_pn_unit_cluster_03_id")
    return str(active_col), str(val_col)


def _select_cluster_column(df: pd.DataFrame, col: str, role: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    if "q_pn_unit_cluster_id" in df.columns:
        logger.warning(
            "Configured %s=%r not found in metadata; falling back to existing "
            "`q_pn_unit_cluster_id`.",
            role,
            col,
        )
        return df["q_pn_unit_cluster_id"]
    raise KeyError(
        f"Configured {role}={col!r} not found in metadata and no "
        "`q_pn_unit_cluster_id` fallback column is available."
    )


def add_cluster_id_columns(
    metadata_df: pd.DataFrame,
    cfg: dict | DictConfig | None = None,
) -> pd.DataFrame:
    """Materialize the active and val-exclusion cluster columns.

    Writes two canonical in-memory columns used downstream:
      * ``q_pn_unit_cluster_id`` -- the configured active clustering (sampling).
      * ``q_pn_unit_cluster_val_id`` -- the configured val-exclusion clustering.
    """
    out = metadata_df.copy()
    active_col, val_col = resolve_cluster_columns(cfg)
    active_series = _select_cluster_column(out, active_col, "cluster_id_col")
    val_series = _select_cluster_column(out, val_col, "val_exclusion_cluster_id_col")
    out["q_pn_unit_cluster_id"] = active_series.to_numpy()
    out["q_pn_unit_cluster_val_id"] = val_series.to_numpy()
    logger.info(
        "Cluster columns resolved: active(sampling)=%r -> q_pn_unit_cluster_id, "
        "val_exclusion=%r -> q_pn_unit_cluster_val_id.",
        active_col,
        val_col,
    )
    return out


def add_derived_pn_unit_flags(
    metadata_df: pd.DataFrame,
    cfg: dict | DictConfig | None = None,
) -> pd.DataFrame:
    out = metadata_df.copy()
    cfg = cfg or {}
    nuc_chain_types = [chain_type.value for chain_type in aw_enums.ChainType.get_nucleic_acids()]
    if "q_pn_unit_is_nuc" not in out.columns:
        out["q_pn_unit_is_nuc"] = (
            out.get("q_pn_unit_is_polymer", pd.Series(False, index=out.index)).fillna(False).astype(bool)
            & out.get("q_pn_unit_type", pd.Series(np.nan, index=out.index)).isin(nuc_chain_types)
        )
    if "q_pn_unit_nucleic_acid_group_id" not in out.columns:
        out = add_nucleic_acid_group_columns(
            out,
            nucleic_acid_dist_threshold=cfg.get(
                "nucleic_acid_group_distance_cutoff",
                DEFAULT_NUCLEIC_ACID_GROUP_DISTANCE_CUTOFF,
            ),
            nucleic_acid_ligand_max_residues=cfg.get(
                "nucleic_acid_ligand_max_residues",
                DEFAULT_NUCLEIC_ACID_LIGAND_MAX_RESIDUES,
            ),
        )
    if "q_pn_unit_is_artifact" not in out.columns:
        artifact_codes = load_small_molecule_artifact_codes(cfg)
        out["q_pn_unit_is_artifact"] = series_has_any_exact_ccd(
            out.get("q_pn_unit_non_polymer_res_names"),
            artifact_codes,
            index=out.index,
        ) if artifact_codes else False
    return out


def collect_external_evidence(
    metadata_df: pd.DataFrame,
    output_col: str = "q_pn_unit_has_external_evidence",
    allowed_evidence_columns: list[str] | None = None,
    require_columns: bool = True,
) -> pd.DataFrame:
    """Attach a boolean external-evidence flag from configured source columns."""
    out = metadata_df.copy()
    allowed_evidence_columns = list(allowed_evidence_columns or [])

    missing = [col for col in allowed_evidence_columns if col not in out.columns]
    if missing and require_columns:
        raise KeyError(f"Metadata is missing configured evidence columns: {missing}")

    present_cols = [col for col in allowed_evidence_columns if col in out.columns]
    if present_cols:
        evidence = out[present_cols].fillna(False).astype(bool).any(axis=1)
    else:
        evidence = pd.Series(False, index=out.index)
    out[output_col] = evidence.astype(bool)
    return out


def resolve_evidence_columns(cfg: dict | DictConfig) -> list[str]:
    return cfg.get(
        "allowed_evidence_columns",
        [
            "metal_medba_evidence",
            "metal_pubmed_evidence",
        ],
    )


def add_chain_counts_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "protein_cluster_multiset" in df.columns:
        df["n_prot"] = df["protein_cluster_multiset"].apply(lambda clusters: len(clusters))
        interface_type = df.get("interface_type", pd.Series("", index=df.index)).fillna("")
        df["n_nuc"] = 0 # (interface_type == "nuc_lig_protein").astype(int) #! need to change 260608
        df["n_nuc_ligand"] = (interface_type == "nuc_lig_protein").astype(int)
        df["n_peptide"] = (interface_type == "peptide_protein").astype(int)
        df["n_small_molecule"] = (interface_type == "bmsm_protein").astype(int)
        df["n_metal"] = (interface_type == "bmm_protein").astype(int)
        return df

    df["n_prot"] = df["q_pn_unit_is_protein"].fillna(False).astype(bool).astype(int)
    df["n_nuc"] = df.get("q_pn_unit_is_nuc", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(int)
    df["n_nuc_ligand"] = df.get("q_pn_unit_is_nuc_ligand", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(int)
    df["n_peptide"] = df.get("q_pn_unit_is_peptide", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(int)
    df["n_small_molecule"] = df.get("q_pn_unit_is_small_molecule", pd.Series(False, index=df.index)).fillna(False).astype(bool).astype(int)
    df["n_metal"] = df["q_pn_unit_is_metal"].fillna(False).astype(bool).astype(int)
    return df


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def apply_filters(
    filters: list[str] | None,
    df: pd.DataFrame,
    allow_empty: bool = False,
) -> pd.DataFrame:
    if filters is None:
        return df
    for query in filters:
        df = apply_query(query, df, allow_empty=allow_empty)
    return df


def apply_query(query: str, df: pd.DataFrame, allow_empty: bool = False) -> pd.DataFrame:
    before = len(df)
    df = df.query(query)
    validate_filter_impact(query, before, len(df), allow_empty=allow_empty)
    return df


def validate_filter_impact(
    query: str,
    original_num_rows: int,
    filtered_num_rows: int,
    allow_empty: bool = False,
) -> None:
    if original_num_rows == 0:
        logger.warning("Query %r was applied to an empty dataset.", query)
        return
    if filtered_num_rows == original_num_rows:
        logger.warning("Query %r on dataset did not remove any rows.", query)
    elif filtered_num_rows == 0 and not allow_empty:
        raise ValueError(f"Query {query!r} on dataset removed all rows.")
    else:
        logger.info(
            "Query %r filtered dataset from %d to %d rows.",
            query,
            original_num_rows,
            filtered_num_rows,
        )


# --------------------------------------------------------------------------- #
# Train index construction
# --------------------------------------------------------------------------- #
def process_train_metadata_df(
    metadata_path: str,
    cfg: DictConfig,
    allowed_evidence_columns: list[str],
    phase: str,
) -> tuple[pd.DataFrame, list]:
    metadata_df = read_parquet_with_metadata(metadata_path)
    metadata_df = ensure_example_id_column(metadata_df)
    metadata_df = add_cluster_id_columns(metadata_df, cfg)
    metadata_df = add_derived_pn_unit_flags(metadata_df, cfg)
    metadata_df = collect_external_evidence(
        metadata_df,
        allowed_evidence_columns=allowed_evidence_columns,
        require_columns=cfg.get("require_evidence_columns", True),
    )
    metadata_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
    metadata_df, val_cluster_ids = add_phase_split(metadata_df, cfg)
    metadata_df = metadata_df[metadata_df["phase"] == phase]
    metadata_df = apply_filters(cfg.train_filters.metadata_filter, metadata_df)
    if cfg.query_pn_unit_iids_only:
        metadata_df = filter_metadata_to_query_pn_unit_iids_only(metadata_df, cfg)
    return metadata_df, val_cluster_ids


def add_phase_split(metadata_df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, list]:
    metadata_df = metadata_df.copy()
    val_cluster_ids: list = []
    val_split = _load_validation_pdb_ids(cfg)

    if cfg.debug:
        pdb_ids = metadata_df["pdb_id"].unique().tolist()
        n_debug = min(int(cfg.debug_num_ids), len(pdb_ids))
        debug_pdb_list = np.random.choice(pdb_ids, size=n_debug, replace=False)
        split_idx = 3 * n_debug // 4
        metadata_df.loc[metadata_df["pdb_id"].isin(debug_pdb_list[:split_idx]), "phase"] = "train"
        metadata_df.loc[metadata_df["pdb_id"].isin(debug_pdb_list[split_idx:]), "phase"] = "val"
    else:
        metadata_df.loc[~metadata_df["pdb_id"].str.lower().isin(val_split), "phase"] = "train"
        metadata_df.loc[metadata_df["pdb_id"].str.lower().isin(val_split), "phase"] = "val"

    if cfg.exclude_val_cluster:
        val_df = metadata_df[metadata_df["phase"] == "val"].copy()
        val_proteins = apply_filters(
            cfg.train_filters.protein_monomer_chain_filter,
            val_df,
            allow_empty=True,
        )
        val_cluster_ids = list(set(val_proteins["q_pn_unit_cluster_val_id"]))
    return metadata_df, val_cluster_ids


def _load_validation_pdb_ids(cfg: DictConfig) -> set[str]:
    validation_ids_file = cfg.get("validation_ids_file", None)
    if validation_ids_file not in (None, ""):
        with open(validation_ids_file, "r") as f:
            val_split = {x.lower().split(".")[0] for x in f.read().splitlines()}
        logger.info("Loading validation IDs from %s", validation_ids_file)
        return val_split

    val_metadata_path = cfg.get("val_metadata_path", None)
    if val_metadata_path in (None, ""):
        raise ValueError("Either `validation_ids_file` or `val_metadata_path` must be configured.")

    val_df = pd.read_parquet(val_metadata_path, columns=["pdb_id"])
    val_split = {str(pdb_id).lower().split(".")[0] for pdb_id in val_df["pdb_id"].dropna().unique()}
    logger.info("Loading validation IDs from %s", val_metadata_path)
    return val_split


def build_monomer_chain_df(
    metadata_df: pd.DataFrame,
    cfg: DictConfig,
    dataset_name: str,
    val_cluster_ids: list,
) -> pd.DataFrame:
    monomer_df = apply_filters(
        cfg.train_filters.protein_monomer_chain_filter,
        metadata_df.copy(),
    )
    if cfg.exclude_val_cluster and val_cluster_ids:
        before = len(monomer_df)
        monomer_df = monomer_df[~monomer_df["q_pn_unit_cluster_val_id"].isin(val_cluster_ids)]
        logger.info("Excluded %d monomer rows by val cluster.", before - len(monomer_df))

    monomer_df = add_chain_counts_info(monomer_df)
    monomer_df["example_id"] = monomer_df.apply(
        lambda row: generate_example_id(
            [dataset_name, "protein_monomer_chain"],
            row["pdb_id"],
            row["assembly_id"],
            [row["q_pn_unit_iid"]],
        ),
        axis=1,
    )
    monomer_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
    return monomer_df


def filter_metadata_to_query_pn_unit_iids_only(
    metadata_df: pd.DataFrame,
    cfg: DictConfig,
) -> pd.DataFrame:
    protein_df = apply_filters(
        cfg.train_filters.protein_monomer_chain_filter,
        metadata_df.copy(),
    )
    protein_mask = pd.Series(metadata_df.index.isin(protein_df.index), index=metadata_df.index)
    metal_mask = metal_center_mask(metadata_df, cfg)
    small_molecule_mask = small_molecule_center_mask(metadata_df, cfg)
    peptide_mask = peptide_center_mask(metadata_df, cfg)
    nuc_ligand_mask = nucleic_acid_ligand_center_mask(metadata_df, cfg)
    keep_mask = (
        protein_mask
        | metal_mask
        | small_molecule_mask
        | peptide_mask
        | nuc_ligand_mask
    )

    out = metadata_df[keep_mask].copy()
    if out.empty:
        raise ValueError("Metadata scope filter removed all rows.")
    logger.info(
        "Filtered train metadata to scope: %d -> %d rows "
        "(protein_monomer_candidates=%d, metal_center_candidates=%d, "
        "small_molecule_center_candidates=%d, peptide_center_candidates=%d, "
        "nuc_ligand_center_candidates=%d).",
        len(metadata_df),
        len(out),
        int(protein_mask.sum()),
        int(metal_mask.sum()),
        int(small_molecule_mask.sum()),
        int(peptide_mask.sum()),
        int(nuc_ligand_mask.sum()),
    )
    return out


def build_train_interface_df(
    metadata_df: pd.DataFrame,
    cfg: DictConfig,
    dataset_name: str,
    val_cluster_ids: list,
) -> pd.DataFrame:
    interface_filters = cfg.train_filters.get("interface_filter", {}) or {}
    interface_metadata_df = apply_filters(
        interface_filters.get("1", []),
        metadata_df.copy(),
    )
    protein_df = apply_filters(
        cfg.train_filters.protein_monomer_chain_filter,
        interface_metadata_df.copy(),
    )
    interface_df = build_interface_df(
        metadata_df=interface_metadata_df,
        protein_df=protein_df,
        dataset_name=dataset_name,
        cfg=cfg,
    )
    if cfg.exclude_val_cluster and val_cluster_ids and not interface_df.empty:
        val_clusters = set(val_cluster_ids)
        before = len(interface_df)
        interface_df = interface_df[
            ~interface_df["protein_val_cluster_multiset"].apply(
                lambda clusters: any(c in val_clusters for c in clusters)
            )
        ]
        logger.info("Excluded %d interface rows by val cluster.", before - len(interface_df))

    interface_df = add_chain_counts_info(interface_df)
    interface_df = apply_filters(interface_filters.get("2", []), interface_df)
    return interface_df


def parse_train_dfs(
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    cfg: dict | DictConfig,
) -> pd.DataFrame:
    chain_parser = GenericDFParser(pn_unit_iid_colnames=["q_pn_unit_iid"])
    interface_parser = GenericDFParser(pn_unit_iid_colnames=[])

    def parse_interface_row(row):
        parsed = interface_parser.parse(row)
        parsed["query_pn_unit_iids"] = list(row["query_pn_unit_iids"])
        parsed["ligand_pn_unit_iids"] = list(row["ligand_pn_unit_iids"])
        parsed["protein_pn_unit_iids"] = list(row["protein_pn_unit_iids"])
        row_query_only = row.get("query_pn_unit_iids_only", False)
        if isinstance(row_query_only, (float, np.floating)) and pd.isna(row_query_only):
            row_query_only = False
        if cfg.get("query_pn_unit_iids_only", False) or bool(row_query_only):
            parsed["query_pn_unit_iids_only"] = True
        for key in (
            "query_pn_unit_iids",
            "ligand_pn_unit_iids",
            "protein_pn_unit_iids",
            "crop_center_pn_unit_iids",
            "query_pn_unit_iids_only",
        ):
            parsed["extra_info"].pop(key, None)
        return parsed

    parsed_df = pd.concat(
        [
            monomer_df.apply(chain_parser.parse, axis=1),
            interface_df.apply(parse_interface_row, axis=1),
        ],
        axis=0,
    )
    logger.info(
        "Final train dataset has %d monomer rows and %d interface rows.",
        len(monomer_df),
        len(interface_df),
    )
    return parsed_df


# --------------------------------------------------------------------------- #
# Val index construction
# --------------------------------------------------------------------------- #
def process_val_metadata_df(metadata_path: str) -> pd.DataFrame:
    metadata_df = pd.read_parquet(metadata_path)
    metadata_df = ensure_example_id_column(metadata_df)
    if "query_pn_unit_iids" in metadata_df.columns:
        metadata_df["query_pn_unit_iids"] = metadata_df["query_pn_unit_iids"].apply(parse_pn_unit_iids_value)
    elif "q_pn_unit_iid" in metadata_df.columns:
        metadata_df["query_pn_unit_iids"] = metadata_df["q_pn_unit_iid"].apply(lambda value: [str(value)])
    else:
        raise KeyError(
            "Val metadata must contain either `query_pn_unit_iids` "
            "or row-level `q_pn_unit_iid`."
        )
    metadata_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
    logger.info("Final val dataset contains %d pdbs", metadata_df["pdb_id"].nunique())
    return metadata_df


def parse_val_df(metadata_df: pd.DataFrame) -> pd.DataFrame:
    parser = GenericDFParser(pn_unit_iid_colnames=[])

    def parse_val_row(row):
        parsed = parser.parse(row)
        parsed["query_pn_unit_iids"] = list(row["query_pn_unit_iids"])
        parsed["extra_info"].pop("query_pn_unit_iids", None)
        return parsed

    return metadata_df.apply(parse_val_row, axis=1)
