"""Interface-row dispatch and BML-center clique grouping for AtomWorks SD.

The legacy profiles keep using the center-specific builders in :mod:`interface`.
The grouped profile instead forms maximal cliques over eligible interface centers,
associates every protein chain contacting at least one center, and adds iterative
context only to the featurizer query (never to the sampling cluster).
"""

from collections import Counter
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from atomworks.ml.example_id import generate_example_id
from omegaconf import DictConfig, OmegaConf

from allatom_design.data.datasets.atomworks_sd.bml_context import (
    BMLPolicy,
    BML_ANNOTATION_COLUMNS,
    BML_CENTER_METAL_COL,
    BML_CENTER_SMALL_MOLECULE_COL,
    CONTEXT_HALIDE_COL,
    CONTEXT_METAL_COL,
    CONTEXT_NUCLEIC_ACID_CHAINS_COL,
    CONTEXT_PEPTIDE_COL,
    CONTEXT_SMALL_MOLECULE_COL,
    ensure_bml_context_annotations,
)
from allatom_design.data.datasets.atomworks_sd.interface import build_interface_df
from allatom_design.data.utils.pn_unit import (
    natural_key,
    normalize_ligand_ccd_key,
    parse_partner_list,
    split_components,
)

logger = logging.getLogger(__name__)

PRECOMPUTED_LIGAND_GROUP_COLUMN = "q_pn_unit_ligand_group_records"
PRECOMPUTED_LIGAND_GROUP_SCHEMA_VERSION = "1"
_ATTR_SCHEMA_VERSION = "allatom_design.ligand_grouping.schema_version"
_ATTR_CONFIG_SHA256 = "allatom_design.ligand_grouping.config_sha256"
_ATTR_VALIDATION_IDS_SHA256 = "allatom_design.ligand_grouping.validation_ids_sha256"

GROUPING_PER_CENTER = "per_center"
GROUPING_MAXIMAL_CENTER_CLIQUE = "maximal_center_clique"
VALID_INTERFACE_GROUPING_SCHEMES = (
    GROUPING_PER_CENTER,
    GROUPING_MAXIMAL_CENTER_CLIQUE,
)

GROUPED_INTERFACE_TYPE = "ligand_group_protein"

MODALITY_PROTEIN = "protein"
MODALITY_SMALL_MOLECULE = "small_molecule"
MODALITY_METAL = "metal"
MODALITY_PEPTIDE = "peptide"
MODALITY_NUC_LIGAND = "nuc_ligand"
MODALITY_NUC_POLYMER = "nuc_polymer"

_GROUPED_ROW_COLUMNS = (
    "example_id",
    "pdb_id",
    "assembly_id",
    "path",
    "query_pn_unit_iids",
    "crop_center_pn_unit_iids",
    "ligand_pn_unit_iids",
    "protein_pn_unit_iids",
    "center_clique_pn_unit_iids",
    "expanded_context_pn_unit_iids",
    "center_clique_member_types",
    "center_clique_cluster_key",
    "interface_cluster_key",
    "protein_cluster_multiset",
    "protein_val_cluster_multiset",
    "ligand_ccd_key",
    "interface_type",
    "query_pn_unit_iids_only",
    "n_prot",
    "n_designable_prot",
    "n_small_molecule",
    "n_metal",
    "n_peptide",
    "n_nuc_ligand",
    "n_nuc_polymer",
    "n_nuc",
)


def ligand_grouping_semantic_payload(cfg: dict | DictConfig) -> dict[str, Any]:
    """Return the config subset that determines precomputed group membership."""

    def plain(value: Any) -> Any:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        return value

    clustering_cfg = (cfg or {}).get("clustering", {}) or {}
    train_filters = (cfg or {}).get("train_filters", {}) or {}
    interface_filters = train_filters.get("interface_filter", {}) or {}
    return {
        "debug": bool((cfg or {}).get("debug", False)),
        "cluster_id_col": (cfg or {}).get("cluster_id_col"),
        "val_exclusion_cluster_id_col": (cfg or {}).get(
            "val_exclusion_cluster_id_col"
        ),
        "query_pn_unit_iids_only": bool(
            (cfg or {}).get("query_pn_unit_iids_only", False)
        ),
        "bml_center": plain((cfg or {}).get("bml_center", {})),
        "bml_context": plain((cfg or {}).get("bml_context", {})),
        "metadata_filter": plain(train_filters.get("metadata_filter", [])),
        "protein_monomer_chain_filter": plain(
            train_filters.get("protein_monomer_chain_filter", [])
        ),
        "interface_prefilter": plain(interface_filters.get("1", [])),
        "interface_grouping_scheme": clustering_cfg.get(
            "interface_grouping_scheme", GROUPING_PER_CENTER
        ),
        "center_clique_distance_cutoff": clustering_cfg.get(
            "center_clique_distance_cutoff"
        ),
        "max_cliques_per_assembly": clustering_cfg.get(
            "max_cliques_per_assembly", 100_000
        ),
    }


def ligand_grouping_config_sha256(cfg: dict | DictConfig) -> str:
    payload = json.dumps(
        ligand_grouping_semantic_payload(cfg),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def ligand_grouping_validation_ids_sha256(cfg: dict | DictConfig) -> str:
    validation_ids_file = (cfg or {}).get("validation_ids_file")
    if validation_ids_file in (None, ""):
        return ""
    path = Path(str(validation_ids_file))
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ligand_grouping_artifact_attrs(cfg: dict | DictConfig) -> dict[str, str]:
    return {
        _ATTR_SCHEMA_VERSION: PRECOMPUTED_LIGAND_GROUP_SCHEMA_VERSION,
        _ATTR_CONFIG_SHA256: ligand_grouping_config_sha256(cfg),
        _ATTR_VALIDATION_IDS_SHA256: ligand_grouping_validation_ids_sha256(cfg),
    }


def validate_precomputed_ligand_grouping_metadata(
    metadata_df: pd.DataFrame,
    cfg: dict | DictConfig,
) -> None:
    """Fail fast when a present precomputed column is stale or malformed."""

    if PRECOMPUTED_LIGAND_GROUP_COLUMN not in metadata_df.columns:
        raise KeyError(
            f"Precomputed ligand grouping requires column "
            f"{PRECOMPUTED_LIGAND_GROUP_COLUMN!r}."
        )
    missing_annotations = [
        column for column in BML_ANNOTATION_COLUMNS if column not in metadata_df.columns
    ]
    if missing_annotations:
        raise KeyError(
            "Precomputed ligand grouping is missing BML annotation columns: "
            f"{missing_annotations}"
        )

    expected = ligand_grouping_artifact_attrs(cfg)
    mismatches = {
        key: {"expected": value, "actual": metadata_df.attrs.get(key)}
        for key, value in expected.items()
        if metadata_df.attrs.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Precomputed ligand-grouping artifact metadata does not match the "
            f"active data config: {mismatches}"
        )


def build_interface_df_for_clustering(
    metadata_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    dataset_name: str,
    cfg: dict | DictConfig,
) -> pd.DataFrame:
    """Dispatch interface construction without changing legacy builders."""

    clustering_cfg = (cfg or {}).get("clustering", {}) or {}
    grouping_scheme = str(
        clustering_cfg.get("interface_grouping_scheme", GROUPING_PER_CENTER)
    )
    if grouping_scheme not in VALID_INTERFACE_GROUPING_SCHEMES:
        raise ValueError(
            f"Unknown `clustering.interface_grouping_scheme`: {grouping_scheme!r}. "
            f"Expected one of {VALID_INTERFACE_GROUPING_SCHEMES}."
        )
    if grouping_scheme == GROUPING_PER_CENTER:
        return build_interface_df(metadata_df, protein_df, dataset_name, cfg)

    policy = BMLPolicy.from_cfg(cfg)
    if not policy.context.enabled:
        raise ValueError(
            "`maximal_center_clique` grouping requires `bml_context.enabled=true`."
        )
    if policy.context.scheme != "iterative_context_refinement":
        raise ValueError(
            "`maximal_center_clique` grouping requires "
            "`bml_context.scheme=iterative_context_refinement`."
        )
    if PRECOMPUTED_LIGAND_GROUP_COLUMN in metadata_df.columns:
        return build_precomputed_center_clique_interface_df(
            metadata_df=metadata_df,
            protein_df=protein_df,
            dataset_name=dataset_name,
            cfg=cfg,
        )
    return build_center_clique_interface_df(
        metadata_df=metadata_df,
        protein_df=protein_df,
        dataset_name=dataset_name,
        cfg=cfg,
    )


def build_center_clique_interface_df(
    metadata_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    dataset_name: str,
    cfg: dict | DictConfig,
) -> pd.DataFrame:
    """Build mixed-modality interface rows from maximal BML-center cliques."""

    interface_df, _ = build_center_clique_interface_df_with_stats(
        metadata_df=metadata_df,
        protein_df=protein_df,
        dataset_name=dataset_name,
        cfg=cfg,
    )
    return interface_df


def build_center_clique_interface_df_with_stats(
    metadata_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    dataset_name: str,
    cfg: dict | DictConfig,
) -> tuple[pd.DataFrame, Counter]:
    """Build grouped interface rows and return deterministic build diagnostics."""

    policy = BMLPolicy.from_cfg(cfg)
    metadata_df = ensure_bml_context_annotations(metadata_df, cfg)
    clustering_cfg = (cfg or {}).get("clustering", {}) or {}
    distance_cutoff = float(
        clustering_cfg.get(
            "center_clique_distance_cutoff",
            policy.context.contact_distance_cutoff,
        )
    )
    if not math.isclose(
        distance_cutoff,
        policy.context.contact_distance_cutoff,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Grouped clustering uses one contact cutoff for center cliques, proteins, "
            "and direct iterative context; "
            "`clustering.center_clique_distance_cutoff` must equal "
            "`bml_context.contact_distance_cutoff`."
        )
    max_cliques = int(clustering_cfg.get("max_cliques_per_assembly", 100_000))
    if distance_cutoff <= 0:
        raise ValueError("`clustering.center_clique_distance_cutoff` must be positive.")
    if max_cliques <= 0:
        raise ValueError("`clustering.max_cliques_per_assembly` must be positive.")

    center_df = _select_center_candidates(metadata_df)
    designable_protein_df = protein_df[
        protein_df["q_pn_unit_is_protein"].fillna(False).astype(bool)
    ].copy()
    all_protein_df = metadata_df[
        metadata_df["q_pn_unit_is_protein"].fillna(False).astype(bool)
    ].copy()
    context_df = metadata_df[_expandable_nonprotein_context_mask(metadata_df)].copy()

    rows: list[dict[str, Any]] = []
    stats = Counter()
    protein_groups = _group_rows_by_assembly(all_protein_df)
    designable_protein_groups = _group_rows_by_assembly(designable_protein_df)
    context_groups = _group_rows_by_assembly(context_df)

    for assembly_key, assembly_centers in _group_rows_by_assembly(center_df).items():
        protein_rows = protein_groups.get(assembly_key, {})
        designable_protein_iids = set(
            designable_protein_groups.get(assembly_key, {})
        )
        context_rows = context_groups.get(assembly_key, {})
        cliques, graph_stats = enumerate_maximal_center_cliques(
            assembly_centers,
            distance_cutoff=distance_cutoff,
            max_cliques=max_cliques,
        )
        assembly_max_clique_size = graph_stats.pop("max_center_clique_size", 0)
        stats.update(graph_stats)
        stats["max_center_clique_size"] = max(
            stats["max_center_clique_size"],
            assembly_max_clique_size,
        )
        stats["assemblies_with_centers"] += 1
        stats["center_cliques"] += len(cliques)

        for clique_iids in cliques:
            clique_rows = [assembly_centers[iid] for iid in clique_iids]
            protein_iids = _collect_contacted_iids(
                clique_rows,
                protein_rows,
                distance_cutoff=distance_cutoff,
                stats=stats,
                stat_prefix="protein",
            )
            if not protein_iids:
                stats["cliques_without_protein"] += 1
                continue

            context_iids = _collect_contacted_iids(
                clique_rows,
                context_rows,
                distance_cutoff=distance_cutoff,
                stats=stats,
                stat_prefix="context",
            )
            clique_set = set(clique_iids)
            expanded_context_iids = tuple(
                iid for iid in context_iids if iid not in clique_set
            )
            rows.append(
                _build_grouped_interface_row(
                    clique_rows=clique_rows,
                    clique_iids=clique_iids,
                    protein_rows=[protein_rows[iid] for iid in protein_iids],
                    protein_iids=protein_iids,
                    designable_protein_iids=tuple(
                        iid for iid in protein_iids if iid in designable_protein_iids
                    ),
                    expanded_context_iids=expanded_context_iids,
                    dataset_name=dataset_name,
                )
            )

    stats["grouped_interface_rows"] = len(rows)
    logger.info(
        "BML center-clique grouping stats: %s",
        dict(sorted(stats.items())),
    )
    return _rows_to_grouped_interface_df(rows, metadata_df), stats


def ligand_group_records_by_anchor(
    interface_df: pd.DataFrame,
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Serialize grouped-interface identity into one record list per anchor center."""

    records_by_anchor: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    seen_group_ids: set[str] = set()
    for row in interface_df.itertuples(index=False):
        clique_iids = tuple(str(value) for value in row.center_clique_pn_unit_iids)
        member_types = tuple(str(value) for value in row.center_clique_member_types)
        if not clique_iids:
            raise ValueError("Grouped interface row has an empty center clique.")
        if len(clique_iids) != len(member_types):
            raise ValueError(
                "Grouped interface row has mismatched clique/member-type lengths: "
                f"example_id={row.example_id!r}."
            )

        group_id = str(row.example_id)
        if group_id in seen_group_ids:
            raise ValueError(f"Duplicate ligand group ID: {group_id!r}.")
        seen_group_ids.add(group_id)

        anchor = (str(row.pdb_id), str(row.assembly_id), clique_iids[0])
        records_by_anchor.setdefault(anchor, []).append(
            {
                "ligand_group_id": group_id,
                "center_clique_pn_unit_iids": list(clique_iids),
                "center_clique_member_types": list(member_types),
                "protein_pn_unit_iids": [
                    str(value) for value in row.protein_pn_unit_iids
                ],
                "expanded_context_pn_unit_iids": [
                    str(value) for value in row.expanded_context_pn_unit_iids
                ],
            }
        )

    for records in records_by_anchor.values():
        records.sort(key=lambda record: record["ligand_group_id"])
    return records_by_anchor


def build_precomputed_center_clique_interface_df(
    metadata_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    dataset_name: str,
    cfg: dict | DictConfig,
) -> pd.DataFrame:
    """Materialize grouped interface rows from validated metadata records."""

    validate_precomputed_ligand_grouping_metadata(metadata_df, cfg)
    center_groups = _group_rows_by_assembly(_select_center_candidates(metadata_df))
    protein_mask = metadata_df["q_pn_unit_is_protein"].fillna(False).astype(bool)
    protein_columns = [
        "pdb_id",
        "assembly_id",
        "q_pn_unit_iid",
        "q_pn_unit_cluster_id",
        "q_pn_unit_cluster_val_id",
    ]
    all_protein_groups = _group_rows_by_assembly(
        metadata_df.loc[protein_mask, protein_columns]
    )
    designable_protein_groups = _group_iids_by_assembly(
        protein_df[protein_df["q_pn_unit_is_protein"].fillna(False).astype(bool)]
    )
    available_iids_by_assembly = _group_iids_by_assembly(metadata_df)
    serialized_record_count = sum(
        len(_normalize_ligand_group_records(raw_records))
        for raw_records in metadata_df[PRECOMPUTED_LIGAND_GROUP_COLUMN].array
    )

    rows: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for assembly_key, assembly_centers in center_groups.items():
        protein_rows = all_protein_groups.get(assembly_key, {})
        designable_protein_iids = set(
            designable_protein_groups.get(assembly_key, set())
        )
        available_iids = available_iids_by_assembly.get(assembly_key, set())

        for anchor_iid, anchor in assembly_centers.items():
            raw_records = getattr(anchor, PRECOMPUTED_LIGAND_GROUP_COLUMN, None)
            for record in _normalize_ligand_group_records(raw_records):
                group_id = _required_record_string(record, "ligand_group_id")
                if group_id in seen_group_ids:
                    raise ValueError(
                        f"Duplicate precomputed ligand group ID: {group_id!r}."
                    )

                clique_iids = _required_record_strings(
                    record, "center_clique_pn_unit_iids"
                )
                member_types = _required_record_strings(
                    record, "center_clique_member_types"
                )
                protein_iids = _required_record_strings(
                    record, "protein_pn_unit_iids"
                )
                expanded_context_iids = _required_record_strings(
                    record, "expanded_context_pn_unit_iids"
                )
                if not clique_iids or clique_iids[0] != anchor_iid:
                    raise ValueError(
                        "Precomputed clique must be stored on its first canonical "
                        f"center: group={group_id!r}, anchor={anchor_iid!r}, "
                        f"clique={clique_iids}."
                    )
                if len(clique_iids) != len(member_types):
                    raise ValueError(
                        "Precomputed clique/member-type lengths differ for "
                        f"group {group_id!r}."
                    )

                missing_centers = [
                    iid for iid in clique_iids if iid not in assembly_centers
                ]
                missing_proteins = [
                    iid for iid in protein_iids if iid not in protein_rows
                ]
                missing_context = [
                    iid for iid in expanded_context_iids if iid not in available_iids
                ]
                if missing_centers or missing_proteins or missing_context:
                    raise KeyError(
                        "Precomputed ligand group references missing PN units: "
                        f"group={group_id!r}, centers={missing_centers}, "
                        f"proteins={missing_proteins}, context={missing_context}."
                    )

                clique_rows = [assembly_centers[iid] for iid in clique_iids]
                actual_member_types = tuple(
                    str(row.center_modality) for row in clique_rows
                )
                if actual_member_types != member_types:
                    raise ValueError(
                        "Precomputed member modalities do not match active metadata: "
                        f"group={group_id!r}, stored={member_types}, "
                        f"actual={actual_member_types}."
                    )
                for row, modality in zip(clique_rows, actual_member_types):
                    if modality == MODALITY_SMALL_MOLECULE and bool(
                        getattr(
                            row,
                            "q_pn_unit_is_maybe_covalently_linked_to_protein",
                            False,
                        )
                    ):
                        raise ValueError(
                            "Maybe-covalently-linked small molecule appears in a "
                            f"precomputed center clique: group={group_id!r}, "
                            f"iid={row.q_pn_unit_iid!r}."
                        )

                built = _build_grouped_interface_row(
                    clique_rows=clique_rows,
                    clique_iids=clique_iids,
                    protein_rows=[protein_rows[iid] for iid in protein_iids],
                    protein_iids=protein_iids,
                    designable_protein_iids=tuple(
                        iid for iid in protein_iids if iid in designable_protein_iids
                    ),
                    expanded_context_iids=expanded_context_iids,
                    dataset_name=dataset_name,
                )
                if built["example_id"] != group_id:
                    raise ValueError(
                        "Precomputed ligand group ID does not match the production "
                        f"example-ID contract: stored={group_id!r}, "
                        f"rebuilt={built['example_id']!r}."
                    )
                rows.append(built)
                seen_group_ids.add(group_id)

    if len(seen_group_ids) != serialized_record_count:
        raise ValueError(
            "Not every serialized ligand-group record was consumed from an eligible "
            "canonical center row: "
            f"stored={serialized_record_count}, consumed={len(seen_group_ids)}."
        )
    logger.info(
        "Loaded %d precomputed ligand-group interface rows from metadata column %s.",
        len(rows),
        PRECOMPUTED_LIGAND_GROUP_COLUMN,
    )
    return _rows_to_grouped_interface_df(rows, metadata_df)


def _normalize_ligand_group_records(value: Any) -> list[dict[str, Any]]:
    if value is None or (
        not isinstance(value, (list, tuple, np.ndarray)) and pd.isna(value)
    ):
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "Precomputed ligand-group records must be a list of structs, got "
            f"{type(value).__name__}."
        )
    records = list(value)
    if not all(isinstance(record, dict) for record in records):
        raise TypeError("Every precomputed ligand-group record must be a struct/dict.")
    return records


def _required_record_string(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if value is None or str(value) == "":
        raise ValueError(f"Precomputed ligand-group record is missing {key!r}.")
    return str(value)


def _required_record_strings(
    record: dict[str, Any],
    key: str,
) -> tuple[str, ...]:
    value = record.get(key)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"Precomputed ligand-group field {key!r} must be a list of strings."
        )
    return tuple(str(item) for item in value)


def enumerate_maximal_center_cliques(
    center_rows: dict[str, Any],
    *,
    distance_cutoff: float,
    max_cliques: int,
) -> tuple[list[tuple[str, ...]], Counter]:
    """Return canonical maximal cliques for one assembly's center rows."""

    graph = nx.Graph()
    graph.add_nodes_from(center_rows)
    stats = Counter()

    available_iids = set(center_rows)
    directed_edges: set[tuple[str, str]] = set()
    for source_iid, source in center_rows.items():
        for contact in parse_partner_list(
            getattr(source, "q_pn_unit_contacting_pn_unit_iids", None)
        ):
            distance = _strict_contact_distance(contact)
            if distance is None:
                stats["center_contacts_missing_or_invalid_distance"] += 1
                continue
            if distance > distance_cutoff:
                continue
            partner_iids = _resolve_contact_iids(contact, available_iids)
            if not partner_iids:
                # The source contact list contains proteins and context as well
                # as other centers.  Those valid out-of-graph contacts are
                # expected here and are not resolution failures.
                stats["center_contacts_outside_center_graph"] += 1
                continue
            for partner_iid in partner_iids:
                if partner_iid == source_iid:
                    continue
                directed_edges.add((source_iid, partner_iid))
                graph.add_edge(source_iid, partner_iid)

    undirected_edges = {
        tuple(sorted((source, partner), key=natural_key))
        for source, partner in directed_edges
    }
    stats["center_nodes"] = graph.number_of_nodes()
    stats["center_edges"] = graph.number_of_edges()
    stats["one_sided_center_edges"] = sum(
        1
        for left, right in undirected_edges
        if (left, right) not in directed_edges or (right, left) not in directed_edges
    )

    canonical_cliques: set[tuple[str, ...]] = set()
    for clique in nx.find_cliques(graph):
        canonical = tuple(sorted((str(iid) for iid in clique), key=natural_key))
        canonical_cliques.add(canonical)
        if len(canonical_cliques) > max_cliques:
            raise RuntimeError(
                "Maximal center-clique enumeration exceeded "
                f"max_cliques_per_assembly={max_cliques}."
            )
    cliques = sorted(
        canonical_cliques,
        key=lambda clique: tuple(natural_key(iid) for iid in clique),
    )
    stats["max_center_clique_size"] = max((len(clique) for clique in cliques), default=0)
    return cliques, stats


def _select_center_candidates(metadata_df: pd.DataFrame) -> pd.DataFrame:
    modality_masks = {
        MODALITY_SMALL_MOLECULE: _bool_col(metadata_df, BML_CENTER_SMALL_MOLECULE_COL),
        MODALITY_METAL: _bool_col(metadata_df, BML_CENTER_METAL_COL),
        MODALITY_PEPTIDE: _bool_col(metadata_df, "q_pn_unit_is_peptide"),
        MODALITY_NUC_LIGAND: _bool_col(metadata_df, "q_pn_unit_is_nuc_ligand"),
        MODALITY_NUC_POLYMER: _bool_col(metadata_df, "q_pn_unit_is_nuc_polymer"),
    }
    membership_count = sum(mask.astype(int) for mask in modality_masks.values())
    ambiguous = membership_count > 1
    if ambiguous.any():
        ambiguous_iids = metadata_df.loc[ambiguous, "q_pn_unit_iid"].astype(str).tolist()
        raise ValueError(
            "Center candidates must map to exactly one modality; ambiguous PN units: "
            f"{ambiguous_iids[:20]}"
        )

    out = metadata_df[membership_count == 1].copy()
    out["center_modality"] = pd.NA
    for modality, mask in modality_masks.items():
        out.loc[mask.reindex(out.index, fill_value=False), "center_modality"] = modality
    return out


def _expandable_nonprotein_context_mask(metadata_df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=metadata_df.index)
    for column in (
        CONTEXT_SMALL_MOLECULE_COL,
        CONTEXT_METAL_COL,
        CONTEXT_HALIDE_COL,
        CONTEXT_PEPTIDE_COL,
        CONTEXT_NUCLEIC_ACID_CHAINS_COL,
    ):
        mask |= _bool_col(metadata_df, column)
    return mask & ~_bool_col(metadata_df, "q_pn_unit_is_protein")


def _group_rows_by_assembly(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    if df.empty:
        return grouped
    for (pdb_id, assembly_id), group in df.groupby(["pdb_id", "assembly_id"], sort=False):
        rows: dict[str, Any] = {}
        for row in group.itertuples(index=False):
            iid = str(row.q_pn_unit_iid)
            if iid in rows:
                raise ValueError(
                    "PN-unit IID must be unique within an assembly: "
                    f"pdb_id={pdb_id!r}, assembly_id={assembly_id!r}, iid={iid!r}."
                )
            rows[iid] = row
        grouped[(str(pdb_id), str(assembly_id))] = rows
    return grouped


def _group_iids_by_assembly(df: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    grouped: dict[tuple[str, str], set[str]] = {}
    if df.empty:
        return grouped
    columns = ["pdb_id", "assembly_id", "q_pn_unit_iid"]
    for (pdb_id, assembly_id), group in df[columns].groupby(
        ["pdb_id", "assembly_id"], sort=False
    ):
        iids = group["q_pn_unit_iid"].astype(str).tolist()
        iid_set = set(iids)
        if len(iids) != len(iid_set):
            raise ValueError(
                "PN-unit IID must be unique within an assembly: "
                f"pdb_id={pdb_id!r}, assembly_id={assembly_id!r}."
            )
        grouped[(str(pdb_id), str(assembly_id))] = iid_set
    return grouped


def _collect_contacted_iids(
    source_rows: list[Any],
    target_rows: dict[str, Any],
    *,
    distance_cutoff: float,
    stats: Counter,
    stat_prefix: str,
) -> tuple[str, ...]:
    if not target_rows:
        return ()
    target_iids = set(target_rows)
    source_iids = {str(source.q_pn_unit_iid) for source in source_rows}
    contacted: set[str] = set()
    for source in source_rows:
        for contact in parse_partner_list(
            getattr(source, "q_pn_unit_contacting_pn_unit_iids", None)
        ):
            distance = _strict_contact_distance(contact)
            if distance is None:
                stats[f"{stat_prefix}_contacts_missing_or_invalid_distance"] += 1
                continue
            if distance > distance_cutoff:
                continue
            resolved = _resolve_contact_iids(contact, target_iids)
            if not resolved:
                # A center's contact list spans every modality.  Failure to
                # resolve inside this particular target subset normally means
                # the contact belongs to another valid subset, not that its IID
                # is malformed.
                stats[f"source_contacts_outside_{stat_prefix}_targets"] += 1
                continue
            contacted.update(resolved)

    # Contact distance is physically symmetric, but metadata occasionally
    # contains only one directed record. Accept a valid reverse record too.
    for target_iid, target in target_rows.items():
        for contact in parse_partner_list(
            getattr(target, "q_pn_unit_contacting_pn_unit_iids", None)
        ):
            distance = _strict_contact_distance(contact)
            if distance is None or distance > distance_cutoff:
                continue
            if _resolve_contact_iids(contact, source_iids):
                contacted.add(target_iid)
                break
    return tuple(sorted(contacted, key=natural_key))


def _resolve_contact_iids(contact: dict, available_iids: set[str]) -> tuple[str, ...]:
    candidates = [
        str(value)
        for value in (contact.get("pn_unit_iid"), contact.get("chain_iid"))
        if value is not None
    ]
    for candidate in candidates:
        if candidate in available_iids:
            return (candidate,)

    resolved: set[str] = set()
    for candidate in candidates:
        resolved.update(
            component
            for component in split_components(candidate)
            if component in available_iids
        )
    return tuple(sorted(resolved, key=natural_key))


def _strict_contact_distance(contact: Any) -> float | None:
    if not isinstance(contact, dict):
        return None
    try:
        distance = float(contact.get("min_distance"))
    except (TypeError, ValueError):
        return None
    return distance if math.isfinite(distance) and distance >= 0.0 else None


def _build_grouped_interface_row(
    *,
    clique_rows: list[Any],
    clique_iids: tuple[str, ...],
    protein_rows: list[Any],
    protein_iids: tuple[str, ...],
    designable_protein_iids: tuple[str, ...],
    expanded_context_iids: tuple[str, ...],
    dataset_name: str,
) -> dict[str, Any]:
    center = clique_rows[0]
    modalities = tuple(str(row.center_modality) for row in clique_rows)
    modality_counts = Counter(modalities)
    center_cluster_key = tuple(
        sorted(
            (_center_cluster_key(row, modality) for row, modality in zip(clique_rows, modalities)),
            key=repr,
        )
    )
    protein_clusters = tuple(row.q_pn_unit_cluster_id for row in protein_rows)
    protein_val_clusters = tuple(row.q_pn_unit_cluster_val_id for row in protein_rows)
    canonical_protein_clusters = tuple(sorted(protein_clusters, key=repr))
    interface_cluster_key = (
        center_cluster_key,
        tuple((MODALITY_PROTEIN, cluster) for cluster in canonical_protein_clusters),
    )

    query_iids = tuple(
        dict.fromkeys([*clique_iids, *protein_iids, *expanded_context_iids])
    )
    crop_center_iids = tuple(dict.fromkeys([*clique_iids, *protein_iids]))
    row = center._asdict()
    row.pop("crop_center_pn_unit_iids", None)
    row.pop(PRECOMPUTED_LIGAND_GROUP_COLUMN, None)
    row.update(
        {
            "query_pn_unit_iids": query_iids,
            "crop_center_pn_unit_iids": crop_center_iids,
            "ligand_pn_unit_iids": clique_iids,
            "protein_pn_unit_iids": protein_iids,
            "center_clique_pn_unit_iids": clique_iids,
            "expanded_context_pn_unit_iids": expanded_context_iids,
            "center_clique_member_types": modalities,
            "center_clique_cluster_key": center_cluster_key,
            "interface_cluster_key": interface_cluster_key,
            "protein_cluster_multiset": protein_clusters,
            "protein_val_cluster_multiset": protein_val_clusters,
            "ligand_ccd_key": center_cluster_key,
            "interface_type": GROUPED_INTERFACE_TYPE,
            "query_pn_unit_iids_only": True,
            "n_prot": len(protein_iids),
            "n_designable_prot": len(designable_protein_iids),
            "n_small_molecule": modality_counts[MODALITY_SMALL_MOLECULE],
            "n_metal": modality_counts[MODALITY_METAL],
            "n_peptide": modality_counts[MODALITY_PEPTIDE],
            "n_nuc_ligand": modality_counts[MODALITY_NUC_LIGAND],
            "n_nuc_polymer": modality_counts[MODALITY_NUC_POLYMER],
            "n_nuc": (
                modality_counts[MODALITY_NUC_LIGAND]
                + modality_counts[MODALITY_NUC_POLYMER]
            ),
        }
    )
    row["example_id"] = generate_example_id(
        [dataset_name, "interface"],
        row["pdb_id"],
        row["assembly_id"],
        list(crop_center_iids),
    )
    return row


def _center_cluster_key(row: Any, modality: str) -> tuple[str, Any]:
    if modality in {MODALITY_SMALL_MOLECULE, MODALITY_METAL}:
        return modality, normalize_ligand_ccd_key(row.q_pn_unit_non_polymer_res_names)
    cluster_id = row.q_pn_unit_cluster_id
    if pd.isna(cluster_id):
        raise ValueError(
            f"Center PN unit {row.q_pn_unit_iid!r} ({modality}) has no sampling cluster ID."
        )
    return modality, cluster_id


def _rows_to_grouped_interface_df(
    rows: list[dict[str, Any]],
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = list(
        dict.fromkeys(
            [
                *_GROUPED_ROW_COLUMNS,
                *[
                    column
                    for column in metadata_df.columns.tolist()
                    if column != "crop_center_pn_unit_iids"
                ],
            ]
        )
    )
    if not rows:
        out = pd.DataFrame(columns=output_columns)
    else:
        out = pd.DataFrame(rows).reindex(columns=output_columns)
    out.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
    return out


def _bool_col(metadata_df: pd.DataFrame, column: str) -> pd.Series:
    return metadata_df.get(
        column,
        pd.Series(False, index=metadata_df.index),
    ).fillna(False).astype(bool)
