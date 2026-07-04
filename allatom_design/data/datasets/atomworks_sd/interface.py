"""Interface DataFrame construction for the SD dataset.

Builds the metal-protein, small-molecule-protein and protein-protein interface
rows that, together with protein monomer chains, make up the train index.
"""

from collections.abc import Callable
import logging

import pandas as pd
from atomworks.ml.example_id import generate_example_id
from omegaconf import DictConfig

from allatom_design.data.utils.pn_unit import (
    contact_count,
    contact_within_cutoff,
    missing_atom_fraction_from_values,
    normalize_ligand_ccd_key,
    parse_partner_list,
    split_components,
)

from allatom_design.data.datasets.atomworks_sd.bml_context import (
    BMLContextExpansionStats,
    BMLPolicy,
    BML_CENTER_METAL_COL,
    BML_CENTER_SMALL_MOLECULE_COL,
    build_pn_unit_index_lookup,
    context_expansion_source_rows,
    ensure_bml_context_annotations,
    expand_bml_context_query_iids,
)
from allatom_design.data.datasets.atomworks_sd.selectors import (
    nucleic_acid_ligand_center_mask,
    peptide_center_mask,
)

logger = logging.getLogger(__name__)


def build_interface_df(
    metadata_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    dataset_name: str,
    cfg: dict | DictConfig,
) -> pd.DataFrame:

    cfg = cfg or {}
    policy = BMLPolicy.from_cfg(cfg)
    metadata_df = ensure_bml_context_annotations(metadata_df, cfg)
    context_lookup = build_pn_unit_index_lookup(metadata_df)
    context_stats = BMLContextExpansionStats()

    protein_interface_distance_cutoff = float(cfg.get("protein_interface_distance_cutoff", 5.0))
    ligand_contact_distance_cutoff = float(cfg.get("ligand_contact_distance_cutoff", 5.0))

    protein_df = protein_df[protein_df["q_pn_unit_is_protein"].fillna(False).astype(bool)].copy()
    protein_lookup = {
        (row.pdb_id, str(row.assembly_id), row.q_pn_unit_iid): row
        for row in protein_df.itertuples(index=False)
    }

    metal_rows, _ = _build_protein_metal_interface_rows(
        metadata_df=metadata_df,
        protein_lookup=protein_lookup,
        dataset_name=dataset_name,
        policy=policy,
        context_lookup=context_lookup,
        context_stats=context_stats,
    )
    small_molecule_rows = _build_small_molecule_interface_rows(
        metadata_df=metadata_df,
        protein_lookup=protein_lookup,
        dataset_name=dataset_name,
        policy=policy,
        context_lookup=context_lookup,
        context_stats=context_stats,
    )
    peptide_rows = _build_peptide_interface_rows(
        metadata_df=metadata_df,
        protein_lookup=protein_lookup,
        dataset_name=dataset_name,
        distance_cutoff=ligand_contact_distance_cutoff,
        cfg=cfg,
        policy=policy,
        context_lookup=context_lookup,
        context_stats=context_stats,
    )
    nuc_ligand_rows = _build_nucleic_acid_ligand_interface_rows(
        metadata_df=metadata_df,
        protein_lookup=protein_lookup,
        dataset_name=dataset_name,
        distance_cutoff=ligand_contact_distance_cutoff,
        cfg=cfg,
        policy=policy,
        context_lookup=context_lookup,
        context_stats=context_stats,
    )
    protein_rows = _build_protein_interface_rows(
        protein_df=protein_df,
        protein_lookup=protein_lookup,
        dataset_name=dataset_name,
        distance_cutoff=protein_interface_distance_cutoff,
    )
    rows = [
        *metal_rows,
        *small_molecule_rows,
        *peptide_rows,
        *nuc_ligand_rows,
        *protein_rows,
    ]

    output_cols = [
        "example_id",
        "pdb_id",
        "assembly_id",
        "path",
        "query_pn_unit_iids",
        "crop_center_pn_unit_iids",
        "ligand_pn_unit_iids",
        "protein_pn_unit_iids",
        "protein_cluster_multiset",
        "protein_val_cluster_multiset",
        "ligand_ccd_key",
        "interface_type",
        "query_pn_unit_iids_only",
        "n_coordinating_protein_donor_atoms",
        "n_contacting_protein_atoms",
        "n_protein_protein_contacts",
        "q_pn_unit_missing_atom_fraction_nonpolymer",
        *[col for col in metadata_df.columns.tolist() if col != "crop_center_pn_unit_iids"],
    ]
    output_cols = list(dict.fromkeys(output_cols))
    interface_df = pd.DataFrame(rows)
    if interface_df.empty:
        interface_df = pd.DataFrame(columns=output_cols)
    else:
        interface_df = interface_df.reindex(columns=output_cols)
    interface_df.set_index("example_id", inplace=True, drop=False, verify_integrity=True)
    logger.info(
        "Built interface_df with %d rows (%d metal-protein, %d small-molecule-protein, "
        "%d peptide-protein, %d nucleic-acid-ligand-protein, %d protein-protein).",
        len(interface_df),
        len(metal_rows),
        len(small_molecule_rows),
        len(peptide_rows),
        len(nuc_ligand_rows),
        len(protein_rows),
    )
    if context_stats.missing_partner_rows or context_stats.skipped_ineligible_partners:
        logger.info(
            "BML context expansion skipped %d missing partner rows and %d "
            "ineligible non-protein partners.",
            context_stats.missing_partner_rows,
            context_stats.skipped_ineligible_partners,
        )
    return interface_df


def _build_protein_metal_interface_rows(
    metadata_df: pd.DataFrame,
    protein_lookup: dict,
    dataset_name: str,
    policy: BMLPolicy,
    context_lookup: dict,
    context_stats: BMLContextExpansionStats,
) -> tuple[list[dict], int]:
    min_donors = int(policy.center.metal.min_protein_donor_atoms)
    metal_center_df = metadata_df[metadata_df[BML_CENTER_METAL_COL].fillna(False).astype(bool)].copy()

    rows = []
    for center in metal_center_df.itertuples(index=False):
        donor_count, protein_rows = _collect_metal_protein_donor_partners(
            center,
            protein_lookup,
            count_fn=policy.metal_donor.count,
        )
        if donor_count < min_donors or len(protein_rows) == 0:
            continue

        protein_rows = sorted(protein_rows, key=lambda row: row.q_pn_unit_iid)
        protein_iids = tuple(row.q_pn_unit_iid for row in protein_rows)
        protein_clusters = tuple(row.q_pn_unit_cluster_id for row in protein_rows)
        crop_center_iids = [center.q_pn_unit_iid, *protein_iids]
        query_iids = expand_bml_context_query_iids(
            source_rows=context_expansion_source_rows(policy, [center], protein_rows),
            crop_center_pn_unit_iids=crop_center_iids,
            metadata_df=metadata_df,
            lookup=context_lookup,
            policy=policy,
            stats=context_stats,
        )

        row = center._asdict()
        row.pop("crop_center_pn_unit_iids", None)
        row.update(
            {
                "query_pn_unit_iids": query_iids,
                "crop_center_pn_unit_iids": tuple(crop_center_iids),
                "ligand_pn_unit_iids": (center.q_pn_unit_iid,),
                "protein_pn_unit_iids": protein_iids,
                "protein_cluster_multiset": protein_clusters,
                "protein_val_cluster_multiset": tuple(r.q_pn_unit_cluster_val_id for r in protein_rows),
                "ligand_ccd_key": normalize_ligand_ccd_key(center.q_pn_unit_non_polymer_res_names),
                "interface_type": "bmm_protein",
                "n_coordinating_protein_donor_atoms": donor_count,
            }
        )
        row["example_id"] = generate_example_id(
            [dataset_name, "interface"],
            row["pdb_id"],
            row["assembly_id"],
            list(row["crop_center_pn_unit_iids"]),
        )
        rows.append(row)

    return rows, len(metal_center_df)


def _build_small_molecule_interface_rows(
    metadata_df: pd.DataFrame,
    protein_lookup: dict,
    dataset_name: str,
    policy: BMLPolicy,
    context_lookup: dict,
    context_stats: BMLContextExpansionStats,
) -> list[dict]:
    center_mask = metadata_df[BML_CENTER_SMALL_MOLECULE_COL].fillna(False).astype(bool)
    center_df = metadata_df[center_mask].copy()
    min_contacts = policy.center.small_molecule.min_contacting_protein_atoms

    rows = []
    for center in center_df.itertuples(index=False):
        contact_count, protein_rows = _collect_small_molecule_protein_partners(center, protein_lookup)
        if (
            contact_count == 0
            or len(protein_rows) == 0
            or (min_contacts is not None and contact_count < int(min_contacts))
        ):
            continue

        protein_rows = sorted(protein_rows, key=lambda row: row.q_pn_unit_iid)
        protein_iids = tuple(row.q_pn_unit_iid for row in protein_rows)
        protein_clusters = tuple(row.q_pn_unit_cluster_id for row in protein_rows)
        crop_center_iids = [center.q_pn_unit_iid, *protein_iids]
        query_iids = expand_bml_context_query_iids(
            source_rows=context_expansion_source_rows(policy, [center], protein_rows),
            crop_center_pn_unit_iids=crop_center_iids,
            metadata_df=metadata_df,
            lookup=context_lookup,
            policy=policy,
            stats=context_stats,
        )

        row = center._asdict()
        row.pop("crop_center_pn_unit_iids", None)
        row.update(
            {
                "query_pn_unit_iids": query_iids,
                "crop_center_pn_unit_iids": tuple(crop_center_iids),
                "ligand_pn_unit_iids": (center.q_pn_unit_iid,),
                "protein_pn_unit_iids": protein_iids,
                "protein_cluster_multiset": protein_clusters,
                "protein_val_cluster_multiset": tuple(r.q_pn_unit_cluster_val_id for r in protein_rows),
                "ligand_ccd_key": normalize_ligand_ccd_key(center.q_pn_unit_non_polymer_res_names),
                "interface_type": "bmsm_protein",
                "n_contacting_protein_atoms": contact_count,
                "q_pn_unit_missing_atom_fraction_nonpolymer": missing_atom_fraction_from_values(
                    getattr(center, "q_pn_unit_num_resolved_atoms"),
                    getattr(center, "q_pn_unit_expected_heavy_atoms_non_polymer"),
                ),
            }
        )
        row["example_id"] = generate_example_id(
            [dataset_name, "interface"],
            row["pdb_id"],
            row["assembly_id"],
            list(row["crop_center_pn_unit_iids"]),
        )
        rows.append(row)

    return rows


def _build_peptide_interface_rows(
    metadata_df: pd.DataFrame,
    protein_lookup: dict,
    dataset_name: str,
    distance_cutoff: float,
    cfg: dict | DictConfig,
    policy: BMLPolicy,
    context_lookup: dict,
    context_stats: BMLContextExpansionStats,
) -> list[dict]:
    center_df = metadata_df[peptide_center_mask(metadata_df, cfg)].copy()

    rows = []
    for center in center_df.itertuples(index=False):
        contact_count, protein_rows = _collect_contacted_protein_rows(
            [center],
            protein_lookup,
            distance_cutoff=distance_cutoff,
        )
        if not protein_rows:
            continue

        protein_rows = sorted(protein_rows, key=lambda row: row.q_pn_unit_iid)
        protein_iids = tuple(row.q_pn_unit_iid for row in protein_rows)
        protein_clusters = tuple(row.q_pn_unit_cluster_id for row in protein_rows)
        crop_center_iids = [center.q_pn_unit_iid, *protein_iids]
        query_iids = expand_bml_context_query_iids(
            source_rows=context_expansion_source_rows(policy, [center], protein_rows),
            crop_center_pn_unit_iids=crop_center_iids,
            metadata_df=metadata_df,
            lookup=context_lookup,
            policy=policy,
            stats=context_stats,
        )

        row = center._asdict()
        row.pop("crop_center_pn_unit_iids", None)
        row.update(
            {
                "query_pn_unit_iids": query_iids,
                "crop_center_pn_unit_iids": tuple(crop_center_iids),
                "ligand_pn_unit_iids": (center.q_pn_unit_iid,),
                "protein_pn_unit_iids": protein_iids,
                "protein_cluster_multiset": protein_clusters,
                "protein_val_cluster_multiset": tuple(r.q_pn_unit_cluster_val_id for r in protein_rows),
                "ligand_ccd_key": ("peptide_seq_cluster", center.q_pn_unit_cluster_id),
                "interface_type": "peptide_protein",
                "n_contacting_protein_atoms": contact_count,
            }
        )
        row["example_id"] = generate_example_id(
            [dataset_name, "interface"],
            row["pdb_id"],
            row["assembly_id"],
            list(row["crop_center_pn_unit_iids"]),
        )
        rows.append(row)

    return rows


def _build_nucleic_acid_ligand_interface_rows(
    metadata_df: pd.DataFrame,
    protein_lookup: dict,
    dataset_name: str,
    distance_cutoff: float,
    cfg: dict | DictConfig,
    policy: BMLPolicy,
    context_lookup: dict,
    context_stats: BMLContextExpansionStats,
) -> list[dict]:
    center_df = metadata_df[nucleic_acid_ligand_center_mask(metadata_df, cfg)].copy()
    if center_df.empty:
        return []

    rows = []
    for _, group_df in center_df.groupby(
        ["pdb_id", "assembly_id", "q_pn_unit_nucleic_acid_group_iids"],
        sort=False,
    ):
        group_rows = sorted(
            list(group_df.itertuples(index=False)),
            key=lambda row: row.q_pn_unit_iid,
        )
        contact_count, protein_rows = _collect_contacted_protein_rows(
            group_rows,
            protein_lookup,
            distance_cutoff=distance_cutoff,
        )
        if not protein_rows:
            continue

        ligand_iids = tuple(dict.fromkeys(row.q_pn_unit_iid for row in group_rows))
        protein_rows = sorted(protein_rows, key=lambda row: row.q_pn_unit_iid)
        protein_iids = tuple(row.q_pn_unit_iid for row in protein_rows)
        protein_clusters = tuple(row.q_pn_unit_cluster_id for row in protein_rows)
        crop_center_iids = [*ligand_iids, *protein_iids]
        query_iids = expand_bml_context_query_iids(
            source_rows=context_expansion_source_rows(policy, group_rows, protein_rows),
            crop_center_pn_unit_iids=crop_center_iids,
            metadata_df=metadata_df,
            lookup=context_lookup,
            policy=policy,
            stats=context_stats,
        )
        center = group_rows[0]

        row = center._asdict()
        row.pop("crop_center_pn_unit_iids", None)
        row.update(
            {
                "query_pn_unit_iids": query_iids,
                "crop_center_pn_unit_iids": tuple(crop_center_iids),
                "ligand_pn_unit_iids": ligand_iids,
                "protein_pn_unit_iids": protein_iids,
                "protein_cluster_multiset": protein_clusters,
                "protein_val_cluster_multiset": tuple(r.q_pn_unit_cluster_val_id for r in protein_rows),
                "ligand_ccd_key": ("nuc_seq_cluster", _nuc_group_cluster_key(group_rows)),
                "interface_type": "nuc_lig_protein",
                "n_contacting_protein_atoms": contact_count,
            }
        )
        row["example_id"] = generate_example_id(
            [dataset_name, "interface"],
            row["pdb_id"],
            row["assembly_id"],
            list(row["crop_center_pn_unit_iids"]),
        )
        rows.append(row)

    return rows


def _build_protein_interface_rows(
    protein_df: pd.DataFrame,
    protein_lookup: dict,
    dataset_name: str,
    distance_cutoff: float,
) -> list[dict]:
    edge_contacts, neighbors_by_source = _collect_protein_interface_edges(
        protein_df,
        protein_lookup,
        distance_cutoff,
    )

    rows_by_tuple = {}
    for query in protein_df.itertuples(index=False):
        assembly_id = str(query.assembly_id)
        source_key = (query.pdb_id, assembly_id, query.q_pn_unit_iid)
        neighbor_iids = neighbors_by_source.get(source_key, set())
        protein_iids = tuple(sorted({query.q_pn_unit_iid, *neighbor_iids}))
        if len(protein_iids) < 2:
            continue

        tuple_key = (query.pdb_id, assembly_id, protein_iids)
        if tuple_key in rows_by_tuple:
            continue

        protein_rows = [protein_lookup[(query.pdb_id, assembly_id, iid)] for iid in protein_iids]
        protein_clusters = tuple(row.q_pn_unit_cluster_id for row in protein_rows)
        row = protein_rows[0]._asdict()
        row.pop("crop_center_pn_unit_iids", None)
        row.update(
            {
                "query_pn_unit_iids": list(protein_iids),
                "ligand_pn_unit_iids": (),
                "protein_pn_unit_iids": protein_iids,
                "protein_cluster_multiset": protein_clusters,
                "protein_val_cluster_multiset": tuple(r.q_pn_unit_cluster_val_id for r in protein_rows),
                "ligand_ccd_key": ("protein_interface", "none"),
                "interface_type": "protein_protein",
                "query_pn_unit_iids_only": True,
                "n_protein_protein_contacts": _sum_protein_tuple_contacts(
                    query.pdb_id,
                    assembly_id,
                    protein_iids,
                    edge_contacts,
                ),
            }
        )
        row["example_id"] = generate_example_id(
            [dataset_name, "interface"],
            row["pdb_id"],
            row["assembly_id"],
            row["query_pn_unit_iids"],
        )
        rows_by_tuple[tuple_key] = row

    return list(rows_by_tuple.values())


def _collect_protein_interface_edges(
    protein_df: pd.DataFrame,
    protein_lookup: dict,
    distance_cutoff: float,
) -> tuple[dict, dict]:
    edge_contacts = {}
    neighbors_by_source = {}
    for source in protein_df.itertuples(index=False):
        assembly_id = str(source.assembly_id)
        source_key = (source.pdb_id, assembly_id, source.q_pn_unit_iid)
        contacts = parse_partner_list(getattr(source, "q_pn_unit_contacting_pn_unit_iids"))
        for contact in contacts:
            if not isinstance(contact, dict) or not contact_within_cutoff(contact, distance_cutoff):
                continue
            partner_rows = _resolve_partner_rows(
                source.pdb_id,
                source.assembly_id,
                contact.get("pn_unit_iid"),
                contact.get("chain_iid"),
                protein_lookup,
            )
            if not partner_rows:
                continue

            n_contacts = contact_count(contact)
            for partner in partner_rows:
                if partner.q_pn_unit_iid == source.q_pn_unit_iid:
                    continue
                neighbors_by_source.setdefault(source_key, set()).add(partner.q_pn_unit_iid)
                edge_iids = tuple(sorted((source.q_pn_unit_iid, partner.q_pn_unit_iid)))
                edge_key = (source.pdb_id, assembly_id, edge_iids)
                edge_contacts[edge_key] = max(edge_contacts.get(edge_key, 0), n_contacts)
    return edge_contacts, neighbors_by_source


def _sum_protein_tuple_contacts(
    pdb_id: str,
    assembly_id: str,
    protein_iids: tuple[str, ...],
    edge_contacts: dict,
) -> int:
    total = 0
    for i, source_iid in enumerate(protein_iids):
        for partner_iid in protein_iids[i + 1 :]:
            edge_key = (pdb_id, assembly_id, tuple(sorted((source_iid, partner_iid))))
            total += edge_contacts.get(edge_key, 0)
    return int(total)


def _collect_contacted_protein_rows(
    source_rows,
    protein_lookup: dict,
    distance_cutoff: float | None,
) -> tuple[int, list]:
    total = 0
    protein_rows_by_iid = {}
    for source in source_rows:
        contacts = parse_partner_list(getattr(source, "q_pn_unit_contacting_pn_unit_iids"))
        for contact in contacts:
            if not isinstance(contact, dict) or not contact_within_cutoff(contact, distance_cutoff):
                continue
            resolved = _resolve_partner_rows(
                source.pdb_id,
                source.assembly_id,
                contact.get("pn_unit_iid"),
                contact.get("chain_iid"),
                protein_lookup,
            )
            if not resolved:
                continue
            total += contact_count(contact)
            for protein in resolved:
                protein_rows_by_iid[protein.q_pn_unit_iid] = protein
    return int(total), list(protein_rows_by_iid.values())


def _collect_metal_protein_donor_partners(
    center,
    protein_lookup: dict,
    *,
    count_fn: Callable[[dict], int] = contact_count,
) -> tuple[int, list]:
    contacts = parse_partner_list(getattr(center, "q_pn_unit_per_partner_contacts_metal"))
    total = 0
    protein_rows_by_iid = {}
    for contact in contacts:
        if not isinstance(contact, dict):
            continue
        count = count_fn(contact)
        resolved = _resolve_partner_rows(
            center.pdb_id,
            center.assembly_id,
            contact.get("pn_unit_iid"),
            contact.get("chain_iid"),
            protein_lookup,
        )
        if not resolved or count <= 0:
            continue
        total += count
        for protein in resolved:
            protein_rows_by_iid[protein.q_pn_unit_iid] = protein
    return total, list(protein_rows_by_iid.values())


def _collect_small_molecule_protein_partners(center, protein_lookup: dict) -> tuple[int, list]:
    contacts = parse_partner_list(getattr(center, "q_pn_unit_per_partner_contacts_to_protein_small_molecule"))
    total = 0
    protein_rows_by_iid = {}
    for contact in contacts:
        if not isinstance(contact, dict):
            continue
        resolved = _resolve_partner_rows(
            center.pdb_id,
            center.assembly_id,
            contact.get("pn_unit_iid"),
            contact.get("chain_iid"),
            protein_lookup,
        )
        if not resolved:
            continue
        try:
            count = int(contact.get("count", 0))
        except (TypeError, ValueError):
            count = 0
        total += count
        for protein in resolved:
            protein_rows_by_iid[protein.q_pn_unit_iid] = protein
    return total, list(protein_rows_by_iid.values())


def _nuc_group_cluster_key(group_rows) -> tuple:
    cluster_ids = {
        getattr(row, "q_pn_unit_cluster_id")
        for row in group_rows
        if not pd.isna(getattr(row, "q_pn_unit_cluster_id"))
    }
    return tuple(sorted(cluster_ids, key=repr))


def _resolve_partner_rows(pdb_id: str, assembly_id, raw_iid, chain_iid, lookup: dict) -> list:
    candidates = []
    for value in (raw_iid, chain_iid):
        if value is not None:
            candidates.append(str(value))

    seen_candidates = set()
    for candidate in candidates:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        key = (pdb_id, str(assembly_id), candidate)
        if key in lookup:
            return [lookup[key]]

    rows = []
    seen_rows = set()
    for candidate in candidates:
        for iid in split_components(candidate):
            key = (pdb_id, str(assembly_id), iid)
            if iid in seen_rows or key not in lookup:
                continue
            seen_rows.add(iid)
            rows.append(lookup[key])
    return rows
