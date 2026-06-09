"""Nucleic-acid group computation for pn_unit metadata.

Builds connected-component groups of nucleic-acid pn_units from their pairwise
contacts and derives per-group columns (group id/iids, resolved residue counts,
cluster ids, ligand/polymer flags). Shared by the SD dataset metadata pipeline
and the BMSM preprocessing augmentation.
"""

from collections import deque

import pandas as pd

from allatom_design.data.utils.pn_unit import (
    join_sorted,
    natural_key,
    parse_contacts_with_distance,
    split_components,
)

NUCLEIC_ACID_TYPE_VALUES = {3, 4, 7}
DEFAULT_NUCLEIC_ACID_GROUP_DISTANCE_CUTOFF = 4.5
DEFAULT_NUCLEIC_ACID_LIGAND_MAX_RESIDUES = 10


def is_nucleic_acid_pn_unit(g: pd.DataFrame) -> pd.Series:
    if 'q_pn_unit_type' in g.columns:
        return g['q_pn_unit_type'].isin(NUCLEIC_ACID_TYPE_VALUES).fillna(False)
    cols = ['q_pn_unit_is_DNA', 'q_pn_unit_is_RNA', 'q_pn_unit_is_RNA_DNA_hybrid']
    present = [c for c in cols if c in g.columns]
    if present:
        out = pd.Series(False, index=g.index)
        for col in present:
            out = out | g[col].fillna(False).astype(bool)
        return out
    return pd.Series(False, index=g.index)


def _format_group_label(ids):
    return "(" + join_sorted(ids) + ")"


def _format_group_iids(ids):
    return join_sorted(ids)


def _format_cluster_ids(ids):
    clean = []
    for value in ids:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        try:
            clean.append(int(value))
        except (TypeError, ValueError):
            clean.append(str(value))
    return ", ".join(str(v) for v in sorted(set(clean), key=natural_key))


def compute_nucleic_acid_groups_per_pdb(
    g: pd.DataFrame,
    nucleic_acid_dist_threshold: float = DEFAULT_NUCLEIC_ACID_GROUP_DISTANCE_CUTOFF,
    nucleic_acid_ligand_max_residues: int = DEFAULT_NUCLEIC_ACID_LIGAND_MAX_RESIDUES,
) -> pd.DataFrame:
    columns = [
        'q_pn_unit_nucleic_acid_group_id',
        'q_pn_unit_nucleic_acid_group_iids',
        'q_pn_unit_num_resolved_residues_in_nucleic_acid_group',
        'q_pn_unit_nucleic_acid_group_cluster_ids',
        'q_pn_unit_is_nuc_ligand',
        'q_pn_unit_is_nuc_polymer',
    ]
    out = pd.DataFrame(index=g.index)
    out['q_pn_unit_nucleic_acid_group_id'] = pd.NA
    out['q_pn_unit_nucleic_acid_group_iids'] = pd.NA
    out['q_pn_unit_num_resolved_residues_in_nucleic_acid_group'] = 0
    out['q_pn_unit_nucleic_acid_group_cluster_ids'] = pd.NA
    out['q_pn_unit_is_nuc_ligand'] = False
    out['q_pn_unit_is_nuc_polymer'] = False

    na_mask = is_nucleic_acid_pn_unit(g)
    if not na_mask.any():
        return out[columns]

    sub = g.loc[
        na_mask,
        [
            'q_pn_unit_iid',
            'q_pn_unit_contacting_pn_unit_iids',
            'q_pn_unit_num_resolved_residues',
            *(['q_pn_unit_cluster_id'] if 'q_pn_unit_cluster_id' in g.columns else []),
        ],
    ].copy()
    first_by_iid = sub.drop_duplicates(subset=['q_pn_unit_iid']).copy()

    rid_to_comps = {
        str(row.q_pn_unit_iid): tuple(split_components(str(row.q_pn_unit_iid)))
        for row in first_by_iid.itertuples(index=False)
    }
    rid_to_residues = {
        str(row.q_pn_unit_iid): int(row.q_pn_unit_num_resolved_residues)
        for row in first_by_iid.itertuples(index=False)
    }
    if 'q_pn_unit_cluster_id' in first_by_iid.columns:
        rid_to_cluster = {
            str(row.q_pn_unit_iid): row.q_pn_unit_cluster_id
            for row in first_by_iid.itertuples(index=False)
        }
    else:
        rid_to_cluster = {}

    atomic_nodes = set(c for comps in rid_to_comps.values() for c in comps)
    if not atomic_nodes:
        return out[columns]

    rid_contacts_pairs = {}
    for rid_raw, ssub in sub.groupby('q_pn_unit_iid', sort=False):
        rid = str(rid_raw)
        pairs = []
        for val in ssub['q_pn_unit_contacting_pn_unit_iids']:
            pairs.extend(parse_contacts_with_distance(val))
        rid_contacts_pairs[rid] = pairs

    comp_dir_adj = {u: set() for u in atomic_nodes}
    for rid, comps in rid_to_comps.items():
        for u in comps:
            for target_iid, md in rid_contacts_pairs.get(rid, ()):
                if md is None or md > nucleic_acid_dist_threshold:
                    continue
                for v in split_components(target_iid):
                    if v in atomic_nodes and v != u:
                        comp_dir_adj[u].add(v)

    comp_adj = {u: set() for u in atomic_nodes}
    for u, vs in comp_dir_adj.items():
        for v in vs:
            comp_adj[u].add(v)
            comp_adj[v].add(u)

    visited, components = set(), []
    for u in sorted(atomic_nodes, key=natural_key):
        if u in visited:
            continue
        comp_set = {u}
        dq = deque([u])
        visited.add(u)
        while dq:
            x = dq.popleft()
            for y in comp_adj.get(x, ()):
                if y not in visited:
                    visited.add(y)
                    comp_set.add(y)
                    dq.append(y)
        components.append(comp_set)

    comp_iids = []
    for comp_set in components:
        iids = sorted(
            {rid for rid, comps in rid_to_comps.items() if set(comps) & comp_set},
            key=natural_key,
        )
        comp_iids.append(iids)

    rid_to_group_iids = {}
    for rid, comps in rid_to_comps.items():
        merged = set()
        comp_set = set(comps)
        for iids in comp_iids:
            if any(set(rid_to_comps[iid]) & comp_set for iid in iids):
                merged.update(iids)
        rid_to_group_iids[rid] = sorted(merged or {rid}, key=natural_key)

    for rid, group_iids in rid_to_group_iids.items():
        label = _format_group_label(group_iids)
        residues = sum(rid_to_residues[iid] for iid in group_iids)
        cluster_ids = [rid_to_cluster.get(iid) for iid in group_iids]
        row_mask = na_mask & (g['q_pn_unit_iid'].astype(str) == rid)
        out.loc[row_mask, 'q_pn_unit_nucleic_acid_group_id'] = label
        out.loc[row_mask, 'q_pn_unit_nucleic_acid_group_iids'] = _format_group_iids(group_iids)
        out.loc[row_mask, 'q_pn_unit_num_resolved_residues_in_nucleic_acid_group'] = int(residues)
        out.loc[row_mask, 'q_pn_unit_nucleic_acid_group_cluster_ids'] = _format_cluster_ids(cluster_ids)

    group_residues = out['q_pn_unit_num_resolved_residues_in_nucleic_acid_group'].astype(int)
    out.loc[na_mask, 'q_pn_unit_is_nuc_ligand'] = (
        group_residues.loc[na_mask] <= nucleic_acid_ligand_max_residues
    )
    out.loc[na_mask, 'q_pn_unit_is_nuc_polymer'] = (
        group_residues.loc[na_mask] > nucleic_acid_ligand_max_residues
    )
    return out[columns]


def add_nucleic_acid_group_columns(
    df: pd.DataFrame,
    nucleic_acid_dist_threshold: float = DEFAULT_NUCLEIC_ACID_GROUP_DISTANCE_CUTOFF,
    nucleic_acid_ligand_max_residues: int = DEFAULT_NUCLEIC_ACID_LIGAND_MAX_RESIDUES,
) -> pd.DataFrame:
    out = df.copy()
    group_cols = [
        'q_pn_unit_nucleic_acid_group_id',
        'q_pn_unit_nucleic_acid_group_iids',
        'q_pn_unit_num_resolved_residues_in_nucleic_acid_group',
        'q_pn_unit_nucleic_acid_group_cluster_ids',
        'q_pn_unit_is_nuc_ligand',
        'q_pn_unit_is_nuc_polymer',
    ]
    out['q_pn_unit_nucleic_acid_group_id'] = pd.NA
    out['q_pn_unit_nucleic_acid_group_iids'] = pd.NA
    out['q_pn_unit_num_resolved_residues_in_nucleic_acid_group'] = 0
    out['q_pn_unit_nucleic_acid_group_cluster_ids'] = pd.NA
    out['q_pn_unit_is_nuc_ligand'] = False
    out['q_pn_unit_is_nuc_polymer'] = False

    na_mask = is_nucleic_acid_pn_unit(out)
    if not na_mask.any():
        return out

    if {'pdb_id', 'assembly_id'}.issubset(out.columns):
        grouped = out.loc[na_mask].groupby(['pdb_id', 'assembly_id'], sort=False)
    else:
        grouped = [(None, out.loc[na_mask])]

    for _, g in grouped:
        computed = compute_nucleic_acid_groups_per_pdb(
            g,
            nucleic_acid_dist_threshold=nucleic_acid_dist_threshold,
            nucleic_acid_ligand_max_residues=nucleic_acid_ligand_max_residues,
        )
        out.loc[computed.index, group_cols] = computed[group_cols]

    out['q_pn_unit_num_resolved_residues_in_nucleic_acid_group'] = (
        out['q_pn_unit_num_resolved_residues_in_nucleic_acid_group'].fillna(0).astype('int64')
    )
    out['q_pn_unit_is_nuc_ligand'] = out['q_pn_unit_is_nuc_ligand'].fillna(False).astype(bool)
    out['q_pn_unit_is_nuc_polymer'] = out['q_pn_unit_is_nuc_polymer'].fillna(False).astype(bool)
    return out
