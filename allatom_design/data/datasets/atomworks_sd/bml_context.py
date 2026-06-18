"""Biologically meaningful ligand center/context policy for AtomWorks SD.

This module owns the train-time BML metadata contract:

* parse the nested ``bml_center`` / ``bml_context`` config;
* materialize center and context eligibility columns on metadata rows;
* expand interface query pn_units with eligible 5A context while preserving a
  smaller crop-center pn_unit set for AF3-style crop-center sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allatom_design.data.datasets.atomworks_sd.selectors import (
    metal_center_mask,
    small_molecule_center_mask,
)
from allatom_design.data.utils.pn_unit import (
    contact_count,
    contact_within_cutoff,
    natural_key,
    parse_partner_list,
    series_has_any_exact_ccd,
    split_components,
)

logger = logging.getLogger(__name__)


BML_CENTER_SMALL_MOLECULE_COL = "q_pn_unit_is_biologically_meaningful_small_molecule"
BML_CENTER_METAL_COL = "q_pn_unit_is_biologically_meaningful_metal"

CONTEXT_SMALL_MOLECULE_COL = "q_pn_unit_is_context_small_molecule"
CONTEXT_METAL_COL = "q_pn_unit_is_context_metal"
CONTEXT_HALIDE_COL = "q_pn_unit_is_context_halide"
CONTEXT_PEPTIDE_COL = "q_pn_unit_is_context_peptide"
CONTEXT_NUCLEIC_ACID_CHAINS_COL = "q_pn_unit_is_context_nucleic_acid_chains"

BML_CENTER_FLAG_COLUMNS = [
    BML_CENTER_SMALL_MOLECULE_COL,
    BML_CENTER_METAL_COL,
]
CONTEXT_FLAG_COLUMNS = [
    CONTEXT_SMALL_MOLECULE_COL,
    CONTEXT_METAL_COL,
    CONTEXT_HALIDE_COL,
    CONTEXT_PEPTIDE_COL,
    CONTEXT_NUCLEIC_ACID_CHAINS_COL,
]
BML_ANNOTATION_COLUMNS = [*BML_CENTER_FLAG_COLUMNS, *CONTEXT_FLAG_COLUMNS]

LEGACY_BML_CONFIG_KEYS = [
    "metal_external_evidence_policy",
    "allowed_ccd_codes",
    "min_protein_donor_atoms",
    "min_avg_occupancy_nonpolymer",
    "small_molecule_artifact_list_path",
    "small_molecule_artifact_external_evidence_policy",
    "small_molecule_external_evidence_policy",
    "min_contacting_protein_atoms_small_molecule",
    "min_contacting_protein_atom_ratio_small_molecule",
    "min_avg_occupancy_nonpolymer_small_molecule",
    "max_missing_atom_fraction_small_molecule",
    "nucleic_acid_group_distance_cutoff",
    "nucleic_acid_ligand_max_residues",
]

CONTEXT_REQUIRED_COLUMNS = [
    "q_pn_unit_iid",
    "pdb_id",
    "assembly_id",
    "q_pn_unit_contacting_pn_unit_iids",
    "q_pn_unit_per_partner_contacts_small_molecule",
    "q_pn_unit_per_partner_contacts_metal",
    "q_pn_unit_per_partner_contacts_halide",
    "q_pn_unit_expected_heavy_atoms_non_polymer",
    "q_pn_unit_avg_occupancy_nonpolymer",
    "q_pn_unit_non_polymer_res_names",
    "q_pn_unit_is_protein",
    "q_pn_unit_is_small_molecule",
    "q_pn_unit_is_metal",
    "q_pn_unit_is_halide",
    "q_pn_unit_is_peptide",
    "q_pn_unit_is_nuc",
]


@dataclass(frozen=True)
class MetalCenterPolicy:
    external_evidence_policy: str = "no_filter"
    allowed_ccd_codes: list[str] | None = None
    min_protein_donor_atoms: int = 3
    min_avg_occupancy_nonpolymer: float | None = 0.5

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "MetalCenterPolicy":
        cfg = cfg or {}
        return cls(
            external_evidence_policy=str(cfg.get("external_evidence_policy", "no_filter")),
            allowed_ccd_codes=_optional_list(cfg.get("allowed_ccd_codes", None)),
            min_protein_donor_atoms=int(cfg.get("min_protein_donor_atoms", 3)),
            min_avg_occupancy_nonpolymer=_optional_float(cfg.get("min_avg_occupancy_nonpolymer", 0.5)),
        )

    def as_selector_cfg(self) -> dict[str, Any]:
        return {
            "external_evidence_policy": self.external_evidence_policy,
            "metal_external_evidence_policy": self.external_evidence_policy,
            "allowed_ccd_codes": self.allowed_ccd_codes,
            "min_avg_occupancy_nonpolymer": self.min_avg_occupancy_nonpolymer,
        }


@dataclass(frozen=True)
class SmallMoleculeCenterPolicy:
    artifact_list_path: str | None = None
    artifact_external_evidence_policy: str = "no_filter"
    min_contacting_protein_atoms: int | None = 20
    min_contacting_protein_atom_ratio: float | None = None
    min_avg_occupancy_nonpolymer: float | None = 0.5
    max_missing_atom_fraction: float | None = 0.2

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "SmallMoleculeCenterPolicy":
        cfg = cfg or {}
        return cls(
            artifact_list_path=_optional_str(cfg.get("artifact_list_path", None)),
            artifact_external_evidence_policy=str(
                cfg.get("artifact_external_evidence_policy", "no_filter")
            ),
            min_contacting_protein_atoms=_optional_int(
                cfg.get("min_contacting_protein_atoms", 20)
            ),
            min_contacting_protein_atom_ratio=_optional_float(
                cfg.get("min_contacting_protein_atom_ratio", None)
            ),
            min_avg_occupancy_nonpolymer=_optional_float(
                cfg.get("min_avg_occupancy_nonpolymer", 0.5)
            ),
            max_missing_atom_fraction=_optional_float(cfg.get("max_missing_atom_fraction", 0.2)),
        )

    def as_selector_cfg(self) -> dict[str, Any]:
        return {
            "small_molecule_artifact_list_path": self.artifact_list_path,
            "small_molecule_artifact_external_evidence_policy": (
                self.artifact_external_evidence_policy
            ),
            "min_contacting_protein_atoms_small_molecule": (
                self.min_contacting_protein_atoms
            ),
            "min_contacting_protein_atom_ratio_small_molecule": (
                self.min_contacting_protein_atom_ratio
            ),
            "min_avg_occupancy_nonpolymer_small_molecule": (
                self.min_avg_occupancy_nonpolymer
            ),
            "max_missing_atom_fraction_small_molecule": self.max_missing_atom_fraction,
        }


@dataclass(frozen=True)
class NucleicAcidCenterPolicy:
    group_distance_cutoff: float = 4.5
    ligand_max_residues: int = 10

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "NucleicAcidCenterPolicy":
        cfg = cfg or {}
        return cls(
            group_distance_cutoff=float(cfg.get("group_distance_cutoff", 4.5)),
            ligand_max_residues=int(cfg.get("ligand_max_residues", 10)),
        )


@dataclass(frozen=True)
class SmallMoleculeContextPolicy:
    min_contacting_heavy_atoms: int = 20
    min_contacting_heavy_atom_ratio: float | None = 1.0

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "SmallMoleculeContextPolicy":
        cfg = cfg or {}
        return cls(
            min_contacting_heavy_atoms=int(cfg.get("min_contacting_heavy_atoms", 20)),
            min_contacting_heavy_atom_ratio=_optional_float(
                cfg.get("min_contacting_heavy_atom_ratio", 1.0)
            ),
        )


@dataclass(frozen=True)
class MetalContextPolicy:
    min_coordinating_atoms: int = 3
    min_avg_occupancy_nonpolymer: float | None = 0.5

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "MetalContextPolicy":
        cfg = cfg or {}
        return cls(
            min_coordinating_atoms=int(cfg.get("min_coordinating_atoms", 3)),
            min_avg_occupancy_nonpolymer=_optional_float(
                cfg.get("min_avg_occupancy_nonpolymer", 0.5)
            ),
        )


@dataclass(frozen=True)
class HalideContextPolicy:
    allowed_ccd_codes: list[str]
    min_non_carbon_contacts: int = 3
    min_avg_occupancy_nonpolymer: float | None = 0.5

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "HalideContextPolicy":
        cfg = cfg or {}
        allowed = _optional_list(cfg.get("allowed_ccd_codes", ["F"]))
        if not allowed:
            raise ValueError("`bml_context.halide.allowed_ccd_codes` must not be empty.")
        return cls(
            allowed_ccd_codes=allowed,
            min_non_carbon_contacts=int(cfg.get("min_non_carbon_contacts", 3)),
            min_avg_occupancy_nonpolymer=_optional_float(
                cfg.get("min_avg_occupancy_nonpolymer", 0.5)
            ),
        )


@dataclass(frozen=True)
class EnabledContextPolicy:
    enabled: bool = True

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "EnabledContextPolicy":
        cfg = cfg or {}
        return cls(enabled=bool(cfg.get("enabled", True)))


@dataclass(frozen=True)
class BMLCenterPolicy:
    metal: MetalCenterPolicy
    small_molecule: SmallMoleculeCenterPolicy
    nucleic_acid: NucleicAcidCenterPolicy

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "BMLCenterPolicy":
        cfg = cfg or {}
        return cls(
            metal=MetalCenterPolicy.from_cfg(cfg.get("metal", {})),
            small_molecule=SmallMoleculeCenterPolicy.from_cfg(cfg.get("small_molecule", {})),
            nucleic_acid=NucleicAcidCenterPolicy.from_cfg(cfg.get("nucleic_acid", {})),
        )


@dataclass(frozen=True)
class BMLContextPolicy:
    enabled: bool
    contact_distance_cutoff: float
    max_iterations: int
    small_molecule: SmallMoleculeContextPolicy
    metal: MetalContextPolicy
    halide: HalideContextPolicy
    peptide: EnabledContextPolicy
    nucleic_acid: EnabledContextPolicy

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "BMLContextPolicy":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            contact_distance_cutoff=float(cfg.get("contact_distance_cutoff", 5.0)),
            max_iterations=int(cfg.get("max_iterations", 10)),
            small_molecule=SmallMoleculeContextPolicy.from_cfg(cfg.get("small_molecule", {})),
            metal=MetalContextPolicy.from_cfg(cfg.get("metal", {})),
            halide=HalideContextPolicy.from_cfg(cfg.get("halide", {})),
            peptide=EnabledContextPolicy.from_cfg(cfg.get("peptide", {})),
            nucleic_acid=EnabledContextPolicy.from_cfg(cfg.get("nucleic_acid", {})),
        )


@dataclass(frozen=True)
class BMLPolicy:
    center: BMLCenterPolicy
    context: BMLContextPolicy

    @classmethod
    def from_cfg(cls, cfg: dict | DictConfig | None) -> "BMLPolicy":
        cfg = cfg or {}
        validate_no_legacy_bml_keys(cfg)
        if "bml_center" not in cfg:
            raise KeyError(
                "AtomWorks SD data config must define nested `bml_center`. "
                "Legacy flat BML keys are no longer supported."
            )
        return cls(
            center=BMLCenterPolicy.from_cfg(cfg.get("bml_center", {})),
            context=BMLContextPolicy.from_cfg(cfg.get("bml_context", {})),
        )


@dataclass
class BMLContextExpansionStats:
    missing_partner_rows: int = 0
    skipped_ineligible_partners: int = 0


def validate_no_legacy_bml_keys(cfg: dict | DictConfig | None) -> None:
    cfg = cfg or {}
    present = [key for key in LEGACY_BML_CONFIG_KEYS if key in cfg]
    if present:
        raise ValueError(
            "Legacy flat AtomWorks SD BML config keys are no longer supported: "
            f"{present}. Move these settings under nested `bml_center` and "
            "`bml_context`."
        )


def has_nested_bml_config(cfg: dict | DictConfig | None) -> bool:
    cfg = cfg or {}
    return "bml_center" in cfg


def ensure_bml_context_annotations(
    metadata_df: pd.DataFrame,
    cfg: dict | DictConfig,
) -> pd.DataFrame:
    if all(col in metadata_df.columns for col in BML_ANNOTATION_COLUMNS):
        BMLPolicy.from_cfg(cfg)
        return metadata_df
    return annotate_bml_context(metadata_df, cfg)


def annotate_bml_context(
    metadata_df: pd.DataFrame,
    cfg: dict | DictConfig,
) -> pd.DataFrame:
    policy = BMLPolicy.from_cfg(cfg)
    out = metadata_df.copy()

    _require_columns(
        out,
        [
            "q_pn_unit_is_metal",
            "q_pn_unit_is_small_molecule",
            "q_pn_unit_per_partner_contacts_metal",
            "q_pn_unit_per_partner_contacts_to_protein_small_molecule",
            "q_pn_unit_expected_heavy_atoms_non_polymer",
            "q_pn_unit_num_resolved_atoms",
            "q_pn_unit_avg_occupancy_nonpolymer",
            "q_pn_unit_non_polymer_res_names",
            "q_pn_unit_has_external_evidence",
        ],
        context="BML center annotation",
    )
    if policy.context.enabled:
        _require_columns(out, CONTEXT_REQUIRED_COLUMNS, context="BML context annotation")

    pn_unit_kind = _pn_unit_kind(out)
    lookup = build_pn_unit_index_lookup(out)
    metal_edges = _build_contact_edges(
        out,
        "q_pn_unit_per_partner_contacts_metal",
        pn_unit_kind,
        lookup,
    )
    protein_partner_passes = pn_unit_kind.eq("protein")

    metal_protein_donor_counts = _sum_edge_counts(
        metal_edges,
        out.index,
        source_kind="metal",
        partner_passes=protein_partner_passes,
    )
    bmm_mask = (
        metal_center_mask(out, policy.center.metal.as_selector_cfg())
        & (metal_protein_donor_counts >= policy.center.metal.min_protein_donor_atoms)
    )
    bmsm_mask = small_molecule_center_mask(
        out,
        policy.center.small_molecule.as_selector_cfg(),
    )
    out[BML_CENTER_METAL_COL] = bmm_mask.astype(bool)
    out[BML_CENTER_SMALL_MOLECULE_COL] = bmsm_mask.astype(bool)

    for col in CONTEXT_FLAG_COLUMNS:
        out[col] = False
    if not policy.context.enabled:
        return out

    small_molecule_edges = _build_contact_edges(
        out,
        "q_pn_unit_per_partner_contacts_small_molecule",
        pn_unit_kind,
        lookup,
    )
    halide_edges = _build_contact_edges(
        out,
        "q_pn_unit_per_partner_contacts_halide",
        pn_unit_kind,
        lookup,
    )

    context_masks = _solve_context_fixed_point(
        out,
        small_molecule_edges=small_molecule_edges,
        metal_edges=metal_edges,
        halide_edges=halide_edges,
        policy=policy,
    )
    for col, mask in context_masks.items():
        out[col] = mask.astype(bool)
    return out


def build_pn_unit_index_lookup(metadata_df: pd.DataFrame) -> dict[tuple[str, str, str], Any]:
    lookup = {}
    for idx, row in zip(metadata_df.index, metadata_df.itertuples(index=False)):
        lookup[(str(row.pdb_id), str(row.assembly_id), str(row.q_pn_unit_iid))] = idx
    return lookup


def expand_bml_context_query_iids(
    *,
    source_rows,
    crop_center_pn_unit_iids,
    metadata_df: pd.DataFrame,
    lookup: dict[tuple[str, str, str], Any] | None,
    policy: BMLPolicy,
    stats: BMLContextExpansionStats | None = None,
) -> list[str]:
    center_iids = [str(iid) for iid in crop_center_pn_unit_iids]
    if not policy.context.enabled:
        return list(dict.fromkeys(center_iids))

    lookup = lookup or build_pn_unit_index_lookup(metadata_df)
    stats = stats or BMLContextExpansionStats()
    seen = set()
    ordered_centers = []
    for iid in center_iids:
        if iid in seen:
            continue
        seen.add(iid)
        ordered_centers.append(iid)

    extra_iids = set()
    for source in source_rows:
        contacts = parse_partner_list(getattr(source, "q_pn_unit_contacting_pn_unit_iids", None))
        for contact in contacts:
            if not isinstance(contact, dict) or not contact_within_cutoff(
                contact,
                policy.context.contact_distance_cutoff,
            ):
                continue
            partner_indices = _resolve_partner_indices(
                source.pdb_id,
                source.assembly_id,
                contact.get("pn_unit_iid"),
                contact.get("chain_iid"),
                lookup,
            )
            if not partner_indices:
                stats.missing_partner_rows += 1
                continue
            for partner_idx in partner_indices:
                partner_row = metadata_df.loc[partner_idx]
                partner_iid = str(partner_row["q_pn_unit_iid"])
                if partner_iid in seen or partner_iid in extra_iids:
                    continue
                if _row_is_expandable_context(partner_row):
                    extra_iids.add(partner_iid)
                else:
                    stats.skipped_ineligible_partners += 1

    return [*ordered_centers, *sorted(extra_iids, key=natural_key)]


def _solve_context_fixed_point(
    metadata_df: pd.DataFrame,
    *,
    small_molecule_edges: pd.DataFrame,
    metal_edges: pd.DataFrame,
    halide_edges: pd.DataFrame,
    policy: BMLPolicy,
) -> dict[str, pd.Series]:
    index = metadata_df.index
    raw_small_molecule = _bool_col(metadata_df, "q_pn_unit_is_small_molecule")
    raw_metal = _bool_col(metadata_df, "q_pn_unit_is_metal")
    raw_halide = _bool_col(metadata_df, "q_pn_unit_is_halide")
    raw_peptide = _bool_col(metadata_df, "q_pn_unit_is_peptide")
    raw_nuc = _bool_col(metadata_df, "q_pn_unit_is_nuc")
    allowed_halide = series_has_any_exact_ccd(
        metadata_df["q_pn_unit_non_polymer_res_names"],
        policy.context.halide.allowed_ccd_codes,
        index=index,
    )

    ctx_small_molecule = raw_small_molecule.copy()
    ctx_metal = raw_metal.copy()
    ctx_halide = raw_halide & allowed_halide
    ctx_peptide = raw_peptide if policy.context.peptide.enabled else pd.Series(False, index=index)
    ctx_nuc = raw_nuc if policy.context.nucleic_acid.enabled else pd.Series(False, index=index)

    converged = False
    max_iterations = max(1, int(policy.context.max_iterations))
    for iteration in range(1, max_iterations + 1):
        partner_passes = _partner_passes(
            metadata_df,
            ctx_small_molecule=ctx_small_molecule,
            ctx_metal=ctx_metal,
            ctx_halide=ctx_halide,
            ctx_peptide=ctx_peptide,
            ctx_nuc=ctx_nuc,
        )
        next_small_molecule = _context_small_molecule_mask(
            metadata_df,
            small_molecule_edges,
            partner_passes=partner_passes,
            policy=policy.context.small_molecule,
        )
        next_metal = _context_metal_mask(
            metadata_df,
            metal_edges,
            partner_passes=partner_passes,
            policy=policy.context.metal,
        )
        next_halide = _context_halide_mask(
            metadata_df,
            halide_edges,
            partner_passes=partner_passes,
            allowed_halide=allowed_halide,
            policy=policy.context.halide,
        )

        deltas = {
            CONTEXT_SMALL_MOLECULE_COL: int((next_small_molecule != ctx_small_molecule).sum()),
            CONTEXT_METAL_COL: int((next_metal != ctx_metal).sum()),
            CONTEXT_HALIDE_COL: int((next_halide != ctx_halide).sum()),
        }
        ctx_small_molecule = next_small_molecule
        ctx_metal = next_metal
        ctx_halide = next_halide
        logger.info(
            "BML context fixed-point iteration %d: ctx_sm=%d ctx_metal=%d "
            "ctx_halide=%d deltas=%s.",
            iteration,
            int(ctx_small_molecule.sum()),
            int(ctx_metal.sum()),
            int(ctx_halide.sum()),
            deltas,
        )
        if not any(deltas.values()):
            converged = True
            break

    if not converged:
        logger.warning(
            "BML context fixed point did not converge after %d iterations; "
            "continuing with the last eligibility state.",
            max_iterations,
        )

    # Peptide and nucleic-acid context flags are raw modality flags gated only by
    # their config switch; they are included in the fixed-point partner pass but
    # do not themselves iterate.
    return {
        CONTEXT_SMALL_MOLECULE_COL: ctx_small_molecule,
        CONTEXT_METAL_COL: ctx_metal,
        CONTEXT_HALIDE_COL: ctx_halide,
        CONTEXT_PEPTIDE_COL: ctx_peptide,
        CONTEXT_NUCLEIC_ACID_CHAINS_COL: ctx_nuc,
    }


def _context_small_molecule_mask(
    metadata_df: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    partner_passes: pd.Series,
    policy: SmallMoleculeContextPolicy,
) -> pd.Series:
    contacts = _sum_edge_counts(
        edges,
        metadata_df.index,
        source_kind="small_molecule",
        partner_passes=partner_passes,
    )
    expected = pd.to_numeric(
        metadata_df["q_pn_unit_expected_heavy_atoms_non_polymer"],
        errors="coerce",
    )
    denom = expected.where(expected > 0, np.nan)
    ratio = contacts.astype(float) / denom
    mask = (
        _bool_col(metadata_df, "q_pn_unit_is_small_molecule")
        & (contacts >= policy.min_contacting_heavy_atoms)
    )
    if policy.min_contacting_heavy_atom_ratio is not None:
        mask = mask & (ratio.fillna(-np.inf) >= policy.min_contacting_heavy_atom_ratio)
    return mask.fillna(False).astype(bool)


def _context_metal_mask(
    metadata_df: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    partner_passes: pd.Series,
    policy: MetalContextPolicy,
) -> pd.Series:
    contacts = _sum_edge_counts(
        edges,
        metadata_df.index,
        source_kind="metal",
        partner_passes=partner_passes,
    )
    mask = (
        _bool_col(metadata_df, "q_pn_unit_is_metal")
        & (contacts >= policy.min_coordinating_atoms)
    )
    if policy.min_avg_occupancy_nonpolymer is not None:
        occupancy = pd.to_numeric(
            metadata_df["q_pn_unit_avg_occupancy_nonpolymer"],
            errors="coerce",
        )
        mask = mask & (occupancy.fillna(-np.inf) >= policy.min_avg_occupancy_nonpolymer)
    return mask.fillna(False).astype(bool)


def _context_halide_mask(
    metadata_df: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    partner_passes: pd.Series,
    allowed_halide: pd.Series,
    policy: HalideContextPolicy,
) -> pd.Series:
    contacts = _sum_edge_counts(
        edges,
        metadata_df.index,
        source_kind="halide",
        partner_passes=partner_passes,
    )
    mask = (
        _bool_col(metadata_df, "q_pn_unit_is_halide")
        & allowed_halide
        & (contacts >= policy.min_non_carbon_contacts)
    )
    if policy.min_avg_occupancy_nonpolymer is not None:
        occupancy = pd.to_numeric(
            metadata_df["q_pn_unit_avg_occupancy_nonpolymer"],
            errors="coerce",
        )
        mask = mask & (occupancy.fillna(-np.inf) >= policy.min_avg_occupancy_nonpolymer)
    return mask.fillna(False).astype(bool)


def _partner_passes(
    metadata_df: pd.DataFrame,
    *,
    ctx_small_molecule: pd.Series,
    ctx_metal: pd.Series,
    ctx_halide: pd.Series,
    ctx_peptide: pd.Series,
    ctx_nuc: pd.Series,
) -> pd.Series:
    return (
        _bool_col(metadata_df, "q_pn_unit_is_protein")
        | _bool_col(metadata_df, BML_CENTER_SMALL_MOLECULE_COL)
        | _bool_col(metadata_df, BML_CENTER_METAL_COL)
        | ctx_small_molecule
        | ctx_metal
        | ctx_halide
        | ctx_peptide
        | ctx_nuc
    ).fillna(False).astype(bool)


def _build_contact_edges(
    metadata_df: pd.DataFrame,
    column: str,
    pn_unit_kind: pd.Series,
    lookup: dict[tuple[str, str, str], Any],
) -> pd.DataFrame:
    records = []
    missing_partner_rows = 0
    if column not in metadata_df.columns:
        return pd.DataFrame(
            columns=["source_idx", "source_kind", "partner_idx", "partner_kind", "count"]
        )

    for source_idx, source in zip(metadata_df.index, metadata_df.itertuples(index=False)):
        source_kind = pn_unit_kind.loc[source_idx]
        contacts = parse_partner_list(getattr(source, column, None))
        for contact in contacts:
            if not isinstance(contact, dict):
                continue
            partner_indices = _resolve_partner_indices(
                source.pdb_id,
                source.assembly_id,
                contact.get("pn_unit_iid"),
                contact.get("chain_iid"),
                lookup,
            )
            if not partner_indices:
                missing_partner_rows += 1
                continue
            count = contact_count(contact)
            for partner_idx in partner_indices:
                records.append(
                    {
                        "source_idx": source_idx,
                        "source_kind": source_kind,
                        "partner_idx": partner_idx,
                        "partner_kind": pn_unit_kind.loc[partner_idx],
                        "count": count,
                    }
                )
    if missing_partner_rows:
        logger.info(
            "Skipped %d %s contacts whose partner pn_unit row was not present.",
            missing_partner_rows,
            column,
        )
    return pd.DataFrame.from_records(
        records,
        columns=["source_idx", "source_kind", "partner_idx", "partner_kind", "count"],
    )


def _sum_edge_counts(
    edges: pd.DataFrame,
    index: pd.Index,
    *,
    source_kind: str,
    partner_passes: pd.Series,
) -> pd.Series:
    if edges.empty:
        return pd.Series(0, index=index, dtype=int)
    mask = edges["source_kind"].eq(source_kind)
    mask = mask & edges["partner_idx"].map(partner_passes).fillna(False).astype(bool)
    if not mask.any():
        return pd.Series(0, index=index, dtype=int)
    return (
        edges.loc[mask]
        .groupby("source_idx")["count"]
        .sum()
        .reindex(index, fill_value=0)
        .astype(int)
    )


def _pn_unit_kind(metadata_df: pd.DataFrame) -> pd.Series:
    kind = pd.Series("other", index=metadata_df.index, dtype=object)
    kind.loc[_bool_col(metadata_df, "q_pn_unit_is_protein")] = "protein"
    kind.loc[_bool_col(metadata_df, "q_pn_unit_is_nuc")] = "nucleic_acid"
    kind.loc[_bool_col(metadata_df, "q_pn_unit_is_peptide")] = "peptide"
    kind.loc[_bool_col(metadata_df, "q_pn_unit_is_small_molecule")] = "small_molecule"
    kind.loc[_bool_col(metadata_df, "q_pn_unit_is_metal")] = "metal"
    kind.loc[_bool_col(metadata_df, "q_pn_unit_is_halide")] = "halide"
    return kind


def _resolve_partner_indices(
    pdb_id: str,
    assembly_id,
    raw_iid,
    chain_iid,
    lookup: dict[tuple[str, str, str], Any],
) -> list[Any]:
    candidates = []
    for value in (raw_iid, chain_iid):
        if value is not None:
            candidates.append(str(value))

    seen_candidates = set()
    for candidate in candidates:
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        key = (str(pdb_id), str(assembly_id), candidate)
        if key in lookup:
            return [lookup[key]]

    rows = []
    seen_rows = set()
    for candidate in candidates:
        for iid in split_components(candidate):
            key = (str(pdb_id), str(assembly_id), iid)
            if iid in seen_rows or key not in lookup:
                continue
            seen_rows.add(iid)
            rows.append(lookup[key])
    return rows


def _row_is_expandable_context(row: pd.Series) -> bool:
    return (
        _truthy(row.get("q_pn_unit_is_protein", False))
        or _truthy(row.get(BML_CENTER_SMALL_MOLECULE_COL, False))
        or _truthy(row.get(BML_CENTER_METAL_COL, False))
        or _truthy(row.get(CONTEXT_SMALL_MOLECULE_COL, False))
        or _truthy(row.get(CONTEXT_METAL_COL, False))
        or _truthy(row.get(CONTEXT_HALIDE_COL, False))
        or _truthy(row.get(CONTEXT_PEPTIDE_COL, False))
        or _truthy(row.get(CONTEXT_NUCLEIC_ACID_CHAINS_COL, False))
    )


def _bool_col(metadata_df: pd.DataFrame, col: str) -> pd.Series:
    return metadata_df.get(col, pd.Series(False, index=metadata_df.index)).fillna(False).astype(bool)


def _require_columns(metadata_df: pd.DataFrame, columns: list[str], *, context: str) -> None:
    missing = [col for col in columns if col not in metadata_df.columns]
    if missing:
        raise KeyError(f"{context} requires metadata columns that are missing: {missing}")


def _optional_list(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in list(value)]


def _optional_str(value) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _truthy(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return False
    if value is pd.NA:
        return False
    return bool(value)
