"""SD-dataset center-selection masks.

cfg-driven row selectors that decide which pn_units qualify as metal /
small-molecule / peptide / nucleic-acid-ligand interface centers, applying the
dataset's curation policy (external-evidence policy, occupancy and missing-atom
thresholds). The cfg-free parsing/CCD primitives they build on live in
:mod:`allatom_design.data.utils.pn_unit`.
"""

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from allatom_design.data.utils.pn_unit import (
    _normalize_ccd_codes,
    parse_partner_list,
    parse_pn_unit_iids_value,
    series_has_any_exact_ccd,
)


def _get_allowed_ccd_codes(allowed_ccd_codes: list[str] | None) -> set[str]:
    code_set = _normalize_ccd_codes(allowed_ccd_codes)
    if allowed_ccd_codes is not None and not code_set:
        raise ValueError("`allowed_ccd_codes` must be null or contain at least one exact CCD code.")
    return code_set


def metal_center_mask(metadata_df: pd.DataFrame, cfg: dict | DictConfig) -> pd.Series:
    cfg = cfg or {}
    allowed_codes = _get_allowed_ccd_codes(cfg.get("allowed_ccd_codes", None))
    min_occupancy = float(cfg.get("min_avg_occupancy_nonpolymer", 0.5))
    evidence_policy = cfg.get(
        "metal_external_evidence_policy",
        cfg.get("external_evidence_policy", "no_filter"),
    )

    mask = (
        metadata_df["q_pn_unit_is_metal"].fillna(False).astype(bool)
        & series_has_any_exact_ccd(
            metadata_df["q_pn_unit_non_polymer_res_names"],
            allowed_codes,
            index=metadata_df.index,
        )
        & (metadata_df["q_pn_unit_avg_occupancy_nonpolymer"].fillna(-np.inf) >= min_occupancy)
    )
    if evidence_policy == "external_evidence":
        mask = mask & metadata_df["q_pn_unit_has_external_evidence"].fillna(False).astype(bool)
    elif evidence_policy != "no_filter":
        raise ValueError(
            "Unknown `metal_external_evidence_policy` "
            f"{evidence_policy!r}; expected 'no_filter' or 'external_evidence'."
        )
    return mask


def small_molecule_center_mask(metadata_df: pd.DataFrame, cfg: dict | DictConfig) -> pd.Series:
    cfg = cfg or {}
    min_contacts = cfg.get("min_contacting_protein_atoms_small_molecule", 20)
    min_contact_ratio = cfg.get("min_contacting_protein_atom_ratio_small_molecule", None)
    min_occupancy = cfg.get("min_avg_occupancy_nonpolymer_small_molecule", 0.5)
    max_missing_fraction = cfg.get("max_missing_atom_fraction_small_molecule", 0.2)
    evidence_policy = cfg.get(
        "small_molecule_artifact_external_evidence_policy",
        cfg.get("small_molecule_artifact_external_evidence_policy", "no_filter"),
    )

    is_small_molecule = metadata_df["q_pn_unit_is_small_molecule"].fillna(False).astype(bool)
    is_artifact = metadata_df.get(
        "q_pn_unit_is_artifact",
        pd.Series(False, index=metadata_df.index),
    ).fillna(False).astype(bool)
    has_evidence = metadata_df.get(
        "q_pn_unit_has_external_evidence",
        pd.Series(False, index=metadata_df.index),
    ).fillna(False).astype(bool)

    if evidence_policy == "no_filter":
        mask = is_small_molecule
    elif evidence_policy == "filter_all_artifacts":
        mask = is_small_molecule & ~is_artifact
    elif evidence_policy == "external_evidence":
        mask = is_small_molecule | (has_evidence & ~is_artifact)
    else:
        raise ValueError(
            "Unknown `small_molecule_artifact_external_evidence_policy` "
            f"{evidence_policy!r}; expected 'no_filter', "
            "'filter_all_artifacts', or 'external_evidence'."
        )

    # Apply quality thresholds
    if min_contacts is not None:
        mask = mask & (_small_molecule_contacting_protein_atoms(metadata_df) >= int(min_contacts))
    if min_contact_ratio is not None:
        mask = mask & (
            _small_molecule_contacting_protein_atom_ratio(metadata_df).fillna(-np.inf)
            >= float(min_contact_ratio)
        )
    if min_occupancy is not None:
        mask = mask & (
            metadata_df["q_pn_unit_avg_occupancy_nonpolymer"].fillna(-np.inf) >= float(min_occupancy)
        )
    if max_missing_fraction is not None:
        mask = mask & (
            _small_molecule_missing_atom_fraction(metadata_df).fillna(np.inf)
            <= float(max_missing_fraction)
        )
    return mask


def _small_molecule_contacting_protein_atoms(metadata_df: pd.DataFrame) -> pd.Series:
    return metadata_df["q_pn_unit_per_partner_contacts_to_protein_small_molecule"].apply(_sum_contact_counts)


def _small_molecule_contacting_protein_atom_ratio(metadata_df: pd.DataFrame) -> pd.Series:
    contacts = _small_molecule_contacting_protein_atoms(metadata_df).astype(float)
    expected = pd.to_numeric(metadata_df["q_pn_unit_expected_heavy_atoms_non_polymer"], errors="coerce")
    denom = expected.where(expected > 0, np.nan)
    return contacts / denom


def _sum_contact_counts(value) -> int:
    total = 0
    for contact in parse_partner_list(value):
        if not isinstance(contact, dict):
            continue
        try:
            total += int(contact.get("count", 0))
        except (TypeError, ValueError):
            continue
    return total


def _small_molecule_missing_atom_fraction(metadata_df: pd.DataFrame) -> pd.Series:
    resolved = pd.to_numeric(metadata_df["q_pn_unit_num_resolved_atoms"], errors="coerce")
    expected = pd.to_numeric(metadata_df["q_pn_unit_expected_heavy_atoms_non_polymer"], errors="coerce")
    denom = expected.where(expected > 0, np.nan)
    return 1.0 - (resolved / denom)


def peptide_center_mask(metadata_df: pd.DataFrame, cfg: dict | DictConfig | None = None) -> pd.Series:
    return metadata_df.get(
        "q_pn_unit_is_peptide",
        pd.Series(False, index=metadata_df.index),
    ).fillna(False).astype(bool)


def nucleic_acid_ligand_center_mask(
    metadata_df: pd.DataFrame,
    cfg: dict | DictConfig | None = None,
) -> pd.Series:
    return (
        metadata_df.get("q_pn_unit_is_nuc_ligand", pd.Series(False, index=metadata_df.index))
        .fillna(False)
        .astype(bool)
        & metadata_df.get("q_pn_unit_nucleic_acid_group_id", pd.Series(pd.NA, index=metadata_df.index)).notna()
    )
