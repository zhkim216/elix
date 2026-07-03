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
    normalize_ligand_ccd_key,
    parse_partner_list,
    parse_pn_unit_iids_value,
    series_has_any_exact_ccd,
)


def _get_allowed_ccd_codes(allowed_ccd_codes: list[str] | None) -> set[str]:
    code_set = _normalize_ccd_codes(allowed_ccd_codes)
    if allowed_ccd_codes is not None and not code_set:
        raise ValueError("`allowed_ccd_codes` must be null or contain at least one exact CCD code.")
    return code_set


def _bool_column(
    metadata_df: pd.DataFrame,
    column: str,
    *,
    required: bool = False,
) -> pd.Series:
    if column in metadata_df.columns:
        return metadata_df[column].fillna(False).astype(bool)
    if required:
        raise KeyError(f"Metadata is missing required boolean column {column!r}.")
    return pd.Series(False, index=metadata_df.index)


def _policy_values(policy, *, policy_name: str = "external_evidence_policy") -> list[str]:
    if policy is None:
        return ["no_filter"]
    if isinstance(policy, str):
        if policy == "":
            return ["no_filter"]
        return [policy]
    try:
        values = [str(value) for value in policy]
    except TypeError as exc:
        raise ValueError(
            f"`{policy_name}` must be 'no_filter' or a non-empty list of policy names."
        ) from exc
    if not values:
        raise ValueError(f"`{policy_name}` must be 'no_filter' or a non-empty list.")
    if "no_filter" in values and len(values) > 1:
        raise ValueError(f"`{policy_name}` cannot combine 'no_filter' with evidence policies.")
    return values


def _validate_evidence_policies(
    evidence_policies: list[str],
    *,
    policy_name: str,
    allowed_policies: set[str],
) -> None:
    if evidence_policies == ["no_filter"]:
        return
    unknown = [policy for policy in evidence_policies if policy not in allowed_policies]
    if unknown:
        allowed = "', '".join(sorted(allowed_policies | {"no_filter"}))
        raise ValueError(
            f"Unknown `{policy_name}` value(s): {unknown}; expected one of '{allowed}'."
        )


def _metal_external_evidence_mask(
    metadata_df: pd.DataFrame,
    evidence_policies: list[str],
) -> pd.Series:
    _validate_evidence_policies(
        evidence_policies,
        policy_name="metal.external_evidence_policy",
        allowed_policies={"medba", "pubmed"},
    )
    if evidence_policies == ["no_filter"]:
        return pd.Series(True, index=metadata_df.index)

    selected = set(evidence_policies)
    metalsite = _bool_column(
        metadata_df,
        "q_pn_unit_has_medba_metalsite_evidence",
        required=True,
    )
    component_identity = (
        _bool_column(
            metadata_df,
            "q_pn_unit_has_medba_component_identity_evidence",
            required=True,
        )
        if "medba" in selected
        else pd.Series(False, index=metadata_df.index)
    )
    pubmed_identity = (
        _bool_column(
            metadata_df,
            "q_pn_unit_has_metal_pubmed_evidence",
            required=True,
        )
        if "pubmed" in selected
        else pd.Series(False, index=metadata_df.index)
    )
    is_metal = metadata_df["q_pn_unit_is_metal"].fillna(False).astype(bool)
    ccd_key = metadata_df["q_pn_unit_non_polymer_res_names"].map(
        lambda value: normalize_ligand_ccd_key(value)[1]
    )
    pdb_id = metadata_df["pdb_id"].astype(str)
    pdb_ccd_has_metalsite = pd.Series(False, index=metadata_df.index)
    if is_metal.any():
        pdb_ccd_has_metalsite.loc[is_metal] = (
            metalsite.loc[is_metal]
            .groupby([pdb_id.loc[is_metal], ccd_key.loc[is_metal]], sort=False)
            .transform("any")
            .astype(bool)
        )

    fallback_identity = component_identity | pubmed_identity
    exact_metalsite = metalsite if "medba" in selected else pd.Series(False, index=metadata_df.index)
    return (exact_metalsite | (~pdb_ccd_has_metalsite & fallback_identity)).astype(bool)


def metal_center_mask(metadata_df: pd.DataFrame, cfg: dict | DictConfig) -> pd.Series:
    cfg = cfg or {}
    if "bml_center" in cfg:
        cfg = {
            "external_evidence_policy": cfg.get("bml_center", {}).get("metal", {}).get(
                "external_evidence_policy",
                "no_filter",
            ),
            "allowed_ccd_codes": cfg.get("bml_center", {}).get("metal", {}).get(
                "allowed_ccd_codes",
                None,
            ),
            "min_avg_occupancy_nonpolymer": cfg.get("bml_center", {}).get("metal", {}).get(
                "min_avg_occupancy_nonpolymer",
                0.5,
            ),
        }
    allowed_codes = _get_allowed_ccd_codes(cfg.get("allowed_ccd_codes", None))
    min_occupancy = float(cfg.get("min_avg_occupancy_nonpolymer", 0.5))
    evidence_policy = cfg.get("external_evidence_policy", "no_filter")
    evidence_policies = _policy_values(
        evidence_policy,
        policy_name="metal.external_evidence_policy",
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
    if evidence_policies != ["no_filter"]:
        mask = mask & _metal_external_evidence_mask(metadata_df, evidence_policies)
    return mask


def small_molecule_center_mask(metadata_df: pd.DataFrame, cfg: dict | DictConfig) -> pd.Series:
    cfg = cfg or {}
    if "bml_center" in cfg:
        small_molecule_cfg = cfg.get("bml_center", {}).get("small_molecule", {})
        cfg = {
            "external_evidence_policy": small_molecule_cfg.get(
                "external_evidence_policy",
                "no_filter",
            ),
            "min_contacting_protein_atoms_small_molecule": small_molecule_cfg.get(
                "min_contacting_protein_atoms",
                20,
            ),
            "min_contacting_protein_atom_ratio_small_molecule": small_molecule_cfg.get(
                "min_contacting_protein_atom_ratio",
                None,
            ),
            "min_avg_occupancy_nonpolymer_small_molecule": small_molecule_cfg.get(
                "min_avg_occupancy_nonpolymer",
                0.5,
            ),
            "max_missing_atom_fraction_small_molecule": small_molecule_cfg.get(
                "max_missing_atom_fraction",
                0.2,
            ),
            "exclude_maybe_covalently_linked_small_molecules": small_molecule_cfg.get(
                "exclude_maybe_covalently_linked_small_molecules",
                False,
            ),
        }
    min_contacts = cfg.get("min_contacting_protein_atoms_small_molecule", 20)
    min_contact_ratio = cfg.get("min_contacting_protein_atom_ratio_small_molecule", None)
    min_occupancy = cfg.get("min_avg_occupancy_nonpolymer_small_molecule", 0.5)
    max_missing_fraction = cfg.get("max_missing_atom_fraction_small_molecule", 0.2)
    external_evidence_policy = cfg.get("external_evidence_policy", "no_filter")
    evidence_policies = _policy_values(
        external_evidence_policy,
        policy_name="small_molecule.external_evidence_policy",
    )
    _validate_evidence_policies(
        evidence_policies,
        policy_name="small_molecule.external_evidence_policy",
        allowed_policies={"klifs", "prd"},
    )

    is_small_molecule = metadata_df["q_pn_unit_is_small_molecule"].fillna(False).astype(bool)

    if evidence_policies == ["no_filter"]:
        mask = is_small_molecule
    else:
        is_excluded_for_center = _bool_column(
            metadata_df,
            "q_pn_unit_non_polymer_ccd_code_is_excluded_for_center",
            required=True,
        )
        evidence_mask = pd.Series(False, index=metadata_df.index)
        for policy in evidence_policies:
            if policy == "klifs":
                evidence_mask = evidence_mask | _bool_column(
                    metadata_df,
                    "q_pn_unit_has_klifs_evidence",
                    required=True,
                )
            elif policy == "prd":
                evidence_mask = evidence_mask | _bool_column(
                    metadata_df,
                    "q_pn_unit_has_prd_evidence",
                    required=True,
                )
        mask = is_small_molecule & (~is_excluded_for_center | evidence_mask)

    if cfg.get("exclude_maybe_covalently_linked_small_molecules", False):
        mask = mask & ~_bool_column(
            metadata_df,
            "q_pn_unit_is_maybe_covalently_linked_to_protein",
            required=True,
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
    required_cols = (
        "q_pn_unit_is_nuc_ligand",
        "q_pn_unit_nucleic_acid_group_iids",
    )
    missing = [col for col in required_cols if col not in metadata_df.columns]
    if missing:
        raise KeyError(
            "Metadata is missing required precomputed nucleic-acid ligand "
            f"center columns: {missing}"
        )
    return (
        metadata_df["q_pn_unit_is_nuc_ligand"]
        .fillna(False)
        .astype(bool)
        & metadata_df["q_pn_unit_nucleic_acid_group_iids"].notna()
    )
