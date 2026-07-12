"""Exact-global-fraction sampling for single-protein interface rows."""

import logging

import numpy as np
import pandas as pd

from allatom_design.data.datasets.atomworks_sd.sampling_schemes._per_center import (
    _add_pair_cluster_columns,
    _protein_clusters,
    _resolve_interface_type_alphas,
)

logger = logging.getLogger("allatom_design.data.datasets.atomworks_sd.sampling")

_TARGET_FRACTION_OF_MAX_KEY = "target_fraction_of_max"
_TARGET_ULP_MULTIPLIER = 16.0
_MAX_ITERATIONS = 256


def _validate_configured_alphas(alphas_interface: dict[str, float]) -> None:
    bad_alphas = {}
    for alpha_key, raw_alpha in alphas_interface.items():
        try:
            alpha = float(raw_alpha)
        except (TypeError, ValueError, OverflowError):
            bad_alphas[alpha_key] = raw_alpha
            continue
        if not np.isfinite(alpha) or alpha < 0:
            bad_alphas[alpha_key] = raw_alpha
    if bad_alphas:
        raise ValueError(
            "Interface-fraction configured alphas must be finite and "
            f"non-negative; bad={bad_alphas}"
        )


def _target_tolerance(
    target_global_fraction: float,
    maximum_feasible_fraction: float,
) -> float:
    endpoint_margin = min(
        target_global_fraction,
        maximum_feasible_fraction - target_global_fraction,
    )
    return min(
        _TARGET_ULP_MULTIPLIER * float(np.spacing(target_global_fraction)),
        endpoint_margin / 4.0,
    )


def _add_interface_fraction_sampling_weights(
    *,
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
    alphas_interface: dict[str, float],
    cluster_col: str,
    single_protein_context_weight: float,
    cfg: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equalize protein mass at a fraction of the alpha-aware feasible maximum."""

    raw_target_fraction_of_max = (
        None if cfg is None else cfg.get(_TARGET_FRACTION_OF_MAX_KEY)
    )
    target_missing = (
        cfg is None
        or _TARGET_FRACTION_OF_MAX_KEY not in cfg
        or raw_target_fraction_of_max is None
    )
    try:
        target_fraction_of_max = float(raw_target_fraction_of_max)
    except (TypeError, ValueError):
        target_fraction_of_max = np.nan

    monomer_df = monomer_df.copy()
    interface_df = interface_df.copy()
    if cluster_col not in monomer_df.columns:
        raise KeyError(
            f"monomer_df is missing protein cluster column {cluster_col!r}."
        )
    if "protein_cluster_multiset" not in interface_df.columns:
        raise ValueError("interface_df must contain `protein_cluster_multiset`.")

    _validate_configured_alphas(alphas_interface)
    alpha_by_interface_type = _resolve_interface_type_alphas(alphas_interface)
    _add_pair_cluster_columns(interface_df, alpha_by_interface_type)
    interface_df["context_weight"] = float(single_protein_context_weight)
    if interface_df.empty:
        protein_clusters = pd.Series(index=interface_df.index, dtype=object)
    else:
        protein_clusters = interface_df.apply(_protein_clusters, axis=1)

    invalid_rows = protein_clusters.apply(len) != 1
    if "n_prot" in interface_df.columns:
        n_prot = pd.to_numeric(interface_df["n_prot"], errors="coerce")
        invalid_rows |= n_prot.isna() | (n_prot != 1)
    if invalid_rows.any():
        diagnostic_columns = [
            column
            for column in ("example_id", "n_prot", "protein_cluster_multiset")
            if column in interface_df.columns
        ]
        bad_examples = (
            interface_df.loc[invalid_rows, diagnostic_columns]
            .head(10)
            .to_dict("records")
        )
        raise ValueError(
            "Interface-fraction sampling requires exactly one protein cluster "
            f"per interface row; examples={bad_examples}"
        )

    interface_cluster_ids = [clusters[0] for clusters in protein_clusters]
    effective_alphas = interface_df["alpha"].to_numpy(dtype=float)
    if not np.isfinite(effective_alphas).all() or (effective_alphas < 0).any():
        raise ValueError(
            "Interface-fraction effective alphas must be finite "
            "and non-negative."
        )
    alpha_scale = float(effective_alphas.max()) if len(effective_alphas) else 0.0
    normalized_alphas = np.zeros_like(effective_alphas)
    if alpha_scale > 0:
        normalized_alphas = effective_alphas / alpha_scale
    interface_base_weights = (
        normalized_alphas
        / interface_df["pair_cluster_size"].to_numpy(dtype=float)
    )

    monomer_counts = {
        cluster_id: int(count)
        for cluster_id, count in monomer_df[cluster_col].value_counts().items()
        if count > 0
    }
    monomer_cluster_ids = list(monomer_counts)
    missing_monomer_clusters = sorted(
        set(interface_cluster_ids) - set(monomer_cluster_ids),
        key=repr,
    )
    if missing_monomer_clusters:
        raise ValueError(
            "Interface-fraction sampling requires a corresponding monomer cluster "
            f"for every interface cluster; missing={missing_monomer_clusters}"
        )

    interface_base_mass = {cluster_id: 0.0 for cluster_id in monomer_cluster_ids}
    for cluster_id, base_weight in zip(
        interface_cluster_ids,
        interface_base_weights,
    ):
        interface_base_mass[cluster_id] += float(base_weight)
    if any(
        not np.isfinite(base_mass) or base_mass < 0
        for base_mass in interface_base_mass.values()
    ):
        raise ValueError(
            "Interface-fraction per-protein base mass must be finite "
            "and non-negative."
        )

    positive_interface_clusters = [
        cluster_id
        for cluster_id in monomer_cluster_ids
        if interface_base_mass[cluster_id] > 0
    ]
    if not positive_interface_clusters:
        raise ValueError(
            "Interface-fraction sampling has zero effective interface base mass."
        )
    maximum_feasible_fraction = (
        len(positive_interface_clusters) / len(monomer_cluster_ids)
    )
    if (
        target_missing
        or not np.isfinite(target_fraction_of_max)
        or target_fraction_of_max <= 0
        or target_fraction_of_max >= 1
    ):
        raise ValueError(
            "Invalid `clustering.interface_fraction_sampling_weights."
            "target_fraction_of_max`: "
            f"value={raw_target_fraction_of_max!r}, "
            f"max_feasible={maximum_feasible_fraction:.17g}; "
            "value must be finite and strictly between 0 and 1."
        )
    target_global_fraction = (
        target_fraction_of_max * maximum_feasible_fraction
    )
    if (
        not np.isfinite(target_global_fraction)
        or target_global_fraction <= 0
        or target_global_fraction >= maximum_feasible_fraction
    ):
        raise ValueError(
            "The derived global interface target is not representable strictly "
            "between zero and max_feasible: "
            f"target_fraction_of_max={target_fraction_of_max:.17g}, "
            f"target_global={target_global_fraction:.17g}, "
            f"max_feasible={maximum_feasible_fraction:.17g}."
        )
    target_tolerance = _target_tolerance(
        target_global_fraction,
        maximum_feasible_fraction,
    )

    base_masses = np.array(
        [interface_base_mass[cluster_id] for cluster_id in monomer_cluster_ids],
        dtype=float,
    )
    log_base_masses = np.full_like(base_masses, -np.inf)
    positive_base_mass = base_masses > 0
    log_base_masses[positive_base_mass] = (
        np.log(base_masses[positive_base_mass]) - np.log(base_masses.max())
    )

    def _cluster_interface_fractions(log_scale: float) -> np.ndarray:
        # Bisect log(lambda) so every finite positive base mass remains
        # numerically reachable without overflowing lambda itself.
        logits = log_scale + log_base_masses
        fractions = np.empty_like(logits)
        non_negative = logits >= 0
        fractions[non_negative] = 1.0 / (
            1.0 + np.exp(-logits[non_negative])
        )
        exp_logits = np.exp(logits[~non_negative])
        fractions[~non_negative] = exp_logits / (1.0 + exp_logits)
        return fractions

    zero_log_scale_fraction = float(_cluster_interface_fractions(0.0).mean())
    if zero_log_scale_fraction < target_global_fraction:
        lower_log_scale = 0.0
        upper_log_scale = 1.0
        for _ in range(_MAX_ITERATIONS):
            if float(
                _cluster_interface_fractions(upper_log_scale).mean()
            ) >= target_global_fraction:
                break
            lower_log_scale = upper_log_scale
            upper_log_scale *= 2.0
        else:
            raise RuntimeError(
                "Failed to bracket the interface-fraction scale within "
                f"{_MAX_ITERATIONS} iterations."
            )
    else:
        lower_log_scale = -1.0
        upper_log_scale = 0.0
        for _ in range(_MAX_ITERATIONS):
            if float(
                _cluster_interface_fractions(lower_log_scale).mean()
            ) <= target_global_fraction:
                break
            upper_log_scale = lower_log_scale
            lower_log_scale *= 2.0
        else:
            raise RuntimeError(
                "Failed to bracket the interface-fraction scale within "
                f"{_MAX_ITERATIONS} iterations."
            )

    lower_fractions = _cluster_interface_fractions(lower_log_scale)
    upper_fractions = _cluster_interface_fractions(upper_log_scale)
    lower_fraction = float(lower_fractions.mean())
    upper_fraction = float(upper_fractions.mean())
    if abs(lower_fraction - target_global_fraction) <= abs(
        upper_fraction - target_global_fraction
    ):
        best_fractions = lower_fractions
        best_fraction = lower_fraction
    else:
        best_fractions = upper_fractions
        best_fraction = upper_fraction

    for _ in range(_MAX_ITERATIONS):
        log_scale = (lower_log_scale + upper_log_scale) / 2.0
        if log_scale == lower_log_scale or log_scale == upper_log_scale:
            break
        candidate_fractions = _cluster_interface_fractions(log_scale)
        candidate_fraction = float(candidate_fractions.mean())
        if abs(candidate_fraction - target_global_fraction) < abs(
            best_fraction - target_global_fraction
        ):
            best_fractions = candidate_fractions
            best_fraction = candidate_fraction
        if abs(best_fraction - target_global_fraction) <= target_tolerance:
            break
        if candidate_fraction < target_global_fraction:
            lower_log_scale = log_scale
        else:
            upper_log_scale = log_scale
    if abs(best_fraction - target_global_fraction) > target_tolerance:
        raise RuntimeError(
            "Interface-fraction scale did not converge to machine-aware "
            f"accuracy within {_MAX_ITERATIONS} iterations: "
            f"target_global={target_global_fraction:.17g}, "
            f"achieved={best_fraction:.17g}, tolerance={target_tolerance:.17g}."
        )

    fraction_by_cluster = dict(
        zip(monomer_cluster_ids, best_fractions)
    )
    monomer_df["sampling_weight"] = [
        (1.0 - fraction_by_cluster[cluster_id]) / monomer_counts[cluster_id]
        for cluster_id in monomer_df[cluster_col]
    ]
    interface_df["sampling_weight"] = [
        0.0
        if interface_base_mass[cluster_id] == 0
        else fraction_by_cluster[cluster_id]
        * base_weight
        / interface_base_mass[cluster_id]
        for cluster_id, base_weight in zip(
            interface_cluster_ids,
            interface_base_weights,
        )
    ]

    interface_mass = float(interface_df["sampling_weight"].sum())
    total_mass = interface_mass + float(monomer_df["sampling_weight"].sum())
    achieved_fraction = interface_mass / total_mass
    if (
        not np.isfinite(achieved_fraction)
        or achieved_fraction <= 0
        or achieved_fraction >= maximum_feasible_fraction
        or abs(achieved_fraction - target_global_fraction) > target_tolerance
    ):
        raise RuntimeError(
            "Interface-fraction sampling failed final target verification: "
            f"target_fraction_of_max={target_fraction_of_max:.17g}, "
            f"target_global={target_global_fraction:.17g}, "
            f"achieved={achieved_fraction:.17g}, "
            f"max_feasible={maximum_feasible_fraction:.17g}, "
            f"tolerance={target_tolerance:.17g}."
        )

    logger.info(
        "Interface-fraction sampling weights: monomer_rows=%d, "
        "interface_rows=%d, protein_clusters=%d, "
        "target_fraction_of_max=%.12f, max_feasible=%.12f, "
        "target_global=%.12f, achieved=%.12f",
        len(monomer_df),
        len(interface_df),
        len(monomer_cluster_ids),
        target_fraction_of_max,
        maximum_feasible_fraction,
        target_global_fraction,
        achieved_fraction,
    )
    return monomer_df, interface_df
