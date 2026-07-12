import numpy as np
import pandas as pd
import pytest

from allatom_design.data.datasets.atomworks_sd.sampling import add_sampling_weights


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    monomer_df = pd.DataFrame(
        {
            "example_id": ["ma1", "ma2", "mb1", "mc1"],
            "q_pn_unit_cluster_id": ["A", "A", "B", "C"],
            "n_prot": [1, 1, 1, 1],
        }
    )
    interface_df = pd.DataFrame(
        {
            "example_id": ["ia-sm", "ia-metal", "ib-sm"],
            "interface_type": [
                "bmsm_protein",
                "bmm_protein",
                "bmsm_protein",
            ],
            "protein_cluster_multiset": [("A",), ("A",), ("B",)],
            "n_prot": [1, 1, 1],
            "ligand_ccd_key": [
                ("ccd", "LIGA"),
                ("ccd", "MG"),
                ("ccd", "LIGB"),
            ],
            "q_pn_unit_non_polymer_res_names": ["LIGA", "MG", "LIGB"],
        }
    )
    return monomer_df, interface_df


def _alphas(
    *,
    scale: float = 1.0,
    small_molecule: float = 1.0,
    metal: float = 1.0,
) -> dict[str, float]:
    return {
        "alpha_protein_metal": scale * metal,
        "alpha_protein_small_molecule": scale * small_molecule,
        "alpha_protein_nuc_lig": scale,
        "alpha_protein_peptide": scale,
        "alpha_protein_protein": scale,
    }


def _interface_fraction_cfg(target_fraction_of_max) -> dict:
    return {
        "sampling_scheme": "interface_fraction",
        "interface_fraction_sampling_weights": {
            "target_fraction_of_max": target_fraction_of_max,
        },
    }


def _run_interface_fraction(
    *,
    target_fraction_of_max: float = 0.6,
    alphas: dict[str, float] | None = None,
    monomer_df: pd.DataFrame | None = None,
    interface_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if monomer_df is None or interface_df is None:
        default_monomer, default_interface = _frames()
        monomer_df = default_monomer if monomer_df is None else monomer_df
        interface_df = default_interface if interface_df is None else interface_df
    return add_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface=_alphas() if alphas is None else alphas,
        clustering_cfg=_interface_fraction_cfg(target_fraction_of_max),
    )


def _global_interface_fraction(
    monomer_df: pd.DataFrame,
    interface_df: pd.DataFrame,
) -> float:
    interface_mass = interface_df["sampling_weight"].sum()
    return interface_mass / (interface_mass + monomer_df["sampling_weight"].sum())


def _interface_mass_for_cluster(interface_df: pd.DataFrame, cluster_id: str) -> float:
    mask = interface_df["protein_cluster_multiset"].apply(
        lambda clusters: tuple(clusters)[0] == cluster_id
    )
    return interface_df.loc[mask, "sampling_weight"].sum()


def test_absent_legacy_and_percentile_profiles_are_exactly_equal():
    monomer_df, interface_df = _frames()
    kwargs = {
        "alphas_interface": _alphas(),
        "k_percentile": 80.0,
        "single_protein_context_weight": 0.7,
        "multi_protein_context_weight": 1.3,
    }

    absent = add_sampling_weights(monomer_df, interface_df, **kwargs)
    legacy = add_sampling_weights(
        monomer_df,
        interface_df,
        clustering_cfg={"sampling_scheme": "legacy"},
        **kwargs,
    )
    percentile = add_sampling_weights(
        monomer_df,
        interface_df,
        clustering_cfg={"sampling_scheme": "percentile"},
        **kwargs,
    )

    for expected, actual in ((absent, legacy), (absent, percentile)):
        pd.testing.assert_frame_equal(expected[0], actual[0])
        pd.testing.assert_frame_equal(expected[1], actual[1])


def test_interface_fraction_hits_derived_target_and_equalizes_combined_mass():
    target_fraction_of_max = 0.6
    maximum_feasible_fraction = 2.0 / 3.0
    expected_global_fraction = (
        target_fraction_of_max * maximum_feasible_fraction
    )
    monomer_df, interface_df = _run_interface_fraction(
        target_fraction_of_max=target_fraction_of_max
    )

    assert _global_interface_fraction(monomer_df, interface_df) == pytest.approx(
        expected_global_fraction,
        abs=1e-12,
    )
    for cluster_id in ("A", "B", "C"):
        monomer_mass = monomer_df.loc[
            monomer_df["q_pn_unit_cluster_id"] == cluster_id,
            "sampling_weight",
        ].sum()
        interface_mass = _interface_mass_for_cluster(interface_df, cluster_id)
        assert monomer_mass + interface_mass == pytest.approx(1.0, abs=1e-12)


def test_interface_alpha_selectively_changes_preference_at_fixed_global_target():
    _, baseline = _run_interface_fraction(alphas=_alphas(metal=1.0))
    _, metal_upweighted = _run_interface_fraction(alphas=_alphas(metal=4.0))

    baseline_by_id = baseline.set_index("example_id")["sampling_weight"]
    upweighted_by_id = metal_upweighted.set_index("example_id")["sampling_weight"]
    assert baseline_by_id["ia-metal"] / baseline_by_id["ia-sm"] == pytest.approx(
        1.0
    )
    assert upweighted_by_id["ia-metal"] / upweighted_by_id[
        "ia-sm"
    ] == pytest.approx(4.0)
    assert _interface_mass_for_cluster(
        metal_upweighted, "A"
    ) > _interface_mass_for_cluster(baseline, "A")
    assert _interface_mass_for_cluster(
        metal_upweighted, "B"
    ) < _interface_mass_for_cluster(baseline, "B")


def test_common_alpha_scaling_cancels():
    baseline_monomer, baseline_interface = _run_interface_fraction(
        alphas=_alphas(scale=1.0)
    )
    scaled_monomer, scaled_interface = _run_interface_fraction(
        alphas=_alphas(scale=11.0)
    )

    assert scaled_monomer["sampling_weight"].tolist() == pytest.approx(
        baseline_monomer["sampling_weight"].tolist(),
        abs=1e-12,
    )
    assert scaled_interface["sampling_weight"].tolist() == pytest.approx(
        baseline_interface["sampling_weight"].tolist(),
        abs=1e-12,
    )


def test_common_finite_alpha_scaling_near_float_max_cancels():
    baseline_monomer, baseline_interface = _run_interface_fraction(
        alphas=_alphas(scale=1.0)
    )
    scaled_monomer, scaled_interface = _run_interface_fraction(
        alphas=_alphas(scale=1e308)
    )

    assert scaled_monomer["sampling_weight"].tolist() == pytest.approx(
        baseline_monomer["sampling_weight"].tolist(),
        rel=8 * np.finfo(float).eps,
        abs=0.0,
    )
    assert scaled_interface["sampling_weight"].tolist() == pytest.approx(
        baseline_interface["sampling_weight"].tolist(),
        rel=8 * np.finfo(float).eps,
        abs=0.0,
    )
    assert np.isfinite(scaled_interface["alpha"]).all()
    assert scaled_interface["alpha"].max() == 1e308


@pytest.mark.parametrize("bad_alpha", [float("nan"), float("inf"), -1.0])
def test_interface_fraction_rejects_invalid_configured_alpha(bad_alpha):
    alphas = _alphas()
    alphas["alpha_protein_metal"] = bad_alpha

    with pytest.raises(ValueError, match="configured alphas must be finite"):
        _run_interface_fraction(alphas=alphas)


def test_positive_extreme_alpha_ratio_remains_feasible():
    monomer_df, interface_df = _frames()
    interface_df = interface_df.loc[
        interface_df["example_id"].isin(["ia-sm", "ib-sm"])
    ].copy()
    interface_df.loc[
        interface_df["example_id"] == "ib-sm", "interface_type"
    ] = "bmm_protein"

    monomer_df, interface_df = _run_interface_fraction(
        target_fraction_of_max=0.6,
        alphas=_alphas(metal=1e-300),
        monomer_df=monomer_df,
        interface_df=interface_df,
    )

    assert _global_interface_fraction(monomer_df, interface_df) == pytest.approx(
        0.4,
        abs=1e-12,
    )
    assert _interface_mass_for_cluster(interface_df, "B") > 0


def test_unused_categorical_levels_do_not_create_phantom_protein_clusters():
    monomer_df, interface_df = _frames()
    monomer_df["q_pn_unit_cluster_id"] = pd.Categorical(
        monomer_df["q_pn_unit_cluster_id"],
        categories=["A", "B", "C", "unused-D", "unused-E"],
    )

    weighted_monomer, weighted_interface = _run_interface_fraction(
        target_fraction_of_max=0.6,
        monomer_df=monomer_df,
        interface_df=interface_df,
    )

    assert _global_interface_fraction(
        weighted_monomer, weighted_interface
    ) == pytest.approx(0.4, rel=8 * np.finfo(float).eps, abs=0.0)
    assert (
        weighted_monomer["sampling_weight"].sum()
        + weighted_interface["sampling_weight"].sum()
    ) == pytest.approx(3.0)
    for cluster_id in ("A", "B", "C"):
        monomer_mass = weighted_monomer.loc[
            weighted_monomer["q_pn_unit_cluster_id"] == cluster_id,
            "sampling_weight",
        ].sum()
        interface_mass = _interface_mass_for_cluster(
            weighted_interface,
            cluster_id,
        )
        assert monomer_mass + interface_mass == pytest.approx(1.0)

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        _run_interface_fraction(
            target_fraction_of_max=1.0,
            monomer_df=monomer_df,
            interface_df=interface_df,
        )


def test_tiny_positive_fraction_of_max_uses_relative_machine_precision():
    target_fraction_of_max = 1e-15
    expected_global_fraction = target_fraction_of_max * (2.0 / 3.0)
    monomer_df, interface_df = _run_interface_fraction(
        target_fraction_of_max=target_fraction_of_max
    )
    achieved = _global_interface_fraction(monomer_df, interface_df)

    assert abs(achieved - expected_global_fraction) <= 16 * np.spacing(
        expected_global_fraction
    )


def test_representable_near_one_fraction_of_max_does_not_saturate():
    target_fraction_of_max = np.nextafter(1.0, 0.0)
    maximum_feasible_fraction = 2.0 / 3.0
    expected_global_fraction = (
        target_fraction_of_max * maximum_feasible_fraction
    )

    monomer_df, interface_df = _run_interface_fraction(
        target_fraction_of_max=target_fraction_of_max
    )
    achieved = _global_interface_fraction(monomer_df, interface_df)

    assert achieved < maximum_feasible_fraction
    assert abs(achieved - expected_global_fraction) <= 8 * np.spacing(
        expected_global_fraction
    )


def test_duplicate_rows_preserve_pair_and_protein_total_mass():
    monomer_df, interface_df = _frames()
    original_monomer, original_interface = _run_interface_fraction(
        monomer_df=monomer_df,
        interface_df=interface_df,
    )
    duplicate = interface_df.iloc[[0]].assign(example_id="ia-sm-duplicate")
    duplicated_interface_df = pd.concat(
        [interface_df, duplicate],
        ignore_index=True,
    )
    duplicated_monomer, duplicated_interface = _run_interface_fraction(
        monomer_df=monomer_df,
        interface_df=duplicated_interface_df,
    )

    assert duplicated_monomer["sampling_weight"].tolist() == pytest.approx(
        original_monomer["sampling_weight"].tolist(),
        abs=1e-12,
    )
    original_pair_mass = original_interface.groupby("pair_cluster")[
        "sampling_weight"
    ].sum()
    duplicated_pair_mass = duplicated_interface.groupby("pair_cluster")[
        "sampling_weight"
    ].sum()
    pd.testing.assert_series_equal(
        original_pair_mass,
        duplicated_pair_mass,
        check_exact=False,
        atol=1e-12,
        rtol=0.0,
    )
    assert duplicated_interface.loc[
        duplicated_interface["example_id"].isin(["ia-sm", "ia-sm-duplicate"]),
        "pair_cluster_size",
    ].tolist() == [2, 2]


def test_individual_zero_alpha_is_legal():
    monomer_df, interface_df = _run_interface_fraction(
        alphas=_alphas(metal=0.0)
    )

    metal_weight = interface_df.loc[
        interface_df["example_id"] == "ia-metal",
        "sampling_weight",
    ].item()
    assert metal_weight == 0.0
    assert _global_interface_fraction(monomer_df, interface_df) == pytest.approx(
        0.4,
        abs=1e-12,
    )


def test_zero_alpha_reduces_maximum_before_target_derivation():
    monomer_df, interface_df = _run_interface_fraction(
        target_fraction_of_max=0.6,
        alphas=_alphas(small_molecule=0.0, metal=1.0),
    )

    # Only cluster A retains positive interface mass, so max_feasible=1/3.
    assert _global_interface_fraction(monomer_df, interface_df) == pytest.approx(
        0.6 * (1.0 / 3.0),
        abs=1e-12,
    )
    assert interface_df.loc[
        interface_df["interface_type"] == "bmsm_protein",
        "sampling_weight",
    ].eq(0.0).all()


@pytest.mark.parametrize(
    "clustering_cfg",
    [
        {"sampling_scheme": "interface_fraction"},
        {
            "sampling_scheme": "interface_fraction",
            "interface_fraction_sampling_weights": {
                "target_fraction_of_max": None,
            },
        },
        {
            "sampling_scheme": "interface_fraction",
            "interface_fraction_sampling_weights": {
                "target_interface_fraction": 0.4,
            },
        },
    ],
)
def test_interface_fraction_requires_explicit_target(clustering_cfg):
    monomer_df, interface_df = _frames()

    with pytest.raises(ValueError, match="target_fraction_of_max.*max_feasible"):
        add_sampling_weights(
            monomer_df,
            interface_df,
            alphas_interface=_alphas(),
            clustering_cfg=clustering_cfg,
        )


@pytest.mark.parametrize(
    "target_fraction_of_max",
    [0.0, -0.1, float("nan"), float("inf"), 1.0, 1.1],
)
def test_interface_fraction_rejects_invalid_fraction_of_max(
    target_fraction_of_max,
):
    monomer_df, interface_df = _frames()

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        add_sampling_weights(
            monomer_df,
            interface_df,
            alphas_interface=_alphas(),
            clustering_cfg=_interface_fraction_cfg(target_fraction_of_max),
        )


def test_interface_fraction_rejects_underflowed_derived_global_target():
    monomer_df, interface_df = _frames()
    interface_df = interface_df.loc[
        interface_df["example_id"] == "ia-metal"
    ].copy()

    with pytest.raises(ValueError, match="derived global interface target"):
        _run_interface_fraction(
            target_fraction_of_max=np.nextafter(0.0, 1.0),
            monomer_df=monomer_df,
            interface_df=interface_df,
        )


def test_interface_fraction_rejects_multi_protein_cluster_multiset():
    monomer_df, interface_df = _frames()
    interface_df.at[0, "protein_cluster_multiset"] = ("A", "B")
    interface_df.at[0, "n_prot"] = 2

    with pytest.raises(ValueError, match="exactly one protein cluster"):
        _run_interface_fraction(
            monomer_df=monomer_df,
            interface_df=interface_df,
        )


def test_interface_fraction_rejects_n_prot_mismatch():
    monomer_df, interface_df = _frames()
    interface_df.at[0, "n_prot"] = 2

    with pytest.raises(ValueError, match="exactly one protein cluster"):
        _run_interface_fraction(
            monomer_df=monomer_df,
            interface_df=interface_df,
        )


def test_interface_fraction_rejects_interface_without_monomer_cluster():
    monomer_df, interface_df = _frames()
    monomer_df = monomer_df.loc[
        monomer_df["q_pn_unit_cluster_id"] != "B"
    ].copy()

    with pytest.raises(ValueError, match="corresponding monomer cluster.*B"):
        _run_interface_fraction(
            monomer_df=monomer_df,
            interface_df=interface_df,
        )


def test_interface_fraction_rejects_all_zero_effective_interface_mass():
    monomer_df, interface_df = _frames()

    with pytest.raises(ValueError, match="zero effective interface base mass"):
        _run_interface_fraction(
            alphas=_alphas(small_molecule=0.0, metal=0.0),
            monomer_df=monomer_df,
            interface_df=interface_df,
        )


def test_unknown_sampling_scheme_is_rejected():
    monomer_df, interface_df = _frames()

    with pytest.raises(ValueError, match="Unknown `clustering.sampling_scheme`"):
        add_sampling_weights(
            monomer_df,
            interface_df,
            alphas_interface=_alphas(),
            clustering_cfg={"sampling_scheme": "not-a-scheme"},
        )


def test_interface_fraction_rejects_non_per_center_grouping():
    monomer_df, interface_df = _frames()
    clustering_cfg = _interface_fraction_cfg(0.6)
    clustering_cfg["interface_grouping_scheme"] = "maximal_center_clique"

    with pytest.raises(
        ValueError,
        match="interface_fraction.*requires.*per_center.*maximal_center_clique",
    ):
        add_sampling_weights(
            monomer_df,
            interface_df,
            alphas_interface=_alphas(),
            clustering_cfg=clustering_cfg,
        )


def test_ligand_grouped_scheme_still_dispatches_through_public_facade():
    monomer_df = pd.DataFrame(
        {
            "example_id": ["ma1", "ma2", "mb1"],
            "q_pn_unit_cluster_id": ["A", "A", "B"],
            "n_prot": [1, 1, 1],
        }
    )
    interface_df = pd.DataFrame(
        {
            "example_id": ["ia-sm", "ia-metal"],
            "protein_cluster_multiset": [("A",), ("A",)],
            "interface_cluster_key": [("sm", "A"), ("metal", "A")],
            "n_prot": [1, 1],
            "n_small_molecule": [1, 0],
            "n_metal": [0, 1],
            "n_peptide": [0, 0],
            "n_nuc_ligand": [0, 0],
            "n_nuc_polymer": [0, 0],
        }
    )
    grouped_cfg = {
        "sampling_scheme": "ligand_grouped_protein_equalized",
        "ligand_grouped_sampling_weights": {
            "beta_monomer": 1.0,
            "beta_interface": 1.0,
            "alpha_protein": 1.0,
            "alpha_small_molecule": 1.0,
            "alpha_metal": 1.0,
            "alpha_peptide": 1.0,
            "alpha_nuc_ligand": 1.0,
            "alpha_nuc_polymer": 1.0,
        },
    }

    monomer_df, interface_df = add_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface={},
        clustering_cfg=grouped_cfg,
    )

    mass_a = monomer_df.loc[
        monomer_df["q_pn_unit_cluster_id"] == "A", "sampling_weight"
    ].sum() + interface_df["sampling_weight"].sum()
    mass_b = monomer_df.loc[
        monomer_df["q_pn_unit_cluster_id"] == "B", "sampling_weight"
    ].sum()
    assert mass_a == pytest.approx(1.0)
    assert mass_b == pytest.approx(1.0)
