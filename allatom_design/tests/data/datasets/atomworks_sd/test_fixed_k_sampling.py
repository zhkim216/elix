import numpy as np
import pandas as pd
import pytest

from allatom_design.data.datasets.atomworks_sd.sampling import add_sampling_weights


def _alphas(*, small_molecule: float = 1.0) -> dict[str, float]:
    return {
        "alpha_protein_metal": 1.0,
        "alpha_protein_small_molecule": small_molecule,
        "alpha_protein_nuc_lig": 1.0,
        "alpha_protein_peptide": 1.0,
        "alpha_protein_protein": 0.0,
    }


def _monomer_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["ma1", "ma2", "mb1"],
            "q_pn_unit_cluster_id": ["A", "A", "B"],
        }
    )


def _interface_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["ia-sm-1", "ia-sm-2", "ia-metal"],
            "interface_type": ["bmsm_protein", "bmsm_protein", "bmm_protein"],
            "protein_cluster_multiset": [("A",), ("A",), ("A",)],
            "n_prot": [1, 1, 1],
            "ligand_ccd_key": [
                ("ccd", "LIGA"),
                ("ccd", "LIGA"),
                ("ccd", "MG"),
            ],
            "q_pn_unit_non_polymer_res_names": ["LIGA", "LIGA", "MG"],
        }
    )


def _run(*, fixed_k: float, small_molecule: float = 1.0):
    return add_sampling_weights(
        _monomer_df(),
        _interface_df(),
        alphas_interface=_alphas(small_molecule=small_molecule),
        fixed_k=fixed_k,
        clustering_cfg={"sampling_scheme": "fixed_k"},
    )


def test_fixed_k_preserves_pair_cluster_mass_and_equalizes_protein_clusters():
    monomer, interface = _run(fixed_k=4.0)

    small_molecule = interface.loc[interface["interface_type"] == "bmsm_protein"]
    metal = interface.loc[interface["interface_type"] == "bmm_protein"]
    assert small_molecule["sampling_weight"].tolist() == pytest.approx([0.5, 0.5])
    assert metal["sampling_weight"].tolist() == pytest.approx([1.0])
    assert monomer.loc[
        monomer["q_pn_unit_cluster_id"] == "A", "sampling_weight"
    ].tolist() == pytest.approx([1.0, 1.0])
    assert monomer.loc[
        monomer["q_pn_unit_cluster_id"] == "B", "sampling_weight"
    ].tolist() == pytest.approx([4.0])

    mass_a = interface["sampling_weight"].sum() + monomer.loc[
        monomer["q_pn_unit_cluster_id"] == "A", "sampling_weight"
    ].sum()
    mass_b = monomer.loc[
        monomer["q_pn_unit_cluster_id"] == "B", "sampling_weight"
    ].sum()
    assert mass_a == pytest.approx(4.0)
    assert mass_b == pytest.approx(4.0)


def test_fixed_k_caps_alpha_weighted_interface_mass_proportionally():
    monomer, interface = _run(fixed_k=3.0, small_molecule=4.0)

    small_molecule = interface.loc[interface["interface_type"] == "bmsm_protein"]
    metal = interface.loc[interface["interface_type"] == "bmm_protein"]
    assert small_molecule["sampling_weight"].tolist() == pytest.approx([1.2, 1.2])
    assert metal["sampling_weight"].tolist() == pytest.approx([0.6])
    assert monomer.loc[
        monomer["q_pn_unit_cluster_id"] == "A", "sampling_weight"
    ].sum() == pytest.approx(0.0)
    assert interface["sampling_weight"].sum() == pytest.approx(3.0)


def test_fixed_k_uses_requested_mass_when_interfaces_are_empty():
    interface = _interface_df().iloc[0:0].copy()
    monomer, interface = add_sampling_weights(
        _monomer_df(),
        interface,
        alphas_interface=_alphas(),
        fixed_k=3.0,
        clustering_cfg={"sampling_scheme": "fixed_k"},
    )

    assert interface.empty
    assert monomer["sampling_weight"].tolist() == pytest.approx([1.5, 1.5, 3.0])


@pytest.mark.parametrize("fixed_k", [0.0, -1.0, np.nan, np.inf])
def test_fixed_k_rejects_non_positive_or_non_finite_values(fixed_k):
    with pytest.raises(ValueError, match="finite and positive"):
        _run(fixed_k=fixed_k)


def test_fixed_k_is_required_for_the_default_sampling_scheme():
    with pytest.raises(ValueError, match="fixed_k.*required"):
        add_sampling_weights(
            _monomer_df(),
            _interface_df(),
            alphas_interface=_alphas(),
        )


@pytest.mark.parametrize("stale_scheme", ["percentile", "legacy"])
def test_removed_sampling_scheme_names_are_rejected(stale_scheme):
    with pytest.raises(ValueError, match="Unknown.*sampling_scheme"):
        add_sampling_weights(
            _monomer_df(),
            _interface_df(),
            alphas_interface=_alphas(),
            fixed_k=3.0,
            clustering_cfg={"sampling_scheme": stale_scheme},
        )
