import pandas as pd
import pytest

from allatom_design.data.datasets.atomworks_sd.sampling import (
    add_sampling_weights,
    validate_sampling_weights,
)


def _alphas(*, small_molecule: float = 8.0) -> dict[str, float]:
    return {
        "alpha_protein_metal": 1.0,
        "alpha_protein_small_molecule": small_molecule,
        "alpha_protein_nuc_lig": 1.0,
        "alpha_protein_peptide": 1.0,
        "alpha_protein_protein": 0.0,
    }


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    monomer_df = pd.DataFrame(
        {
            "example_id": ["ma1", "ma2", "mb1"],
            "q_pn_unit_cluster_id": ["A", "A", "B"],
        }
    )
    interface_df = pd.DataFrame(
        {
            "example_id": ["ia-sm-1", "ia-sm-2", "ia-metal", "ib-sm"],
            "interface_type": [
                "bmsm_protein",
                "bmsm_protein",
                "bmm_protein",
                "bmsm_protein",
            ],
            "protein_cluster_multiset": [("A",), ("A",), ("A",), ("B",)],
            "n_prot": [1, 1, 1, 1],
            "ligand_ccd_key": [
                ("ccd", "LIGA"),
                ("ccd", "LIGA"),
                ("ccd", "MG"),
                ("ccd", "LIGB"),
            ],
            "q_pn_unit_non_polymer_res_names": ["LIGA", "LIGA", "MG", "LIGB"],
        }
    )
    return monomer_df, interface_df


def _run(
    *,
    small_molecule: float = 8.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monomer_df, interface_df = _frames()
    return add_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface=_alphas(small_molecule=small_molecule),
        clustering_cfg={"sampling_scheme": "pair_cluster_balanced"},
    )


def test_pair_cluster_balanced_uses_uniform_monomers_and_alpha_per_pair_cluster():
    monomer_df, interface_df = _run()

    assert monomer_df["sampling_weight"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert interface_df["sampling_weight"].tolist() == pytest.approx(
        [4.0, 4.0, 1.0, 8.0]
    )

    pair_cluster_mass = interface_df.groupby("pair_cluster")["sampling_weight"].sum()
    assert sorted(pair_cluster_mass.tolist()) == pytest.approx([1.0, 8.0, 8.0])

    monomer_mass = monomer_df.groupby("q_pn_unit_cluster_id")["sampling_weight"].sum()
    assert monomer_mass.to_dict() == pytest.approx({"A": 2.0, "B": 1.0})


def test_pair_cluster_balanced_does_not_require_fixed_k():
    monomer_df, interface_df = _run()

    validate_sampling_weights(monomer_df, interface_df)


def test_pair_cluster_balanced_supports_empty_interface_rows():
    monomer_df, interface_df = _frames()
    interface_df = interface_df.iloc[0:0].copy()

    monomer_df, interface_df = add_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface=_alphas(),
        clustering_cfg={"sampling_scheme": "pair_cluster_balanced"},
    )

    assert monomer_df["sampling_weight"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert interface_df["sampling_weight"].empty
    validate_sampling_weights(monomer_df, interface_df)


def test_pair_cluster_balanced_supports_empty_monomer_rows():
    monomer_df, interface_df = _frames()
    monomer_df = monomer_df.iloc[0:0].copy()

    monomer_df, interface_df = add_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface=_alphas(),
        clustering_cfg={"sampling_scheme": "pair_cluster_balanced"},
    )

    assert monomer_df["sampling_weight"].empty
    validate_sampling_weights(monomer_df, interface_df)


def test_pair_cluster_balanced_negative_alpha_is_rejected_by_weight_validation():
    monomer_df, interface_df = _run(small_molecule=-1.0)

    with pytest.raises(ValueError, match="negative"):
        validate_sampling_weights(monomer_df, interface_df)


def test_pair_cluster_balanced_requires_protein_cluster_multiset():
    monomer_df, interface_df = _frames()
    interface_df = interface_df.drop(columns="protein_cluster_multiset")

    with pytest.raises(ValueError, match="protein_cluster_multiset"):
        add_sampling_weights(
            monomer_df,
            interface_df,
            alphas_interface=_alphas(),
            clustering_cfg={"sampling_scheme": "pair_cluster_balanced"},
        )
