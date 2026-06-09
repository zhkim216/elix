import json

import numpy as np
import pandas as pd
import pytest
import atomworks.enums as aw_enums
from biotite.structure import AtomArray

from omegaconf import OmegaConf

from allatom_design.data.datasets.atomworks_sd.interface import build_interface_df
from allatom_design.data.datasets.atomworks_sd.metadata import (
    add_derived_pn_unit_flags,
    add_chain_counts_info,
    add_phase_split,
    build_train_interface_df,
    collect_external_evidence,
)
from allatom_design.data.datasets.atomworks_sd.sampling import add_sampling_weights
from allatom_design.data.datasets.atomworks_sd.selectors import (
    metal_center_mask,
    parse_pn_unit_iids_value,
    small_molecule_center_mask,
)
from allatom_design.data.transform.custom_transforms import (
    annotate_ligand_pockets,
    annotate_ligand_pockets_calpha,
    annotate_ligand_pockets_pseudocb,
)
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from allatom_design.train_seq_denoiser import build_sd_datamodule


def _contacts(*pairs):
    return json.dumps([
        {"pn_unit_iid": iid, "min_distance": distance, "count": 25, "num_contacts": 25}
        for iid, distance in pairs
    ])


def _distance_contacts(*pairs):
    return json.dumps([
        {"pn_unit_iid": iid, "min_distance": distance}
        for iid, distance in pairs
    ])


def _row(
    iid,
    *,
    chain_type,
    cluster_id,
    contacts="[]",
    is_protein=False,
    is_nuc=False,
    is_peptide=False,
    is_small_molecule=False,
    is_metal=False,
    is_bmsm=False,
    is_bmm=False,
    is_nuc_ligand=False,
    nuc_group_id=None,
    nuc_group_iids=None,
    nuc_group_residues=0,
    nuc_group_cluster_ids=None,
    non_polymer_res_names=None,
    bmsm_ligand_cluster_id=-1,
):
    return {
        "pdb_id": "1abc",
        "assembly_id": "1",
        "path": "/tmp/1abc.cif",
        "q_pn_unit_id": iid.split("_")[0],
        "q_pn_unit_iid": iid,
        "q_pn_unit_type": chain_type,
        "q_pn_unit_sequence_length": nuc_group_residues or 20,
        "q_pn_unit_num_resolved_residues": nuc_group_residues or 20,
        "q_pn_unit_contacting_pn_unit_iids": contacts,
        "q_pn_unit_is_protein": is_protein,
        "q_pn_unit_is_peptide": is_peptide,
        "q_pn_unit_is_nuc": is_nuc,
        "q_pn_unit_is_small_molecule": is_small_molecule,
        "q_pn_unit_is_metal": is_metal,
        "q_pn_unit_is_polymer": is_protein or is_nuc,
        "q_pn_unit_is_biologically_meaningful_small_molecule": is_bmsm,
        "q_pn_unit_is_biologically_meaningful_metal": is_bmm,
        "q_pn_unit_is_nuc_ligand": is_nuc_ligand,
        "q_pn_unit_is_nuc_polymer": is_nuc and not is_nuc_ligand,
        "q_pn_unit_nucleic_acid_group_id": nuc_group_id,
        "q_pn_unit_nucleic_acid_group_iids": nuc_group_iids,
        "q_pn_unit_num_resolved_residues_in_nucleic_acid_group": nuc_group_residues,
        "q_pn_unit_nucleic_acid_group_cluster_ids": nuc_group_cluster_ids,
        "q_pn_unit_cluster_id": cluster_id,
        "q_pn_unit_non_polymer_res_names": non_polymer_res_names,
        "q_pn_unit_bmsm_ligand_cluster_id": bmsm_ligand_cluster_id,
        "q_pn_unit_avg_occupancy_nonpolymer": 1.0,
        "q_pn_unit_per_partner_contacts_metal": contacts,
        "q_pn_unit_per_partner_contacts_to_protein_small_molecule": contacts,
        "q_pn_unit_expected_heavy_atoms_non_polymer": 10,
        "q_pn_unit_num_resolved_atoms": 10,
        "q_pn_unit_is_artifact": False,
        "q_pn_unit_has_external_evidence": True,
        "resolution": 2.0,
        "biologically_meaningful_pn_unit_iids": ["P_1", "S_1", "A_1", "B_1"],
    }


def _interface_cfg(**overrides):
    cfg = {
        "min_protein_donor_atoms": 3,
        "min_avg_occupancy_nonpolymer": 0.5,
        "min_contacting_protein_atoms_small_molecule": 20,
        "min_avg_occupancy_nonpolymer_small_molecule": 0.5,
        "max_missing_atom_fraction_small_molecule": 0.2,
        "metal_external_evidence_policy": "no_filter",
        "small_molecule_external_evidence_policy": "no_filter",
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _protein_df(df):
    return df[df["q_pn_unit_is_protein"].fillna(False).astype(bool)].copy()


def _policy_row(
    iid,
    *,
    is_protein=False,
    is_metal=False,
    is_small_molecule=False,
    ccd=None,
    cluster_id=1,
    resolution=2.0,
    medba=False,
    pubmed=False,
    bird=False,
    artifact=False,
):
    return {
        "pdb_id": iid.lower(),
        "assembly_id": "1",
        "path": f"/tmp/{iid.lower()}.cif",
        "example_id": f"raw-{iid}",
        "q_pn_unit_iid": iid,
        "q_pn_unit_is_protein": is_protein,
        "q_pn_unit_is_metal": is_metal,
        "q_pn_unit_is_small_molecule": is_small_molecule,
        "q_pn_unit_non_polymer_res_names": ccd,
        "q_pn_unit_cluster_id": cluster_id,
        "q_pn_unit_num_resolved_residues": 100 if is_protein else 1,
        "q_pn_unit_avg_occupancy_nonpolymer": 1.0,
        "q_pn_unit_per_partner_contacts_to_protein_small_molecule": "[]",
        "q_pn_unit_expected_heavy_atoms_non_polymer": 10,
        "q_pn_unit_num_resolved_atoms": 10,
        "q_pn_unit_is_artifact": artifact,
        "resolution": resolution,
        "metal_medba_evidence": medba,
        "metal_pubmed_evidence": pubmed,
        "q_pn_unit_has_bird_prd_identifier": bird,
    }


def test_collect_external_evidence_uses_configured_medba_pubmed_bird_columns():
    df = pd.DataFrame(
        [
            _policy_row("MG_1", is_metal=True, ccd="MG", medba=True),
            _policy_row("ZN_1", is_metal=True, ccd="ZN", pubmed=True),
            _policy_row("ATP_1", is_small_molecule=True, ccd="ATP", bird=True),
            _policy_row("P_1", is_protein=True),
        ]
    )

    out = collect_external_evidence(
        df,
        allowed_evidence_columns=[
            "metal_medba_evidence",
            "metal_pubmed_evidence",
            "q_pn_unit_has_bird_prd_identifier",
        ],
    )

    assert out["q_pn_unit_has_external_evidence"].tolist() == [True, True, True, False]

    with pytest.raises(KeyError, match="missing_evidence"):
        collect_external_evidence(df, allowed_evidence_columns=["missing_evidence"])


def test_split_metal_and_small_molecule_policies_are_independent():
    df = pd.DataFrame(
        [
            _policy_row("MG_1", is_metal=True, ccd="MG", medba=True),
            _policy_row("ZN_1", is_metal=True, ccd="ZN"),
            _policy_row("ATP_1", is_small_molecule=True, ccd="ATP", artifact=False),
            _policy_row("PEG_1", is_small_molecule=True, ccd="PEG", artifact=True, bird=True),
        ]
    )
    df = collect_external_evidence(
        df,
        allowed_evidence_columns=[
            "metal_medba_evidence",
            "metal_pubmed_evidence",
            "q_pn_unit_has_bird_prd_identifier",
        ],
    )
    cfg = OmegaConf.create(
        {
            "metal_external_evidence_policy": "external_evidence",
            "small_molecule_external_evidence_policy": "filter_all_artifacts",
            "allowed_ccd_codes": None,
            "min_avg_occupancy_nonpolymer": 0.5,
            "min_contacting_protein_atoms_small_molecule": None,
            "min_avg_occupancy_nonpolymer_small_molecule": 0.5,
            "max_missing_atom_fraction_small_molecule": 0.2,
        }
    )

    assert metal_center_mask(df, cfg).tolist() == [True, False, False, False]
    assert small_molecule_center_mask(df, cfg).tolist() == [False, False, True, False]


def test_add_derived_flags_reconstructs_missing_nucleic_acid_group_columns():
    df = pd.DataFrame(
        [
            {
                "pdb_id": "1abc",
                "assembly_id": "1",
                "q_pn_unit_iid": "A_1",
                "q_pn_unit_type": 3,
                "q_pn_unit_contacting_pn_unit_iids": '[{"pn_unit_iid": "B_1", "min_distance": 4.0}]',
                "q_pn_unit_num_resolved_residues": 4,
                "q_pn_unit_cluster_id": 10,
            },
            {
                "pdb_id": "1abc",
                "assembly_id": "1",
                "q_pn_unit_iid": "B_1",
                "q_pn_unit_type": 3,
                "q_pn_unit_contacting_pn_unit_iids": '[{"pn_unit_iid": "A_1", "min_distance": 4.0}]',
                "q_pn_unit_num_resolved_residues": 4,
                "q_pn_unit_cluster_id": 11,
            },
        ]
    )

    out = add_derived_pn_unit_flags(df, {})

    assert out["q_pn_unit_nucleic_acid_group_id"].tolist() == ["(A_1, B_1)", "(A_1, B_1)"]
    assert out["q_pn_unit_is_nuc_ligand"].tolist() == [True, True]


def test_val_cluster_ids_follow_configured_monomer_resolution_filter(tmp_path):
    val_metadata = tmp_path / "metadata_for_training_nativeval.parquet"
    pd.DataFrame({"pdb_id": ["val_low", "val_high"]}).to_parquet(val_metadata)
    metadata_df = pd.DataFrame(
        [
            _policy_row("P_LOW", is_protein=True, cluster_id=101, resolution=4.0),
            _policy_row("P_HIGH", is_protein=True, cluster_id=102, resolution=8.0),
        ]
    )
    metadata_df.loc[0, "pdb_id"] = "val_low"
    metadata_df.loc[1, "pdb_id"] = "val_high"

    base_cfg = {
        "validation_ids_file": None,
        "val_metadata_path": str(val_metadata),
        "debug": False,
        "exclude_val_cluster": True,
    }
    cfg_45 = OmegaConf.create(
        {
            **base_cfg,
            "train_filters": {
                "protein_monomer_chain_filter": [
                    "(q_pn_unit_is_protein and resolution < 4.5 and 20 <= q_pn_unit_num_resolved_residues < 2048)"
                ]
            },
        }
    )
    cfg_90 = OmegaConf.create(
        {
            **base_cfg,
            "train_filters": {
                "protein_monomer_chain_filter": [
                    "(q_pn_unit_is_protein and resolution < 9.0 and 20 <= q_pn_unit_num_resolved_residues < 2048)"
                ]
            },
        }
    )
    cfg_30 = OmegaConf.create(
        {
            **base_cfg,
            "train_filters": {
                "protein_monomer_chain_filter": [
                    "(q_pn_unit_is_protein and resolution < 3.0 and 20 <= q_pn_unit_num_resolved_residues < 2048)"
                ]
            },
        }
    )

    _, val_clusters_45 = add_phase_split(metadata_df, cfg_45)
    _, val_clusters_90 = add_phase_split(metadata_df, cfg_90)
    _, val_clusters_30 = add_phase_split(metadata_df, cfg_30)

    assert set(val_clusters_45) == {101}
    assert set(val_clusters_90) == {101, 102}
    assert val_clusters_30 == []


def test_build_sd_datamodule_routes_atomworks_sd_and_proto_alias_to_refactor(monkeypatch):
    import allatom_design.data.datasets.atomworks_sd.datamodule as datamodule_module

    def fake_init(self, cfg):
        self.cfg = cfg

    monkeypatch.setattr(datamodule_module.AtomworksSDDataModule, "__init__", fake_init)

    atomworks_dm = build_sd_datamodule(OmegaConf.create({"dataset_impl": "atomworks_sd"}))
    proto_dm = build_sd_datamodule(OmegaConf.create({"dataset_impl": "proto"}))

    assert isinstance(atomworks_dm, datamodule_module.AtomworksSDDataModule)
    assert isinstance(proto_dm, datamodule_module.AtomworksSDDataModule)


def test_build_interface_df_adds_one_row_per_nucleic_acid_ligand_group():
    df = pd.DataFrame(
        [
            _row("P_1", chain_type=6, cluster_id=100, is_protein=True),
            _row(
                "S_1",
                chain_type=8,
                cluster_id=200,
                contacts=_contacts(("P_1", 3.8)),
                is_small_molecule=True,
                is_bmsm=True,
                non_polymer_res_names="ATP",
                bmsm_ligand_cluster_id=7,
            ),
            _row(
                "A_1",
                chain_type=3,
                cluster_id=10,
                contacts=_contacts(("B_1", 4.5), ("P_1", 4.0)),
                is_nuc=True,
                is_nuc_ligand=True,
                nuc_group_id="(A_1, B_1)",
                nuc_group_iids="A_1, B_1",
                nuc_group_residues=10,
                nuc_group_cluster_ids="10, 11",
            ),
            _row(
                "B_1",
                chain_type=7,
                cluster_id=11,
                contacts=_contacts(("A_1", 4.5)),
                is_nuc=True,
                is_nuc_ligand=True,
                nuc_group_id="(A_1, B_1)",
                nuc_group_iids="A_1, B_1",
                nuc_group_residues=10,
                nuc_group_cluster_ids="10, 11",
            ),
        ]
    )

    interface_df = build_interface_df(
        metadata_df=df,
        protein_df=_protein_df(df),
        dataset_name="toy",
        cfg=_interface_cfg(),
    )

    assert set(interface_df["interface_type"]) == {"bmsm_protein", "nuc_lig_protein"}

    nuc_row = interface_df[interface_df["interface_type"] == "nuc_lig_protein"].iloc[0]
    assert nuc_row["ligand_pn_unit_iids"] == ("A_1", "B_1")
    assert nuc_row["protein_pn_unit_iids"] == ("P_1",)
    assert nuc_row["query_pn_unit_iids"] == ["A_1", "B_1", "P_1"]
    assert nuc_row["ligand_ccd_key"] == ("nuc_seq_cluster", (10, 11))

    counted = add_chain_counts_info(interface_df.copy())
    nuc_counted = counted[counted["interface_type"] == "nuc_lig_protein"].iloc[0]
    assert nuc_counted["n_nuc"] == 1
    assert nuc_counted["n_small_molecule"] == 0


def test_nuc_lig_interface_uses_alpha_protein_nuc_lig():
    monomer_df = pd.DataFrame(
        [{"q_pn_unit_cluster_id": 100, "q_pn_unit_is_protein": True}],
        index=["monomer"],
    )
    interface_df = pd.DataFrame(
        [
            {
                "interface_type": "nuc_lig_protein",
                "protein_cluster_multiset": (100,),
                "ligand_ccd_key": ("nuc_seq_cluster", (10, 11)),
            }
        ],
        index=["iface"],
    )

    _, weighted_interface = add_sampling_weights(
        monomer_df=monomer_df,
        interface_df=interface_df,
        alphas_interface={
            "alpha_protein_small_molecule": 0.0,
            "alpha_protein_nuc_lig": 2.0,
        },
        k_percentile=100.0,
    )

    assert weighted_interface.loc["iface", "alpha"] == 2.0
    assert weighted_interface.loc["iface", "sampling_weight"] == 2.0


def test_build_interface_df_adds_bmm_protein_rows():
    df = pd.DataFrame(
        [
            _row("P_1", chain_type=6, cluster_id=100, is_protein=True),
            _row(
                "M_1",
                chain_type=10,
                cluster_id=300,
                contacts=_contacts(("P_1", 2.2)),
                is_metal=True,
                is_bmm=True,
                non_polymer_res_names="MG",
            ),
        ]
    )

    interface_df = build_interface_df(
        metadata_df=df,
        protein_df=_protein_df(df),
        dataset_name="toy",
        cfg=_interface_cfg(),
    )

    assert set(interface_df["interface_type"]) == {"bmm_protein"}
    row = interface_df.iloc[0]
    assert row["query_pn_unit_iids"] == ["M_1", "P_1"]
    assert row["ligand_pn_unit_iids"] == ("M_1",)
    assert row["protein_pn_unit_iids"] == ("P_1",)
    assert row["ligand_ccd_key"] == ("ccd", "MG")

    counted = add_chain_counts_info(interface_df.copy())
    assert counted.iloc[0]["n_metal"] == 1
    assert counted.iloc[0]["n_small_molecule"] == 0


def test_bmm_interface_uses_alpha_protein_metal():
    monomer_df = pd.DataFrame(
        [{"q_pn_unit_cluster_id": 100, "q_pn_unit_is_protein": True}],
        index=["monomer"],
    )
    interface_df = pd.DataFrame(
        [
            {
                "interface_type": "bmm_protein",
                "protein_cluster_multiset": (100,),
                "ligand_ccd_key": ("ccd", "MG"),
            }
        ],
        index=["iface"],
    )

    _, weighted_interface = add_sampling_weights(
        monomer_df=monomer_df,
        interface_df=interface_df,
        alphas_interface={
            "alpha_protein_metal": 3.0,
        },
        k_percentile=100.0,
    )

    assert weighted_interface.loc["iface", "alpha"] == 3.0
    assert weighted_interface.loc["iface", "sampling_weight"] == 3.0


def test_train_interface_filter_keeps_only_single_protein_ligand_contexts():
    df = pd.DataFrame(
        [
            _row("P_1", chain_type=6, cluster_id=100, is_protein=True),
            _row("P_2", chain_type=6, cluster_id=101, is_protein=True),
            _row(
                "T_1",
                chain_type=6,
                cluster_id=200,
                contacts=_contacts(("P_1", 4.0), ("P_2", 4.0)),
                is_peptide=True,
                is_protein=False,
            ),
            _row(
                "A_1",
                chain_type=3,
                cluster_id=10,
                contacts=_contacts(("B_1", 4.5), ("P_1", 4.0)),
                is_nuc=True,
                is_nuc_ligand=True,
                nuc_group_id="(A_1, B_1)",
                nuc_group_iids="A_1, B_1",
                nuc_group_residues=10,
                nuc_group_cluster_ids="10, 11",
            ),
            _row(
                "B_1",
                chain_type=7,
                cluster_id=11,
                contacts=_contacts(("A_1", 4.5)),
                is_nuc=True,
                is_nuc_ligand=True,
                nuc_group_id="(A_1, B_1)",
                nuc_group_iids="A_1, B_1",
                nuc_group_residues=10,
                nuc_group_cluster_ids="10, 11",
            ),
        ]
    )

    cfg = OmegaConf.create(
        {
            **OmegaConf.to_container(_interface_cfg(), resolve=True),
            "exclude_val_cluster": False,
            "train_filters": {
                "protein_monomer_chain_filter": [
                    "(q_pn_unit_is_protein and resolution < 9.0 and 20 <= q_pn_unit_num_resolved_residues < 2048)"
                ],
                "interface_filter": {
                    "1": ["resolution < 3.5"],
                    "2": [
                        "interface_type in ['bmm_protein', 'bmsm_protein', 'nuc_lig_protein', 'peptide_protein']",
                        "n_prot == 1",
                    ],
                },
            },
        }
    )

    interface_df = build_train_interface_df(
        metadata_df=df,
        cfg=cfg,
        dataset_name="toy",
        val_cluster_ids=[],
    )

    assert interface_df["interface_type"].tolist() == ["nuc_lig_protein"]
    assert interface_df.iloc[0]["n_prot"] == 1


def test_polymer_ligands_do_not_use_small_molecule_contact_atom_gate():
    df = pd.DataFrame(
        [
            _row("P_1", chain_type=6, cluster_id=100, is_protein=True),
            _row(
                "T_1",
                chain_type=6,
                cluster_id=200,
                contacts=_distance_contacts(("P_1", 4.0)),
                is_peptide=True,
            ),
            _row(
                "A_1",
                chain_type=3,
                cluster_id=10,
                contacts=_distance_contacts(("B_1", 4.5), ("P_1", 4.0)),
                is_nuc=True,
                is_nuc_ligand=True,
                nuc_group_id="(A_1, B_1)",
                nuc_group_iids="A_1, B_1",
                nuc_group_residues=10,
                nuc_group_cluster_ids="10, 11",
            ),
            _row(
                "B_1",
                chain_type=7,
                cluster_id=11,
                contacts=_distance_contacts(("A_1", 4.5)),
                is_nuc=True,
                is_nuc_ligand=True,
                nuc_group_id="(A_1, B_1)",
                nuc_group_iids="A_1, B_1",
                nuc_group_residues=10,
                nuc_group_cluster_ids="10, 11",
            ),
        ]
    )
    cfg = _interface_cfg(
        min_contacting_protein_atoms_small_molecule=999,
        max_missing_atom_fraction_small_molecule=0.0,
    )

    interface_df = build_interface_df(
        metadata_df=df,
        protein_df=_protein_df(df),
        dataset_name="toy",
        cfg=cfg,
    )

    assert set(interface_df["interface_type"]) == {"peptide_protein", "nuc_lig_protein"}


def test_peptide_and_nuc_lig_interfaces_use_separate_alphas():
    monomer_df = pd.DataFrame(
        [{"q_pn_unit_cluster_id": 100}],
        index=["monomer"],
    )
    interface_df = pd.DataFrame(
        [
            {
                "interface_type": "peptide_protein",
                "protein_cluster_multiset": (100,),
                "ligand_ccd_key": ("peptide_seq_cluster", 200),
            },
            {
                "interface_type": "nuc_lig_protein",
                "protein_cluster_multiset": (100,),
                "ligand_ccd_key": ("nuc_seq_cluster", (10, 11)),
            },
        ],
        index=["peptide", "nuc"],
    )

    _, weighted_interface = add_sampling_weights(
        monomer_df=monomer_df,
        interface_df=interface_df,
        alphas_interface={
            "alpha_protein_peptide": 2.0,
            "alpha_protein_nuc_lig": 3.0,
        },
        k_percentile=100.0,
    )

    assert weighted_interface.loc["peptide", "alpha"] == 2.0
    assert weighted_interface.loc["nuc", "alpha"] == 3.0


def test_parse_pn_unit_iids_accepts_numpy_arrays_and_strings():
    assert parse_pn_unit_iids_value(np.array(["A_1", "B_1"], dtype=object)) == ["A_1", "B_1"]
    assert parse_pn_unit_iids_value("['A_1', 'B_1']") == ["A_1", "B_1"]
    assert parse_pn_unit_iids_value(("A_1", "B_1")) == ["A_1", "B_1"]


def _protein_atom_array(res_names):
    atom_names = ["N", "CA", "C", "O"]
    n_protein_atoms = len(res_names) * len(atom_names)
    arr = AtomArray(n_protein_atoms + 1)
    coords = []
    res_id = []
    res_name = []
    atom_name = []
    chain_type = []
    hetero = []
    occupancy = []
    chain_id = []
    pn_unit_iid = []
    is_polymer = []
    is_covalent_modification = []
    atomize = []

    for i, rn in enumerate(res_names, start=1):
        base_x = float(i - 1) * 6.0
        for atom_idx, an in enumerate(atom_names):
            coords.append([base_x + atom_idx * 0.1, 0.0, 0.0])
            res_id.append(i)
            res_name.append(rn)
            atom_name.append(an)
            chain_type.append(aw_enums.ChainType.POLYPEPTIDE_L)
            hetero.append(False)
            occupancy.append(1.0)
            chain_id.append("A")
            pn_unit_iid.append("A_1")
            is_polymer.append(True)
            is_covalent_modification.append(False)
            atomize.append(False)

    coords.append([0.2, 0.0, 0.0])
    res_id.append(1)
    res_name.append("MG")
    atom_name.append("MG")
    chain_type.append(aw_enums.ChainType.NON_POLYMER)
    hetero.append(True)
    occupancy.append(1.0)
    chain_id.append("B")
    pn_unit_iid.append("B_1")
    is_polymer.append(False)
    is_covalent_modification.append(False)
    atomize.append(False)

    arr.coord = np.array(coords, dtype=float)
    arr.res_id = np.array(res_id)
    arr.res_name = np.array(res_name, dtype=object)
    arr.atom_name = np.array(atom_name, dtype=object)
    arr.hetero = np.array(hetero, dtype=bool)
    arr.set_annotation("occupancy", np.array(occupancy, dtype=float))
    arr.chain_id = np.array(chain_id, dtype=object)
    arr.set_annotation("chain_type", np.array(chain_type, dtype=object))
    arr.set_annotation("pn_unit_iid", np.array(pn_unit_iid, dtype=object))
    arr.set_annotation("is_polymer", np.array(is_polymer, dtype=bool))
    arr.set_annotation(
        "is_covalent_modification",
        np.array(is_covalent_modification, dtype=bool),
    )
    arr.set_annotation("atomize", np.array(atomize, dtype=bool))
    return arr


def test_mg_single_atom_pocket_annotation_uses_n_min_ligand_atoms():
    atom_array = _protein_atom_array(["ALA", "GLY"])

    default_annotated = annotate_ligand_pockets(
        atom_array.copy(),
        pocket_distance=5.0,
        n_min_ligand_atoms=5,
        annotation_name="is_ligand_pocket_5_default",
    )
    mg_annotated = annotate_ligand_pockets(
        atom_array.copy(),
        pocket_distance=5.0,
        n_min_ligand_atoms=1,
        annotation_name="is_ligand_pocket_5_mg",
    )

    default_residue_mask = apply_and_spread_residue_wise(
        default_annotated,
        default_annotated.get_annotation("is_ligand_pocket_5_default"),
        function=np.any,
    )
    mg_residue_mask = apply_and_spread_residue_wise(
        mg_annotated,
        mg_annotated.get_annotation("is_ligand_pocket_5_mg"),
        function=np.any,
    )
    assert int(default_residue_mask.sum()) == 0
    assert int(mg_residue_mask[mg_annotated.atom_name == "CA"].sum()) == 1


def test_calpha_pocket_annotation_marks_near_ca_residue():
    atom_array = _protein_atom_array(["ALA", "GLY"])

    annotated = annotate_ligand_pockets_calpha(
        atom_array.copy(),
        pocket_distance=5.0,
        n_min_ligand_atoms=1,
        annotation_name="is_ligand_pocket_ca",
    )

    ca_mask = annotated.atom_name == "CA"
    assert annotated.get_annotation("is_ligand_pocket_ca")[ca_mask].tolist() == [True, False]


def test_calpha_pocket_annotation_does_not_cross_same_res_id_chains():
    atom_names = ["N", "CA", "C", "O"]
    coords = []
    res_id = []
    res_name = []
    atom_name = []
    chain_type = []
    hetero = []
    occupancy = []
    chain_id = []
    pn_unit_iid = []
    is_polymer = []
    is_covalent_modification = []
    atomize = []

    for chain, iid, base_x in [("A", "A_1", 0.0), ("C", "C_1", 100.0)]:
        for atom_idx, an in enumerate(atom_names):
            coords.append([base_x + atom_idx * 0.1, 0.0, 0.0])
            res_id.append(1)
            res_name.append("ALA")
            atom_name.append(an)
            chain_type.append(aw_enums.ChainType.POLYPEPTIDE_L)
            hetero.append(False)
            occupancy.append(1.0)
            chain_id.append(chain)
            pn_unit_iid.append(iid)
            is_polymer.append(True)
            is_covalent_modification.append(False)
            atomize.append(False)

    coords.append([0.1, 0.0, 0.0])
    res_id.append(1)
    res_name.append("MG")
    atom_name.append("MG")
    chain_type.append(aw_enums.ChainType.NON_POLYMER)
    hetero.append(True)
    occupancy.append(1.0)
    chain_id.append("B")
    pn_unit_iid.append("B_1")
    is_polymer.append(False)
    is_covalent_modification.append(False)
    atomize.append(False)

    expanded = AtomArray(len(coords))
    expanded.coord = np.array(coords, dtype=float)
    expanded.res_id = np.array(res_id)
    expanded.res_name = np.array(res_name, dtype=object)
    expanded.atom_name = np.array(atom_name, dtype=object)
    expanded.hetero = np.array(hetero, dtype=bool)
    expanded.set_annotation("occupancy", np.array(occupancy, dtype=float))
    expanded.chain_id = np.array(chain_id, dtype=object)
    expanded.set_annotation("chain_type", np.array(chain_type, dtype=object))
    expanded.set_annotation("pn_unit_iid", np.array(pn_unit_iid, dtype=object))
    expanded.set_annotation("is_polymer", np.array(is_polymer, dtype=bool))
    expanded.set_annotation("is_covalent_modification", np.array(is_covalent_modification, dtype=bool))
    expanded.set_annotation("atomize", np.array(atomize, dtype=bool))

    annotated = annotate_ligand_pockets_calpha(
        expanded,
        pocket_distance=5.0,
        n_min_ligand_atoms=1,
        annotation_name="is_ligand_pocket_ca",
    )

    pocket = annotated.get_annotation("is_ligand_pocket_ca")
    assert bool(pocket[(annotated.chain_id == "A") & (annotated.atom_name == "CA")][0])
    assert not bool(pocket[(annotated.chain_id == "C") & (annotated.atom_name == "CA")][0])


def test_pseudocb_pocket_annotation_marks_near_pseudocb_residue():
    atom_array = _protein_atom_array(["ALA", "GLY"])

    annotated = annotate_ligand_pockets_pseudocb(
        atom_array.copy(),
        pocket_distance=5.0,
        n_min_ligand_atoms=1,
        annotation_name="is_ligand_pocket_pcb",
    )

    ca_mask = annotated.atom_name == "CA"
    assert annotated.get_annotation("is_ligand_pocket_pcb")[ca_mask].tolist() == [True, False]


def test_pseudocb_pocket_annotation_does_not_cross_same_res_id_chains():
    atom_names = ["N", "CA", "C", "O"]
    coords = []
    res_id = []
    res_name = []
    atom_name = []
    chain_type = []
    hetero = []
    occupancy = []
    chain_id = []
    pn_unit_iid = []
    is_polymer = []
    is_covalent_modification = []
    atomize = []

    for chain, iid, base_x in [("A", "A_1", 0.0), ("C", "C_1", 100.0)]:
        for atom_idx, an in enumerate(atom_names):
            coords.append([base_x + atom_idx * 0.1, 0.0, 0.0])
            res_id.append(1)
            res_name.append("ALA")
            atom_name.append(an)
            chain_type.append(aw_enums.ChainType.POLYPEPTIDE_L)
            hetero.append(False)
            occupancy.append(1.0)
            chain_id.append(chain)
            pn_unit_iid.append(iid)
            is_polymer.append(True)
            is_covalent_modification.append(False)
            atomize.append(False)

    coords.append([0.1, 0.0, 0.0])
    res_id.append(1)
    res_name.append("MG")
    atom_name.append("MG")
    chain_type.append(aw_enums.ChainType.NON_POLYMER)
    hetero.append(True)
    occupancy.append(1.0)
    chain_id.append("B")
    pn_unit_iid.append("B_1")
    is_polymer.append(False)
    is_covalent_modification.append(False)
    atomize.append(False)

    expanded = AtomArray(len(coords))
    expanded.coord = np.array(coords, dtype=float)
    expanded.res_id = np.array(res_id)
    expanded.res_name = np.array(res_name, dtype=object)
    expanded.atom_name = np.array(atom_name, dtype=object)
    expanded.hetero = np.array(hetero, dtype=bool)
    expanded.set_annotation("occupancy", np.array(occupancy, dtype=float))
    expanded.chain_id = np.array(chain_id, dtype=object)
    expanded.set_annotation("chain_type", np.array(chain_type, dtype=object))
    expanded.set_annotation("pn_unit_iid", np.array(pn_unit_iid, dtype=object))
    expanded.set_annotation("is_polymer", np.array(is_polymer, dtype=bool))
    expanded.set_annotation("is_covalent_modification", np.array(is_covalent_modification, dtype=bool))
    expanded.set_annotation("atomize", np.array(atomize, dtype=bool))

    annotated = annotate_ligand_pockets_pseudocb(
        expanded,
        pocket_distance=5.0,
        n_min_ligand_atoms=1,
        annotation_name="is_ligand_pocket_pcb",
    )

    pocket = annotated.get_annotation("is_ligand_pocket_pcb")
    assert bool(pocket[(annotated.chain_id == "A") & (annotated.atom_name == "CA")][0])
    assert not bool(pocket[(annotated.chain_id == "C") & (annotated.atom_name == "CA")][0])
