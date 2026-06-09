import json

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from allatom_design.data.datasets.atomworks_sd_dataset_proto import (
    ProtoSDDataset,
    add_proto_chain_counts_info,
    add_proto_sampling_weights,
    build_proto_interface_df,
    collect_external_evidence,
    _add_proto_modality_columns,
    _proto_center_mask,
)
from allatom_design.train_seq_denoiser import build_sd_datamodule


def _contacts(*items):
    return json.dumps(
        [
            {"pn_unit_iid": pn_unit_iid, "chain_iid": pn_unit_iid, "count": count}
            for pn_unit_iid, count in items
        ]
    )


def _protein_contacts(*items):
    return json.dumps(
        [
            {
                "pn_unit_iid": pn_unit_iid,
                "num_contacts": count,
                "min_distance": min_distance,
            }
            for pn_unit_iid, count, min_distance in items
        ]
    )


def _row(
    iid,
    *,
    is_protein=False,
    is_metal=False,
    is_small_molecule=False,
    ccd=None,
    cluster_id=1,
    contacts="[]",
    protein_contacts="[]",
    sm_protein_contacts="[]",
    medba_evidence=False,
    pubmed_evidence=False,
    binding_evidence=False,
    procog_evidence=False,
    rellig_present=False,
    avg_occupancy=None,
    resolved_atoms=10,
    expected_atoms=10,
):
    if avg_occupancy is None:
        avg_occupancy = 1.0 if is_metal or is_small_molecule else np.nan
    return {
        "pdb_id": "1abc",
        "assembly_id": "1",
        "path": "/tmp/1abc.cif",
        "example_id": f"raw-{iid}",
        "q_pn_unit_id": iid.split("_")[0],
        "q_pn_unit_iid": iid,
        "q_pn_unit_type": 6 if is_protein else 8,
        "q_pn_unit_sequence_length": 50 if is_protein else np.nan,
        "q_pn_unit_num_resolved_residues": 50 if is_protein else 1,
        "q_pn_unit_is_protein": is_protein,
        "q_pn_unit_is_metal": is_metal,
        "q_pn_unit_is_polymer": is_protein,
        "q_pn_unit_is_halide": False,
        "q_pn_unit_is_small_molecule": is_small_molecule,
        "q_pn_unit_non_polymer_res_names": ccd,
        "q_pn_unit_avg_occupancy_nonpolymer": avg_occupancy,
        "q_pn_unit_per_partner_contacts_metal": contacts,
        "q_pn_unit_contacting_pn_unit_iids": protein_contacts,
        "q_pn_unit_per_partner_contacts_to_protein_small_molecule": sm_protein_contacts,
        "q_pn_unit_expected_heavy_atoms_non_polymer": expected_atoms,
        "q_pn_unit_num_resolved_atoms": resolved_atoms,
        "q_pn_unit_cluster_id": cluster_id,
        "q_pn_unit_external_binding_relevance_evidence": binding_evidence,
        "q_pn_unit_procoggraph_cognate_ligand_evidence": procog_evidence,
        "q_pn_unit_rellig_source_row_present": rellig_present,
        "metal_medba_evidence": medba_evidence,
        "metal_pubmed_evidence": pubmed_evidence,
    }


def _artifact_tsv(tmp_path, *codes):
    path = tmp_path / "artifact_sources_by_ccd_JH_ver2.tsv"
    path.write_text("ccd_code\n" + "\n".join(codes) + "\n")
    return path


def test_collect_external_evidence_ors_medba_and_pubmed_for_metal_rows_only():
    df = pd.DataFrame(
        [
            _row("Z_1", is_metal=True, ccd="ZN", medba_evidence=True),
            _row("F_1", is_metal=True, ccd="FE", pubmed_evidence=True),
            _row("M_1", is_metal=True, ccd="MG"),
            _row("P_1", is_protein=True, medba_evidence=True, pubmed_evidence=True),
        ]
    )

    out = collect_external_evidence(df)

    assert out["has_external_evidence"].tolist() == [True, True, False, False]


def test_collect_external_evidence_requires_sources_only_when_requested():
    df = pd.DataFrame([_row("Z_1", is_metal=True, ccd="ZN")]).drop(columns=["metal_pubmed_evidence"])

    with pytest.raises(KeyError, match="metal_pubmed_evidence"):
        collect_external_evidence(df, require_columns=True)

    out = collect_external_evidence(df, require_columns=False)
    assert out["has_external_evidence"].tolist() == [False]


def test_proto_center_mask_supports_all_metal_default_ccd_filter_and_external_evidence():
    df = pd.DataFrame(
        [
            _row("M_1", is_metal=True, ccd="MG"),
            _row("Z_1", is_metal=True, ccd="ZN", pubmed_evidence=True),
            _row("F_1", is_metal=True, ccd="FE"),
            _row("S_1", is_protein=False, ccd="ATP", pubmed_evidence=True),
        ]
    )
    df.loc[df["q_pn_unit_iid"] == "F_1", "q_pn_unit_avg_occupancy_nonpolymer"] = 0.25
    df = collect_external_evidence(df)

    no_filter = _proto_center_mask(
        df,
        {"external_evidence_policy": "no_filter", "allowed_ccd_codes": None, "min_avg_occupancy_nonpolymer": 0.5},
    )
    assert no_filter.tolist() == [True, True, False, False]

    external = _proto_center_mask(
        df,
        {"external_evidence_policy": "external_evidence", "allowed_ccd_codes": None, "min_avg_occupancy_nonpolymer": 0.5},
    )
    assert external.tolist() == [False, True, False, False]

    mg_only = _proto_center_mask(
        df,
        {"external_evidence_policy": "no_filter", "allowed_ccd_codes": ["MG"], "min_avg_occupancy_nonpolymer": 0.5},
    )
    assert mg_only.tolist() == [True, False, False, False]

    with pytest.raises(ValueError, match="allowed_ccd_codes"):
        _proto_center_mask(df, {"allowed_ccd_codes": []})


def test_proto_center_mask_applies_artifact_evidence_gate_and_physical_sm_filters(tmp_path):
    artifact_path = _artifact_tsv(tmp_path, "BME", "PEG", "SO4")
    df = pd.DataFrame(
        [
            _row("B_1", is_small_molecule=True, ccd="BME", sm_protein_contacts=_contacts(("P_1", 25))),
            _row(
                "P_1",
                is_small_molecule=True,
                ccd="PEG",
                sm_protein_contacts=_contacts(("P_1", 25)),
                procog_evidence=True,
            ),
            _row("A_1", is_small_molecule=True, ccd="ADP", sm_protein_contacts=_contacts(("P_1", 25))),
            _row(
                "R_1",
                is_small_molecule=True,
                ccd="SO4",
                sm_protein_contacts=_contacts(("P_1", 25)),
                rellig_present=True,
            ),
            _row(
                "C_1",
                is_small_molecule=True,
                ccd="SO4",
                sm_protein_contacts=_contacts(("P_1", 5)),
                binding_evidence=True,
            ),
            _row(
                "O_1",
                is_small_molecule=True,
                ccd="SO4",
                sm_protein_contacts=_contacts(("P_1", 25)),
                binding_evidence=True,
                avg_occupancy=0.25,
            ),
            _row(
                "M_1",
                is_small_molecule=True,
                ccd="SO4",
                sm_protein_contacts=_contacts(("P_1", 25)),
                binding_evidence=True,
                resolved_atoms=7,
                expected_atoms=10,
            ),
            _row("Z_1", is_metal=True, ccd="ZN", contacts=_contacts(("P_1", 3)), pubmed_evidence=True),
        ]
    )
    df = collect_external_evidence(df)

    mask = _proto_center_mask(
        df,
        {
            "external_evidence_policy": "external_evidence",
            "small_molecule_artifact_list_path": str(artifact_path),
            "small_molecule_external_evidence_policy": "binding_or_procog",
            "min_contacting_protein_atoms_small_molecule": 20,
            "min_avg_occupancy_nonpolymer_small_molecule": 0.5,
            "max_missing_atom_fraction_small_molecule": 0.2,
        },
    )

    assert mask.tolist() == [False, True, True, False, False, False, False, True]


def test_add_proto_modality_columns_marks_artifact_ccds_from_configured_list(tmp_path):
    artifact_path = _artifact_tsv(tmp_path, "bme", "PEG")
    df = pd.DataFrame(
        [
            _row("B_1", is_small_molecule=True, ccd="BME"),
            _row("A_1", is_small_molecule=True, ccd="ADP"),
            _row("P_1", is_protein=True),
            _row("X_1", is_small_molecule=True, ccd="ATP, PEG"),
        ]
    )

    out = _add_proto_modality_columns(
        df,
        {"small_molecule_artifact_list_path": str(artifact_path)},
    )

    assert out["q_pn_unit_is_artifact"].tolist() == [True, False, False, True]


def test_add_proto_modality_columns_defaults_artifact_flag_false_without_list():
    df = pd.DataFrame([_row("B_1", is_small_molecule=True, ccd="BME")])

    out = _add_proto_modality_columns(df, {})

    assert out["q_pn_unit_is_artifact"].tolist() == [False]


def test_filter_metadata_to_proto_scope_keeps_protein_monomers_and_selected_metals():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("P_short", is_protein=True, cluster_id=101),
            _row("M_1", is_metal=True, ccd="MG", cluster_id=200),
            _row("Z_1", is_metal=True, ccd="ZN", cluster_id=201),
            _row("S_1", is_protein=False, ccd="ATP", cluster_id=203),
        ]
    )
    metadata_df.loc[metadata_df["q_pn_unit_iid"] == "P_short", "q_pn_unit_num_resolved_residues"] = 5
    metadata_df = collect_external_evidence(metadata_df)
    metadata_df.set_index("example_id", inplace=True, drop=False)

    dataset = object.__new__(ProtoSDDataset)
    dataset.proto_cfg = {"external_evidence_policy": "no_filter", "allowed_ccd_codes": ["ZN"]}
    dataset.cfg = OmegaConf.create(
        {
            "train_filters": {
                "protein_monomer_chain_filter": [
                    "(q_pn_unit_is_protein and 20 <= q_pn_unit_num_resolved_residues < 2048)"
                ]
            }
        }
    )

    scoped = dataset._filter_metadata_to_proto_scope(metadata_df)

    assert scoped["q_pn_unit_iid"].tolist() == ["P_1", "Z_1"]


def test_build_proto_interface_df_includes_multi_protein_metal_interfaces_and_actual_ccd_key():
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("P_2", is_protein=True, cluster_id=101),
            _row(
                "Z_1",
                is_metal=True,
                ccd="ZN",
                cluster_id=300,
                contacts=_contacts(("P_1", 2), ("P_2", 2)),
            ),
        ]
    )
    metadata_df = collect_external_evidence(metadata_df)

    interface_df = build_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        proto_cfg={"min_protein_donor_atoms": 3, "min_avg_occupancy_nonpolymer": 0.5},
    )

    assert len(interface_df) == 1
    row = interface_df.iloc[0]
    assert row["query_pn_unit_iids"] == ["Z_1", "P_1", "P_2"]
    assert row["ligand_pn_unit_iids"] == ("Z_1",)
    assert row["protein_pn_unit_iids"] == ("P_1", "P_2")
    assert row["protein_cluster_multiset"] == (100, 101)
    assert row["ligand_ccd_key"] == ("ccd", "ZN")
    assert row["n_coordinating_protein_donor_atoms"] == 4


def test_build_proto_interface_df_adds_artifact_gated_small_molecule_interfaces(tmp_path):
    artifact_path = _artifact_tsv(tmp_path, "BME", "PEG")
    metadata_df = pd.DataFrame(
        [
            _row("P_1", is_protein=True, cluster_id=100),
            _row("P_2", is_protein=True, cluster_id=101),
            _row(
                "B_1",
                is_small_molecule=True,
                ccd="BME",
                cluster_id=200,
                sm_protein_contacts=_contacts(("P_1", 12), ("P_2", 13)),
                binding_evidence=True,
            ),
            _row(
                "A_1",
                is_small_molecule=True,
                ccd="ADP",
                cluster_id=201,
                sm_protein_contacts=_contacts(("P_1", 21)),
            ),
            _row(
                "X_1",
                is_small_molecule=True,
                ccd="PEG",
                cluster_id=202,
                sm_protein_contacts=_contacts(("P_1", 30)),
            ),
        ]
    )
    metadata_df = collect_external_evidence(metadata_df)

    interface_df = build_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        proto_cfg={
            "min_protein_donor_atoms": 3,
            "min_avg_occupancy_nonpolymer": 0.5,
            "small_molecule_artifact_list_path": str(artifact_path),
        },
    )

    sm_df = interface_df[interface_df["interface_type"] == "bmsm_protein"].sort_values("q_pn_unit_iid")
    assert sm_df["q_pn_unit_iid"].tolist() == ["A_1", "B_1"]

    bme = sm_df[sm_df["q_pn_unit_iid"] == "B_1"].iloc[0]
    assert bme["query_pn_unit_iids"] == ["B_1", "P_1", "P_2"]
    assert bme["ligand_pn_unit_iids"] == ("B_1",)
    assert bme["protein_pn_unit_iids"] == ("P_1", "P_2")
    assert bme["protein_cluster_multiset"] == (100, 101)
    assert bme["ligand_ccd_key"] == ("ccd", "BME")
    assert bme["n_contacting_protein_atoms"] == 25


def test_build_proto_interface_df_adds_unique_protein_interface_tuples_with_contact_sum():
    metadata_df = pd.DataFrame(
        [
            _row(
                "P_1",
                is_protein=True,
                cluster_id=100,
                protein_contacts=_protein_contacts(("P_2", 10, 3.0), ("P_3", 5, 4.0)),
            ),
            _row(
                "P_2",
                is_protein=True,
                cluster_id=101,
                protein_contacts=_protein_contacts(("P_1", 8, 3.0), ("P_3", 7, 4.5)),
            ),
            _row(
                "P_3",
                is_protein=True,
                cluster_id=102,
                protein_contacts=_protein_contacts(("P_1", 4, 4.0), ("P_2", 6, 4.5)),
            ),
            _row("S_1", is_protein=False, ccd="ATP", cluster_id=200),
        ]
    )
    metadata_df = collect_external_evidence(metadata_df)

    interface_df = build_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        proto_cfg={"min_protein_donor_atoms": 3, "min_avg_occupancy_nonpolymer": 0.5},
    )

    ppi_df = interface_df[interface_df["interface_type"] == "protein_protein"]
    assert len(ppi_df) == 1
    row = ppi_df.iloc[0]
    assert row["query_pn_unit_iids"] == ["P_1", "P_2", "P_3"]
    assert row["ligand_pn_unit_iids"] == ()
    assert row["protein_pn_unit_iids"] == ("P_1", "P_2", "P_3")
    assert row["protein_cluster_multiset"] == (100, 101, 102)
    assert row["ligand_ccd_key"] == ("protein_interface", "none")
    assert bool(row["query_pn_unit_iids_only"]) is True
    assert row["n_protein_protein_contacts"] == 22


def test_parse_train_dfs_uses_row_level_ppi_query_only_and_global_override():
    metadata_df = pd.DataFrame(
        [
            _row(
                "P_1",
                is_protein=True,
                cluster_id=100,
                protein_contacts=_protein_contacts(("P_2", 10, 3.0)),
            ),
            _row(
                "P_2",
                is_protein=True,
                cluster_id=101,
                protein_contacts=_protein_contacts(("P_1", 8, 3.0)),
            ),
            _row("Z_1", is_metal=True, ccd="ZN", cluster_id=300, contacts=_contacts(("P_1", 3))),
        ]
    )
    metadata_df = collect_external_evidence(metadata_df)
    interface_df = build_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        proto_cfg={"min_protein_donor_atoms": 3, "min_avg_occupancy_nonpolymer": 0.5},
    )
    monomer_df = pd.DataFrame([_row("P_9", is_protein=True, cluster_id=999)])
    monomer_df["example_id"] = "toy-protein-monomer"
    monomer_df.set_index("example_id", inplace=True, drop=False)

    dataset = object.__new__(ProtoSDDataset)
    dataset.protein_monomer_chain_df = monomer_df
    dataset.interface_df = interface_df
    dataset.proto_cfg = {"query_pn_unit_iids_only": False}

    parsed_df = dataset._parse_train_dfs()
    ppi_id = interface_df[interface_df["interface_type"] == "protein_protein"].index[0]
    metal_id = interface_df[interface_df["interface_type"] == "bmm_protein"].index[0]
    parsed_ppi = parsed_df.loc[ppi_id]
    parsed_metal = parsed_df.loc[metal_id]

    assert parsed_ppi["query_pn_unit_iids_only"] is True
    assert parsed_ppi["query_pn_unit_iids"] == ["P_1", "P_2"]
    assert parsed_metal.get("query_pn_unit_iids_only") is not True
    assert "query_pn_unit_iids_only" not in parsed_ppi["extra_info"]

    dataset.proto_cfg = {"query_pn_unit_iids_only": True}
    parsed_df = dataset._parse_train_dfs()
    parsed_metal = parsed_df.loc[metal_id]

    assert parsed_metal["query_pn_unit_iids_only"] is True


def test_build_proto_interface_df_filters_protein_interfaces_by_distance_cutoff():
    metadata_df = pd.DataFrame(
        [
            _row(
                "P_1",
                is_protein=True,
                cluster_id=100,
                protein_contacts=_protein_contacts(("P_2", 10, 6.0)),
            ),
            _row(
                "P_2",
                is_protein=True,
                cluster_id=101,
                protein_contacts=_protein_contacts(("P_1", 8, 6.0)),
            ),
        ]
    )
    metadata_df = collect_external_evidence(metadata_df)

    interface_df = build_proto_interface_df(
        metadata_df,
        protein_df=metadata_df[metadata_df["q_pn_unit_is_protein"]].copy(),
        dataset_name="toy",
        proto_cfg={"min_protein_donor_atoms": 3, "min_avg_occupancy_nonpolymer": 0.5},
    )

    assert interface_df.empty


def test_add_proto_sampling_weights_uses_actual_metal_ccd_key():
    monomer_df = pd.DataFrame(
        [
            {"q_pn_unit_cluster_id": 100},
            {"q_pn_unit_cluster_id": 200},
        ],
        index=["mono1", "mono2"],
    )
    interface_df = pd.DataFrame(
        [
            {
                "protein_cluster_multiset": (100,),
                "interface_type": "bmm_protein",
                "ligand_ccd_key": ("ccd", "ZN"),
            },
            {
                "protein_cluster_multiset": (200,),
                "interface_type": "bmm_protein",
                "ligand_ccd_key": ("ccd", "FE"),
            },
        ],
        index=["zn_iface", "fe_iface"],
    )

    _, weighted_interface = add_proto_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface={"alpha_protein_metal": 1.0},
        k_percentile=100.0,
    )

    assert weighted_interface.loc["zn_iface", "pair_cluster"][0] == ("ccd", "ZN")
    assert weighted_interface.loc["fe_iface", "pair_cluster"][0] == ("ccd", "FE")


def test_add_proto_sampling_weights_uses_interface_type_alpha_for_ppi_and_small_molecule():
    monomer_df = pd.DataFrame(
        [
            {"q_pn_unit_cluster_id": 100},
            {"q_pn_unit_cluster_id": 200},
        ],
        index=["mono1", "mono2"],
    )
    interface_df = pd.DataFrame(
        [
            {
                "protein_cluster_multiset": (100,),
                "interface_type": "bmm_protein",
                "ligand_ccd_key": ("ccd", "ZN"),
            },
            {
                "protein_cluster_multiset": (100, 200),
                "interface_type": "protein_protein",
                "ligand_ccd_key": ("protein_interface", "none"),
            },
            {
                "protein_cluster_multiset": (200,),
                "interface_type": "bmsm_protein",
                "ligand_ccd_key": ("ccd", "ADP"),
            },
        ],
        index=["metal_iface", "ppi_iface", "sm_iface"],
    )

    _, weighted_interface = add_proto_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface={
            "alpha_protein_metal": 3.0,
            "alpha_protein_small_molecule": 4.0,
            "alpha_protein_protein": 2.0,
        },
        k_percentile=100.0,
    )

    assert weighted_interface.loc["metal_iface", "alpha"] == 3.0
    assert weighted_interface.loc["ppi_iface", "alpha"] == 2.0
    assert weighted_interface.loc["sm_iface", "alpha"] == 4.0
    assert weighted_interface.loc["ppi_iface", "pair_cluster"][0] == ("protein_interface", "none")


def test_add_proto_sampling_weights_applies_context_weight_before_equalization():
    monomer_df = pd.DataFrame(
        [
            {"q_pn_unit_cluster_id": 100},
            {"q_pn_unit_cluster_id": 200},
            {"q_pn_unit_cluster_id": 300},
        ],
        index=["mono1", "mono2", "mono3"],
    )
    interface_df = pd.DataFrame(
        [
            {
                "protein_cluster_multiset": (100,),
                "interface_type": "bmm_protein",
                "ligand_ccd_key": ("ccd", "ZN"),
                "n_prot": 1,
            },
            {
                "protein_cluster_multiset": (200, 300),
                "interface_type": "bmm_protein",
                "ligand_ccd_key": ("ccd", "FE"),
                "n_prot": 2,
            },
            {
                "protein_cluster_multiset": (100, 200),
                "interface_type": "protein_protein",
                "ligand_ccd_key": ("protein_interface", "none"),
                "n_prot": 2,
            },
        ],
        index=["single_metal", "multi_metal", "ppi"],
    )

    _, weighted_interface = add_proto_sampling_weights(
        monomer_df,
        interface_df,
        alphas_interface={
            "alpha_protein_metal": 2.0,
            "alpha_protein_protein": 5.0,
        },
        k_percentile=100.0,
        single_protein_context_weight=0.25,
        multi_protein_context_weight=0.1,
    )

    assert weighted_interface.loc["single_metal", "context_weight"] == 0.25
    assert weighted_interface.loc["multi_metal", "context_weight"] == 0.1
    assert weighted_interface.loc["ppi", "context_weight"] == 0.1
    assert weighted_interface.loc["single_metal", "sampling_weight"] == 0.5
    assert weighted_interface.loc["multi_metal", "sampling_weight"] == 0.2
    assert weighted_interface.loc["ppi", "sampling_weight"] == 0.5


def test_add_proto_chain_counts_info_marks_ppi_without_metal():
    interface_df = pd.DataFrame(
        [
            {
                "protein_cluster_multiset": (100, 101, 102),
                "interface_type": "protein_protein",
            },
            {
                "protein_cluster_multiset": (100,),
                "interface_type": "bmm_protein",
            },
            {
                "protein_cluster_multiset": (101,),
                "interface_type": "bmsm_protein",
            },
        ],
        index=["ppi", "metal", "sm"],
    )

    counted = add_proto_chain_counts_info(interface_df)

    assert counted.loc["ppi", "n_prot"] == 3
    assert counted.loc["ppi", "n_metal"] == 0
    assert counted.loc["ppi", "n_small_molecule"] == 0
    assert counted.loc["metal", "n_prot"] == 1
    assert counted.loc["metal", "n_metal"] == 1
    assert counted.loc["metal", "n_small_molecule"] == 0
    assert counted.loc["sm", "n_prot"] == 1
    assert counted.loc["sm", "n_metal"] == 0
    assert counted.loc["sm", "n_small_molecule"] == 1


def test_getitem_leaves_cached_atom_array_filtering_to_featurizer():
    dataset = object.__new__(ProtoSDDataset)
    dataset.phase = "val"
    dataset.parsed_df = pd.DataFrame(
        [
            {
                "example_id": "toy-example",
                "query_pn_unit_iids": ["Z_1", "P_1"],
                "extra_info": {"pdb_id": "1abc"},
            }
        ],
        index=["toy-example"],
    )
    dataset._load_cached_example = lambda pdb_id: {"atom_array": "full cached assembly"}
    dataset.featurizer = lambda example: example

    result = ProtoSDDataset.__getitem__(dataset, 0)

    assert result["atom_array"] == "full cached assembly"
    assert result["query_pn_unit_iids"] == ["Z_1", "P_1"]
    assert result["phase"] == "val"


def test_build_sd_datamodule_accepts_proto_selector(monkeypatch):
    import allatom_design.data.datasets.atomworks_sd_dataset_proto as proto_module

    def fake_init(self, cfg):
        self.cfg = cfg

    monkeypatch.setattr(proto_module.AtomworksSDProtoDataModule, "__init__", fake_init)

    datamodule = build_sd_datamodule(OmegaConf.create({"dataset_impl": "proto"}))

    assert isinstance(datamodule, proto_module.AtomworksSDProtoDataModule)
