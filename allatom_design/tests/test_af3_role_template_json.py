from __future__ import annotations

import json

import numpy as np
import pytest
from biotite.structure import AtomArray

import atomworks.enums as aw_enums
from allatom_design.eval.structure_prediction.af3_json import (
    build_af3_chain_id_to_pn_unit_iid,
    make_af3_json,
)


def _sample_dict(tmp_path, *, template_pn_unit_iids: list[str]) -> dict:
    atom_array = AtomArray(7)
    atom_array.coord = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [4.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    atom_array.chain_id = np.asarray(["A"] * 3 + ["C"] * 3 + ["L"])
    atom_array.res_id = np.asarray([1, 2, 3, 1, 2, 3, 1])
    atom_array.res_name = np.asarray(["ALA"] * 3 + ["GLY"] * 3 + ["FE"])
    atom_array.atom_name = np.asarray(["CA"] * 6 + ["FE"])
    atom_array.element = np.asarray(["C"] * 6 + ["FE"])
    atom_array.hetero = np.asarray([False] * 6 + [True])
    atom_array.set_annotation(
        "pn_unit_iid", np.asarray(["A_1"] * 3 + ["C_1"] * 3 + ["L_1"])
    )
    atom_array.set_annotation(
        "chain_type",
        np.asarray(
            [aw_enums.ChainType.POLYPEPTIDE_L] * 6
            + [aw_enums.ChainType.NON_POLYMER],
            dtype=object,
        ),
    )
    template_path = tmp_path / "template.cif"
    template_path.write_text("template fixture")
    return {
        "toy": {
            "designed_sample_id": ["toy_sample0"],
            "designed_sample_atom_array": [atom_array],
            "designed_sample_path_for_af3_tc": [str(template_path)],
            "pdb_chain_info": {
                "protein_pn_unit_iids": ["A_1", "C_1"],
                "ligand_pn_unit_iids": ["L_1"],
                "ligand_ccd_codes": ["FE"],
            },
            "pn_unit_roles": {
                "binder_pn_unit_iids": ["A_1"],
                "context_pn_unit_iids": ["C_1", "L_1"],
                "frame_pn_unit_iids": ["A_1"],
                "template_pn_unit_iids": template_pn_unit_iids,
            },
        }
    }


def test_tc_json_templates_only_the_selected_protein_chain(tmp_path) -> None:
    sample_dict = _sample_dict(tmp_path, template_pn_unit_iids=["C_1"])
    tc_dir = tmp_path / "tc"

    make_af3_json(
        af3_ss_input_dir=None,
        af3_tc_input_dir=tc_dir,
        sample_dict=sample_dict,
        json_config={"model_seeds": [42]},
        make_ss_input=False,
        make_tc_input=True,
    )

    with sample_dict["toy"]["af3_tc_json_paths"][0].open() as handle:
        payload = json.load(handle)
    proteins = {
        entry["protein"]["id"]: entry["protein"]["templates"]
        for entry in payload["sequences"]
        if "protein" in entry
    }
    assert proteins["A"] == []
    assert len(proteins["C"]) == 1


def test_ss_and_tc_json_share_chain_ids_for_transformed_pn_units(tmp_path) -> None:
    sample_dict = _sample_dict(tmp_path, template_pn_unit_iids=["C_1"])
    subsample = sample_dict["toy"]
    atom_array = subsample["designed_sample_atom_array"][0]
    atom_array.chain_id = np.asarray(
        ["CA" if chain_id == "C" else chain_id for chain_id in atom_array.chain_id]
    )
    source_iid_by_chain_id = {"A": "A_2", "CA": "CA_3", "L": "L_2"}
    atom_array.pn_unit_iid = np.asarray(
        [source_iid_by_chain_id[chain_id] for chain_id in atom_array.chain_id]
    )
    subsample["pdb_chain_info"]["protein_pn_unit_iids"] = ["A_2", "CA_3"]
    subsample["pdb_chain_info"]["ligand_pn_unit_iids"] = ["L_2"]
    subsample["pn_unit_roles"] = {
        "binder_pn_unit_iids": ["A_2"],
        "context_pn_unit_iids": ["CA_3", "L_2"],
        "frame_pn_unit_iids": ["A_2"],
        "template_pn_unit_iids": ["CA_3"],
    }

    make_af3_json(
        af3_ss_input_dir=tmp_path / "ss",
        af3_tc_input_dir=tmp_path / "tc",
        sample_dict=sample_dict,
        json_config={"model_seeds": [42]},
        make_ss_input=True,
        make_tc_input=True,
    )

    for json_paths_key in ("af3_ss_json_paths", "af3_tc_json_paths"):
        with subsample[json_paths_key][0].open() as handle:
            payload = json.load(handle)
        assert [
            sequence["protein"]["id"]
            if "protein" in sequence
            else sequence["ligand"]["id"]
            for sequence in payload["sequences"]
        ] == ["A", "CA", "L"]


def test_ss_sparse_index_is_default_and_gap_filling_is_explicit(tmp_path) -> None:
    sample_dict = _sample_dict(tmp_path, template_pn_unit_iids=["C_1"])
    atom_array = sample_dict["toy"]["designed_sample_atom_array"][0]
    atom_array.res_id[:3] = np.asarray([10, 12, 13])
    sample_dict["toy"]["native_res_name_by_chain_res_id"] = {("A", 11): "GLY"}

    make_af3_json(
        af3_ss_input_dir=tmp_path / "sparse",
        sample_dict=sample_dict,
        json_config={"model_seeds": [42]},
        make_tc_input=False,
    )
    with sample_dict["toy"]["af3_ss_json_paths"][0].open() as handle:
        sparse_payload = json.load(handle)
    assert sparse_payload["sequences"][0]["protein"]["sequence"] == "AAA"
    assert sample_dict["toy"]["af3_ss_residue_index_by_chain"] == [
        {"A": [10, 12, 13]}
    ]

    gap_filled_sample_dict = _sample_dict(
        tmp_path,
        template_pn_unit_iids=["C_1"],
    )
    gap_filled_atoms = gap_filled_sample_dict["toy"][
        "designed_sample_atom_array"
    ][0]
    gap_filled_atoms.res_id[:3] = np.asarray([10, 12, 13])
    gap_filled_sample_dict["toy"]["native_res_name_by_chain_res_id"] = {
        ("A", 11): "GLY"
    }
    make_af3_json(
        af3_ss_input_dir=tmp_path / "gap_filled",
        sample_dict=gap_filled_sample_dict,
        json_config={
            "model_seeds": [42],
            # (JH) fixed: this is the only switch that restores legacy gap filling.
            "ss_preserve_residue_index_gaps": False,
        },
        make_tc_input=False,
    )
    with gap_filled_sample_dict["toy"]["af3_ss_json_paths"][0].open() as handle:
        gap_filled_payload = json.load(handle)
    assert gap_filled_payload["sequences"][0]["protein"]["sequence"] == "AGAA"
    assert gap_filled_sample_dict["toy"]["af3_ss_residue_index_by_chain"] == [{}]


def test_af3_chain_mapping_preserves_multiletter_id_with_empty_ligands() -> None:
    assert build_af3_chain_id_to_pn_unit_iid(
        protein_pn_unit_iids=["CA_1"],
        ligand_pn_unit_iids=[],
    ) == {"CA": "CA_1"}


def test_af3_chain_mapping_accepts_atomworks_compound_pn_iid() -> None:
    assert build_af3_chain_id_to_pn_unit_iid(
        protein_pn_unit_iids=["CA_3"],
        ligand_pn_unit_iids=["B_1,C_1"],
    ) == {"CA": "CA_3", "B": "B_1,C_1"}


def test_af3_chain_mapping_rejects_compound_transformation_mismatch() -> None:
    with pytest.raises(ValueError, match="same transformation ID"):
        build_af3_chain_id_to_pn_unit_iid(
            protein_pn_unit_iids=[],
            ligand_pn_unit_iids=["B_1,C_2"],
        )


def test_af3_chain_mapping_rejects_source_iid_collision() -> None:
    with pytest.raises(ValueError, match="AF3 chain ID collision"):
        build_af3_chain_id_to_pn_unit_iid(
            protein_pn_unit_iids=["A_2"],
            ligand_pn_unit_iids=["A_3"],
        )


@pytest.mark.parametrize(
    ("protein_pn_unit_iids", "ligand_pn_unit_iids"),
    [
        (["A_1", "A_1"], []),
        (["A_1"], ["A_1"]),
    ],
    ids=["within-list", "cross-list"],
)
def test_af3_chain_mapping_rejects_duplicate_exact_iid(
    protein_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
) -> None:
    with pytest.raises(ValueError, match="Duplicate AF3 chain ID"):
        build_af3_chain_id_to_pn_unit_iid(
            protein_pn_unit_iids=protein_pn_unit_iids,
            ligand_pn_unit_iids=ligand_pn_unit_iids,
        )


@pytest.mark.parametrize("pn_unit_iid", ["A", "_2", "A_", "A_1,", "A_1,C"])
def test_af3_chain_mapping_rejects_malformed_source_iid(pn_unit_iid: str) -> None:
    with pytest.raises(ValueError, match="Malformed pn_unit_iid component"):
        build_af3_chain_id_to_pn_unit_iid(
            protein_pn_unit_iids=[pn_unit_iid],
            ligand_pn_unit_iids=[],
        )


def test_tc_json_rejects_nonprotein_template_unit(tmp_path) -> None:
    sample_dict = _sample_dict(tmp_path, template_pn_unit_iids=["L_1"])
    with pytest.raises(ValueError, match="template.*protein"):
        make_af3_json(
            af3_ss_input_dir=None,
            af3_tc_input_dir=tmp_path / "tc",
            sample_dict=sample_dict,
            json_config={"model_seeds": [42]},
            make_ss_input=False,
            make_tc_input=True,
        )


def test_legacy_tc_json_rejects_ambiguous_multiple_protein_templates(tmp_path) -> None:
    sample_dict = _sample_dict(tmp_path, template_pn_unit_iids=["C_1"])
    sample_dict["toy"].pop("pn_unit_roles")

    with pytest.raises(ValueError, match="requires exactly one protein"):
        make_af3_json(
            af3_ss_input_dir=None,
            af3_tc_input_dir=tmp_path / "tc",
            sample_dict=sample_dict,
            json_config={"model_seeds": [42]},
            make_ss_input=False,
            make_tc_input=True,
        )
