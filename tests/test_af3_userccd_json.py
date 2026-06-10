from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from biotite.structure import AtomArray, get_residue_starts

import atomworks.enums as aw_enums
import allatom_design.eval.utils.design_sequence_two_stage as two_stage
from allatom_design.eval.utils import folding_utils
from allatom_design.eval.utils.data_utils import (
    collect_design_outputs,
    resolve_query_pn_unit_iids,
)
from allatom_design.eval.utils.folding_utils import make_af3_json
from allatom_design.utils.atom_array_utils import (
    get_res_name_by_chain_res_id,
    insert_unk_residues_for_gaps_in_atom_array,
)


def _make_protein_metal_atom_array() -> AtomArray:
    records = [
        ("A", 1, "ALA", False, "N", "N", [0.0, 0.0, 0.0], "A_1"),
        ("A", 1, "ALA", False, "CA", "C", [1.0, 0.0, 0.0], "A_1"),
        ("A", 1, "ALA", False, "C", "C", [2.0, 0.0, 0.0], "A_1"),
        ("A", 1, "ALA", False, "O", "O", [3.0, 0.0, 0.0], "A_1"),
        ("B", 1, "l01", True, "CU1", "CU", [1.0, 2.0, 3.0], "B_1"),
    ]
    atom_array = AtomArray(len(records))
    atom_array.chain_id = np.array([record[0] for record in records])
    atom_array.res_id = np.array([record[1] for record in records])
    atom_array.res_name = np.array([record[2] for record in records])
    atom_array.hetero = np.array([record[3] for record in records], dtype=bool)
    atom_array.atom_name = np.array([record[4] for record in records])
    atom_array.element = np.array([record[5] for record in records])
    atom_array.coord = np.array([record[6] for record in records], dtype=np.float32)
    atom_array.set_annotation("pn_unit_iid", np.array([record[7] for record in records]))
    return atom_array


def _make_gap_protein_atom_array(residue_names: dict[int, str]) -> AtomArray:
    res_ids = sorted(residue_names)
    atom_array = AtomArray(len(res_ids))
    atom_array.chain_id = np.array(["A"] * len(res_ids))
    atom_array.res_id = np.array(res_ids)
    atom_array.res_name = np.array([residue_names[res_id] for res_id in res_ids])
    atom_array.atom_name = np.array(["CA"] * len(res_ids))
    atom_array.element = np.array(["C"] * len(res_ids))
    atom_array.coord = np.array([[float(res_id), 0.0, 0.0] for res_id in res_ids], dtype=np.float32)
    atom_array.hetero = np.zeros(len(res_ids), dtype=bool)
    atom_array.occupancy = np.ones(len(res_ids), dtype=float)
    atom_array.b_factor = np.zeros(len(res_ids), dtype=float)

    atom_array.set_annotation("alt_atom_id", np.array(["CA"] * len(res_ids)))
    atom_array.set_annotation("atom_id", np.arange(1, len(res_ids) + 1))
    atom_array.set_annotation("stereo", np.array(["S"] * len(res_ids)))
    atom_array.set_annotation("is_aromatic", np.zeros(len(res_ids), dtype=bool))
    atom_array.set_annotation("is_backbone_atom", np.ones(len(res_ids), dtype=bool))
    atom_array.set_annotation("is_polymer", np.ones(len(res_ids), dtype=bool))
    atom_array.set_annotation("charge", np.zeros(len(res_ids), dtype=int))
    atom_array.set_annotation("atomic_number", np.full(len(res_ids), 6, dtype=int))
    atom_array.set_annotation("atomize", np.ones(len(res_ids), dtype=bool))
    atom_array.set_annotation("is_covalent_modification", np.zeros(len(res_ids), dtype=bool))
    atom_array.set_annotation("uses_alt_atom_id", np.zeros(len(res_ids), dtype=bool))
    atom_array.set_annotation("ins_code", np.array([""] * len(res_ids)))
    atom_array.set_annotation("pn_unit_id", np.array(["A"] * len(res_ids)))
    atom_array.set_annotation("molecule_id", np.zeros(len(res_ids), dtype=int))
    atom_array.set_annotation("chain_entity", np.array(["1"] * len(res_ids)))
    atom_array.set_annotation("pn_unit_entity", np.array(["1"] * len(res_ids)))
    atom_array.set_annotation("molecule_entity", np.array(["1"] * len(res_ids)))
    atom_array.set_annotation("transformation_id", np.array(["1"] * len(res_ids)))
    atom_array.set_annotation("chain_iid", np.array(["A_1"] * len(res_ids)))
    atom_array.set_annotation("pn_unit_iid", np.array(["A_1"] * len(res_ids)))
    atom_array.set_annotation("molecule_iid", np.zeros(len(res_ids), dtype=int))
    atom_array.set_annotation(
        "chain_type",
        np.full(len(res_ids), int(aw_enums.ChainType.POLYPEPTIDE_L), dtype=int),
    )
    return atom_array


def _make_modified_protein_atom_array() -> AtomArray:
    records = [
        ("A", 1, "ALA", False, "N", "N", [0.0, 0.0, 0.0], "A_1"),
        ("A", 1, "ALA", False, "CA", "C", [1.0, 0.0, 0.0], "A_1"),
        ("A", 1, "ALA", False, "C", "C", [2.0, 0.0, 0.0], "A_1"),
        ("A", 1, "ALA", False, "O", "O", [3.0, 0.0, 0.0], "A_1"),
        ("A", 2, "MLY", True, "N", "N", [0.0, 1.0, 0.0], "A_1"),
        ("A", 2, "MLY", True, "CA", "C", [1.0, 1.0, 0.0], "A_1"),
        ("A", 2, "MLY", True, "C", "C", [2.0, 1.0, 0.0], "A_1"),
        ("A", 2, "MLY", True, "O", "O", [3.0, 1.0, 0.0], "A_1"),
        ("A", 3, "GLY", False, "N", "N", [0.0, 2.0, 0.0], "A_1"),
        ("A", 3, "GLY", False, "CA", "C", [1.0, 2.0, 0.0], "A_1"),
        ("A", 3, "GLY", False, "C", "C", [2.0, 2.0, 0.0], "A_1"),
        ("A", 3, "GLY", False, "O", "O", [3.0, 2.0, 0.0], "A_1"),
    ]
    atom_array = AtomArray(len(records))
    atom_array.chain_id = np.array([record[0] for record in records])
    atom_array.res_id = np.array([record[1] for record in records])
    atom_array.res_name = np.array([record[2] for record in records])
    atom_array.hetero = np.array([record[3] for record in records], dtype=bool)
    atom_array.atom_name = np.array([record[4] for record in records])
    atom_array.element = np.array([record[5] for record in records])
    atom_array.coord = np.array([record[6] for record in records], dtype=np.float32)
    atom_array.set_annotation("pn_unit_iid", np.array([record[7] for record in records]))
    return atom_array


def _make_designed_protein_ligand_atom_array(*, ligand_res_name: str = "UNK") -> AtomArray:
    records = [
        ("A", 1, "ALA", False, "N", "N", [0.0, 0.0, 0.0], "A_1", aw_enums.ChainType.POLYPEPTIDE_L),
        ("A", 1, "ALA", False, "CA", "C", [1.0, 0.0, 0.0], "A_1", aw_enums.ChainType.POLYPEPTIDE_L),
        ("A", 1, "ALA", False, "C", "C", [2.0, 0.0, 0.0], "A_1", aw_enums.ChainType.POLYPEPTIDE_L),
        ("A", 1, "ALA", False, "O", "O", [3.0, 0.0, 0.0], "A_1", aw_enums.ChainType.POLYPEPTIDE_L),
        ("E", 1, ligand_res_name, True, "C1", "C", [1.0, 2.0, 3.0], "E_1", aw_enums.ChainType.NON_POLYMER),
    ]
    atom_array = AtomArray(len(records))
    atom_array.chain_id = np.array([record[0] for record in records])
    atom_array.res_id = np.array([record[1] for record in records])
    atom_array.res_name = np.array([record[2] for record in records])
    atom_array.hetero = np.array([record[3] for record in records], dtype=bool)
    atom_array.atom_name = np.array([record[4] for record in records])
    atom_array.element = np.array([record[5] for record in records])
    atom_array.coord = np.array([record[6] for record in records], dtype=np.float32)
    atom_array.set_annotation("pn_unit_iid", np.array([record[7] for record in records]))
    atom_array.set_annotation("chain_type", np.array([int(record[8]) for record in records], dtype=np.int8))
    return atom_array


def _res_names_by_id(atom_array: AtomArray) -> dict[int, str]:
    starts = get_residue_starts(atom_array)
    return {
        int(atom_array.res_id[idx]): str(atom_array.res_name[idx])
        for idx in starts
    }


def _write_af3_ligand_json(json_path: Path, ccd_code: str) -> None:
    json_path.write_text(
        json.dumps(
            {
                "name": json_path.stem,
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "A",
                            "unpairedMsa": "",
                            "pairedMsa": "",
                            "templates": [],
                        }
                    },
                    {"ligand": {"id": "B", "ccdCodes": [ccd_code]}},
                ],
                "modelSeeds": [42],
                "dialect": "alphafold3",
                "version": 2,
            }
        )
    )


def _minimal_af3_inference_config(ss_config: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        base={
            "model_dir": "/tmp/model",
            "db_dir": "/tmp/db",
            "flash_attention_implementation": "triton",
        },
        ss=ss_config or {
            "num_recycles": 1,
            "num_diffusion_samples": 1,
            "max_templates": 0,
            "ligand_protein_template_conditioning_mode": 0,
        },
    )


def test_af3_json_detects_glycan_ligand_ccd_from_af3_sets(tmp_path: Path) -> None:
    glycan_json = tmp_path / "glycan.json"
    non_glycan_json = tmp_path / "non_glycan.json"
    _write_af3_ligand_json(glycan_json, "NAG")
    _write_af3_ligand_json(non_glycan_json, "FAD")

    assert folding_utils._json_needs_fix_standalone_glycans(glycan_json, {}) is True
    assert folding_utils._json_needs_fix_standalone_glycans(non_glycan_json, {}) is False


def test_run_af3_single_sequence_subprocess_adds_glycan_flag_only_for_glycans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = []

    def fake_run(cmd, check, env):
        commands.append(cmd)

    monkeypatch.setattr(folding_utils.subprocess, "run", fake_run)

    glycan_json = tmp_path / "glycan.json"
    non_glycan_json = tmp_path / "non_glycan.json"
    _write_af3_ligand_json(glycan_json, "NAG")
    _write_af3_ligand_json(non_glycan_json, "FAD")
    inference_config = _minimal_af3_inference_config()

    folding_utils.run_af3_single_sequence(
        json_path=str(glycan_json),
        out_dir=str(tmp_path / "glycan_out"),
        runner_path="/tmp/run_alphafold.py",
        inference_config=inference_config,
        use_subprocess=True,
    )
    folding_utils.run_af3_single_sequence(
        json_path=str(non_glycan_json),
        out_dir=str(tmp_path / "non_glycan_out"),
        runner_path="/tmp/run_alphafold.py",
        inference_config=inference_config,
        use_subprocess=True,
    )

    assert "--fix_standalone_glycans=True" in commands[0]
    assert "--fix_standalone_glycans=True" not in commands[1]


def test_run_af3_inprocess_passes_auto_glycan_flag_to_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passed_flags = []

    class FakeRunner:
        def process_fold_input(self, **kwargs):
            passed_flags.append(kwargs["fix_standalone_glycans"])

    monkeypatch.setattr(
        folding_utils,
        "_get_af3_model_runner_and_config",
        lambda runner_path, inference_config, mode: (FakeRunner(), object(), object()),
    )

    glycan_json = tmp_path / "glycan.json"
    non_glycan_json = tmp_path / "non_glycan.json"
    _write_af3_ligand_json(glycan_json, "NAG")
    _write_af3_ligand_json(non_glycan_json, "FAD")

    folding_utils._run_af3_inprocess(
        json_path=str(glycan_json),
        out_dir=str(tmp_path / "glycan_out"),
        runner_path="/tmp/run_alphafold.py",
        inference_config={"ss": {}},
        mode="ss",
    )
    folding_utils._run_af3_inprocess(
        json_path=str(non_glycan_json),
        out_dir=str(tmp_path / "non_glycan_out"),
        runner_path="/tmp/run_alphafold.py",
        inference_config={"ss": {}},
        mode="ss",
    )

    assert passed_flags == [True, False]


def test_fix_standalone_glycans_config_override_wins_over_auto_detection(tmp_path: Path) -> None:
    glycan_json = tmp_path / "glycan.json"
    non_glycan_json = tmp_path / "non_glycan.json"
    _write_af3_ligand_json(glycan_json, "NAG")
    _write_af3_ligand_json(non_glycan_json, "FAD")

    assert (
        folding_utils._json_needs_fix_standalone_glycans(
            glycan_json,
            {"fix_standalone_glycans": False},
        )
        is False
    )
    assert (
        folding_utils._json_needs_fix_standalone_glycans(
            non_glycan_json,
            {"fix_standalone_glycans": True},
        )
        is True
    )


def test_make_af3_json_uses_userccd_component_ids_without_metal_rewrite(tmp_path: Path) -> None:
    sample_dict = {
        "length_150_Cu_sample_0": {
            "designed_sample_id": ["length_150_Cu_sample_0_sample0"],
            "designed_sample_atom_array": [_make_protein_metal_atom_array()],
            "pdb_chain_info": {
                "protein_pn_unit_iids": ["A_1"],
                "ligand_pn_unit_iids": ["B_1"],
                "ligand_ccd_codes": ["l01"],
                "af3_ligand_ccd_codes": ["S179002"],
            },
        }
    }

    make_af3_json(
        af3_ss_input_dir=tmp_path,
        sample_dict=sample_dict,
        json_config={
            "model_seeds": [42],
            "version": 4,
            "user_ccd_path": "/tmp/example_components_userccd.cif",
        },
    )

    payload = json.loads((tmp_path / "length_150_Cu_sample_0_sample0.json").read_text())
    ligand_entries = [entry["ligand"] for entry in payload["sequences"] if "ligand" in entry]

    assert payload["version"] == 4
    assert payload["userCCDPath"] == "/tmp/example_components_userccd.cif"
    assert ligand_entries == [{"id": "B", "ccdCodes": ["S179002"]}]


def test_resolve_query_pn_unit_iids_uses_json_fallback_when_legacy_column_is_blank() -> None:
    atom_array = _make_designed_protein_ligand_atom_array()
    sampling_inputs_df = pd.DataFrame(
        {
            "pdb_id": ["8clq"],
            "query_pn_unit_iids": [""],
            "query_pn_unit_iids_json": ['["A_1", "E_1"]'],
        }
    )

    assert resolve_query_pn_unit_iids(
        atom_array=atom_array,
        sampling_inputs_df=sampling_inputs_df,
        pdb_id="8clq",
    ) == ["A_1", "E_1"]


def test_collect_design_outputs_preserves_sampling_ligand_ccd_for_af3_json(tmp_path: Path) -> None:
    sampling_inputs_df = pd.DataFrame(
        {
            "pdb_id": ["8clq"],
            "query_pn_unit_iids": [""],
            "query_pn_unit_iids_json": ['["A_1", "E_1"]'],
            "ccd_code": ["ZGR"],
        }
    )
    outputs = {
        "8clq": {
            "designed_sample_id": ["8clq_sample0"],
            "seq_recovery_metrics": [{"seq_recovery_ratio": 1.0}],
            "designed_sample_atom_array": [_make_designed_protein_ligand_atom_array(ligand_res_name="UNK")],
            "designed_sample_seq": ["A"],
            "designed_sample_path": [str(tmp_path / "8clq_sample0.cif")],
            "designed_sample_path_for_af3_tc": [str(tmp_path / "8clq_sample0_tc.cif")],
            "U": [0.0],
        }
    }
    sample_dict = {
        "8clq": {
            "input_sample_path": str(tmp_path / "8clq.cif"),
            "input_sample_id": "8clq",
        }
    }

    collected = collect_design_outputs(
        sample_dict=sample_dict,
        outputs=outputs,
        log_dir_per_ckpt=tmp_path,
        csv_suffix="",
        guidance_cfg=None,
        sampling_inputs_df=sampling_inputs_df,
    )

    assert collected["8clq"]["pdb_chain_info"]["protein_pn_unit_iids"] == ["A_1"]
    assert collected["8clq"]["pdb_chain_info"]["ligand_pn_unit_iids"] == ["E_1"]
    assert collected["8clq"]["pdb_chain_info"]["ligand_ccd_codes"] == ["ZGR"]

    af3_dir = tmp_path / "af3_inputs"
    af3_dir.mkdir()
    make_af3_json(
        af3_ss_input_dir=af3_dir,
        sample_dict=collected,
        json_config={"model_seeds": [42], "version": 2},
    )
    payload = json.loads((af3_dir / "8clq_sample0.json").read_text())
    ligand_entries = [entry["ligand"] for entry in payload["sequences"] if "ligand" in entry]

    assert ligand_entries == [{"id": "E", "ccdCodes": ["ZGR"]}]


def test_stage2_inputs_preserve_stage1_pdb_chain_info(monkeypatch) -> None:
    def fake_create_pos_constraint_dict_from_pocket(**kwargs):
        return {
            "pdb_key": kwargs["pdb_key"],
            "fixed_pos_seq": "",
            "fixed_pos_scn": "",
        }, None

    monkeypatch.setattr(
        two_stage,
        "create_pos_constraint_dict_from_pocket",
        fake_create_pos_constraint_dict_from_pocket,
    )
    pdb_chain_info = {
        "protein_pn_unit_iids": ["A_1"],
        "ligand_pn_unit_iids": ["E_1"],
        "ligand_ccd_codes": ["ZGR"],
    }
    stage2_sample_dict, _, _ = two_stage._build_stage2_inputs_and_constraints(
        stage1_sample_dict_per_ckpt={
            "8clq": {
                "input_sample_path": "/tmp/8clq.cif",
                "designed_sample_id": ["8clq_twostage_ps_psample0"],
                "designed_sample_path": ["/tmp/8clq_twostage_ps_psample0.cif"],
                "designed_sample_atom_array": [object()],
                "designed_sample_seq": ["A"],
                "pdb_chain_info": pdb_chain_info,
            }
        },
        stage1_ckpt_info={"global_step": 1, "epoch": 1, "ckpt_path": "/tmp/ckpt.pt"},
        stage1_log_dir_per_ckpt=Path("/tmp/stage1"),
        stage1_region="pocket",
        stage2_region="scaffold",
        stage2_constraint_type="pocket",
        stage1_model_label="stage1",
        pocket_distance=8.0,
        pocket_annotation_method="calpha",
        use_calpha_for_pocket_annotation=True,
    )

    assert stage2_sample_dict["8clq_twostage_ps_psample0"]["pdb_chain_info"] == pdb_chain_info
    assert stage2_sample_dict["8clq_twostage_ps_psample0"]["pdb_chain_info"] is not pdb_chain_info


def test_make_af3_json_uses_closest_canonical_sequence_for_protein_ptm(tmp_path: Path) -> None:
    sample_dict = {
        "sample": {
            "designed_sample_id": ["sample0"],
            "designed_sample_atom_array": [_make_modified_protein_atom_array()],
            "pdb_chain_info": {
                "protein_pn_unit_iids": ["A_1"],
                "ligand_pn_unit_iids": [],
                "ligand_ccd_codes": [],
            },
        }
    }

    make_af3_json(
        af3_ss_input_dir=tmp_path,
        sample_dict=sample_dict,
        json_config={"model_seeds": [42], "version": 2},
    )

    payload = json.loads((tmp_path / "sample0.json").read_text())
    protein_entry = payload["sequences"][0]["protein"]

    assert protein_entry["sequence"] == "AKG"
    assert protein_entry["modifications"] == [
        {"ptmType": "MLY", "ptmPosition": 2}
    ]
    assert "bondedAtomPairs" not in payload

    folding_input = pytest.importorskip(
        "alphafold3.common.folding_input",
        reason="local AlphaFold 3 parser is not importable",
    )
    protein_chain = folding_input.ProteinChain.from_dict({"protein": protein_entry})

    assert protein_chain.to_dict()["protein"]["sequence"] == "AKG"
    assert protein_chain.to_ccd_sequence() == ["ALA", "MLY", "GLY"]


def test_insert_gap_residues_defaults_to_unk_without_native_lookup() -> None:
    atom_array = _make_gap_protein_atom_array({1: "ALA", 4: "TYR"})

    with_gaps = insert_unk_residues_for_gaps_in_atom_array(atom_array)

    assert _res_names_by_id(with_gaps) == {
        1: "ALA",
        2: "UNK",
        3: "UNK",
        4: "TYR",
    }


def test_insert_gap_residues_uses_native_lookup_when_available() -> None:
    atom_array = _make_gap_protein_atom_array({1: "ALA", 4: "TYR"})
    native_lookup = {
        ("A", 1): "ALA",
        ("A", 2): "GLY",
        ("A", 3): "SER",
        ("A", 4): "VAL",
    }

    with_gaps = insert_unk_residues_for_gaps_in_atom_array(
        atom_array,
        missing_res_name_by_chain_res_id=native_lookup,
    )

    assert _res_names_by_id(with_gaps) == {
        1: "ALA",
        2: "GLY",
        3: "SER",
        4: "TYR",
    }


def test_make_af3_json_fills_missing_gap_sequence_from_native_lookup(tmp_path: Path) -> None:
    designed_atom_array = _make_gap_protein_atom_array({1: "ALA", 4: "TYR"})
    native_atom_array = _make_gap_protein_atom_array({1: "ALA", 2: "GLY", 3: "SER", 4: "VAL"})
    ss_dir = tmp_path / "ss"
    tc_dir = tmp_path / "tc"
    ss_dir.mkdir()
    tc_dir.mkdir()
    template_path = tmp_path / "template.cif"
    template_path.write_text("")
    sample_dict = {
        "sample": {
            "designed_sample_id": ["sample0"],
            "designed_sample_atom_array": [designed_atom_array],
            "designed_sample_path_for_af3_tc": [str(template_path)],
            "native_res_name_by_chain_res_id": get_res_name_by_chain_res_id(native_atom_array),
            "pdb_chain_info": {
                "protein_pn_unit_iids": ["A_1"],
                "ligand_pn_unit_iids": [],
                "ligand_ccd_codes": [],
            },
        }
    }

    make_af3_json(
        af3_ss_input_dir=ss_dir,
        af3_tc_input_dir=tc_dir,
        sample_dict=sample_dict,
        json_config={"model_seeds": [42], "version": 2},
        make_tc_input=True,
    )

    ss_payload = json.loads((ss_dir / "sample0.json").read_text())
    protein_entry = ss_payload["sequences"][0]["protein"]
    assert protein_entry["sequence"] == "AGSY"

    tc_payload = json.loads((tc_dir / "sample0.json").read_text())
    tc_protein_entry = tc_payload["sequences"][0]["protein"]
    assert tc_protein_entry["sequence"] == "AGSY"
    assert tc_protein_entry["templates"][0]["queryIndices"] == [0, 3]
