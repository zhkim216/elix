from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray
from omegaconf import OmegaConf

from allatom_design.eval.structure_prediction import af3_evaluation
from allatom_design.eval.sampling.sequence_design import evaluation as sequence_evaluation


def _roles(
    *,
    context: list[str] | None = None,
    template: list[str] | None = None,
) -> dict[str, list[str]]:
    return {
        "binder_pn_unit_iids": ["A_1"],
        "context_pn_unit_iids": context or [],
        "frame_pn_unit_iids": ["A_1"],
        "template_pn_unit_iids": template or [],
    }


def _struct_pred_cfg(mode: str = "ss"):
    return OmegaConf.create(
        {
            "af3": {
                "require_complete_predictions": False,
                "runner_path": "/fake/run_alphafold.py",
                "json_config": {"model_seeds": [42], "version": 2},
                "inference_config": {
                    "base": {},
                    mode: {
                        "num_diffusion_samples": 1,
                        "strict_input_fingerprint": False,
                    },
                },
            }
        }
    )


def test_auto_mode_resolver_requires_roles_and_uses_template_presence() -> None:
    assert af3_evaluation._resolve_af3_mode({"pn_unit_roles": _roles()}) == "ss"
    assert (
        af3_evaluation._resolve_af3_mode(
            {"pn_unit_roles": _roles(template=["A_1"])}
        )
        == "tc"
    )

    with pytest.raises(ValueError, match="requires pn_unit_roles"):
        af3_evaluation._resolve_af3_mode({})
    with pytest.raises(ValueError, match="at most one"):
        af3_evaluation._resolve_af3_mode(
            {
                "pn_unit_roles": {
                    **_roles(),
                    "context_pn_unit_iids": ["B_1"],
                    "template_pn_unit_iids": ["A_1", "B_1"],
                }
            }
        )
    with pytest.raises(ValueError, match="protein PN unit"):
        af3_evaluation._resolve_af3_mode(
            {
                "pn_unit_roles": {
                    **_roles(),
                    "context_pn_unit_iids": ["L_1"],
                    "template_pn_unit_iids": ["L_1"],
                },
                "pdb_chain_info": {"protein_pn_unit_iids": ["A_1"]},
            }
        )


def test_auto_partition_validates_every_entry_before_mutation() -> None:
    sample_dict = {
        "ss_input": {"pn_unit_roles": _roles()},
        "tc_input": {"pn_unit_roles": _roles(template=["A_1"])},
    }

    partitions = af3_evaluation._partition_sample_dict_by_af3_mode(sample_dict)

    assert list(partitions["ss"]) == ["ss_input"]
    assert list(partitions["tc"]) == ["tc_input"]
    assert "af3_ss_json_paths" not in sample_dict["ss_input"]
    assert "af3_tc_json_paths" not in sample_dict["tc_input"]


def test_legacy_wrappers_forward_fixed_modes(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        af3_evaluation,
        "_evaluate_af3_mode",
        lambda **kwargs: calls.append(kwargs) or [],
    )
    common_kwargs = {
        "sample_dict": {},
        "out_dir": tmp_path,
        "struct_pred_cfg": _struct_pred_cfg(),
        "cif_parse_cfg": None,
        "preprocess_cfg": None,
        "featurizer_cfg": None,
        "pocket_cfg": None,
    }

    af3_evaluation.evaluate_af3_self_consistency(**common_kwargs)
    af3_evaluation.evaluate_af3_docking_consistency(**common_kwargs)

    assert [call["mode"] for call in calls] == ["ss", "tc"]
    assert all(call["enforce_require_complete_predictions"] for call in calls)


def test_auto_evaluator_batches_ss_then_tc_and_raises_after_both(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events = []

    def fake_evaluate(**kwargs):
        input_ids = list(kwargs["sample_dict"])
        events.append((kwargs["mode"], input_ids))
        return [
            {
                "designed_sample_id": input_id,
                "af3_status": "complete",
            }
            for input_id in input_ids
        ]

    def fake_raise(**kwargs):
        events.append(("raise", [row["designed_sample_id"] for row in kwargs["status_rows"]]))

    monkeypatch.setattr(af3_evaluation, "_evaluate_af3_mode", fake_evaluate)
    monkeypatch.setattr(
        af3_evaluation,
        "_raise_if_required_predictions_incomplete",
        fake_raise,
    )
    sample_dict = {
        "tc_input": {"pn_unit_roles": _roles(template=["A_1"])},
        "ss_input": {"pn_unit_roles": _roles()},
    }

    rows = af3_evaluation.evaluate_af3_structure_prediction(
        sample_dict=sample_dict,
        out_dir=tmp_path,
        struct_pred_cfg=_struct_pred_cfg(),
        cif_parse_cfg=None,
        preprocess_cfg=None,
        featurizer_cfg=None,
        pocket_cfg=None,
    )

    assert events == [
        ("ss", ["ss_input"]),
        ("tc", ["tc_input"]),
        ("raise", ["ss_input", "tc_input"]),
    ]
    assert [row["designed_sample_id"] for row in rows] == ["ss_input", "tc_input"]


def _prediction_atom_array(
    chain_ids: list[str],
    *,
    pn_unit_iids: list[str] | None = None,
) -> AtomArray:
    atom_array = AtomArray(len(chain_ids))
    atom_array.coord = np.arange(len(chain_ids) * 3, dtype=float).reshape(-1, 3)
    atom_array.chain_id = np.asarray(chain_ids)
    atom_array.set_annotation(
        "pn_unit_iid",
        np.asarray(pn_unit_iids or [f"{chain_id}_1" for chain_id in chain_ids]),
    )
    atom_array.atom_name = np.asarray(["CA"] * len(chain_ids))
    atom_array.res_name = np.asarray(["ALA"] * len(chain_ids))
    atom_array.res_id = np.asarray([1] * len(chain_ids))
    atom_array.set_annotation("confidence", np.arange(len(chain_ids), dtype=float))
    return atom_array


def _patch_single_prediction_lifecycle(
    monkeypatch,
    tmp_path: Path,
    *,
    parse_result=None,
    parse_error: Exception | None = None,
):
    prediction_path = tmp_path / "prediction_model.cif"

    def fake_make_af3_json(*, sample_dict, make_ss_input, make_tc_input, **kwargs):
        key = "af3_ss_json_paths" if make_ss_input else "af3_tc_json_paths"
        for subsample_dict in sample_dict.values():
            json_path = tmp_path / "design.json"
            chain_mapping = af3_evaluation.build_af3_chain_id_to_pn_unit_iid(
                protein_pn_unit_iids=subsample_dict["pdb_chain_info"]["protein_pn_unit_iids"],
                ligand_pn_unit_iids=subsample_dict["pdb_chain_info"]["ligand_pn_unit_iids"],
            )
            sequences = [
                {
                    "protein": {
                        "id": chain_id,
                        "sequence": "A",
                        "modifications": [],
                    }
                }
                for chain_id, pn_unit_iid in chain_mapping.items()
                if pn_unit_iid in subsample_dict["pdb_chain_info"]["protein_pn_unit_iids"]
            ]
            json_path.write_text(json.dumps({"sequences": sequences}))
            subsample_dict[key] = [json_path]
        return sample_dict

    monkeypatch.setattr(af3_evaluation, "make_af3_json", fake_make_af3_json)
    monkeypatch.setattr(
        af3_evaluation,
        "expected_prediction_count_from_json",
        lambda *args, **kwargs: 1,
    )
    monkeypatch.setattr(
        af3_evaluation,
        "summarize_af3_prediction_outputs",
        lambda **kwargs: {
            "model_cif_paths": [prediction_path],
            "malformed_sample_dirs": [],
            "n_found": 1,
            "n_malformed": 0,
            "n_surplus": 0,
            "input_fingerprint_ok": None,
            "input_fingerprint_error": "",
        },
    )
    from allatom_design.eval.structure_prediction import inputs

    if parse_error is not None:
        def fail_parse(**kwargs):
            raise parse_error

        monkeypatch.setattr(inputs, "prepare_af3_prediction", fail_parse)
    else:
        monkeypatch.setattr(
            inputs,
            "prepare_af3_prediction",
            lambda **kwargs: parse_result,
        )


def test_common_lifecycle_records_role_error_when_prediction_parse_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_single_prediction_lifecycle(
        monkeypatch,
        tmp_path,
        parse_error=ValueError("bad prediction cif"),
    )
    saved = {}
    monkeypatch.setattr(
        af3_evaluation,
        "_save_metrics_results",
        lambda **kwargs: saved.update(kwargs),
    )
    sample_dict = {
        "input": {
            "designed_sample_id": ["design"],
            "designed_sample_atom_array": [_prediction_atom_array(["A", "L"])],
            "pdb_chain_info": {
                "protein_pn_unit_iids": ["A_1"],
                "ligand_pn_unit_iids": [],
                "ligand_ccd_codes": [],
            },
            "pn_unit_roles": _roles(),
        }
    }

    status_rows = af3_evaluation._evaluate_af3_mode(
        mode="ss",
        sample_dict=sample_dict,
        out_dir=tmp_path,
        struct_pred_cfg=_struct_pred_cfg(),
        cif_parse_cfg=None,
        preprocess_cfg=None,
        featurizer_cfg=None,
        pocket_cfg=OmegaConf.create({"pocket_distance_for_docking_metrics": 6.0}),
        calculate_metrics_only=True,
        csv_suffix="",
        input_sample_is_designed=True,
        free_atom_arrays_progressively=False,
        enforce_require_complete_predictions=False,
    )

    status = status_rows[0]
    assert status["af3_status"] == "metric_failed"
    assert status["n_role_metric_rows"] == 1
    assert status["n_role_metric_errors"] == 1
    assert "prediction parse failed" in status["sc_errors"]
    assert saved["role_metric_rows"][0]["metric_family"] == "role_contract"
    assert saved["role_metric_rows"][0]["prediction_id"] == "diffusion_0"


def test_sc_failure_is_reported_without_legacy_docking_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pred_atom_array = _prediction_atom_array(["A", "L"])
    _patch_single_prediction_lifecycle(
        monkeypatch,
        tmp_path,
        parse_result={"atom_array": pred_atom_array},
    )
    from allatom_design.eval import metrics

    def fail_sc(**kwargs):
        raise RuntimeError("sc failed")

    monkeypatch.setattr(metrics, "compute_self_consistency_metrics_atomarray", fail_sc)
    saved = {}
    monkeypatch.setattr(
        af3_evaluation,
        "_save_metrics_results",
        lambda **kwargs: saved.update(kwargs),
    )
    sample_dict = {
        "input": {
            "designed_sample_id": ["design"],
            "designed_sample_atom_array": [_prediction_atom_array(["A", "L"])],
            "pdb_chain_info": {
                "protein_pn_unit_iids": ["A_1"],
                "ligand_pn_unit_iids": ["L_1"],
                "ligand_ccd_codes": ["ATP"],
            },
        }
    }

    status_rows = af3_evaluation._evaluate_af3_mode(
        mode="ss",
        sample_dict=sample_dict,
        out_dir=tmp_path,
        struct_pred_cfg=_struct_pred_cfg(),
        cif_parse_cfg=None,
        preprocess_cfg=None,
        featurizer_cfg=None,
        pocket_cfg=OmegaConf.create({"pocket_distance_for_docking_metrics": 6.0}),
        calculate_metrics_only=True,
        csv_suffix="",
        input_sample_is_designed=True,
        free_atom_arrays_progressively=False,
        enforce_require_complete_predictions=False,
    )

    status = status_rows[0]
    assert status["n_sc_success"] == 0
    assert status["af3_status"] == "metric_failed"
    assert "n_docking_success" not in status
    assert "designed_sample_id_to_per_pred_docking_metrics" not in saved
    assert saved["role_metric_rows"] is None


def test_role_metric_error_result_is_not_counted_as_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pred_atom_array = _prediction_atom_array(["A", "L"])
    _patch_single_prediction_lifecycle(
        monkeypatch,
        tmp_path,
        parse_result={"atom_array": pred_atom_array},
    )
    from allatom_design.eval import metrics
    from allatom_design.eval.metrics import role_aware

    monkeypatch.setattr(
        metrics,
        "compute_self_consistency_metrics_atomarray",
        lambda **kwargs: {"sc_ca_rmsd": 1.0},
    )
    monkeypatch.setattr(
        role_aware,
        "compute_role_aware_metrics_atomarray",
        lambda **kwargs: [
            {
                "status": "error",
                "error": "No common pocket CA residues",
                "metric_family": "frame_placement",
            }
        ],
    )
    saved = {}
    monkeypatch.setattr(
        af3_evaluation,
        "_save_metrics_results",
        lambda **kwargs: saved.update(kwargs),
    )
    sample_dict = {
        "input": {
            "designed_sample_id": ["design"],
            "designed_sample_atom_array": [_prediction_atom_array(["A", "L"])],
            "pdb_chain_info": {
                "protein_pn_unit_iids": ["A_1"],
                "ligand_pn_unit_iids": ["L_1"],
                "ligand_ccd_codes": ["ATP"],
            },
            "pn_unit_roles": _roles(context=["L_1"]),
        }
    }

    status_rows = af3_evaluation._evaluate_af3_mode(
        mode="ss",
        sample_dict=sample_dict,
        out_dir=tmp_path,
        struct_pred_cfg=_struct_pred_cfg(),
        cif_parse_cfg=None,
        preprocess_cfg=None,
        featurizer_cfg=None,
        pocket_cfg=OmegaConf.create({"pocket_distance_for_docking_metrics": 6.0}),
        calculate_metrics_only=True,
        csv_suffix="",
        input_sample_is_designed=True,
        free_atom_arrays_progressively=False,
        enforce_require_complete_predictions=False,
    )

    status = status_rows[0]
    assert status["n_sc_success"] == 1
    assert status["n_role_metric_predictions_success"] == 0
    assert status["n_role_metric_errors"] == 1
    assert status["af3_status"] == "metric_failed"
    assert "No common pocket CA residues" in status["role_metric_errors"]
    assert saved["role_metric_rows"][0]["error"] == "No common pocket CA residues"


def test_prediction_identity_restore_is_in_place_and_preserves_other_fields() -> None:
    pred_atom_array = _prediction_atom_array(
        ["B", "B", "F"],
        pn_unit_iids=["B_1", "B_1", "F_1"],
    )
    original_coord = pred_atom_array.coord.copy()
    original_confidence = pred_atom_array.confidence.copy()
    original_chain_id = pred_atom_array.chain_id.copy()

    restored = af3_evaluation._restore_af3_prediction_pn_unit_iids(
        pred_atom_array,
        af3_chain_id_to_pn_unit_iid={"B": "B_2", "F": "F_2"},
    )

    assert restored is pred_atom_array
    assert restored.pn_unit_iid.tolist() == ["B_2", "B_2", "F_2"]
    np.testing.assert_array_equal(restored.chain_id, original_chain_id)
    np.testing.assert_array_equal(restored.coord, original_coord)
    np.testing.assert_array_equal(restored.confidence, original_confidence)


def test_prediction_identity_restore_rejects_missing_chain() -> None:
    pred_atom_array = _prediction_atom_array(["B"])

    with pytest.raises(ValueError, match=r"missing=\['F'\]"):
        af3_evaluation._restore_af3_prediction_pn_unit_iids(
            pred_atom_array,
            af3_chain_id_to_pn_unit_iid={"B": "B_2", "F": "F_2"},
        )


def test_prediction_identity_restore_rejects_unexpected_chain() -> None:
    pred_atom_array = _prediction_atom_array(["B", "X"])

    with pytest.raises(ValueError, match=r"unexpected=\['X'\]"):
        af3_evaluation._restore_af3_prediction_pn_unit_iids(
            pred_atom_array,
            af3_chain_id_to_pn_unit_iid={"B": "B_2"},
        )


def test_prediction_res_id_restore_round_trips_source_label_ids(tmp_path: Path) -> None:
    designed = AtomArray(4)
    designed.coord = np.zeros((4, 3))
    designed.chain_id = np.asarray(["A"] * 4)
    designed.atom_name = np.asarray(["N", "CA", "N", "CA"])
    designed.res_name = np.asarray(["MET", "MET", "LYS", "LYS"])
    designed.res_id = np.asarray([7, 7, 8, 8])
    designed.set_annotation("pn_unit_iid", np.asarray(["A_1"] * 4))

    predicted = designed.copy()
    predicted.res_id = np.asarray([1, 1, 2, 2])
    json_path = tmp_path / "input.json"
    json_path.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "MK",
                            "modifications": [],
                        }
                    }
                ]
            }
        )
    )

    restored = af3_evaluation._restore_af3_prediction_protein_res_ids(
        predicted,
        designed_sample_atom_array=designed,
        protein_pn_unit_iids=["A_1"],
        af3_chain_id_to_pn_unit_iid={"A": "A_1"},
        json_path=json_path,
    )

    assert restored is predicted
    assert restored.res_id.tolist() == [7, 7, 8, 8]


def test_prediction_res_id_restore_rejects_serialized_sequence_mismatch(
    tmp_path: Path,
) -> None:
    designed = _prediction_atom_array(["A"], pn_unit_iids=["A_1"])
    predicted = designed.copy()
    predicted.res_name[0] = "GLY"
    json_path = tmp_path / "input.json"
    json_path.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "A",
                            "modifications": [],
                        }
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="exact serialized JSON sequence"):
        af3_evaluation._restore_af3_prediction_protein_res_ids(
            predicted,
            designed_sample_atom_array=designed,
            protein_pn_unit_iids=["A_1"],
            af3_chain_id_to_pn_unit_iid={"A": "A_1"},
            json_path=json_path,
        )


@pytest.mark.parametrize(
    ("modifications", "predicted_res_name"),
    [
        ([], "MET"),
        ([{"ptmType": "MSE", "ptmPosition": 1}], "MSE"),
    ],
)
def test_prediction_res_id_restore_uses_exact_json_modified_residue_contract(
    tmp_path: Path,
    modifications: list[dict[str, object]],
    predicted_res_name: str,
) -> None:
    designed = _prediction_atom_array(["A"], pn_unit_iids=["A_1"])
    designed.res_id[0] = 7
    designed.res_name[0] = "MSE"
    predicted = designed.copy()
    predicted.res_id[0] = 1
    predicted.res_name[0] = predicted_res_name
    json_path = tmp_path / "input.json"
    json_path.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "M",
                            "modifications": modifications,
                        }
                    }
                ]
            }
        )
    )

    restored = af3_evaluation._restore_af3_prediction_protein_res_ids(
        predicted,
        designed_sample_atom_array=designed,
        protein_pn_unit_iids=["A_1"],
        af3_chain_id_to_pn_unit_iid={"A": "A_1"},
        json_path=json_path,
    )

    assert restored.res_id.tolist() == [7]
    assert restored.res_name.tolist() == [predicted_res_name]


def test_prediction_res_id_restore_rejects_lost_json_modification(
    tmp_path: Path,
) -> None:
    designed = _prediction_atom_array(["A"], pn_unit_iids=["A_1"])
    designed.res_name[0] = "MSE"
    predicted = designed.copy()
    predicted.res_name[0] = "MET"
    json_path = tmp_path / "input.json"
    json_path.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "M",
                            "modifications": [
                                {"ptmType": "MSE", "ptmPosition": 1}
                            ],
                        }
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="MSE!=MET"):
        af3_evaluation._restore_af3_prediction_protein_res_ids(
            predicted,
            designed_sample_atom_array=designed,
            protein_pn_unit_iids=["A_1"],
            af3_chain_id_to_pn_unit_iid={"A": "A_1"},
            json_path=json_path,
        )


@pytest.mark.parametrize("mode", ["ss", "tc"])
def test_common_lifecycle_restores_source_iids_for_active_metric_consumers(
    monkeypatch,
    tmp_path: Path,
    mode: str,
) -> None:
    pred_atom_array = _prediction_atom_array(
        ["B", "F"],
        pn_unit_iids=["B_1", "F_1"],
    )
    _patch_single_prediction_lifecycle(
        monkeypatch,
        tmp_path,
        parse_result={"atom_array": pred_atom_array},
    )
    from allatom_design.eval import metrics
    from allatom_design.eval.metrics import role_aware

    metric_inputs: dict[str, tuple[list[str], list[str]]] = {}

    def record_metric_input(metric_name: str, pred_atom_array: AtomArray) -> None:
        metric_inputs[metric_name] = (
            pred_atom_array.chain_id.tolist(),
            pred_atom_array.pn_unit_iid.tolist(),
        )

    def fake_role_metrics(**kwargs):
        record_metric_input("role", kwargs["pred_atom_array"])
        return [{"status": "ok", "error": "", "metric_family": "test"}]

    def fake_sc_metrics(**kwargs):
        record_metric_input("sc", kwargs["pred_atom_array"])
        return {"sc_ca_rmsd": 1.0}

    monkeypatch.setattr(
        role_aware,
        "compute_role_aware_metrics_atomarray",
        fake_role_metrics,
    )
    monkeypatch.setattr(
        metrics,
        "compute_self_consistency_metrics_atomarray",
        fake_sc_metrics,
    )
    monkeypatch.setattr(af3_evaluation, "_save_metrics_results", lambda **kwargs: None)
    sample_dict = {
        "input": {
            "designed_sample_id": ["design"],
            "designed_sample_atom_array": [
                _prediction_atom_array(
                    ["B", "F"],
                    pn_unit_iids=["B_2", "F_2"],
                )
            ],
            "pdb_chain_info": {
                "protein_pn_unit_iids": ["B_2"],
                "ligand_pn_unit_iids": ["F_2"],
                "ligand_ccd_codes": ["ATP"],
            },
            "pn_unit_roles": {
                "binder_pn_unit_iids": ["B_2"],
                "context_pn_unit_iids": ["F_2"],
                "frame_pn_unit_iids": ["B_2"],
                "template_pn_unit_iids": ["B_2"] if mode == "tc" else [],
            },
        }
    }

    status_rows = af3_evaluation._evaluate_af3_mode(
        mode=mode,
        sample_dict=sample_dict,
        out_dir=tmp_path,
        struct_pred_cfg=_struct_pred_cfg(mode),
        cif_parse_cfg=None,
        preprocess_cfg=None,
        featurizer_cfg=None,
        pocket_cfg=OmegaConf.create({"pocket_distance_for_docking_metrics": 6.0}),
        calculate_metrics_only=True,
        csv_suffix="",
        input_sample_is_designed=True,
        free_atom_arrays_progressively=False,
        enforce_require_complete_predictions=False,
    )

    assert status_rows[0]["af3_status"] == "complete"
    assert metric_inputs == {
        "role": (["B", "F"], ["B_2", "F_2"]),
        "sc": (["B", "F"], ["B_2", "F_2"]),
    }


def _checkpoint_cfg(*, auto: bool, legacy_ss: bool, legacy_tc: bool):
    return OmegaConf.create(
        {
            "struct_pred_cfg": {
                "evaluate_structure_prediction": auto,
                "evaluate_self_consistency": legacy_ss,
                "evaluate_docking_consistency": legacy_tc,
                "calculate_metrics_only": False,
            },
            "cif_cfg": {"parse": {"af3_predictions": {}}},
            "preprocess_cfg": {"af3_predictions": {}},
            "featurizer_cfg": {"prepare_af3_predictions": {}},
            "pocket_cfg": {},
            "wandb": {"no_wandb": True},
        }
    )


def test_checkpoint_dispatch_rejects_auto_with_legacy_flags(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        sequence_evaluation.evaluate_af3_for_checkpoint(
            cfg=_checkpoint_cfg(auto=True, legacy_ss=True, legacy_tc=False),
            sample_dict_per_ckpt={},
            log_dir_per_ckpt=tmp_path,
            ckpt_info={"global_step": 1, "epoch": 2},
            csv_suffix="",
            free_atom_arrays_after_self_consistency=True,
            input_sample_is_designed=True,
        )


def test_checkpoint_dispatch_uses_unified_auto_evaluator(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        sequence_evaluation,
        "evaluate_af3_structure_prediction",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        sequence_evaluation,
        "evaluate_af3_self_consistency",
        lambda **kwargs: pytest.fail("legacy SS evaluator should not run"),
    )
    monkeypatch.setattr(
        sequence_evaluation,
        "evaluate_af3_docking_consistency",
        lambda **kwargs: pytest.fail("legacy TC evaluator should not run"),
    )
    sample_dict = {"input": {"pn_unit_roles": _roles()}}

    sequence_evaluation.evaluate_af3_for_checkpoint(
        cfg=_checkpoint_cfg(auto=True, legacy_ss=False, legacy_tc=False),
        sample_dict_per_ckpt=sample_dict,
        log_dir_per_ckpt=tmp_path,
        ckpt_info={"global_step": 1, "epoch": 2},
        csv_suffix="_array_0",
        free_atom_arrays_after_self_consistency=False,
        input_sample_is_designed=False,
    )

    assert len(calls) == 1
    assert calls[0]["sample_dict"] is sample_dict
    assert calls[0]["free_atom_arrays_progressively"] is True
    assert calls[0]["input_sample_is_designed"] is False
