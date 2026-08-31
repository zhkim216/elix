from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from biotite.structure import AtomArray

import atomworks.enums as aw_enums
from allatom_design.eval.metrics import role_aware
from allatom_design.eval.metrics.role_aware import (
    build_role_metric_plan,
    compute_role_aware_metrics_atomarray,
)
from allatom_design.eval.sampling.sequence_design.outputs import (
    _attach_designed_sample_chain_info,
    _filter_query_pn_unit_iids,
)
from allatom_design.eval.utils.sampling_inputs import (
    ROLE_SAMPLING_COLUMNS,
    derive_role_sample_id,
    normalize_pn_unit_roles,
    normalize_role_sampling_inputs_df,
)


def _roles(
    *,
    binder: list[str] | None = None,
    context: list[str] | None = None,
    frame: list[str] | None = None,
    template: list[str] | None = None,
) -> dict[str, list[str]]:
    return {
        "binder_pn_unit_iids": binder or ["A_1"],
        "context_pn_unit_iids": context or [],
        "frame_pn_unit_iids": frame or ["A_1"],
        "template_pn_unit_iids": template or [],
    }


def _synthetic_system(*, include_small_molecule: bool = False) -> tuple[AtomArray, AtomArray]:
    pn_unit_iids = ["C_1"] * 3 + ["A_1"] * 3 + ["E_1"]
    chain_ids = ["C"] * 3 + ["A"] * 3 + ["E"]
    res_ids = [1, 2, 3, 1, 2, 3, 1]
    res_names = ["ALA", "GLY", "SER", "ALA", "GLY", "SER", "FE"]
    atom_names = ["CA"] * 6 + ["FE"]
    elements = ["C"] * 6 + ["FE"]
    chain_types = [aw_enums.ChainType.POLYPEPTIDE_L] * 6 + [
        aw_enums.ChainType.NON_POLYMER
    ]
    coordinates = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 2.0, 0.0],
        [7.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    if include_small_molecule:
        pn_unit_iids += ["F_1", "F_1"]
        chain_ids += ["F", "F"]
        res_ids += [1, 1]
        res_names += ["LIG", "LIG"]
        atom_names += ["C1", "C2"]
        elements += ["C", "C"]
        chain_types += [aw_enums.ChainType.NON_POLYMER] * 2
        coordinates += [[2.0, 1.0, 1.0], [3.0, 1.0, 1.0]]

    sample = AtomArray(len(coordinates))
    sample.coord = np.asarray(coordinates, dtype=float)
    sample.chain_id = np.asarray(chain_ids)
    sample.res_id = np.asarray(res_ids)
    sample.res_name = np.asarray(res_names)
    sample.atom_name = np.asarray(atom_names)
    sample.element = np.asarray(elements)
    sample.hetero = np.asarray([False] * 6 + [True] * (len(coordinates) - 6))
    sample.set_annotation("pn_unit_iid", np.asarray(pn_unit_iids))
    sample.set_annotation("chain_type", np.asarray(chain_types, dtype=object))
    sample.set_annotation(
        "is_polymer",
        np.asarray([chain_type == aw_enums.ChainType.POLYPEPTIDE_L for chain_type in chain_types]),
    )
    sample.set_annotation("is_covalent_modification", np.zeros(len(sample), dtype=bool))
    sample.set_annotation("atomize", np.zeros(len(sample), dtype=bool))
    sample.set_annotation("occupancy", np.ones(len(sample), dtype=float))

    pred = sample.copy()
    pred_local = sample.coord.copy()
    pred_local[np.asarray(pn_unit_iids) == "A_1", 2] += 3.0
    pred_local[np.asarray(pn_unit_iids) == "E_1", 2] += 4.0
    if include_small_molecule:
        pred_local[np.asarray(pn_unit_iids) == "F_1", 2] += 5.0
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pred.coord = pred_local @ rotation.T + np.asarray([10.0, -4.0, 7.0])
    return sample, pred


def test_role_algebra_preserves_order_and_rejects_invalid_overlap_or_subset() -> None:
    normalized = normalize_pn_unit_roles(
        _roles(
            context=["B_1", "C_1"],
            frame=["C_1"],
            template=["C_1"],
        )
    )
    assert normalized == {
        "binder_pn_unit_iids": ["A_1"],
        "context_pn_unit_iids": ["B_1", "C_1"],
        "frame_pn_unit_iids": ["C_1"],
        "template_pn_unit_iids": ["C_1"],
    }

    with pytest.raises(ValueError, match="overlap"):
        normalize_pn_unit_roles(_roles(context=["A_1"]))
    with pytest.raises(ValueError, match="frame_pn_unit_iids must be a subset"):
        normalize_pn_unit_roles(_roles(frame=["Z_1"]))
    with pytest.raises(ValueError, match="template_pn_unit_iids must be a subset"):
        normalize_pn_unit_roles(_roles(template=["Z_1"]))
    with pytest.raises(ValueError, match="at most one"):
        normalize_pn_unit_roles(_roles(context=["B_1"], template=["A_1", "B_1"]))


def test_role_plan_keeps_intrinsic_binder_and_separates_every_nonframe_target() -> None:
    operations = build_role_metric_plan(
        _roles(
            context=["E_1", "F_1", "C_1"],
            frame=["C_1"],
            template=["C_1"],
        )
    )
    intrinsic_targets = {
        operation.target_pn_unit_iid
        for operation in operations
        if operation.metric_family == "intrinsic_sc"
    }
    placement_targets = {
        operation.target_pn_unit_iid
        for operation in operations
        if operation.metric_family == "frame_placement"
        and operation.operation_scope == "target"
    }
    assert intrinsic_targets == {"A_1"}
    assert placement_targets == {"A_1", "E_1", "F_1"}
    assert "C_1" not in placement_targets
    assert any(operation.metric_family == "frame_fit" for operation in operations)


def test_binder_that_is_also_frame_still_gets_intrinsic_sc() -> None:
    operations = build_role_metric_plan(
        _roles(context=["E_1"], frame=["A_1"], template=["A_1"])
    )
    assert any(
        operation.metric_family == "intrinsic_sc"
        and operation.target_pn_unit_iid == "A_1"
        for operation in operations
    )
    assert {
        operation.target_pn_unit_iid
        for operation in operations
        if operation.metric_family == "frame_placement"
    } == {"E_1"}


def test_frame_geometry_drives_protein_and_metal_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    sample, pred = _synthetic_system()
    monkeypatch.setattr(role_aware, "_mean_plddt", lambda **_: 80.0)
    rows = compute_role_aware_metrics_atomarray(
        sample_atom_array=sample,
        pred_atom_array=pred,
        pn_unit_roles=_roles(
            context=["E_1", "C_1"],
            frame=["C_1"],
            template=["C_1"],
        ),
    )
    by_id = {row["operation_id"]: row for row in rows}

    assert by_id["intrinsic_sc:A_1"]["ca_rmsd"] == pytest.approx(0.0, abs=1e-5)
    assert by_id["frame_fit:C_1"]["ca_rmsd"] == pytest.approx(0.0, abs=1e-5)
    assert by_id["frame_placement:C_1->A_1"]["ca_rmsd"] == pytest.approx(3.0, abs=1e-5)
    assert by_id["frame_placement:C_1->E_1"]["ligand_rmsd"] == pytest.approx(4.0, abs=1e-5)
    assert by_id["frame_placement:C_1->E_1"]["rmsd_mode"] == "metal"
    assert by_id["frame_placement:C_1->E_1"]["alignment_mode"] == "template_full_ca"
    assert by_id["frame_placement:C_1->E_1"]["frame_matched_ca_count"] == 3
    assert {row["status"] for row in rows} == {"ok"}


def test_metal_and_small_molecule_targets_are_measured_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, pred = _synthetic_system(include_small_molecule=True)
    monkeypatch.setattr(role_aware, "_mean_plddt", lambda **_: 80.0)
    rows = compute_role_aware_metrics_atomarray(
        sample_atom_array=sample,
        pred_atom_array=pred,
        pn_unit_roles=_roles(
            context=["E_1", "F_1", "C_1"],
            frame=["C_1"],
        ),
    )
    placement_rows = {
        row["target_pn_unit_iid"]: row
        for row in rows
        if row["metric_family"] == "frame_placement"
    }

    assert placement_rows["E_1"]["target_kind"] == "metal"
    assert placement_rows["E_1"]["alignment_mode"] == "all_atom_pocket_ca"
    assert placement_rows["E_1"]["pocket_distance"] == 10.0
    assert placement_rows["E_1"]["frame_matched_ca_count"] == 6
    assert placement_rows["E_1"]["ligand_rmsd"] >= 0.0
    assert placement_rows["F_1"]["target_kind"] == "small_molecule"
    assert placement_rows["F_1"]["rmsd_mode"] == "ccd_exact"
    assert placement_rows["F_1"]["alignment_mode"] == "all_atom_pocket_ca"
    assert placement_rows["F_1"]["ligand_rmsd"] >= 0.0
    assert placement_rows["E_1"]["status"] == "ok"
    assert placement_rows["F_1"]["status"] == "ok"


def test_invalid_global_frame_does_not_block_independent_ligand_pocket_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample, pred = _synthetic_system()
    keep = ~((sample.pn_unit_iid == "C_1") & (sample.res_id == 3))
    sample = sample[keep]
    pred = pred[keep]
    monkeypatch.setattr(role_aware, "_mean_plddt", lambda **_: 80.0)
    rows = compute_role_aware_metrics_atomarray(
        sample_atom_array=sample,
        pred_atom_array=pred,
        pn_unit_roles=_roles(context=["E_1", "C_1"], frame=["C_1"]),
    )
    intrinsic = next(row for row in rows if row["metric_family"] == "intrinsic_sc")
    placements = [row for row in rows if row["metric_family"] == "frame_placement"]
    assert intrinsic["status"] == "ok"
    assert placements
    placement_by_target = {row["target_pn_unit_iid"]: row for row in placements}
    assert placement_by_target["A_1"]["status"] == "error"
    assert "Frame alignment failed" in placement_by_target["A_1"]["error"]
    assert placement_by_target["E_1"]["status"] == "ok"
    assert placement_by_target["E_1"]["alignment_mode"] == "all_atom_pocket_ca"
    assert any("at least 3" in row["error"] for row in rows if row["metric_family"] == "frame_fit")


def test_designed_reference_uses_calpha_12a_pocket(monkeypatch: pytest.MonkeyPatch) -> None:
    reference, pred = _synthetic_system()
    monkeypatch.setattr(role_aware, "_mean_plddt", lambda **_: 80.0)
    rows = compute_role_aware_metrics_atomarray(
        reference_atom_array=reference,
        sample_atom_array=reference,
        pred_atom_array=pred,
        reference_is_designed=True,
        pn_unit_roles=_roles(context=["E_1"], frame=["A_1"]),
    )
    placement = next(row for row in rows if row["target_pn_unit_iid"] == "E_1")
    assert placement["status"] == "ok"
    assert placement["alignment_mode"] == "calpha_pocket_ca"
    assert placement["pocket_annotation_method"] == "calpha"
    assert placement["pocket_distance"] == 12.0


def test_reference_may_differ_and_modified_design_matches_canonical_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    designed, pred = _synthetic_system()
    reference = designed.copy()
    reference.res_name[reference.pn_unit_iid == "C_1"] = "VAL"
    modified_mask = (designed.pn_unit_iid == "C_1") & (designed.res_id == 2)
    designed.res_name[modified_mask] = "MSE"
    pred.res_name[modified_mask] = "MET"
    monkeypatch.setattr(role_aware, "_mean_plddt", lambda **_: 80.0)

    rows = compute_role_aware_metrics_atomarray(
        reference_atom_array=reference,
        sample_atom_array=designed,
        pred_atom_array=pred,
        pn_unit_roles=_roles(context=["E_1", "C_1"], frame=["C_1"], template=["C_1"]),
    )
    placement = next(row for row in rows if row["target_pn_unit_iid"] == "E_1")
    assert placement["status"] == "ok"

    pred.res_name[modified_mask] = "THR"
    rows = compute_role_aware_metrics_atomarray(
        reference_atom_array=reference,
        sample_atom_array=designed,
        pred_atom_array=pred,
        pn_unit_roles=_roles(context=["E_1", "C_1"], frame=["C_1"], template=["C_1"]),
    )
    placement = next(row for row in rows if row["target_pn_unit_iid"] == "E_1")
    assert placement["status"] == "error"
    assert "Designed and predicted" in placement["error"]


def test_ligand_alignment_saves_full_prediction_with_target_specific_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sample, pred = _synthetic_system()
    saved: list[tuple[AtomArray, object]] = []
    monkeypatch.setattr(role_aware, "_mean_plddt", lambda **_: 80.0)
    monkeypatch.setattr(
        role_aware,
        "save_cif_file",
        lambda atom_array, path: saved.append((atom_array, path)),
    )
    pred_path = tmp_path / "prediction_model.cif"
    rows = compute_role_aware_metrics_atomarray(
        reference_atom_array=sample,
        sample_atom_array=sample,
        pred_atom_array=pred,
        pred_sample_path=pred_path,
        pn_unit_roles=_roles(context=["E_1", "C_1"], frame=["C_1"], template=["C_1"]),
    )
    placement = next(row for row in rows if row["target_pn_unit_iid"] == "E_1")
    assert placement["aligned_path"].endswith("prediction_model_E_1_pocket_aligned.cif")
    assert len(saved) == 1
    assert len(saved[0][0]) == len(pred)


def test_ca_correspondence_does_not_fall_back_to_relative_offsets() -> None:
    sample, pred = _synthetic_system()
    pred.res_id[pred.pn_unit_iid == "A_1"] += 100
    with pytest.raises(ValueError, match="exact residue IDs"):
        role_aware._match_ca(
            sample_atom_array=sample,
            pred_atom_array=pred,
            pn_unit_iids=["A_1"],
        )


def test_role_snapshot_is_attached_to_sample_bundle_entry() -> None:
    sample, _ = _synthetic_system()
    roles = _roles(
        context=["E_1", "C_1"],
        frame=["C_1"],
        template=["C_1"],
    )
    sample_id = derive_role_sample_id(
        pdb_key="toy",
        binder_pn_unit_iids=roles["binder_pn_unit_iids"],
        context_pn_unit_iids=roles["context_pn_unit_iids"],
    )
    sampling_row = {
        "pdb_key": "toy",
        **roles,
        "fixed_pos_seq": "",
        "fixed_pos_scn": "",
        "fixed_pos_override_seq": "",
        "pos_restrict_aatype": "",
    }
    sampling_inputs_df = normalize_role_sampling_inputs_df(
        pd.DataFrame([sampling_row], columns=ROLE_SAMPLING_COLUMNS)
    )
    sample_dict = {
        sample_id: {
            "input_sample_id": sample_id,
            "designed_sample_atom_array": [sample],
        }
    }

    _attach_designed_sample_chain_info(
        sample_dict_per_ckpt=sample_dict,
        sampling_inputs_df=sampling_inputs_df,
    )

    assert sample_dict[sample_id]["pn_unit_roles"] == roles


def test_explicit_protein_only_query_does_not_restore_unselected_ligands() -> None:
    proteins, ligands = _filter_query_pn_unit_iids(
        protein_pn_unit_iids=["A_1", "C_1"],
        ligand_pn_unit_iids=["L_1"],
        query_pn_unit_iids=["A_1"],
    )
    assert proteins == ["A_1"]
    assert ligands == []


def test_ca_correspondence_keeps_sequence_offsets_before_filtering_nan_ca() -> None:
    sample = AtomArray(4)
    sample.coord = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    sample.chain_id = np.asarray(["A"] * 4)
    sample.res_id = np.asarray([1, 2, 3, 4])
    sample.res_name = np.asarray(["ALA"] * 4)
    sample.atom_name = np.asarray(["CA"] * 4)
    sample.element = np.asarray(["C"] * 4)
    sample.hetero = np.asarray([False] * 4)
    sample.set_annotation("pn_unit_iid", np.asarray(["A_1"] * 4))
    sample.set_annotation(
        "chain_type",
        np.asarray([aw_enums.ChainType.POLYPEPTIDE_L] * 4, dtype=object),
    )

    pred = sample.copy()
    pred.coord[0] = np.nan

    match = role_aware._match_ca(
        sample_atom_array=sample,
        pred_atom_array=pred,
        pn_unit_iids=["A_1"],
    )

    paired_res_ids = list(
        zip(
            sample.res_id[match.sample_indices].tolist(),
            pred.res_id[match.pred_indices].tolist(),
        )
    )
    assert paired_res_ids == [(2, 2), (3, 3), (4, 4)]
