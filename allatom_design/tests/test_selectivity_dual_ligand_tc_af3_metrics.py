import pandas as pd

from allatom_design.eval.structure_prediction import run_selectivity_dual_ligand_tc_af3 as runner


def _selectivity_sampling_row() -> pd.Series:
    return pd.Series(
        {
            "selectivity_pair_id": "selpair_0000",
            "scaffold_side": 1,
            "native_ligand_side": 1,
            "transformed_ligand_side": 2,
            "ligand_1_pn_unit_iid": "X_1",
            "ligand_2_pn_unit_iid": "Y_1",
            "ligand_1_original_pn_unit_iid": "C_1",
            "ligand_2_original_pn_unit_iid": "D_1",
            "ligand_1_ccd_code": "L1A",
            "ligand_2_ccd_code": "L2B",
            "ligand_1_smiles": "CCO",
        }
    )


def test_compute_per_ligand_docking_metric_rows_dispatches_ligands_separately(monkeypatch) -> None:
    calls = []

    def fake_compute_docking_metrics_atomarray(**kwargs):
        calls.append(kwargs)
        ligand_iid = kwargs["ligand_pn_unit_iids"][0]
        return {
            "ligand_rmsd": 1.0 if ligand_iid == "X_1" else 2.0,
            "binding_site_rmsd": 0.5,
            "ligand_plddt": 80.0,
            "binding_site_plddt": 70.0,
            "iptm": 0.8,
            "interface_min_pae": 3.0,
        }

    monkeypatch.setattr(
        runner,
        "compute_docking_metrics_atomarray",
        fake_compute_docking_metrics_atomarray,
    )

    rows = runner.compute_per_ligand_docking_metric_rows(
        pred_atom_array=object(),
        sample_atom_array=object(),
        pred_sample_path="/tmp/pred.cif",
        input_sample_id="input0",
        designed_sample_id="design0",
        sampling_row=_selectivity_sampling_row(),
        diffusion_idx=0,
        receptor_pn_unit_iids=["A_1"],
        ligand_pn_unit_iids=["X_1", "Y_1"],
        ligand_ccd_codes=["L1A", "L2B"],
        pocket_distance_for_docking_metrics=8.0,
        input_sample_is_designed=False,
        guidance_mode="selectivity",
        guidance_scale=1.0,
        guidance_target_ligand_side=2,
    )

    assert [call["ligand_pn_unit_iids"] for call in calls] == [["X_1"], ["Y_1"]]
    assert [call["ligand_ccd_codes"] for call in calls] == [["L1A"], ["L2B"]]
    assert [call["reference_ligand_pn_unit_iids"] for call in calls] == [["C_1"], ["D_1"]]
    assert [call["ligand_smiles"] for call in calls] == [["CCO"], None]
    assert [row["ligand_side"] for row in rows] == [1, 2]
    assert [row["guidance_mode"] for row in rows] == ["selectivity", "selectivity"]
    assert [row["guidance_scale"] for row in rows] == [1.0, 1.0]
    assert [row["guidance_target_ligand_side"] for row in rows] == [2, 2]
    assert [row["positive_ligand_side"] for row in rows] == [2, 2]
    assert [row["negative_ligand_side"] for row in rows] == [1, 1]
    assert [row["positive_ligand_pn_unit_iid"] for row in rows] == ["Y_1", "Y_1"]
    assert [row["negative_ligand_pn_unit_iid"] for row in rows] == ["X_1", "X_1"]
    assert [row["positive_branch_label"] for row in rows] == ["ligand_2", "ligand_2"]
    assert [row["negative_branch_label"] for row in rows] == ["ligand_1", "ligand_1"]
    assert [row["positive_ligand_role"] for row in rows] == ["transformed", "transformed"]
    assert [row["negative_ligand_role"] for row in rows] == ["native", "native"]
    assert [row["ligand_rmsd"] for row in rows] == [1.0, 2.0]
    assert all(row["metric_status"] == "ok" for row in rows)


def test_designed_selectivity_tc_metrics_use_current_ligand_iids_as_reference(monkeypatch) -> None:
    calls = []

    def fake_compute_docking_metrics_atomarray(**kwargs):
        calls.append(kwargs)
        return {
            "ligand_rmsd": 1.0,
            "binding_site_rmsd": 0.5,
            "ligand_plddt": 80.0,
            "binding_site_plddt": 70.0,
            "iptm": 0.8,
            "interface_min_pae": 3.0,
        }

    monkeypatch.setattr(
        runner,
        "compute_docking_metrics_atomarray",
        fake_compute_docking_metrics_atomarray,
    )

    rows = runner.compute_per_ligand_docking_metric_rows(
        pred_atom_array=object(),
        sample_atom_array=object(),
        pred_sample_path="/tmp/pred.cif",
        input_sample_id="input0",
        designed_sample_id="design0",
        sampling_row=_selectivity_sampling_row(),
        diffusion_idx=0,
        receptor_pn_unit_iids=["A_1"],
        ligand_pn_unit_iids=["X_1", "Y_1"],
        ligand_ccd_codes=["L1A", "L2B"],
        pocket_distance_for_docking_metrics=8.0,
        input_sample_is_designed=True,
        guidance_mode="selectivity",
        guidance_scale=1.0,
        guidance_target_ligand_side=2,
    )

    assert [call["reference_ligand_pn_unit_iids"] for call in calls] == [["X_1"], ["Y_1"]]
    assert [row["original_ligand_pn_unit_iid"] for row in rows] == ["C_1", "D_1"]
    assert all(row["metric_status"] == "ok" for row in rows)
