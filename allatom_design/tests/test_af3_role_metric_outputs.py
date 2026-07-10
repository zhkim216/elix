from __future__ import annotations

import pandas as pd

from allatom_design.eval.metrics.role_aware import ROLE_METRIC_COLUMNS
from allatom_design.eval.structure_prediction.af3_evaluation import _save_metrics_results


def test_generic_af3_outputs_include_sc_and_role_metrics_only(tmp_path) -> None:
    sc_metrics = {
        "design0": {
            "input_sample_id": "input0",
            "diffusion_0": {"sc_ca_rmsd": 1.25, "avg_ca_plddt": 82.0},
        }
    }
    role_row = {column: None for column in ROLE_METRIC_COLUMNS}
    role_row.update(
        input_sample_id="input0",
        designed_sample_id="design0",
        prediction_id="diffusion_0",
        metric_family="frame_placement",
        operation_scope="target",
        operation_id="frame_placement:A_1->E_1",
        status="ok",
        error="",
        alignment_pn_unit_iids="A_1",
        measurement_pn_unit_iids="E_1",
        target_pn_unit_iid="E_1",
        target_kind="metal",
        rmsd_mode="metal",
        ligand_rmsd=2.5,
        ligand_plddt=70.0,
    )

    _save_metrics_results(
        out_dir=tmp_path,
        designed_sample_id_to_per_pred_sc_metrics=sc_metrics,
        role_metric_rows=[role_row],
        prediction_status_rows=None,
    )

    sc_out = pd.read_csv(tmp_path / "all_sc_metrics_per_designed_sample.csv")
    role_out = pd.read_csv(tmp_path / "all_role_metrics_per_prediction.csv")
    assert sc_out.loc[0, "diffusion_0"] == str(sc_metrics["design0"]["diffusion_0"])
    assert not (tmp_path / "all_docking_metrics_per_prediction.csv").exists()
    assert list(role_out.columns) == list(ROLE_METRIC_COLUMNS)
    assert role_out.loc[0, "operation_id"] == "frame_placement:A_1->E_1"
    assert role_out.loc[0, "target_pn_unit_iid"] == "E_1"
