from __future__ import annotations

from allatom_design.eval.structure_prediction.af3_evaluation import (
    _finalize_prediction_status,
    _new_prediction_status_row,
)


def test_malformed_prediction_directory_cannot_finish_complete() -> None:
    row = _new_prediction_status_row(
        input_sample_id="input0",
        designed_sample_id="design0",
        mode="ss",
        n_expected_predictions=1,
    )
    row.update(
        n_found_predictions=1,
        n_malformed_prediction_dirs=1,
        n_sc_success=1,
    )

    _finalize_prediction_status(row, has_role_metrics=False)

    assert row["af3_status"] == "malformed_predictions"
