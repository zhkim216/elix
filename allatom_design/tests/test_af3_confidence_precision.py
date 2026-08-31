from __future__ import annotations

import json

import numpy as np
import pytest
from biotite.structure import AtomArray

from allatom_design.eval.metrics.af3_confidence import extract_af3_confidence_metrics


def test_atom_plddt_mean_preserves_float32_precision(tmp_path) -> None:
    confidence_path = tmp_path / "prediction_confidences.json"
    confidence_path.write_text(json.dumps({"atom_plddts": [26.55, 24.427]}))

    atom_array = AtomArray(2)
    atom_array.coord = np.zeros((2, 3), dtype=float)
    mask = np.asarray([True, False])

    mean = extract_af3_confidence_metrics(
        confidence_file_path=str(confidence_path),
        atom_array=atom_array,
        mask=mask,
        metrics_to_extract="atom_plddts",
        return_mean=True,
    )

    assert mean == pytest.approx(26.55, abs=1e-6)
