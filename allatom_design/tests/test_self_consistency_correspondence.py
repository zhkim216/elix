from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray

import atomworks.enums as aw_enums
from allatom_design.eval.metrics.self_consistency import (
    compute_self_consistency_metrics_atomarray,
)


def _two_chain_ca_array(*, chain_order: tuple[str, str], translation=None) -> AtomArray:
    coordinates = {
        "A": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "B": np.asarray([[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]]),
    }
    atom_array = AtomArray(6)
    atom_array.coord = np.concatenate([coordinates[chain] for chain in chain_order])
    if translation is not None:
        atom_array.coord += np.asarray(translation)
    atom_array.chain_id = np.asarray(
        [chain for chain in chain_order for _ in range(3)]
    )
    atom_array.res_id = np.asarray([1, 2, 3, 1, 2, 3])
    atom_array.res_name = np.asarray(["ALA", "GLY", "SER"] * 2)
    atom_array.atom_name = np.asarray(["CA"] * 6)
    atom_array.element = np.asarray(["C"] * 6)
    atom_array.hetero = np.asarray([False] * 6)
    atom_array.set_annotation(
        "pn_unit_iid",
        np.asarray([f"{chain}_1" for chain in chain_order for _ in range(3)]),
    )
    atom_array.set_annotation(
        "chain_type",
        np.asarray([aw_enums.ChainType.POLYPEPTIDE_L] * 6, dtype=object),
    )
    return atom_array


def test_self_consistency_preserves_keyed_pairs_when_chain_order_differs(
    monkeypatch,
    tmp_path,
) -> None:
    sample = _two_chain_ca_array(chain_order=("A", "B"))
    prediction = _two_chain_ca_array(
        chain_order=("B", "A"),
        translation=(10.0, -3.0, 2.0),
    )
    monkeypatch.setattr(
        "allatom_design.eval.metrics.self_consistency.extract_af3_confidence_metrics",
        lambda **_: 80.0,
    )

    metrics = compute_self_consistency_metrics_atomarray(
        sample_atom_array=sample,
        pred_atom_array=prediction,
        pred_sample_path=tmp_path / "toy_model.cif",
        save_aligned=False,
    )

    assert metrics["sc_ca_rmsd"] < 1e-5
    assert metrics["avg_ca_plddt"] == 80.0
