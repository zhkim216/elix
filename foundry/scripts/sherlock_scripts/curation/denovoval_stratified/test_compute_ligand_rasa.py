from __future__ import annotations

import numpy as np

from compute_ligand_rasa import target_ligand_heavy_mask


class FakeAtomArray:
    def __init__(self) -> None:
        self.coord = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [float("nan"), 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        )
        self.element = np.asarray(["C", "C", "H", "C", "C"])
        self.res_name = np.asarray(["GLU", "GLU", "GLU", "GLU", "MET"])
        self.hetero = np.asarray([False, True, True, True, True])


def test_target_mask_excludes_same_named_protein_residue() -> None:
    mask = target_ligand_heavy_mask(FakeAtomArray(), "glu")

    assert mask.tolist() == [False, True, False, False, False]
