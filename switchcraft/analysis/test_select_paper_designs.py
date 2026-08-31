import math
from pathlib import Path

import numpy as np

import analysis.select_paper_designs as selection
from analysis.select_paper_designs import (
    SelectionError,
    _motif_reference_ca,
    _validate_ligand_ca,
    allostery_gates,
    ligand_discrimination_gates,
    ligand_modification_gates,
    motif_switching_gates,
    selection_memberships,
)


def test_allostery_equality_boundaries() -> None:
    positive = allostery_gates("positive_allostery", 1.0, 0.5, 1.0, 0.5)
    assert not positive["state_A_direction_pass"]
    assert positive["state_B_direction_pass"]
    assert positive["state_A_std_pass"] and positive["state_B_std_pass"]

    negative = allostery_gates("negative_allostery", 1.0, 0.5, 1.0, 0.5)
    assert negative["state_A_direction_pass"]
    assert not negative["state_B_direction_pass"]


def test_allostery_appendix_and_main_delta_conflict() -> None:
    result = allostery_gates("positive_allostery", 1.25, 0.2, 0.75, 0.2)
    assert result["appendix_b1_pass"]
    assert not result["main_text_4_1_pass"]
    assert result["appendix_b1_score"] == 0.0

    main_equality = allostery_gates("positive_allostery", 1.5, 0.2, 0.5, 0.2)
    assert main_equality["appendix_b1_pass"]
    assert not main_equality["main_text_4_1_pass"]
    assert main_equality["main_text_4_1_score"] == 0.0


def test_ligand_modification_or_and_both_overlap() -> None:
    literal_only = ligand_modification_gates(2.1, 0.9, 1.1)
    assert literal_only["appendix_b3_literal_or_pass"]
    assert not literal_only["appendix_b3_both_states_pass"]
    assert math.isclose(
        literal_only["appendix_b3_literal_or_score"], literal_only["cross_AB_margin"]
    )

    overlap = ligand_modification_gates(2.1, 0.9, 0.8)
    assert overlap["appendix_b3_literal_or_pass"]
    assert overlap["appendix_b3_both_states_pass"]
    memberships = selection_memberships({"family": "ligand_modification", **overlap})
    assert [name for name, _ in memberships] == [
        "appendix_b3_literal_or",
        "appendix_b3_both_states",
    ]


def test_ligand_modification_strict_equalities_fail() -> None:
    assert not ligand_modification_gates(2.0, 0.5, 0.5)["cross_AB_pass"]
    assert not ligand_modification_gates(3.0, 1.0, 0.5)["intra_A_pass"]


def test_motif_switching_direction_equalities() -> None:
    metrics = {
        "motif_3ixt_state_A_mean": 1.0,
        "motif_3ixt_state_A_std": 0.5,
        "motif_1ycr_state_A_mean": 1.0,
        "motif_1ycr_state_A_std": 0.5,
        "motif_3ixt_state_B_mean": 1.0,
        "motif_3ixt_state_B_std": 0.5,
        "motif_1ycr_state_B_mean": 1.0,
        "motif_1ycr_state_B_std": 0.5,
    }
    result = motif_switching_gates(metrics)
    assert result["motif_3ixt_state_A_direction_pass"]
    assert not result["motif_1ycr_state_A_direction_pass"]
    assert not result["motif_3ixt_state_B_direction_pass"]
    assert result["motif_1ycr_state_B_direction_pass"]
    assert not result["motif_3ixt_switch_pass"]
    assert not result["motif_1ycr_switch_pass"]
    assert not result["appendix_b2_pass"]


def test_motif_switching_per_motif_flags_and_scores() -> None:
    metrics = {
        "motif_3ixt_state_A_mean": 0.5,
        "motif_3ixt_state_A_std": 0.1,
        "motif_1ycr_state_A_mean": 1.5,
        "motif_1ycr_state_A_std": 0.1,
        "motif_3ixt_state_B_mean": 1.5,
        "motif_3ixt_state_B_std": 0.1,
        "motif_1ycr_state_B_mean": 1.2,
        "motif_1ycr_state_B_std": 0.1,
    }
    result = motif_switching_gates(metrics)
    assert result["motif_3ixt_switch_pass"]
    assert math.isclose(result["motif_3ixt_switch_score"], 0.4)
    assert not result["motif_1ycr_switch_pass"]
    assert not result["appendix_b2_pass"]


def test_ligand_ca_requires_exactly_50_valid_residues() -> None:
    ca = np.zeros((50, 3), dtype=np.float32)
    valid = np.ones(50, dtype=bool)
    assert _validate_ligand_ca(ca, valid, Path("valid.pdb")) is ca
    for invalid_ca, invalid_mask in (
        (np.zeros((51, 3), dtype=np.float32), np.ones(51, dtype=bool)),
        (ca, np.r_[np.ones(49, dtype=bool), False]),
    ):
        try:
            _validate_ligand_ca(invalid_ca, invalid_mask, Path("invalid.pdb"))
        except SelectionError:
            pass
        else:
            raise AssertionError("invalid ligand CA contract was accepted")


def test_motif_reference_ca_is_cached_per_process() -> None:
    selection._MOTIF_REFERENCE_CA_CACHE.clear()
    original = selection._parse_protein
    calls = []

    def fake_parse(path: Path):
        calls.append(path)
        return object(), np.ones((3, 3), dtype=np.float32), np.ones(3, dtype=bool)

    selection._parse_protein = fake_parse
    try:
        first = _motif_reference_ca(Path("reference.pdb"))
        second = _motif_reference_ca(Path("reference.pdb"))
    finally:
        selection._parse_protein = original
        selection._MOTIF_REFERENCE_CA_CACHE.clear()
    assert first is second
    assert len(calls) == 1


def test_ligand_discrimination_strict_equalities_fail() -> None:
    metrics = {
        "intra_A_mean": 1.0,
        "intra_B_mean": 0.9,
        "intra_C_mean": 0.9,
        "cross_AB_mean": 1.0,
        "cross_AC_mean": 1.1,
        "cross_BC_mean": 1.1,
    }
    result = ligand_discrimination_gates(metrics)
    assert not result["intra_A_pass"]
    assert not result["cross_AB_pass"]
    assert not result["appendix_b5_pass"]
