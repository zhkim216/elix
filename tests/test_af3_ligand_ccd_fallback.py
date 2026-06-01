import numpy as np
from biotite.structure import AtomArray

from allatom_design.eval.eval_utils.folding_utils import _resolve_af3_ligand_ccd_code


def _ligand_atom_array(
    *,
    elements: list[str],
    res_names: list[str],
    pn_unit_iid: str = "B_1",
) -> AtomArray:
    atom_array = AtomArray(len(elements))
    atom_array.coord = np.zeros((len(elements), 3), dtype=np.float32)
    atom_array.chain_id = np.array(["B"] * len(elements))
    atom_array.res_id = np.ones(len(elements), dtype=int)
    atom_array.ins_code = np.array([""] * len(elements))
    atom_array.res_name = np.array(res_names)
    atom_array.hetero = np.ones(len(elements), dtype=bool)
    atom_array.atom_name = np.array([f"{element}{idx + 1}" for idx, element in enumerate(elements)])
    atom_array.element = np.array(elements)
    atom_array.occupancy = np.ones(len(elements), dtype=np.float32)
    atom_array.set_annotation("pn_unit_iid", np.array([pn_unit_iid] * len(elements)))
    return atom_array


def test_single_atom_metal_ligand_uses_element_for_synthetic_ccd() -> None:
    atom_array = _ligand_atom_array(elements=["CU"], res_names=["l01"])

    assert (
        _resolve_af3_ligand_ccd_code(
            designed_sample_atom_array=atom_array,
            ligand_pn_unit_iid="B_1",
            ligand_ccd_code="l01",
        )
        == "CU"
    )


def test_existing_metal_ccd_is_normalized_without_atom_fallback() -> None:
    atom_array = _ligand_atom_array(elements=["CU"], res_names=["CU"])

    assert (
        _resolve_af3_ligand_ccd_code(
            designed_sample_atom_array=atom_array,
            ligand_pn_unit_iid="B_1",
            ligand_ccd_code="cu",
        )
        == "CU"
    )


def test_multi_atom_ligand_keeps_synthetic_ccd() -> None:
    atom_array = _ligand_atom_array(elements=["FE", "S"], res_names=["l01", "l01"])

    assert (
        _resolve_af3_ligand_ccd_code(
            designed_sample_atom_array=atom_array,
            ligand_pn_unit_iid="B_1",
            ligand_ccd_code="l01",
        )
        == "l01"
    )


def test_single_atom_nonmetal_ligand_keeps_synthetic_ccd() -> None:
    atom_array = _ligand_atom_array(elements=["C"], res_names=["l01"])

    assert (
        _resolve_af3_ligand_ccd_code(
            designed_sample_atom_array=atom_array,
            ligand_pn_unit_iid="B_1",
            ligand_ccd_code="l01",
        )
        == "l01"
    )
