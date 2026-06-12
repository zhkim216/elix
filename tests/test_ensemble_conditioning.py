import numpy as np
import torch
from biotite.structure import AtomArray

from allatom_design.eval.utils.ensemble_conditioning import (
    apply_ensemble_noise,
    repeat_batch_for_ensembles,
)


def _make_atom_array(n_atoms: int = 4) -> AtomArray:
    atom_array = AtomArray(n_atoms)
    atom_array.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ][:n_atoms],
        dtype=np.float32,
    )
    atom_array.atom_name = np.array(["N", "CA", "C", "O"][:n_atoms])
    atom_array.occupancy = np.ones(n_atoms, dtype=np.float32)
    atom_array.set_annotation("token_id", np.zeros(n_atoms, dtype=np.int64))
    return atom_array


def _make_batch() -> dict:
    atom_array = _make_atom_array()
    coords = torch.as_tensor(atom_array.coord, dtype=torch.float32).unsqueeze(0)
    batch = {
        "coords": coords,
        "noised_coords": coords.clone(),
        "noised_ca_coords": torch.zeros(1, 1, 3),
        "noised_n_coords": torch.zeros(1, 1, 3),
        "noised_c_coords": torch.zeros(1, 1, 3),
        "noised_o_coords": torch.zeros(1, 1, 3),
        "noised_pseudo_cb_coords": torch.zeros(1, 1, 3),
        "atom_pad_mask": torch.ones(1, 4),
        "atom_resolved_mask": torch.ones(1, 4),
        "atom_is_protein_chain": torch.ones(1, 4),
        "atom_is_metal_chain": torch.zeros(1, 4),
        "atom_is_small_molecule_chain": torch.zeros(1, 4),
        "atom_is_prot_std_aa": torch.ones(1, 4),
        "token_pad_mask": torch.ones(1, 1),
        "atom_array": [atom_array],
        "example_id": ["example"],
    }
    return batch


def test_repeat_batch_for_ensembles_repeats_tensors_and_lists():
    batch = _make_batch()

    repeated = repeat_batch_for_ensembles(batch, 3)

    assert repeated["coords"].shape[0] == 3
    assert repeated["example_id"] == ["example", "example", "example"]
    torch.testing.assert_close(repeated["coords"][0], batch["coords"][0])
    torch.testing.assert_close(repeated["coords"][2], batch["coords"][0])


def test_apply_ensemble_noise_uses_category_specific_std():
    batch = _make_batch()
    batch["atom_is_protein_chain"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    batch["atom_is_metal_chain"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    batch["atom_is_small_molecule_chain"] = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    batch["atom_is_prot_std_aa"] = torch.zeros(1, 4)

    noised = apply_ensemble_noise(
        batch,
        {
            "enabled": True,
            "num_ensembles": 1,
            "reduce": "mean",
            "noise_seed": 11,
            "noise_std": {
                "protein": 0.0,
                "metal": 0.2,
                "nonpolymer": 0.0,
            },
        },
    )

    delta = noised["noised_coords"] - batch["coords"]
    torch.testing.assert_close(delta[0, 0], torch.zeros(3))
    assert not torch.allclose(delta[0, 1], torch.zeros(3))
    torch.testing.assert_close(delta[0, 2], torch.zeros(3))
    torch.testing.assert_close(delta[0, 3], torch.zeros(3))


def test_apply_ensemble_noise_recomputes_token_backbone_fields_from_noised_atoms():
    batch = _make_batch()

    noised = apply_ensemble_noise(
        batch,
        {
            "enabled": True,
            "num_ensembles": 1,
            "reduce": "mean",
            "noise_seed": 7,
            "noise_std": {
                "protein": 0.1,
                "metal": 0.0,
                "nonpolymer": 0.0,
            },
        },
    )

    noised_coords = noised["noised_coords"][0]
    torch.testing.assert_close(noised["noised_n_coords"][0, 0], noised_coords[0])
    torch.testing.assert_close(noised["noised_ca_coords"][0, 0], noised_coords[1])
    torch.testing.assert_close(noised["noised_c_coords"][0, 0], noised_coords[2])
    torch.testing.assert_close(noised["noised_o_coords"][0, 0], noised_coords[3])

    b_vec = noised_coords[1] - noised_coords[0]
    c_vec = noised_coords[2] - noised_coords[1]
    a_vec = torch.cross(b_vec, c_vec, dim=-1)
    expected_pseudo_cb = (
        -0.58273431 * a_vec
        + 0.56802827 * b_vec
        - 0.54067466 * c_vec
        + noised_coords[1]
    )
    torch.testing.assert_close(noised["noised_pseudo_cb_coords"][0, 0], expected_pseudo_cb)
