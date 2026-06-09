import torch
from omegaconf import OmegaConf

from allatom_design.model.seq_denoiser.denoisers.elix_mpnn_denoiser import ElixMPNNDenoiser
from allatom_design.model.seq_denoiser.mask_selector import MaskSelector


def _make_selector(scn_context_ratio: float) -> MaskSelector:
    cfg = OmegaConf.create(
        {
            "restype_masking_schedule": "constant_t",
            "atom_masking_schedule": "constant_t",
            "masking_cfg": {"constant_t": {"t": 0.0}},
            "restype_masking_cfg": {"constant_t": {"t": 0.0}},
            "atom_masking_cfg": {"constant_t": {"t": 0.0}},
            "scn_context_ratio": scn_context_ratio,
            "pseudo_ligand_backbone_mask_radius": 1,
        }
    )
    return MaskSelector(cfg)


def _make_batch() -> dict[str, torch.Tensor]:
    # token 0: protein residue with N, CA, C, O, CB, CG
    # token 1: protein residue with backbone only
    # token 2: non-protein atom
    atom_to_token_map = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2]])
    atom_resolved_mask = torch.ones(1, 11)
    atom_pad_mask = torch.ones(1, 11)

    prot_bb_atom_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]], dtype=torch.float32)
    prot_scn_atom_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    atom_is_prot_std_aa = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32)

    return {
        "atom_to_token_map": atom_to_token_map,
        "atom_resolved_mask": atom_resolved_mask,
        "atom_pad_mask": atom_pad_mask,
        "token_is_prot_std_aa": torch.tensor([[1, 1, 0]], dtype=torch.float32),
        "token_resolved_mask": torch.ones(1, 3),
        "token_pad_mask": torch.ones(1, 3),
        "atom_is_prot_std_aa": atom_is_prot_std_aa,
        "prot_bb_atom_mask": prot_bb_atom_mask,
        "prot_scn_atom_mask": prot_scn_atom_mask,
        "prot_scn_wo_cb_atom_mask": torch.tensor(
            [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32
        ),
        "can_be_pseudo_ligand": torch.zeros(1, 11),
    }


def test_selected_sidechain_context_keeps_backbone_and_cbeta():
    selector = _make_selector(scn_context_ratio=1.0)
    batch = _make_batch()

    atom_cond_mask, token_mask, context_atom_mask = selector.sample_atom_cond_mask(batch)

    assert torch.equal(token_mask, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.equal(
        atom_cond_mask,
        torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32),
    )
    assert torch.equal(
        context_atom_mask,
        torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32),
    )


def test_zero_sidechain_context_ratio_keeps_default_backbone_only_for_protein():
    selector = _make_selector(scn_context_ratio=0.0)
    batch = _make_batch()

    atom_cond_mask, token_mask, context_atom_mask = selector.sample_atom_cond_mask(batch)

    assert torch.equal(token_mask, torch.zeros(1, 3))
    assert torch.equal(
        atom_cond_mask,
        torch.tensor([[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.float32),
    )
    assert torch.equal(context_atom_mask, torch.zeros(1, 11))


def test_pseudo_context_mask_no_longer_excludes_training_graph_nodes():
    batch = {
        "seq_cond_mask": torch.zeros(1, 3),
        "token_resolved_mask": torch.ones(1, 3),
        "token_pad_mask": torch.ones(1, 3),
        "atom_cond_mask": torch.ones(1, 2),
        "atom_resolved_mask": torch.ones(1, 2),
        "atom_pad_mask": torch.ones(1, 2),
        "token_is_prot_std_aa": torch.tensor([[1.0, 1.0, 0.0]]),
        "pseudo_context_mask": torch.tensor([[1.0, 0.0, 0.0]]),
    }

    out = ElixMPNNDenoiser.build_masks(None, batch, is_sampling=False)

    assert torch.equal(out["protein_residue_node_mask"], torch.tensor([[1.0, 1.0, 0.0]]))
