from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import allatom_design.data.const as const
from allatom_design.model.seq_denoiser.denoisers.elix_mpnn_denoiser import (
    ElixMPNNDenoiser,
)
from allatom_design.model.seq_denoiser.denoisers.seq_design import frustration


def test_pairwise_frustration_uses_full_q23_population_statistics() -> None:
    q = 23
    values = torch.arange(q * q, dtype=torch.float32).reshape(q, q)
    J = torch.stack((values, torch.zeros_like(values))).reshape(1, 2, 1, q, q)
    mask_ij = torch.tensor([[[1.0], [0.0]]])

    F = frustration.compute_pairwise_frustration(J, mask_ij)

    expected = (values - values.mean()) / values.std(correction=0)
    torch.testing.assert_close(F[0, 0, 0], expected)
    torch.testing.assert_close(F[0, 0, 0].mean(), torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        F[0, 0, 0].std(correction=0),
        torch.tensor(1.0),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.equal(F[0, 1, 0], torch.zeros_like(values))
    assert torch.isfinite(F).all()


def test_mix_pairwise_couplings_matches_explicit_alpha_j_plus_beta_f() -> None:
    generator = torch.Generator().manual_seed(7)
    J = torch.randn((1, 2, 1, 23, 23), generator=generator)
    mask_ij = torch.ones((1, 2, 1))
    alpha = 0.6
    beta = 0.4

    F = frustration.compute_pairwise_frustration(J, mask_ij)
    mixed = frustration.mix_pairwise_couplings(
        J,
        mask_ij,
        alpha=alpha,
        beta=beta,
    )

    torch.testing.assert_close(mixed, alpha * J + beta * F)

    shifted = frustration.mix_pairwise_couplings(
        J + 3.0,
        mask_ij,
        alpha=alpha,
        beta=beta,
    )
    torch.testing.assert_close(shifted - mixed, torch.full_like(mixed, alpha * 3.0))


def test_valid_zero_variance_edge_remains_undefined_without_floor() -> None:
    J = torch.ones((1, 1, 1, 23, 23))
    mask_ij = torch.ones((1, 1, 1))

    F_ij = frustration.compute_pairwise_frustration(J, mask_ij)

    assert torch.isnan(F_ij).all()


@pytest.mark.parametrize(
    "config_path",
    [
        "allatom_design/configs/seq_des/elix_mpnn_inference.yaml",
        "allatom_design/configs_local/seq_des/elix_mpnn_inference.yaml",
    ],
)
def test_sampling_config_defaults_disable_frustration(config_path: str) -> None:
    cfg = OmegaConf.load(config_path)
    assert OmegaConf.to_container(cfg.potts_sampling_cfg.frustration, resolve=True) == {
        "enabled": False,
        "alpha": 1.0,
        "beta": 0.0,
    }


def test_frustration_rejects_guidance_until_branch_order_is_defined() -> None:
    denoiser = object.__new__(ElixMPNNDenoiser)
    batch = {"seq_cond_mask": torch.zeros((1, 1))}
    sampling_inputs = {
        "potts_sampling_cfg": {
            "potts_only_cond": False,
            "frustration": {"enabled": True, "alpha": 1.0, "beta": 0.5},
            "guidance_cfg": {"enabled": True, "mode": "cond_uncond"},
        }
    }

    with pytest.raises(
        NotImplementedError,
        match="frustration.*guidance",
    ):
        denoiser.potts_sample(batch, sampling_inputs)


def test_potts_sample_applies_frustration_only_when_enabled() -> None:
    class _CapturingGraphPotts:
        def __init__(self) -> None:
            self.J = None

        def sample(self, h, J, edge_idx, mask_i, mask_ij, **kwargs):
            self.J = J.detach().clone()
            return kwargs["S"].clone(), torch.zeros(h.shape[0])

    denoiser = object.__new__(ElixMPNNDenoiser)
    torch.nn.Module.__init__(denoiser)
    denoiser.use_potts_encoding = False
    denoiser.sequence_encoding = const.AF3_ENCODING
    capturing_decoder = _CapturingGraphPotts()
    denoiser.elix_mpnn = SimpleNamespace(decoder_S_potts=capturing_decoder)

    q = const.AF3_ENCODING.n_tokens
    generator = torch.Generator().manual_seed(11)
    raw_J = torch.randn((1, 2, 1, q, q), generator=generator)
    potts_aux = {
        "h": torch.zeros((1, 2, q)),
        "J": raw_J.clone(),
        "edge_idx": torch.tensor([[[1], [0]]]),
        "mask_i": torch.ones((1, 2)),
        "mask_ij": torch.ones((1, 2, 1)),
    }
    restype_idx = torch.tensor([[0, 1]])
    batch = {
        "restype": F.one_hot(restype_idx, num_classes=q).float(),
        "seq_cond_mask": torch.zeros((1, 2)),
        "token_pad_mask": torch.ones((1, 2)),
    }

    def _compute_potts_params(self, batch, sampling_inputs):
        batch["protein_residue_node_mask"] = potts_aux["mask_i"]
        batch["token_exists_mask"] = potts_aux["mask_i"]
        return potts_aux, batch, sampling_inputs

    def _postprocess(self, S_list, batch, per_sample_aux=None):
        return {"test": S_list}, {"test": per_sample_aux}

    def _keep_sampled_tokens(self, S, batch):
        return S

    denoiser.compute_potts_params = MethodType(_compute_potts_params, denoiser)
    denoiser._postprocess_sampled_sequences = MethodType(_postprocess, denoiser)
    denoiser._set_non_protein_tokens = MethodType(_keep_sampled_tokens, denoiser)

    sampling_inputs = {
        "omit_aas": None,
        "num_seqs_per_pdb": 1,
        "potts_sampling_cfg": {
            "potts_only_cond": False,
            "regularization": None,
            "potts_sweeps": 1,
            "potts_proposal": "dlmc",
            "potts_temperature": 1.0,
            "rejection_step": False,
            "frustration": {"enabled": False, "alpha": 0.75, "beta": 0.25},
        },
    }

    denoiser.potts_sample(batch, sampling_inputs)
    torch.testing.assert_close(capturing_decoder.J, raw_J)

    potts_aux["J"] = raw_J.clone()
    sampling_inputs["potts_sampling_cfg"]["frustration"]["enabled"] = True
    denoiser.potts_sample(batch, sampling_inputs)
    expected = frustration.mix_pairwise_couplings(
        raw_J,
        potts_aux["mask_ij"],
        alpha=0.75,
        beta=0.25,
    )
    torch.testing.assert_close(capturing_decoder.J, expected)
