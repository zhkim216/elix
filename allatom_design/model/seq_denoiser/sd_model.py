import copy
import math
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn
from biotite.structure import AtomArray
from omegaconf import DictConfig
from torchtyping import TensorType

from allatom_design.model.seq_denoiser.mask_selector import MaskSelector
from allatom_design.model.seq_denoiser.denoisers.elix_mpnn_denoiser import \
    ElixMPNNDenoiser
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser


class SeqDenoiser(nn.Module):
    """
    Sequence denoiser model.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.task

        self.denoiser = get_denoiser(cfg.denoiser)

        # Mask selector
        self.mask_selector = MaskSelector(cfg.mask_selector)


    def setup(self):
        # Initialize denoiser pre-trained weights if needed
        self.denoiser.setup()


    def forward(self,
                batch: dict[str, TensorType["b ..."]],
                t: TensorType["b", float] | None = None
                ) -> dict[str, TensorType["b ..."]]:
        outputs = {}

        # Copy batch to avoid modifying the original
        batch = copy.deepcopy(batch)

        with torch.no_grad():
            # Sample sequence and atom conditioning masks
            batch["seq_cond_mask"] = self.mask_selector.sample_seq_cond_mask(batch, t)  # 1 if we should condition on the restype, 0 otherwise
            batch["atom_cond_mask"], scn_token_mask, scn_atom_mask = self.mask_selector.sample_atom_cond_mask(batch)
            batch["sidechain_context_token_mask"] = scn_token_mask
            batch["sidechain_context_atom_mask"] = scn_atom_mask
            batch["seq_cond_mask"] = (batch["seq_cond_mask"] + scn_token_mask).clamp(max=1.0)

        # Denoise sequence
        _, aux_preds = self.denoiser(batch)

        # Additional outputs for computing loss
        outputs.update(aux_preds)

        return outputs


    def sample(self,
               batch: dict[str, TensorType["b ..."]],
               sampling_inputs: dict[str, Any],
               potts_aux_provider: Callable | None = None,
               ) -> tuple[dict[str, list[AtomArray]], dict[str, Any]]:

        # Handle inference noise labels
        batch["noise_labels"] = sampling_inputs.get("noise_labels", None)
        batch["noise"] = None

        if batch["noise_labels"] is not None:
            raise NotImplementedError("Noise labels are not implemented yet")

        if sampling_inputs.get("t", None) is not None:
            batch["t"] = torch.full((batch["token_pad_mask"].shape[0],), fill_value=sampling_inputs["t"], device=batch["token_pad_mask"].device)

        # Choose sampling method
        if sampling_inputs.get("use_potts_sampling", False):
            id_to_atom_arrays, aux = self.denoiser.potts_sample(
                batch,
                sampling_inputs,
                potts_aux_provider=potts_aux_provider,
            )
        else:
            raise ValueError("No sampling method specified. Set use_potts_sampling=True.")

        return id_to_atom_arrays, aux


    def score_samples(self, batch: dict[str, TensorType["b ..."]], sampling_inputs: dict[str, Any]):
        """
        Score samples using Potts parameters computed from input backbones.
        """
        batch["noise_labels"] = sampling_inputs.get("noise_labels", None)
        batch["noise"] = None

        if batch["noise_labels"] is not None:
            raise NotImplementedError("Noise labels are not implemented yet")

        if sampling_inputs["add_noise"]:
            raise NotImplementedError("Adding noise is not implemented yet")

        potts_decoder_aux, batch = self.denoiser.compute_potts_params(batch, sampling_inputs=sampling_inputs)

        return potts_decoder_aux, batch


def get_denoiser(cfg: DictConfig) -> BaseSeqDenoiser:
    """
    Get the denoiser specified in the config.
    """
    if cfg.name == "elix_mpnn":
        return ElixMPNNDenoiser(cfg)
    raise ValueError(f"Unknown denoiser: {cfg.name}")
