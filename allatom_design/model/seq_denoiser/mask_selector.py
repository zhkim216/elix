import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType


class MaskSelector:
    def __init__(self, cfg: DictConfig):
        """
        Handles selecting masks for training the sequence design model.
        """
        super().__init__()
        self.cfg = cfg

        self.restype_masking_schedule = cfg.restype_masking_schedule
        self.restype_masking_cfg = OmegaConf.to_container(cfg.restype_masking_cfg[self.restype_masking_schedule], resolve=True)  # to dict to avoid dataloader issues?
        self.atom_masking_schedule = cfg.atom_masking_schedule
        self.atom_masking_cfg = OmegaConf.to_container(cfg.atom_masking_cfg[self.atom_masking_schedule], resolve=True)
        self.scn_context_ratio = cfg.scn_context_ratio


    def sample_seq_cond_mask(self,
                             batch: dict[str, TensorType["b ..."]],
                             t: TensorType["b", float] | None = None
                             ) -> TensorType["b n_tokens", float]:
        """
        Create a mask denoting which restypes to mask out.
        0 if we should mask, 1 if we should keep. Non-protein restypes are always kept (1).
        """
        B, N = batch["token_pad_mask"].shape
        device = batch["token_pad_mask"].device

        if t is None:
            # Sample timestep
            t = self._sample_t(B, device=device, schedule=self.restype_masking_schedule, cfg=self.restype_masking_cfg)

        # Create mask based on timestep
        seq_cond_mask = torch.rand(B, N, device=device) < rearrange(t, "b -> b 1")

        # Non-protein and non-standard restypes are always kept
        standard_aa_prot_token_mask = batch["token_is_prot_std_aa"] * batch["token_resolved_mask"] * batch["token_pad_mask"]

        seq_cond_mask = torch.where(~standard_aa_prot_token_mask.bool(),
                                    torch.ones_like(seq_cond_mask),
                                    seq_cond_mask)

        seq_cond_mask = seq_cond_mask * batch["token_pad_mask"] * batch["token_resolved_mask"]  # mask out padding, non-resolved entries


        return seq_cond_mask


    def sample_atom_cond_mask(
        self, batch: dict[str, TensorType["b ..."]]
    ) -> tuple[TensorType["b n_atoms", float], TensorType["b n_tokens", float], TensorType["b n_tokens", float]]:
        """
        Create atom-level conditioning and sidechain-context masks.

        Returns:
            atom_cond_mask:                 [B, n_atoms]  — 1 = keep atom, 0 = mask
            sidechain_context_token_mask:   [B, n_tokens] — 1 = token with sidechain context visible
            sidechain_context_atom_mask:    [B, n_atoms]  — 1 = sidechain context atom, 0 otherwise
        """
        B, N = batch["token_pad_mask"].shape
        device = batch["atom_resolved_mask"].device

        atom_cond_mask = batch["atom_resolved_mask"].clone()

        standard_aa_prot_token_mask = batch["token_is_prot_std_aa"] * batch["token_resolved_mask"] * batch["token_pad_mask"]
        standard_aa_prot_atom_mask = batch["atom_is_prot_std_aa"] * batch["atom_resolved_mask"] * batch["atom_pad_mask"]
        standard_aa_prot_bb_atom_mask = standard_aa_prot_atom_mask * batch["prot_bb_atom_mask"]
        standard_aa_prot_scn_atom_mask = standard_aa_prot_atom_mask * batch["prot_scn_atom_mask"]

        token_has_sidechain = torch.zeros(B, N, device=device)
        token_has_sidechain.scatter_reduce_(
            1,
            batch["atom_to_token_map"],
            standard_aa_prot_scn_atom_mask,
            reduce="amax",
            include_self=False,
        )
        token_eligible = standard_aa_prot_token_mask * (token_has_sidechain > 0).float()

        # --- Select scn_context_ratio fraction from eligible tokens ---
        target_count = (token_eligible.sum(dim=-1) * self.scn_context_ratio).long()
        random_priority = torch.where(
            token_eligible.bool(),
            torch.rand_like(token_eligible),
            torch.full_like(token_eligible, -float("inf"))
        )
        rank = random_priority.argsort(dim=-1, descending=True).argsort(dim=-1)
        sidechain_context_token_mask = token_eligible * (rank < target_count.unsqueeze(-1)).float()

        # --- Build atom_cond_mask ---
        atomwise_sidechain_context_token_mask = (
            sidechain_context_token_mask.gather(dim=-1, index=batch["atom_to_token_map"])
            * batch["atom_pad_mask"]
            * batch["atom_resolved_mask"]
        )

        selected_token_atom_mask = (standard_aa_prot_bb_atom_mask + standard_aa_prot_scn_atom_mask)
        prot_atom_mask = torch.where(
            atomwise_sidechain_context_token_mask.bool(),
            selected_token_atom_mask,
            standard_aa_prot_bb_atom_mask,
        )
        prot_atom_mask = prot_atom_mask * batch["atom_pad_mask"] * batch["atom_resolved_mask"]
        sidechain_context_atom_mask = atomwise_sidechain_context_token_mask * standard_aa_prot_scn_atom_mask

        atom_cond_mask = torch.where(standard_aa_prot_atom_mask.bool(), prot_atom_mask, atom_cond_mask)
        atom_cond_mask = atom_cond_mask * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

        return atom_cond_mask, sidechain_context_token_mask, sidechain_context_atom_mask

    def _sample_t(
        self,
        B: int = None,
        device: torch.device = None,
        schedule: str = None,
        cfg: dict = None,
    ) -> TensorType["b", float]:
        """
        Sample a timestep from the masking schedule.
        t = probability of keeping the restype unmasked

        Args:
            B: batch size
            device: torch device
            schedule: masking schedule name (defaults to restype_masking_schedule)
            cfg: masking config dict (defaults to restype_masking_cfg)
        """
        # Use defaults if not provided (backward compatible)
        if schedule is None:
            schedule = self.restype_masking_schedule
        if cfg is None:
            cfg = self.restype_masking_cfg

        if schedule == "constant_t":
            t = torch.ones(B, device=device) * cfg["t"]
        elif schedule.startswith("uniform"):
            # sample time from uniform distribution
            t_min, t_max = cfg["t_min"], cfg["t_max"]
            t = torch.rand(B, device=device) * (t_max - t_min) + t_min

            # apply transformation to t
            if schedule == "uniform_t":
                t = t
            elif schedule == "uniform_squared_t":
                t = t ** 2
            elif schedule == "uniform_cubed_t":
                t = t ** 3
            elif schedule == "uniform_cosine_t":
                t = 1 - torch.cos(t * np.pi / 2)
            elif schedule == "uniform_sqrt_t":
                t = t ** 0.5
            elif schedule == "uniform_cbrt_t":
                t = t ** (1/3)
        else:
            raise ValueError(f"Unknown masking schedule: {schedule}")

        return t
