import copy
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from atomworks.ml.utils.token import apply_token_wise, spread_token_wise
from biotite.structure import AtomArray
from einops import rearrange
from omegaconf import DictConfig
from torchtyping import TensorType
from tqdm import tqdm

import allatom_design.data.const as const
import allatom_design.data.transform.potts_encoding as potts_encoding
import allatom_design.model.seq_denoiser.denoisers.seq_design.potts as potts
from allatom_design.utils.feature_utils import slice_feats
from allatom_design.utils.tensor_utils import to
from allatom_design.model.seq_denoiser.denoisers.denoiser import \
    BaseSeqDenoiser
from allatom_design.model.seq_denoiser.denoisers.seq_design import complexity, frustration
from allatom_design.model.seq_denoiser.denoisers.seq_design.inference_schedule import (
    build_energy_density_schedule_trace,
    build_heat_capacity_schedule_trace,
)

logger = logging.getLogger(__name__)


def _sample_initial_sequence_probes(
    *,
    logits_init: torch.Tensor,
    finalized_mask_sample: torch.Tensor,
    sampling_sequence: torch.Tensor,
    first_sequence: torch.Tensor,
    num_sequences: int,
) -> torch.Tensor:
    """Draw distinct legal probes without changing generation starts."""

    if num_sequences < 1:
        raise ValueError("schedule calibration requires at least one probe sequence")
    batch_size = first_sequence.shape[0]
    selected = [[first_sequence[batch_index].clone()] for batch_index in range(batch_size)]
    max_draws = max(256, 64 * num_sequences)
    for _ in range(max_draws):
        if all(len(batch_probes) == num_sequences for batch_probes in selected):
            break
        _, _, probe = potts.init_sampling_masks(
            logits_init,
            mask_sample=finalized_mask_sample,
            S=sampling_sequence,
        )
        for batch_index, batch_probes in enumerate(selected):
            if len(batch_probes) == num_sequences:
                continue
            candidate = probe[batch_index]
            if all(not torch.equal(candidate, existing) for existing in batch_probes):
                batch_probes.append(candidate.clone())
    unique_counts = [len(batch_probes) for batch_probes in selected]
    if any(count != num_sequences for count in unique_counts):
        raise RuntimeError(
            f"failed to draw {num_sequences} distinct calibration sequences; "
            f"unique counts={unique_counts} after {max_draws} draws"
        )
    return torch.stack(
        [torch.stack(batch_probes, dim=0) for batch_probes in selected], dim=1
    )


class ElixMPNNDenoiser(BaseSeqDenoiser):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg
        self.task = cfg.task
        self.use_potts_encoding = bool(cfg.mpnn.get("use_potts_encoding", False))
        self.sequence_encoding = potts_encoding.selected_sequence_encoding(self.use_potts_encoding)

        # Keep the historical attribute name because sampling and analysis
        # consumers access ``elix_mpnn.decoder_S_potts`` directly.
        from allatom_design.model.seq_denoiser.denoisers.seq_design.elix_mpnn import (
            ElixMPNN,
        )

        self.elix_mpnn = ElixMPNN(cfg.mpnn)


    def forward(self,
                batch: dict[str, TensorType["b ..."]],
                is_sampling: bool = False,
                sampling_inputs: dict[str, Any] | None = None,
                ) -> tuple[TensorType["b n c", float],  # seq_logits
                           dict[str, TensorType["b ..."]]]:
        if self.use_potts_encoding:
            batch = potts_encoding.apply_potts_standard_aa_mask(batch)

        # Build some helpful masks based on conditioning sequence and atoms
        batch = self.build_masks(batch, is_sampling)

        if self.use_potts_encoding:
            batch = potts_encoding.apply_potts_restype_encoding(batch)
        else:
            batch["target_restype"] = batch["restype"].argmax(dim=-1)

        # Run model
        seq_logits, mpnn_feats = self.elix_mpnn(batch, is_sampling)

        # Outputs
        aux_preds = {
            "seq_logits": seq_logits,
            "potts_decoder_aux": mpnn_feats.get("potts_decoder_aux", None),
            "sidechain_prediction_aux": mpnn_feats.get("sidechain_prediction_aux", None),
            "pcp_shared_atom_count": mpnn_feats.get(
                "pcp_shared_atom_count",
                None,
            ),
            "seq_cond_mask": batch["seq_cond_mask"],
            "atom_cond_mask": batch["atom_cond_mask"],
            "sidechain_context_token_mask": batch["sidechain_context_token_mask"],
            "token_exists_mask": batch["token_exists_mask"],
            "protein_residue_node_mask": batch["protein_residue_node_mask"],
            "target_restype": batch["target_restype"],
            "restype": batch["restype"],
        }

        return seq_logits, aux_preds


    def build_masks(self, batch: dict[str, TensorType["b ..."]], is_sampling) -> dict[str, TensorType["b ..."]]:
        """
        Build various masks for ElixMPNN.

        Ensures that the conditioning masks only contain non-pad, resolved entries.
        Also, updates batch (in place) with:
        - atomwise_token_idx: Tensor["b n_atoms", int]: index of the token that the atom belongs to, 0 for pad atoms
        - atomwise_seq_cond_mask: Tensor["b n_atoms", float]: 1 if the atom is part of an unmasked residue type, or 0 otherwise
        - token_exists_mask: Tensor["b n_tokens", float]: 1 if there exists any unmasked atom in the token, or 0 otherwise
        """

        # Ensure the conditioning masks only contain non-pad, resolved entries.
        batch["seq_cond_mask"] = batch["seq_cond_mask"] * batch["token_resolved_mask"] * batch["token_pad_mask"]
        batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_resolved_mask"] * batch["atom_pad_mask"]
        batch["sidechain_context_token_mask"] = batch.get(
            "sidechain_context_token_mask",
            torch.zeros_like(batch["seq_cond_mask"]),
        )
        batch["sidechain_context_token_mask"] = (
            batch["sidechain_context_token_mask"]
            * batch["token_resolved_mask"]
            * batch["token_pad_mask"]
        )

        # Build mask for which tokens to include in the token-level grpah
        ## ensure center atom is present, since graph nodes are the center atom
        batch["token_exists_mask"] = batch["token_resolved_mask"].float()  # [b, n_tokens], "whether the token exists in the residue-level graph"

        ## sometimes, it's helpful to mask out certain tokens from the graph (e.g. for protein-only design in lcaliby or exclude hetero residues in sampling)
        token_exists_override = batch.get("token_exists_override", torch.ones_like(batch["token_exists_mask"]))
        batch["token_exists_mask"] = batch["token_exists_mask"] * token_exists_override

        # Mask out hetero residues in protein residue graphs for sampling, if specified.
        #Todo: Need to implement functionality for redesigning hetero residues into standard AA in the future.
        residuewise_hetero_mask = batch.get("residuewise_hetero_mask", torch.ones_like(batch["token_exists_mask"]))
        atomwise_hetero_mask = batch.get("atomwise_hetero_mask", torch.ones_like(batch["atom_resolved_mask"]))

        if not is_sampling:
            batch["protein_residue_node_mask"] = (
                batch["token_is_prot_std_aa"] *
                batch["token_exists_mask"] *
                batch["token_pad_mask"]
            )

        else:
            #Todo: Need to implement functionality for redesigning hetero residues into standard AA in the future.
            batch["protein_residue_node_mask"] = (
                batch["token_is_prot_std_aa"] *
                residuewise_hetero_mask *
                batch["token_exists_mask"] *
                batch["token_pad_mask"]
            )

            batch["atom_cond_mask"] = batch["atom_cond_mask"] * atomwise_hetero_mask

        return batch

    @staticmethod
    def _guidance_branch_batch(
        batch: dict[str, TensorType["b ..."]],
        atom_cond_mask: torch.Tensor,
    ) -> dict[str, TensorType["b ..."]]:
        branch_batch = dict(batch)
        branch_batch["atom_cond_mask"] = atom_cond_mask
        return branch_batch

    @staticmethod
    def _require_guidance_atom_cond_mask(
        guidance_cfg: dict[str, Any],
        key: str,
        batch: dict[str, TensorType["b ..."]],
    ) -> torch.Tensor:
        if key not in guidance_cfg:
            raise ValueError(f"guidance.mode=selectivity requires guidance_cfg.{key}")
        mask = guidance_cfg[key]
        if not torch.is_tensor(mask):
            raise TypeError(f"guidance_cfg.{key} must be a torch.Tensor, got {type(mask).__name__}")
        if mask.shape != batch["atom_cond_mask"].shape:
            raise ValueError(
                f"guidance_cfg.{key} shape {tuple(mask.shape)} does not match "
                f"atom_cond_mask shape {tuple(batch['atom_cond_mask'].shape)}"
            )
        return mask.to(device=batch["atom_cond_mask"].device, dtype=batch["atom_cond_mask"].dtype)

    @classmethod
    def _resolve_potts_guidance_branches(
        cls,
        batch: dict[str, TensorType["b ..."]],
        guidance_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        guidance_mode = str(guidance_cfg.get("mode", "cond_uncond"))
        if guidance_mode == "cond_uncond":
            uncond_mode = guidance_cfg.get("uncond_mode", "protein_only")
            if uncond_mode != "protein_only":
                raise NotImplementedError(
                    f"Unsupported uncond_mode={uncond_mode!r}. Only 'protein_only' is implemented."
                )
            negative_mask = batch["atom_cond_mask"] * batch["atom_is_protein_chain"]
            return {
                "mode": guidance_mode,
                "positive_batch": batch,
                "negative_batch": cls._guidance_branch_batch(batch, negative_mask),
                "positive_branch_label": "cond",
                "negative_branch_label": "protein_only",
            }

        if guidance_mode == "selectivity":
            raise NotImplementedError(
                "legacy composite selectivity branch masks are no longer supported; "
                "selectivity pair guidance uses common_token_idx inputs"
            )

        raise NotImplementedError(
            f"Unsupported guidance.mode={guidance_mode!r}. Expected 'cond_uncond' or 'selectivity'."
        )

    @staticmethod
    def _require_selectivity_pair_guidance(
        guidance_cfg: dict[str, Any],
        batch: dict[str, TensorType["b ..."]],
    ) -> dict[str, Any]:
        required = (
            "common_token_idx",
            "common_valid_mask",
            "common_designable_mask",
            "positive_batch_idx",
            "negative_batch_idx",
        )
        missing = [key for key in required if key not in guidance_cfg]
        if missing:
            raise ValueError(f"selectivity pair guidance missing keys: {missing}")
        if batch["restype"].shape[0] != 2:
            raise ValueError(
                "selectivity pair guidance requires exactly two examples per batch; "
                f"got batch size {batch['restype'].shape[0]}"
            )
        common_token_idx = guidance_cfg["common_token_idx"]
        if not torch.is_tensor(common_token_idx) or common_token_idx.shape[0] != 2:
            raise ValueError("guidance_cfg.common_token_idx must be a tensor with shape [2, n_common]")
        return {
            "common_token_idx": common_token_idx.to(device=batch["restype"].device, dtype=torch.long),
            "common_valid_mask": guidance_cfg["common_valid_mask"].to(device=batch["restype"].device).bool(),
            "common_designable_mask": guidance_cfg["common_designable_mask"].to(device=batch["restype"].device).bool(),
            "positive_batch_idx": int(guidance_cfg["positive_batch_idx"]),
            "negative_batch_idx": int(guidance_cfg["negative_batch_idx"]),
            "positive_branch_label": str(guidance_cfg.get("positive_branch_label", "positive")),
            "negative_branch_label": str(guidance_cfg.get("negative_branch_label", "negative")),
        }

    @staticmethod
    def _project_pos_restrict_aatype_to_common_axis(
        pos_restrict_aatype: tuple[torch.Tensor, torch.Tensor] | None,
        common_token_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if pos_restrict_aatype is None:
            return None
        restrict_pos_mask, allowed_aatype_mask = pos_restrict_aatype
        idx0 = common_token_idx[0]
        idx1 = common_token_idx[1]
        restrict_common = restrict_pos_mask[0, idx0].bool() | restrict_pos_mask[1, idx1].bool()
        allowed_common = allowed_aatype_mask[0, idx0] * allowed_aatype_mask[1, idx1]
        if bool((restrict_common & (allowed_common.sum(-1) == 0)).any()):
            raise ValueError("selectivity pair guidance has incompatible amino-acid restrictions on common residues")
        return restrict_common[None].float(), allowed_common[None].float()

    def _project_pos_restrict_aatype_to_encoding(
        self,
        pos_restrict_aatype: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        return potts_encoding.project_pos_restrict_aatype_to_encoding(
            pos_restrict_aatype,
            use_potts_encoding=self.use_potts_encoding,
        )

    def _sampling_ban_indices(self, omit_aas: list[str] | tuple[str, ...] | None) -> list[int]:
        ban_aas = {"X"}
        if omit_aas is not None:
            ban_aas = ban_aas | set(omit_aas)
        if not self.use_potts_encoding:
            ban_S = const.AF3_ENCODING.encode_aa_seq(ban_aas)
            ban_S = ban_S + const.AF3_ENCODING.encode(const.AF3_ENCODING.non_protein_tokens)
            return sorted(set(ban_S))

        ban_S = self.sequence_encoding.encode_aa_seq(ban_aas)
        special_tokens = (
            const.UNKNOWN_AA,
            const.POTTS_NON_PROTEIN_TOKEN,
            const.POTTS_MASK_TOKEN,
            const.POTTS_PAD_TOKEN,
        )
        ban_S = ban_S + self.sequence_encoding.encode(special_tokens)
        return sorted(set(ban_S))

    def _validate_potts_aux_alphabet(self, potts_decoder_aux: dict[str, torch.Tensor]) -> None:
        n_states = potts_decoder_aux["h"].shape[-1]
        if n_states != self.sequence_encoding.n_tokens:
            raise ValueError(
                f"Potts aux alphabet size {n_states} does not match selected sequence "
                f"encoding size {self.sequence_encoding.n_tokens}."
            )

    def _validate_restype_alphabet(self, batch: dict[str, torch.Tensor]) -> None:
        n_states = batch["restype"].shape[-1]
        if n_states != self.sequence_encoding.n_tokens:
            raise ValueError(
                f"Batch restype alphabet size {n_states} does not match selected sequence "
                f"encoding size {self.sequence_encoding.n_tokens}."
            )

    @staticmethod
    def _project_potts_branch_to_common_axis(
        *,
        potts_decoder_aux: dict[str, torch.Tensor],
        batch_idx: int,
        common_idx: torch.Tensor,
        S_full: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h = potts_decoder_aux["h"]
        J = potts_decoder_aux["J"]
        edge_idx = potts_decoder_aux["edge_idx"]
        mask_i = potts_decoder_aux["mask_i"]
        mask_ij = potts_decoder_aux["mask_ij"]
        _, N, K, C, _ = J.shape
        M = int(common_idx.numel())
        device = h.device

        common_idx = common_idx.to(device=device, dtype=torch.long)
        common_pos = torch.full((N,), -1, dtype=torch.long, device=device)
        common_pos[common_idx] = torch.arange(M, dtype=torch.long, device=device)

        h_new = h[batch_idx, common_idx].clone()
        J_new = J.new_zeros((1, M, M, C, C))
        mask_ij_new = mask_ij.new_zeros((1, M, M))

        for common_i, src_i_raw in enumerate(common_idx.tolist()):
            src_i = int(src_i_raw)
            for edge_k in range(K):
                if mask_ij[batch_idx, src_i, edge_k] <= 0:
                    continue
                src_j = int(edge_idx[batch_idx, src_i, edge_k].item())
                common_j = int(common_pos[src_j].item())
                if common_j >= 0:
                    J_new[0, common_i, common_j] += J[batch_idx, src_i, edge_k]
                    mask_ij_new[0, common_i, common_j] = 1
                else:
                    fixed_j = int(S_full[batch_idx, src_j].item())
                    h_new[common_i] += 0.5 * J[batch_idx, src_i, edge_k, :, fixed_j]

        for src_i in range(N):
            if int(common_pos[src_i].item()) >= 0:
                continue
            fixed_i = int(S_full[batch_idx, src_i].item())
            for edge_k in range(K):
                if mask_ij[batch_idx, src_i, edge_k] <= 0:
                    continue
                src_j = int(edge_idx[batch_idx, src_i, edge_k].item())
                common_j = int(common_pos[src_j].item())
                if common_j >= 0:
                    h_new[common_j] += 0.5 * J[batch_idx, src_i, edge_k, fixed_i, :]

        return {
            "h": h_new[None],
            "J": J_new,
            "edge_idx": torch.arange(M, device=device, dtype=torch.long).view(1, 1, M).expand(1, M, M),
            "mask_i": mask_i[batch_idx, common_idx][None],
            "mask_ij": mask_ij_new,
        }

    @staticmethod
    def _pair_aux_tensor(value: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if value is None:
            return None
        return value.expand(batch_size).detach().cpu()

    def _potts_sample_selectivity_pair(
        self,
        batch: dict[str, TensorType["b ..."]],
        sampling_inputs: dict[str, Any],
        guidance_cfg: dict[str, Any],
    ) -> tuple[dict[str, list[AtomArray]], dict[str, Any]]:
        selectivity_pair = self._require_selectivity_pair_guidance(guidance_cfg, batch)
        potts_sampling_cfg = sampling_inputs["potts_sampling_cfg"]
        if "tied_sampling_ids" in batch:
            raise NotImplementedError("selectivity pair guidance is not supported with tied_sampling")
        if potts_sampling_cfg["potts_proposal"] != "dlmc":
            raise NotImplementedError("selectivity pair guidance requires potts_proposal='dlmc'")

        potts_decoder_aux, batch, sampling_inputs = self.compute_potts_params(batch, sampling_inputs)
        self._validate_potts_aux_alphabet(potts_decoder_aux)
        self._validate_restype_alphabet(batch)
        common_token_idx = selectivity_pair["common_token_idx"]
        common_valid_mask = selectivity_pair["common_valid_mask"]
        common_designable_mask = selectivity_pair["common_designable_mask"]
        positive_batch_idx = selectivity_pair["positive_batch_idx"]
        negative_batch_idx = selectivity_pair["negative_batch_idx"]
        positive_common_idx = common_token_idx[positive_batch_idx]
        negative_common_idx = common_token_idx[negative_batch_idx]

        B, N, C = batch["restype"].shape
        S_full_init = batch["restype"].argmax(dim=-1)
        positive_aux = self._project_potts_branch_to_common_axis(
            potts_decoder_aux=potts_decoder_aux,
            batch_idx=positive_batch_idx,
            common_idx=positive_common_idx,
            S_full=S_full_init,
        )
        negative_aux = self._project_potts_branch_to_common_axis(
            potts_decoder_aux=potts_decoder_aux,
            batch_idx=negative_batch_idx,
            common_idx=negative_common_idx,
            S_full=S_full_init,
        )

        mask_sample_full = (
            (1 - batch["seq_cond_mask_potts"])
            * batch["token_pad_mask"]
            * batch["protein_residue_node_mask"]
        )
        mask_sample_common = (
            mask_sample_full[0, common_token_idx[0]]
            * mask_sample_full[1, common_token_idx[1]]
            * common_valid_mask.float()
            * common_designable_mask.float()
            * positive_aux["mask_i"][0]
            * negative_aux["mask_i"][0]
        )
        if not bool(mask_sample_common.bool().any()):
            raise ValueError("selectivity pair common axis has no sampleable residues")

        logits_init = torch.zeros((1, common_token_idx.shape[1], C), device=batch["restype"].device).float()
        S_common_init = S_full_init[positive_batch_idx, positive_common_idx][None]
        ban_S = self._sampling_ban_indices(sampling_inputs.get("omit_aas", None))
        pos_restrict_common = self._project_pos_restrict_aatype_to_common_axis(
            sampling_inputs.get("pos_restrict_aatype", None),
            common_token_idx,
        )
        pos_restrict_common = self._project_pos_restrict_aatype_to_encoding(pos_restrict_common)
        mask_sample_common, _, S_common_init = potts.init_sampling_masks(
            logits_init,
            mask_sample=mask_sample_common[None],
            S=S_common_init,
            ban_S=ban_S,
            pos_restrict_aatype=pos_restrict_common,
        )

        regularization = potts_sampling_cfg["regularization"]
        penalty_func = None
        if regularization == "LCP":
            C_complexity = batch["asym_id"][positive_batch_idx, positive_common_idx][None]
            C_complexity = C_complexity * positive_aux["mask_i"]
            penalty_func = lambda _S: complexity.complexity_lcp(_S, C_complexity)
        elif regularization not in {None, "none", "None"}:
            raise NotImplementedError(f"Unsupported selectivity pair regularization={regularization!r}")

        iter_items = self._build_schedule_iter_items(guidance_cfg)

        pocket_distance = float(guidance_cfg.get("pocket_distance", 10.0))
        pocket_mask, _ = self._compute_ligand_pocket_mask(batch, pocket_distance=pocket_distance)
        pocket_common = (
            pocket_mask[0, common_token_idx[0]].bool()
            | pocket_mask[1, common_token_idx[1]].bool()
        ).float()[None]
        n_pocket = pocket_common.sum(-1).clamp(min=1.0)
        n_common = common_valid_mask.float().sum().view(1).clamp(min=1.0)

        S_samples: list[torch.Tensor] = []
        per_sample_aux: list[dict[str, Any]] = []
        for sched in iter_items:
            sched_cfg = None if sched["type"] == "constant" else sched
            for _ in tqdm(
                range(sampling_inputs["num_seqs_per_pdb"]),
                desc=f"Sampling selectivity pair ({sched['label']})",
                leave=False,
            ):
                S_common, U_mixed = self.elix_mpnn.decoder_S_potts.sample(
                    positive_aux["h"],
                    positive_aux["J"],
                    positive_aux["edge_idx"],
                    positive_aux["mask_i"],
                    positive_aux["mask_ij"],
                    S=S_common_init,
                    mask_sample=mask_sample_common,
                    temperature=potts_sampling_cfg["potts_temperature"],
                    num_sweeps=potts_sampling_cfg["potts_sweeps"],
                    penalty_func=penalty_func,
                    proposal=potts_sampling_cfg["potts_proposal"],
                    dlmc_dt=float(potts_sampling_cfg.get("dlmc_dt", 0.1)),
                    rejection_step=potts_sampling_cfg.get("rejection_step", False),
                    verbose=False,
                    h_uncond=negative_aux["h"],
                    J_uncond=negative_aux["J"],
                    edge_idx_uncond=negative_aux["edge_idx"],
                    gamma=sched["gamma_max"],
                    gamma_schedule_cfg=sched_cfg,
                )
                S_full = S_full_init.clone()
                S_full[positive_batch_idx, positive_common_idx] = S_common[0]
                S_full[negative_batch_idx, negative_common_idx] = S_common[0]
                S_full = self._set_non_protein_tokens(S_full, batch)

                U_positive, _, U_positive_per_res = potts.compute_potts_energy(
                    S_common,
                    positive_aux["h"],
                    positive_aux["J"],
                    positive_aux["edge_idx"],
                    return_per_res=True,
                )
                U_negative, _, U_negative_per_res = potts.compute_potts_energy(
                    S_common,
                    negative_aux["h"],
                    negative_aux["J"],
                    negative_aux["edge_idx"],
                    return_per_res=True,
                )
                U_positive_pocket = (U_positive_per_res * pocket_common).sum(-1)
                U_negative_pocket = (U_negative_per_res * pocket_common).sum(-1)
                per_sample_aux.append({
                    "U": self._pair_aux_tensor(U_mixed, B),
                    "gamma": sched["gamma_max"],
                    "guidance_scale": sched["gamma_max"],
                    "guidance_mode": "selectivity",
                    "schedule_label": sched["label"],
                    "positive_branch_label": selectivity_pair["positive_branch_label"],
                    "negative_branch_label": selectivity_pair["negative_branch_label"],
                    "U_positive": self._pair_aux_tensor(U_positive, B),
                    "U_negative": self._pair_aux_tensor(U_negative, B),
                    "U_positive_per_res": self._pair_aux_tensor(U_positive / n_common, B),
                    "U_negative_per_res": self._pair_aux_tensor(U_negative / n_common, B),
                    "U_positive_pocket": self._pair_aux_tensor(U_positive_pocket, B),
                    "U_negative_pocket": self._pair_aux_tensor(U_negative_pocket, B),
                    "U_positive_pocket_per_res": self._pair_aux_tensor(U_positive_pocket / n_pocket, B),
                    "U_negative_pocket_per_res": self._pair_aux_tensor(U_negative_pocket / n_pocket, B),
                    "U_cond": self._pair_aux_tensor(U_positive, B),
                    "U_uncond": self._pair_aux_tensor(U_negative, B),
                    "U_cond_per_res": self._pair_aux_tensor(U_positive / n_common, B),
                    "U_uncond_per_res": self._pair_aux_tensor(U_negative / n_common, B),
                    "U_cond_pocket": self._pair_aux_tensor(U_positive_pocket, B),
                    "U_uncond_pocket": self._pair_aux_tensor(U_negative_pocket, B),
                    "U_cond_pocket_per_res": self._pair_aux_tensor(U_positive_pocket / n_pocket, B),
                    "U_uncond_pocket_per_res": self._pair_aux_tensor(U_negative_pocket / n_pocket, B),
                    "N_pocket": self._pair_aux_tensor(n_pocket, B),
                })
                S_samples.append(S_full.cpu())

        del potts_decoder_aux
        return self._postprocess_sampled_sequences(S_samples, batch, per_sample_aux=per_sample_aux)

    @staticmethod
    def _build_schedule_iter_items(guidance_cfg: dict[str, Any]) -> list[dict[str, Any]]:
        """Normalize guidance schedule config into a flat list of iteration items.

        Each item is ``{label, type, gamma_max[, tau]}``. ``schedule_list`` (when
        set) takes precedence over ``guidance_scale_list`` / the legacy
        ``gamma_list``. For constant entries ``gamma_schedule_cfg`` should be
        passed as ``None`` downstream. A scale value of ``None`` yields a
        no-guidance constant item (``gamma_max=None``).
        """
        guidance_scale_list = list(
            guidance_cfg.get("guidance_scale_list", guidance_cfg.get("gamma_list", [1.0]))
        )
        schedule_list_raw = guidance_cfg.get("schedule_list", None)
        if schedule_list_raw is not None:
            iter_items = []
            for sched_raw in schedule_list_raw:
                sched = dict(sched_raw)
                if "type" not in sched or "gamma_max" not in sched:
                    raise ValueError(
                        f"schedule_list entry must include 'type' and 'gamma_max'; got {sched!r}"
                    )
                iter_items.append({
                    "label": str(sched.get("label", f"{sched['type']}_g{float(sched['gamma_max']):.2f}")),
                    "type": str(sched["type"]),
                    "gamma_max": float(sched["gamma_max"]),
                    **({"tau": float(sched["tau"])} if "tau" in sched else {}),
                })
            return iter_items
        return [
            {
                "label": (f"gamma_{float(g):.2f}" if g is not None else "no-guidance"),
                "type": "constant",
                "gamma_max": (float(g) if g is not None else None),
            }
            for g in guidance_scale_list
        ]

    @staticmethod
    def _compute_branch_energies(
        S_sample: torch.Tensor,
        potts_decoder_aux: dict[str, torch.Tensor],
        potts_decoder_aux_negative: dict[str, torch.Tensor],
        pocket_mask: torch.Tensor,
        n_protein: torch.Tensor,
        n_pocket: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Post-hoc physical Potts energies on cond/uncond branches for one sample.

        Returns CPU tensors used as the (x, y) coordinates of the Pareto plot
        downstream. Pocket-restricted totals sum per-residue contributions over
        pocket residues only; per-residue averages clamp the denominators so
        ``N_pocket==0`` (no ligand / nothing within distance) stays finite.
        """
        U_cond_post, _, U_cond_per_res_post = potts.compute_potts_energy(
            S_sample,
            potts_decoder_aux["h"],
            potts_decoder_aux["J"],
            potts_decoder_aux["edge_idx"],
            return_per_res=True,
        )
        U_uncond_post, _, U_uncond_per_res_post = potts.compute_potts_energy(
            S_sample,
            potts_decoder_aux_negative["h"],
            potts_decoder_aux_negative["J"],
            potts_decoder_aux_negative["edge_idx"],
            return_per_res=True,
        )
        U_cond_pocket = (U_cond_per_res_post * pocket_mask).sum(-1)
        U_uncond_pocket = (U_uncond_per_res_post * pocket_mask).sum(-1)
        safe_np = n_pocket.clamp(min=1.0)
        safe_n = n_protein.clamp(min=1.0)
        return {
            "U_cond": U_cond_post.cpu(),
            "U_uncond": U_uncond_post.cpu(),
            "U_cond_per_res": (U_cond_post / safe_n).cpu(),
            "U_uncond_per_res": (U_uncond_post / safe_n).cpu(),
            "U_cond_pocket": U_cond_pocket.cpu(),
            "U_uncond_pocket": U_uncond_pocket.cpu(),
            "U_cond_pocket_per_res": (U_cond_pocket / safe_np).cpu(),
            "U_uncond_pocket_per_res": (U_uncond_pocket / safe_np).cpu(),
            "N_pocket": n_pocket.cpu(),
        }

    def potts_sample(self,
                     batch: dict[str, TensorType["b ..."]],
                     sampling_inputs: dict[str, Any],
                     potts_aux_provider: Callable | None = None,
                     potts_mixing_provider: Callable | None = None,
                     mixing_scaffold_batch: dict[str, TensorType["b ..."]] | None = None,
                     mixing_scaffold_sampling_inputs: dict[str, Any] | None = None,
                     ) -> tuple[dict[str, list[AtomArray]], dict[str, Any]]:
        """
        Potts sampling for sequence design.

        When ``potts_sampling_cfg.guidance_cfg.enabled`` is true, Potts
        parameters are computed for positive and negative branches. In
        ``cond_uncond`` mode these are the ligand-conditioned and protein-only
        branches. In ``selectivity`` mode the eval layer supplies selectivity-pair
        common-token tensors and sampling runs on the aligned common axis. The
        sampler then runs DLMC on the linearly-mixed parameters

            h_mix = scale * h_positive + (1 - scale) * h_negative
            J_mix = scale * J_positive + (1 - scale) * J_negative

        sweeping over ``guidance_scale_list`` or the legacy ``gamma_list``.
        For each sampled sequence we also record post-hoc physical Potts
        energies for both branches.

        Returns:
            output_feats: list[dict[str, TensorType["b ..."]]]: list of length (n_samples_per_pdb) of output features for each sample
            aux: dict[str, Any]: auxiliary outputs
        """
        # Preserve scaffold conditioning before the primary checkpoint applies
        # its own ``potts_only_cond`` policy or ensemble aggregation.
        mixing_source_batch = None
        if potts_mixing_provider is not None:
            mixing_source_batch = dict(mixing_scaffold_batch or batch)
            mixing_source_batch["seq_cond_mask"] = mixing_source_batch[
                "seq_cond_mask"
            ].clone()
            mixing_source_batch["seq_cond_mask_potts"] = mixing_source_batch[
                "seq_cond_mask"
            ].clone()

        # If specified, condition on sequence only in the potts model
        batch["seq_cond_mask_potts"] = batch["seq_cond_mask"].clone()
        if sampling_inputs["potts_sampling_cfg"].get("potts_only_cond", False):
            print("Conditioning on sequence only in the potts model")
            batch["seq_cond_mask"] = torch.zeros_like(batch["seq_cond_mask"])  # zero out model-level sequence conditioning mask

        # Parse guidance config (optional).
        potts_sampling_cfg = sampling_inputs["potts_sampling_cfg"]
        guidance_cfg = potts_sampling_cfg.get("guidance_cfg", None)
        use_guidance = bool(guidance_cfg) and bool(guidance_cfg.get("enabled", False))
        frustration_cfg = potts_sampling_cfg.get("frustration", {}) or {}
        use_frustration = bool(frustration_cfg.get("enabled", False))
        if use_frustration and use_guidance:
            raise NotImplementedError(
                "Pairwise frustration sampling cannot yet be combined with Potts guidance "
                "or selectivity guidance."
            )
        if potts_aux_provider is not None and use_guidance:
            raise NotImplementedError(
                "A custom Potts aux provider cannot be combined with Potts guidance."
            )
        if potts_mixing_provider is not None:
            if use_guidance:
                raise NotImplementedError(
                    "Potts-model mixing cannot be combined with guidance."
                )
            if use_frustration:
                raise NotImplementedError(
                    "Potts-model mixing cannot be combined with frustration."
                )
        if use_guidance and str(guidance_cfg.get("mode", "cond_uncond")) == "selectivity":
            return self._potts_sample_selectivity_pair(
                batch=batch,
                sampling_inputs=sampling_inputs,
                guidance_cfg=guidance_cfg,
            )
        if use_guidance:
            if "tied_sampling_ids" in batch:
                raise NotImplementedError(
                    "Potts guidance is not supported together with tied_sampling."
                )
            guidance_branches = self._resolve_potts_guidance_branches(batch, guidance_cfg)
            guidance_mode = guidance_branches["mode"]
            batch_positive = guidance_branches["positive_batch"]
            batch_negative = guidance_branches["negative_batch"]
            pocket_distance = float(guidance_cfg.get("pocket_distance", 10.0))
        else:
            guidance_branches = None
            guidance_mode = None
            batch_positive = None
            batch_negative = None
            pocket_distance = None

        # Compute cond potts parameters, or use a runtime provider that returns
        # aux tensors on this same token axis.
        if use_guidance:
            potts_decoder_aux, batch_positive, sampling_inputs = self.compute_potts_params(
                batch_positive,
                sampling_inputs,
            )
            batch["protein_residue_node_mask"] = batch_positive["protein_residue_node_mask"]
            batch["token_exists_mask"] = batch_positive["token_exists_mask"]
            if self.use_potts_encoding:
                batch["restype"] = batch_positive["restype"]
                batch["target_restype"] = batch_positive["target_restype"]
        elif potts_aux_provider is None:
            potts_decoder_aux, batch, sampling_inputs = self.compute_potts_params(batch, sampling_inputs)
        else:
            potts_decoder_aux, batch, sampling_inputs = potts_aux_provider(
                batch=batch,
                sampling_inputs=sampling_inputs,
            )
        self._validate_potts_aux_alphabet(potts_decoder_aux)
        self._validate_restype_alphabet(batch)

        local_mixture = None
        if potts_mixing_provider is not None:
            local_mixture = potts_mixing_provider(
                primary_batch=batch,
                scaffold_batch=mixing_source_batch,
                primary_potts_aux=potts_decoder_aux,
                scaffold_sampling_inputs=mixing_scaffold_sampling_inputs,
            )

        # Apply the sampling-only transform after any tied/ensemble aggregation
        # has produced the final sparse Potts coupling tensor. The Potts head,
        # training losses, and standalone scoring path continue to use raw J.
        if use_frustration:
            potts_decoder_aux["J"] = frustration.mix_pairwise_couplings(
                potts_decoder_aux["J"],
                potts_decoder_aux["coupling_mask"],
                alpha=float(frustration_cfg["alpha"]),
                beta=float(frustration_cfg["beta"]),
            )

        # Compute negative branch Potts parameters if guidance is on.
        potts_decoder_aux_negative = None
        pocket_mask = None
        n_protein = None
        n_pocket = None
        if use_guidance:
            potts_decoder_aux_negative, _, _ = self.compute_potts_params(batch_negative, sampling_inputs)
            # Pocket mask is structure-only; compute once per batch, reuse for
            # every (gamma, sample). N_pocket==0 is possible (no ligand atoms
            # or nothing within pocket_distance) — handled via clamp below.
            pocket_mask, n_protein = self._compute_ligand_pocket_mask(
                batch, pocket_distance=pocket_distance,
            )  # [B, N], [B]
            n_pocket = pocket_mask.sum(-1)  # [B]

        # Set up Potts sampling
        regularization = potts_sampling_cfg["regularization"]
        potts_sweeps = potts_sampling_cfg["potts_sweeps"]
        potts_proposal = potts_sampling_cfg["potts_proposal"]
        potts_temperature = potts_sampling_cfg["potts_temperature"]
        dlmc_dt = float(potts_sampling_cfg.get("dlmc_dt", 0.1))
        temperature_schedule_cfg = potts_sampling_cfg.get(
            "temperature_schedule", None
        )
        use_temperature_schedule = bool(temperature_schedule_cfg) and bool(
            temperature_schedule_cfg.get("enabled", False)
        )
        rejection_step = potts_sampling_cfg.get("rejection_step", potts_proposal == "chromatic")

        B, N, C = batch["restype"].shape
        logits_init = torch.zeros((B, N, C), device=batch["restype"].device).float()

        # Handle banned amino acids and aatype restrictions
        ban_S = self._sampling_ban_indices(sampling_inputs.get("omit_aas", None))

        # Initialize random sequence and sampling masks
        mask_sample = (
            (1 - batch["seq_cond_mask_potts"])
            * batch["token_pad_mask"]
            * batch["protein_residue_node_mask"]
        )  # 1 where we can sample, 0 where we can't
        pos_restrict_aatype = self._project_pos_restrict_aatype_to_encoding(
            sampling_inputs.get("pos_restrict_aatype", None)
        )

        # ``potts_only_cond`` intentionally clears the model-facing sequence
        # conditioning mask before the forward pass.  With compact Potts
        # encoding that turns every standard-AA model input into ``<M>``.
        # Fixed sampler positions must still start from their unmasked target
        # identities, otherwise the fixed state (and its pairwise energies)
        # incorrectly remains the mask token.
        sampling_sequence = batch.get("target_restype")
        if sampling_sequence is None:
            sampling_sequence = batch["restype"].argmax(dim=-1)
        if sampling_sequence.shape != batch["restype"].shape[:-1]:
            raise ValueError(
                "Potts sampling target/restype shape mismatch: "
                f"target={tuple(sampling_sequence.shape)}, "
                f"restype={tuple(batch['restype'].shape)}"
            )

        mask_sample, _, S_init = potts.init_sampling_masks(
            logits_init,
            mask_sample=mask_sample,
            S=sampling_sequence,
            ban_S=ban_S,
            pos_restrict_aatype=pos_restrict_aatype,
        )

        # Complexity regularization
        penalty_func = None
        mask_ij_coloring = None
        edge_idx_coloring = None
        if regularization == "LCP":
            C_complexity = batch["asym_id"] - torch.min(batch["asym_id"]) + 1  # renumber asym_id to have min value of 1
            C_complexity = C_complexity * batch["protein_residue_node_mask"]
            #! fixed, 251110
            # mask out i) non-protein chains, ii) pad tokens, iii) tokens that don't exist in the graph
            # complexity is only calculated for the residues where C_complexity > 0
            penalty_func = lambda _S: complexity.complexity_lcp(_S, C_complexity)

        precomputed_temperature_schedule_trace = None
        endpoint_mode = "fixed"
        schedule_name = None
        if use_temperature_schedule:
            endpoint_mode = str(temperature_schedule_cfg.get("endpoint_mode", "fixed"))
            schedule_name = str(temperature_schedule_cfg["name"])
        if schedule_name == "heat_capacity":
            if endpoint_mode != "fixed":
                raise ValueError(
                    "heat_capacity requires fixed start/end temperature endpoints"
                )
            if potts_proposal != "dlmc" or rejection_step:
                raise NotImplementedError(
                    "heat_capacity requires proposal='dlmc' without a rejection step"
                )
            if (
                local_mixture is not None
                and local_mixture.mixing_scheme != "mean_energy"
            ):
                raise NotImplementedError(
                    "heat_capacity with local transition mixing is supported only "
                    "for mixing_scheme='mean_energy'"
                )
            calibration_cfg = temperature_schedule_cfg.get(
                "heat_capacity_calibration", {}
            )
            num_probe_sequences = int(calibration_cfg.get("num_sequences", 16))
            anneal_temperature = float(
                calibration_cfg.get("anneal_temperature", 1.0)
            )
            anneal_steps = int(calibration_cfg.get("anneal_steps", 50))
            calibration_seed = int(calibration_cfg.get("seed", 0))
            fork_devices = []
            if logits_init.is_cuda:
                device_index = logits_init.device.index
                fork_devices = [
                    torch.cuda.current_device() if device_index is None else device_index
                ]
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(calibration_seed)
                if logits_init.is_cuda:
                    with torch.cuda.device(logits_init.device):
                        torch.cuda.manual_seed(calibration_seed)
                probe_sequences = _sample_initial_sequence_probes(
                    logits_init=logits_init,
                    finalized_mask_sample=mask_sample,
                    sampling_sequence=sampling_sequence,
                    first_sequence=S_init,
                    num_sequences=num_probe_sequences,
                )
                annealed_probe_sequences = []
                for probe_sequence in probe_sequences:
                    probe_sample_kwargs = {}
                    if local_mixture is not None:
                        probe_sample_kwargs["local_mixture"] = local_mixture
                    annealed_probe, _ = potts.sample_potts(
                        potts_decoder_aux["h"],
                        potts_decoder_aux["J"],
                        potts_decoder_aux["edge_idx"],
                        potts_decoder_aux["mask_i"],
                        potts_decoder_aux["mask_ij"],
                        S=probe_sequence,
                        mask_sample=mask_sample,
                        num_sweeps=anneal_steps,
                        temperature=anneal_temperature,
                        temperature_init=anneal_temperature,
                        penalty_func=penalty_func,
                        proposal="dlmc",
                        dlmc_dt=dlmc_dt,
                        rejection_step=False,
                        verbose=False,
                        **probe_sample_kwargs,
                    )
                    annealed_probe_sequences.append(annealed_probe)
                annealed_probe_sequences = torch.stack(
                    annealed_probe_sequences, dim=0
                )

            annealed_local_energy = []
            for annealed_probe in annealed_probe_sequences:
                if local_mixture is None:
                    _, local_energy = potts._compute_dlmc_local_energy(
                        annealed_probe,
                        potts_decoder_aux["h"],
                        potts_decoder_aux["J"],
                        potts_decoder_aux["edge_idx"],
                        penalty_func=penalty_func,
                    )
                else:
                    _, local_energy = potts._compute_mean_energy_local_energy(
                        annealed_probe,
                        potts_decoder_aux["h"],
                        potts_decoder_aux["J"],
                        potts_decoder_aux["edge_idx"],
                        local_mixture,
                        penalty_func=penalty_func,
                    )
                annealed_local_energy.append(local_energy.detach())
            annealed_local_energy = torch.stack(annealed_local_energy, dim=0)
            scheduled_site_mask = (
                (mask_sample.sum(dim=-1) > 1) & (potts_decoder_aux["mask_i"] > 0)
            )
            precomputed_temperature_schedule_trace = (
                build_heat_capacity_schedule_trace(
                    annealed_local_energy=annealed_local_energy,
                    legal_state_mask=mask_sample > 0,
                    site_mask=scheduled_site_mask,
                    initial_probe_sequences=probe_sequences.detach(),
                    annealed_probe_sequences=annealed_probe_sequences.detach(),
                    start_temperature=float(
                        temperature_schedule_cfg.get("start_temperature", 3.0)
                    ),
                    end_temperature=float(potts_temperature),
                    total_steps=int(potts_sweeps),
                    end_hold_steps=int(
                        temperature_schedule_cfg.get("end_hold_steps", 50)
                    ),
                    curve_points=int(calibration_cfg.get("curve_points", 200)),
                    heat_capacity_floor=float(
                        calibration_cfg.get("heat_capacity_floor", 0.1)
                    ),
                    bisection_iterations=int(
                        calibration_cfg.get("alpha_bisection_iterations", 64)
                    ),
                    calibration_anneal_temperature=anneal_temperature,
                    calibration_anneal_steps=anneal_steps,
                )
            )
        elif endpoint_mode != "fixed":
            if endpoint_mode != "initial_random_mean_total_U_per_residue":
                raise ValueError(f"Unknown temperature endpoint_mode={endpoint_mode!r}")
            calibration_cfg = temperature_schedule_cfg.get("energy_calibration", {})
            num_probe_sequences = int(calibration_cfg.get("num_sequences", 16))
            probe_sequences = _sample_initial_sequence_probes(
                logits_init=logits_init,
                finalized_mask_sample=mask_sample,
                sampling_sequence=sampling_sequence,
                first_sequence=S_init,
                num_sequences=num_probe_sequences,
            )
            with torch.no_grad():
                probe_total_energies = torch.stack(
                    [
                        potts.compute_potts_energy(
                            probe_sequence,
                            potts_decoder_aux["h"],
                            potts_decoder_aux["J"],
                            potts_decoder_aux["edge_idx"],
                        )[0]
                        for probe_sequence in probe_sequences
                    ],
                    dim=0,
                )
            _, frozen_local_energy = potts._compute_dlmc_local_energy(
                S_init,
                potts_decoder_aux["h"],
                potts_decoder_aux["J"],
                potts_decoder_aux["edge_idx"],
                penalty_func=penalty_func,
            )
            scheduled_site_mask = (
                (mask_sample.sum(dim=-1) > 1) & (potts_decoder_aux["mask_i"] > 0)
            )
            precomputed_temperature_schedule_trace = (
                build_energy_density_schedule_trace(
                    str(temperature_schedule_cfg["name"]),
                    probe_total_energies=probe_total_energies.detach(),
                    probe_sequences=probe_sequences.detach(),
                    lengths=potts_decoder_aux["mask_i"].sum(dim=-1),
                    local_energy=frozen_local_energy.detach(),
                    legal_state_mask=mask_sample > 0,
                    site_mask=scheduled_site_mask,
                    start_reduced_energy=float(
                        calibration_cfg.get("start_reduced_energy", 0.01)
                    ),
                    end_reduced_energy=float(
                        calibration_cfg.get("end_reduced_energy", 10.0)
                    ),
                    total_steps=int(potts_sweeps),
                    end_hold_steps=int(
                        temperature_schedule_cfg.get("end_hold_steps", 50)
                    ),
                    variance_floor=float(
                        temperature_schedule_cfg.get("variance_floor", 1e-12)
                    ),
                )
            )

        S = []  # keep track of sequences for each sample
        per_sample_aux: list[dict[str, Any]] = []

        # Energy aux keys filled per sample under guidance; set to None
        # otherwise. Per-residue / pocket entries are shape [B] when filled.
        energy_keys = (
            "U_cond",
            "U_uncond",
            "U_cond_per_res",
            "U_uncond_per_res",
            "U_cond_pocket",
            "U_uncond_pocket",
            "U_cond_pocket_per_res",
            "U_uncond_pocket_per_res",
            "N_pocket",
        )

        num_seqs_per_pdb = sampling_inputs["num_seqs_per_pdb"]

        # Outer iteration: schedule_list (when set) takes precedence over
        # scale lists. Each item is normalized to {label, type, gamma_max}; for
        # constant entries we pass gamma_schedule_cfg=None so the legacy
        # constant-γ path runs unchanged.
        if use_guidance:
            iter_items = self._build_schedule_iter_items(guidance_cfg)
        else:
            iter_items = [{"label": "no-guidance", "type": "constant", "gamma_max": None}]

        # Design sequences: outer loop over schedule (or constant gamma),
        # inner loop over samples-per-pdb.
        for sched in iter_items:
            sched_label = sched["label"]
            sched_gamma_max = sched["gamma_max"]
            sched_cfg = None if sched["type"] == "constant" else sched
            desc = f"schedule={sched_label}"
            if precomputed_temperature_schedule_trace is None:
                shared_temperature_schedule = None
            else:
                shared_temperature_schedule = (
                    precomputed_temperature_schedule_trace.temperatures.detach()
                )
            for sequence_index in tqdm(
                range(num_seqs_per_pdb),
                desc=f"Sampling sequences ({desc})",
                leave=False,
            ):

                sample_kwargs = dict(
                    S=S_init,
                    mask_sample=mask_sample,
                    temperature=potts_temperature,
                    num_sweeps=potts_sweeps,
                    penalty_func=penalty_func,
                    proposal=potts_proposal,
                    dlmc_dt=dlmc_dt,
                    rejection_step=rejection_step,
                    verbose=False,
                    edge_idx_coloring=edge_idx_coloring,
                    mask_ij_coloring=mask_ij_coloring,
                )
                if use_temperature_schedule:
                    sample_kwargs.update(
                        temperature_init=float(
                            temperature_schedule_cfg.get("start_temperature", 1.0)
                        ),
                        temperature_schedule_cfg=temperature_schedule_cfg,
                        temperature_schedule_override=shared_temperature_schedule,
                        return_temperature_schedule=(
                            shared_temperature_schedule is None
                        ),
                    )
                if use_guidance:
                    sample_kwargs.update(
                        h_uncond=potts_decoder_aux_negative["h"],
                        J_uncond=potts_decoder_aux_negative["J"],
                        edge_idx_uncond=potts_decoder_aux_negative["edge_idx"],
                        gamma=sched_gamma_max,
                        gamma_schedule_cfg=sched_cfg,
                    )
                if local_mixture is not None:
                    sample_kwargs["local_mixture"] = local_mixture
                    if local_mixture.mixing_scheme == "pcebm":
                        sample_kwargs["return_mixing_diagnostics"] = True
                sample_output = self.elix_mpnn.decoder_S_potts.sample(
                    potts_decoder_aux["h"],
                    potts_decoder_aux["J"],
                    potts_decoder_aux["edge_idx"],
                    potts_decoder_aux["mask_i"],
                    potts_decoder_aux["mask_ij"],
                    **sample_kwargs,
                )
                temperature_schedule_trace = (
                    precomputed_temperature_schedule_trace
                    if sequence_index == 0
                    else None
                )
                mixing_diagnostics = None
                if use_temperature_schedule and shared_temperature_schedule is None:
                    (
                        S_sample,
                        U_sample,
                        temperature_schedule_trace,
                    ) = sample_output
                    shared_temperature_schedule = (
                        temperature_schedule_trace.temperatures.detach()
                    )
                elif (
                    local_mixture is not None
                    and local_mixture.mixing_scheme == "pcebm"
                ):
                    S_sample, U_sample, mixing_diagnostics = sample_output
                else:
                    S_sample, U_sample = sample_output

                # Set all tokens that don't exist in the graph to unknown
                S_sample = self._set_non_protein_tokens(S_sample, batch)

                if use_guidance:
                    energies = self._compute_branch_energies(
                        S_sample,
                        potts_decoder_aux,
                        potts_decoder_aux_negative,
                        pocket_mask,
                        n_protein,
                        n_pocket,
                    )
                else:
                    energies = {key: None for key in energy_keys}

                sample_entry = {
                    "U": U_sample.cpu(),
                    "gamma": sched_gamma_max,
                    "guidance_scale": sched_gamma_max,
                    "guidance_mode": guidance_mode if use_guidance else None,
                    "schedule_label": sched_label if use_guidance else None,
                    "positive_branch_label": (
                        guidance_branches["positive_branch_label"] if use_guidance else None
                    ),
                    "negative_branch_label": (
                        guidance_branches["negative_branch_label"] if use_guidance else None
                    ),
                    "U_positive": energies["U_cond"],
                    "U_negative": energies["U_uncond"],
                    "U_positive_per_res": energies["U_cond_per_res"],
                    "U_negative_per_res": energies["U_uncond_per_res"],
                    "U_positive_pocket": energies["U_cond_pocket"],
                    "U_negative_pocket": energies["U_uncond_pocket"],
                    "U_positive_pocket_per_res": energies["U_cond_pocket_per_res"],
                    "U_negative_pocket_per_res": energies["U_uncond_pocket_per_res"],
                    **{key: energies[key] for key in energy_keys},
                }
                if use_temperature_schedule:
                    sample_entry["temperature_schedule_trace"] = (
                        temperature_schedule_trace
                    )
                if local_mixture is not None:
                    sample_entry["transition_mixing_scheme"] = (
                        local_mixture.mixing_scheme
                    )
                    if local_mixture.alpha is not None:
                        alpha = local_mixture.alpha[..., 0]
                        alpha_mask = batch["protein_residue_node_mask"].bool()
                        alpha_count = alpha_mask.sum(dim=-1).clamp(min=1)
                        sample_entry.update(
                            transition_alpha_min=torch.where(
                                alpha_mask,
                                alpha,
                                torch.full_like(alpha, torch.inf),
                            ).min(dim=-1).values.detach().cpu(),
                            transition_alpha_max=torch.where(
                                alpha_mask,
                                alpha,
                                torch.full_like(alpha, -torch.inf),
                            ).max(dim=-1).values.detach().cpu(),
                            transition_alpha_mean=(
                                (alpha * alpha_mask).sum(dim=-1) / alpha_count
                            ).detach().cpu(),
                        )
                    if mixing_diagnostics is not None:
                        sample_entry.update(
                            transition_lambda_pocket_final=(
                                mixing_diagnostics.lambda_pocket_final.cpu()
                            ),
                            transition_lambda_pocket_min=(
                                mixing_diagnostics.lambda_pocket_min.cpu()
                            ),
                            transition_lambda_pocket_max=(
                                mixing_diagnostics.lambda_pocket_max.cpu()
                            ),
                            transition_lambda_pocket_mean=(
                                mixing_diagnostics.lambda_pocket_mean.cpu()
                            ),
                        )
                per_sample_aux.append(sample_entry)
                S.append(S_sample.cpu())

        # Free GPU potts parameters before postprocessing
        del potts_decoder_aux
        if potts_decoder_aux_negative is not None:
            del potts_decoder_aux_negative

        return self._postprocess_sampled_sequences(S, batch, per_sample_aux=per_sample_aux)


    @staticmethod
    def _compute_ligand_pocket_mask(
        batch: dict[str, torch.Tensor],
        pocket_distance: float = 10.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-batch ligand pocket mask over token positions.

        A protein residue is "in the pocket" iff its C-alpha coordinate is
        within ``pocket_distance`` Å of any resolved ligand atom in the same
        batch item. The C-alpha is read from ``batch['noised_ca_coords']``,
        which at sequence-design time has zero structure noise.

        Returns:
            pocket_mask: float tensor ``[B, N_tokens]``, 1 at pocket residues,
                0 elsewhere.
            n_protein: float tensor ``[B]``, count of valid protein residues
                per batch item (used for whole-protein per-residue averaging).
        """
        ca = batch["noised_ca_coords"]                        # [B, N, 3]
        coords = batch["coords"]                              # [B, N_atoms, 3]
        lig_atom_mask = (
            batch["atom_is_small_molecule_chain"].bool()
            & batch["atom_resolved_mask"].bool()
            & batch["atom_pad_mask"].bool()
        )                                                     # [B, N_atoms]

        # Entries with zero C-alpha coords flag non-standard / unresolved tokens.
        # Combine with the existing protein-residue node mask to be safe.
        ca_valid = (
            (ca.norm(dim=-1) > 1e-6)
            & batch["protein_residue_node_mask"].bool()
        )                                                     # [B, N]
        n_protein = ca_valid.sum(-1).float()                  # [B]

        B, N, _ = ca.shape
        pocket = torch.zeros((B, N), device=ca.device, dtype=torch.float32)
        d2_threshold = float(pocket_distance) ** 2
        for b in range(B):
            if not lig_atom_mask[b].any():
                continue
            lig_b = coords[b][lig_atom_mask[b]]               # [L_b, 3]
            ca_b = ca[b]                                      # [N, 3]
            d2 = ((ca_b[:, None, :] - lig_b[None, :, :]) ** 2).sum(-1)
            min_d2 = d2.min(dim=1).values                     # [N]
            pocket[b] = (min_d2 < d2_threshold).float()
        pocket = pocket * ca_valid.float()
        return pocket, n_protein

    def _set_non_protein_tokens(self,
                                S: TensorType["b n", int],
                                batch: dict[str, TensorType["b ..."]],
                                ) -> TensorType["b n", int]:
        """Set non-protein-residue-node positions to appropriate unknown tokens."""
        non_protein = ~batch["protein_residue_node_mask"].bool()
        if self.use_potts_encoding:
            token_pad_mask = batch["token_pad_mask"].bool()
            protein_context = potts_encoding.protein_context_token_mask(batch)
            S = torch.where(
                non_protein & token_pad_mask & protein_context,
                S.new_full((), self.sequence_encoding.token_to_idx[const.UNKNOWN_AA]),
                S,
            )
            S = torch.where(
                non_protein & token_pad_mask & ~protein_context,
                S.new_full((), self.sequence_encoding.token_to_idx[const.POTTS_NON_PROTEIN_TOKEN]),
                S,
            )
            S = torch.where(
                ~token_pad_mask,
                S.new_full((), self.sequence_encoding.token_to_idx[const.POTTS_PAD_TOKEN]),
                S,
            )
            return S

        S = torch.where(non_protein & (batch["is_protein"] | batch["is_ligand"]),
                        const.AF3_ENCODING.token_to_idx[const.UNKNOWN_AA], S)
        S = torch.where(non_protein & batch["is_rna"],
                        const.AF3_ENCODING.token_to_idx[const.UNKNOWN_RNA], S)
        S = torch.where(non_protein & batch["is_dna"],
                        const.AF3_ENCODING.token_to_idx[const.UNKNOWN_DNA], S)
        return S


    def _postprocess_sampled_sequences(
        self,
        S_list: list[TensorType["b n", int]],
        batch: dict[str, TensorType["b ..."]],
        per_sample_aux: list[dict] | None = None,
    ) -> tuple[dict[str, list[AtomArray]], dict[str, Any]]:
        """Thread sampled integer sequences onto atom arrays.

        Args:
            S_list: list of [B, N] integer tensors, one per sample.
            batch: the batch dict (will be moved to CPU).
            per_sample_aux: optional list of dicts with per-sample auxiliary data (e.g. {"U": tensor}).
        """
        batch = to(batch, device="cpu")

        id_to_atom_arrays = defaultdict(list)
        id_to_aux = defaultdict(list)
        for si in range(len(S_list)):
            atom_arrays = copy.deepcopy(batch["atom_array"])

            for bi in range(len(atom_arrays)):
                token_pad_mask = batch["token_pad_mask"][bi].bool()
                atom_pad_mask = batch["atom_pad_mask"][bi].bool()

                new_restype = S_list[si][bi][token_pad_mask]
                new_coords = batch["coords"][bi][atom_pad_mask]

                example_id = batch["example_id"][bi]
                atom_array = atom_arrays[bi]
                # ``seq_cond_mask`` may have been cleared for model-only
                # conditioning.  Output threading must preserve the original
                # fixed-position contract used by the Potts sampler.
                seq_cond_mask = batch.get(
                    "seq_cond_mask_potts",
                    batch["seq_cond_mask"],
                )[bi][token_pad_mask]
                atom_cond_mask = batch["atom_cond_mask"][bi][atom_pad_mask]
                atom_resolved_mask = batch["atom_resolved_mask"][bi][atom_pad_mask]

                # Update only sampled standard protein residues.
                protein_residue_node_mask = batch["protein_residue_node_mask"][bi][token_pad_mask].bool()
                update_seq_mask = (
                    ~seq_cond_mask.bool() & protein_residue_node_mask
                ).numpy().astype(bool)
                atomwise_update_seq_mask = spread_token_wise(atom_array, update_seq_mask)
                decoded_resnames = self.sequence_encoding.idx_to_token[new_restype.numpy()]
                atomwise_resnames = spread_token_wise(atom_array, decoded_resnames)
                atomwise_resnames = np.where(atomwise_update_seq_mask,
                                             atomwise_resnames,
                                             atom_array.get_annotation("res_name"))
                atom_array.set_annotation("res_name", atomwise_resnames)

                # Update coords.
                update_coords_mask = (atom_cond_mask * atom_resolved_mask).numpy().astype(bool)
                atom_array.coord = np.where(update_coords_mask[..., None],
                                            new_coords.numpy(),
                                            np.nan)

                id_to_atom_arrays[example_id].append(atom_array)

                # Auxiliary outputs.
                sample_aux = {"S": new_restype.cpu()}

                def _extract_scalar(entry, batch_idx):
                    if entry is None:
                        return float("nan")
                    if torch.is_tensor(entry):
                        return entry[batch_idx].cpu().item()
                    return float("nan")

                if per_sample_aux is not None:
                    aux_si = per_sample_aux[si]
                    sample_aux["U"] = _extract_scalar(aux_si.get("U"), bi)
                    if "gamma" in aux_si:
                        sample_aux["gamma"] = aux_si["gamma"]  # scalar or None
                    if "guidance_scale" in aux_si:
                        sample_aux["guidance_scale"] = aux_si["guidance_scale"]  # scalar or None
                    for key in (
                        "guidance_mode",
                        "positive_branch_label",
                        "negative_branch_label",
                        "transition_mixing_scheme",
                    ):
                        if key in aux_si:
                            sample_aux[key] = aux_si[key]
                    if "schedule_label" in aux_si:
                        sample_aux["schedule_label"] = aux_si["schedule_label"]  # str or None
                    for key in (
                        "U_positive",
                        "U_negative",
                        "U_positive_per_res",
                        "U_negative_per_res",
                        "U_positive_pocket",
                        "U_negative_pocket",
                        "U_positive_pocket_per_res",
                        "U_negative_pocket_per_res",
                        "U_cond",
                        "U_uncond",
                        "U_cond_per_res",
                        "U_uncond_per_res",
                        "U_cond_pocket",
                        "U_uncond_pocket",
                        "U_cond_pocket_per_res",
                        "U_uncond_pocket_per_res",
                        "N_pocket",
                        "transition_alpha_min",
                        "transition_alpha_max",
                        "transition_alpha_mean",
                        "transition_lambda_pocket_final",
                        "transition_lambda_pocket_min",
                        "transition_lambda_pocket_max",
                        "transition_lambda_pocket_mean",
                    ):
                        if key in aux_si:
                            sample_aux[key] = _extract_scalar(aux_si.get(key), bi)
                    temperature_trace = aux_si.get("temperature_schedule_trace")
                    if temperature_trace is not None:
                        sample_aux["temperature_schedule_trace"] = (
                            temperature_trace.batch_dict(bi)
                        )
                else:
                    sample_aux["U"] = float("nan")
                id_to_aux[example_id].append(sample_aux)

        return id_to_atom_arrays, id_to_aux


    def compute_potts_params(self, batch: dict[str, TensorType["b ..."]],
                             sampling_inputs: dict[str, Any]) -> tuple[dict[str, TensorType["b ..."]], dict[str, TensorType["b ..."]], dict[str, Any]]:
        """
        Run model and collect potts parameters over a batch of samples.

        If "tied_sampling_ids" is in batch, we will aggregate potts parameters across tied groups and slice batch to representative elements.

        Returns:
            potts_decoder_aux: dict[str, TensorType["b ..."]]: potts parameters
            batch: dict[str, TensorType["b ..."]]: batch with token_exists_mask added
            sampling_inputs: dict[str, Any]: sampling inputs with pos_restrict_aatype sliced to representative elements
        """
        subbatch_size = sampling_inputs["batch_size"]
        B = batch["restype"].shape[0]

        # Run model and collect potts parameters
        potts_decoder_aux = {}  # potts parameters
        token_exists_mask = []
        protein_residue_node_mask = []  # keep track of the residues that exist in the graph
        restype = []
        target_restype = []
        for bi in tqdm(range(0, B, subbatch_size), desc="Computing potts parameters", leave=False):
            subbatch = slice_feats(batch, slice(bi, bi + subbatch_size))

            _, aux_preds_i = self(subbatch, is_sampling=True, sampling_inputs=sampling_inputs)

            for k, v in aux_preds_i["potts_decoder_aux"].items():
                potts_decoder_aux.setdefault(k, []).append(v)
            protein_residue_node_mask.append(aux_preds_i["protein_residue_node_mask"])
            token_exists_mask.append(aux_preds_i["token_exists_mask"])
            if self.use_potts_encoding:
                restype.append(aux_preds_i["restype"])
                target_restype.append(aux_preds_i["target_restype"])
            del aux_preds_i  # free seq_logits, h_V, h_ESV etc.
        potts_decoder_aux = {k: torch.cat(v, dim=0) for k, v in potts_decoder_aux.items()}

        token_exists_mask = torch.cat(token_exists_mask, dim=0)
        protein_residue_node_mask = torch.cat(protein_residue_node_mask, dim=0)
        batch["protein_residue_node_mask"] = protein_residue_node_mask  # store in batch for downstream use
        batch["token_exists_mask"] = token_exists_mask  # store in batch for downstream use
        if self.use_potts_encoding:
            batch["restype"] = torch.cat(restype, dim=0)
            batch["target_restype"] = torch.cat(target_restype, dim=0)

        # Handle tied sampling
        if "tied_sampling_ids" in batch:
            tied_sampling_inputs = _construct_tied_sampling_inputs(batch)

            # slice to representative elements
            unique_rep_idxs = tied_sampling_inputs["unique_rep_idx"].tolist()
            batch = slice_feats(batch, unique_rep_idxs)  # get representative batch elements

            if sampling_inputs.get("pos_restrict_aatype", None) is not None:
                sampling_inputs["pos_restrict_aatype"] = [x[unique_rep_idxs] for x in sampling_inputs["pos_restrict_aatype"]]

            # aggregate potts parameters across tied groups
            potts_decoder_aux = _aggregate_potts_params(
                potts_decoder_aux,
                tied_sampling_inputs,
                reduce=tied_sampling_inputs["reduce"],
            )

        return potts_decoder_aux, batch, sampling_inputs


def _aggregate_potts_params(potts_decoder_aux: dict[str, TensorType["b ..."]],
                            tied_sampling_inputs: dict[str, Any],
                            use_mean: bool | None = True,
                            reduce: str | None = None,
                            ) -> dict[str, TensorType["b ..."]]:
    """
    Aggregate potts parameters across tied groups.

    If reduce is "mean", we take the mean of the potts parameters across the tied
    groups (equivalent to geometric mean in probability space). If reduce is
    "sqrt", summed parameters are divided by sqrt(group size). If reduce is
    "weighted_mean", per-member weights from tied_sampling_inputs["weights"] are
    used and normalized within each tied group. The legacy use_mean argument is
    preserved for callers/tests that use sum vs mean.
    """
    if reduce is None:
        reduce = "mean" if use_mean else "sum"
    if reduce not in {"sum", "mean", "sqrt", "weighted_mean"}:
        raise ValueError(f"Unknown Potts aggregation reduce: {reduce!r}")

    h = potts_decoder_aux["h"]
    J = potts_decoder_aux["J"]
    edge_idx = potts_decoder_aux["edge_idx"]
    mask_i = potts_decoder_aux["mask_i"]
    mask_ij = potts_decoder_aux["mask_ij"]
    coupling_mask = potts_decoder_aux["coupling_mask"]
    inverse, unique_ids = tied_sampling_inputs["inverse"], tied_sampling_inputs["unique_ids"]
    B = h.shape[0]

    weights = None
    weight_sums = None
    raw_weights = tied_sampling_inputs.get("weights", None)
    if raw_weights is not None and reduce != "weighted_mean":
        raise ValueError(
            "tied sampling weights require Potts aggregation reduce='weighted_mean'"
        )
    if reduce == "weighted_mean":
        if raw_weights is None:
            raise ValueError("Potts aggregation reduce='weighted_mean' requires tied sampling weights")
        weights = raw_weights.to(device=h.device, dtype=h.dtype)
        if weights.ndim != 1 or weights.shape[0] != B:
            raise ValueError(
                "tied sampling weights must be a 1D tensor with one value per batch item; "
                f"got shape={tuple(weights.shape)}, batch_size={B}"
            )
        if torch.any(weights < 0):
            raise ValueError("tied sampling weights must be non-negative")
        weight_sums = h.new_zeros(unique_ids.shape[0]).index_add(0, inverse, weights)
        if torch.any(weight_sums <= 0):
            raise ValueError("tied sampling weights must sum to a positive value per group")

    # handle 1D features
    counts = torch.bincount(inverse)
    h_source = h if weights is None else h * weights.view(B, *([1] * (h.ndim - 1)))
    h_new = h.new_zeros(unique_ids.shape[0], *h.shape[1:]).index_add(0, inverse, h_source)
    node_counts = mask_i.new_zeros(unique_ids.shape[0], *mask_i.shape[1:]).index_add(0, inverse, mask_i)
    mask_i_new = (node_counts == counts.view(-1, 1)).float()  # node i is unmasked only if node i is present across all inputs in the tied group

    # handle 2D features
    n_grp = unique_ids.shape[0]
    _, N, K = edge_idx.shape
    C = J.shape[-1]
    edge_counts = mask_ij.new_zeros(n_grp, N, N)
    coupling_counts = coupling_mask.new_zeros(n_grp, N, N)
    J_new = J.new_zeros(n_grp, N, N, C, C)
    for bi in range(B):
        g = inverse[bi]
        J_source = J[bi] if weights is None else J[bi] * weights[bi].to(dtype=J.dtype)

        edge_indices_flat = (edge_idx[bi] + torch.arange(N, device=edge_idx.device)[:, None] * N).reshape(-1)
        edge_counts[g].view(-1).index_add_(0, edge_indices_flat, mask_ij[bi].view(-1))  # count number of edges between each pair of nodes
        coupling_support = coupling_mask[bi]
        if weights is not None:
            coupling_support = coupling_support * (weights[bi] > 0).to(
                dtype=coupling_support.dtype
            )
        coupling_counts[g].view(-1).index_add_(
            0,
            edge_indices_flat,
            coupling_support.reshape(-1),
        )
        J_new[g].view(-1, C, C).index_add_(0, edge_indices_flat, J_source.view(-1, C, C))  # add in the pairwise interactions for this graph

    mask_ij_new = (edge_counts > 0) * (mask_i_new[:, :, None] * mask_i_new[:, None, :])  # edge i,j is present only if both nodes are present and there exists some edge between them
    coupling_mask_new = (coupling_counts > 0) * (
        mask_i_new[:, :, None] * mask_i_new[:, None, :]
    )
    edge_idx_new = torch.arange(N, device=edge_idx.device).expand(1, 1, -1).repeat(n_grp, N, 1)  # new edge indices are given in the full NxN grid

    if reduce == "mean":
        J_new = J_new / counts.view(-1, 1, 1, 1, 1)
        h_new = h_new / counts.view(-1, 1, 1)
    elif reduce == "sqrt":
        scale = torch.sqrt(counts.to(dtype=J_new.dtype))
        J_new = J_new / scale.view(-1, 1, 1, 1, 1)
        h_new = h_new / scale.to(dtype=h_new.dtype).view(-1, 1, 1)
    elif reduce == "weighted_mean":
        J_new = J_new / weight_sums.to(dtype=J_new.dtype).view(-1, 1, 1, 1, 1)
        h_new = h_new / weight_sums.to(dtype=h_new.dtype).view(-1, 1, 1)

    potts_decoder_aux_new = {
        "h": h_new,
        "J": J_new,
        "edge_idx": edge_idx_new,
        "mask_i": mask_i_new,
        "mask_ij": mask_ij_new,
        "coupling_mask": coupling_mask_new,
    }

    return potts_decoder_aux_new


def _construct_tied_sampling_inputs(batch: dict[str, TensorType["b ..."]]) -> dict[str, Any]:
    tied_sampling_inputs = {"tied_sampling_ids": batch["tied_sampling_ids"]}
    device = batch["tied_sampling_ids"].device
    B = batch["restype"].shape[0]
    tied_sampling_inputs["unique_ids"], tied_sampling_inputs["inverse"] = tied_sampling_inputs["tied_sampling_ids"].unique(return_inverse=True)
    reduce = str(batch.get("tied_sampling_aggregation_scheme", "mean"))
    if reduce not in {"mean", "sqrt", "weighted_mean"}:
        raise ValueError(
            "tied_sampling_aggregation_scheme must be 'mean', 'sqrt', "
            f"or 'weighted_mean'; got {reduce!r}"
        )
    tied_sampling_inputs["reduce"] = reduce

    # use first index of each tied group as the representative index
    batch_idx = torch.arange(B, device=device)
    n_unique_ids = tied_sampling_inputs["unique_ids"].shape[0]
    first_idxs = torch.full((n_unique_ids, ), B, device=device)
    first_idxs.scatter_reduce_(0, tied_sampling_inputs["inverse"], batch_idx, reduce="amin", include_self=True)
    tied_sampling_inputs["unique_rep_idx"] = first_idxs
    tied_sampling_inputs["rep_idx"] = first_idxs[tied_sampling_inputs["inverse"]]
    if "tied_sampling_weights" in batch:
        if reduce != "weighted_mean":
            raise ValueError(
                "tied_sampling_weights require "
                "tied_sampling_aggregation_scheme='weighted_mean'"
            )
        weights = batch["tied_sampling_weights"].to(device=device, dtype=torch.float32)
        if weights.ndim != 1 or weights.shape[0] != B:
            raise ValueError(
                "tied_sampling_weights must be a 1D tensor with one value per batch item"
            )
        if torch.any(weights < 0):
            raise ValueError("tied_sampling_weights must be non-negative")
        group_weight_sums = torch.zeros(n_unique_ids, device=device, dtype=weights.dtype)
        group_weight_sums.index_add_(0, tied_sampling_inputs["inverse"], weights)
        if torch.any(group_weight_sums <= 0):
            raise ValueError("Each tied_sampling group must have positive total weight")
        tied_sampling_inputs["weights"] = weights / group_weight_sums[tied_sampling_inputs["inverse"]]
    elif reduce == "weighted_mean":
        raise ValueError(
            "tied_sampling_aggregation_scheme='weighted_mean' requires "
            "tied_sampling_weights"
        )
    return tied_sampling_inputs
