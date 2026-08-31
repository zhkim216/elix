"""Model-local input and output sequence encodings for ElixMPNN."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F
from atomworks.constants import STANDARD_AA, UNKNOWN_AA

import allatom_design.data.const as const

MASK_TOKEN = "<M>"
INPUT_TOKENS = (*STANDARD_AA, UNKNOWN_AA, MASK_TOKEN)
OUTPUT_TOKENS = tuple(STANDARD_AA)


class SequenceEncoding:
    """Fixed residue-name encoding used by one model tensor axis."""

    def __init__(self, tokens: Sequence[str]):
        self._tokens = tuple(tokens)
        self._token_to_idx = {token: index for index, token in enumerate(self._tokens)}

    @property
    def tokens(self) -> tuple[str, ...]:
        return self._tokens

    @property
    def token_to_idx(self) -> dict[str, int]:
        return self._token_to_idx

    @cached_property
    def idx_to_token(self) -> np.ndarray:
        return np.array(self._tokens)

    def encode(self, residue_names: Sequence[str]) -> list[int]:
        return [self._token_to_idx[name] for name in residue_names]

    def encode_aa_seq(self, amino_acids: Sequence[str]) -> list[int]:
        residue_names = [const.PROT_LETTER_TO_TOKEN[aa] for aa in amino_acids]
        return self.encode(residue_names)

    def decode_aa_seq(self, token_indices: Sequence[int]) -> str:
        return "".join(
            const.PROT_TOKEN_TO_LETTER[token]
            for token in self.idx_to_token[list(token_indices)]
        )


INPUT_ENCODING = SequenceEncoding(INPUT_TOKENS)
OUTPUT_ENCODING = SequenceEncoding(OUTPUT_TOKENS)


def standard_aa_token_mask(
    batch: dict[str, torch.Tensor], *, is_sampling: bool = False
) -> torch.Tensor:
    """Resolved, non-hetero, non-atomized standard protein residues."""
    restype_idx = batch["restype"].argmax(dim=-1)
    standard_aa_idx = torch.tensor(
        [const.AF3_ENCODING.token_to_idx[token] for token in STANDARD_AA],
        device=restype_idx.device,
        dtype=restype_idx.dtype,
    )
    is_standard_aa = (restype_idx.unsqueeze(-1) == standard_aa_idx).any(dim=-1)
    return _eligible_protein_token_mask(batch, is_sampling=is_sampling) & is_standard_aa


def unknown_aa_token_mask(
    batch: dict[str, torch.Tensor], *, is_sampling: bool = False
) -> torch.Tensor:
    """Resolved, non-hetero, non-atomized protein UNK residues."""
    restype_idx = batch["restype"].argmax(dim=-1)
    unk_idx = const.AF3_ENCODING.token_to_idx[UNKNOWN_AA]
    return _eligible_protein_token_mask(
        batch, is_sampling=is_sampling
    ) & (restype_idx == unk_idx)


def apply_sequence_encoding(
    batch: dict[str, torch.Tensor],
    *,
    is_sampling: bool = False,
    allow_unk_redesign: bool = False,
) -> dict[str, torch.Tensor]:
    """Add model input/target tensors and graph/Potts/supervision masks.

    The AF3-width ``batch['restype']`` tensor is read only and remains unchanged.
    """
    if allow_unk_redesign and not is_sampling:
        raise ValueError("allow_unk_redesign is supported only during sampling")

    restype_idx = batch["restype"].argmax(dim=-1)
    standard_mask = standard_aa_token_mask(batch, is_sampling=is_sampling)
    unknown_mask = unknown_aa_token_mask(batch, is_sampling=is_sampling)
    graph_mask = standard_mask | unknown_mask

    seq_cond_mask = batch["seq_cond_mask"].bool()
    sampler_cond_mask = batch.get("seq_cond_mask_potts", batch["seq_cond_mask"]).bool()
    redesignable_unknown = unknown_mask & ~sampler_cond_mask & allow_unk_redesign
    potts_mask = standard_mask | redesignable_unknown

    target_restype = torch.where(
        standard_mask,
        restype_idx,
        torch.zeros_like(restype_idx),
    ).long()
    input_idx = target_restype.clone()
    input_idx = torch.where(
        unknown_mask,
        input_idx.new_full((), INPUT_ENCODING.token_to_idx[UNKNOWN_AA]),
        input_idx,
    )
    input_mask = (standard_mask & ~seq_cond_mask) | redesignable_unknown
    input_idx = torch.where(
        input_mask,
        input_idx.new_full((), INPUT_ENCODING.token_to_idx[MASK_TOKEN]),
        input_idx,
    )

    batch["sequence_input"] = F.one_hot(
        input_idx, num_classes=len(INPUT_TOKENS)
    ).to(device=batch["restype"].device, dtype=torch.float32)
    batch["target_restype"] = target_restype
    batch["protein_residue_node_mask"] = graph_mask.to(
        dtype=batch["token_pad_mask"].dtype
    )
    batch["potts_node_mask"] = potts_mask.to(dtype=batch["token_pad_mask"].dtype)
    batch["sequence_target_mask"] = standard_mask.to(
        dtype=batch["token_pad_mask"].dtype
    )
    return batch


def project_pos_restrict_aatype(
    pos_restrict_aatype: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Project external 20-, 22-, or AF3-width restrictions to 20 outputs."""
    if pos_restrict_aatype is None:
        return None
    restrict_pos_mask, allowed_aatype_mask = pos_restrict_aatype
    source_width = allowed_aatype_mask.shape[-1]
    supported_widths = {
        len(OUTPUT_TOKENS),
        len(INPUT_TOKENS),
        const.AF3_ENCODING.n_tokens,
    }
    if source_width not in supported_widths:
        raise ValueError(
            "pos_restrict_aatype allowed_aatype_mask has unsupported alphabet size "
            f"{source_width}; expected one of {sorted(supported_widths)}"
        )
    return restrict_pos_mask, allowed_aatype_mask[..., : len(OUTPUT_TOKENS)]


def protein_context_token_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "token_is_protein_chain" in batch:
        return batch["token_is_protein_chain"].bool()
    return _bool_feature(batch, "is_protein", default=False)


def _eligible_protein_token_mask(
    batch: dict[str, torch.Tensor], *, is_sampling: bool
) -> torch.Tensor:
    token_is_hetero = _bool_feature(batch, "token_is_hetero", default=False)
    is_atomized = _bool_feature(batch, "is_atomized", default=False)
    token_exists_mask = batch.get(
        "token_exists_mask", batch["token_resolved_mask"]
    ).bool()
    residuewise_hetero_mask = torch.ones_like(
        batch["token_pad_mask"], dtype=torch.bool
    )
    if is_sampling:
        residuewise_hetero_mask = batch.get(
            "residuewise_hetero_mask",
            residuewise_hetero_mask,
        ).bool()
    return (
        batch["token_pad_mask"].bool()
        & token_exists_mask
        & protein_context_token_mask(batch)
        & ~token_is_hetero
        & ~is_atomized
        & residuewise_hetero_mask
    )


def _bool_feature(
    batch: dict[str, torch.Tensor], key: str, *, default: bool
) -> torch.Tensor:
    if key in batch:
        return batch[key].bool()
    return torch.full_like(
        batch["token_pad_mask"], fill_value=default, dtype=torch.bool
    )
