"""Model-local sequence encoding helpers for compact Potts alphabets."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import allatom_design.data.const as const


def selected_sequence_encoding(use_potts_encoding: bool):
    """Return the sequence encoding used by the denoiser/model path."""
    return const.POTTS_ENCODING if use_potts_encoding else const.AF3_ENCODING


def strict_standard_aa_token_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Standard, non-hetero, non-atomized protein/peptide tokens in an AF3 batch."""
    restype_idx = batch["restype"].argmax(dim=-1)
    token_pad_mask = batch["token_pad_mask"].bool()
    protein_context = _protein_context_mask(batch)
    token_is_hetero = _bool_feature(batch, "token_is_hetero", default=False)
    is_atomized = _bool_feature(batch, "is_atomized", default=False)

    standard_aa_idx = torch.tensor(
        [const.AF3_ENCODING.token_to_idx[token] for token in const.STANDARD_AA],
        device=restype_idx.device,
        dtype=restype_idx.dtype,
    )
    is_standard_aa = (restype_idx.unsqueeze(-1) == standard_aa_idx).any(dim=-1)
    return token_pad_mask & protein_context & is_standard_aa & ~token_is_hetero & ~is_atomized


def protein_context_token_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Protein-chain context tokens, falling back to legacy is_protein if needed."""
    return _protein_context_mask(batch)


def apply_potts_standard_aa_mask(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace token_is_prot_std_aa with the compact-Potts strict AA definition."""
    batch["token_is_prot_std_aa"] = strict_standard_aa_token_mask(batch).to(
        dtype=batch["token_is_prot_std_aa"].dtype
    )
    return batch


def apply_potts_restype_encoding(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Project AF3 restype to compact 24-class Potts input and target tensors.

    ``batch["restype"]`` becomes the model input: masked standard AA positions
    carry ``<M>`` and synthetic pads carry ``<pad>``. ``batch["target_restype"]``
    keeps the unmasked target index for losses and Potts scoring.
    """
    restype = batch["restype"]
    target_idx = potts_target_restype_indices(batch)
    model_idx = target_idx.clone()
    standard_aa_mask = strict_standard_aa_token_mask(batch)
    seq_cond_mask = batch.get(
        "seq_cond_mask",
        torch.ones_like(batch["token_pad_mask"], dtype=batch["token_pad_mask"].dtype),
    ).bool()
    mask_idx = const.POTTS_ENCODING.token_to_idx[const.POTTS_MASK_TOKEN]
    model_idx = torch.where(standard_aa_mask & ~seq_cond_mask, model_idx.new_full((), mask_idx), model_idx)

    batch["target_restype"] = target_idx
    batch["restype"] = F.one_hot(model_idx, num_classes=const.POTTS_ENCODING.n_tokens).to(
        device=restype.device,
        dtype=torch.float32,
    )
    return batch


def potts_target_restype_indices(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return compact-Potts target indices without applying the mask token."""
    restype_idx = batch["restype"].argmax(dim=-1)
    token_pad_mask = batch["token_pad_mask"].bool()
    standard_aa_mask = strict_standard_aa_token_mask(batch)
    protein_context = _protein_context_mask(batch)

    unk_idx = const.POTTS_ENCODING.token_to_idx[const.UNKNOWN_AA]
    non_protein_idx = const.POTTS_ENCODING.token_to_idx[const.POTTS_NON_PROTEIN_TOKEN]
    pad_idx = const.POTTS_ENCODING.token_to_idx[const.POTTS_PAD_TOKEN]

    target_idx = restype_idx.new_full(restype_idx.shape, non_protein_idx)
    target_idx = torch.where(protein_context & token_pad_mask & ~standard_aa_mask, target_idx.new_full((), unk_idx), target_idx)
    target_idx = torch.where(standard_aa_mask, restype_idx, target_idx)
    target_idx = torch.where(token_pad_mask, target_idx, target_idx.new_full((), pad_idx))
    return target_idx


def project_pos_restrict_aatype_to_encoding(
    pos_restrict_aatype: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    use_potts_encoding: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Convert external AA restrictions to the selected model alphabet."""
    if pos_restrict_aatype is None:
        return None
    if not use_potts_encoding:
        return pos_restrict_aatype

    restrict_pos_mask, allowed_aatype_mask = pos_restrict_aatype
    if allowed_aatype_mask.shape[-1] == const.POTTS_ENCODING.n_tokens:
        return restrict_pos_mask, allowed_aatype_mask
    if allowed_aatype_mask.shape[-1] not in (len(const.STANDARD_AA), const.AF3_ENCODING.n_tokens):
        raise ValueError(
            "pos_restrict_aatype allowed_aatype_mask has unsupported alphabet size "
            f"{allowed_aatype_mask.shape[-1]}; expected {len(const.STANDARD_AA)}, "
            f"{const.AF3_ENCODING.n_tokens}, or {const.POTTS_ENCODING.n_tokens}"
        )

    projected = allowed_aatype_mask.new_zeros(
        (*allowed_aatype_mask.shape[:-1], const.POTTS_ENCODING.n_tokens)
    )
    n_standard = len(const.STANDARD_AA)
    projected[..., :n_standard] = allowed_aatype_mask[..., :n_standard]
    return restrict_pos_mask, projected


def _protein_context_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    if "token_is_protein_chain" in batch:
        return batch["token_is_protein_chain"].bool()
    return _bool_feature(batch, "is_protein", default=False)


def _bool_feature(batch: dict[str, torch.Tensor], key: str, *, default: bool) -> torch.Tensor:
    if key in batch:
        return batch[key].bool()
    return torch.full_like(batch["token_pad_mask"], fill_value=default, dtype=torch.bool)
