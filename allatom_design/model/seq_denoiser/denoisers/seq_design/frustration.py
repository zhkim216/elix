"""Pairwise frustration transforms for Potts sequence sampling."""

import torch


def _validate_pairwise_inputs(J: torch.Tensor, mask_ij: torch.Tensor) -> None:
    if J.ndim != 5:
        raise ValueError(f"J must have shape [B, N, K, Q, Q], got {tuple(J.shape)}")
    if J.shape[-2] != J.shape[-1]:
        raise ValueError(f"J must have square state axes, got {tuple(J.shape[-2:])}")
    if tuple(mask_ij.shape) != tuple(J.shape[:-2]):
        raise ValueError(
            "mask_ij must match the [B, N, K] prefix of J; "
            f"got mask_ij={tuple(mask_ij.shape)}, J={tuple(J.shape)}"
        )
    if not torch.is_floating_point(J):
        raise TypeError(f"J must be floating point, got dtype={J.dtype}")


def compute_pairwise_frustration(
    J: torch.Tensor,
    mask_ij: torch.Tensor,
) -> torch.Tensor:
    """Standardize each valid edge's full ``Q x Q`` coupling matrix.

    For every sparse graph edge ``(i, j)``, this computes

    ``F_ij = (J_ij - mean(J_ij)) / std(J_ij)``

    over both state axes using the population standard deviation. No variance
    floor or clamp is applied. Masked graph edges are returned as exact zeros;
    a valid edge with zero variance therefore retains the formula's undefined
    division behavior.
    """
    _validate_pairwise_inputs(J, mask_ij)

    mu_ij = J.mean(dim=(-2, -1), keepdim=True)
    sigma_ij = J.std(dim=(-2, -1), correction=0, keepdim=True)
    edge_mask = mask_ij.bool()[..., None, None]
    return torch.where(edge_mask, (J - mu_ij) / sigma_ij, torch.zeros_like(J))


def mix_pairwise_couplings(
    J: torch.Tensor,
    mask_ij: torch.Tensor,
    *,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Return ``alpha * J + beta * F`` using the exact frustration transform."""
    _validate_pairwise_inputs(J, mask_ij)

    if beta == 0.0:
        if alpha == 1.0:
            return J
        return alpha * J

    F_ij = compute_pairwise_frustration(J, mask_ij)
    return F_ij.mul_(beta).add_(J, alpha=alpha)
