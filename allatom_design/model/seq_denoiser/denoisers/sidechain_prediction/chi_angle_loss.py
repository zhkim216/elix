import torch
import torch.nn.functional as F
from torchtyping import TensorType


def wrap_degrees(angles: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angles + 180.0, 360.0) - 180.0


def chi_bin_centers(
    num_bins: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    bin_width = 360.0 / num_bins
    return torch.arange(num_bins, device=device, dtype=dtype) * bin_width - 180.0


def encode_chi_angles(
    chi_angles: torch.Tensor,
    num_bins: int,
    std_dev: float | None = None,
) -> torch.Tensor:
    """Circular Gaussian encoding over chi bin centers."""
    bin_width = 360.0 / num_bins
    if std_dev is None:
        std_dev = bin_width / 2.0

    centers = chi_bin_centers(num_bins, chi_angles.device, chi_angles.dtype)
    deltas = torch.minimum(
        torch.remainder(centers - chi_angles.unsqueeze(-1), 360.0),
        torch.remainder(chi_angles.unsqueeze(-1) - centers, 360.0),
    )
    encoding = torch.exp(-0.5 * (deltas / std_dev) ** 2)
    return encoding / encoding.sum(dim=-1, keepdim=True)


def chi_target_bins_and_offsets(
    chi_angles: TensorType["... x", float],
    num_bins: int,
) -> tuple[TensorType["... x", int], TensorType["... x", float]]:
    safe_angles = chi_angles.nan_to_num(nan=0.0)
    target_bins = encode_chi_angles(safe_angles, num_bins).argmax(dim=-1)

    centers = chi_bin_centers(num_bins, chi_angles.device, chi_angles.dtype)
    bin_centers = centers[target_bins]
    offset_pos = torch.remainder(chi_angles - bin_centers, 360.0)
    offset_neg = -torch.remainder(bin_centers - chi_angles, 360.0)
    use_pos = offset_pos.abs() <= offset_neg.abs()
    target_offsets = torch.where(use_pos, offset_pos, offset_neg)
    return target_bins, target_offsets


def masked_chi_cross_entropy(
    logits: TensorType["b n x c", float],
    chi_angles: TensorType["b n x", float],
    mask: TensorType["b n x", float],
) -> torch.Tensor:
    target_bins, _ = chi_target_bins_and_offsets(chi_angles, logits.shape[-1])
    cel = F.cross_entropy(
        logits.flatten(end_dim=-2),
        target_bins.flatten(),
        reduction="none",
    ).view_as(mask)
    denom = mask.sum().clamp(min=1.0)
    return (cel * mask).sum() / denom


def masked_chi_offset_mse(
    pred_offsets: TensorType["b n x", float],
    target_offsets: TensorType["b n x", float],
    mask: TensorType["b n x", float],
) -> torch.Tensor:
    mse = F.mse_loss(pred_offsets, target_offsets.nan_to_num(nan=0.0), reduction="none")
    denom = mask.sum().clamp(min=1.0)
    return (mse * mask).sum() / denom


def masked_chi_accuracy(
    logits: TensorType["b n x c", float],
    chi_angles: TensorType["b n x", float],
    mask: TensorType["b n x", float],
) -> torch.Tensor:
    target_bins, _ = chi_target_bins_and_offsets(chi_angles, logits.shape[-1])
    correct = (logits.argmax(dim=-1) == target_bins).float()
    denom = mask.sum().clamp(min=1.0)
    return (correct * mask).sum() / denom
