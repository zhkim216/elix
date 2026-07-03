from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from allatom_design.model.seq_denoiser.denoisers.sidechain_prediction.chi_angle_loss import (
    chi_bin_centers,
    chi_target_bins_and_offsets,
    encode_chi_angles,
    wrap_degrees,
)


class ChiAnglePredictionHead(nn.Module):
    """Predict sidechain chi angle bins and offsets from Potts node embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_chi: int = 4,
        num_bins: int = 72,
        dropout_p: float = 0.0,
        ):
        super().__init__()
        self.num_chi = num_chi
        self.num_bins = num_bins

        self.node_norm = nn.LayerNorm(input_dim)
        self.chi_prediction_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim + chi_idx * num_bins, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, num_bins),
                )
                for chi_idx in range(num_chi)
            ]
        )
        self.chi_offset_prediction_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * num_bins, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_p),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_chi)
            ]
        )

    def forward(
        self,
        h_V_potts: TensorType["b n d", float],
        chi_angles: Optional[TensorType["b n x", float]] = None,
    ) -> dict[str, TensorType["b n ...", float]]:
        B, N, _ = h_V_potts.shape
        node_features = self.node_norm(h_V_potts).reshape(B * N, -1)

        target_bins = None
        target_offsets = None
        gt_chi_encoding = None
        if chi_angles is not None:
            chi_angles = chi_angles[..., : self.num_chi]
            target_bins, target_offsets = chi_target_bins_and_offsets(chi_angles, self.num_bins)
            gt_chi_encoding = encode_chi_angles(
                chi_angles.reshape(B * N, self.num_chi),
                self.num_bins,
            ).nan_to_num()
            target_bins = target_bins.reshape(B * N, self.num_chi)
            target_offsets = target_offsets.reshape(B, N, self.num_chi)

        prev_chi = node_features.new_empty((B * N, 0))
        centers = chi_bin_centers(self.num_bins, h_V_potts.device, h_V_potts.dtype)
        logits_by_chi = []
        offsets_by_chi = []
        angles_by_chi = []

        for chi_idx, chi_layer in enumerate(self.chi_prediction_layers):
            chi_logits = chi_layer(torch.cat([node_features, prev_chi], dim=-1))
            if target_bins is not None:
                selected_bins = target_bins[:, chi_idx]
                selected_one_hot = F.one_hot(selected_bins, num_classes=self.num_bins).to(chi_logits.dtype)
            else:
                selected_bins = chi_logits.argmax(dim=-1)
                selected_one_hot = F.one_hot(selected_bins, num_classes=self.num_bins).to(chi_logits.dtype)

            pred_offset = self.chi_offset_prediction_layers[chi_idx](
                torch.cat([chi_logits, selected_one_hot], dim=-1)
            ).squeeze(-1)
            pred_angle = wrap_degrees(centers[selected_bins] + pred_offset)

            if gt_chi_encoding is not None:
                next_chi_encoding = gt_chi_encoding[:, chi_idx]
            else:
                next_chi_encoding = encode_chi_angles(pred_angle, self.num_bins)
            prev_chi = torch.cat([prev_chi, next_chi_encoding], dim=-1)

            logits_by_chi.append(chi_logits.view(B, N, self.num_bins))
            offsets_by_chi.append(pred_offset.view(B, N))
            angles_by_chi.append(pred_angle.view(B, N))

        outputs = {
            "chi_logits": torch.stack(logits_by_chi, dim=2),
            "chi_offsets": torch.stack(offsets_by_chi, dim=2),
            "predicted_chi_angles": torch.stack(angles_by_chi, dim=2),
        }
        if target_offsets is not None:
            outputs["target_chi_offsets"] = target_offsets
        return outputs
