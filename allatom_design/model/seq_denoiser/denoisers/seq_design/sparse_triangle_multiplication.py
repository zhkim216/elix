"""Sparse triangle multiplications for protein and local-context pair graphs."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

def _apply_mask(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values
    return values * mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)


def _gather_residue_axis(features: Tensor, neighbor_idx: Tensor) -> Tensor:
    """Gather ``features[B, N, ...]`` at ``neighbor_idx[B, N, K]``."""
    batch, centers, neighbors = neighbor_idx.shape
    tail_shape = features.shape[2:]
    flat_idx = neighbor_idx.to(
        device=features.device,
        dtype=torch.long,
    ).reshape(batch, centers * neighbors)
    index = flat_idx.reshape(
        batch,
        centers * neighbors,
        *([1] * len(tail_shape)),
    ).expand(batch, centers * neighbors, *tail_shape)
    gathered = torch.gather(features, dim=1, index=index)
    return gathered.reshape(batch, centers, neighbors, *tail_shape)


def build_shared_atom_lookup(
    context_atom_idx: Tensor,
    context_mask: Tensor,
    neighbor_idx: Tensor,
    protein_pair_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Align residue-local context slots that refer to the same atom."""
    context_atom_idx = context_atom_idx.to(
        device=neighbor_idx.device,
        dtype=torch.long,
    )
    context_mask = context_mask.to(
        device=neighbor_idx.device,
        dtype=torch.bool,
    )
    neighbor_idx = neighbor_idx.to(dtype=torch.long)
    protein_pair_mask = protein_pair_mask.to(
        device=neighbor_idx.device,
        dtype=torch.bool,
    )
    _, num_residues, _ = neighbor_idx.shape
    center_atom_idx = context_atom_idx[:, :, None, :, None]
    center_context_mask = context_mask[:, :, None, :, None]
    neighbor_atom_idx = _gather_residue_axis(
        context_atom_idx,
        neighbor_idx,
    )[:, :, :, None, :]
    neighbor_context_mask = _gather_residue_axis(
        context_mask,
        neighbor_idx,
    )[:, :, :, None, :]
    shared_atom_matches = (
        (center_atom_idx == neighbor_atom_idx)
        & center_context_mask
        & neighbor_context_mask
    )
    center_idx = torch.arange(
        num_residues,
        device=neighbor_idx.device,
    ).view(1, num_residues, 1)
    shared_atom_edge_mask = (
        protein_pair_mask
        & (neighbor_idx != center_idx)
        & shared_atom_matches.any(dim=(-1, -2))
    )
    return shared_atom_matches, shared_atom_edge_mask


def build_protein_pair_edge_lookup(
    neighbor_idx: Tensor,
    pair_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return valid sparse PP edges and each residue's neighbor slot."""
    neighbor_idx = neighbor_idx.to(dtype=torch.long)
    pair_mask = pair_mask.to(
        device=neighbor_idx.device,
        dtype=torch.bool,
    )
    batch, num_residues, num_neighbors = neighbor_idx.shape
    protein_pair_edge_mask = (
        pair_mask
        & (neighbor_idx >= 0)
        & (neighbor_idx < num_residues)
    )
    safe_neighbor_idx = torch.where(
        protein_pair_edge_mask,
        neighbor_idx,
        torch.full_like(neighbor_idx, num_residues),
    )
    sorted_neighbor_idx, sorted_neighbor_slots = torch.sort(
        safe_neighbor_idx,
        dim=-1,
    )
    residue_idx = torch.arange(
        num_residues,
        device=neighbor_idx.device,
    ).view(1, 1, num_residues).expand(
        batch,
        num_residues,
        -1,
    ).contiguous()
    neighbor_slot_position = torch.searchsorted(
        sorted_neighbor_idx,
        residue_idx,
    )
    safe_slot_position = neighbor_slot_position.clamp_max(
        num_neighbors - 1
    )
    matched_neighbor_idx = torch.gather(
        sorted_neighbor_idx,
        dim=-1,
        index=safe_slot_position,
    )
    matched_neighbor_slot = torch.gather(
        sorted_neighbor_slots,
        dim=-1,
        index=safe_slot_position,
    )
    has_neighbor = (
        (neighbor_slot_position < num_neighbors)
        & (matched_neighbor_idx == residue_idx)
    )
    neighbor_slot_by_residue = torch.where(
        has_neighbor,
        matched_neighbor_slot,
        torch.full_like(matched_neighbor_slot, -1),
    ).to(torch.int32)
    return protein_pair_edge_mask, neighbor_slot_by_residue


def build_protein_pair_edge_slot_matches(
    neighbor_idx: Tensor,
    pair_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Align sparse edge slots between neighboring protein residues.

    ``neighbor_row_edge_slot_matches[b, i, r, q, s]`` is true when slot
    ``s`` in neighbor row ``E_idx[b, i, r]`` points to residue
    ``E_idx[b, i, q]``. The same tensor aligns outgoing right operands with
    axes ``[r, q]`` and incoming right operands with axes ``[q, r]``.
    """
    protein_pair_edge_mask, neighbor_slot_by_residue = (
        build_protein_pair_edge_lookup(neighbor_idx, pair_mask)
    )
    neighbor_idx = neighbor_idx.to(
        device=protein_pair_edge_mask.device,
        dtype=torch.long,
    )
    batch, num_residues, num_neighbors = neighbor_idx.shape
    safe_neighbor_idx = torch.where(
        protein_pair_edge_mask,
        neighbor_idx,
        torch.zeros_like(neighbor_idx),
    )
    batch_index = torch.arange(
        batch,
        device=neighbor_idx.device,
    ).view(batch, 1, 1, 1)
    neighbor_row_residue = safe_neighbor_idx.unsqueeze(-1)
    requested_residue = safe_neighbor_idx.unsqueeze(-2)
    neighbor_row_edge_slot = neighbor_slot_by_residue[
        batch_index,
        neighbor_row_residue,
        requested_residue,
    ]
    neighbor_row_path_valid = (
        protein_pair_edge_mask.unsqueeze(-1)
        & protein_pair_edge_mask.unsqueeze(-2)
        & (neighbor_row_edge_slot >= 0)
    )

    compact_batch_index = batch_index.squeeze(-1)
    center_residue = torch.arange(
        num_residues,
        device=neighbor_idx.device,
    ).view(1, num_residues, 1)
    incoming_center_edge_slot = neighbor_slot_by_residue[
        compact_batch_index,
        safe_neighbor_idx,
        center_residue,
    ]
    incoming_center_edge_valid = (
        protein_pair_edge_mask & (incoming_center_edge_slot >= 0)
    )
    edge_slot = torch.arange(
        num_neighbors,
        device=neighbor_idx.device,
        dtype=neighbor_row_edge_slot.dtype,
    )
    neighbor_row_edge_slot_matches = (
        neighbor_row_edge_slot.unsqueeze(-1) == edge_slot
    ) & neighbor_row_path_valid.unsqueeze(-1)
    incoming_center_edge_slot_matches = (
        incoming_center_edge_slot.unsqueeze(-1) == edge_slot
    ) & incoming_center_edge_valid.unsqueeze(-1)
    return (
        protein_pair_edge_mask,
        safe_neighbor_idx,
        neighbor_row_edge_slot_matches.contiguous(),
        incoming_center_edge_slot_matches.contiguous(),
    )


class SparseSharedAtomTriangleMultiplication(nn.Module):
    """Update sparse PP pairs from shared-atom PC x PC triangles.

    Context slots are residue-local, so the two PC operands are aligned by the
    global atom index stored in each slot. KNN target edges are chunked to bound
    the ``[B, N, K, M, H]`` intermediate.
    """

    def __init__(
        self,
        dim_protein_context: int,
        dim_protein_pair: int,
        dim_hidden: int,
        *,
        edge_chunk_size: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        if dim_hidden <= 0:
            raise ValueError("dim_hidden must be positive")
        if dim_protein_pair <= 0:
            raise ValueError("dim_protein_pair must be positive")
        if edge_chunk_size <= 0:
            raise ValueError("edge_chunk_size must be positive")

        self.shared_atom_product_sum_norm = nn.LayerNorm(dim_hidden)
        self.output_projection = nn.Linear(
            dim_hidden,
            dim_protein_pair,
            bias=False,
        )
        self.output_gate = nn.Linear(
            dim_protein_pair,
            dim_protein_pair,
            bias=False,
        )
        self.edge_chunk_size = edge_chunk_size
        self.pc_norm = nn.LayerNorm(dim_protein_context)
        self.left_projection = nn.Linear(
            dim_protein_context,
            dim_hidden,
            bias=False,
        )
        self.right_projection = nn.Linear(
            dim_protein_context,
            dim_hidden,
            bias=False,
        )
        self.protein_pair_norm = nn.LayerNorm(dim_protein_pair)
        self.dropout = nn.Dropout(dropout_p)
        self.residual_norm = nn.LayerNorm(dim_protein_pair)

    def forward(
        self,
        protein_context_pairs: Tensor,
        protein_pairs: Tensor,
        neighbor_idx: Tensor,
        shared_atom_matches: Tensor,
        *,
        shared_atom_edge_mask: Tensor,
        protein_pair_mask: Tensor,
    ) -> Tensor:
        shared_atom_matches = shared_atom_matches.to(
            device=protein_context_pairs.device,
            dtype=torch.bool,
        )
        shared_atom_edge_mask = shared_atom_edge_mask.to(
            device=protein_pairs.device,
            dtype=torch.bool,
        )
        protein_pair_mask = protein_pair_mask.to(
            device=protein_pairs.device,
            dtype=torch.bool,
        )

        batch, num_residues, num_neighbors = neighbor_idx.shape
        if protein_pairs.shape[:3] != (batch, num_residues, num_neighbors):
            raise ValueError(
                "protein_pairs and neighbor_idx must share [B, N, K]"
            )

        normalized_pc = self.pc_norm(protein_context_pairs)
        center_left = self.left_projection(normalized_pc)
        all_right = self.right_projection(normalized_pc)
        updated_chunks: list[Tensor] = []
        for start in range(0, num_neighbors, self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, num_neighbors)
            chunk_idx = neighbor_idx[:, :, start:stop]
            chunk_pairs = protein_pairs[:, :, start:stop]
            chunk_pair_mask = protein_pair_mask[:, :, start:stop]
            chunk_shared_atom_matches = shared_atom_matches[
                :, :, start:stop
            ]
            chunk_shared_atom_edge_mask = shared_atom_edge_mask[
                :, :, start:stop
            ]

            neighbor_right = _gather_residue_axis(all_right, chunk_idx)
            aligned_neighbor = torch.einsum(
                "bntij,bntjh->bntih",
                chunk_shared_atom_matches.to(neighbor_right.dtype),
                neighbor_right,
            )
            shared_atom_product_sum = (
                center_left.unsqueeze(2) * aligned_neighbor
            ).sum(dim=-2)
            normalized_target = self.protein_pair_norm(chunk_pairs)
            update = self.output_projection(
                self.shared_atom_product_sum_norm(shared_atom_product_sum)
            )
            update = update * torch.sigmoid(
                self.output_gate(normalized_target)
            )
            update = _apply_mask(
                update,
                chunk_shared_atom_edge_mask,
            )
            candidate = self.residual_norm(
                chunk_pairs + self.dropout(update)
            )
            updated = torch.where(
                chunk_shared_atom_edge_mask.unsqueeze(-1),
                candidate,
                chunk_pairs,
            )
            updated_chunks.append(
                _apply_mask(updated, chunk_pair_mask)
            )

        return torch.cat(updated_chunks, dim=2)


class SparseProteinPairTriangleMultiplication(nn.Module):
    """Triangle multiplication on sparse PP edges using edge-slot matches."""

    def __init__(
        self,
        dim_pair: int = 128,
        dim_hidden: int = 128,
        *,
        direction: Literal["outgoing", "incoming"],
        target_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        if dim_hidden <= 0:
            raise ValueError("dim_hidden must be positive")
        if dim_pair <= 0:
            raise ValueError("dim_pair must be positive")
        if direction not in {"outgoing", "incoming"}:
            raise ValueError(
                "direction must be 'outgoing' or 'incoming'"
            )
        if target_chunk_size <= 0:
            raise ValueError("target_chunk_size must be positive")

        self.protein_pair_path_product_sum_norm = nn.LayerNorm(dim_hidden)
        self.output_projection = nn.Linear(
            dim_hidden,
            dim_pair,
            bias=False,
        )
        self.output_gate = nn.Linear(
            dim_pair,
            dim_pair,
            bias=False,
        )
        self.direction = direction
        self.dim_hidden = dim_hidden
        self.target_chunk_size = target_chunk_size
        self.input_norm = nn.LayerNorm(dim_pair)
        self.input_projection = nn.Linear(
            dim_pair,
            2 * dim_hidden,
            bias=False,
        )
        self.input_gate = nn.Linear(
            dim_pair,
            2 * dim_hidden,
            bias=False,
        )

    def forward(
        self,
        pairs: Tensor,
        protein_pair_edge_mask: Tensor,
        safe_neighbor_idx: Tensor,
        neighbor_row_edge_slot_matches: Tensor,
        incoming_center_edge_slot_matches: Tensor | None = None,
    ) -> Tensor:
        batch, num_residues, num_neighbors, _ = pairs.shape
        if protein_pair_edge_mask.shape != (
            batch,
            num_residues,
            num_neighbors,
        ):
            raise ValueError(
                "protein_pair_edge_mask must match pairs along [B, N, K]"
            )
        if safe_neighbor_idx.shape != (batch, num_residues, num_neighbors):
            raise ValueError(
                "safe_neighbor_idx must have shape [B, N, K]"
            )
        expected_match_shape = (
            batch,
            num_residues,
            num_neighbors,
            num_neighbors,
            num_neighbors,
        )
        if neighbor_row_edge_slot_matches.shape != expected_match_shape:
            raise ValueError(
                "neighbor_row_edge_slot_matches must have shape "
                "[B, N, K, K, K]"
            )
        if self.direction == "incoming":
            if incoming_center_edge_slot_matches is None:
                raise ValueError(
                    "incoming triangle requires "
                    "incoming_center_edge_slot_matches"
                )
            if incoming_center_edge_slot_matches.shape != (
                batch,
                num_residues,
                num_neighbors,
                num_neighbors,
            ):
                raise ValueError(
                    "incoming_center_edge_slot_matches must have shape "
                    "[B, N, K, K]"
                )
        protein_pair_edge_mask = protein_pair_edge_mask.to(
            device=pairs.device,
            dtype=torch.bool,
        )
        safe_neighbor_idx = safe_neighbor_idx.to(
            device=pairs.device,
            dtype=torch.long,
        )
        neighbor_row_edge_slot_matches = (
            neighbor_row_edge_slot_matches.to(
                device=pairs.device,
                dtype=torch.bool,
            )
        )
        if incoming_center_edge_slot_matches is not None:
            incoming_center_edge_slot_matches = (
                incoming_center_edge_slot_matches.to(
                    device=pairs.device,
                    dtype=torch.bool,
                )
            )

        normalized = self.input_norm(pairs)
        projected = (
            torch.sigmoid(self.input_gate(normalized))
            * self.input_projection(normalized)
        )
        left, right = projected.chunk(2, dim=-1)

        output_chunks: list[Tensor] = []

        if self.direction == "incoming":
            candidate_left = _gather_residue_axis(left, safe_neighbor_idx)
            incoming_first_leg = torch.einsum(
                "bncs,bncsh->bnch",
                incoming_center_edge_slot_matches.to(candidate_left.dtype),
                candidate_left,
            )
            candidate_right = _gather_residue_axis(right, safe_neighbor_idx)

        for start in range(0, num_neighbors, self.target_chunk_size):
            stop = min(start + self.target_chunk_size, num_neighbors)
            target_edge_valid = protein_pair_edge_mask[
                :, :, start:stop
            ]
            if self.direction == "outgoing":
                target_right = _gather_residue_axis(
                    right,
                    safe_neighbor_idx[:, :, start:stop],
                )
                aligned_right = torch.einsum(
                    "bntcs,bntsh->bntch",
                    neighbor_row_edge_slot_matches[
                        :, :, start:stop
                    ].to(target_right.dtype),
                    target_right,
                )
                first_leg = left.unsqueeze(2)
            else:
                aligned_right = torch.einsum(
                    "bncts,bncsh->bntch",
                    neighbor_row_edge_slot_matches[
                        :, :, :, start:stop
                    ].to(candidate_right.dtype),
                    candidate_right,
                )
                first_leg = incoming_first_leg.unsqueeze(2)
            protein_pair_path_product_sum = (
                first_leg
                * aligned_right
            ).sum(dim=3)

            normalized_target = normalized[:, :, start:stop]
            update = self.output_projection(
                self.protein_pair_path_product_sum_norm(
                    protein_pair_path_product_sum
                )
            )
            update = update * torch.sigmoid(
                self.output_gate(normalized_target)
            )
            update = _apply_mask(
                update,
                target_edge_valid,
            )
            output_chunks.append(update)

        return torch.cat(output_chunks, dim=2)
