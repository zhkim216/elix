import numpy as np
import torch
import torch.nn as nn


def _zero_sum_gauge(h: torch.Tensor, J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = h - h.mean(-1, keepdim=True)
    J = (
        J
        - J.mean(-1, keepdim=True)
        - J.mean(-2, keepdim=True)
        + J.mean(dim=[-1, -2], keepdim=True)
    )
    return h, J


class FactorPottsHead(nn.Module):
    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        num_states: int,
        init_scale: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.num_states = num_states
        self.init_scale = init_scale
        self.log_scale = nn.Parameter(np.log(init_scale) * torch.ones(1))
        self.W_h = nn.Linear(self.dim_nodes, self.num_states, bias=True)
        self.W_J_left = nn.Linear(self.dim_edges, self.num_states ** 2, bias=True)
        self.W_J_right = nn.Linear(self.dim_edges, self.num_states ** 2, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        mask_i: torch.Tensor,
        mask_J: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.exp(self.log_scale)
        h = scale * mask_i.unsqueeze(-1) * self.W_h(node_h)
        mask_J = scale * mask_J.unsqueeze(-1)
        shape_J = list(edge_h.size())[:3] + ([self.num_states] * 2)
        J_left = (mask_J * self.W_J_left(edge_h)).view(shape_J)
        J_right = (mask_J * self.W_J_right(edge_h)).view(shape_J)
        J = torch.matmul(J_left, J_right)
        J = self.dropout(J)
        return _zero_sum_gauge(h, J)


class MultiHeadFactorPotts(nn.Module):
    """
    Multi-head Factor Potts model.
    """
    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        dim_multi_head: int,
        num_states: int,
        num_heads: int,
        reduce: str = "mean",
        init_scale: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if reduce not in {"mean", "sqrt"}:
            raise ValueError(f"Unknown multi-head factor reduce: {reduce!r}")
        if dim_multi_head % num_heads != 0:
            raise ValueError(
                f"dim_multi_head must be divisible by num_heads for multi_head_factor; "
                f"got dim_multi_head={dim_multi_head}, num_heads={num_heads}"
            )

        self.dim_multi_head = dim_multi_head
        self.num_heads = num_heads
        self.reduce = reduce
        self.w_h = nn.Linear(dim_nodes, self.dim_multi_head)
        self.w_e = nn.Linear(dim_edges, self.dim_multi_head)
        self.heads = nn.ModuleList(
            [
                FactorPottsHead(
                    dim_nodes=self.dim_multi_head // num_heads,
                    dim_edges=self.dim_multi_head // num_heads,
                    num_states=num_states,
                    init_scale=init_scale,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        mask_i: torch.Tensor,
        mask_J: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N = node_h.shape[:2]
        K = edge_h.shape[2]
        H = self.num_heads

        node_h = self.w_h(node_h).view(B, N, H, self.dim_multi_head // H)
        edge_h = self.w_e(edge_h).view(B, N, K, H, self.dim_multi_head // H)

        h, J = self.heads[0](
            node_h[:, :, 0, :],
            edge_h[:, :, :, 0, :],
            mask_i,
            mask_J,
        )
        for head_idx, head in enumerate(self.heads[1:], start=1):
            h_head, J_head = head(
                node_h[:, :, head_idx, :],
                edge_h[:, :, :, head_idx, :],
                mask_i,
                mask_J,
            )
            h = h + h_head
            J = J + J_head

        if self.reduce == "mean":
            scale = float(self.num_heads)
        elif self.reduce == "sqrt":
            scale = float(np.sqrt(self.num_heads))
        else:
            raise RuntimeError(f"Unexpected multi-head factor reduce: {self.reduce!r}")
        return h / scale, J / scale
