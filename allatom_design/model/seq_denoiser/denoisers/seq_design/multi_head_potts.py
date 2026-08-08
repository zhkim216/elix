import numpy as np
import torch
import torch.nn as nn


def _zero_sum_gauge(
    h: torch.Tensor,
    J: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        self.W_J_left = nn.Linear(
            self.dim_edges,
            self.num_states**2,
            bias=True,
        )
        self.W_J_right = nn.Linear(
            self.dim_edges,
            self.num_states**2,
            bias=True,
        )
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


class _ResidualAdapter(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear_in = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.linear_out = nn.Linear(hidden_dim, dim)
        self.reset_identity_parameters()

    def reset_identity_parameters(self) -> None:
        nn.init.zeros_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

    def update(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear_out(
            self.activation(self.linear_in(self.norm(features)))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.update(features)


def _gate_statistics(
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
    prefix: str,
) -> dict[str, torch.Tensor]:
    weights_float = weights.detach().float()
    valid_float = valid_mask.detach().float()
    valid_count = valid_float.sum()
    weight_sum = (weights_float * valid_float.unsqueeze(-1)).sum(
        dim=tuple(range(weights.ndim - 1))
    )

    entropy = -(
        weights_float * torch.log(weights_float.clamp_min(1e-8))
    ).sum(dim=-1) / np.log(weights.shape[-1])
    entropy_sum = (entropy * valid_float).sum()

    is_max = weights == weights.max(dim=-1, keepdim=True).values
    is_tie = is_max.sum(dim=-1) > 1
    unique_max = is_max & ~is_tie.unsqueeze(-1)
    argmax_count = (
        unique_max.float() * valid_float.unsqueeze(-1)
    ).sum(dim=tuple(range(weights.ndim - 1)))
    tie_count = (is_tie.float() * valid_float).sum()

    return {
        f"{prefix}_weight_sum": weight_sum,
        f"{prefix}_count": valid_count,
        f"{prefix}_entropy_sum": entropy_sum,
        f"{prefix}_argmax_count": argmax_count,
        f"{prefix}_tie_count": tie_count,
    }


def _multi_head_statistics(
    heads: nn.ModuleList,
    head_h: torch.Tensor,
    head_J: torch.Tensor,
    mask_i: torch.Tensor,
    mask_J: torch.Tensor,
    node_weights: torch.Tensor,
    edge_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    gated_head_h = head_h * node_weights.permute(2, 0, 1).unsqueeze(-1)
    gated_head_J = (
        head_J
        * edge_weights.permute(3, 0, 1, 2).unsqueeze(-1).unsqueeze(-1)
    )
    stats = {
        "head_exp_log_scale": torch.stack(
            [torch.exp(head.log_scale.detach()) for head in heads]
        ).flatten().float(),
        "head_gated_h_sq_sum": gated_head_h.detach().float().square().sum(
            dim=(1, 2, 3)
        ),
        "head_h_count": mask_i.float().sum() * head_h.shape[-1],
        "head_gated_J_sq_sum": gated_head_J.detach().float().square().sum(
            dim=(1, 2, 3, 4, 5)
        ),
        "head_J_count": mask_J.float().sum() * (head_J.shape[-1] ** 2),
    }
    stats.update(_gate_statistics(node_weights, mask_i, "node_gate"))
    stats.update(_gate_statistics(edge_weights, mask_J, "edge_gate"))
    return {name: value.detach() for name, value in stats.items()}


def _aggregate_head_outputs(
    head_h: torch.Tensor,
    head_J: torch.Tensor,
    node_weights: torch.Tensor,
    edge_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    h = (
        head_h.permute(1, 2, 0, 3) * node_weights.unsqueeze(-1)
    ).sum(dim=2)
    J = (
        head_J.permute(1, 2, 3, 0, 4, 5)
        * edge_weights.unsqueeze(-1).unsqueeze(-1)
    ).sum(dim=3)
    return h, J


class SharedEdgeBottleneckMultiHeadFactorPotts(nn.Module):
    """Gated Potts heads over one shared nonlinear edge bottleneck."""

    def __init__(
        self,
        dim_nodes: int,
        dim_edges: int,
        shared_edge_dim: int,
        num_states: int,
        num_heads: int,
        node_adapter_hidden_dim: int,
        init_scale: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_heads < 2:
            raise ValueError(
                "shared-edge multi-head Potts requires num_heads >= 2, "
                f"got {num_heads}"
            )
        if not 1 <= shared_edge_dim < dim_edges:
            raise ValueError(
                "shared_edge_dim must satisfy 1 <= shared_edge_dim < dim_edges; "
                f"got shared_edge_dim={shared_edge_dim}, dim_edges={dim_edges}"
            )

        self.dim_nodes = dim_nodes
        self.dim_edges = dim_edges
        self.shared_edge_dim = shared_edge_dim
        self.num_states = num_states
        self.num_heads = num_heads

        self.node_gate = nn.Linear(dim_nodes, num_heads)
        self.edge_gate = nn.Linear(shared_edge_dim, num_heads)
        self.node_adapter = _ResidualAdapter(
            dim_nodes,
            node_adapter_hidden_dim,
        )
        self.edge_projection = nn.Linear(dim_edges, shared_edge_dim)
        self.edge_adapter = _ResidualAdapter(
            shared_edge_dim,
            shared_edge_dim,
        )
        self.heads = nn.ModuleList(
            [
                FactorPottsHead(
                    dim_nodes=dim_nodes,
                    dim_edges=shared_edge_dim,
                    num_states=num_states,
                    init_scale=init_scale,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.reset_special_initialization()

    def reset_special_initialization(self) -> None:
        """Restore the Xavier bottleneck, identity adapters, and uniform gates."""
        nn.init.xavier_uniform_(self.edge_projection.weight, gain=1.0)
        nn.init.zeros_(self.edge_projection.bias)
        nn.init.zeros_(self.node_gate.weight)
        nn.init.zeros_(self.node_gate.bias)
        nn.init.zeros_(self.edge_gate.weight)
        nn.init.zeros_(self.edge_gate.bias)
        self.node_adapter.reset_identity_parameters()
        self.edge_adapter.reset_identity_parameters()

    def forward(
        self,
        node_h: torch.Tensor,
        edge_h: torch.Tensor,
        mask_i: torch.Tensor,
        mask_J: torch.Tensor,
        return_stats: bool = False,
    ):
        node_h = self.node_adapter(node_h)
        edge_h = self.edge_adapter(self.edge_projection(edge_h))

        node_weights = torch.softmax(self.node_gate(node_h), dim=-1)
        edge_weights = torch.softmax(self.edge_gate(edge_h), dim=-1)
        head_outputs = [
            head(node_h, edge_h, mask_i, mask_J)
            for head in self.heads
        ]

        head_h = torch.stack([output[0] for output in head_outputs], dim=0)
        head_J = torch.stack([output[1] for output in head_outputs], dim=0)
        h, J = _aggregate_head_outputs(
            head_h,
            head_J,
            node_weights,
            edge_weights,
        )
        if not return_stats:
            return h, J
        stats = _multi_head_statistics(
            self.heads,
            head_h,
            head_J,
            mask_i,
            mask_J,
            node_weights,
            edge_weights,
        )
        return h, J, stats
