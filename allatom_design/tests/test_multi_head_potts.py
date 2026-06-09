import math

import pytest
import torch

from allatom_design.model.seq_denoiser.denoisers.seq_design.potts import GraphPotts
from allatom_design.model.seq_denoiser.denoisers.seq_design.multi_head_potts import (
    MultiHeadFactorPotts,
)


def test_graph_potts_multi_head_factor_shapes_and_parameters():
    torch.manual_seed(0)
    batch_size, num_nodes, num_neighbors = 2, 4, 4
    dim_nodes, dim_edges, dim_multi_head, num_states = 5, 7, 6, 3
    model = GraphPotts(
        dim_nodes=dim_nodes,
        dim_edges=dim_edges,
        dim_multi_head=dim_multi_head,
        num_states=num_states,
        parameterization="multi_head_factor",
        num_heads=3,
        reduce="mean",
        symmetric_J=True,
        dropout=0.0,
    )

    node_h = torch.randn(batch_size, num_nodes, dim_nodes)
    edge_h = torch.randn(batch_size, num_nodes, num_neighbors, dim_edges)
    edge_idx = torch.arange(num_nodes).view(1, 1, num_nodes).expand(
        batch_size, num_nodes, num_nodes
    )
    mask_i = torch.ones(batch_size, num_nodes)
    mask_ij = torch.ones(batch_size, num_nodes, num_neighbors)

    h, J = model(node_h, edge_h, edge_idx, mask_i, mask_ij)

    assert h.shape == (batch_size, num_nodes, num_states)
    assert J.shape == (
        batch_size,
        num_nodes,
        num_neighbors,
        num_states,
        num_states,
    )
    param_names = dict(model.named_parameters())
    assert "multi_head_factor.w_h.weight" in param_names
    assert "multi_head_factor.w_e.weight" in param_names
    assert "multi_head_factor.heads.0.W_h.weight" in param_names
    assert "multi_head_factor.heads.2.W_J_right.weight" in param_names


def test_multi_head_factor_sqrt_reduce_scales_mean_reduce():
    torch.manual_seed(1)
    num_heads = 2
    module = MultiHeadFactorPotts(
        dim_nodes=6,
        dim_edges=8,
        dim_multi_head=6,
        num_states=3,
        num_heads=num_heads,
        reduce="mean",
        dropout=0.0,
    )
    module.eval()

    node_h = torch.randn(2, 4, 6)
    edge_h = torch.randn(2, 4, 3, 8)
    mask_i = torch.ones(2, 4)
    mask_J = torch.ones(2, 4, 3)

    h_mean, J_mean = module(node_h, edge_h, mask_i, mask_J)
    module.reduce = "sqrt"
    h_sqrt, J_sqrt = module(node_h, edge_h, mask_i, mask_J)

    scale = math.sqrt(num_heads)
    torch.testing.assert_close(h_sqrt, h_mean * scale)
    torch.testing.assert_close(J_sqrt, J_mean * scale)


def test_multi_head_factor_rejects_unknown_reduce():
    with pytest.raises(ValueError, match="Unknown multi-head factor reduce"):
        MultiHeadFactorPotts(
            dim_nodes=6,
            dim_edges=8,
            dim_multi_head=6,
            num_states=3,
            num_heads=2,
            reduce="sum",
        )


def test_multi_head_factor_rejects_unsplittable_projected_dimension():
    with pytest.raises(ValueError, match="dim_multi_head must be divisible"):
        MultiHeadFactorPotts(
            dim_nodes=5,
            dim_edges=7,
            dim_multi_head=10,
            num_states=3,
            num_heads=3,
        )


def test_multi_head_factor_allows_unsplittable_input_dimensions():
    module = MultiHeadFactorPotts(
        dim_nodes=5,
        dim_edges=7,
        dim_multi_head=6,
        num_states=3,
        num_heads=3,
        dropout=0.0,
    )

    node_h = torch.randn(2, 4, 5)
    edge_h = torch.randn(2, 4, 3, 7)
    mask_i = torch.ones(2, 4)
    mask_J = torch.ones(2, 4, 3)

    h, J = module(node_h, edge_h, mask_i, mask_J)

    assert h.shape == (2, 4, 3)
    assert J.shape == (2, 4, 3, 3, 3)


def test_graph_potts_factor_does_not_require_dim_multi_head():
    model = GraphPotts(
        dim_nodes=5,
        dim_edges=7,
        num_states=3,
        parameterization="factor",
        symmetric_J=False,
        dropout=0.0,
    )

    node_h = torch.randn(2, 4, 5)
    edge_h = torch.randn(2, 4, 3, 7)
    edge_idx = torch.arange(3).view(1, 1, 3).expand(2, 4, 3)
    mask_i = torch.ones(2, 4)
    mask_ij = torch.ones(2, 4, 3)

    h, J = model(node_h, edge_h, edge_idx, mask_i, mask_ij)

    assert h.shape == (2, 4, 3)
    assert J.shape == (2, 4, 3, 3, 3)
