import pytest
import torch
from omegaconf import OmegaConf

from allatom_design.model.seq_denoiser.denoisers.seq_design.elix_mpnn import (
    ContextModule,
    ElixMPNN,
)


def _make_elix_mpnn_cfg(
    *,
    hidden_dim: int = 8,
    expansion_mode: str | None = "node_concat_context_skip",
    parameterization: str = "factor",
    ligand_conditioning: bool = True,
    use_context_skip_connection: bool = False,
):
    return OmegaConf.create(
        {
            "hidden_dim": hidden_dim,
            "k_neighbors": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dropout_p": 0.0,
            "ligand_conditioning": ligand_conditioning,
            "use_context_skip_connection": use_context_skip_connection,
            "lmpnn_module": {
                "num_context_feature_processor_layers": 1,
                "num_context_feature_aggregator_layers": 1,
                "context_edge_update": False,
            },
            "token_features": {
                "num_rbf": 4,
                "min_rbf_mean": 2.0,
                "max_rbf_mean": 22.0,
                "protein_graph_rbf_type": "ca",
                "k_neighbors": 4,
                "num_positional_embeddings": 4,
                "hidden_dim": hidden_dim,
                "protein_ligand_interaction_rbf_type": "cb",
                "add_angle_features": False,
                "use_multichain_encoding": True,
                "ligand_conditioning": ligand_conditioning,
                "use_ligand_context": True,
                "ligand_atom_context_num": 3,
                "masked_distance_fill": 1000.0,
            },
            "potts": {
                "k_neighbors_potts": 4,
                "max_dist_potts": None,
                "use_potts": True,
                "norm_potts_inputs": False,
                "parameterization": parameterization,
                "num_factors": None,
                "symmetric_J": False,
                "multi_head": {
                    "dim_multi_head": 8,
                    "reduce": "mean",
                    "num_heads": 1,
                },
            },
            "expansion_mode": expansion_mode,
        }
    )


def test_node_concat_context_skip_uses_factor_potts_with_context_skip_inputs():
    hidden_dim = 8
    model = ElixMPNN(_make_elix_mpnn_cfg(hidden_dim=hidden_dim))

    assert model.return_context_skip
    assert not model.use_context_skip_connection
    assert model.dim_nodes_potts == 2 * hidden_dim
    assert model.dim_edges_potts == 5 * hidden_dim
    assert model.decoder_S_potts.parameterization == "factor"
    assert model.decoder_S_potts.dim_nodes == 2 * hidden_dim
    assert model.decoder_S_potts.dim_edges == 5 * hidden_dim
    assert model.decoder_S_potts.W_h.in_features == 2 * hidden_dim
    assert model.decoder_S_potts.W_J_left.in_features == 5 * hidden_dim


def test_node_concat_context_skip_rejects_non_factor_potts():
    cfg = _make_elix_mpnn_cfg(parameterization="multi_head_factor")

    with pytest.raises(ValueError, match="node_concat_context_skip.*factor"):
        ElixMPNN(cfg)


def test_node_concat_context_skip_requires_ligand_conditioning():
    cfg = _make_elix_mpnn_cfg(ligand_conditioning=False)

    with pytest.raises(AssertionError, match="node_concat_context_skip.*ligand_conditioning"):
        ElixMPNN(cfg)


def test_node_concat_context_skip_edge_segment_order():
    model = ElixMPNN(_make_elix_mpnn_cfg(hidden_dim=2))
    h_V = torch.tensor(
        [
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ]
        ]
    )
    h_E = torch.tensor(
        [
            [
                [[100.0, 101.0], [110.0, 111.0]],
                [[200.0, 201.0], [210.0, 211.0]],
                [[300.0, 301.0], [310.0, 311.0]],
            ]
        ]
    )
    h_V_C_skip = torch.tensor(
        [
            [
                [1000.0, 1001.0],
                [2000.0, 2001.0],
                [3000.0, 3001.0],
            ]
        ]
    )
    E_idx = torch.tensor([[[1, 2], [0, 2], [0, 1]]])

    expanded = model._expand_potts_edges(h_V, h_E, E_idx, h_V_C_skip)

    assert expanded.shape == (1, 3, 2, 10)
    torch.testing.assert_close(expanded[..., 0:2], h_V.unsqueeze(-2).expand(-1, -1, 2, -1))
    torch.testing.assert_close(expanded[..., 2:4], h_E)
    torch.testing.assert_close(
        expanded[..., 4:6],
        torch.tensor(
            [
                [
                    [[20.0, 21.0], [30.0, 31.0]],
                    [[10.0, 11.0], [30.0, 31.0]],
                    [[10.0, 11.0], [20.0, 21.0]],
                ]
            ]
        ),
    )
    torch.testing.assert_close(
        expanded[..., 6:8],
        h_V_C_skip.unsqueeze(-2).expand(-1, -1, 2, -1),
    )
    torch.testing.assert_close(
        expanded[..., 8:10],
        torch.tensor(
            [
                [
                    [[2000.0, 2001.0], [3000.0, 3001.0]],
                    [[1000.0, 1001.0], [3000.0, 3001.0]],
                    [[1000.0, 1001.0], [2000.0, 2001.0]],
                ]
            ]
        ),
    )


def test_node_concat_context_skip_concats_skip_to_potts_nodes():
    model = ElixMPNN(_make_elix_mpnn_cfg(hidden_dim=2))
    h_V = torch.tensor(
        [
            [
                [10.0, 11.0],
                [20.0, 21.0],
                [30.0, 31.0],
            ]
        ]
    )
    h_V_C_skip = torch.tensor(
        [
            [
                [1000.0, 1001.0],
                [2000.0, 2001.0],
                [3000.0, 3001.0],
            ]
        ]
    )

    expanded = model._expand_potts_nodes(h_V, h_V_C_skip)

    assert expanded.shape == (1, 3, 4)
    torch.testing.assert_close(expanded[..., 0:2], h_V)
    torch.testing.assert_close(expanded[..., 2:4], h_V_C_skip)


@pytest.mark.parametrize("context_edge_update", [False, True])
def test_context_module_can_return_skip_without_decoder_skip(context_edge_update):
    torch.manual_seed(0)
    batch_size, num_nodes, num_context_atoms, hidden_dim = 2, 5, 3, 8
    module = ContextModule(
        hidden_dim=hidden_dim,
        dropout_p=0.0,
        num_processor_layers=1,
        num_aggregator_layers=1,
        context_edge_update=context_edge_update,
        return_context_skip=True,
    )

    h_V = torch.randn(batch_size, num_nodes, hidden_dim)
    h_E = torch.randn(batch_size, num_nodes, 4, hidden_dim)
    V = torch.randn(batch_size, num_nodes, num_context_atoms, hidden_dim)
    Y_nodes = torch.randn(batch_size, num_nodes, num_context_atoms, hidden_dim)
    Y_edges = torch.randn(batch_size, num_nodes, num_context_atoms, num_context_atoms, hidden_dim)
    Y_m = torch.ones(batch_size, num_nodes, num_context_atoms)
    protein_residue_node_mask = torch.ones(batch_size, num_nodes)

    out, h_V_C_skip = module(
        h_V=h_V,
        h_E=h_E,
        V=V,
        Y_nodes=Y_nodes,
        Y_edges=Y_edges,
        Y_m=Y_m,
        protein_residue_node_mask=protein_residue_node_mask,
    )

    assert out.shape == h_V.shape
    assert h_V_C_skip is not None
    assert h_V_C_skip.shape == h_V.shape
