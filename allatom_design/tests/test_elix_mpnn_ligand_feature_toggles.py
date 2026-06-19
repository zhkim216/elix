import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from allatom_design.data.transform.custom_transforms import _encode_atom_chirality_tags
from allatom_design.model.seq_denoiser.denoisers.seq_design.elix_mpnn import ElixMPNN
from allatom_design.model.seq_denoiser.denoisers.seq_design.tokenfeatures import TokenFeatures


def _make_elix_mpnn_cfg(**token_feature_overrides):
    token_features = {
        "num_rbf": 4,
        "min_rbf_mean": 2.0,
        "max_rbf_mean": 22.0,
        "protein_graph_rbf_type": "ca",
        "k_neighbors": 4,
        "num_positional_embeddings": 4,
        "hidden_dim": 8,
        "protein_ligand_interaction_rbf_type": "cb",
        "add_angle_features": False,
        "use_multichain_encoding": True,
        "ligand_conditioning": True,
        "use_ligand_context": True,
        "ligand_atom_context_num": 3,
        "masked_distance_fill": 1000.0,
    }
    token_features.update(token_feature_overrides)
    return OmegaConf.create(
        {
            "hidden_dim": 8,
            "k_neighbors": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dropout_p": 0.0,
            "ligand_conditioning": True,
            "use_context_skip_connection": False,
            "lmpnn_module": {
                "num_context_feature_processor_layers": 1,
                "num_context_feature_aggregator_layers": 1,
                "context_edge_update": False,
            },
            "token_features": token_features,
            "potts": {
                "k_neighbors_potts": 4,
                "max_dist_potts": None,
                "use_potts": False,
                "norm_potts_inputs": False,
                "parameterization": "factor",
                "num_factors": None,
                "symmetric_J": False,
                "multi_head": {
                    "dim_multi_head": 8,
                    "reduce": "mean",
                    "num_heads": 1,
                },
            },
            "expansion_mode": None,
        }
    )


def _make_token_features_cfg(protein_graph_rbf_type: str):
    return OmegaConf.create(
        {
            "num_rbf": 4,
            "min_rbf_mean": 2.0,
            "max_rbf_mean": 22.0,
            "protein_graph_rbf_type": protein_graph_rbf_type,
            "k_neighbors": 4,
            "num_positional_embeddings": 4,
            "hidden_dim": 8,
            "ligand_conditioning": False,
            "masked_distance_fill": 1000.0,
        }
    )


def _make_token_features_batch():
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    return {
        "noised_coords": coords,
        "token_to_center_atom": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "protein_residue_node_mask": torch.ones(1, 4),
        "residue_index": torch.arange(4).unsqueeze(0),
        "asym_id": torch.zeros(1, 4, dtype=torch.long),
        "noised_ca_coords": coords.clone(),
        "noised_n_coords": coords + torch.tensor([[[0.0, 0.1, 0.0]]]),
        "noised_c_coords": coords + torch.tensor([[[0.1, 0.0, 0.0]]]),
        "noised_o_coords": coords + torch.tensor([[[0.0, 0.0, 0.1]]]),
        "noised_pseudo_cb_coords": coords + torch.tensor([[[0.1, 0.1, 0.0]]]),
    }


@pytest.mark.parametrize("protein_graph_rbf_type", ["ca", "ncacocb"])
def test_token_features_do_not_require_pocket_rbf_mask(protein_graph_rbf_type):
    token_features = TokenFeatures(_make_token_features_cfg(protein_graph_rbf_type))
    batch = _make_token_features_batch()

    assert "pocket_rbf_mask" not in batch

    E, E_idx, V, Y_nodes, Y_edges, Y_m, D_neighbors = token_features(batch)

    assert E.shape == (1, 4, 4, 8)
    assert E_idx.shape == (1, 4, 4)
    assert D_neighbors.shape == (1, 4, 4)
    assert V is None
    assert Y_nodes is None
    assert Y_edges is None
    assert Y_m is None


def test_token_features_reject_invalid_protein_graph_rbf_type():
    with pytest.raises(ValueError, match="Invalid protein_graph_rbf_type"):
        TokenFeatures(_make_token_features_cfg("pocket_rbf"))


def test_default_ligand_feature_toggles_preserve_existing_projection_shapes():
    model = ElixMPNN(_make_elix_mpnn_cfg())
    token_features = model.token_features
    state_shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    assert token_features.type_linear.in_features == 147
    assert token_features.y_nodes.in_features == 147
    assert token_features.y_edges.in_features == token_features.num_rbf
    assert token_features.ligand_formal_charge_linear is None
    assert token_features.ligand_aromatic_atom_linear is None
    assert token_features.ligand_chirality_tag_linear is None
    assert token_features.ligand_aromatic_edge_linear is None
    assert state_shapes["token_features.type_linear.weight"] == (64, 147)
    assert state_shapes["token_features.type_linear.bias"] == (64,)
    assert state_shapes["token_features.y_nodes.weight"] == (8, 147)
    assert state_shapes["token_features.y_edges.weight"] == (8, 4)


def test_enabled_ligand_features_use_named_additive_zero_initialized_projections():
    model = ElixMPNN(
        _make_elix_mpnn_cfg(
            use_ligand_formal_charge=True,
            use_ligand_aromatic_atom_feature=True,
            use_ligand_aromatic_edge_feature=True,
            use_ligand_chirality_tag=True,
        )
    )
    token_features = model.token_features

    assert token_features.type_linear.in_features == 147
    assert token_features.y_nodes.in_features == 147
    assert token_features.y_edges.in_features == token_features.num_rbf
    assert not hasattr(token_features, "ligand_atom_interaction_feature_linear")
    assert not hasattr(token_features, "ligand_atom_node_feature_linear")
    assert not hasattr(token_features, "ligand_edge_feature_linear")

    assert token_features.ligand_formal_charge_linear.in_features == 1
    assert token_features.ligand_formal_charge_linear.out_features == 8
    assert token_features.ligand_formal_charge_linear.bias is None
    assert token_features.ligand_aromatic_atom_linear.in_features == 1
    assert token_features.ligand_aromatic_atom_linear.out_features == 8
    assert token_features.ligand_aromatic_atom_linear.bias is None
    assert token_features.ligand_chirality_tag_linear.in_features == 3
    assert token_features.ligand_chirality_tag_linear.out_features == 8
    assert token_features.ligand_chirality_tag_linear.bias is None
    assert token_features.ligand_aromatic_edge_linear.in_features == 1
    assert token_features.ligand_aromatic_edge_linear.out_features == 8
    assert token_features.ligand_aromatic_edge_linear.bias is None

    assert torch.count_nonzero(token_features.ligand_formal_charge_linear.weight) == 0
    assert torch.count_nonzero(token_features.ligand_aromatic_atom_linear.weight) == 0
    assert torch.count_nonzero(token_features.ligand_chirality_tag_linear.weight) == 0
    assert torch.count_nonzero(token_features.ligand_aromatic_edge_linear.weight) == 0


def test_atom_chirality_tags_encode_known_stereo_and_unknown_as_zero():
    stereo_values = np.array(["N", "R", "S", "?", "", "r", None, b"S"], dtype=object)

    np.testing.assert_array_equal(
        _encode_atom_chirality_tags(stereo_values),
        np.array([0, 1, 2, 0, 0, 1, 0, 2], dtype=np.int64),
    )
