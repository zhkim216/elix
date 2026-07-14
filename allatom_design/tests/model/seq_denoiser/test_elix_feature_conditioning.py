import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from allatom_design.data.const import PERIODIC_TABLE_FEATURES
from allatom_design.model.seq_denoiser.denoisers.seq_design.elix_mpnn import ElixMPNN
from allatom_design.model.seq_denoiser.denoisers.seq_design.tokenfeatures import (
    TokenFeatures,
    build_f_block_features,
    gather_dense_pair_features,
)
from allatom_design.utils.checkpoint_utils import (
    migrate_elix_feature_projection_state_dict,
)


def _token_feature_cfg(**overrides):
    cfg = {
        "masked_distance_fill": 1000.0,
        "k_neighbors": 2,
        "num_positional_embeddings": 4,
        "hidden_dim": 4,
        "num_rbf": 2,
        "min_rbf_mean": 2.0,
        "max_rbf_mean": 22.0,
        "protein_graph_rbf_type": "ca",
        "use_multichain_encoding": True,
        "ligand_conditioning": True,
        "use_ligand_context": True,
        "ligand_atom_context_num": 4,
        "protein_ligand_interaction_rbf_type": "cb",
        "add_angle_features": False,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _token_features(**overrides) -> TokenFeatures:
    return TokenFeatures(_token_feature_cfg(**overrides))


def _elix_mpnn(**token_feature_overrides) -> ElixMPNN:
    return ElixMPNN(
        OmegaConf.create(
            {
                "hidden_dim": 4,
                "k_neighbors": 2,
                "num_encoder_layers": 1,
                "num_decoder_layers": 1,
                "use_mpnn_decoder": True,
                "use_potts_encoding": False,
                "dropout_p": 0.0,
                "ligand_conditioning": True,
                "use_context_skip_connection": True,
                "lmpnn_module": {
                    "num_context_feature_processor_layers": 1,
                    "num_context_feature_aggregator_layers": 1,
                    "context_edge_update": False,
                },
                "token_features": _token_feature_cfg(**token_feature_overrides),
                "potts": {"use_potts": False},
                "sidechain_prediction": {"enabled": False},
            }
        )
    )


def test_legacy_group_period_and_f_block_features_are_chemically_indexed() -> None:
    model = _token_features(use_ligand_f_block_features=True)
    atomic_numbers = torch.tensor([[[56, 57, 71, 72, 89, 103, 104]]])

    legacy, _, f_block = model._embed_ligand_atom_types(atomic_numbers)

    assert legacy.shape[-1] == 147
    expected_group = torch.tensor(PERIODIC_TABLE_FEATURES[1])[atomic_numbers]
    expected_period = torch.tensor(PERIODIC_TABLE_FEATURES[2])[atomic_numbers]
    torch.testing.assert_close(legacy[..., 120:139].argmax(-1), expected_group)
    torch.testing.assert_close(legacy[..., 139:147].argmax(-1), expected_period)
    torch.testing.assert_close(f_block, build_f_block_features(atomic_numbers))

    assert f_block[0, 0, 0].sum().item() == 0.0  # Ba
    assert f_block[0, 0, 1, 0].item() == 1.0  # La flag
    assert f_block[0, 0, 1, 2].item() == 1.0  # La subgroup 0
    assert f_block[0, 0, 2, 16].item() == 1.0  # Lu subgroup 14
    assert f_block[0, 0, 3].sum().item() == 0.0  # Hf
    assert f_block[0, 0, 4, 1].item() == 1.0  # Ac flag
    assert f_block[0, 0, 4, 17].item() == 1.0  # Ac subgroup 0
    assert f_block[0, 0, 5, 31].item() == 1.0  # Lr subgroup 14
    assert f_block[0, 0, 6].sum().item() == 0.0  # Rf


def test_legacy_periodic_table_features_match_representative_chemistry() -> None:
    atomic_numbers = torch.tensor([[[1, 2, 6, 26, 57, 71, 89, 103, 118]]])
    model = _token_features()

    legacy, _, _ = model._embed_ligand_atom_types(atomic_numbers)

    expected_groups = torch.tensor([[[1, 18, 14, 8, 3, 3, 3, 3, 18]]])
    expected_periods = torch.tensor([[[1, 1, 2, 4, 6, 6, 7, 7, 7]]])
    torch.testing.assert_close(legacy[..., 120:139].argmax(-1), expected_groups)
    torch.testing.assert_close(legacy[..., 139:147].argmax(-1), expected_periods)


def test_asinh_charge_and_cached_chirality_v2_feed_nodes_and_interactions() -> None:
    model = _token_features(
        use_ligand_f_block_features=True,
        use_ligand_asinh_formal_charge=True,
        use_ligand_cached_rdkit_chirality=True,
    )
    atom_features, _ = model._build_ligand_atom_features(
        {
            "atom_formal_charge": torch.tensor([[2.0, -3.0, 0.0, 1.0]]),
            "atom_cached_rdkit_chirality_tag": torch.tensor([[0, 1, 2, 3]]),
            "atom_cached_rdkit_chirality_mask": torch.tensor([[1, 1, 1, 0]]),
        },
        ligand_mask=torch.ones(1, 4),
    )
    torch.testing.assert_close(
        atom_features["cached_rdkit_chirality"],
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        ),
    )
    gathered = model._gather_ligand_atom_features(
        ligand_atom_features=atom_features,
        Y_idx=torch.tensor([[[0, 1, 2, 3]]]),
        B=1,
        N=1,
        device=torch.device("cpu"),
    )

    model.norm_y_nodes = nn.Identity()
    model.norm_y_edges = nn.Identity()
    nn.init.zeros_(model.y_nodes.weight)
    nn.init.zeros_(model.y_edges.weight)
    nn.init.zeros_(model.ligand_f_block_node_linear.weight)
    nn.init.zeros_(model.ligand_asinh_formal_charge_linear.weight)
    nn.init.zeros_(model.ligand_cached_rdkit_chirality_v2_node_linear.weight)
    model.ligand_f_block_node_linear.weight.data[0, 0] = 2.0
    model.ligand_asinh_formal_charge_linear.weight.data[1, 0] = 3.0
    model.ligand_cached_rdkit_chirality_v2_node_linear.weight.data[2, 0] = 5.0
    model.ligand_cached_rdkit_chirality_v2_node_linear.weight.data[3, 2] = 7.0

    atomic_numbers = torch.tensor([[[57, 26, 26, 26]]])
    legacy, _, f_block = model._embed_ligand_atom_types(atomic_numbers)
    y_nodes, _ = model._embed_ligand_subgraph_features(
        Y=torch.zeros(1, 1, 4, 3),
        Y_m=torch.ones(1, 1, 4, dtype=torch.long),
        Y_t_features=legacy,
        Y_f_block_features=f_block,
        Y_atom_features=gathered,
        Y_aromatic=None,
        Y_bond_order=None,
        Y_token_bond_edges=None,
    )

    expected = torch.zeros_like(y_nodes)
    expected[0, 0, 0] = torch.tensor([2.0, 3.0 * torch.asinh(torch.tensor(2.0)), 5.0, 0.0])
    expected[0, 0, 1, 1] = 3.0 * torch.asinh(torch.tensor(-3.0))
    expected[0, 0, 2, 3] = 7.0
    expected[0, 0, 3, 1] = 3.0 * torch.asinh(torch.tensor(1.0))
    torch.testing.assert_close(y_nodes, expected)

    model.norm_nodes = nn.Identity()
    nn.init.zeros_(model.node_project_down.weight)
    nn.init.zeros_(model.node_project_down.bias)
    nn.init.zeros_(model.ligand_f_block_interaction_linear.weight)
    nn.init.zeros_(model.ligand_asinh_formal_charge_interaction_linear.weight)
    nn.init.zeros_(model.ligand_cached_rdkit_chirality_v2_interaction_linear.weight)
    model.ligand_asinh_formal_charge_interaction_linear.weight.data[0, 0] = 11.0
    model.ligand_cached_rdkit_chirality_v2_interaction_linear.weight.data[1, 2] = 13.0
    model.node_project_down.weight.data[0, 2] = 1.0
    model.node_project_down.weight.data[1, 3] = 1.0
    interaction = model._embed_ligand_interaction_features(
        Y=torch.zeros(1, 1, 4, 3),
        Y_t_embedded=torch.zeros(1, 1, 4, 64),
        Y_f_block_features=f_block,
        Y_atom_features=gathered,
        Y_token_bond_interactions=None,
        noised_backbone_pseudo_cb_coords=torch.zeros(1, 1, 5, 3),
    )
    expected_interaction = torch.zeros_like(interaction)
    expected_interaction[0, 0, :, 0] = 11.0 * torch.asinh(
        torch.tensor([2.0, -3.0, 0.0, 1.0])
    )
    expected_interaction[0, 0, 2, 1] = 13.0
    # The invalid final S tag would project to 13 without the post-projection gate.
    torch.testing.assert_close(interaction, expected_interaction)


def test_hydrogenbond_features_keep_hba_hbd_order_and_feed_nodes_and_interactions() -> None:
    model = _token_features(add_hydrogenbond_feature=True)
    ligand_features, _ = model._build_ligand_atom_features(
        {
            "atom_is_HBA": torch.tensor([[1, 0, 1, 0]]),
            "atom_is_HBD": torch.tensor([[0, 1, 1, 1]]),
            "atom_hydrogenbond_feature_mask": torch.tensor([[1, 1, 0, 1]]),
        },
        ligand_mask=torch.tensor([[1, 1, 1, 0]]),
    )
    torch.testing.assert_close(
        ligand_features["hydrogenbond"],
        torch.tensor(
            [
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        ),
    )

    gathered = model._gather_ligand_atom_features(
        ligand_atom_features=ligand_features,
        Y_idx=torch.tensor([[[0, 1, 2, 3]]]),
        B=1,
        N=1,
        device=torch.device("cpu"),
    )
    model.norm_y_nodes = nn.Identity()
    model.norm_y_edges = nn.Identity()
    nn.init.zeros_(model.y_nodes.weight)
    nn.init.zeros_(model.y_edges.weight)
    nn.init.zeros_(model.ligand_hydrogenbond_node_linear.weight)
    model.ligand_hydrogenbond_node_linear.weight.data[0, 0] = 2.0
    model.ligand_hydrogenbond_node_linear.weight.data[1, 1] = 3.0

    y_nodes, _ = model._embed_ligand_subgraph_features(
        Y=torch.zeros(1, 1, 4, 3),
        Y_m=torch.ones(1, 1, 4, dtype=torch.long),
        Y_t_features=torch.zeros(1, 1, 4, 147),
        Y_f_block_features=torch.zeros(1, 1, 4, 32),
        Y_atom_features=gathered,
        Y_aromatic=None,
        Y_bond_order=None,
        Y_token_bond_edges=None,
    )
    torch.testing.assert_close(
        y_nodes[0, 0],
        torch.tensor(
            [[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0] * 4, [0.0] * 4]
        ),
    )

    model.norm_nodes = nn.Identity()
    nn.init.zeros_(model.node_project_down.weight)
    nn.init.zeros_(model.node_project_down.bias)
    nn.init.zeros_(model.ligand_hydrogenbond_interaction_linear.weight)
    model.ligand_hydrogenbond_interaction_linear.weight.data[0, 0] = 5.0
    model.ligand_hydrogenbond_interaction_linear.weight.data[1, 1] = 7.0
    # With two RBF channels, atom-type channels 0 and 1 occupy concat columns 2 and 3.
    model.node_project_down.weight.data[0, 2] = 1.0
    model.node_project_down.weight.data[1, 3] = 1.0
    interaction = model._embed_ligand_interaction_features(
        Y=torch.zeros(1, 1, 4, 3),
        Y_t_embedded=torch.zeros(1, 1, 4, 64),
        Y_f_block_features=torch.zeros(1, 1, 4, 32),
        Y_atom_features=gathered,
        Y_token_bond_interactions=None,
        noised_backbone_pseudo_cb_coords=torch.zeros(1, 1, 5, 3),
    )
    torch.testing.assert_close(
        interaction[0, 0],
        torch.tensor(
            [[5.0, 0.0, 0.0, 0.0], [0.0, 7.0, 0.0, 0.0], [0.0] * 4, [0.0] * 4]
        ),
    )


def test_bond_order_keeps_only_small_molecule_metal_or_nucleic_acid_pairs() -> None:
    model = _token_features(use_ligand_bond_order=True)
    bond_order = torch.zeros(1, 4, 4, dtype=torch.long)
    bond_order[0, 0, 1] = bond_order[0, 1, 0] = 2
    bond_order[0, 0, 2] = bond_order[0, 2, 0] = 1
    bond_order[0, 1, 3] = bond_order[0, 3, 1] = 4
    batch = {
        "atom_ligand_bond_order": bond_order,
        "atom_pad_mask": torch.ones(1, 4),
        "atom_is_small_molecule_chain": torch.tensor([[1, 0, 0, 0]]),
        "atom_is_metal_chain": torch.zeros(1, 4),
        "atom_is_nucleic_acid_chain": torch.tensor([[0, 1, 0, 0]]),
    }

    features = model._build_ligand_bond_order_features(
        batch=batch,
        Y_idx=torch.tensor([[[0, 1, 2, 3]]]),
        Y_m=torch.ones(1, 1, 4, dtype=torch.long),
    )

    assert features.shape == (1, 1, 4, 4, 5)
    assert features[0, 0, 0, 1, 1].item() == 1.0
    assert features[0, 0, 1, 0, 1].item() == 1.0
    assert features[0, 0, 0, 2].sum().item() == 0.0
    assert features[0, 0, 1, 3].sum().item() == 0.0
    assert features[0, 0, 0, 0].sum().item() == 0.0


def test_token_bonds_feed_context_edges_and_masked_protein_interactions() -> None:
    model = _token_features(use_token_bonds=True, ligand_atom_context_num=3)
    token_bonds = torch.zeros(1, 4, 4)
    token_bonds[0, 0, 2] = token_bonds[0, 2, 0] = 1
    token_bonds[0, 2, 3] = token_bonds[0, 3, 2] = 1
    token_bonds[0, 1, 3] = token_bonds[0, 3, 1] = 1
    batch = {
        "token_bonds": token_bonds,
        "atom_to_token_map": torch.tensor([[2, 3, 1]]),
    }
    y_idx = torch.tensor([[[0, 1, 2], [0, 1, 2]]])
    y_m = torch.tensor([[[1, 1, 1], [1, 1, 0]]])

    edge_bonds, interaction_bonds = model._build_token_bond_features(
        batch=batch,
        Y_idx=y_idx,
        Y_m=y_m,
        protein_residue_node_mask=torch.tensor([[1, 0]]),
    )

    assert edge_bonds[0, 0, 0, 1, 0].item() == 1.0  # token 2 <-> token 3
    assert edge_bonds[0, 0, 1, 2, 0].item() == 1.0  # token 3 <-> token 1
    assert edge_bonds[0, 1, :, 2].sum().item() == 0.0  # padded context endpoint
    assert interaction_bonds[0, 0, :, 0].tolist() == [1.0, 0.0, 0.0]
    assert interaction_bonds[0, 1].sum().item() == 0.0  # non-protein current token


def test_new_pair_and_interaction_projections_enter_the_existing_paths() -> None:
    pair_model = _token_features(use_ligand_bond_order=True, use_token_bonds=True)
    pair_model.norm_y_nodes = nn.Identity()
    pair_model.norm_y_edges = nn.Identity()
    nn.init.zeros_(pair_model.y_nodes.weight)
    nn.init.zeros_(pair_model.y_edges.weight)
    nn.init.zeros_(pair_model.ligand_bond_order_linear.weight)
    nn.init.zeros_(pair_model.token_bond_edge_linear.weight)
    pair_model.ligand_bond_order_linear.weight.data[0, 1] = 2.0
    pair_model.token_bond_edge_linear.weight.data[3, 0] = 3.0
    bond_order = torch.zeros(1, 1, 2, 2, 5)
    bond_order[0, 0, 0, 1, 1] = 1.0
    token_bonds = torch.zeros(1, 1, 2, 2, 1)
    token_bonds[0, 0, 0, 1, 0] = 1.0

    _, y_edges = pair_model._embed_ligand_subgraph_features(
        Y=torch.zeros(1, 1, 2, 3),
        Y_m=torch.ones(1, 1, 2, dtype=torch.long),
        Y_t_features=torch.zeros(1, 1, 2, 147),
        Y_f_block_features=torch.zeros(1, 1, 2, 32),
        Y_atom_features={},
        Y_aromatic=None,
        Y_bond_order=bond_order,
        Y_token_bond_edges=token_bonds,
    )
    torch.testing.assert_close(
        y_edges[0, 0, 0, 1],
        torch.tensor([2.0, 0.0, 0.0, 3.0]),
    )

    interaction_model = _token_features(
        use_ligand_f_block_features=True,
        use_token_bonds=True,
    )
    interaction_model.norm_nodes = nn.Identity()
    nn.init.zeros_(interaction_model.node_project_down.weight)
    nn.init.zeros_(interaction_model.node_project_down.bias)
    nn.init.zeros_(interaction_model.ligand_f_block_interaction_linear.weight)
    nn.init.zeros_(interaction_model.token_bond_interaction_linear.weight)
    interaction_model.ligand_f_block_interaction_linear.weight.data[0, 0] = 4.0
    # The first atom-type channel follows the two RBF channels in the concat.
    interaction_model.node_project_down.weight.data[0, 2] = 1.0
    interaction_model.token_bond_interaction_linear.weight.data[1, 0] = 5.0
    f_block = torch.zeros(1, 1, 1, 32)
    f_block[..., 0] = 1.0

    interaction = interaction_model._embed_ligand_interaction_features(
        Y=torch.zeros(1, 1, 1, 3),
        Y_t_embedded=torch.zeros(1, 1, 1, 64),
        Y_f_block_features=f_block,
        Y_atom_features={},
        Y_token_bond_interactions=torch.ones(1, 1, 1, 1),
        noised_backbone_pseudo_cb_coords=torch.zeros(1, 1, 5, 3),
    )
    torch.testing.assert_close(
        interaction[0, 0, 0],
        torch.tensor([4.0, 5.0, 0.0, 0.0]),
    )


def test_enabled_new_features_require_their_producer_keys_and_off_paths_do_not() -> None:
    charge_model = _token_features(use_ligand_asinh_formal_charge=True)
    with pytest.raises(KeyError, match="atom_formal_charge"):
        charge_model._build_ligand_atom_features({}, torch.ones(1, 1))

    chirality_model = _token_features(use_ligand_cached_rdkit_chirality=True)
    with pytest.raises(KeyError, match="atom_cached_rdkit_chirality_tag"):
        chirality_model._build_ligand_atom_features({}, torch.ones(1, 1))
    with pytest.raises(KeyError, match="atom_cached_rdkit_chirality_mask"):
        chirality_model._build_ligand_atom_features(
            {"atom_cached_rdkit_chirality_tag": torch.zeros(1, 1)},
            torch.ones(1, 1),
        )
    masked_chirality, _ = chirality_model._build_ligand_atom_features(
        {
            "atom_cached_rdkit_chirality_tag": torch.tensor([[3]]),
            "atom_cached_rdkit_chirality_mask": torch.zeros(1, 1),
        },
        torch.ones(1, 1),
    )
    torch.testing.assert_close(
        masked_chirality["cached_rdkit_chirality"],
        torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
    )

    hydrogenbond_model = _token_features(add_hydrogenbond_feature=True)
    with pytest.raises(KeyError, match="add_hydrogenbond_feature=true.*atom_is_HBA"):
        hydrogenbond_model._build_ligand_atom_features({}, torch.ones(1, 1))
    with pytest.raises(KeyError, match="add_hydrogenbond_feature=true.*atom_is_HBD"):
        hydrogenbond_model._build_ligand_atom_features(
            {"atom_is_HBA": torch.zeros(1, 1)},
            torch.ones(1, 1),
        )
    with pytest.raises(
        KeyError,
        match="add_hydrogenbond_feature=true.*atom_hydrogenbond_feature_mask",
    ):
        hydrogenbond_model._build_ligand_atom_features(
            {
                "atom_is_HBA": torch.zeros(1, 1),
                "atom_is_HBD": torch.zeros(1, 1),
            },
            torch.ones(1, 1),
        )

    bond_model = _token_features(use_ligand_bond_order=True)
    with pytest.raises(KeyError, match="atom_ligand_bond_order"):
        bond_model._build_ligand_bond_order_features(
            {"atom_pad_mask": torch.ones(1, 1)},
            torch.zeros(1, 1, 1, dtype=torch.long),
            torch.ones(1, 1, 1),
        )

    token_bond_model = _token_features(use_token_bonds=True)
    with pytest.raises(KeyError, match="token_bonds"):
        token_bond_model._build_token_bond_features(
            {"atom_to_token_map": torch.zeros(1, 1, dtype=torch.long)},
            torch.zeros(1, 1, 1, dtype=torch.long),
            torch.ones(1, 1, 1),
            torch.ones(1, 1),
        )

    disabled = _token_features()
    disabled_atom_features, _ = disabled._build_ligand_atom_features({}, torch.ones(1, 1))
    assert "hydrogenbond" not in disabled_atom_features
    assert disabled._build_ligand_bond_order_features({}, None, None) is None
    assert disabled._build_token_bond_features({}, None, None, None) == (None, None)


def test_full_elix_uses_xavier_for_optional_features_and_keeps_context_skip_zero() -> None:
    model = _elix_mpnn(
        use_ligand_f_block_features=True,
        use_ligand_asinh_formal_charge=True,
        use_ligand_cached_rdkit_chirality=True,
        use_ligand_bond_order=True,
        use_token_bonds=True,
        add_hydrogenbond_feature=True,
    )

    for name in (
        "ligand_f_block_node_linear",
        "ligand_f_block_interaction_linear",
        "ligand_asinh_formal_charge_linear",
        "ligand_asinh_formal_charge_interaction_linear",
        "ligand_cached_rdkit_chirality_v2_node_linear",
        "ligand_cached_rdkit_chirality_v2_interaction_linear",
        "ligand_bond_order_linear",
        "token_bond_edge_linear",
        "token_bond_interaction_linear",
        "ligand_hydrogenbond_node_linear",
        "ligand_hydrogenbond_interaction_linear",
    ):
        projection = getattr(model.token_features, name)
        assert projection.bias is None
        assert torch.count_nonzero(projection.weight).item() > 0

    for decoder_layer in model.decoder_layers:
        assert torch.count_nonzero(decoder_layer.W_ctx.weight).item() == 0
        assert torch.count_nonzero(decoder_layer.W_ctx.bias).item() == 0


def test_asinh_charge_defaults_true_in_yaml_but_false_when_cfg_omits_it() -> None:
    omitted = _token_features()
    assert omitted.use_ligand_asinh_formal_charge is False
    assert omitted.ligand_asinh_formal_charge_linear is None
    assert omitted.ligand_asinh_formal_charge_interaction_linear is None
    assert not hasattr(omitted, "use_ligand_formal_charge")
    assert not hasattr(omitted, "ligand_formal_charge_linear")

    removed_raw_flag = _token_features(use_ligand_formal_charge=True)
    removed_features, _ = removed_raw_flag._build_ligand_atom_features(
        {},
        torch.ones(1, 1),
    )
    assert "formal_charge" not in removed_features

    for path in (
        "allatom_design/configs/seq_denoiser/denoiser/elix_mpnn.yaml",
        "allatom_design/configs_local/seq_denoiser/denoiser/debug_elix_mpnn.yaml",
    ):
        cfg = OmegaConf.load(path)
        assert cfg.mpnn.token_features.use_ligand_asinh_formal_charge is True
        assert "use_ligand_formal_charge" not in cfg.mpnn.token_features


def test_default_off_flags_preserve_the_legacy_parameter_set_and_initialization() -> None:
    torch.manual_seed(13)
    implicit_defaults = _token_features()
    torch.manual_seed(13)
    explicit_defaults = _token_features(
        use_ligand_f_block_features=False,
        use_ligand_asinh_formal_charge=False,
        use_ligand_cached_rdkit_chirality=False,
        use_ligand_bond_order=False,
        use_token_bonds=False,
        add_hydrogenbond_feature=False,
    )

    implicit_state = implicit_defaults.state_dict()
    explicit_state = explicit_defaults.state_dict()
    assert implicit_state.keys() == explicit_state.keys()
    for key in implicit_state:
        torch.testing.assert_close(implicit_state[key], explicit_state[key])


class _TinyTokenFeatures(nn.Module):
    def __init__(self, feature_shapes: dict[str, tuple[int, int]] | None = None):
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        for name, (input_dim, output_dim) in (feature_shapes or {}).items():
            setattr(self, name, nn.Linear(input_dim, output_dim, bias=False))


class _TinyElixMPNN(nn.Module):
    def __init__(self, feature_shapes: dict[str, tuple[int, int]] | None = None):
        super().__init__()
        self.token_features = _TinyTokenFeatures(feature_shapes)


class _TinyDenoiser(nn.Module):
    def __init__(self, feature_shapes: dict[str, tuple[int, int]] | None = None):
        super().__init__()
        self.elix_mpnn = _TinyElixMPNN(feature_shapes)


class _TinyModel(nn.Module):
    def __init__(self, feature_shapes: dict[str, tuple[int, int]] | None = None):
        super().__init__()
        self.denoiser = _TinyDenoiser(feature_shapes)


class _TinyLit(nn.Module):
    def __init__(self, feature_shapes: dict[str, tuple[int, int]] | None = None):
        super().__init__()
        self.model = _TinyModel(feature_shapes)


_CURRENT_OPTIONAL_PROJECTION_SHAPES = {
    "ligand_f_block_node_linear": (32, 2),
    "ligand_f_block_interaction_linear": (32, 64),
    "ligand_asinh_formal_charge_linear": (1, 2),
    "ligand_asinh_formal_charge_interaction_linear": (1, 64),
    "ligand_cached_rdkit_chirality_v2_node_linear": (4, 2),
    "ligand_cached_rdkit_chirality_v2_interaction_linear": (4, 64),
    "ligand_bond_order_linear": (5, 2),
    "token_bond_edge_linear": (1, 2),
    "token_bond_interaction_linear": (1, 2),
    "ligand_hydrogenbond_node_linear": (3, 2),
    "ligand_hydrogenbond_interaction_linear": (3, 64),
}


def test_checkpoint_migration_keeps_xavier_new_weights_and_stays_strict() -> None:
    torch.manual_seed(7)
    legacy_model = _TinyLit()
    legacy_state = legacy_model.state_dict()

    disabled_model = _TinyLit()
    enabled_model = _TinyLit(_CURRENT_OPTIONAL_PROJECTION_SHAPES)
    initialized_state = {
        key: value.clone()
        for key, value in enabled_model.state_dict().items()
    }
    disabled_model.load_state_dict(
        migrate_elix_feature_projection_state_dict(disabled_model, legacy_state),
        strict=True,
    )
    migrated = migrate_elix_feature_projection_state_dict(enabled_model, legacy_state)
    enabled_model.load_state_dict(migrated, strict=True)
    new_keys = set(enabled_model.state_dict()) - set(legacy_state)
    assert len(new_keys) == len(_CURRENT_OPTIONAL_PROJECTION_SHAPES)
    for key in new_keys:
        torch.testing.assert_close(migrated[key], initialized_state[key])
        assert torch.count_nonzero(migrated[key]).item() > 0

    compile_style_state = {
        key.replace("model.", "model._orig_mod.", 1): value
        for key, value in legacy_state.items()
    }
    disabled_model.load_state_dict(
        migrate_elix_feature_projection_state_dict(disabled_model, compile_style_state),
        strict=True,
    )

    missing_legacy = dict(legacy_state)
    missing_legacy.pop("model.denoiser.elix_mpnn.token_features.base.weight")
    with pytest.raises(RuntimeError, match="missing keys"):
        migrate_elix_feature_projection_state_dict(enabled_model, missing_legacy)

    unexpected_legacy = dict(legacy_state)
    unexpected_legacy["unrelated.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected keys"):
        migrate_elix_feature_projection_state_dict(enabled_model, unexpected_legacy)


def test_checkpoint_migration_rejects_old_feature_semantics_and_shape_changes() -> None:
    old_cached = _TinyLit(
        {"ligand_cached_rdkit_chirality_linear": (4, 2)}
    ).state_dict()
    current_cached = _TinyLit(
        {
            "ligand_cached_rdkit_chirality_v2_node_linear": (4, 2),
            "ligand_cached_rdkit_chirality_v2_interaction_linear": (4, 64),
        }
    )
    with pytest.raises(RuntimeError, match="unexpected keys.*ligand_cached_rdkit_chirality_linear"):
        migrate_elix_feature_projection_state_dict(current_cached, old_cached)

    old_hydrogenbond = _TinyLit(
        {
            "ligand_hydrogenbond_node_linear": (2, 2),
            "ligand_hydrogenbond_interaction_linear": (2, 64),
        }
    ).state_dict()
    current_hydrogenbond = _TinyLit(
        {
            "ligand_hydrogenbond_node_linear": (3, 2),
            "ligand_hydrogenbond_interaction_linear": (3, 64),
        }
    )
    with pytest.raises(RuntimeError, match="shape mismatches.*ligand_hydrogenbond"):
        migrate_elix_feature_projection_state_dict(
            current_hydrogenbond,
            old_hydrogenbond,
        )


def test_checkpoint_migration_accepts_asinh_node_only_and_compatible_bond_order() -> None:
    old_asinh = _TinyLit(
        {"ligand_asinh_formal_charge_linear": (1, 2)}
    ).state_dict()
    current_asinh = _TinyLit(
        {
            "ligand_asinh_formal_charge_linear": (1, 2),
            "ligand_asinh_formal_charge_interaction_linear": (1, 64),
        }
    )
    initialized_interaction = (
        current_asinh.model.denoiser.elix_mpnn.token_features
        .ligand_asinh_formal_charge_interaction_linear.weight.detach().clone()
    )
    migrated_asinh = migrate_elix_feature_projection_state_dict(
        current_asinh,
        old_asinh,
    )
    torch.testing.assert_close(
        migrated_asinh[
            "model.denoiser.elix_mpnn.token_features.ligand_asinh_formal_charge_linear.weight"
        ],
        old_asinh[
            "model.denoiser.elix_mpnn.token_features.ligand_asinh_formal_charge_linear.weight"
        ],
    )
    torch.testing.assert_close(
        migrated_asinh[
            "model.denoiser.elix_mpnn.token_features.ligand_asinh_formal_charge_interaction_linear.weight"
        ],
        initialized_interaction,
    )

    old_bond_order = _TinyLit(
        {"ligand_bond_order_linear": (5, 2)}
    ).state_dict()
    current_bond_order = _TinyLit(
        {"ligand_bond_order_linear": (5, 2)}
    )
    migrated_bond_order = migrate_elix_feature_projection_state_dict(
        current_bond_order,
        old_bond_order,
    )
    torch.testing.assert_close(
        migrated_bond_order[
            "model.denoiser.elix_mpnn.token_features.ligand_bond_order_linear.weight"
        ],
        old_bond_order[
            "model.denoiser.elix_mpnn.token_features.ligand_bond_order_linear.weight"
        ],
    )


def test_dense_pair_gather_uses_each_context_index_on_both_axes() -> None:
    pair_features = torch.arange(16).reshape(1, 4, 4)
    gathered = gather_dense_pair_features(
        pair_features,
        torch.tensor([[[0, 2]]]),
    )
    torch.testing.assert_close(gathered, torch.tensor([[[[0, 2], [8, 10]]]]))
