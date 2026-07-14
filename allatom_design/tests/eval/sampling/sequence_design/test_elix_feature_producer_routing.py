from types import SimpleNamespace

from omegaconf import OmegaConf

from allatom_design.eval.sampling.sequence_design.inputs import (
    featurizer_cfg_with_model_feature_producers,
)


def _model(*, cached_chirality: bool, bond_order: bool, hydrogenbond: bool):
    token_features = SimpleNamespace(
        use_ligand_cached_rdkit_chirality=cached_chirality,
        use_ligand_bond_order=bond_order,
        add_hydrogenbond_feature=hydrogenbond,
    )
    return SimpleNamespace(
        denoiser=SimpleNamespace(
            elix_mpnn=SimpleNamespace(token_features=token_features),
        )
    )


def test_design_featurizer_producers_follow_loaded_model_and_keep_runtime_cache_path() -> None:
    original = OmegaConf.create(
        {
            "residue_cache_dir": "/runtime/cache",
            "use_ligand_cached_rdkit_chirality": False,
            "use_ligand_bond_order": False,
            "add_hydrogenbond_feature": False,
        }
    )

    routed = featurizer_cfg_with_model_feature_producers(
        original,
        _model(cached_chirality=True, bond_order=True, hydrogenbond=True),
    )

    assert routed.residue_cache_dir == "/runtime/cache"
    assert routed.use_ligand_cached_rdkit_chirality is True
    assert routed.use_ligand_bond_order is True
    assert routed.add_hydrogenbond_feature is True
    assert original.use_ligand_cached_rdkit_chirality is False
    assert original.use_ligand_bond_order is False
    assert original.add_hydrogenbond_feature is False


def test_design_featurizer_disables_unused_optional_producers() -> None:
    routed = featurizer_cfg_with_model_feature_producers(
        {
            "residue_cache_dir": "/runtime/cache",
            "use_ligand_cached_rdkit_chirality": True,
            "use_ligand_bond_order": True,
            "add_hydrogenbond_feature": True,
        },
        _model(cached_chirality=False, bond_order=False, hydrogenbond=False),
    )

    assert routed.use_ligand_cached_rdkit_chirality is False
    assert routed.use_ligand_bond_order is False
    assert routed.add_hydrogenbond_feature is False


def test_design_featurizer_routing_preserves_legacy_or_mock_calls() -> None:
    cfg = {"residue_cache_dir": "/runtime/cache"}

    assert featurizer_cfg_with_model_feature_producers(None, object()) is None
    assert featurizer_cfg_with_model_feature_producers(cfg, object()) is cfg


def test_design_featurizer_routing_resolves_parent_interpolations_before_copy() -> None:
    root = OmegaConf.create(
        {
            "cache_root": "/runtime/cache",
            "design": {"residue_cache_dir": "${cache_root}"},
        }
    )

    routed = featurizer_cfg_with_model_feature_producers(
        root.design,
        _model(cached_chirality=True, bond_order=False, hydrogenbond=True),
    )

    assert routed.residue_cache_dir == "/runtime/cache"
    assert routed.use_ligand_cached_rdkit_chirality is True
    assert routed.use_ligand_bond_order is False
    assert routed.add_hydrogenbond_feature is True
