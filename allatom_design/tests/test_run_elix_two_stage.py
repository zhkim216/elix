from pathlib import Path

import pytest
from omegaconf import OmegaConf

from allatom_design.eval.sampling.run_elix import (
    _copy_stage_design_cfg,
    _stage_cfg,
    _two_stage_regions,
)
from allatom_design.eval.utils.constraint_utils import resolve_pocket_annotation_method
from allatom_design.eval.utils import design_sequence as design_sequence_utils
from allatom_design.eval.utils import design_sequence_two_stage as two_stage_utils


def test_format_designed_sample_id_default_and_twostage_prefix() -> None:
    assert (
        design_sequence_utils._format_designed_sample_id(
            example_id="orig",
            sample_idx=0,
        )
        == "orig_sample0"
    )
    assert (
        design_sequence_utils._format_designed_sample_id(
            example_id="orig",
            sample_idx=1,
            tag="gamma0.50",
        )
        == "orig_gamma0.50_sample1"
    )
    assert (
        design_sequence_utils._format_designed_sample_id(
            example_id="orig",
            sample_idx=0,
            sample_token_prefix="twostage_ps_p",
        )
        == "orig_twostage_ps_psample0"
    )
    assert (
        design_sequence_utils._format_designed_sample_id(
            example_id="orig_twostage_ps_psample0",
            sample_idx=0,
            sample_token_prefix="s",
        )
        == "orig_twostage_ps_psample0_ssample0"
    )


def test_two_stage_direction_mapping() -> None:
    assert _two_stage_regions("ps") == ("pocket", "scaffold", "pocket")
    assert _two_stage_regions("sp") == ("scaffold", "pocket", "scaffold")


def test_stage_cfg_requires_checkpoint_source() -> None:
    cfg = OmegaConf.create({
        "pocket": {
            "model_cfg": {"model_name": "elix_mpnn", "denoiser_train_dir": None},
            "sampling_cfg": {"base_cfg_path": "/base.yaml"},
        }
    })

    with pytest.raises(ValueError, match="two_stage.pocket.model_cfg.denoiser_train_dir"):
        _stage_cfg(cfg, "pocket")


def test_resolve_pocket_annotation_method_prefers_enum_and_keeps_boolean_fallback() -> None:
    assert resolve_pocket_annotation_method(None, False) == "all_atom"
    assert resolve_pocket_annotation_method(None, True) == "calpha"
    assert resolve_pocket_annotation_method("pseudocb", True) == "pseudocb"
    with pytest.raises(ValueError, match="pocket_annotation_method"):
        resolve_pocket_annotation_method("cbeta", False)


def test_copy_stage_design_cfg_sets_sampling_prefix_and_resolves_interpolation() -> None:
    cfg = OmegaConf.create({
        "model_cfg": {"model_name": "elix_mpnn"},
        "sampling_cfg": {
            "base_cfg_path": "/base.yaml",
            "overrides": {"num_seqs_per_pdb": 2},
            "guidance": {"enabled": False},
        },
        "two_stage": {
            "pocket": {
                "model_cfg": {"model_name": "elix_mpnn", "denoiser_train_dir": "/pocket"},
                "sampling_cfg": {
                    "base_cfg_path": "${sampling_cfg.base_cfg_path}",
                    "overrides": "${sampling_cfg.overrides}",
                    "guidance": "${sampling_cfg.guidance}",
                },
            }
        },
    })

    stage_cfg = _copy_stage_design_cfg(
        cfg,
        cfg.two_stage.pocket,
        sample_token_prefix="twostage_ps_p",
    )

    assert stage_cfg.model_cfg.denoiser_train_dir == "/pocket"
    assert stage_cfg.sampling_cfg.base_cfg_path == "/base.yaml"
    assert stage_cfg.sampling_cfg.overrides.num_seqs_per_pdb == 2
    assert stage_cfg.sampling_cfg.overrides.sample_token_prefix == "twostage_ps_p"


def test_build_stage2_inputs_and_constraints_uses_stage1_sample_ids(monkeypatch) -> None:
    captured_kwargs = []

    def fake_create_pos_constraint_dict_from_pocket(**kwargs):
        captured_kwargs.append(kwargs)
        return {
            "pdb_key": kwargs["pdb_key"],
            "fixed_pos_seq": "A1-3",
            "fixed_pos_scn": None,
            "pocket_distance": kwargs["pocket_distance"],
            "constraint_type": kwargs["constraint_type"],
            "num_constrained_residues": 3,
        }, {}

    monkeypatch.setattr(
        two_stage_utils,
        "create_pos_constraint_dict_from_pocket",
        fake_create_pos_constraint_dict_from_pocket,
    )

    stage1_sample_dict = {
        "orig": {
            "input_sample_path": "/inputs/orig.cif",
            "designed_sample_id": ["orig_twostage_ps_psample0"],
            "designed_sample_path": ["/stage1/orig_twostage_ps_psample0.cif"],
            "designed_sample_atom_array": [object()],
            "designed_sample_seq": ["AAA"],
        }
    }
    stage2_sample_dict, pos_constraint_df, lineage = (
        two_stage_utils._build_stage2_inputs_and_constraints(
            stage1_sample_dict_per_ckpt=stage1_sample_dict,
            stage1_ckpt_info={
                "ckpt_path": "/ckpts/stage1.ckpt",
                "global_step": 10,
                "epoch": 2,
            },
            stage1_log_dir_per_ckpt=Path("/stage1/step_10_epoch_2"),
            stage1_region="pocket",
            stage2_region="scaffold",
            stage2_constraint_type="pocket",
            stage1_model_label="pocket",
            pocket_distance=5.0,
            pocket_annotation_method="pseudocb",
            use_calpha_for_pocket_annotation=False,
        )
    )

    assert list(stage2_sample_dict) == ["orig_twostage_ps_psample0"]
    assert stage2_sample_dict["orig_twostage_ps_psample0"]["input_sample_path"] == (
        "/stage1/orig_twostage_ps_psample0.cif"
    )
    assert pos_constraint_df["pdb_key"].tolist() == ["orig_twostage_ps_psample0"]
    assert pos_constraint_df["fixed_pos_seq"].tolist() == ["A1-3"]
    assert lineage["orig_twostage_ps_psample0"]["original_sample_id"] == "orig"
    assert captured_kwargs[0]["pocket_annotation_method"] == "pseudocb"
    assert lineage["orig_twostage_ps_psample0"]["pocket_annotation_method"] == "pseudocb"
