import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.pocket_constraints import (
    create_pos_constraint_dict_from_pocket,
    resolve_pocket_annotation_method,
)
from allatom_design.eval.sampling.sequence_design.constraints import (
    RUNTIME_POS_CONSTRAINT_COLUMNS,
)
from allatom_design.eval.sampling.sequence_design.config import get_stage2_potts_only_cond
from allatom_design.eval.sampling.sequence_design.checkpoints import ckpt_label
from allatom_design.eval.sampling.sequence_design.core import (
    SequenceDesignRunSpec,
    iter_design_sequence_for_run_spec,
)


@dataclass(frozen=True)
class TwoStageDesignContext:
    direction: str
    stage1_region: str
    stage2_region: str
    stage2_constraint_type: str
    stage1_design_cfg: DictConfig
    stage2_design_cfg: DictConfig
    stage1_model_label: str
    stage2_model_label: str
    pocket_distance: float
    pocket_annotation_method: str | None
    use_calpha_for_pocket_annotation: bool

    @property
    def stage1_guidance_cfg(self) -> DictConfig | None:
        return self.stage1_design_cfg.sampling_cfg.get("guidance", None)

    @property
    def stage2_guidance_cfg(self) -> DictConfig | None:
        return self.stage2_design_cfg.sampling_cfg.get("guidance", None)


@dataclass(frozen=True)
class _Stage2DesignArtifacts:
    sample_dict: dict
    pos_constraint_df: pd.DataFrame
    lineage_by_stage1_sample_id: dict[str, dict[str, Any]]
    base_log_dir: Path
    constraint_csv: Path


def _two_stage_regions(direction: str) -> tuple[str, str, str]:
    if direction == "ps":
        return "pocket", "scaffold", "pocket"
    if direction == "sp":
        return "scaffold", "pocket", "scaffold"
    raise ValueError(f"two_stage.direction must be 'ps' or 'sp', got {direction!r}")


def _stage_cfg(two_stage_cfg: DictConfig, region: str) -> DictConfig:
    cfg = two_stage_cfg.get(region, None)
    if cfg is None:
        raise ValueError(f"two_stage.{region} is required")
    model_cfg = cfg.get("model_cfg", None)
    if model_cfg is None:
        raise ValueError(f"two_stage.{region}.model_cfg is required")
    if cfg.get("sampling_cfg", None) is None:
        raise ValueError(f"two_stage.{region}.sampling_cfg is required")
    if model_cfg.get("model_name", None) is None:
        raise ValueError(f"two_stage.{region}.model_cfg.model_name is required")
    if model_cfg.get("ckpt_path", None) is None:
        if model_cfg.get("denoiser_train_dir", None) is None:
            raise ValueError(
                f"two_stage.{region}.model_cfg.denoiser_train_dir is required "
                f"unless two_stage.{region}.model_cfg.ckpt_path is set"
            )
        if model_cfg.get("ckpt_cfg", None) is None:
            raise ValueError(
                f"two_stage.{region}.model_cfg.ckpt_cfg is required "
                f"unless two_stage.{region}.model_cfg.ckpt_path is set"
            )
    return cfg


def _stage_label(stage_cfg: DictConfig, region: str) -> str:
    label = stage_cfg.get("label", None)
    if label is not None:
        return str(label)
    train_dir = stage_cfg.model_cfg.get("denoiser_train_dir", None)
    if train_dir is None:
        return region
    return Path(str(train_dir)).name


def _copy_stage_design_cfg(cfg: DictConfig, stage_cfg: DictConfig, sample_token_prefix: str) -> DictConfig:
    design_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(design_cfg, False)
    design_cfg.model_cfg = OmegaConf.create(OmegaConf.to_container(stage_cfg.model_cfg, resolve=True))
    design_cfg.sampling_cfg = OmegaConf.create(OmegaConf.to_container(stage_cfg.sampling_cfg, resolve=True))
    if design_cfg.sampling_cfg.get("overrides", None) is None:
        design_cfg.sampling_cfg.overrides = {}
    design_cfg.sampling_cfg.overrides.sample_token_prefix = sample_token_prefix
    return design_cfg


def build_two_stage_design_context(cfg: DictConfig) -> TwoStageDesignContext:
    """Resolve two-stage direction, per-stage configs, labels, and pocket policy."""
    two_stage_cfg = cfg.get("two_stage", None)
    if two_stage_cfg is None or not two_stage_cfg.get("enabled", False):
        raise ValueError("two_stage.enabled is required for two-stage design")
    if cfg.get("pos_constraint_csv", None) is not None:
        raise ValueError("two_stage mode always runs stage1 as free design; pos_constraint_csv must be null")

    direction = two_stage_cfg.get("direction", None)
    stage1_region, stage2_region, stage2_constraint_type = _two_stage_regions(direction)
    stage1_cfg = _stage_cfg(two_stage_cfg, stage1_region)
    stage2_cfg = _stage_cfg(two_stage_cfg, stage2_region)
    stage1_label = _stage_label(stage1_cfg, stage1_region)
    stage2_label = _stage_label(stage2_cfg, stage2_region)

    stage1_token_prefix = f"twostage_{direction}_{stage1_region[0]}"
    stage2_token_prefix = stage2_region[0]
    stage1_design_cfg = _copy_stage_design_cfg(cfg, stage1_cfg, stage1_token_prefix)
    stage2_design_cfg = _copy_stage_design_cfg(cfg, stage2_cfg, stage2_token_prefix)
    use_calpha_for_pocket_annotation = bool(
        two_stage_cfg.get("use_calpha_for_pocket_annotation", False)
    )
    pocket_annotation_method = resolve_pocket_annotation_method(
        pocket_annotation_method=two_stage_cfg.get("pocket_annotation_method", None),
        use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
    )

    return TwoStageDesignContext(
        direction=direction,
        stage1_region=stage1_region,
        stage2_region=stage2_region,
        stage2_constraint_type=stage2_constraint_type,
        stage1_design_cfg=stage1_design_cfg,
        stage2_design_cfg=stage2_design_cfg,
        stage1_model_label=stage1_label,
        stage2_model_label=stage2_label,
        pocket_distance=two_stage_cfg.get("pocket_distance", 5.0),
        pocket_annotation_method=pocket_annotation_method,
        use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
    )


def _build_stage2_inputs_and_constraints(
    *,
    stage1_sample_dict_per_ckpt: dict,
    stage1_ckpt_info: dict[str, Any],
    stage1_log_dir_per_ckpt: Path,
    stage1_region: str,
    stage2_region: str,
    stage2_constraint_type: str,
    stage1_model_label: str,
    pocket_distance: float,
    pocket_annotation_method: str | None,
    use_calpha_for_pocket_annotation: bool,
) -> tuple[dict, pd.DataFrame, dict[str, dict[str, Any]]]:
    stage2_sample_dict: dict[str, dict[str, Any]] = {}
    constraint_rows = []
    lineage_by_stage1_sample_id: dict[str, dict[str, Any]] = {}

    for original_sample_id, stage1_entry in stage1_sample_dict_per_ckpt.items():
        designed_sample_ids = stage1_entry.get("designed_sample_id", [])
        designed_sample_paths = stage1_entry.get("designed_sample_path", [])
        designed_atom_arrays = stage1_entry.get("designed_sample_atom_array", [])
        designed_sample_seqs = stage1_entry.get("designed_sample_seq", [])

        for sample_idx, stage1_sample_id in enumerate(designed_sample_ids):
            stage1_sample_path = designed_sample_paths[sample_idx]
            stage1_atom_array = designed_atom_arrays[sample_idx]
            stage1_sample_seq = (
                designed_sample_seqs[sample_idx]
                if sample_idx < len(designed_sample_seqs)
                else None
            )

            constraint_row, _ = create_pos_constraint_dict_from_pocket(
                pdb_key=stage1_sample_id,
                atom_array=stage1_atom_array,
                pocket_distance=pocket_distance,
                constraint_type=stage2_constraint_type,
                pocket_annotation_method=pocket_annotation_method,
                use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
                sample_path=stage1_sample_path,
                return_ligand_mpnn_format=False,
            )
            constraint_rows.append(constraint_row)

            stage2_sample_dict[stage1_sample_id] = {
                "input_sample_path": stage1_sample_path,
                "input_sample_id": stage1_sample_id,
            }
            if "pdb_chain_info" in stage1_entry:
                stage2_sample_dict[stage1_sample_id]["pdb_chain_info"] = copy.deepcopy(
                    stage1_entry["pdb_chain_info"]
                )
            lineage_by_stage1_sample_id[stage1_sample_id] = {
                "stage1_region": stage1_region,
                "stage2_region": stage2_region,
                "original_sample_id": original_sample_id,
                "original_sample_path": stage1_entry.get("input_sample_path"),
                "stage1_model_label": stage1_model_label,
                "stage1_step": stage1_ckpt_info["global_step"],
                "stage1_epoch": stage1_ckpt_info["epoch"],
                "stage1_checkpoint_path": stage1_ckpt_info["ckpt_path"],
                "stage1_checkpoint_dir": str(stage1_log_dir_per_ckpt),
                "stage1_sample_id": stage1_sample_id,
                "stage1_sample_path": stage1_sample_path,
                "stage1_sample_seq": stage1_sample_seq,
                "fixed_region": stage2_constraint_type,
                "pocket_distance": pocket_distance,
                "pocket_annotation_method": pocket_annotation_method,
            }

    if len(stage2_sample_dict) == 0:
        raise ValueError("Stage 1 produced no samples for stage 2")

    constraint_df = pd.DataFrame(constraint_rows)
    missing_columns = [
        col for col in RUNTIME_POS_CONSTRAINT_COLUMNS if col not in constraint_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Generated stage2 constraints missing columns: {missing_columns}")
    pos_constraint_df = constraint_df.loc[:, list(RUNTIME_POS_CONSTRAINT_COLUMNS)]
    if pos_constraint_df["pdb_key"].duplicated().any():
        duplicated = pos_constraint_df[pos_constraint_df["pdb_key"].duplicated()]["pdb_key"].tolist()
        raise ValueError(f"Duplicate stage2 constraint pdb_key values: {duplicated[:5]}")
    if set(stage2_sample_dict.keys()) != set(pos_constraint_df["pdb_key"]):
        raise ValueError("Stage2 sample IDs must exactly match generated constraint pdb_key values")

    return stage2_sample_dict, pos_constraint_df, lineage_by_stage1_sample_id


def _build_twostage_manifest_rows(
    *,
    direction: str,
    lineage_by_stage1_sample_id: dict[str, dict[str, Any]],
    stage2_sample_dict_per_ckpt: dict,
    stage2_ckpt_info: dict[str, Any],
    stage2_log_dir_per_ckpt: Path,
    stage2_model_label: str,
    stage2_constraint_csv: Path,
    stage2_design_cfg: DictConfig,
) -> list[dict[str, Any]]:
    rows = []
    potts_only_cond = get_stage2_potts_only_cond(stage2_design_cfg)

    for stage1_sample_id, stage2_entry in stage2_sample_dict_per_ckpt.items():
        lineage = lineage_by_stage1_sample_id[stage1_sample_id]
        for sample_idx, stage2_sample_id in enumerate(stage2_entry.get("designed_sample_id", [])):
            rows.append({
                **lineage,
                "direction": direction,
                "stage2_model_label": stage2_model_label,
                "stage2_step": stage2_ckpt_info["global_step"],
                "stage2_epoch": stage2_ckpt_info["epoch"],
                "stage2_checkpoint_path": stage2_ckpt_info["ckpt_path"],
                "stage2_checkpoint_dir": str(stage2_log_dir_per_ckpt),
                "stage2_sample_id": stage2_sample_id,
                "stage2_sample_path": stage2_entry["designed_sample_path"][sample_idx],
                "stage2_sample_seq": stage2_entry["designed_sample_seq"][sample_idx],
                "constraint_csv": str(stage2_constraint_csv),
                "potts_only_cond": potts_only_cond,
            })

    return rows


def _stage2_base_log_dir(
    *,
    log_dir: Path,
    direction: str,
    stage1_region: str,
    stage2_region: str,
    stage1_model_label: str,
    stage1_ckpt_info: dict[str, Any],
    stage2_model_label: str,
) -> Path:
    stage1_label = f"{stage1_model_label}_{ckpt_label(stage1_ckpt_info)}"
    return (
        log_dir
        / direction
        / f"stage2_{stage2_region}"
        / f"from_{stage1_region}_{stage1_label}"
        / stage2_model_label
    )


def _stage2_constraint_csv_path(
    *,
    stage2_base_log_dir: Path,
    stage2_constraint_type: str,
    pocket_distance: float,
    csv_suffix: str,
) -> Path:
    return (
        stage2_base_log_dir
        / f"pos_constraints_{stage2_constraint_type}_{pocket_distance}A{csv_suffix}.csv"
    )


def _prepare_stage2_design_artifacts(
    *,
    stage1_sample_dict_per_ckpt: dict,
    stage1_ckpt_info: dict[str, Any],
    stage1_log_dir_per_ckpt: Path,
    log_dir: Path,
    direction: str,
    stage1_region: str,
    stage2_region: str,
    stage2_constraint_type: str,
    stage1_model_label: str,
    stage2_model_label: str,
    pocket_distance: float,
    pocket_annotation_method: str | None,
    use_calpha_for_pocket_annotation: bool,
    csv_suffix: str,
) -> _Stage2DesignArtifacts:
    stage2_sample_dict, stage2_pos_constraint_df, lineage_by_stage1_sample_id = (
        _build_stage2_inputs_and_constraints(
            stage1_sample_dict_per_ckpt=stage1_sample_dict_per_ckpt,
            stage1_ckpt_info=stage1_ckpt_info,
            stage1_log_dir_per_ckpt=stage1_log_dir_per_ckpt,
            stage1_region=stage1_region,
            stage2_region=stage2_region,
            stage2_constraint_type=stage2_constraint_type,
            stage1_model_label=stage1_model_label,
            pocket_distance=pocket_distance,
            pocket_annotation_method=pocket_annotation_method,
            use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
        )
    )
    stage2_base_log_dir = _stage2_base_log_dir(
        log_dir=log_dir,
        direction=direction,
        stage1_region=stage1_region,
        stage2_region=stage2_region,
        stage1_model_label=stage1_model_label,
        stage1_ckpt_info=stage1_ckpt_info,
        stage2_model_label=stage2_model_label,
    )
    stage2_base_log_dir.mkdir(parents=True, exist_ok=True)
    stage2_constraint_csv = _stage2_constraint_csv_path(
        stage2_base_log_dir=stage2_base_log_dir,
        stage2_constraint_type=stage2_constraint_type,
        pocket_distance=pocket_distance,
        csv_suffix=csv_suffix,
    )
    stage2_pos_constraint_df.to_csv(stage2_constraint_csv, index=False)

    return _Stage2DesignArtifacts(
        sample_dict=stage2_sample_dict,
        pos_constraint_df=stage2_pos_constraint_df,
        lineage_by_stage1_sample_id=lineage_by_stage1_sample_id,
        base_log_dir=stage2_base_log_dir,
        constraint_csv=stage2_constraint_csv,
    )


def _build_stage2_run_spec(
    *,
    seed: int,
    stage2_artifacts: _Stage2DesignArtifacts,
    stage2_design_cfg: DictConfig,
    stage2_cif_parse_cfg: DictConfig | None,
    stage2_preprocess_cfg: DictConfig | None,
    featurizer_cfg: DictConfig | None,
    cif_save_cfg: DictConfig | None,
    protein_only: bool,
    pocket_distances_for_seq_recovery: list[float] | None,
    pocket_distance_bins: list[tuple[float, float]] | None,
    pocket_n_min_ligand_atoms_for_seq_recovery: int,
    csv_suffix: str,
    stage2_guidance_cfg: DictConfig | None,
) -> SequenceDesignRunSpec:
    return SequenceDesignRunSpec(
        seed=seed,
        input_sample_is_designed=True,
        sample_dict=stage2_artifacts.sample_dict,
        design_cfg=stage2_design_cfg,
        cif_parse_cfg=stage2_cif_parse_cfg,
        preprocess_cfg=stage2_preprocess_cfg,
        featurizer_cfg=featurizer_cfg,
        cif_save_cfg=cif_save_cfg,
        sampling_inputs_df=None,
        selectivity_residue_alignment_df=None,
        log_dir=stage2_artifacts.base_log_dir,
        pos_constraint_df=stage2_artifacts.pos_constraint_df,
        protein_only=protein_only,
        pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
        pocket_distance_bins=pocket_distance_bins,
        pocket_n_min_ligand_atoms_for_seq_recovery=pocket_n_min_ligand_atoms_for_seq_recovery,
        csv_suffix=csv_suffix,
        guidance_cfg=stage2_guidance_cfg,
    )


def design_sequence_two_stage(
    *,
    seed: int = 0,
    direction: str,
    input_sample_is_designed: bool = False,
    sample_dict: dict | None = None,
    stage1_design_cfg: DictConfig | None = None,
    stage2_design_cfg: DictConfig | None = None,
    stage1_region: str,
    stage2_region: str,
    stage2_constraint_type: str,
    stage1_model_label: str,
    stage2_model_label: str,
    stage1_cif_parse_cfg: DictConfig | None = None,
    stage1_preprocess_cfg: DictConfig | None = None,
    stage2_cif_parse_cfg: DictConfig | None = None,
    stage2_preprocess_cfg: DictConfig | None = None,
    featurizer_cfg: DictConfig | None = None,
    cif_save_cfg: DictConfig | None = None,
    stage1_sampling_inputs_df: pd.DataFrame | None = None,
    log_dir: Path | None = None,
    protein_only: bool = False,
    pocket_distances_for_seq_recovery: list[float] | None = None,
    pocket_distance_bins: list[tuple[float, float]] | None = None,
    pocket_n_min_ligand_atoms_for_seq_recovery: int = 5,
    csv_suffix: str = "",
    stage1_guidance_cfg: DictConfig | None = None,
    stage2_guidance_cfg: DictConfig | None = None,
    pocket_distance: float = 5.0,
    pocket_annotation_method: str | None = None,
    use_calpha_for_pocket_annotation: bool = False,
) -> Iterator[tuple[dict, Path, dict, list[dict[str, Any]]]]:
    if sample_dict is None:
        raise ValueError("sample_dict must be provided")
    if stage1_design_cfg is None:
        raise ValueError("stage1_design_cfg must be provided")
    if stage2_design_cfg is None:
        raise ValueError("stage2_design_cfg must be provided")
    if log_dir is None:
        raise ValueError("log_dir must be provided")
    pocket_annotation_method = resolve_pocket_annotation_method(
        pocket_annotation_method=pocket_annotation_method,
        use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
    )

    stage1_log_dir = log_dir / direction / f"stage1_{stage1_region}" / stage1_model_label
    stage1_run_spec = SequenceDesignRunSpec(
        seed=seed,
        input_sample_is_designed=input_sample_is_designed,
        sample_dict=sample_dict,
        design_cfg=stage1_design_cfg,
        cif_parse_cfg=stage1_cif_parse_cfg,
        preprocess_cfg=stage1_preprocess_cfg,
        featurizer_cfg=featurizer_cfg,
        cif_save_cfg=cif_save_cfg,
        sampling_inputs_df=stage1_sampling_inputs_df,
        selectivity_residue_alignment_df=None,
        log_dir=stage1_log_dir,
        pos_constraint_df=None,
        protein_only=protein_only,
        pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
        pocket_distance_bins=pocket_distance_bins,
        pocket_n_min_ligand_atoms_for_seq_recovery=pocket_n_min_ligand_atoms_for_seq_recovery,
        csv_suffix=csv_suffix,
        guidance_cfg=stage1_guidance_cfg,
    )
    stage1_iter = iter_design_sequence_for_run_spec(stage1_run_spec)

    for stage1_sample_dict_per_ckpt, stage1_log_dir_per_ckpt, stage1_ckpt_info in stage1_iter:
        stage2_artifacts = _prepare_stage2_design_artifacts(
            stage1_sample_dict_per_ckpt=stage1_sample_dict_per_ckpt,
            stage1_ckpt_info=stage1_ckpt_info,
            stage1_log_dir_per_ckpt=stage1_log_dir_per_ckpt,
            log_dir=log_dir,
            direction=direction,
            stage1_region=stage1_region,
            stage2_region=stage2_region,
            stage2_constraint_type=stage2_constraint_type,
            stage1_model_label=stage1_model_label,
            stage2_model_label=stage2_model_label,
            pocket_distance=pocket_distance,
            pocket_annotation_method=pocket_annotation_method,
            use_calpha_for_pocket_annotation=use_calpha_for_pocket_annotation,
            csv_suffix=csv_suffix,
        )

        stage2_run_spec = _build_stage2_run_spec(
            seed=seed,
            stage2_artifacts=stage2_artifacts,
            stage2_design_cfg=stage2_design_cfg,
            stage2_cif_parse_cfg=stage2_cif_parse_cfg,
            stage2_preprocess_cfg=stage2_preprocess_cfg,
            featurizer_cfg=featurizer_cfg,
            cif_save_cfg=cif_save_cfg,
            protein_only=protein_only,
            pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
            pocket_distance_bins=pocket_distance_bins,
            pocket_n_min_ligand_atoms_for_seq_recovery=pocket_n_min_ligand_atoms_for_seq_recovery,
            csv_suffix=csv_suffix,
            stage2_guidance_cfg=stage2_guidance_cfg,
        )
        stage2_iter = iter_design_sequence_for_run_spec(stage2_run_spec)

        for stage2_sample_dict_per_ckpt, stage2_log_dir_per_ckpt, stage2_ckpt_info in stage2_iter:
            manifest_rows = _build_twostage_manifest_rows(
                direction=direction,
                lineage_by_stage1_sample_id=stage2_artifacts.lineage_by_stage1_sample_id,
                stage2_sample_dict_per_ckpt=stage2_sample_dict_per_ckpt,
                stage2_ckpt_info=stage2_ckpt_info,
                stage2_log_dir_per_ckpt=stage2_log_dir_per_ckpt,
                stage2_model_label=stage2_model_label,
                stage2_constraint_csv=stage2_artifacts.constraint_csv,
                stage2_design_cfg=stage2_design_cfg,
            )
            yield stage2_sample_dict_per_ckpt, stage2_log_dir_per_ckpt, stage2_ckpt_info, manifest_rows
