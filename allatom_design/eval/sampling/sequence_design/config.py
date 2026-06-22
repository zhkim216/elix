from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.config import require_cfg_value


def resolve_sampling_cfg(cfg: DictConfig) -> DictConfig:
    sampling_cfg = require_cfg_value(cfg, "sampling_cfg", "sequence design config")
    sampling_cfg_path = sampling_cfg.get("base_cfg_path", None)
    if sampling_cfg_path is None:
        raise ValueError("sampling_cfg.base_cfg_path is required")

    resolved_sampling_cfg = OmegaConf.load(str(sampling_cfg_path))
    overrides = sampling_cfg.get("overrides", {})
    if overrides is not None:
        if isinstance(overrides, DictConfig):
            overrides = OmegaConf.to_container(overrides, resolve=True)
        else:
            overrides = dict(overrides)
        resolved_sampling_cfg = OmegaConf.merge(resolved_sampling_cfg, overrides)
    return resolved_sampling_cfg


def resolve_input_cfgs(cfg: DictConfig) -> tuple[DictConfig, DictConfig]:
    if cfg.input_sample_is_designed:
        return cfg.cif_cfg.parse.designed_samples, cfg.preprocess_cfg.designed_samples
    return cfg.cif_cfg.parse.native, cfg.preprocess_cfg.native


def get_stage2_potts_only_cond(stage2_design_cfg: DictConfig) -> Any:
    return OmegaConf.select(
        stage2_design_cfg,
        "sampling_cfg.overrides.potts_sampling_cfg.potts_only_cond",
        default=None,
    )
