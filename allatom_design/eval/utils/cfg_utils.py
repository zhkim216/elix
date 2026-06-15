from omegaconf import DictConfig, OmegaConf
from typing import Any

def _select_cfg_value(cfg: DictConfig | dict | None, key: str, default=None):
    if cfg is None:
        return default
    selectable_cfg = cfg if isinstance(cfg, DictConfig) else OmegaConf.create(cfg)
    return OmegaConf.select(selectable_cfg, key, default=default)

def require_cfg_value(cfg: DictConfig | dict, key: str, owner: str) -> Any:
    if cfg is None or key not in cfg or cfg.get(key) is None:
        raise ValueError(f"{owner}.{key} is required")
    return cfg.get(key)

def get_config_value(config: dict | DictConfig | None, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)

def get_json_config_value(json_config: dict | DictConfig, *keys: str, default=None):
    for key in keys:
        if hasattr(json_config, "get"):
            value = json_config.get(key, None)
            if value is not None:
                return value
    return default

def config_value_as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)

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
    return _select_cfg_value(
        stage2_design_cfg,
        "sampling_cfg.overrides.potts_sampling_cfg.potts_only_cond",
        default=None,
    )

def guidance_is_enabled(cfg: DictConfig | dict | None) -> bool:
    direct_enabled = _select_cfg_value(cfg, "enabled", default=None)
    if direct_enabled is not None:
        return config_value_as_bool(direct_enabled)
    return config_value_as_bool(
        _select_cfg_value(cfg, "sampling_cfg.guidance.enabled", default=False)
    )
