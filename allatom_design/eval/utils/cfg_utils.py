from omegaconf import DictConfig, OmegaConf
from typing import Any

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
    sampling_cfg = stage2_design_cfg.get("sampling_cfg", None)
    if sampling_cfg is None:
        return None
    overrides = sampling_cfg.get("overrides", None)
    if overrides is None:
        return None
    potts_cfg = overrides.get("potts_sampling_cfg", None)
    if potts_cfg is None:
        return None
    return potts_cfg.get("potts_only_cond", None)

def guidance_is_enabled(cfg: DictConfig | dict | None) -> bool:
    if cfg is None:
        return False

    if cfg.get("enabled", None) is not None:
        guidance_cfg = cfg
    else:
        sampling_cfg = cfg.get("sampling_cfg", None)
        if sampling_cfg is None:
            return False
        guidance_cfg = sampling_cfg.get("guidance", None)
        if guidance_cfg is None:
            return False

    return guidance_cfg.get("enabled", False)
