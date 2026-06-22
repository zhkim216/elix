from typing import Any

from omegaconf import DictConfig, OmegaConf


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
