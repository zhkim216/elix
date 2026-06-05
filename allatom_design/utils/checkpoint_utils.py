import copy
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf


_ELIX_MPNN_CONFIG_RENAMES = (
    ("debug_lc_atom_mpnn_refactored", "debug_elix_mpnn"),
    ("lc_atom_mpnn_refactored", "elix_mpnn"),
    ("debug_lc_atom_mpnn", "debug_elix_mpnn"),
    ("lc_atom_mpnn_inference", "elix_mpnn_inference"),
    ("atom_mpnn_inference", "elix_mpnn_inference"),
    ("lc_atom_mpnn", "elix_mpnn"),
    ("atom_mpnn", "elix_mpnn"),
)


def _elix_mpnn_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _elix_mpnn_config_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_elix_mpnn_config_value(item) for item in value]
    if isinstance(value, str):
        for old, new in _ELIX_MPNN_CONFIG_RENAMES:
            value = value.replace(old, new)
    return value


def elix_mpnn_config(cfg: DictConfig | ListConfig | dict[str, Any]) -> DictConfig | ListConfig | dict[str, Any]:
    """
    Normalize saved AtomMPNN config references to ElixMPNN.

    Source configs should use the new ElixMPNN names directly.
    """
    if isinstance(cfg, (DictConfig, ListConfig)):
        cfg_container = OmegaConf.to_container(cfg, resolve=False)
        return OmegaConf.create(_elix_mpnn_config_value(cfg_container))
    return _elix_mpnn_config_value(copy.deepcopy(cfg))


def elix_mpnn_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Rename old checkpoint parameter keys for the denoiser module attribute.
    """
    return {
        src_key.replace(".denoiser.atom_mpnn.", ".denoiser.elix_mpnn."): value
        for src_key, value in state_dict.items()
    }


def get_cfg_from_ckpt(
    ckpt_path: str,
    return_as_dict: bool = False,
) -> tuple[DictConfig | dict[str, Any], dict[str, Any]]:
    """
    Load the config directly from the cfg arg passed into the model during training.

    Also returns the model checkpoint.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = elix_mpnn_config(ckpt["hyper_parameters"]["cfg"])
    cfg = OmegaConf.create(cfg_dict)

    if return_as_dict:
        return cfg_dict, ckpt
    return cfg, ckpt


def repair_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Repair the state dict to avoid issues with loading the model checkpoint due to torch.compile().

    https://github.com/pytorch/pytorch/issues/101107
    """
    repaired_state_dict = {
        src_key.replace("_orig_mod.", ""): value
        for src_key, value in state_dict.items()
    }
    return elix_mpnn_state_dict(repaired_state_dict)
