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

_NEW_ELIX_FEATURE_PROJECTION_KEYS = frozenset(
    {
        "model.denoiser.elix_mpnn.token_features.ligand_f_block_node_linear.weight",
        "model.denoiser.elix_mpnn.token_features.ligand_f_block_interaction_linear.weight",
        "model.denoiser.elix_mpnn.token_features.ligand_asinh_formal_charge_linear.weight",
        "model.denoiser.elix_mpnn.token_features.ligand_asinh_formal_charge_interaction_linear.weight",
        "model.denoiser.elix_mpnn.token_features.ligand_cached_rdkit_chirality_v2_node_linear.weight",
        "model.denoiser.elix_mpnn.token_features.ligand_cached_rdkit_chirality_v2_interaction_linear.weight",
        "model.denoiser.elix_mpnn.token_features.ligand_bond_order_linear.weight",
        "model.denoiser.elix_mpnn.token_features.token_bond_edge_linear.weight",
        "model.denoiser.elix_mpnn.token_features.token_bond_interaction_linear.weight",
        "model.denoiser.elix_mpnn.token_features.ligand_hydrogenbond_node_linear.weight",
        "model.denoiser.elix_mpnn.token_features.ligand_hydrogenbond_interaction_linear.weight",
    }
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


def migrate_elix_feature_projection_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Fill only newly enabled ELIX feature projections from current initialization.

    The returned mapping uses the current module's actual keys, including a
    possible ``_orig_mod`` segment introduced by ``torch.compile``. All other
    missing or unexpected state keys remain strict errors.
    """

    def normalized_key(key: str) -> str:
        return key.replace("_orig_mod.", "")

    def index_by_normalized_key(keys) -> dict[str, str]:
        indexed = {}
        for key in keys:
            normalized = normalized_key(key)
            if normalized in indexed:
                raise RuntimeError(
                    "State dict has duplicate keys after torch.compile normalization: "
                    f"{indexed[normalized]!r} and {key!r}"
                )
            indexed[normalized] = key
        return indexed

    state_dict = elix_mpnn_state_dict(state_dict)
    current_state = module.state_dict()
    current_keys = index_by_normalized_key(current_state)
    incoming_keys = index_by_normalized_key(state_dict)

    missing_keys = set(current_keys) - set(incoming_keys)
    unexpected_keys = set(incoming_keys) - set(current_keys)
    allowed_missing_keys = _NEW_ELIX_FEATURE_PROJECTION_KEYS & set(current_keys)
    disallowed_missing_keys = missing_keys - allowed_missing_keys
    shape_mismatches = {
        key: (
            tuple(state_dict[incoming_keys[key]].shape),
            tuple(current_state[current_keys[key]].shape),
        )
        for key in set(current_keys) & set(incoming_keys)
        if state_dict[incoming_keys[key]].shape
        != current_state[current_keys[key]].shape
    }

    if disallowed_missing_keys or unexpected_keys or shape_mismatches:
        error_parts = []
        if disallowed_missing_keys:
            error_parts.append(
                f"missing keys: {sorted(disallowed_missing_keys)}"
            )
        if unexpected_keys:
            error_parts.append(
                f"unexpected keys: {sorted(unexpected_keys)}"
            )
        if shape_mismatches:
            error_parts.append(
                "shape mismatches: "
                + ", ".join(
                    f"{key} incoming={incoming_shape} current={current_shape}"
                    for key, (incoming_shape, current_shape) in sorted(
                        shape_mismatches.items()
                    )
                )
            )
        raise RuntimeError(
            "ELIX feature checkpoint migration rejected incompatible state dict; "
            + "; ".join(error_parts)
        )

    migrated_state = {}
    for normalized, current_key in current_keys.items():
        if normalized in incoming_keys:
            migrated_state[current_key] = state_dict[incoming_keys[normalized]]
        else:
            migrated_state[current_key] = current_state[current_key]
    return migrated_state
