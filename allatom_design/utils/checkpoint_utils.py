import copy
import re
import warnings
from pathlib import Path
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

_STEP_TRAINING_CHECKPOINT_PATTERN = re.compile(
    r"^sd-step(?P<step>\d+)-epoch(?P<epoch>\d+)(?:-v(?P<version>\d+))?\.ckpt$"
)
_EPOCH_TRAINING_CHECKPOINT_PATTERN = re.compile(
    r"^sd-epoch(?P<epoch>\d+)(?:-v(?P<version>\d+))?\.ckpt$"
)
_FULL_TRAINING_CHECKPOINT_KEYS = {
    "epoch",
    "global_step",
    "loops",
    "lr_schedulers",
    "optimizer_states",
    "state_dict",
}

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


def _checkpoint_sort_key(path: Path, pattern: re.Pattern[str]) -> tuple[int, int, int]:
    match = pattern.fullmatch(path.name)
    if match is None:
        return (-1, -1, path.stat().st_mtime_ns)
    primary = int(match.groupdict().get("step") or match.group("epoch"))
    version = int(match.groupdict().get("version") or 0)
    return (primary, version, path.stat().st_mtime_ns)


def _latest_valid_training_checkpoint(
    candidates: list[Path],
    pattern: re.Pattern[str],
) -> tuple[Path, int] | None:
    for candidate in sorted(
        candidates,
        key=lambda path: _checkpoint_sort_key(path, pattern),
        reverse=True,
    ):
        try:
            match = pattern.fullmatch(candidate.name)
            if match is None:
                raise ValueError("filename does not match a full-training checkpoint pattern")

            checkpoint = torch.load(candidate, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, dict):
                raise TypeError(f"expected a checkpoint dict, got {type(checkpoint).__name__}")

            missing_keys = sorted(_FULL_TRAINING_CHECKPOINT_KEYS - checkpoint.keys())
            if missing_keys:
                raise ValueError(f"missing full-training keys: {missing_keys}")

            global_step = checkpoint["global_step"]
            if isinstance(global_step, bool) or not isinstance(global_step, int):
                raise TypeError(f"global_step must be an int, got {global_step!r}")

            filename_step = match.groupdict().get("step")
            if filename_step is not None and int(filename_step) != global_step:
                raise ValueError(
                    f"filename step {filename_step} does not match global_step {global_step}"
                )
        except Exception as exc:
            warnings.warn(
                f"Skipping invalid training checkpoint {candidate}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        return candidate, global_step

    return None


def select_latest_training_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the latest valid non-EMA checkpoint that can resume training.

    Step checkpoints and the epoch checkpoint are validated as full Lightning
    training checkpoints. Invalid newest files are skipped so an interrupted
    write can fall back to the previous valid checkpoint. EMA and Lightning HPC
    checkpoints are intentionally outside the accepted filename patterns.
    """
    checkpoint_dir = Path(checkpoint_dir)
    step_candidates = list(checkpoint_dir.glob("sd-step*-epoch*.ckpt"))
    epoch_candidates = list(checkpoint_dir.glob("sd-epoch*.ckpt"))
    if not step_candidates and not epoch_candidates:
        return None

    valid_candidates = [
        candidate
        for candidate in (
            _latest_valid_training_checkpoint(
                step_candidates,
                _STEP_TRAINING_CHECKPOINT_PATTERN,
            ),
            _latest_valid_training_checkpoint(
                epoch_candidates,
                _EPOCH_TRAINING_CHECKPOINT_PATTERN,
            ),
        )
        if candidate is not None
    ]
    if not valid_candidates:
        raise RuntimeError(
            f"No valid full-training checkpoint found in {checkpoint_dir}; "
            "refusing to start fresh over an existing failed run"
        )

    selected_path, _ = max(
        valid_candidates,
        key=lambda candidate: (candidate[1], candidate[0].stat().st_mtime_ns),
    )
    return selected_path


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
