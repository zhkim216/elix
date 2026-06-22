from __future__ import annotations

import glob
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import hydra
from joblib import Parallel
from omegaconf import DictConfig

from allatom_design.eval.config import require_cfg_value
from allatom_design.eval.sampling.sequence_design.config import resolve_sampling_cfg
from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.utils.checkpoint_utils import get_cfg_from_ckpt, repair_state_dict

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted


def get_checkpoints(design_cfg: DictConfig) -> list[dict[str, Any]]:
    """Get sequence-design model checkpoints from a direct path or train dir."""
    model_cfg = require_cfg_value(design_cfg, "model_cfg", "sequence design config")
    direct_ckpt = model_cfg.get("ckpt_path", None)

    prefix_map = {"atom_denoiser": "ad", "seq_denoiser": "sd"}
    model_name_to_type = {
        "elix_mpnn": "seq_denoiser",
        "atom_mpnn": "seq_denoiser",
        "lc_atom_mpnn": "seq_denoiser",
    }
    model_type = model_cfg.get("model_type", None)
    if model_type is None:
        model_type = model_name_to_type.get(model_cfg.get("model_name", None))
    prefix = prefix_map.get(model_type)
    if prefix is None:
        raise ValueError(
            f"Invalid model_cfg.model_type/model_name: "
            f"{model_cfg.get('model_type', None)}/{model_cfg.get('model_name', None)}. "
            "Expected model_type in ['atom_denoiser', 'seq_denoiser'] or a known "
            "sequence design model_name."
        )
    pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)(?:-ema\\d+\\.\\d+)?\\.ckpt$")

    if direct_ckpt is not None:
        match = pattern.search(Path(direct_ckpt).name)
        if match is None:
            raise ValueError(f"Unexpected checkpoint filename: {Path(direct_ckpt).name}")
        return [
            {
                "ckpt_path": str(direct_ckpt),
                "global_step": int(match.group(1)),
                "epoch": int(match.group(2)),
            }
        ]

    ckpt_cfg = model_cfg.get("ckpt_cfg", None)
    if ckpt_cfg is None:
        raise ValueError("model_cfg.ckpt_cfg is required when model_cfg.ckpt_path is null")

    ema_ckpt_dir = f"{model_cfg.denoiser_train_dir}/checkpoints/ema"
    non_ema_ckpt_dir = f"{model_cfg.denoiser_train_dir}/checkpoints"
    use_ema = ckpt_cfg.get("use_ema", model_cfg.get("use_ema", False))
    if use_ema:
        if Path(ema_ckpt_dir).exists():
            print(f"Using EMA checkpoints from {ema_ckpt_dir}")
            pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)-ema(\\d+\\.\\d+)\\.ckpt$")
            ckpts = glob.glob(f"{ema_ckpt_dir}/*.ckpt")
        else:
            print(f"Warning: No EMA checkpoints found in {ema_ckpt_dir}, using non-EMA checkpoints")
            pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)\\.ckpt$")
            ckpts = glob.glob(f"{non_ema_ckpt_dir}/*.ckpt")
    else:
        print(f"Using non-EMA checkpoints from {non_ema_ckpt_dir}")
        pattern = re.compile(f"{prefix}-step(\\d+)-epoch(\\d+)\\.ckpt$")
        ckpts = glob.glob(f"{non_ema_ckpt_dir}/*.ckpt")

    all_ckpt_infos = []
    for ckpt in natsorted(ckpts):
        match = pattern.search(Path(ckpt).name)
        if match is None:
            continue
        all_ckpt_infos.append(
            {
                "ckpt_path": ckpt,
                "global_step": int(match.group(1)),
                "epoch": int(match.group(2)),
            }
        )
    all_ckpt_infos.sort(
        key=lambda ckpt_info: (
            ckpt_info["global_step"],
            ckpt_info["epoch"],
            ckpt_info["ckpt_path"],
        )
    )

    if ckpt_cfg.start_step is not None or ckpt_cfg.end_step is not None:
        filtered_ckpt_infos = []
        for ckpt_info in all_ckpt_infos:
            global_step = ckpt_info["global_step"]
            if (
                (ckpt_cfg.start_step is None or global_step >= ckpt_cfg.start_step)
                and (ckpt_cfg.end_step is None or global_step <= ckpt_cfg.end_step)
            ):
                filtered_ckpt_infos.append(ckpt_info)
    else:
        filtered_ckpt_infos = all_ckpt_infos

    ckpt_infos = filtered_ckpt_infos[:: ckpt_cfg.eval_every_n_ckpts]

    selected_paths = {ckpt_info["ckpt_path"] for ckpt_info in ckpt_infos}
    if (
        ckpt_cfg.eval_last_ckpt
        and all_ckpt_infos
        and all_ckpt_infos[-1]["ckpt_path"] not in selected_paths
    ):
        ckpt_infos.append(all_ckpt_infos[-1])

    return ckpt_infos


def ckpt_label(ckpt_info: dict[str, Any]) -> str:
    """Return the canonical 'step_{global_step}_epoch_{epoch}' label for a checkpoint."""
    return f"step_{ckpt_info['global_step']}_epoch_{ckpt_info['epoch']}"


def load_seq_des_model(
    cfg: DictConfig | None = None,
    device: str | None = None,
    ckpt_path: str | None = None,
    load_sampling_cfg: bool = True,
) -> dict[str, Any]:
    """Load a sequence-design model plus resolved sampling config."""
    if cfg is None:
        raise ValueError("sequence design config is required")

    model_cfg = require_cfg_value(cfg, "model_cfg", "sequence design config")
    model_name = require_cfg_value(model_cfg, "model_name", "model_cfg")
    seq_des_model = {"model_name": model_name, "cfg": cfg, "device": device}

    resolved_ckpt_path = ckpt_path or model_cfg.get("ckpt_path", None)
    if resolved_ckpt_path is None:
        raise ValueError("model_cfg.ckpt_path is required unless ckpt_path is passed explicitly")

    ckpt_model_cfg, checkpoint = get_cfg_from_ckpt(str(resolved_ckpt_path))
    if ckpt_model_cfg.get("train", {}).get("compile", {}).get("compile_model", False):
        ckpt_model_cfg.train.compile.compile_model = False
    state_dict = repair_state_dict(checkpoint["state_dict"])
    for obsolete_key in ("model.bb_std", "model.bb_mean", "model.scn_mean", "model.scn_std"):
        state_dict.pop(obsolete_key, None)
    lit_sd_model = LitSeqDenoiser(ckpt_model_cfg).eval()
    lit_sd_model.load_state_dict(state_dict, strict=True)
    if device is not None:
        lit_sd_model = lit_sd_model.to(device)
    data_cfg = hydra.utils.instantiate(ckpt_model_cfg.data)

    seq_des_model["model"] = lit_sd_model.model
    seq_des_model["data_cfg"] = data_cfg
    seq_des_model["ckpt_path"] = str(resolved_ckpt_path)

    if load_sampling_cfg:
        seq_des_model["sampling_cfg"] = resolve_sampling_cfg(cfg)

    return seq_des_model


def parallel_context(num_workers: int):
    if num_workers <= 1:
        return nullcontext()
    if Parallel is None:
        raise ImportError("joblib is required when num_workers > 1") from None
    return Parallel(n_jobs=num_workers)
