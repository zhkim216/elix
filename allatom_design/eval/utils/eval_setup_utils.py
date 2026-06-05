import glob
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import wandb
import hydra
from omegaconf import DictConfig
from contextlib import nullcontext
from joblib import Parallel

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

from allatom_design.model.seq_denoiser.lit_sd_model import LitSeqDenoiser
from allatom_design.utils.checkpoint_utils import get_cfg_from_ckpt
from allatom_design.eval.utils.cfg_utils import require_cfg_value, resolve_sampling_cfg


###########################################################
# PDB file retrieval
###########################################################
def get_pdb_files(pdb_dir: str | None,
                  pdb_name_list: str | None,
                  pdb_name_ext: str | None = None,
                  n_subsample: int | None = None,
                  # slurm array parameters for parallelization
                  array_id: int | None = None,
                  num_arrays: int | None = None,
                  skip_pdb_names: list[str] | None = None,
                  # recursive search for nested directory structures (e.g. CCD code subfolders)
                  recursive: bool = False,
                  split_by_subfolder: bool = False,
                  # sample index filtering (e.g. {CCD}_len{L}_{IDX}.cif
                  # or {PREFIX}_{IDX}_model_{M}.cif)
                  sample_indices: list[int] | None = None,
                  # sample length filtering (e.g. {CCD}_len{L}_{IDX}.cif
                  # or {PREFIX}_len_{L}_{IDX}_model_{M}.cif)
                  sample_lengths: list[int] | None = None,
                  ) -> list[str]:
    """
    Retrieve a list of PDB files from a directory, either by specifying a list of pdb_names or by getting all files.

    Args:
        pdb_dir: Directory containing PDB files
        pdb_name_list: Optional path to a file containing PDB keys (one per line)
        pdb_name_ext: Optional extension to append to each key when pdb_name_list is provided
        array_id: Set by Slurm array job. Null means run all.
        num_arrays: Number of total arrays. If array_id is null, this can remain 1.
        skip_pdb_names: List of PDB names to skip
        recursive: If True, recursively search subdirectories for files.
            Useful for nested directory structures (e.g. CCD code subfolders).
        split_by_subfolder: If True and array_id is set, split by top-level subfolder
            instead of splitting the flat file list. Each array task gets one or more
            subfolders. Only used when recursive=True.
        sample_indices: Optional list of sample indices to keep. Filters files whose
            filename matches compact {PREFIX}_len{L}_{IDX}.ext or legacy
            {PREFIX}_{IDX}_model_{M}.ext patterns, keeping only those where IDX is in
            sample_indices.
        sample_lengths: Optional list of sample lengths to keep. Filters files whose
            filename matches compact {PREFIX}_len{L}_{IDX}.ext or legacy
            {PREFIX}_len_{L}_{IDX}_model_{M}.ext patterns, keeping only those where L
            is in sample_lengths.

    Returns:
        List of PDB file paths, naturally sorted if retrieving all files

    Raises:
        ValueError: If no PDB files are found in the directory when pdb_name_list is None
    """
    # Read in PDB files from directory or list of PDB names
    if pdb_name_list is not None:
        if isinstance(pdb_name_list, np.ndarray):
            pdb_name_list = pdb_name_list.tolist()
            pdb_names = [f"{Path(name).with_suffix(pdb_name_ext)}" for name in pdb_name_list]
            pdb_files = [f"{pdb_dir}/{name}" for name in pdb_names]
            print(f"Found {len(pdb_files)} PDB files from key list")
        else:
            # get PDBs with keys in the list
            with open(pdb_name_list, "r") as f:
                pdb_names = f.read().splitlines()
            if pdb_name_ext:
                # replace extension with pdb_name_ext
                pdb_names = [f"{Path(name).with_suffix(pdb_name_ext)}" for name in pdb_names]
            pdb_files = [f"{pdb_dir}/{name}" for name in pdb_names]
            print(f"Found {len(pdb_files)} PDB files from key list")
    elif recursive:
        # Recursively search subdirectories for files
        if split_by_subfolder and array_id is not None:
            # Split by top-level subfolder: each array task processes one or more subfolders
            subfolders = natsorted([
                d for d in Path(pdb_dir).iterdir() if d.is_dir()
            ])
            print(f"Found {len(subfolders)} subfolders in {pdb_dir}")

            chunk_size = math.ceil(len(subfolders) / num_arrays)
            start_idx = array_id * chunk_size
            end_idx = min(start_idx + chunk_size, len(subfolders))
            selected_subfolders = subfolders[start_idx:end_idx]
            print(f"Array {array_id}/{num_arrays}: processing subfolders {[s.name for s in selected_subfolders]}")

            # Collect all files from selected subfolders
            pdb_files = []
            for subfolder in selected_subfolders:
                files = natsorted([str(f) for f in subfolder.iterdir() if f.is_file()])
                pdb_files.extend(files)
        else:
            # Flat recursive search across all subdirectories
            pdb_files = natsorted([
                str(f) for f in Path(pdb_dir).rglob("*") if f.is_file()
            ])

        # Filter by extension if pdb_name_ext is provided
        if pdb_name_ext:
            pdb_files = [f for f in pdb_files if f.endswith(pdb_name_ext)]

        print(f"Found {len(pdb_files)} PDB files recursively in {pdb_dir}")
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found recursively in directory {pdb_dir}")
    else:
        # get all PDBs in the directory
        pdb_files = natsorted(list(glob.glob(f"{pdb_dir}/*")))

        # Filter by extension if pdb_name_ext is provided
        if pdb_name_ext:
            pdb_files = [f for f in pdb_files if f.endswith(pdb_name_ext)]
        else:
            # Filter out non-structure files (e.g. .pt, .pkl, .json)
            supported_exts = {".pdb", ".cif", ".mmcif", ".ent"}
            pdb_files = [f for f in pdb_files if Path(f).suffix.lower() in supported_exts]

        print(f"Found {len(pdb_files)} PDB files in {pdb_dir}")
        if len(pdb_files) == 0:
            raise ValueError(f"No PDB files found in directory {pdb_dir}")

    # Skip existing PDBs
    if skip_pdb_names is not None:
        skip_pdb_names = set(skip_pdb_names)
        pdb_files = [f for f in pdb_files if Path(f).name not in skip_pdb_names]

    # Filter by sample indices.
    if sample_indices is not None:
        sample_idx_patterns = [
            re.compile(r"_(\d+)_model_\d+(?:$|_)"),
            re.compile(r"_len\d+_(\d+)$"),
        ]
        filtered_files = []
        for f in pdb_files:
            stem = Path(f).stem
            match = None
            for pattern in sample_idx_patterns:
                match = pattern.search(stem)
                if match:
                    break
            if match and int(match.group(1)) in sample_indices:
                filtered_files.append(f)
        print(f"Filtered by sample_indices {sample_indices}: {len(pdb_files)} -> {len(filtered_files)} files")
        pdb_files = filtered_files

    # Filter by sample lengths.
    if sample_lengths is not None:
        sample_len_patterns = [
            re.compile(r"_len_(\d+)_"),
            re.compile(r"_len(\d+)_"),
        ]
        filtered_files = []
        for f in pdb_files:
            stem = Path(f).stem
            match = None
            for pattern in sample_len_patterns:
                match = pattern.search(stem)
                if match:
                    break
            if match and int(match.group(1)) in sample_lengths:
                filtered_files.append(f)
        print(f"Filtered by sample_lengths {sample_lengths}: {len(pdb_files)} -> {len(filtered_files)} files")
        pdb_files = filtered_files

    # Parallelization: split PDB files into chunks based on array id
    # (skip if already split by subfolder above)
    if array_id is not None and not (recursive and split_by_subfolder):
        chunk_size = math.ceil(len(pdb_files) / num_arrays)

        start_idx = array_id * chunk_size
        end_idx = min(start_idx + chunk_size, len(pdb_files))
        pdb_files = pdb_files[start_idx:end_idx]

    # Optionally take a random subset, preserving order
    if n_subsample is not None:
        n_subsample = min(n_subsample, len(pdb_files))
        chosen_indices = sorted(np.random.choice(len(pdb_files), n_subsample, replace=False))
        pdb_files = [pdb_files[i] for i in chosen_indices]

    print(f"Using {len(pdb_files)} PDB files")

    return pdb_files


###########################################################
# Model checkpoint utils
###########################################################

def get_checkpoints(design_cfg: DictConfig) -> list[dict[str, Any]]:
    """
    Get model checkpoints from a training directory, preferring EMA checkpoints if available.

    Args:
        denoiser_train_dir: Path to the denoiser training directory
        model_cfg.model_type: Either "atom_denoiser" or "seq_denoiser".
            If omitted, sequence-design model names such as "elix_mpnn" are
            treated as "seq_denoiser".
        eval_every_n_ckpts: Only evaluate every nth checkpoint
        start_step: Optional starting step to filter checkpoints (skip checkpoints before this step)
        end_step: Optional ending step to filter checkpoints (skip checkpoints after this step)
        use_ema: Whether to use EMA checkpoints
        eval_last_ckpt: Always include the last checkpoint even if not selected by eval_every_n_ckpts

    Returns:
        List of checkpoint info dicts, sorted by step/epoch. Each dict contains
        `ckpt_path`, `global_step`, and `epoch`.
    """
    model_cfg = require_cfg_value(design_cfg, "model_cfg", "sequence design config")
    direct_ckpt = model_cfg.get("ckpt_path", None)

    # Map model type to checkpoint prefix
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
        return [{
            "ckpt_path": str(direct_ckpt),
            "global_step": int(match.group(1)),
            "epoch": int(match.group(2)),
        }]

    ckpt_cfg = model_cfg.get("ckpt_cfg", None)
    if ckpt_cfg is None:
        raise ValueError("model_cfg.ckpt_cfg is required when model_cfg.ckpt_path is null")

    # Check for EMA checkpoints
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

    # Filter and sort checkpoints
    all_ckpt_infos = []
    for ckpt in natsorted(ckpts):
        match = pattern.search(Path(ckpt).name)
        if match is None:
            continue
        all_ckpt_infos.append({
            "ckpt_path": ckpt,
            "global_step": int(match.group(1)),
            "epoch": int(match.group(2)),
        })

    # Filter by start_step and end_step if provided
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

    ckpt_infos = filtered_ckpt_infos[::ckpt_cfg.eval_every_n_ckpts]

    # Include the last checkpoint if eval_last_ckpt is True and it's not already included
    selected_paths = {ckpt_info["ckpt_path"] for ckpt_info in ckpt_infos}
    if (
        ckpt_cfg.eval_last_ckpt
        and all_ckpt_infos
        and all_ckpt_infos[-1]["ckpt_path"] not in selected_paths
    ):
        ckpt_infos.append(all_ckpt_infos[-1])

    return ckpt_infos

def load_seq_des_model(cfg: DictConfig = None,
                      device: str = None,
                      ckpt_path: str | None = None,
                      load_sampling_cfg: bool = True) -> dict[str, Any]:
    """
    Load in a sequence design model from model_cfg and sampling_cfg.
    """
    if cfg is None:
        raise ValueError("sequence design config is required")

    model_cfg = require_cfg_value(cfg, "model_cfg", "sequence design config")
    model_name = require_cfg_value(model_cfg, "model_name", "model_cfg")
    seq_des_model = {"model_name": model_name, "cfg": cfg, "device": device}

    resolved_ckpt_path = ckpt_path or model_cfg.get("ckpt_path", None)
    if resolved_ckpt_path is None:
        raise ValueError("model_cfg.ckpt_path is required unless ckpt_path is passed explicitly")

    # Load the model from checkpoint
    lit_sd_model = LitSeqDenoiser.load_from_checkpoint(str(resolved_ckpt_path)).eval()
    ckpt_model_cfg, _ = get_cfg_from_ckpt(str(resolved_ckpt_path))
    data_cfg = hydra.utils.instantiate(ckpt_model_cfg.data)

    seq_des_model["model"] = lit_sd_model.model
    seq_des_model["data_cfg"] = data_cfg
    seq_des_model["ckpt_path"] = str(resolved_ckpt_path)

    if load_sampling_cfg:
        seq_des_model["sampling_cfg"] = resolve_sampling_cfg(cfg)

    return seq_des_model

###########################################################
# Weights & Biases setup
###########################################################
def wandb_setup(
    base_out_dir: str,
    no_wandb: bool,
    project: str | None,
    wandb_id: str | None,
    group: str | None,
    exp_name: str | None,
    cfg_dict: dict = None,
) -> str:
    """
    Set up Weights & Biases (wandb) tracking and return the log directory.
    Log directory is set to base_out_dir/exp_name.

    Args:
        no_wandb: If True, disable wandb logging
        project: wandb project name
        wandb_id: wandb entity ID
        group: Group name for the experiment
        exp_name: Name of the experiment
        base_out_dir: Base output directory for logs
        cfg_dict: Configuration dictionary to log

    Returns:
        Path: Log directory path
    """
    if exp_name is None:
        exp_name = "debug"

    # Set up log directory
    log_dir = str(Path(base_out_dir, exp_name))
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if not no_wandb:
        # Create wandb dir
        wandb_dir = str(Path(base_out_dir, "wandb"))
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)

        # Set wandb cache directory
        wandb_cache_dir = str(Path(base_out_dir, "cache", "wandb"))
        os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir

        wandb.init(
            project=project,
            entity=wandb_id,
            group=group,
            name=exp_name,
            config=cfg_dict,
            dir=wandb_dir,
        )

        # Define custom x-axis metric to allow non-monotonic step logging
        # This is needed when evaluating multiple checkpoints in different phases
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("*", step_metric="trainer/global_step")

    return log_dir


# Parallel context manager
def _parallel_context(num_workers: int):
    if num_workers <= 1:
        return nullcontext()
    if Parallel is None:
        raise ImportError("joblib is required when num_workers > 1") from None
    return Parallel(n_jobs=num_workers)
