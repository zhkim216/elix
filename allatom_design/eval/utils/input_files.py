import glob
import math
import re
from pathlib import Path

import numpy as np

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted


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
