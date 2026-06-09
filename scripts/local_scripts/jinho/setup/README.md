# AF3AD Desktop Setup

This directory contains setup scripts for running allatom-design on the lab
desktop with an RTX 2080 Ti.

## Files Description

### `install_elix_local.sh`
Complete environment installation script that performs:
- Installs packages into the active `elix_local` micromamba environment
- Installs PyTorch 2.7.0+cu126 and matching PyG wheels
- Installs AlphaFold3 build tools and AlphaFold3 runtime dependencies from
  `alphafold3/pyproject.toml`, including JAX 0.9.1
- Builds HMMER 3.4 from source (with seq_limit patch)
- Installs AlphaFold3 in editable mode with `--no-build-isolation`
- Installs allatom_design dependencies
- Installs AtomWorks in editable mode with `--no-deps`
- Installs allatom_design in editable mode
- Configures RTX 2080 Ti / Turing compatibility environment variables

### `activate_elix_local.sh`
Environment activation helper that configures HMMER, local AtomWorks mirror paths,
and RTX 2080 Ti runtime variables.

### `setup_rtx2080ti_env.sh`
Script for setting RTX 2080 Ti compatibility environment variables.

## Usage

### 1. Initial Installation
```bash
# Run from allatom-design root directory
export MAMBA_ROOT_PREFIX=/home/yjhk/model-dev/envs/micromamba
micromamba create -n elix_local python=3.12 zlib uv -c conda-forge -y
micromamba activate elix_local
bash scripts/local_scripts/jinho/setup/install_elix_local.sh
```

### 2. Environment Activation
```bash
# Method 1: Using script
micromamba activate elix_local
source scripts/local_scripts/jinho/setup/activate_elix_local.sh
```

### 3. Test Run
```bash
python allatom_design/train_seq_denoiser.py --config-path configs_local/seq_denoiser --config-name debug_elix.yaml
```

## Environment Configuration

### Python Packages
- **PyTorch**: 2.7.0+cu126
- **JAX**: 0.9.1 from the AlphaFold3 dependency set
- **Lightning**: 2.5.6
- **AlphaFold3**: Installed in editable mode with C++ extensions built locally
- **AtomWorks**: installed in editable mode with `--no-deps`
- **Allatom-design**: installed in editable mode with core dependencies (mashumaro, nglview, pandas, seaborn, matplotlib, torchtyping, einops, biopython, ihm, modelcif)
- **Extras**: biotite==1.2.0, hydride, py3Dmol, pymol-remote, pyarrow==17.0.0, cython, cytoolz, typer, openbabel-wheel, pathspec

### System Tools
- **HMMER**: 3.4 (with seq_limit patch applied)
- **Installation Path**: `/home/yjhk/model-dev/software/hmmer-3.4-af3/`

### RTX 2080 Ti Compatibility Settings
The following environment variables are automatically configured:
```bash
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=0.8
export CUDA_VISIBLE_DEVICES=0
export JAX_LOG_COMPILES=0
export TORCH_COMPILE=0
export AF3_FLASH_ATTENTION_IMPLEMENTATION=xla
export PDB_MIRROR_PATH="/home/yjhk/model-dev/datasets/pdb_mirror"
export CCD_MIRROR_PATH="/home/yjhk/model-dev/datasets/ccd_mirror"
export PATH="/home/yjhk/model-dev/software/hmmer-3.4-af3/bin:$PATH"
```

## Atomworks Installation

The script `install_elix_local.sh` installs the `atomworks` package from the `atomworks` directory in editable mode (`-e .`) while skipping dependencies (`--no-deps`). This allows source changes to take effect immediately during development.

## Allatom-design Installation

The script installs `allatom_design` from the repository root in editable mode.
The `elix_local` environment uses Python 3.12 to satisfy the AlphaFold3
runtime and editable C++ build requirements.

## Expected pip check Conflicts

- `atomworks` declares `rdkit<2025.9`, while AlphaFold3 uses
  `rdkit==2025.09.4`.
- `torch==2.7.0+cu126` declares `nvidia-cudnn-cu12==9.5.1.17`, while JAX 0.9.1
  uses `nvidia-cudnn-cu12==9.22.0.52`.

These metadata conflicts are expected for this environment; validate runtime
behavior with imports, AF3 runner tests, and a focused `run_elix` smoke.

## Key Changes Made

### Boltz Dependency Removal
Instead of installing the `boltz` package directly, necessary utility functions were copied to `allatom_design/model/seq_denoiser/utils.py`:
- `LinearNoBias`
- `SwiGLU`
- `center_random_augmentation`
- `randomly_rotate`
- `random_rotations`
- Other geometry functions

### Import Path Updates
- `allatom_design/model/seq_denoiser/denoisers/seq_design/elix_mpnn.py`
- `allatom_design/data/feature/seq_des_featurizer.py`
- `allatom_design/data/feature/motif_featurizer.py`

## Troubleshooting

### CUDA-related Errors
RTX 2080 Ti uses Turing architecture (Compute Capability 7.5). For AlphaFold3
inference, pass `--flash_attention_implementation=xla`; the activation helper
also sets the required XLA compatibility flag.

### Memory Issues
`XLA_CLIENT_MEM_FRACTION=0.8` limits GPU memory usage, and `TF_FORCE_UNIFIED_MEMORY=true` enables unified memory.

### DataLoader Worker Errors
Adjust `num_workers` in config files or reduce `batch_size` if needed.

## Directory Structure
```
/home/yjhk/model-dev/
├── envs/
│   └── micromamba/     # micromamba root prefix
├── cache/
│   └── uv/             # UV cache directory
├── software/
│   └── hmmer-3.4-af3/  # patched HMMER installation
└── allatom-design/     # Main codebase
```

```
/home/yjhk/model-dev/envs/micromamba/
└── envs/
    └── elix_local/  # micromamba environment
```
