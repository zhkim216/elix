#!/bin/bash

# =============================================================================
# AF3AD Desktop Environment Setup Script (Hybrid: Mamba + UV)
# =============================================================================
# This script sets up the complete environment for allatom-design on the lab
# desktop RTX 2080 Ti
# using a hybrid approach:
# 1. Mamba (Conda) for Python, uv, and C/C++ build headers such as zlib
# 2. UV for Python package management and editable installs
#
# Prerequisite:
#   export MAMBA_ROOT_PREFIX=/home/yjhk/model-dev/envs/micromamba
#   micromamba create -n elix_local python=3.12 zlib uv -c conda-forge -y
#   micromamba activate elix_local
#
# Usage: bash scripts/local_scripts/jinho/setup/install_elix_local.sh
# =============================================================================

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
MODEL_DEV_ROOT="${MODEL_DEV_ROOT:-$HOME/model-dev}"
ENV_NAME="${ENV_NAME:-elix_local}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$MODEL_DEV_ROOT/cache/uv}"
HMMER_BUILD_DIR="${HMMER_BUILD_DIR:-$MODEL_DEV_ROOT/build/hmmer-3.4-af3}"
HMMER_INSTALL_DIR="${HMMER_INSTALL_DIR:-$MODEL_DEV_ROOT/software/hmmer-3.4-af3}"
HMMER_LINK_DIR="${HMMER_LINK_DIR:-$MODEL_DEV_ROOT/software/hmmer-af3}"

# Check if correct conda environment is active
if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "Error: '$ENV_NAME' Conda environment is not active."
    echo "Current environment: ${CONDA_DEFAULT_ENV:-None}"
    echo "Please run: micromamba activate $ENV_NAME"
    exit 1
fi

python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit(
        f"Error: AlphaFold3 requires Python >=3.12, got {sys.version.split()[0]}"
    )
PY

echo "Starting AF3AD desktop environment setup (Hybrid Mode)..."
echo "Configuration:"
echo "  Active Environment: $CONDA_DEFAULT_ENV"
echo "  Python Location: $(which python)"
echo "  Repo Dir: $REPO_DIR"
echo "  UV Cache Dir: $UV_CACHE_DIR"
echo "  HMMER Build Dir: $HMMER_BUILD_DIR"
echo "  HMMER Install Dir: $HMMER_INSTALL_DIR"
echo ""

mkdir -p "$UV_CACHE_DIR" "$(dirname "$HMMER_BUILD_DIR")" "$(dirname "$HMMER_INSTALL_DIR")"

# AlphaFold3's editable C++ build needs conda-provided headers such as zlib.
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export ZLIB_ROOT="${ZLIB_ROOT:-$CONDA_PREFIX}"
export CMAKE_ARGS="${CMAKE_ARGS:--DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DZLIB_ROOT=$CONDA_PREFIX}"

# Step 1: Install pip & build tools via UV
echo "Step 1: Updating pip and build tools..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install --upgrade pip setuptools wheel
# Install exceptiongroup because some AtomWorks dependency paths still import it.
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install exceptiongroup
echo "✓ Basic tools updated (including exceptiongroup)"

# Step 2: Install PyTorch / PyG from repo-pinned CUDA 12.6 requirements.
echo "Step 2: Installing PyTorch 2.7.0+cu126 and PyG wheels..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install \
    -r "$REPO_DIR/requirements_split/local/uv-compatible-torch.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu126
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install -r "$REPO_DIR/requirements_split/local/pip-only-torch.txt"
echo "✓ PyTorch installed"

# Step 3: Install AlphaFold3 build tools. Runtime deps, including JAX, are
# installed from alphafold3/pyproject.toml in Step 5.
echo "Step 3: Installing AlphaFold3 build tools..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install \
    scikit_build_core pybind11 "cmake>=3.28" ninja setuptools_scm
echo "✓ AlphaFold3 build tools installed"

# Step 4: Install HMMER from source with seq_limit patch
echo "Step 4: Installing HMMER from source..."
mkdir -p "$HMMER_BUILD_DIR" "$HMMER_INSTALL_DIR"
cd "$HMMER_BUILD_DIR"

if [ ! -f "hmmer-3.4.tar.gz" ]; then
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
fi

# Verify checksum
echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz" | sha256sum --check

if [ ! -d "hmmer-3.4" ]; then
    tar zxf hmmer-3.4.tar.gz
fi

# Apply seq_limit patch logic
PATCH_PATH="$(dirname "$0")/../../../../alphafold3/docker/jackhmmer_seq_limit.patch"
# Fallback to absolute path search if relative fails
if [ ! -f "$PATCH_PATH" ]; then
    PATCH_PATH="$REPO_DIR/alphafold3/docker/jackhmmer_seq_limit.patch"
fi

if [ ! -f "$PATCH_PATH" ]; then
    echo "Error: jackhmmer_seq_limit.patch not found at $PATCH_PATH"
    exit 1
fi

cp "$PATCH_PATH" .
cd hmmer-3.4
if [ ! -f ".patch_applied" ]; then
    echo "Applying jackhmmer_seq_limit.patch..."
    patch -p1 < ../jackhmmer_seq_limit.patch
    touch .patch_applied
fi

./configure --prefix "$HMMER_INSTALL_DIR"
make -j$(nproc)
make install
cd easel && make install
ln -sfn "$(basename "$HMMER_INSTALL_DIR")" "$HMMER_LINK_DIR"
"$HMMER_INSTALL_DIR/bin/jackhmmer" -h | grep -q -- "--seq_limit"
echo "✓ HMMER installed"

# Step 5: Install AlphaFold3
echo "Step 5: Installing AlphaFold3..."
# Go back to allatom-design root
cd "$REPO_DIR"

UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install \
    -e ./alphafold3 --no-build-isolation --index-strategy unsafe-best-match
if command -v build_data &> /dev/null; then
    build_data
fi
echo "✓ AlphaFold3 installed"

# Step 6: Install allatom_design dependencies
echo "Step 6: Installing allatom_design dependencies..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install mashumaro==3.14
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install nglview==3.1.4
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install lightning==2.5.6 hydra-core wandb
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install pandas seaborn matplotlib
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install torchtyping einops biopython
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install ihm modelcif typing_extensions
echo "✓ All dependencies installed"

# Step 7: Atomworks & Utils
echo "Step 7: Installing atomworks dependencies..."
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install \
"biotite==1.2.0" \
"hydride>=1.2.3,<2" \
"py3Dmol>=2.2.1,<3" \
"pymol-remote>=0.0.5" \
"pyarrow==17.0.0" \
"cython>=3,<4" \
"cytoolz>=0.12.3,<1" \
"typer>=0.12.5,<1"
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install "openbabel-wheel==3.1.1.22" pathspec

echo "Step 7.2: Installing atomworks..."
if [ -d "atomworks" ]; then
    cd atomworks
    # Keep the AlphaFold3 dependency set; AtomWorks is installed without deps.
    UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install -e . --no-deps
fi
echo "✓ atomworks installed"

echo "Step 7.3: Installing allatom_design..."
cd "$REPO_DIR"
# Dependencies were installed explicitly above; keep the editable project install
# from changing the AF3-compatible dependency set.
python -m pip install -e . --no-deps --ignore-requires-python
echo "✓ allatom_design installed"

# JAX 0.9.1 requires newer cuDNN than the exact torch 2.7.0 metadata pin.
UV_CACHE_DIR="$UV_CACHE_DIR" uv pip install \
    nvidia-cudnn-cu12==9.22.0.52 --index-strategy unsafe-best-match

# Step 8: Cleanup
echo "Step 8: Cleaning up..."
rm -rf "$HMMER_BUILD_DIR"

echo ""
echo "============================================================================="
echo "AF3AD Desktop Environment Setup Complete!"
echo "============================================================================="
echo "Environment: $ENV_NAME"
echo "Python Version: $(python --version)"
echo ""
echo "To activate environment variables for the RTX 2080 Ti desktop:"
echo "  source scripts/local_scripts/jinho/setup/activate_elix_local.sh"
echo ""
