#!/usr/bin/env bash

# Canonical Sherlock environment wiring for Elix with Torch 2.8.
# Source this before entering the container or wrapping sbatch jobs.

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6.1 || true
fi

# Cache and build locations.
ELIX_USER="${USER:-$(id -un)}"
ELIX_HOME="${HOME:-/home/users/$ELIX_USER}"
export SCRATCH="${SCRATCH:-/scratch/users/$ELIX_USER}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/cache/torch}"
export HF_HOME="${HF_HOME:-$SCRATCH/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/cache/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/cache/.cache}"
# Keep Python bytecode beside its source. A shared absolute-path pycache tree on
# Lustre amplifies metadata traffic and duplicates directory hierarchies.
unset PYTHONPYCACHEPREFIX
export TORCHINDUCTOR_CACHE_DIR="${ELIX_TORCH280_INDUCTOR_CACHE_DIR:-$SCRATCH/cache/inductor_cache_torch280}"
export TRITON_CACHE_DIR="${ELIX_TORCH280_TRITON_CACHE_DIR:-$SCRATCH/cache/triton_cache_torch280}"
export TORCH_EXTENSIONS_DIR="${ELIX_TORCH280_EXTENSIONS_DIR:-$SCRATCH/cache/torch_extensions_torch280}"
export UV_ENV_ROOT="${UV_ENV_ROOT:-$SCRATCH/envs/uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$UV_ENV_ROOT/python}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$SCRATCH/cache/jax_compilation_cache}"
export ELIX_TORCH280_VENV="${ELIX_TORCH280_VENV:-$UV_ENV_ROOT/elix-torch280}"
export VENV="$ELIX_TORCH280_VENV"

mkdir -p \
  "$UV_ENV_ROOT" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" \
  "$JAX_COMPILATION_CACHE_DIR"

# CUDA setup. CUDA_HOST is the host module path; CUDA_HOME is the in-container path.
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOST="${CUDA_HOST:-$(dirname "$(dirname "$(command -v nvcc)")")}"
fi
export CUDA_HOST="${CUDA_HOST:-/usr/local/cuda}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if [ -f "$CUDA_HOME/targets/x86_64-linux/lib/stubs/libcuda.so" ]; then
  export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-$CUDA_HOME/targets/x86_64-linux/lib/stubs}"
else
  export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-$CUDA_HOME/lib64/stubs}"
fi

# AlphaFold3/JAX settings. The disable_hlo_passes flag is needed on V100/T4-class GPUs.
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false --xla_disable_hlo_passes=custom-kernel-fusion-rewriter}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-true}"
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.95}"

# Elix paths. Torch 2.8 is the only supported Elix environment.
export SIF="${SIF:-$SCRATCH/containers/elix.sif}"
export PROJECT_ROOT="${PROJECT_ROOT:-$ELIX_HOME/code/elix}"
export TMALIGN_BINARY="${TMALIGN_BINARY:-/oak/stanford/groups/possu/jinho/software/TMalign}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Elix Torch 2.8 environment is not installed: $VENV" >&2
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 1
  fi
  exit 1
fi

export PATH="$VENV/bin:/hmmer/bin:$CUDA_HOME/bin:$PATH"

echo "Elix Torch 2.8 Sherlock environment loaded:"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA_HOST: $CUDA_HOST"
echo "  SIF: $SIF"
echo "  VENV: $VENV"
echo "  UV_ENV_ROOT: $UV_ENV_ROOT"
echo "  TMALIGN_BINARY: $TMALIGN_BINARY"
