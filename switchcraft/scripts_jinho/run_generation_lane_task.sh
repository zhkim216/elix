#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-/home/users/zhkim216/code/switchcraft/scripts_jinho}"
SWITCHCRAFT_ROOT="${SWITCHCRAFT_ROOT:-/scratch/users/zhkim216/code/switchcraft}"
SWITCHCRAFT_ENV="${SWITCHCRAFT_ENV:-/scratch/users/zhkim216/envs/uv/switchcraft-py311-cu126}"
SWITCHCRAFT_SIF="${SWITCHCRAFT_SIF:-/scratch/users/zhkim216/containers/elix.sif}"
SWITCHCRAFT_CACHE_ROOT="${SWITCHCRAFT_CACHE_ROOT:-/scratch/users/zhkim216/cache/switchcraft}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-/bin/singularity}"
LANE_MANIFEST="${LANE_MANIFEST:?LANE_MANIFEST must be exported by the controller}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

ROW_NUMBER=$((ARRAY_TASK_ID + 2))
ROW="$(awk -v row_number="$ROW_NUMBER" 'NR == row_number { print; exit }' "$LANE_MANIFEST")"
if [[ -z "$ROW" ]]; then
  echo "No lane task at array index $ARRAY_TASK_ID (row $ROW_NUMBER)" >&2
  exit 2
fi

IFS=$'\t' read -r \
  TASK_ID LANE CASE_ID FAMILY NUM_STATES DESIGN_INDICES MOTIFS CONFIG_PATH \
  OUTPATH DESIGN_ROOT NUM_DESIGNS <<< "$ROW"

if [[ "$TASK_ID" != "$ARRAY_TASK_ID" ]]; then
  echo "Task ID mismatch: expected $ARRAY_TASK_ID, found $TASK_ID" >&2
  exit 2
fi
if [[ ! -x "$CONTAINER_RUNTIME" ]]; then
  echo "Container runtime is not executable: $CONTAINER_RUNTIME" >&2
  exit 2
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
GPU_CACHE_KEY="$(sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//' <<< "$GPU_NAME")"
if [[ -z "$GPU_CACHE_KEY" ]]; then
  echo "Could not derive a cache key from GPU name: $GPU_NAME" >&2
  exit 2
fi
SKU_CACHE_ROOT="$SWITCHCRAFT_CACHE_ROOT/by-gpu-sku/$GPU_CACHE_KEY"
mkdir -p \
  "$SKU_CACHE_ROOT/cuequivariance-triton" \
  "$SKU_CACHE_ROOT/torch" \
  "$SKU_CACHE_ROOT/torch_extensions" \
  "$SKU_CACHE_ROOT/torchinductor" \
  "$SKU_CACHE_ROOT/triton"

echo "lane=$LANE case_id=$CASE_ID family=$FAMILY task_id=$TASK_ID"
echo "design_indices=$DESIGN_INDICES gpu=$GPU_NAME cache=$SKU_CACHE_ROOT"
echo "config=$CONFIG_PATH design_root=$DESIGN_ROOT"

# Some partitions raise the allocated CPUs/task above the requested TRES. Slurm
# then exports conflicting SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK values,
# which makes srun fail before the worker starts. Use the actual allocation.
unset SLURM_TRES_PER_TASK
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-2}" \
  "$CONTAINER_RUNTIME" exec --nv \
  --bind /scratch/users/zhkim216:/scratch/users/zhkim216,/home/users/zhkim216:/home/users/zhkim216 \
  "$SWITCHCRAFT_SIF" \
  bash -lc '
    set -euo pipefail
    source "$1/bin/activate"
    export HOME=/home/users/zhkim216
    export XDG_CACHE_HOME="${11}"
    export CUEQ_TRITON_CACHE_DIR="${11}/cuequivariance-triton"
    export TORCH_HOME="${11}/torch"
    export TORCH_EXTENSIONS_DIR="${11}/torch_extensions"
    export TORCHINDUCTOR_CACHE_DIR="${11}/torchinductor"
    export TRITON_CACHE_DIR="${11}/triton"
    export BOLTZ_USE_CUEQ=1
    export PYTHONUNBUFFERED=1
    exec python -u "$2/run_worker.py" \
      --switchcraft-root "$3" \
      --config "$4" \
      --outpath "$5" \
      --design-root "$6" \
      --num-designs "$7" \
      --design-indices "$8" \
      --num-states "$9" \
      --motifs "${10}"
  ' bash \
  "$SWITCHCRAFT_ENV" \
  "$SCRIPT_ROOT" \
  "$SWITCHCRAFT_ROOT" \
  "$CONFIG_PATH" \
  "$OUTPATH" \
  "$DESIGN_ROOT" \
  "$NUM_DESIGNS" \
  "$DESIGN_INDICES" \
  "$NUM_STATES" \
  "$MOTIFS" \
  "$SKU_CACHE_ROOT"
