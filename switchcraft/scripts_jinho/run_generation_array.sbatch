#!/bin/bash
#SBATCH --job-name=switchcraft_gen
#SBATCH --partition=possu,owners
#SBATCH --account=possu
#SBATCH --qos=normal
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPU_SKU:A100_SXM4|GPU_SKU:A100_PCIE|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%A_%a.err

set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-/home/users/zhkim216/code/switchcraft/scripts_jinho}"
SWITCHCRAFT_ROOT="${SWITCHCRAFT_ROOT:-/scratch/users/zhkim216/code/switchcraft}"
SWITCHCRAFT_ENV="${SWITCHCRAFT_ENV:-/scratch/users/zhkim216/envs/uv/switchcraft-py311-cu126}"
SWITCHCRAFT_SIF="${SWITCHCRAFT_SIF:-/scratch/users/zhkim216/containers/elix.sif}"
SWITCHCRAFT_CACHE_ROOT="${SWITCHCRAFT_CACHE_ROOT:-/scratch/users/zhkim216/cache/switchcraft}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-/bin/singularity}"
WORK_ITEMS="${WORK_ITEMS:-/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/manifests/work_items.tsv}"
WORK_OFFSET="${WORK_OFFSET:?WORK_OFFSET must be exported by submit_generation.sh}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

GLOBAL_WORK_INDEX=$((WORK_OFFSET + ARRAY_TASK_ID))
ROW_NUMBER=$((GLOBAL_WORK_INDEX + 2))
ROW="$(awk -v row_number="$ROW_NUMBER" 'NR == row_number { print; exit }' "$WORK_ITEMS")"
if [[ -z "$ROW" ]]; then
  echo "No work item at global index $GLOBAL_WORK_INDEX (row $ROW_NUMBER)" >&2
  exit 2
fi

IFS=$'\t' read -r \
  WORK_INDEX CASE_ID FAMILY WORKER_ID NUM_WORKERS NUM_DESIGNS NUM_STATES \
  MOTIF_KEY CONFIG_PATH OUTPATH DESIGN_ROOT <<< "$ROW"

if [[ "$WORK_INDEX" != "$GLOBAL_WORK_INDEX" ]]; then
  echo "Work index mismatch: expected $GLOBAL_WORK_INDEX, found $WORK_INDEX" >&2
  exit 2
fi

echo "case_id=$CASE_ID family=$FAMILY worker=$WORKER_ID/$NUM_WORKERS"
echo "config=$CONFIG_PATH"
echo "design_root=$DESIGN_ROOT"
if [[ ! -x "$CONTAINER_RUNTIME" ]]; then
  echo "Container runtime is not executable: $CONTAINER_RUNTIME" >&2
  exit 2
fi
mkdir -p \
  "$SWITCHCRAFT_CACHE_ROOT/cuequivariance-triton" \
  "$SWITCHCRAFT_CACHE_ROOT/torch" \
  "$SWITCHCRAFT_CACHE_ROOT/torch_extensions" \
  "$SWITCHCRAFT_CACHE_ROOT/torchinductor" \
  "$SWITCHCRAFT_CACHE_ROOT/triton"

srun --ntasks=1 "$CONTAINER_RUNTIME" exec --nv \
  --bind /scratch/users/zhkim216:/scratch/users/zhkim216,/home/users/zhkim216:/home/users/zhkim216 \
  "$SWITCHCRAFT_SIF" \
  bash -lc '
    set -euo pipefail
    source "$1/bin/activate"
    export HOME=/home/users/zhkim216
    export XDG_CACHE_HOME="${12}"
    export CUEQ_TRITON_CACHE_DIR="${12}/cuequivariance-triton"
    export TORCH_HOME="${12}/torch"
    export TORCH_EXTENSIONS_DIR="${12}/torch_extensions"
    export TORCHINDUCTOR_CACHE_DIR="${12}/torchinductor"
    export TRITON_CACHE_DIR="${12}/triton"
    export BOLTZ_USE_CUEQ=1
    export PYTHONUNBUFFERED=1
    exec python -u "$2/run_worker.py" \
      --switchcraft-root "$3" \
      --config "$4" \
      --outpath "$5" \
      --design-root "$6" \
      --num-designs "$7" \
      --num-workers "$8" \
      --worker-id "$9" \
      --num-states "${10}" \
      --motifs "${11}"
  ' bash \
  "$SWITCHCRAFT_ENV" \
  "$SCRIPT_ROOT" \
  "$SWITCHCRAFT_ROOT" \
  "$CONFIG_PATH" \
  "$OUTPATH" \
  "$DESIGN_ROOT" \
  "$NUM_DESIGNS" \
  "$NUM_WORKERS" \
  "$WORKER_ID" \
  "$NUM_STATES" \
  "$MOTIF_KEY" \
  "$SWITCHCRAFT_CACHE_ROOT"
