#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/benchmarking/run_cofolding_physics_tc_atp_charge3_sweep.sbatch
#
# Five array tasks split by template-pair scale. Each task runs all recycle
# values for native ATP and ATP charge_3.
#SBATCH --job-name=tc_atp_charge3
#SBATCH -p possu,bioe
#SBATCH --array=0-4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/cofolding_tc_atp_charge3_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/cofolding_tc_atp_charge3_%A_%a.err

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

PAIR_SCALES=(0.1 0.15 0.2 0.25 0.3)
NUM_RECYCLES_VALUES=(1 3 5 7 10)
EXPERIMENT_IDS=(
  cdk2_1b38_atp_modification_native_atp
  cdk2_1b38_atp_modification_atp_charge_3
)

NUM_PAIR_SCALES=${#PAIR_SCALES[@]}
if [[ "${TASK_ID}" -ge "${NUM_PAIR_SCALES}" ]]; then
  echo "TASK_ID=${TASK_ID} is outside selected pair scales=${NUM_PAIR_SCALES}; exiting without work."
  exit 0
fi

PAIR_SCALE=${PAIR_SCALES[$TASK_ID]}
PAIR_SCALE_LABEL=${PAIR_SCALE//./p}

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/users/zhkim216/datasets/cofolding_physics_tc_benchmark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/users/zhkim216/out_dir/cofolding_physics_tc_benchmark}"
RUN_LABEL="${RUN_LABEL:-atp_native_vs_charge3_pairscale_recycles_sweep}"
MODE="${MODE:-run-af3}"
RUNNER_PATH="${RUNNER_PATH:-${BASE_DIR}/alphafold3/src/alphafold3/run_alphafold.py}"
MODEL_DIR="${MODEL_DIR:-/scratch/users/zhkim216/model_params/af3}"
DB_DIR="${DB_DIR:-/scratch/users/zhkim216/af3_databases}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-5}"
POCKET_DISTANCE_FOR_DOCKING_METRICS="${POCKET_DISTANCE_FOR_DOCKING_METRICS:-8.0}"
USE_SUBPROCESS="${USE_SUBPROCESS:-false}"

case "${MODE}" in
  run-af3)
    MODE_FLAG="--run-af3"
    ;;
  metrics-only)
    MODE_FLAG="--metrics-only"
    ;;
  generate-inputs-only)
    MODE_FLAG="--generate-inputs-only"
    ;;
  *)
    echo "Invalid MODE=${MODE}; expected run-af3, metrics-only, or generate-inputs-only." >&2
    exit 2
    ;;
esac

EXPERIMENT_ARGS=()
for EXPERIMENT_ID in "${EXPERIMENT_IDS[@]}"; do
  EXPERIMENT_ARGS+=(--experiment-id "${EXPERIMENT_ID}")
done

EXTRA_ARGS=()
if [[ -n "${PREDICTIONS_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--predictions-root "${PREDICTIONS_ROOT}")
fi
if [[ "${USE_SUBPROCESS}" == "true" ]]; then
  EXTRA_ARGS+=(--use-subprocess)
fi

mkdir -p /scratch/users/zhkim216/job_output

[[ -d "${BASE_DIR}" ]] || { echo "Missing BASE_DIR=${BASE_DIR}" >&2; exit 1; }
[[ -d "${DATASET_ROOT}" ]] || { echo "Missing DATASET_ROOT=${DATASET_ROOT}" >&2; exit 1; }
[[ -f "${RUNNER_PATH}" ]] || { echo "Missing RUNNER_PATH=${RUNNER_PATH}" >&2; exit 1; }
[[ -d "${MODEL_DIR}" ]] || { echo "Missing MODEL_DIR=${MODEL_DIR}" >&2; exit 1; }
[[ -d "${DB_DIR}" ]] || { echo "Missing DB_DIR=${DB_DIR}" >&2; exit 1; }

cd "${BASE_DIR}"
export PYTHONWARNINGS="ignore"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "${COMPUTE_CAP}" | cut -d. -f1)

if [[ "${MAJOR}" -ge 8 ]]; then
  FLASH_ATTN_IMPL="${FLASH_ATTN_IMPL:-triton}"
else
  FLASH_ATTN_IMPL="${FLASH_ATTN_IMPL:-xla}"
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi

echo "Starting cofolding physics ATP native vs charge_3 sweep"
echo "TASK_ID=${TASK_ID}/${NUM_PAIR_SCALES}"
echo "PAIR_SCALE=${PAIR_SCALE}"
echo "NUM_RECYCLES_VALUES=${NUM_RECYCLES_VALUES[*]}"
echo "EXPERIMENT_IDS=${EXPERIMENT_IDS[*]}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "RUN_LABEL=${RUN_LABEL}"
echo "MODE=${MODE}"
echo "NUM_DIFFUSION_SAMPLES=${NUM_DIFFUSION_SAMPLES}"
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"

for NUM_RECYCLES in "${NUM_RECYCLES_VALUES[@]}"; do
  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}/scale_${PAIR_SCALE_LABEL}_recycles${NUM_RECYCLES}"
  mkdir -p "${RUN_OUTPUT_DIR}"

  echo "Running pair_scale=${PAIR_SCALE}, num_recycles=${NUM_RECYCLES}, output=${RUN_OUTPUT_DIR}"
  python3 -m allatom_design.eval.benchmarking.cofolding_physics_tc_benchmark.run_tc_benchmark \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${RUN_OUTPUT_DIR}" \
    --condition holo_protein_only_template \
    "${EXPERIMENT_ARGS[@]}" \
    --array-id 0 \
    --num-arrays 1 \
    ${MODE_FLAG} \
    --runner-path "${RUNNER_PATH}" \
    --model-dir "${MODEL_DIR}" \
    --db-dir "${DB_DIR}" \
    --num-recycles "${NUM_RECYCLES}" \
    --num-diffusion-samples "${NUM_DIFFUSION_SAMPLES}" \
    --template-pair-scale "${PAIR_SCALE}" \
    --flash-attention-implementation "${FLASH_ATTN_IMPL}" \
    --pocket-distance-for-docking-metrics "${POCKET_DISTANCE_FOR_DOCKING_METRICS}" \
    "${EXTRA_ARGS[@]}"
done

echo "Completed ATP native vs charge_3 sweep for pair_scale=${PAIR_SCALE}"
