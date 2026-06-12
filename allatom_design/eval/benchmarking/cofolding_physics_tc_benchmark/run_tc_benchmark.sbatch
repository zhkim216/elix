#!/bin/bash
# Submit from Sherlock, overriding DATASET_ROOT/MODEL_DIR/DB_DIR if needed:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     allatom_design/eval/benchmarking/cofolding_physics_tc_benchmark/run_tc_benchmark.sbatch
#
# Default array 0-13 covers the three AF3 conditions with CHUNK_SIZE=200 for the
# current prepared dataset. For holo-only runs, submit with --array=0-5 and
# CONDITIONS=holo_protein_only_template.
#SBATCH --job-name=cofolding_tc_af3
#SBATCH -p possu,bioe
#SBATCH --array=0-13
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/cofolding_tc_af3_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/cofolding_tc_af3_%A_%a.err

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/users/zhkim216/datasets/cofolding_physics_tc_benchmark}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/users/zhkim216/out_dir/cofolding_physics_tc_benchmark}"
RUN_LABEL="${RUN_LABEL:-af3_tc_recycles1_diff1_chunk200}"
CONDITIONS="${CONDITIONS:-no_template_control,holo_protein_only_template,mutant_protein_only_template}"
CHUNK_SIZE="${CHUNK_SIZE:-200}"
MODE="${MODE:-run-af3}"
RUNNER_PATH="${RUNNER_PATH:-${BASE_DIR}/alphafold3/src/alphafold3/run_alphafold.py}"
MODEL_DIR="${MODEL_DIR:-/scratch/users/zhkim216/model_params/af3}"
DB_DIR="${DB_DIR:-/scratch/users/zhkim216/af3_databases}"
NUM_RECYCLES="${NUM_RECYCLES:-1}"
NUM_DIFFUSION_SAMPLES="${NUM_DIFFUSION_SAMPLES:-1}"
POCKET_DISTANCE_FOR_DOCKING_METRICS="${POCKET_DISTANCE_FOR_DOCKING_METRICS:-8.0}"
FLASH_ATTN_IMPL="${FLASH_ATTN_IMPL:-xla}"
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

IFS=',' read -r -a CONDITION_VALUES <<< "${CONDITIONS}"
CONDITION_ARGS=()
for CONDITION in "${CONDITION_VALUES[@]}"; do
  [[ -n "${CONDITION}" ]] || continue
  CONDITION_ARGS+=(--condition "${CONDITION}")
done

EXTRA_ARGS=()
if [[ -n "${PREDICTIONS_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--predictions-root "${PREDICTIONS_ROOT}")
fi
if [[ "${USE_SUBPROCESS}" == "true" ]]; then
  EXTRA_ARGS+=(--use-subprocess)
fi

RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_LABEL}/array_${TASK_ID}"

mkdir -p /scratch/users/zhkim216/job_output "${RUN_OUTPUT_DIR}"

[[ -d "${BASE_DIR}" ]] || { echo "Missing BASE_DIR=${BASE_DIR}" >&2; exit 1; }
[[ -d "${DATASET_ROOT}" ]] || { echo "Missing DATASET_ROOT=${DATASET_ROOT}" >&2; exit 1; }
[[ -f "${RUNNER_PATH}" ]] || { echo "Missing RUNNER_PATH=${RUNNER_PATH}" >&2; exit 1; }
[[ -d "${MODEL_DIR}" ]] || { echo "Missing MODEL_DIR=${MODEL_DIR}" >&2; exit 1; }
[[ -d "${DB_DIR}" ]] || { echo "Missing DB_DIR=${DB_DIR}" >&2; exit 1; }

cd "${BASE_DIR}"
export PYTHONWARNINGS="ignore"

echo "Starting cofolding physics TC AF3 benchmark: task=${TASK_ID}, mode=${MODE}"
echo "DATASET_ROOT=${DATASET_ROOT}"
echo "OUTPUT_DIR=${RUN_OUTPUT_DIR}"
echo "CONDITIONS=${CONDITIONS}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "NUM_RECYCLES=${NUM_RECYCLES}"
echo "NUM_DIFFUSION_SAMPLES=${NUM_DIFFUSION_SAMPLES}"
echo "FLASH_ATTN_IMPL=${FLASH_ATTN_IMPL}"

python3 -m allatom_design.eval.benchmarking.cofolding_physics_tc_benchmark.run_tc_benchmark \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${RUN_OUTPUT_DIR}" \
  "${CONDITION_ARGS[@]}" \
  --chunk-size "${CHUNK_SIZE}" \
  --array-id "${TASK_ID}" \
  ${MODE_FLAG} \
  --runner-path "${RUNNER_PATH}" \
  --model-dir "${MODEL_DIR}" \
  --db-dir "${DB_DIR}" \
  --num-recycles "${NUM_RECYCLES}" \
  --num-diffusion-samples "${NUM_DIFFUSION_SAMPLES}" \
  --flash-attention-implementation "${FLASH_ATTN_IMPL}" \
  --pocket-distance-for-docking-metrics "${POCKET_DISTANCE_FOR_DOCKING_METRICS}" \
  "${EXTRA_ARGS[@]}"

echo "Completed cofolding physics TC AF3 benchmark: task=${TASK_ID}, output=${RUN_OUTPUT_DIR}"
