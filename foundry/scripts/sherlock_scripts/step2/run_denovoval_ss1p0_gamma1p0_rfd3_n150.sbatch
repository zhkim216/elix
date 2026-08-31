#!/usr/bin/env bash
#SBATCH --job-name=denovoval_ss1p0_g1p0_n150
#SBATCH --partition=possu,bioe,owners
#SBATCH --array=0-307
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_MEM:48GB|GPU_MEM:80GB|GPU_MEM:141GB"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/users/zhkim216/job_output/rfd3_sample_generation/denovoval_ss1p0_g1p0_n150_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/rfd3_sample_generation/denovoval_ss1p0_g1p0_n150_%A_%a.err
#
# Initial RFdiffusion3 generation for the frozen 154-CCD denovoval cohort:
#   154 CCDs (HEC excluded) x lengths {150, 300} x 150 samples.
# No array concurrency throttle is applied.

set -euo pipefail

# (JH) fixed: reject direct sbatch execution so the full cohort cannot start a nested container.
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "This sbatch must run through the Foundry RFD3 container wrapper." >&2
  echo "Submit with: /home/users/zhkim216/code/elix/foundry/scripts/sherlock_scripts/setup/wrap_sbatch_in_container_foundry_rfd3.sh ${BASH_SOURCE[0]}" >&2
  exit 2
fi

SOURCE_DATA_ROOT="/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step2/source_data"
CCD_LIST_FILE="${SOURCE_DATA_ROOT}/denovoval_step2_ccds.txt"
INPUTS_ROOT="${SOURCE_DATA_ROOT}/rfd3_inputs"
OUTPUT_ROOT="/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_ss1p0_gamma1p0/generated_samples"

LENGTHS=(150 300)
EXPECTED_CCD_COUNT=154
DIFFUSION_BATCH_SIZE=10
N_BATCHES=15
EXPECTED_SAMPLES=$((DIFFUSION_BATCH_SIZE * N_BATCHES))
STEP_SCALE=1.0
GAMMA_0=1.0
NOISE_SCALE=1.003
SEED=216

test -f "${CCD_LIST_FILE}"
mapfile -t SOURCE_CCDS < "${CCD_LIST_FILE}"

CCDS=()
for ccd in "${SOURCE_CCDS[@]}"; do
  if [[ -n "${ccd}" && "${ccd}" != "HEC" ]]; then
    CCDS+=("${ccd}")
  fi
done

if [[ "${#CCDS[@]}" -ne "${EXPECTED_CCD_COUNT}" ]]; then
  echo "Expected ${EXPECTED_CCD_COUNT} CCDs after excluding HEC, found ${#CCDS[@]}." >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LENGTH_COUNT="${#LENGTHS[@]}"
TOTAL_TASKS=$(( ${#CCDS[@]} * LENGTH_COUNT ))
if [[ "${PRINT_ARRAY_PLAN:-false}" == "true" ]]; then
  echo "NUM_TASKS=${TOTAL_TASKS} CCDS=${#CCDS[@]} LENGTHS=${LENGTHS[*]} EXPECTED_SAMPLES=${EXPECTED_SAMPLES}"
  exit 0
fi
if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside 0-$((TOTAL_TASKS - 1))." >&2
  exit 2
fi

CCD="${CCDS[$((TASK_ID / LENGTH_COUNT))]}"
LENGTH="${LENGTHS[$((TASK_ID % LENGTH_COUNT))]}"
INPUT_JSON="${INPUTS_ROOT}/${CCD}_len${LENGTH}.json"
OUT_DIR="${OUTPUT_ROOT}/${CCD}_len${LENGTH}"

if [[ "${PRINT_TASK_PLAN:-false}" == "true" ]]; then
  echo "${TASK_ID} ${CCD} ${LENGTH} ${INPUT_JSON} ${OUT_DIR}"
  exit 0
fi

test -f "${INPUT_JSON}"
test -f "${CKPT_PATH}"
test -d "${CCD_MIRROR_PATH}"
# (JH) fixed: fail closed when the standalone editable RFD3 environment was not activated.
test -x "${RFD3_PYTHON}"
test -x "${RFD3_BIN}"

echo "denovoval high-temperature generation: task=${TASK_ID} ccd=${CCD} length=${LENGTH}"
echo "step_scale=${STEP_SCALE} gamma_0=${GAMMA_0} noise_scale=${NOISE_SCALE} seed=${SEED}"
echo "diffusion_batch_size=${DIFFUSION_BATCH_SIZE} n_batches=${N_BATCHES} expected_samples=${EXPECTED_SAMPLES}"
echo "input_json=${INPUT_JSON}"
echo "out_dir=${OUT_DIR}"

# (JH) fixed: execute both Python and rfd3 from the mounted uv environment in the active container.
"${RFD3_PYTHON}" "${FOUNDRY_ROOT}/scripts/sherlock_scripts/step2/run_rfd3_inputs.py" \
    --input-json "${INPUT_JSON}" \
    --out-dir "${OUT_DIR}" \
    --ckpt-path "${CKPT_PATH}" \
    --diffusion-batch-size "${DIFFUSION_BATCH_SIZE}" \
    --n-batches "${N_BATCHES}" \
    --step-scale "${STEP_SCALE}" \
    --gamma-0 "${GAMMA_0}" \
    --noise-scale "${NOISE_SCALE}" \
    --seed "${SEED}" \
    --rfd3-cmd "${RFD3_BIN}"

CIF_COUNT="$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.cif.gz' -printf '%f\n' | wc -l)"
JSON_COUNT="$(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.json' -printf '%f\n' | wc -l)"
if [[ "${CIF_COUNT}" -ne "${EXPECTED_SAMPLES}" || "${JSON_COUNT}" -ne "${EXPECTED_SAMPLES}" ]]; then
  echo "Incomplete task output: ccd=${CCD} length=${LENGTH} cif=${CIF_COUNT}/${EXPECTED_SAMPLES} json=${JSON_COUNT}/${EXPECTED_SAMPLES}" >&2
  exit 3
fi

if [[ -n "${SLURM_RETRY_STATUS_ROOT:-}" ]]; then
  TASK_STATUS_DIR="${SLURM_RETRY_STATUS_ROOT}/tasks/$(printf '%06d' "${TASK_ID}")"
  DONE_TMP="${TASK_STATUS_DIR}/done.json.tmp.${SLURM_JOB_ID:-$$}"
  mkdir -p "${TASK_STATUS_DIR}"
  printf '{"task_id":%d,"ccd":"%s","length":%d,"expected_samples":%d,"cif_count":%d,"json_count":%d,"output_dir":"%s","job_id":"%s","attempt":"%s"}\n' \
    "${TASK_ID}" "${CCD}" "${LENGTH}" "${EXPECTED_SAMPLES}" "${CIF_COUNT}" "${JSON_COUNT}" "${OUT_DIR}" \
    "${SLURM_JOB_ID:-}" "${SLURM_RETRY_ATTEMPT:-}" > "${DONE_TMP}"
  mv "${DONE_TMP}" "${TASK_STATUS_DIR}/done.json"
fi

echo "Task complete: ccd=${CCD} length=${LENGTH} cif=${CIF_COUNT} json=${JSON_COUNT}"
