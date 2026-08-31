#!/usr/bin/env bash
#SBATCH --job-name=rfd3_atp_temp_smoke
#SBATCH --partition=possu,bioe,owners
#SBATCH --array=0-11%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/rfd3_sample_generation/rfd3_atp_temp_smoke_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/rfd3_sample_generation/rfd3_atp_temp_smoke_%A_%a.err
#
# ATP-only RFdiffusion3 temperature-grid smoke:
#   step_scale in {1.25, 1.0}
#   gamma_0 in {0.6, 0.8, 1.0}
#   length in {150, 300}
#   two samples per condition with a shared seed.

set -euo pipefail

# (JH) fixed: reject direct sbatch execution so the ATP smoke cannot start a nested container.
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "This sbatch must run through the Foundry RFD3 container wrapper." >&2
  echo "Submit with: /home/users/zhkim216/code/elix/foundry/scripts/sherlock_scripts/setup/wrap_sbatch_in_container_foundry_rfd3.sh ${BASH_SOURCE[0]}" >&2
  exit 2
fi

STEP_SCALES=(1.25 1.25 1.25 1.0 1.0 1.0)
GAMMA_ZEROES=(0.6 0.8 1.0 0.6 0.8 1.0)
SETTING_TAGS=(
  ss1p25_g0p6
  ss1p25_g0p8
  ss1p25_g1p0
  ss1p0_g0p6
  ss1p0_g0p8
  ss1p0_g1p0
)
LENGTHS=(150 300)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LENGTH_COUNT="${#LENGTHS[@]}"
SETTING_ID=$((TASK_ID / LENGTH_COUNT))
LENGTH_ID=$((TASK_ID % LENGTH_COUNT))

if (( SETTING_ID >= ${#SETTING_TAGS[@]} )); then
  echo "Task ${TASK_ID} is outside the ATP temperature grid." >&2
  exit 2
fi

STEP_SCALE="${STEP_SCALES[$SETTING_ID]}"
GAMMA_0="${GAMMA_ZEROES[$SETTING_ID]}"
SETTING_TAG="${SETTING_TAGS[$SETTING_ID]}"
LENGTH="${LENGTHS[$LENGTH_ID]}"

SOURCE_DATA_ROOT="/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step2/source_data"
INPUT_JSON="${SOURCE_DATA_ROOT}/rfd3_inputs/ATP_len${LENGTH}.json"
OUTPUT_ROOT="/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval_temperature_pilot/atp_smoke/generated_samples"
OUT_DIR="${OUTPUT_ROOT}/${SETTING_TAG}/ATP_len${LENGTH}"

DIFFUSION_BATCH_SIZE=2
N_BATCHES=1
NOISE_SCALE=1.003
SEED=216

if [[ "${PRINT_TASK_PLAN:-false}" == "true" ]]; then
  echo "${TASK_ID} ${SETTING_TAG} ${LENGTH} ${STEP_SCALE} ${GAMMA_0} ${OUT_DIR}"
  exit 0
fi

test -f "${INPUT_JSON}"
test -f "${CKPT_PATH}"
test -d "${CCD_MIRROR_PATH}"
# (JH) fixed: fail closed when the standalone editable RFD3 environment was not activated.
test -x "${RFD3_PYTHON}"
test -x "${RFD3_BIN}"

echo "ATP temperature smoke: task=${TASK_ID} setting=${SETTING_TAG} length=${LENGTH}"
echo "step_scale=${STEP_SCALE} gamma_0=${GAMMA_0} noise_scale=${NOISE_SCALE} seed=${SEED}"
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
