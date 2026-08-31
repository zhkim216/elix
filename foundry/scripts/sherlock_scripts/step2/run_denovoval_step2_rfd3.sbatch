#!/usr/bin/env bash
#SBATCH --job-name=denovoval_step2_rfd3
#SBATCH --partition=possu,bioe,owners
#SBATCH --array=0-309
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/rfd3_sample_generation/denovoval_step2_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/rfd3_sample_generation/denovoval_step2_%A_%a.err
#
# Diffused-ligand + all-buried RFD3 backbone generation for denovoval step2.
# Task -> (ccd, length): ccd_idx = task / len(LENGTHS), length = LENGTHS[task % len(LENGTHS)].
#
# Full run  (default):  --array=0-<(n_ok_ccds*2 - 1)>   STAGE=full  LENGTHS="150 300" DIFFUSION_BATCH_SIZE=10 N_BATCHES=30
# Pilot run:            --array=0-<(n_ok_ccds - 1)>      STAGE=pilot LENGTHS="150"     DIFFUSION_BATCH_SIZE=2  N_BATCHES=1
# n_ok_ccds = line count of denovoval_step2_ccds.txt (from prepare_rfd3_inputs.py manifest).

set -euo pipefail

# (JH) fixed: reject direct sbatch execution so this script never starts a nested container.
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "This sbatch must run through the Foundry RFD3 container wrapper." >&2
  echo "Submit with: /home/users/zhkim216/code/elix/foundry/scripts/sherlock_scripts/setup/wrap_sbatch_in_container_foundry_rfd3.sh ${BASH_SOURCE[0]}" >&2
  exit 2
fi

STAGE="${STAGE:-full}"
# Accept comma- or space-separated lengths (prepare_rfd3_inputs.py uses commas).
IFS=', ' read -r -a LENGTHS <<< "${LENGTHS:-150 300}"
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-10}"
N_BATCHES="${N_BATCHES:-30}"

if [[ -z "${SCRATCH:-}" ]]; then
  echo "SCRATCH must be set on Sherlock." >&2
  exit 2
fi
if [[ -z "${CKPT_PATH:-}" ]]; then
  echo "CKPT_PATH must point to the RFD3 checkpoint." >&2
  exit 2
fi

# (JH) fixed: container image selection belongs to the outer wrapper; keep the established data selection here.
SOURCE_DATA_ROOT="/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step2/source_data"
INPUTS_ROOT="${INPUTS_ROOT:-${SOURCE_DATA_ROOT}/rfd3_inputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH}/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step2/generated_samples/${STAGE}}"
CCD_LIST_FILE="${CCD_LIST_FILE:-${SOURCE_DATA_ROOT}/denovoval_step2_ccds.txt}"

if [[ ! -f "${CCD_LIST_FILE}" ]]; then
  echo "CCD list not found: ${CCD_LIST_FILE} (run prepare_rfd3_inputs.py first)." >&2
  exit 2
fi
mapfile -t CCDS < "${CCD_LIST_FILE}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LENGTH_COUNT="${#LENGTHS[@]}"
if (( LENGTH_COUNT == 0 )); then
  echo "LENGTHS resolved to zero entries." >&2
  exit 2
fi
TOTAL_TASKS=$(( ${#CCDS[@]} * LENGTH_COUNT ))
if (( TASK_ID >= TOTAL_TASKS )); then
  echo "Task ${TASK_ID} >= ${TOTAL_TASKS} runnable specs (${#CCDS[@]} ccds x ${LENGTH_COUNT} lengths); nothing to do." >&2
  exit 0
fi
CCD="${CCDS[$((TASK_ID / LENGTH_COUNT))]}"
LENGTH="${LENGTHS[$((TASK_ID % LENGTH_COUNT))]}"

INPUT_JSON="${INPUTS_ROOT}/${CCD}_len${LENGTH}.json"
OUT_DIR="${OUTPUT_ROOT}/${CCD}_len${LENGTH}"

# (JH) fixed: require the centralized editable RFD3 environment instead of falling back to the SIF Python.
test -x "${RFD3_PYTHON}"
test -x "${RFD3_BIN}"
test -f "${CKPT_PATH}"
test -d "${CCD_MIRROR_PATH}"

# (JH) fixed: run the consumer and RFD3 console script directly in the already-active container.
"${RFD3_PYTHON}" "${FOUNDRY_ROOT}/scripts/sherlock_scripts/step2/run_rfd3_inputs.py" \
  --input-json "${INPUT_JSON}" \
  --out-dir "${OUT_DIR}" \
  --ckpt-path "${CKPT_PATH}" \
  --diffusion-batch-size "${DIFFUSION_BATCH_SIZE}" \
  --n-batches "${N_BATCHES}" \
  --rfd3-cmd "${RFD3_BIN}"
