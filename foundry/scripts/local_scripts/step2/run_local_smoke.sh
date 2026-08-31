#!/usr/bin/env bash
# Local smoke for denovoval step2 diffused-ligand + all-buried RFD3 generation.
#
# Runs prep + rfd3 for 4 representative CCDs (SM / metal ion / organometallic cofactor /
# Fe-S cluster) at len150 with n=2, on the local GPU, using the foundry_rfd3 micromamba env
# (py3.12 with rc-foundry installed editable -> rfd3 + atomworks). Validates CIF writing,
# prevalidate, CFG RASA burial, and backbone emission before the Sherlock pilot/full runs.
#
# Canonical pipeline code lives in scripts/sherlock_scripts/step2/.

set -euo pipefail

# foundry_rfd3: py3.12 micromamba env with rc-foundry installed editable (== the foundry repo
# source), so `-m rfd3.cli` runs the same code as Sherlock's SIF. No PYTHONPATH shim needed.
ENV_PY="${ENV_PY:-/home/yjhk/model-dev/envs/micromamba/envs/foundry_rfd3/bin/python}"
REPO_ROOT="${REPO_ROOT:-/home/yjhk/model-dev/foundry}"
CKPT_PATH="${CKPT_PATH:-${HOME}/.foundry/checkpoints/rfd3_latest.ckpt}"
export CCD_MIRROR_PATH="${CCD_MIRROR_PATH:-/home/yjhk/model-dev/datasets/ccd_mirror}"

STEP2_DIR="${REPO_ROOT}/scripts/sherlock_scripts/step2"
CURATION_STEP2="/home/yjhk/model-dev/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step2"
SOURCE_DIR="${CURATION_STEP2}/source_data"
OUT_ROOT="${CURATION_STEP2}/generated_samples/smoke"

CCDS=(ATP ZN HEM SF4)
DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-2}"
N_BATCHES="${N_BATCHES:-1}"

test -f "${CKPT_PATH}" || { echo "RFD3 checkpoint not found: ${CKPT_PATH}" >&2; exit 2; }
test -d "${CCD_MIRROR_PATH}" || { echo "CCD mirror not found: ${CCD_MIRROR_PATH}" >&2; exit 2; }

# 1. Stage CIFs + JSONs for the 4 smoke CCDs (len150).
ccd_args=()
for ccd in "${CCDS[@]}"; do ccd_args+=(--ccd "${ccd}"); done
"${ENV_PY}" "${STEP2_DIR}/prepare_rfd3_inputs.py" \
  --source-dir "${SOURCE_DIR}" \
  --ccd-mirror "${CCD_MIRROR_PATH}" \
  --lengths 150 \
  "${ccd_args[@]}" \
  --overwrite

# 2. Run RFD3 for each smoke CCD.
for ccd in "${CCDS[@]}"; do
  "${ENV_PY}" "${STEP2_DIR}/run_rfd3_inputs.py" \
    --input-json "${SOURCE_DIR}/rfd3_inputs/${ccd}_len150.json" \
    --out-dir "${OUT_ROOT}/${ccd}_len150" \
    --ckpt-path "${CKPT_PATH}" \
    --diffusion-batch-size "${DIFFUSION_BATCH_SIZE}" \
    --n-batches "${N_BATCHES}" \
    --rfd3-cmd "${ENV_PY} -m rfd3.cli"
done

echo "Local smoke complete. Outputs under ${OUT_ROOT}/"
