#!/usr/bin/env bash
#SBATCH --job-name=denovoval_step2_smoke
#SBATCH --partition=possu,bioe
#SBATCH --array=0-9
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/rfd3_sample_generation/denovoval_step2_smoke_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/rfd3_sample_generation/denovoval_step2_smoke_%A_%a.err
#
# Ten-task Sherlock smoke for denovoval step2 RFD3 sampling.
# Default: first 10 staged CCDs, length 150, one design each.
# Requires prepare_rfd3_inputs.py to have already written source_data/rfd3_inputs/
# and source_data/denovoval_step2_ccds.txt.

set -euo pipefail

# (JH) fixed: keep the smoke chained to the full launcher while enforcing the same outer-container entrypoint.
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "This sbatch must run through the Foundry RFD3 container wrapper." >&2
  echo "Submit with: /home/users/zhkim216/code/elix/foundry/scripts/sherlock_scripts/setup/wrap_sbatch_in_container_foundry_rfd3.sh ${BASH_SOURCE[0]}" >&2
  exit 2
fi

export STAGE="${STAGE:-smoke10_len150}"
export LENGTHS="${LENGTHS:-150}"
export DIFFUSION_BATCH_SIZE="${DIFFUSION_BATCH_SIZE:-1}"
export N_BATCHES="${N_BATCHES:-1}"

exec bash "${FOUNDRY_ROOT}/scripts/sherlock_scripts/step2/run_denovoval_step2_rfd3.sbatch"
