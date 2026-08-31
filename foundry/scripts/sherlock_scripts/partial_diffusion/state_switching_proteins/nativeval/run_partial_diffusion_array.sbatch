#!/usr/bin/env bash
#SBATCH --job-name=ss_nativeval_pd_t2_n32_array
#SBATCH --partition=possu
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --array=0-49%7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR2@300
#SBATCH --export=ALL,FOUNDRY_REQUEUE_ON_USR2=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/zhkim216/job_output/ss_nativeval_pd_t2_n32_array_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/ss_nativeval_pd_t2_n32_array_%A_%a.err

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "Submit through the Foundry/RFD3 sbatch wrapper:" >&2
  echo "  bash $script_dir/../../../setup/wrap_sbatch_in_container_foundry_rfd3.sh ${BASH_SOURCE[0]}" >&2
  exit 2
fi

[[ -x "${RFD3_PYTHON:-}" ]] || { echo "Missing RFD3_PYTHON: ${RFD3_PYTHON:-unset}" >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Missing SLURM_ARRAY_TASK_ID" >&2; exit 2; }

inputs_json=/scratch/users/zhkim216/datasets/evaluation_datasets/state_switching_proteins/nativeval/rfd3_partial/partialt2/inputs.json
out_dir=/scratch/users/zhkim216/datasets/evaluation_datasets/state_switching_proteins/nativeval/rfd3_partial/partialt2/samples_n32

role_id=$("$RFD3_PYTHON" - "$inputs_json" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys

inputs_path, task_id_text = sys.argv[1:]
role_ids = list(json.load(open(inputs_path, encoding="utf-8")))
if len(role_ids) != 50:
    raise SystemExit(f"Expected 50 roles in {inputs_path}, found {len(role_ids)}")
task_id = int(task_id_text)
if not 0 <= task_id < len(role_ids):
    raise SystemExit(f"Array task {task_id} is outside [0, {len(role_ids) - 1}]")
print(role_ids[task_id])
PY
)

export INPUTS_JSON="$inputs_json"
export ROLE_ID="$role_id"
export SAMPLING_LOG="$out_dir/array_logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${role_id}.log"

echo "array_job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID} role=$role_id"
exec "$script_dir/run_partial_diffusion.sh"
