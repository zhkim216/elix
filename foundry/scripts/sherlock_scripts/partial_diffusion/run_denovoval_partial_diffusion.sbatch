#!/usr/bin/env bash
#SBATCH --job-name=rfd3_denovo_pd
#SBATCH --partition=bioe,owners
#SBATCH --constraint=GPU_SKU:A40|GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5
#SBATCH --array=0-153
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR2@300
#SBATCH --export=ALL,FOUNDRY_REQUEUE_ON_USR2=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/zhkim216/job_output/rfd3_denovo_pd_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/rfd3_denovo_pd_%A_%a.err

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "Submit through wrap_sbatch_in_container_foundry_rfd3.sh" >&2
  exit 2
fi

output_root=${OUTPUT_ROOT:-/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/ensembles}
array_index=${ARRAY_INDEX:-$output_root/array_index.txt}
num_samples=${NUM_SAMPLES:-32}
seed=${SEED:-42}
strict_failures=${STRICT_FAILURES:-0}
foundry_sif=${FOUNDRY_SIF:-/scratch/users/zhkim216/containers/foundry-0.2.0-slim-f176bafd42cf.sif}
checkpoint=${CKPT_PATH:-/scratch/users/zhkim216/model_params/rfd3/rfd3_latest.ckpt}
ccd_mirror=${CCD_MIRROR_PATH:-/scratch/users/zhkim216/datasets/ccd_mirror}

for required in "$array_index" "$foundry_sif" "$checkpoint"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 2; }
done
[[ -x "${RFD3_PYTHON:-}" ]] || { echo "Missing RFD3_PYTHON" >&2; exit 2; }
[[ -x "${RFD3_ENV:-}/bin/rfd3" ]] || { echo "Missing RFD3 CLI" >&2; exit 2; }
[[ -d "$ccd_mirror" ]] || { echo "Missing CCD mirror: $ccd_mirror" >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Missing array task ID" >&2; exit 2; }

task_list=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$array_index")
[[ -n "$task_list" && -f "$task_list" ]] || {
  echo "Missing task list for array index ${SLURM_ARRAY_TASK_ID}: $task_list" >&2
  exit 2
}
mapfile -t input_jsons < "$task_list"
task_name=$(basename "$task_list" .txt)
echo "task=${SLURM_ARRAY_TASK_ID} name=${task_name} inputs=${#input_jsons[@]} restart=${SLURM_RESTART_COUNT:-0}"

container_sha256=${CONTAINER_SHA256:-$(sha256sum "$foundry_sif" | awk '{print $1}')}
checkpoint_sha256=${CHECKPOINT_SHA256:-$(sha256sum "$checkpoint" | awk '{print $1}')}
export CCD_MIRROR_PATH="$ccd_mirror"
export FOUNDRY_CHECKPOINT_DIRS="$(dirname -- "$checkpoint")"

failed_inputs=0
for input_json in "${input_jsons[@]}"; do
  if [[ ! -f "$input_json" ]]; then
    echo "Missing prepared input JSON; continuing: $input_json" >&2
    failed_inputs=$((failed_inputs + 1))
    continue
  fi
  command=(
    "$RFD3_PYTHON" "$script_dir/run_partial_diffusion.py"
    --input-json "$input_json"
    --output-root "$output_root"
    --num-samples "$num_samples"
    --checkpoint "$checkpoint"
    --checkpoint-sha256 "$checkpoint_sha256"
    --container "$foundry_sif"
    --container-sha256 "$container_sha256"
    --seed "$seed"
    --rfd3-cmd "$RFD3_ENV/bin/rfd3"
    --resume
  )
  if ! "${command[@]}"; then
    echo "Partial diffusion failed; continuing: $input_json" >&2
    failed_inputs=$((failed_inputs + 1))
  fi
done

status_dir="$output_root/task_attempts"
mkdir -p "$status_dir"
status_tmp="$status_dir/.${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tmp.$$"
status_path="$status_dir/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
printf '{"job_id":"%s","array_job_id":"%s","array_task_id":%d,"restart_count":%d,"task_name":"%s","num_inputs":%d,"failed_inputs":%d}\n' \
  "${SLURM_JOB_ID}" "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}" "${SLURM_ARRAY_TASK_ID}" \
  "${SLURM_RESTART_COUNT:-0}" "$task_name" "${#input_jsons[@]}" "$failed_inputs" > "$status_tmp"
mv "$status_tmp" "$status_path"

if ((failed_inputs > 0)); then
  echo "Task attempted every input with ${failed_inputs} failure(s)." >&2
  if [[ "$strict_failures" == "1" ]]; then
    exit 1
  fi
fi
echo "Task complete: ${task_name}"
