#!/usr/bin/env bash
#SBATCH --job-name=rfd3_partial
#SBATCH --partition=possu,bioe,owners
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/rfd3_partial_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/rfd3_partial_%A_%a.err

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# (JH) fixed: This job must be submitted through the single outer container wrapper.
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "Run this sbatch through the Foundry/RFD3 wrapper:" >&2
  echo "  bash scripts/sherlock_scripts/setup/wrap_sbatch_in_container_foundry_rfd3.sh scripts/sherlock_scripts/partial_diffusion/run_partial_diffusion.sbatch" >&2
  exit 2
fi

output_root=${OUTPUT_ROOT:-/scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/ensembles}
array_index=${ARRAY_INDEX:-$output_root/array_index.txt}
num_samples=${NUM_SAMPLES:-32}
diffusion_batch_size=${DIFFUSION_BATCH_SIZE:-1}
n_batches=${N_BATCHES:-}
seed=${SEED:-42}
step_scale=${STEP_SCALE:-1.5}
gamma_0=${GAMMA_0:-0.6}
noise_scale=${NOISE_SCALE:-1.003}
max_attempts=${MAX_ATTEMPTS:-1}
foundry_sif=${FOUNDRY_SIF:-/scratch/users/zhkim216/containers/foundry-0.2.0-slim-f176bafd42cf.sif}
checkpoint=${CKPT_PATH:-/scratch/users/zhkim216/model_params/rfd3/rfd3_latest.ckpt}
ccd_mirror=${CCD_MIRROR_PATH:-/scratch/users/zhkim216/datasets/ccd_mirror}

# (JH) fixed: Fail closed if activation did not expose the standalone RFD3 environment.
if [[ -z "${RFD3_ENV:-}" || ! -d "$RFD3_ENV" ]]; then
  echo "Missing RFD3 environment: ${RFD3_ENV:-unset}" >&2
  exit 2
fi
if [[ -z "${RFD3_PYTHON:-}" || ! -x "$RFD3_PYTHON" ]]; then
  echo "Missing RFD3 Python: ${RFD3_PYTHON:-unset}" >&2
  exit 2
fi
if [[ ! -x "$RFD3_ENV/bin/rfd3" ]]; then
  echo "Missing RFD3 CLI: $RFD3_ENV/bin/rfd3" >&2
  exit 2
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Submit this script as an array over zero-based lines in ARRAY_INDEX." >&2
  exit 2
fi
for required in "$array_index" "$foundry_sif" "$checkpoint"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done
shard_list=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$array_index")
if [[ -z "$shard_list" || ! -f "$shard_list" ]]; then
  echo "No shard list for array index $SLURM_ARRAY_TASK_ID: $shard_list" >&2
  exit 2
fi
mapfile -t input_jsons < "$shard_list"
if [[ ${#input_jsons[@]} -eq 0 ]]; then
  echo "No prepared inputs in $shard_list" >&2
  exit 2
fi

container_sha256=${CONTAINER_SHA256:-$(sha256sum "$foundry_sif" | awk '{print $1}')}
checkpoint_sha256=${CHECKPOINT_SHA256:-$(sha256sum "$checkpoint" | awk '{print $1}')}

# (JH) fixed: Execute the repo helper directly; container ownership stays in the wrapper.
export CCD_MIRROR_PATH="$ccd_mirror"
export FOUNDRY_CHECKPOINT_DIRS="$(dirname -- "$checkpoint")"
# (JH) fixed: attempt every input in the shard, then leave the task failed if any input failed.
failed_inputs=0
for input_json in "${input_jsons[@]}"; do
  if [[ ! -f "$input_json" ]]; then
    echo "Missing prepared input JSON: $input_json" >&2
    exit 2
  fi
  command=(
    "$RFD3_PYTHON" "$script_dir/run_partial_diffusion.py"
    --input-json "$input_json"
    --output-root "$output_root"
    --num-samples "$num_samples"
    --diffusion-batch-size "$diffusion_batch_size"
    --checkpoint "$checkpoint"
    --checkpoint-sha256 "$checkpoint_sha256"
    --container "$foundry_sif"
    --container-sha256 "$container_sha256"
    --seed "$seed"
    --step-scale "$step_scale"
    --gamma-0 "$gamma_0"
    --noise-scale "$noise_scale"
    --rfd3-cmd "$RFD3_ENV/bin/rfd3"
  )
  if [[ -n "$n_batches" ]]; then
    command+=(--n-batches "$n_batches")
  fi
  if [[ "${OVERWRITE:-0}" == "1" ]]; then
    command+=(--overwrite)
  fi
  if [[ "${RESUME:-0}" == "1" ]]; then
    command+=(--resume)
  fi
  succeeded=0
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    attempt_command=("${command[@]}")
    if ((attempt > 1)) && [[ "${OVERWRITE:-0}" != "1" && "${RESUME:-0}" != "1" ]]; then
      attempt_command+=(--resume)
    fi
    if "${attempt_command[@]}"; then
      succeeded=1
      break
    fi
    echo "Partial diffusion attempt $attempt/$max_attempts failed: $input_json" >&2
  done
  if ((succeeded == 0)); then
    echo "Partial diffusion failed; continuing with the remaining shard inputs: $input_json" >&2
    failed_inputs=$((failed_inputs + 1))
  fi
done

if ((failed_inputs > 0)); then
  echo "Shard completed with $failed_inputs failed input(s)." >&2
  exit 1
fi
