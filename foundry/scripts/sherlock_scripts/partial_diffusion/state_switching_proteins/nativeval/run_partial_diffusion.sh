#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shell_wrapper="$script_dir/../../../setup/shell_in_container_foundry_rfd3.sh"

# Run through the same editable RFD3 container boundary on both interactive and
# batch nodes. The sbatch wrapper sets FOUNDRY_CONTAINER_ACTIVE=1 for us.
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  exec "$shell_wrapper" -- "$script_dir/run_partial_diffusion.sh"
fi

inputs_json=${INPUTS_JSON:-/scratch/users/zhkim216/datasets/evaluation_datasets/state_switching_proteins/nativeval/rfd3_partial/partialt2/inputs.json}
out_dir=/scratch/users/zhkim216/datasets/evaluation_datasets/state_switching_proteins/nativeval/rfd3_partial/partialt2/samples_n32
checkpoint=/scratch/users/zhkim216/model_params/rfd3/rfd3_latest.ckpt
role_id=${ROLE_ID:-}
sampling_log=${SAMPLING_LOG:-$out_dir/sampling.log}

for required_file in "$inputs_json" "$checkpoint"; do
  [[ -f "$required_file" ]] || { echo "Missing required file: $required_file" >&2; exit 2; }
done
[[ -x "${RFD3_BIN:-}" ]] || { echo "Missing RFD3_BIN: ${RFD3_BIN:-unset}" >&2; exit 2; }
[[ -x "${RFD3_PYTHON:-}" ]] || { echo "Missing RFD3_PYTHON: ${RFD3_PYTHON:-unset}" >&2; exit 2; }
[[ -d "${CCD_MIRROR_PATH:-}" ]] || { echo "Missing CCD_MIRROR_PATH: ${CCD_MIRROR_PATH:-unset}" >&2; exit 2; }

mkdir -p "$out_dir"
mkdir -p "$(dirname -- "$sampling_log")"

if [[ -n "$role_id" ]]; then
  "$RFD3_PYTHON" - "$inputs_json" "$role_id" <<'PY'
import json
import sys

inputs_path, role_id = sys.argv[1:]
inputs = json.load(open(inputs_path, encoding="utf-8"))
if role_id not in inputs:
    raise SystemExit(f"ROLE_ID is not present in {inputs_path}: {role_id}")
PY
fi

command=(
  "$RFD3_BIN" design
  "ckpt_path=$checkpoint"
  "inputs=$inputs_json"
  "out_dir=$out_dir"
  diffusion_batch_size=1
  n_batches=32
  seed=42
  skip_existing=true
  prevalidate_inputs=True
  inference_sampler.step_scale=1.5
  inference_sampler.gamma_0=0.6
  inference_sampler.noise_scale=1.003
  inference_sampler.use_classifier_free_guidance=False
  dump_prediction_metadata_json=True
  output_full_json=True
)
if [[ -n "$role_id" ]]; then
  command+=("json_keys_subset=$role_id")
  echo "Selected role: $role_id"
fi

printf 'Running:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}" 2>&1 | tee -a "$sampling_log"
