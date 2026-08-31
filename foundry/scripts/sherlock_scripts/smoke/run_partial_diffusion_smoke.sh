#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shell_wrapper="$script_dir/../setup/shell_in_container_foundry_rfd3.sh"
helper="$script_dir/partial_diffusion_smoke.py"

foundry_sif=${FOUNDRY_SIF:-/scratch/users/zhkim216/containers/foundry-0.2.0-slim-f176bafd42cf.sif}
checkpoint=${CKPT_PATH:-/scratch/users/zhkim216/model_params/rfd3/rfd3_latest.ckpt}
ccd_mirror=${CCD_MIRROR_PATH:-/scratch/users/zhkim216/datasets/ccd_mirror}
source_cif=${SOURCE_CIF:-/scratch/users/zhkim216/out_dir/benchmarks/nativeval/ligandmpnn/sequence_design/staging/full/semantic_cifs/8r5n_binder_A_1_context_C_1.cif}
output_root=${OUTPUT_ROOT:-/scratch/users/zhkim216/out_dir/benchmarks/nativeval/rfd3_partial/smoke_atp_partialt5}
sample_id=8r5n_binder_A_1_context_C_1
seed=${SEED:-42}
overwrite=${OVERWRITE:-0}

# (JH) fixed: Host smoke calls enter the shared outer container exactly once.
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  if [[ ! -x "$shell_wrapper" ]]; then
    echo "Missing Foundry/RFD3 shell wrapper: $shell_wrapper" >&2
    exit 2
  fi
  export FOUNDRY_SIF="$foundry_sif"
  export CKPT_PATH="$checkpoint"
  export CCD_MIRROR_PATH="$ccd_mirror"
  export SOURCE_CIF="$source_cif"
  export OUTPUT_ROOT="$output_root"
  export SEED="$seed"
  export OVERWRITE="$overwrite"
  export FOUNDRY_CHECKPOINT_DIRS="$(dirname -- "$checkpoint")"
  exec "$shell_wrapper" -- "$script_dir/run_partial_diffusion_smoke.sh" "$@"
fi

# (JH) fixed: Smoke helpers and CLI must come from the editable RFD3 environment.
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

for required_path in "$foundry_sif" "$checkpoint" "$source_cif" "$script_dir/partial_diffusion_smoke.py"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required file: $required_path" >&2
    exit 2
  fi
done
if [[ "$overwrite" != "0" && "$overwrite" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1, got: $overwrite" >&2
  exit 2
fi

input_dir="$output_root/inputs"
aligned_dir="$output_root/aligned"
input_json="$input_dir/atp_four_conditions_partialt5.json"
preflight_json="$output_root/runtime_preflight_partialt5.json"
sampling_log="$output_root/rfd3_sampling_partialt5.log"
manifest_json="$output_root/smoke_manifest_partialt5.json"

if [[ "$overwrite" == "0" ]] && find "$aligned_dir" -maxdepth 1 -type f -name '*_partialt5.*' -print -quit 2>/dev/null | grep -q .; then
  echo "Aligned partial_t=5 outputs already exist; set OVERWRITE=1 to replace them." >&2
  exit 2
fi

mkdir -p "$input_dir" "$aligned_dir" "$output_root"
raw_dir=$(mktemp -d "$output_root/.raw_partialt5.XXXXXX")
cleanup_raw() {
  status=$?
  if [[ $status -eq 0 ]]; then
    rm -rf -- "$raw_dir"
  else
    echo "Smoke failed; preserving raw outputs for diagnosis: $raw_dir" >&2
  fi
}
trap cleanup_raw EXIT

container_sha256=$(sha256sum "$foundry_sif" | awk '{print $1}')
checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')

# (JH) fixed: Helpers run directly because container setup is owned by the caller.
export CCD_MIRROR_PATH="$ccd_mirror"
export FOUNDRY_CHECKPOINT_DIRS="$(dirname -- "$checkpoint")"
run_helper() {
  if [[ "$overwrite" == "1" ]]; then
    "$RFD3_PYTHON" "$helper" "$@" --overwrite
  else
    "$RFD3_PYTHON" "$helper" "$@"
  fi
}

"$RFD3_PYTHON" "$helper" preflight \
  --checkpoint "$checkpoint" | tee "$preflight_json"

run_helper prepare \
  --source "$source_cif" \
  --output "$input_json"

"$RFD3_ENV/bin/rfd3" design \
  "ckpt_path=$checkpoint" \
  "inputs=$input_json" \
  "out_dir=$raw_dir" \
  diffusion_batch_size=1 \
  n_batches=1 \
  "seed=$seed" \
  skip_existing=False \
  prevalidate_inputs=True \
  inference_sampler.use_classifier_free_guidance=False \
  dump_prediction_metadata_json=True \
  output_full_json=True 2>&1 | tee "$sampling_log"

conditions=(
  protein_diffusion_sequence_fixed
  protein_atp_diffusion_sequence_fixed
  protein_diffusion_sequence_unfixed
  protein_atp_diffusion_sequence_unfixed
)
input_stem=${input_json##*/}
input_stem=${input_stem%.json}
for condition in "${conditions[@]}"; do
  raw_base="$raw_dir/${input_stem}_${condition}_0_model_0"
  raw_cif="$raw_base.cif.gz"
  raw_json="$raw_base.json"
  final_stem="${sample_id}__${condition}_ca_aligned_partialt5"
  final_cif="$aligned_dir/$final_stem.cif.gz"
  final_json="$aligned_dir/$final_stem.json"
  if [[ ! -f "$raw_cif" || ! -f "$raw_json" ]]; then
    echo "Missing raw RFD3 output pair for $condition: $raw_base" >&2
    exit 1
  fi
  run_helper align \
    --condition "$condition" \
    --reference "$source_cif" \
    --prediction "$raw_cif" \
    --prediction-metadata "$raw_json" \
    --output-cif "$final_cif" \
    --output-json "$final_json" \
    --container "$foundry_sif" \
    --container-sha256 "$container_sha256" \
    --checkpoint "$checkpoint" \
    --checkpoint-sha256 "$checkpoint_sha256"
done

run_helper summarize \
  --aligned-dir "$aligned_dir" \
  --sample-id "$sample_id" \
  --preflight-json "$preflight_json" \
  --output "$manifest_json"

echo "Partial-diffusion smoke complete: $manifest_json"
