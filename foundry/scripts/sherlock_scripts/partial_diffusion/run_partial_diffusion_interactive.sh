#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_partial_diffusion_interactive.sh \
    --output-root PATH --role-id ROLE_ID --num-samples N \
    [--partial-ts 2,5,10] [--seed 42] [--overwrite]

The output root must already contain preparation_manifest.json and partialt*/inputs/.
EOF
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shell_wrapper="$script_dir/../setup/shell_in_container_foundry_rfd3.sh"
original_args=("$@")
output_root=
role_id=
num_samples=
partial_ts=2,5,10
seed=42
overwrite=0
foundry_sif=${FOUNDRY_SIF:-/scratch/users/zhkim216/containers/foundry-0.2.0-slim-f176bafd42cf.sif}
checkpoint=${CKPT_PATH:-/scratch/users/zhkim216/model_params/rfd3/rfd3_latest.ckpt}
ccd_mirror=${CCD_MIRROR_PATH:-/scratch/users/zhkim216/datasets/ccd_mirror}

while (($#)); do
  case "$1" in
    --output-root) output_root=$2; shift 2 ;;
    --role-id) role_id=$2; shift 2 ;;
    --num-samples) num_samples=$2; shift 2 ;;
    --partial-ts) partial_ts=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --foundry-sif) foundry_sif=$2; shift 2 ;;
    --checkpoint) checkpoint=$2; shift 2 ;;
    --overwrite) overwrite=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$output_root" || -z "$role_id" || -z "$num_samples" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$num_samples" =~ ^[1-9][0-9]*$ ]]; then
  echo "--num-samples must be a positive integer" >&2
  exit 2
fi

# (JH) fixed: Host calls delegate the complete sweep once, preventing nested containers.
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  if [[ ! -x "$shell_wrapper" ]]; then
    echo "Missing Foundry/RFD3 shell wrapper: $shell_wrapper" >&2
    exit 2
  fi
  export FOUNDRY_SIF="$foundry_sif"
  export CKPT_PATH="$checkpoint"
  export CCD_MIRROR_PATH="$ccd_mirror"
  export FOUNDRY_CHECKPOINT_DIRS="$(dirname -- "$checkpoint")"
  exec "$shell_wrapper" -- "$script_dir/run_partial_diffusion_interactive.sh" "${original_args[@]}"
fi

# (JH) fixed: The outer container must expose the standalone editable RFD3 environment.
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
for required in "$foundry_sif" "$checkpoint" "$output_root/preparation_manifest.json"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 2
  fi
done
if [[ ! -d "$ccd_mirror" ]]; then
  echo "Missing CCD mirror: $ccd_mirror" >&2
  exit 2
fi

container_sha256=$(sha256sum "$foundry_sif" | awk '{print $1}')
checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
IFS=',' read -r -a partial_t_values <<< "$partial_ts"

# (JH) fixed: Sampling subprocesses resolve from the same RFD3 environment.
export CCD_MIRROR_PATH="$ccd_mirror"
export FOUNDRY_CHECKPOINT_DIRS="$(dirname -- "$checkpoint")"

for partial_t in "${partial_t_values[@]}"; do
  if [[ ! "$partial_t" =~ ^[1-9][0-9]*$ ]]; then
    echo "Partial-t values must be positive integers: $partial_t" >&2
    exit 2
  fi
  condition="partialt${partial_t}"
  input_json="$output_root/$condition/inputs/$role_id.json"
  if [[ ! -f "$input_json" ]]; then
    echo "Missing prepared input JSON: $input_json" >&2
    exit 2
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
  )
  if [[ "$overwrite" == "1" ]]; then
    command+=(--overwrite)
  fi
  echo "Running $condition for $role_id ($num_samples samples)"
  "${command[@]}"
done
