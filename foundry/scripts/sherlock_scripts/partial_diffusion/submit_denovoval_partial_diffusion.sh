#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
wrapper="$script_dir/../setup/wrap_sbatch_in_container_foundry_rfd3.sh"
prepare_job_script="$script_dir/prepare_denovoval_inputs.sbatch"
run_job_script="$script_dir/run_denovoval_partial_diffusion.sbatch"
finalize_job_script="$script_dir/finalize_denovoval_partial_diffusion.sbatch"
output_root=${OUTPUT_ROOT:-/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/ensembles}
foundry_sif=${FOUNDRY_SIF:-/scratch/users/zhkim216/containers/foundry-0.2.0-slim-f176bafd42cf.sif}
checkpoint=${CKPT_PATH:-/scratch/users/zhkim216/model_params/rfd3/rfd3_latest.ckpt}
dry_run=0

if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
if (($#)); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
for required in "$wrapper" "$prepare_job_script" "$run_job_script" "$finalize_job_script" "$foundry_sif" "$checkpoint"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done

container_sha256=$(sha256sum "$foundry_sif" | awk '{print $1}')
checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')
common_export="ALL,OUTPUT_ROOT=$output_root,CONTAINER_SHA256=$container_sha256,CHECKPOINT_SHA256=$checkpoint_sha256"

if ((dry_run)); then
  "$wrapper" --dry-run \
    --sbatch-arg="--export=ALL,OUTPUT_ROOT=$output_root,PREPARE_OVERWRITE=1" \
    "$prepare_job_script"
  "$wrapper" --dry-run --sbatch-arg=--array=0-0 \
    --sbatch-arg="--export=$common_export,ARRAY_INDEX=$output_root/smoke_array_index.txt,STRICT_FAILURES=1" \
    "$run_job_script"
  "$wrapper" --dry-run --sbatch-arg=--array=0-153 \
    --sbatch-arg="--export=$common_export,ARRAY_INDEX=$output_root/array_index.txt,STRICT_FAILURES=0" \
    "$run_job_script"
  printf 'sbatch --dependency=afterany:<full_job_id> --export=ALL,OUTPUT_ROOT=%q %q\n' \
    "$output_root" "$finalize_job_script"
  exit 0
fi

mkdir -p "$output_root"

submit_wrapped() {
  local output job_id
  output=$("$wrapper" "$@")
  printf '%s\n' "$output" >&2
  job_id=$(printf '%s\n' "$output" | awk '
    /^[0-9]+(;.*)?$/ {split($1, p, ";"); id=p[1]}
    /^Submitted batch job [0-9]+$/ {id=$4}
    END {if (id == "") exit 1; print id}
  ')
  printf '%s\n' "$job_id"
}

prep_job_id=$(submit_wrapped \
  --sbatch-arg=--parsable \
  --sbatch-arg="--export=ALL,OUTPUT_ROOT=$output_root,PREPARE_OVERWRITE=1" \
  "$prepare_job_script")
smoke_job_id=$(submit_wrapped \
  --sbatch-arg=--parsable \
  --sbatch-arg=--array=0-0 \
  --sbatch-arg="--dependency=afterok:$prep_job_id" \
  --sbatch-arg=--job-name=rfd3_denovo_smoke \
  --sbatch-arg="--export=$common_export,ARRAY_INDEX=$output_root/smoke_array_index.txt,STRICT_FAILURES=1" \
  "$run_job_script")
full_job_id=$(submit_wrapped \
  --sbatch-arg=--parsable \
  --sbatch-arg=--array=0-153 \
  --sbatch-arg="--dependency=afterok:$smoke_job_id" \
  --sbatch-arg=--job-name=rfd3_denovo_full \
  --sbatch-arg="--export=$common_export,ARRAY_INDEX=$output_root/array_index.txt,STRICT_FAILURES=0" \
  "$run_job_script")
finalizer_job_id=$(sbatch --parsable \
  --dependency="afterany:$full_job_id" \
  --export="ALL,OUTPUT_ROOT=$output_root" \
  "$finalize_job_script")
finalizer_job_id=${finalizer_job_id%%;*}

submitted_at=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
submission_tmp="$output_root/.submission.json.tmp.$$"
submission_path="$output_root/submission.json"
printf '{\n  "submitted_at_utc": "%s",\n  "output_root": "%s",\n  "prepare_job_id": "%s",\n  "smoke_job_id": "%s",\n  "full_job_id": "%s",\n  "finalizer_job_id": "%s",\n  "container_sha256": "%s",\n  "checkpoint_sha256": "%s"\n}\n' \
  "$submitted_at" "$output_root" "$prep_job_id" "$smoke_job_id" "$full_job_id" \
  "$finalizer_job_id" "$container_sha256" "$checkpoint_sha256" > "$submission_tmp"
mv "$submission_tmp" "$submission_path"

echo "prepare_job_id=$prep_job_id"
echo "smoke_job_id=$smoke_job_id"
echo "full_job_id=$full_job_id"
echo "finalizer_job_id=$finalizer_job_id"
echo "submission_manifest=$submission_path"
