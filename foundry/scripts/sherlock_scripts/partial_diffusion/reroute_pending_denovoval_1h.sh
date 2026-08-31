#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
wrapper="$script_dir/../setup/wrap_sbatch_in_container_foundry_rfd3.sh"
builder="$script_dir/build_denovoval_1h_tasks.py"
run_job_script="$script_dir/run_denovoval_partial_diffusion.sbatch"
finalize_job_script="$script_dir/finalize_denovoval_partial_diffusion.sbatch"
output_root=${OUTPUT_ROOT:-/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/ensembles}
original_job_id=35175515
old_finalizer_job_id=35175516
execute=0

usage() {
  cat >&2 <<EOF
Usage: $0 [--execute] [--original-job-id ID] [--old-finalizer-job-id ID]

Without --execute, print the current live-array split without changing Slurm state.
EOF
}

while (($#)); do
  case "$1" in
    --execute) execute=1; shift ;;
    --original-job-id) original_job_id=$2; shift 2 ;;
    --old-finalizer-job-id) old_finalizer_job_id=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

for required in "$wrapper" "$builder" "$run_job_script" "$finalize_job_script" \
  "$output_root/array_index.txt" "$output_root/submission.json"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 2; }
done

job_description=$(scontrol show job -o "$original_job_id")
grep -q 'JobName=rfd3_denovo_full' <<< "$job_description" || {
  echo "Unexpected original job: $original_job_id" >&2
  exit 2
}

held_original=0
new_job_id=""
new_finalizer_job_id=""
transition_complete=0
original_tasks_cancelled=0
cleanup_on_error() {
  status=$?
  if ((transition_complete == 0)); then
    if ((original_tasks_cancelled == 0)); then
      if [[ -n "$new_job_id" ]]; then
        scancel "$new_job_id" 2>/dev/null || true
      fi
      if [[ -n "$new_finalizer_job_id" ]]; then
        scancel "$new_finalizer_job_id" 2>/dev/null || true
      fi
    else
      echo "Original pending tasks were cancelled; leaving replacement job ${new_job_id:-unknown} held for recovery." >&2
    fi
    if ((held_original)); then
      scontrol release "$original_job_id" 2>/dev/null || true
    fi
  fi
  exit "$status"
}
trap cleanup_on_error ERR INT TERM

if ((execute)); then
  echo "Holding pending components of original array $original_job_id."
  scontrol hold "$original_job_id"
  held_original=1
fi

mapfile -t active_rows < <(
  squeue -h -r -j "$original_job_id" -o '%i|%T|%R' | sort -t_ -k2,2n
)
pending_task_ids=()
preserved_task_ids=()
for row in "${active_rows[@]}"; do
  IFS='|' read -r slurm_task_id state reason <<< "$row"
  if [[ ! "$slurm_task_id" =~ ^${original_job_id}_([0-9]+)$ ]]; then
    echo "Unexpected array task ID: $slurm_task_id" >&2
    exit 2
  fi
  task_id=${BASH_REMATCH[1]}
  if [[ "$state" == "PENDING" ]]; then
    pending_task_ids+=("$task_id")
  else
    preserved_task_ids+=("$task_id")
  fi
done
if ((${#pending_task_ids[@]} == 0)); then
  echo "No pending original tasks remain."
  if ((held_original)); then
    scontrol release "$original_job_id" || true
  fi
  trap - ERR INT TERM
  exit 0
fi

pending_csv=$(IFS=,; echo "${pending_task_ids[*]}")
printf 'original_job_id=%s\n' "$original_job_id"
printf 'pending_task_count=%d\n' "${#pending_task_ids[@]}"
printf 'preserved_active_task_count=%d\n' "${#preserved_task_ids[@]}"
printf 'pending_task_ids=%s\n' "$pending_csv"
if ((execute == 0)); then
  echo "Dry run only; no scheduler or file mutation performed."
  trap - ERR INT TERM
  exit 0
fi

run_stamp=$(date --utc +%Y%m%dT%H%M%SZ)
reroute_dir="$output_root/reroutes/one_hour_$run_stamp"
/usr/bin/python3 "$builder" \
  --output-root "$output_root" \
  --output-dir "$reroute_dir" \
  --original-task-ids "$pending_csv" \
  --source-cifs-per-task 4 >/dev/null

new_task_count=$(jq -r '.n_new_array_tasks' "$reroute_dir/reshard_manifest.json")
if [[ ! "$new_task_count" =~ ^[1-9][0-9]*$ ]] || ((new_task_count > 1000)); then
  echo "Invalid new array task count: $new_task_count" >&2
  exit 2
fi

container_sha256=$(jq -r '.container_sha256' "$output_root/submission.json")
checkpoint_sha256=$(jq -r '.checkpoint_sha256' "$output_root/submission.json")
[[ "$container_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid container digest" >&2; exit 2; }
[[ "$checkpoint_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid checkpoint digest" >&2; exit 2; }
common_export="ALL,OUTPUT_ROOT=$output_root,CONTAINER_SHA256=$container_sha256,CHECKPOINT_SHA256=$checkpoint_sha256,ARRAY_INDEX=$reroute_dir/array_index.txt,STRICT_FAILURES=0"
gpu_constraint='GPU_SKU:A40|GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5'

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

new_job_id=$(submit_wrapped \
  --sbatch-arg=--parsable \
  --sbatch-arg=--hold \
  --sbatch-arg="--array=0-$((new_task_count - 1))" \
  --sbatch-arg=--time=01:00:00 \
  --sbatch-arg=--partition=bioe,owners \
  --sbatch-arg="--constraint=$gpu_constraint" \
  --sbatch-arg=--job-name=rfd3_denovo_1h \
  --sbatch-arg="--export=$common_export" \
  "$run_job_script")

cancel_targets=()
for task_id in "${pending_task_ids[@]}"; do
  cancel_targets+=("${original_job_id}_${task_id}")
done
scancel "${cancel_targets[@]}"
original_tasks_cancelled=1

declare -A selected_targets=()
for target in "${cancel_targets[@]}"; do
  selected_targets["$target"]=1
done
remaining_targets=""
for check_attempt in $(seq 1 15); do
  remaining_targets=""
  while IFS='|' read -r active_task_id active_state; do
    if [[ -n "${selected_targets[$active_task_id]+selected}" ]]; then
      remaining_targets+="${active_task_id}|${active_state}"$'\n'
    fi
  done < <(squeue -h -r -j "$original_job_id" -o '%i|%T')
  if [[ -z "$remaining_targets" ]]; then
    break
  fi
  sleep 1
done
if [[ -n "$remaining_targets" ]]; then
  echo "Selected original tasks remain after cancellation:" >&2
  printf '%s' "$remaining_targets" >&2
  exit 2
fi

new_finalizer_job_id=$(sbatch --parsable \
  --dependency="afterany:$original_job_id:$new_job_id" \
  --export="ALL,OUTPUT_ROOT=$output_root" \
  "$finalize_job_script")
new_finalizer_job_id=${new_finalizer_job_id%%;*}
scancel "$old_finalizer_job_id"

scontrol release "$original_job_id" 2>/dev/null || true
held_original=0
scontrol release "$new_job_id"

preserved_json=$(printf '%s\n' "${preserved_task_ids[@]}" | jq -Rsc \
  'split("\n") | map(select(length > 0) | tonumber)')
submitted_at=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
submission_tmp="$reroute_dir/.submission.json.tmp.$$"
jq -n \
  --arg submitted_at_utc "$submitted_at" \
  --arg output_root "$output_root" \
  --arg reroute_dir "$reroute_dir" \
  --arg original_job_id "$original_job_id" \
  --arg old_finalizer_job_id "$old_finalizer_job_id" \
  --arg new_job_id "$new_job_id" \
  --arg new_finalizer_job_id "$new_finalizer_job_id" \
  --argjson preserved_original_task_ids "$preserved_json" \
  --argjson n_cancelled_original_tasks "${#pending_task_ids[@]}" \
  --argjson n_new_array_tasks "$new_task_count" \
  '{
    schema_version: 1,
    submitted_at_utc: $submitted_at_utc,
    output_root: $output_root,
    reroute_dir: $reroute_dir,
    original_job_id: $original_job_id,
    old_finalizer_job_id: $old_finalizer_job_id,
    preserved_original_task_ids: $preserved_original_task_ids,
    n_cancelled_original_tasks: $n_cancelled_original_tasks,
    new_job_id: $new_job_id,
    n_new_array_tasks: $n_new_array_tasks,
    new_finalizer_job_id: $new_finalizer_job_id,
    partition: "bioe,owners",
    time_limit: "01:00:00"
  }' > "$submission_tmp"
mv "$submission_tmp" "$reroute_dir/submission.json"
active_submission_tmp="$output_root/.active_reroute_submission.json.tmp.$$"
cp "$reroute_dir/submission.json" "$active_submission_tmp"
mv "$active_submission_tmp" "$output_root/active_reroute_submission.json"

transition_complete=1
trap - ERR INT TERM
echo "reroute_dir=$reroute_dir"
echo "new_job_id=$new_job_id"
echo "new_finalizer_job_id=$new_finalizer_job_id"
echo "new_array_tasks=$new_task_count"
