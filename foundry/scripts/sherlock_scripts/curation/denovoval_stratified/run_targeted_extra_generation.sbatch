#!/usr/bin/env bash
#SBATCH --job-name=denovo_extra4
#SBATCH --partition=possu,bioe,owners
#SBATCH --array=0-4
#SBATCH --gres=gpu:1
#SBATCH --constraint="GPU_MEM:48GB|GPU_MEM:80GB|GPU_MEM:141GB"
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/extra_generation/logs/%x_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/extra_generation/logs/%x_%A_%a.err

set -euo pipefail

if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "This sbatch must run through the Foundry RFD3 container wrapper." >&2
  exit 2
fi

version_root="${VERSION_ROOT:-/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1}"
task_manifest="${TASK_MANIFEST:-${version_root}/extra_generation/task_manifest.tsv}"
run_tag="${RUN_TAG:-seed8_n100}"
output_root="${OUTPUT_ROOT:-${version_root}/extra_generation/generated_samples/${run_tag}}"
diffusion_batch_size="${DIFFUSION_BATCH_SIZE:-10}"
n_batches="${N_BATCHES:-10}"
noise_scale="${NOISE_SCALE:-1.003}"
seed="${SEED:-8}"

[[ -f "${task_manifest}" ]] || { echo "Missing task manifest: ${task_manifest}" >&2; exit 2; }
[[ "${diffusion_batch_size}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid DIFFUSION_BATCH_SIZE" >&2; exit 2; }
[[ "${n_batches}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid N_BATCHES" >&2; exit 2; }
[[ "${seed}" =~ ^[0-9]+$ ]] || { echo "Invalid SEED" >&2; exit 2; }

mapfile -t task_rows < <(tail -n +2 "${task_manifest}")
task_id="${SLURM_ARRAY_TASK_ID:-0}"
if (( task_id < 0 || task_id >= ${#task_rows[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${task_id} is outside 0-$((${#task_rows[@]} - 1))." >&2
  exit 2
fi

IFS=$'\t' read -r manifest_task_id condition group ccd length input_json step_scale gamma_0 <<< "${task_rows[task_id]}"
[[ "${manifest_task_id}" == "${task_id}" ]] || { echo "Task manifest order mismatch" >&2; exit 2; }
[[ -f "${input_json}" ]] || { echo "Missing input JSON: ${input_json}" >&2; exit 2; }

expected_samples=$((diffusion_batch_size * n_batches))
out_dir="${output_root}/${condition}/${group}"

if [[ "${PRINT_TASK_PLAN:-false}" == "true" ]]; then
  printf 'task=%s condition=%s group=%s input=%s step_scale=%s gamma_0=%s seed=%s expected_samples=%s out_dir=%s\n' \
    "${task_id}" "${condition}" "${group}" "${input_json}" "${step_scale}" "${gamma_0}" \
    "${seed}" "${expected_samples}" "${out_dir}"
  exit 0
fi

test -x "${RFD3_PYTHON}"
test -x "${RFD3_BIN}"
test -f "${CKPT_PATH}"
test -d "${CCD_MIRROR_PATH}"

mkdir -p "${out_dir}"
echo "targeted denovoval generation: task=${task_id} condition=${condition} group=${group}"
echo "step_scale=${step_scale} gamma_0=${gamma_0} noise_scale=${noise_scale} seed=${seed}"
echo "diffusion_batch_size=${diffusion_batch_size} n_batches=${n_batches} expected_samples=${expected_samples}"
echo "input_json=${input_json}"
echo "out_dir=${out_dir}"

"${RFD3_PYTHON}" "${FOUNDRY_ROOT}/scripts/sherlock_scripts/step2/run_rfd3_inputs.py" \
  --input-json "${input_json}" \
  --out-dir "${out_dir}" \
  --ckpt-path "${CKPT_PATH}" \
  --diffusion-batch-size "${diffusion_batch_size}" \
  --n-batches "${n_batches}" \
  --step-scale "${step_scale}" \
  --gamma-0 "${gamma_0}" \
  --noise-scale "${noise_scale}" \
  --seed "${seed}" \
  --rfd3-cmd "${RFD3_BIN}"

cif_count=$(find "${out_dir}" -maxdepth 1 -type f -name '*.cif.gz' -printf '.' | wc -c)
json_count=$(find "${out_dir}" -maxdepth 1 -type f -name '*.json' -printf '.' | wc -c)
if [[ "${cif_count}" -ne "${expected_samples}" || "${json_count}" -ne "${expected_samples}" ]]; then
  echo "Incomplete output: cif=${cif_count}/${expected_samples} json=${json_count}/${expected_samples}" >&2
  exit 3
fi

"${RFD3_PYTHON}" - "${out_dir}" "${seed}" "${expected_samples}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_seed = int(sys.argv[2])
expected_count = int(sys.argv[3])
paths = sorted(out_dir.glob("*.json"))
if len(paths) != expected_count:
    raise SystemExit(f"Expected {expected_count} JSON files, found {len(paths)}")
bad = [str(path) for path in paths if json.loads(path.read_text()).get("seed") != expected_seed]
if bad:
    raise SystemExit(f"Unexpected seed metadata in {bad[:3]}")
PY

echo "Task complete: task=${task_id} condition=${condition} group=${group} cif=${cif_count} json=${json_count}"
