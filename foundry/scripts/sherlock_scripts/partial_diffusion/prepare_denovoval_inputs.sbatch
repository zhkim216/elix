#!/usr/bin/env bash
#SBATCH --job-name=rfd3_denovo_prep
#SBATCH --partition=bioe,owners
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/rfd3_denovo_prep_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/rfd3_denovo_prep_%j.err

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "Submit through wrap_sbatch_in_container_foundry_rfd3.sh" >&2
  exit 2
fi

output_root=${OUTPUT_ROOT:-/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/ensembles}
command=(
  "$RFD3_PYTHON" "$script_dir/prepare_denovoval_inputs.py"
  --source-dir /scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/cifs
  --output-root "$output_root"
  --partial-t 2 5
  --smoke-sample-id A1ISN_len300_9
  --expected-cif-count 3400
  --expected-ccd-count 154
)
if [[ "${PREPARE_OVERWRITE:-0}" == "1" ]]; then
  command+=(--overwrite)
fi
printf 'command:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
