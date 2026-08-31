#!/usr/bin/env bash
#SBATCH --job-name=dnv_rasa_hetero
#SBATCH --partition=normal,owners,possu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/runs/seed8_n100_hetero_rasa/logs/%x_%j.out
#SBATCH --error=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/runs/seed8_n100_hetero_rasa/logs/%x_%j.err

set -euo pipefail

expected_venv=/scratch/users/zhkim216/envs/uv/elix-torch280
python_bin="${expected_venv}/bin/python"
scripts_root=/home/users/zhkim216/code/elix/foundry/scripts/sherlock_scripts/curation/denovoval_stratified
run_root=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/runs/seed8_n100_hetero_rasa
input_manifest="${run_root}/rasa/input_manifest.tsv"
results_root="${run_root}/rasa/results"
ccd_metadata=/scratch/users/zhkim216/datasets/evaluation_datasets/ccd_list/denovoval_ccd_list.tsv

if [[ "${VIRTUAL_ENV:-}" != "${expected_venv}" ]]; then
  echo "Run this job through wrap_sbatch_in_container_elix.sh." >&2
  exit 2
fi
for path in \
  "${python_bin}" \
  "${scripts_root}/prepare_augmented_rasa_manifest.py" \
  "${scripts_root}/compute_ligand_rasa.py" \
  "${scripts_root}/validate_augmented_rasa.py" \
  "${ccd_metadata}"; do
  [[ -f "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 2; }
done
for path in \
  "${input_manifest}" \
  "${run_root}/rasa/input_manifest.json" \
  "${results_root}" \
  "${run_root}/rasa/validation.json"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite RASA artifact: ${path}" >&2
    exit 2
  fi
done

date --iso-8601=seconds
hostname
"${python_bin}" -B "${scripts_root}/prepare_augmented_rasa_manifest.py" \
  --run-root "${run_root}"
"${python_bin}" -B "${scripts_root}/compute_ligand_rasa.py" \
  --manifest "${input_manifest}" \
  --ccd-metadata "${ccd_metadata}" \
  --output-root "${results_root}" \
  --probe-radius 1.4 \
  --point-number 1000 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunksize 32 \
  --progress-interval 100 \
  --skip-plots
"${python_bin}" -B "${scripts_root}/validate_augmented_rasa.py" \
  --run-root "${run_root}"
date --iso-8601=seconds
