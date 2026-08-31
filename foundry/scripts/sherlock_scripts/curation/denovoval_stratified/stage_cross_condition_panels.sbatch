#!/usr/bin/env bash
#SBATCH --job-name=dnv_joint_stage_v1
#SBATCH --partition=normal,owners,possu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/logs/%x_%j.out
#SBATCH --error=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/logs/%x_%j.err

set -euo pipefail

expected_venv=/scratch/users/zhkim216/envs/uv/elix-torch280
python_bin="${expected_venv}/bin/python"
script=/home/users/zhkim216/code/elix/foundry/scripts/sherlock_scripts/curation/denovoval_stratified/curate_cross_condition_panels.py
output_root=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1
ss1p0_source_root="${output_root}/work/ss1p0_source"
archive=/scratch/users/zhkim216/datasets/denovoval_stratified.tar.gz
foldseek_bin=/oak/stanford/groups/possu/jinho/software/foldseek/bin/foldseek

if [[ "${VIRTUAL_ENV:-}" != "${expected_venv}" ]]; then
  echo "Run this job through wrap_sbatch_in_container_elix.sh." >&2
  exit 2
fi
for path in \
  "${python_bin}" \
  "${script}" \
  "${archive}" \
  "${foldseek_bin}" \
  "${output_root}/work"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}/denovoval_re" || -e "${output_root}/denovoval_subset" ]]; then
  echo "Materialized output already exists under ${output_root}." >&2
  exit 2
fi

date --iso-8601=seconds
hostname
raw_count=0
if [[ -d "${ss1p0_source_root}/raw" ]]; then
  raw_count=$(find "${ss1p0_source_root}/raw" -type f -name '*.cif.gz' -printf '.' | wc -c)
fi
if [[ "${raw_count}" -ne 30800 ]] || \
   [[ ! -s "${ss1p0_source_root}/curation_steps/step1/selection/candidate_audit.tsv" ]]; then
  echo "Staging ss1p0 raw archive entries; existing_raw_count=${raw_count}"
  mkdir -p "${ss1p0_source_root}"
  tar -xzf "${archive}" \
    --touch \
    -C "${ss1p0_source_root}" \
    --strip-components=2 \
    denovoval_stratified/ss1p0_gamma1p0/raw \
    denovoval_stratified/ss1p0_gamma1p0/curation_steps/step1/ligand_rasa/ligand_rasa.tsv \
    denovoval_stratified/ss1p0_gamma1p0/curation_steps/step1/selection/candidate_audit.tsv \
    denovoval_stratified/ss1p0_gamma1p0/curation_steps/step1/selection/selected_samples.tsv
fi
raw_count=$(find "${ss1p0_source_root}/raw" -type f -name '*.cif.gz' -printf '.' | wc -c)
if [[ "${raw_count}" -ne 30800 ]]; then
  echo "Expected 30800 staged ss1p0 raw CIFs, found ${raw_count}." >&2
  exit 2
fi
for path in \
  "${ss1p0_source_root}/curation_steps/step1/ligand_rasa/ligand_rasa.tsv" \
  "${ss1p0_source_root}/curation_steps/step1/selection/candidate_audit.tsv" \
  "${ss1p0_source_root}/curation_steps/step1/selection/selected_samples.tsv"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing staged ss1p0 metadata: ${path}" >&2
    exit 2
  fi
done

smoke_root="${output_root}/smoke/12C_len150"
"${python_bin}" -B "${script}" \
  --output-root "${smoke_root}" \
  --ss1p0-raw-root "${ss1p0_source_root}/raw" \
  --ss1p0-audit "${ss1p0_source_root}/curation_steps/step1/selection/candidate_audit.tsv" \
  --foldseek-bin "${foldseek_bin}" \
  --workers 1 \
  --threads-per-foldseek 4 \
  --groups 12C_len150
date --iso-8601=seconds
