#!/usr/bin/env bash
#SBATCH --job-name=ss1g1_step4_rasa
#SBATCH --partition=normal,owners,possu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_stratified/ss1p0_gamma1p0/curation_steps/step1/logs/%x_%j.out
#SBATCH --error=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_stratified/ss1p0_gamma1p0/curation_steps/step1/logs/%x_%j.err

set -euo pipefail

step1_root=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_stratified/ss1p0_gamma1p0/curation_steps/step1
scripts_root="${step1_root}/scripts"
foldseek_root="${step1_root}/foldseek_clustering"
output_root="${step1_root}/ligand_rasa"
ccd_metadata=/scratch/users/zhkim216/datasets/evaluation_datasets/ccd_list/denovoval_ccd_list.tsv
python_bin=/scratch/users/zhkim216/envs/uv/elix-torch280/bin/python

if [[ "${VIRTUAL_ENV:-}" != "/scratch/users/zhkim216/envs/uv/elix-torch280" ]]; then
  echo "Run this job through wrap_sbatch_in_container_elix.sh." >&2
  exit 2
fi
for path in \
  "${foldseek_root}/input_manifest.tsv" \
  "${ccd_metadata}" \
  "${scripts_root}/compute_ligand_rasa.py" \
  "${scripts_root}/validate_step4_ligand_rasa.py"; do
  test -f "${path}"
done
for output_path in \
  "${output_root}" \
  "${step1_root}/STEP4_REPORT.md" \
  "${step1_root}/step4_validation.json"; do
  if [[ -e "${output_path}" ]]; then
    echo "Output already exists; refusing to overwrite: ${output_path}" >&2
    exit 2
  fi
done

date --iso-8601=seconds
hostname
"${python_bin}" -B -c \
  'import sys, numpy, pandas, matplotlib, biotite; print(sys.executable); print(numpy.__version__, pandas.__version__, matplotlib.__version__, biotite.__version__)'

"${python_bin}" -B "${scripts_root}/compute_ligand_rasa.py" \
  --manifest "${foldseek_root}/input_manifest.tsv" \
  --ccd-metadata "${ccd_metadata}" \
  --output-root "${output_root}" \
  --probe-radius 1.4 \
  --point-number 1000 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --chunksize 32 \
  --progress-interval 1000

"${python_bin}" -B "${scripts_root}/validate_step4_ligand_rasa.py" \
  --input-manifest "${foldseek_root}/input_manifest.tsv" \
  --ccd-metadata "${ccd_metadata}" \
  --output-root "${output_root}" \
  --report-md "${step1_root}/STEP4_REPORT.md" \
  --report-json "${step1_root}/step4_validation.json" \
  --lengths 150 300 \
  --expected-ccd-count 154 \
  --expected-per-condition 100

date --iso-8601=seconds
