#!/usr/bin/env bash
#SBATCH --job-name=dnv_s1p0_d32
#SBATCH --partition=possu,bioe,owners
#SBATCH --constraint=GPU_SKU:A40|GPU_SKU:A100_PCIE|GPU_SKU:A100_SXM4|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/logs/%x_%j.out
#SBATCH --error=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/logs/%x_%j.err

set -euo pipefail

expected_venv=/scratch/users/zhkim216/envs/uv/elix-torch280
python_bin="${expected_venv}/bin/python"
rfd3_bin="${expected_venv}/bin/rfd3"
script_root=/home/users/zhkim216/code/elix/foundry/scripts/sherlock_scripts
curation_root="${script_root}/curation/denovoval_stratified"
partial_root="${script_root}/partial_diffusion"
version_root=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1
output_root="${version_root}/canary/ss1p0_direct32"
checkpoint=/scratch/users/zhkim216/model_params/rfd3/rfd3_latest.ckpt
container=/scratch/users/zhkim216/containers/elix.sif
export CCD_MIRROR_PATH=/scratch/users/zhkim216/datasets/ccd_mirror
export FOUNDRY_CHECKPOINT_DIRS=/scratch/users/zhkim216/model_params/rfd3

if [[ "${VIRTUAL_ENV:-}" != "${expected_venv}" ]]; then
  echo "Run this job through wrap_sbatch_in_container_elix.sh." >&2
  exit 2
fi
for path in \
  "${python_bin}" \
  "${rfd3_bin}" \
  "${checkpoint}" \
  "${container}" \
  "${version_root}/selection/main_ensemble_replacements.tsv" \
  "${version_root}/denovoval_re" \
  "${curation_root}/prepare_ss1p0_direct32_canary.py" \
  "${partial_root}/run_partial_diffusion.py" \
  "${partial_root}/audit_denovoval_partial_diffusion.py"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required canary input: ${path}" >&2
    exit 2
  fi
done

date --iso-8601=seconds
hostname
"${python_bin}" -B "${curation_root}/prepare_ss1p0_direct32_canary.py"
input_json=$(<"${output_root}/canary_input.txt")
"${python_bin}" -B "${partial_root}/run_partial_diffusion.py" \
  --input-json "${input_json}" \
  --output-root "${output_root}" \
  --num-samples 32 \
  --diffusion-batch-size 32 \
  --n-batches 1 \
  --checkpoint "${checkpoint}" \
  --seed 7 \
  --step-scale 1.0 \
  --gamma-0 1.0 \
  --noise-scale 1.003 \
  --rfd3-cmd "${rfd3_bin}" \
  --container "${container}" \
  --resume
"${python_bin}" -B "${partial_root}/audit_denovoval_partial_diffusion.py" \
  --output-root "${output_root}" \
  --expected-num-samples 32
date --iso-8601=seconds
