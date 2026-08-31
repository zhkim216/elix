#!/usr/bin/env bash
#SBATCH --job-name=rfd3_denovo_audit
#SBATCH --partition=bioe,owners
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/zhkim216/job_output/rfd3_denovo_audit_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/rfd3_denovo_audit_%j.err

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_root=${OUTPUT_ROOT:-/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval/ensembles}

exec /usr/bin/python3 "$script_dir/audit_denovoval_partial_diffusion.py" \
  --output-root "$output_root" \
  --expected-num-samples 32
