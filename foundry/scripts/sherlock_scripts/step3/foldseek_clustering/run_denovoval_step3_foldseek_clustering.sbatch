#!/usr/bin/env bash
#SBATCH --job-name=denovoval_step3_foldseek
#SBATCH --partition=possu,bioe,owners
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/denovoval_step3_foldseek_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/denovoval_step3_foldseek_%j.err

set -euo pipefail

FOUNDRY_ROOT="${FOUNDRY_ROOT:-/home/users/zhkim216/code/elix/foundry}"
SCRIPT="${FOUNDRY_ROOT}/scripts/sherlock_scripts/step3/foldseek_clustering/cluster_denovoval_step3_foldseek.py"

INPUT_ROOT="${INPUT_ROOT:-/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step2/generated_samples/full}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step3/foldseek_clustering}"
FOLDSEEK_BIN="${FOLDSEEK_BIN:-/oak/stanford/groups/possu/jinho/software/foldseek/bin/foldseek}"
CLUSTER_C="${CLUSTER_C:-0.8}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p /scratch/users/zhkim216/job_output

"${PYTHON_BIN}" "${SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --foldseek-bin "${FOLDSEEK_BIN}" \
  --cluster-c "${CLUSTER_C}" \
  --threads "${THREADS}" \
  "$@"
