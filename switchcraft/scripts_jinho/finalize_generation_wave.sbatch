#!/bin/bash
#SBATCH --job-name=switchcraft_wave_finalize
#SBATCH --partition=normal
#SBATCH --account=possu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%j.out
#SBATCH --error=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%j.err

set -euo pipefail

RUN_ROOT="${RUN_ROOT:?RUN_ROOT must be exported by the controller}"
BATCH_INDEX="${BATCH_INDEX:?BATCH_INDEX must be exported by the controller}"
WAVE_INDEX="${WAVE_INDEX:?WAVE_INDEX must be exported by the controller}"
ATTEMPT="${ATTEMPT:?ATTEMPT must be exported by the controller}"

exec python -B /home/users/zhkim216/code/switchcraft/scripts_jinho/two_lane_generation.py \
  finalize-wave \
  --run-root "$RUN_ROOT" \
  --batch-index "$BATCH_INDEX" \
  --wave-index "$WAVE_INDEX" \
  --attempt "$ATTEMPT"
