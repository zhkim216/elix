#!/bin/bash
#SBATCH --job-name=switchcraft_transition
#SBATCH --partition=normal
#SBATCH --account=possu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%j.out
#SBATCH --error=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%j.err

set -euo pipefail

SCRIPT_ROOT=/home/users/zhkim216/code/switchcraft/scripts_jinho
CONTROLLER="$SCRIPT_ROOT/two_lane_generation.py"

python -B "$CONTROLLER" prepare
python -B "$CONTROLLER" submit --batch-index 0 --attempt 1 --dry-run
exec python -B "$CONTROLLER" submit --batch-index 0 --attempt 1
