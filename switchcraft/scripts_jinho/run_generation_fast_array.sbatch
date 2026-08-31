#!/bin/bash
#SBATCH --job-name=switchcraft_fast
#SBATCH --partition=bioe,possu,owners
#SBATCH --account=possu
#SBATCH --qos=normal
#SBATCH --gpus-per-node=1
#SBATCH --constraint=GPU_SKU:A40|GPU_SKU:A100_SXM4|GPU_SKU:A100_PCIE|GPU_SKU:H100_SXM5|GPU_SKU:H200_SXM5
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%A_%a.err

set -euo pipefail
exec /home/users/zhkim216/code/switchcraft/scripts_jinho/run_generation_lane_task.sh
