#!/usr/bin/env bash
#SBATCH --job-name=ss_nativeval_pd_t2_n32
#SBATCH --partition=possu,bioe,owners
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR2@300
#SBATCH --export=ALL,FOUNDRY_REQUEUE_ON_USR2=1
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/zhkim216/job_output/ss_nativeval_pd_t2_n32_%j.out
#SBATCH --error=/scratch/users/zhkim216/job_output/ss_nativeval_pd_t2_n32_%j.err

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  echo "Submit through the Foundry/RFD3 sbatch wrapper:" >&2
  echo "  bash $script_dir/../../../setup/wrap_sbatch_in_container_foundry_rfd3.sh ${BASH_SOURCE[0]}" >&2
  exit 2
fi

exec "$script_dir/run_partial_diffusion.sh"
