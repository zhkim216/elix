#!/usr/bin/env bash
# Tiny local smoke for the denovoval step3 Foldseek clustering entrypoint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${HERE}/cluster_denovoval_step3_foldseek.py" \
  --conditions NOS_len150 NOS_len300 \
  --max-structures-per-condition 4 \
  --threads 2 \
  --output-root /scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval/step3/foldseek_clustering_smoke \
  --overwrite \
  "$@"
