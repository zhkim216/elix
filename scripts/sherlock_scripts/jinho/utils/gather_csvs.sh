#!/bin/bash
set -euo pipefail

# Edit EXP_DIRS and OUTPUT_TAR below, then run: bash gather_csvs.sh
#
# Gathers seq-design metric CSV outputs. Set RECURSIVE=true for two-stage
# outputs, where step_* directories are nested under stage1_* and stage2_*.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXP_DIRS=(
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp41_cfg0_denovoval_af3
    /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/eval_exp41_cfg1_denovoval_af3
)

OUTPUT_TAR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/collected_csvs/exp41_denovoval_af3_csvs.tar.gz

# Add --array-jobs flag to also collect *_array_N.csv files.
ARRAY_JOBS=false

# Keep true for two-stage outputs; harmless for flat/one-stage outputs.
RECURSIVE=true

ARGS=()
if [ "$ARRAY_JOBS" = true ]; then
    ARGS+=(--array-jobs)
fi
if [ "$RECURSIVE" = true ]; then
    ARGS+=(--recursive)
fi

python3 "$SCRIPT_DIR/gather_csvs.py" "${ARGS[@]}" "$OUTPUT_TAR" "${EXP_DIRS[@]}"
