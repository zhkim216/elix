#!/bin/bash
set -euo pipefail

# Edit the experiment block below, then run: bash gather_csvs.sh
#
# Gathers seq-design metric CSV outputs. Set RECURSIVE=true for two-stage
# outputs, where step_* directories are nested under stage1_* and stage2_*.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
COLLECTED_CSV_DIR=${BASE_OUT_DIR}/collected_csvs
VALSET_NAME=nativeval_sm_metal
EXP_NAME_SUFFIX=_tc_numseqs1

MODEL_NAMES=(
    elix_exp2_cfg0
    elix_exp2_cfg1
    elix_exp2_cfg2
    elix_exp2_cfg3
    elix_exp2_cfg4
)

EXP_DIRS=(
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    EXP_DIRS+=("${BASE_OUT_DIR}/${MODEL_NAME}_${VALSET_NAME}${EXP_NAME_SUFFIX}")
done

OUTPUT_TAR=${COLLECTED_CSV_DIR}/elix_exp2_nativeval_tc_csvs.tar.gz

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
