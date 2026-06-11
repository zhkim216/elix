#!/bin/bash
set -euo pipefail

# Merge array CSV files for all seq_des_multi experiments
# Usage: bash merge_all_experiments.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
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

echo "=============================================="
echo "Merging CSV files for ${#EXP_DIRS[@]} experiments"
echo "=============================================="

for EXP_DIR in "${EXP_DIRS[@]}"; do
    if [ -d "${EXP_DIR}" ]; then
        echo ""
        echo "Processing: $(basename "${EXP_DIR}")"
        mapfile -t ARRAY_CSV_DIRS < <(
            find "${EXP_DIR}" -type f -name '*_array_*.csv' -printf '%h\n' | sort -u
        )

        if [ "${#ARRAY_CSV_DIRS[@]}" -eq 0 ]; then
            echo "  SKIP: no *_array_*.csv files found under ${EXP_DIR}"
            continue
        fi

        echo "  Found ${#ARRAY_CSV_DIRS[@]} directories with array CSV shards"
        for CSV_DIR in "${ARRAY_CSV_DIRS[@]}"; do
            echo "  Merging: ${CSV_DIR#${EXP_DIR}/}"
            python3 "${SCRIPT_DIR}/merge_array_csvs.py" \
                --results_dir "${CSV_DIR}"
        done
    else
        echo "SKIP (not found): $(basename "${EXP_DIR}")"
    fi
done

echo ""
echo "=============================================="
echo "All done!"
echo "=============================================="
