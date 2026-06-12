#!/bin/bash
set -euo pipefail

# Edit the block below, then run:
#   bash gather_csvs.sh
#
# Examples:
#   ACTION="collect"
#   OUTPUT_DIR="/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/collected_csvs"
#   OUTFILE_NAME="tc_af3_nativeval_pairsweep.tar.gz"
#   EXP_DIRS=(
#     /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/tc_af3_nativeval_sm_metal_nativeseq_recycles1_pairscale0p1
#     /scratch/users/zhkim216/out_dir/eval_ligand_seq_des/tc_af3_nativeval_sm_metal_nativeseq_recycles1_pairscale0p25
#   )

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACTION="${ACTION:-collect}"  # scan, merge, gather, or collect
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/collected_csvs}"
OUTFILE_NAME="${OUTFILE_NAME:-tc_af3_nativeval_pairsweep.tar.gz}"
INCLUDE_ARRAY_SHARDS="${INCLUDE_ARRAY_SHARDS:-false}"
ALLOW_GAPS="${ALLOW_GAPS:-false}"
ALLOW_BROKEN="${ALLOW_BROKEN:-false}"
FORCE="${FORCE:-false}"

EXP_DIRS=(
)

if [ "${#EXP_DIRS[@]}" -eq 0 ]; then
    echo "Edit EXP_DIRS in gather_csvs.sh before running." >&2
    exit 2
fi

OUTPUT_TAR="${OUTPUT_DIR}/${OUTFILE_NAME}"
ARGS=("${ACTION}")

case "${ACTION}" in
    scan)
        ;;
    merge)
        ;;
    gather|collect)
        ARGS+=(--output-tar "${OUTPUT_TAR}")
        ;;
    *)
        echo "Invalid ACTION=${ACTION}; expected scan, merge, gather, or collect." >&2
        exit 2
        ;;
esac

if [ "${INCLUDE_ARRAY_SHARDS}" = "true" ] && [ "${ACTION}" != "merge" ]; then
    ARGS+=(--include-array-shards)
fi
if [ "${ALLOW_GAPS}" = "true" ] && { [ "${ACTION}" = "merge" ] || [ "${ACTION}" = "collect" ]; }; then
    ARGS+=(--allow-gaps)
fi
if [ "${ALLOW_BROKEN}" = "true" ] && { [ "${ACTION}" = "merge" ] || [ "${ACTION}" = "collect" ]; }; then
    ARGS+=(--allow-broken)
fi
if [ "${FORCE}" = "true" ] && { [ "${ACTION}" = "merge" ] || [ "${ACTION}" = "collect" ]; }; then
    ARGS+=(--force)
fi
ARGS+=("${EXP_DIRS[@]}")

echo "ACTION=${ACTION}"
echo "EXP_DIRS=${#EXP_DIRS[@]}"
echo "OUTPUT_TAR=${OUTPUT_TAR}"

python3 "$SCRIPT_DIR/eval_csv_artifacts.py" "${ARGS[@]}"
