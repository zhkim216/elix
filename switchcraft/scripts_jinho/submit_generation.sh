#!/bin/bash
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ITEMS="/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/manifests/work_items.tsv"
SBATCH_FILE="$SCRIPT_ROOT/run_generation_array.sbatch"
CHUNK_SIZE=1000
POSSU_MAX_CONCURRENT=8
DO_SUBMIT=0

usage() {
  echo "Usage: $0 [--submit] [--work-items PATH] [--chunk-size N] [--possu-max-concurrent N]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit)
      DO_SUBMIT=1
      shift
      ;;
    --work-items)
      WORK_ITEMS="$2"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --possu-max-concurrent)
      POSSU_MAX_CONCURRENT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$WORK_ITEMS" ]]; then
  echo "Missing work-items file: $WORK_ITEMS" >&2
  exit 2
fi
if ! [[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "--chunk-size must be a positive integer" >&2
  exit 2
fi
if [[ "$CHUNK_SIZE" -gt 1000 ]]; then
  echo "--chunk-size must not exceed Sherlock max_array_tasks=1000" >&2
  exit 2
fi
if ! [[ "$POSSU_MAX_CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "--possu-max-concurrent must be a positive integer" >&2
  exit 2
fi

TOTAL_WORK_ITEMS=$(( $(wc -l < "$WORK_ITEMS") - 1 ))
if [[ "$TOTAL_WORK_ITEMS" -le 0 ]]; then
  echo "No work items found in $WORK_ITEMS" >&2
  exit 2
fi

echo "work_items=$WORK_ITEMS"
echo "total_work_items=$TOTAL_WORK_ITEMS"
echo "chunk_size=$CHUNK_SIZE possu_max_concurrent=$POSSU_MAX_CONCURRENT submit=$DO_SUBMIT"

OFFSET=0
CHUNK_INDEX=0
while [[ "$OFFSET" -lt "$TOTAL_WORK_ITEMS" ]]; do
  REMAINING=$((TOTAL_WORK_ITEMS - OFFSET))
  COUNT=$CHUNK_SIZE
  if [[ "$REMAINING" -lt "$COUNT" ]]; then
    COUNT=$REMAINING
  fi
  ARRAY_START=$OFFSET
  ARRAY_END=$((OFFSET + COUNT - 1))
  if [[ "$CHUNK_INDEX" -eq 0 ]]; then
    PARTITION=owners
    ARRAY_SPEC="${ARRAY_START}-${ARRAY_END}"
  else
    PARTITION=possu
    ARRAY_SPEC="${ARRAY_START}-${ARRAY_END}%${POSSU_MAX_CONCURRENT}"
  fi
  COMMAND=(
    sbatch
    --parsable
    "--partition=$PARTITION"
    "--array=$ARRAY_SPEC"
    "--export=ALL,WORK_OFFSET=0,WORK_ITEMS=$WORK_ITEMS"
  )
  COMMAND+=("$SBATCH_FILE")
  printf 'chunk=%d global_range=%d-%d count=%d partition=%s command=' \
    "$CHUNK_INDEX" "$ARRAY_START" "$ARRAY_END" "$COUNT" "$PARTITION"
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  if [[ "$DO_SUBMIT" -eq 1 ]]; then
    SUBMIT_OUTPUT="$("${COMMAND[@]}")"
    JOB_ID="${SUBMIT_OUTPUT%%;*}"
    if ! [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
      echo "Could not parse submitted job ID from: $SUBMIT_OUTPUT" >&2
      exit 1
    fi
    echo "submitted_job_id=$JOB_ID"
  fi
  OFFSET=$((OFFSET + COUNT))
  CHUNK_INDEX=$((CHUNK_INDEX + 1))
done

echo "chunks=$CHUNK_INDEX"
if [[ "$DO_SUBMIT" -eq 0 ]]; then
  echo "Dry run only. Re-run with --submit to call sbatch."
fi
