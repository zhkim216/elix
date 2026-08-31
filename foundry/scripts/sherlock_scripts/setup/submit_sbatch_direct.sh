#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") [--sbatch-arg <arg>]... <script.sbatch>" >&2
}

SBATCH_ARGS=()
SBATCH_SCRIPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sbatch-arg)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      SBATCH_ARGS+=("$2")
      shift 2
      ;;
    --sbatch-arg=*)
      SBATCH_ARGS+=("${1#--sbatch-arg=}")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      usage
      exit 2
      ;;
    *)
      [[ -z "${SBATCH_SCRIPT}" ]] || { usage; exit 2; }
      SBATCH_SCRIPT="$1"
      shift
      ;;
  esac
done

[[ -n "${SBATCH_SCRIPT}" && -f "${SBATCH_SCRIPT}" ]] || { usage; exit 2; }
exec sbatch "${SBATCH_ARGS[@]}" "${SBATCH_SCRIPT}"
