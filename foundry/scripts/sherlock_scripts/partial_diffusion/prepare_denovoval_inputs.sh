#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shell_wrapper="$script_dir/../setup/shell_in_container_foundry_rfd3.sh"
ccd_mirror=${CCD_MIRROR_PATH:-/scratch/users/zhkim216/datasets/ccd_mirror}

if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  [[ -x "$shell_wrapper" ]] || { echo "Missing wrapper: $shell_wrapper" >&2; exit 2; }
  exec "$shell_wrapper" -- "$script_dir/prepare_denovoval_inputs.sh" "$@"
fi

[[ -n "${RFD3_PYTHON:-}" && -x "$RFD3_PYTHON" ]] || {
  echo "Missing RFD3 Python: ${RFD3_PYTHON:-unset}" >&2
  exit 2
}
[[ -d "$ccd_mirror" ]] || { echo "Missing CCD mirror: $ccd_mirror" >&2; exit 2; }

export CCD_MIRROR_PATH="$ccd_mirror"
exec "$RFD3_PYTHON" "$script_dir/prepare_denovoval_inputs.py" "$@"
