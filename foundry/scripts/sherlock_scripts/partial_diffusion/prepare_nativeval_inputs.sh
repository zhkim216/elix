#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
shell_wrapper="$script_dir/../setup/shell_in_container_foundry_rfd3.sh"
ccd_mirror=${CCD_MIRROR_PATH:-/scratch/users/zhkim216/datasets/ccd_mirror}

# (JH) fixed: Host calls enter the shared Foundry/RFD3 container exactly once.
if [[ "${FOUNDRY_CONTAINER_ACTIVE:-0}" != "1" ]]; then
  if [[ ! -x "$shell_wrapper" ]]; then
    echo "Missing Foundry/RFD3 shell wrapper: $shell_wrapper" >&2
    exit 2
  fi
  exec "$shell_wrapper" -- "$script_dir/prepare_nativeval_inputs.sh" "$@"
fi

# (JH) fixed: Prepared inputs use the editable RFD3 environment, not SIF Python.
if [[ -z "${RFD3_ENV:-}" || ! -d "$RFD3_ENV" ]]; then
  echo "Missing RFD3 environment: ${RFD3_ENV:-unset}" >&2
  exit 2
fi
if [[ -z "${RFD3_PYTHON:-}" || ! -x "$RFD3_PYTHON" ]]; then
  echo "Missing RFD3 Python: ${RFD3_PYTHON:-unset}" >&2
  exit 2
fi
if [[ ! -d "$ccd_mirror" ]]; then
  echo "Missing CCD mirror: $ccd_mirror" >&2
  exit 2
fi

# (JH) fixed: Call the repo helper by its real bound path inside the outer container.
export CCD_MIRROR_PATH="$ccd_mirror"
exec "$RFD3_PYTHON" "$script_dir/prepare_nativeval_inputs.py" "$@"
