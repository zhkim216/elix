#!/usr/bin/env bash
# Source this file on Sherlock before running Foundry/RFD3 sampling scripts.
#
# Usage:
#   cd /home/users/zhkim216/code/elix/foundry
#   source scripts/sherlock_scripts/setup/activate_foundry_sherlock.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script is meant to be sourced, not executed:" >&2
  echo "  source ${BASH_SOURCE[0]}" >&2
  exit 2
fi

export FOUNDRY_ROOT="${FOUNDRY_ROOT:-/home/users/zhkim216/code/elix/foundry}"
# (JH) fixed: keep the editable RFD3 environment explicit and separate from the SIF Python.
# (JH) fixed: this checkout has no SCM tag metadata, so a plain editable install may
# report rc-foundry==0.0.0. Reinstall with SETUPTOOLS_SCM_PRETEND_VERSION=0.2.0;
# the partial-diffusion preflight intentionally rejects any other package version.
FOUNDRY_USER="${USER:-$(id -un)}"
export SCRATCH="${SCRATCH:-/scratch/users/${FOUNDRY_USER}}"
export RFD3_ENV="${RFD3_ENV:-${SCRATCH}/envs/uv/rfd3}"
export RFD3_PYTHON="${RFD3_PYTHON:-${RFD3_ENV}/bin/python}"
export RFD3_BIN="${RFD3_BIN:-${RFD3_ENV}/bin/rfd3}"
export FOUNDRY_SIF="${FOUNDRY_SIF:-/scratch/users/zhkim216/containers/foundry-0.2.0-slim-f176bafd42cf.sif}"
export CCD_MIRROR_PATH="${CCD_MIRROR_PATH:-/scratch/users/zhkim216/datasets/ccd_mirror}"
export CKPT_PATH="${CKPT_PATH:-/scratch/users/zhkim216/model_params/rfd3/rfd3_latest.ckpt}"
export USE_APPTAINER="${USE_APPTAINER:-1}"
export APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
export FOUNDRY_CHECKPOINT_DIRS="${FOUNDRY_CHECKPOINT_DIRS:-$(dirname "${CKPT_PATH}")}"

if [[ "${FOUNDRY_SHERLOCK_QUIET:-0}" != "1" ]]; then
  echo "FOUNDRY_ROOT=${FOUNDRY_ROOT}"
  # (JH) fixed: show the selected standalone env so launcher logs expose the Python owner.
  echo "RFD3_ENV=${RFD3_ENV}"
  echo "RFD3_PYTHON=${RFD3_PYTHON}"
  echo "RFD3_BIN=${RFD3_BIN}"
  echo "FOUNDRY_SIF=${FOUNDRY_SIF}"
  echo "CCD_MIRROR_PATH=${CCD_MIRROR_PATH}"
  echo "CKPT_PATH=${CKPT_PATH}"
  echo "USE_APPTAINER=${USE_APPTAINER}"
fi
