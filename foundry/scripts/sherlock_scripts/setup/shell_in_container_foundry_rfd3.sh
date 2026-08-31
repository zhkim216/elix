#!/usr/bin/env bash
set -euo pipefail

# (JH) fixed: centralize the one interactive/command container boundary for editable RFD3.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $(basename "$0") [-- <command> [args...]]" >&2
  echo "  No arguments       Open an interactive RFD3 shell in the Foundry SIF." >&2
  echo "  -- <command> ...   Run one command in the same container environment." >&2
}

# (JH) fixed: require an explicit '--' boundary so command arguments cannot be mistaken for wrapper flags.
COMMAND=()
case "${1:-}" in
  "")
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  --)
    shift
    if [[ $# -eq 0 ]]; then
      usage
      exit 2
    fi
    COMMAND=("$@")
    ;;
  *)
    usage
    exit 2
    ;;
esac

# shellcheck disable=SC1091
source "$SCRIPT_DIR/activate_foundry_sherlock.sh"

# (JH) fixed: fail before entering Apptainer when any required runtime artifact is absent.
[[ -f "$FOUNDRY_SIF" ]] || { echo "Foundry SIF not found: $FOUNDRY_SIF" >&2; exit 2; }
[[ -d "$FOUNDRY_ROOT" ]] || { echo "Foundry repo not found: $FOUNDRY_ROOT" >&2; exit 2; }
[[ -x "$RFD3_PYTHON" && -x "$RFD3_BIN" && -f "$RFD3_ENV/bin/activate" ]] || {
  echo "Standalone RFD3 environment not found or incomplete: $RFD3_ENV" >&2
  echo "Expected executable RFD3_PYTHON=$RFD3_PYTHON and RFD3_BIN=$RFD3_BIN." >&2
  exit 2
}
APPTAINER_REQUESTED="$APPTAINER_BIN"
if ! APPTAINER_BIN="$(command -v "$APPTAINER_REQUESTED")"; then
  echo "Apptainer executable not found: $APPTAINER_REQUESTED" >&2
  exit 2
fi

# (JH) fixed: bind the standalone env and repo explicitly even though both may sit below broader binds.
binds=()
add_bind() {
  if [[ -n "${1:-}" ]]; then
    binds+=("$1")
  fi
}
add_bind "$SCRATCH"
add_bind "$FOUNDRY_ROOT"
add_bind "$RFD3_ENV"
add_bind "$SCRIPT_DIR"
BIND_LIST="$(IFS=,; echo "${binds[*]}")"

# (JH) fixed: select only the editable env in-container and disable nested self-containerization.
CONTAINER_PATH="$RFD3_ENV/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
container_env=(
  --env "PATH=$CONTAINER_PATH"
  --env "VIRTUAL_ENV=$RFD3_ENV"
  --env "PYTHONPATH="
  --env "PYTHONNOUSERSITE=1"
  --env "SCRATCH=$SCRATCH"
  --env "FOUNDRY_ROOT=$FOUNDRY_ROOT"
  --env "FOUNDRY_SIF=$FOUNDRY_SIF"
  --env "RFD3_ENV=$RFD3_ENV"
  --env "RFD3_PYTHON=$RFD3_PYTHON"
  --env "RFD3_BIN=$RFD3_BIN"
  --env "CCD_MIRROR_PATH=$CCD_MIRROR_PATH"
  --env "CKPT_PATH=$CKPT_PATH"
  --env "FOUNDRY_CHECKPOINT_DIRS=$FOUNDRY_CHECKPOINT_DIRS"
  --env "FOUNDRY_CONTAINER_ACTIVE=1"
  --env "USE_APPTAINER=0"
)

echo "[foundry-rfd3-container] image: $FOUNDRY_SIF"
echo "[foundry-rfd3-container] env: $RFD3_ENV"
echo "[foundry-rfd3-container] repo: $FOUNDRY_ROOT"
if [[ ${#COMMAND[@]} -gt 0 ]]; then
  printf '[foundry-rfd3-container] command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
else
  echo "[foundry-rfd3-container] command: interactive shell"
fi

# (JH) fixed: do not use --cleanenv; inherited scheduler variables must remain available in command mode.
exec "$APPTAINER_BIN" exec --nv \
  --bind "$BIND_LIST" \
  "${container_env[@]}" \
  "$FOUNDRY_SIF" \
  bash -lc '
set -euo pipefail
unset PYTHONPATH
# shellcheck disable=SC1090
source "$RFD3_ENV/bin/activate"
cd "$FOUNDRY_ROOT"
if [[ $# -gt 0 ]]; then
  exec "$@"
fi
exec bash --noprofile --norc -i
' foundry-rfd3-shell "${COMMAND[@]}"
