#!/usr/bin/env bash
set -euo pipefail

# (JH) fixed: make this wrapper the sole Apptainer owner for RFD3 sbatch jobs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  echo "Usage: $(basename "$0") [--dry-run] [--sbatch-arg <arg>]... <script.sbatch>" >&2
  echo "  --dry-run      Generate the wrapper and saved original copy without submitting." >&2
  echo "  --sbatch-arg   Pass one argument to sbatch; repeat as needed." >&2
}

# (JH) fixed: keep wrapper options separate from repeatable arguments passed verbatim to sbatch.
DRY_RUN=0
SBATCH_ARGS=()
JOB=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --sbatch-arg)
      if [[ $# -lt 2 ]]; then
        usage
        exit 2
      fi
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
      if [[ -n "$JOB" ]]; then
        usage
        exit 2
      fi
      JOB="$1"
      shift
      ;;
  esac
done

if [[ -z "$JOB" ]]; then
  usage
  exit 2
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/activate_foundry_sherlock.sh"

# (JH) fixed: fail before writing or submitting a wrapper when the job runtime cannot exist.
[[ -f "$JOB" ]] || { echo "Sbatch script not found: $JOB" >&2; exit 2; }
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

# (JH) fixed: preserve the original job and keep generated wrappers in a scratch-only directory.
JOB_DIR="$(cd "$(dirname "$JOB")" && pwd)"
JOB_BASE="$(basename "$JOB")"
JOB_ABS="$JOB_DIR/$JOB_BASE"
DEFAULT_REQUEUE_ON_USR2=0
if grep -Eq '^[[:space:]]*#SBATCH[[:space:]]+--export=([^,]*,)*FOUNDRY_REQUEUE_ON_USR2=1(,|[[:space:]]|$)' "$JOB_ABS"; then
  DEFAULT_REQUEUE_ON_USR2=1
fi
WRAP_DIR="$SCRATCH/slurm_foundry_rfd3_container_wrappers"
mkdir -p "$WRAP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
WRAP="$WRAP_DIR/${JOB_BASE%.sbatch}.foundry-rfd3.container.${STAMP}.sbatch"
ORIG_COPY="$WRAP_DIR/${JOB_BASE%.sbatch}.original.${STAMP}.sbatch"
cp "$JOB_ABS" "$ORIG_COPY"

# (JH) fixed: bind every runtime boundary needed by the original job and generated wrapper.
binds=()
add_bind() {
  if [[ -n "${1:-}" ]]; then
    binds+=("$1")
  fi
}
add_bind "$SCRATCH"
add_bind "$FOUNDRY_ROOT"
add_bind "$RFD3_ENV"
add_bind "$JOB_DIR"
add_bind "$WRAP_DIR"
BIND_LIST="$(IFS=,; echo "${binds[*]}")"
CONTAINER_PATH="$RFD3_ENV/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# (JH) fixed: copy #SBATCH directives ahead of executable code while running the saved job only inside one container.
{
  echo '#!/usr/bin/env bash'
  grep -E '^[[:space:]]*#SBATCH' "$JOB_ABS" || true
  printf 'APPTAINER_BIN=%q\n' "$APPTAINER_BIN"
  printf 'FOUNDRY_SIF=%q\n' "$FOUNDRY_SIF"
  printf 'FOUNDRY_ROOT=%q\n' "$FOUNDRY_ROOT"
  printf 'RFD3_ENV=%q\n' "$RFD3_ENV"
  printf 'RFD3_PYTHON=%q\n' "$RFD3_PYTHON"
  printf 'RFD3_BIN=%q\n' "$RFD3_BIN"
  printf 'SCRATCH=%q\n' "$SCRATCH"
  printf 'CCD_MIRROR_PATH=%q\n' "$CCD_MIRROR_PATH"
  printf 'CKPT_PATH=%q\n' "$CKPT_PATH"
  printf 'FOUNDRY_CHECKPOINT_DIRS=%q\n' "$FOUNDRY_CHECKPOINT_DIRS"
  printf 'JOB_ABS=%q\n' "$JOB_ABS"
  printf 'ORIG_COPY=%q\n' "$ORIG_COPY"
  printf 'BIND_LIST=%q\n' "$BIND_LIST"
  printf 'CONTAINER_PATH=%q\n' "$CONTAINER_PATH"
  printf 'DEFAULT_REQUEUE_ON_USR2=%q\n' "$DEFAULT_REQUEUE_ON_USR2"
  cat <<'EOF'
set -euo pipefail

echo "[foundry-rfd3-container] image: $FOUNDRY_SIF"
echo "[foundry-rfd3-container] env: $RFD3_ENV"
echo "[foundry-rfd3-container] original script: $ORIG_COPY"

export FOUNDRY_REQUEUE_ON_USR2="${FOUNDRY_REQUEUE_ON_USR2:-$DEFAULT_REQUEUE_ON_USR2}"
FOUNDRY_REQUEUE_SIGNAL_RECEIVED=0
foundry_requeue_on_usr2() {
  if [[ "${FOUNDRY_REQUEUE_ON_USR2:-0}" != "1" ]]; then
    return 0
  fi
  FOUNDRY_REQUEUE_SIGNAL_RECEIVED=1
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    requeue_target="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    requeue_target="${SLURM_JOB_ID}"
  else
    echo "[foundry-rfd3-container] Cannot requeue: Slurm job ID is unavailable." >&2
    return 0
  fi
  echo "[foundry-rfd3-container] Requeueing ${requeue_target} after USR2."
  if ! "${FOUNDRY_SCONTROL_BIN:-/usr/bin/scontrol}" requeue "${requeue_target}"; then
    echo "[foundry-rfd3-container] Failed to requeue ${requeue_target}; container continues." >&2
    FOUNDRY_REQUEUE_SIGNAL_RECEIVED=0
  fi
}
trap foundry_requeue_on_usr2 USR2

# (JH) fixed: force editable-env ownership and tell migrated jobs they are already containerized.
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

# (JH) fixed: omit --cleanenv so all SLURM_* task, array, and allocation variables reach the original job.
"$APPTAINER_BIN" exec --nv \
  --bind "$BIND_LIST" \
  "${container_env[@]}" \
  "$FOUNDRY_SIF" \
  bash -lc '
set -euo pipefail
unset PYTHONPATH
# shellcheck disable=SC1090
source "$RFD3_ENV/bin/activate"
cd "$FOUNDRY_ROOT"
exec bash "$1"
' foundry-rfd3-job "$JOB_ABS" &
FOUNDRY_CONTAINER_PID=$!

while true; do
  FOUNDRY_REQUEUE_SIGNAL_RECEIVED=0
  if wait "${FOUNDRY_CONTAINER_PID}"; then
    exit 0
  else
    container_status=$?
  fi
  if [[ "${FOUNDRY_REQUEUE_SIGNAL_RECEIVED}" == "1" && "${container_status}" -gt 128 ]]; then
    continue
  fi
  exit "${container_status}"
done
EOF
} > "$WRAP"

chmod +x "$WRAP"
echo "[wrapper] Original script saved to: $ORIG_COPY"
echo "[wrapper] Generated: $WRAP"
if [[ ${#SBATCH_ARGS[@]} -gt 0 ]]; then
  printf '[wrapper] sbatch args:'
  printf ' %q' "${SBATCH_ARGS[@]}"
  printf '\n'
fi

# (JH) fixed: dry-run stops after materialization, guaranteeing no scheduler mutation.
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[wrapper] Dry run: not submitting."
  exit 0
fi

echo "[wrapper] Submitting: $WRAP"
if [[ ${#SBATCH_ARGS[@]} -gt 0 ]]; then
  exec sbatch "${SBATCH_ARGS[@]}" "$WRAP"
fi
exec sbatch "$WRAP"
