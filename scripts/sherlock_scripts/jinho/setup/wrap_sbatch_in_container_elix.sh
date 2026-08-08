#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_setup_elix.sh"

usage() {
  echo "Usage: $(basename "$0") [--schrodinger] [--dry-run] [--sbatch-arg <arg>] <sbatch_script.sbatch>" >&2
  echo "  --schrodinger  Bind Schrodinger support paths and license env." >&2
  echo "  --dry-run      Write the generated wrapper but do not submit it." >&2
  echo "  --sbatch-arg   Pass one argument through to the final sbatch call; repeat as needed." >&2
}

ENABLE_SCHRODINGER="${ELIX_ENABLE_SCHRODINGER:-0}"
DRY_RUN="${ELIX_WRAP_DRY_RUN:-0}"
SBATCH_ARGS=()
JOB=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --schrodinger)
      ENABLE_SCHRODINGER=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --sbatch-arg)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
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
      exit 1
      ;;
    *)
      if [ -n "$JOB" ]; then
        usage
        exit 1
      fi
      JOB="$1"
      shift
      ;;
  esac
done

if [ -z "$JOB" ]; then
  usage
  exit 1
fi

if [ "$ENABLE_SCHRODINGER" = "1" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/schrodinger_env.sh"
fi

ELIX_USER="${USER:-$(id -un)}"
ELIX_HOME="${HOME:-/home/users/$ELIX_USER}"
ELIX_SCRATCH="${SCRATCH:-/scratch/users/$ELIX_USER}"
IMG="${SIF:-$ELIX_SCRATCH/containers/elix.sif}"
REPO_DIR="${PROJECT_ROOT:-$ELIX_HOME/code/elix}"
ENV_DIR="${VENV:-$ELIX_SCRATCH/envs/uv/elix-torch280}"

if [ -x /bin/singularity ]; then
  APPTAINER_BIN="${APPTAINER_BIN:-/bin/singularity}"
else
  APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
fi

[[ -f "$JOB" ]] || { echo "Not found: $JOB" >&2; exit 1; }

JOB_DIR="$(cd "$(dirname "$JOB")" && pwd)"
JOB_BASE="$(basename "$JOB")"
JOB_ABS="$JOB_DIR/$JOB_BASE"

DEFAULT_REQUEUE_ON_USR2=0
if grep -Eq '^[[:space:]]*#SBATCH[[:space:]]+--export=([^,]*,)*ELIX_REQUEUE_ON_USR2=1(,|[[:space:]]|$)' "$JOB_ABS"; then
  DEFAULT_REQUEUE_ON_USR2=1
fi

WRAP_DIR="${SCRATCH:-/tmp}/slurm_elix_container_wrappers"
mkdir -p "$WRAP_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
WRAP="$WRAP_DIR/${JOB_BASE%.sbatch}.elix.container.${STAMP}.sbatch"
ORIG_COPY="$WRAP_DIR/${JOB_BASE%.sbatch}.original.${STAMP}.sbatch"
cp "$JOB_ABS" "$ORIG_COPY"

binds=()
add_bind() {
  if [ -n "${1:-}" ]; then
    binds+=("$1")
  fi
}

add_bind "$SCRATCH"
add_bind "${GROUP_HOME:-}"
add_bind "$REPO_DIR"
add_bind "$ENV_DIR"
add_bind "$JOB_DIR"
add_bind "$WRAP_DIR"
add_bind "$TORCH_HOME"
add_bind "$HF_HOME"
add_bind "$PIP_CACHE_DIR"
add_bind "$XDG_CACHE_HOME"
add_bind "$TORCHINDUCTOR_CACHE_DIR"
add_bind "$TRITON_CACHE_DIR"
add_bind "$TORCH_EXTENSIONS_DIR"
add_bind "$UV_CACHE_DIR"
add_bind "$UV_PYTHON_INSTALL_DIR"
add_bind "$JAX_COMPILATION_CACHE_DIR"
if [ -d "$CUDA_HOST" ]; then
  add_bind "$CUDA_HOST:$CUDA_HOME:ro"
fi
if [ "$ENABLE_SCHRODINGER" = "1" ]; then
  add_bind "${OAK_LIBS:-}"
  if [ -n "${MACHINE_ID_FILE:-}" ] && [ -f "$MACHINE_ID_FILE" ]; then
    add_bind "$MACHINE_ID_FILE:/etc/machine-id"
  fi
fi

BIND_LIST="$(IFS=,; echo "${binds[*]}")"
CONTAINER_PATH="$ENV_DIR/bin:/hmmer/bin:$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

{
  echo '#!/usr/bin/env bash'
  grep -E '^[[:space:]]*#SBATCH' "$JOB_ABS" || true
  cat <<EOF
set -euo pipefail
echo "[elix-container] image: $IMG"
echo "[elix-container] binds: $BIND_LIST"
echo "[elix-container] original script saved to: $ORIG_COPY"
echo "[elix-container] schrodinger: $ENABLE_SCHRODINGER"

source "$SCRIPT_DIR/env_setup_elix.sh"

export ELIX_REQUEUE_ON_USR2="\${ELIX_REQUEUE_ON_USR2:-$DEFAULT_REQUEUE_ON_USR2}"
ELIX_REQUEUE_SIGNAL_RECEIVED=0
elix_requeue_on_usr2() {
  if [ "\${ELIX_REQUEUE_ON_USR2:-0}" != "1" ]; then
    return 0
  fi

  ELIX_REQUEUE_SIGNAL_RECEIVED=1
  if [ -n "\${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "\${SLURM_ARRAY_TASK_ID:-}" ]; then
    requeue_target="\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}"
  elif [ -n "\${SLURM_JOB_ID:-}" ]; then
    requeue_target="\${SLURM_JOB_ID}"
  else
    echo "[elix-container] Cannot requeue: SLURM job ID is unavailable." >&2
    return 0
  fi

  echo "[elix-container] Requeueing \${requeue_target} after USR2; the next run will use the latest regular checkpoint."
  if ! "\${ELIX_SCONTROL_BIN:-/usr/bin/scontrol}" requeue "\${requeue_target}"; then
    echo "[elix-container] Failed to requeue \${requeue_target}; the container process will keep running." >&2
  fi
}
trap elix_requeue_on_usr2 USR2

container_env=(
  --env "PATH=$CONTAINER_PATH"
  --env "PYTHONPATH=$REPO_DIR:\${PYTHONPATH:-}"
  --env "CUDA_HOME=$CUDA_HOME"
  --env "LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}"
  --env "TRITON_LIBCUDA_PATH=$TRITON_LIBCUDA_PATH"
  --env "LIBRARY_PATH=$TRITON_LIBCUDA_PATH:\${LIBRARY_PATH:-}"
  --env "TMALIGN_BINARY=$TMALIGN_BINARY"
  --env "TORCH_HOME=$TORCH_HOME"
  --env "HF_HOME=$HF_HOME"
  --env "PIP_CACHE_DIR=$PIP_CACHE_DIR"
  --env "XDG_CACHE_HOME=$XDG_CACHE_HOME"
  --env "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR"
  --env "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
  --env "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
  --env "UV_ENV_ROOT=$UV_ENV_ROOT"
  --env "UV_CACHE_DIR=$UV_CACHE_DIR"
  --env "UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR"
  --env "XLA_FLAGS=$XLA_FLAGS"
  --env "XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
  --env "XLA_CLIENT_MEM_FRACTION=$XLA_CLIENT_MEM_FRACTION"
  --env "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"
  --env "VENV=$ENV_DIR"
  --env "SCRATCH=$SCRATCH"
  --env "PROJECT_ROOT=$REPO_DIR"
)
EOF
  if [ "$ENABLE_SCHRODINGER" = "1" ]; then
    cat <<EOF
container_env+=(
  --env "SCHRODINGER_LD_LIBS=${SCHRODINGER_LD_LIBS:-}"
  --env "SCHRODINGER=${SCHRODINGER:-}"
  --env "SCHROD_LICENSE_FILE=${SCHROD_LICENSE_FILE:-}"
)
EOF
  fi
  cat <<EOF

$APPTAINER_BIN exec --nv \\
  --bind "$BIND_LIST" \\
  "\${container_env[@]}" \\
  "$IMG" \\
  bash -lc "set -euo pipefail; source '$ENV_DIR/bin/activate'; cd '$REPO_DIR'; exec bash '$JOB_ABS'" &
ELIX_CONTAINER_PID=\$!

while true; do
  ELIX_REQUEUE_SIGNAL_RECEIVED=0
  if wait "\${ELIX_CONTAINER_PID}"; then
    exit 0
  else
    container_status=\$?
  fi
  if [ "\${ELIX_REQUEUE_SIGNAL_RECEIVED}" = "1" ] && [ "\${container_status}" -gt 128 ]; then
    continue
  fi
  exit "\${container_status}"
done
EOF
} > "$WRAP"

chmod +x "$WRAP"
echo "[wrapper] Original script saved to: $ORIG_COPY"
echo "[wrapper] Generated: $WRAP"
if [ "$DRY_RUN" = "1" ]; then
  if [ "${#SBATCH_ARGS[@]}" -gt 0 ]; then
    printf '[wrapper] sbatch args:'
    printf ' %q' "${SBATCH_ARGS[@]}"
    printf '\n'
  fi
  echo "[wrapper] Dry run: not submitting."
  exit 0
fi
echo "[wrapper] Submitting: $WRAP"
if [ "${#SBATCH_ARGS[@]}" -gt 0 ]; then
  sbatch "${SBATCH_ARGS[@]}" "$WRAP"
else
  sbatch "$WRAP"
fi
