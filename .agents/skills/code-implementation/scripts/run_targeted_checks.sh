#!/usr/bin/env bash
set -euo pipefail

WITH_TESTS=0
REPO_ROOT="$(pwd)"
declare -a CHANGED_FILES=()

usage() {
  cat <<'EOF'
Usage:
  run_targeted_checks.sh [--with-tests] [--repo-root PATH] <changed-file>...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-tests)
      WITH_TESTS=1
      shift
      ;;
    --repo-root)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --repo-root requires a value"
        usage
        exit 2
      fi
      REPO_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      CHANGED_FILES+=("$1")
      shift
      ;;
  esac
done

if [[ ${#CHANGED_FILES[@]} -eq 0 ]]; then
  echo "ERROR: at least one changed file path is required"
  usage
  exit 2
fi

cd "$REPO_ROOT"
PYCACHE_DIR="$(mktemp -d /tmp/code-implementation-pycache-XXXXXX)"
export PYTHONPYCACHEPREFIX="$PYCACHE_DIR"
trap 'rm -rf "$PYCACHE_DIR"' EXIT

declare -a PY_FILES=()
declare -a YAML_FILES=()
declare -a SH_FILES=()
declare -a RESOLVED_FILES=()

for path in "${CHANGED_FILES[@]}"; do
  rel="$path"
  if [[ "$path" = /* && "$path" == "$REPO_ROOT/"* ]]; then
    rel="${path#"$REPO_ROOT"/}"
  fi
  if [[ ! -e "$rel" ]]; then
    echo "WARN: skipping missing file: $rel"
    continue
  fi
  RESOLVED_FILES+=("$rel")
  case "$rel" in
    *.py) PY_FILES+=("$rel") ;;
    *.yml|*.yaml) YAML_FILES+=("$rel") ;;
    *.sh|*.bash) SH_FILES+=("$rel") ;;
  esac
done

if [[ ${#RESOLVED_FILES[@]} -eq 0 ]]; then
  echo "ERROR: no valid changed files to validate"
  exit 2
fi

echo "== Targeted checks =="
printf '  - %s\n' "${RESOLVED_FILES[@]}"

if [[ ${#PY_FILES[@]} -gt 0 ]]; then
  echo
  echo "== Python compile check =="
  python - "${PY_FILES[@]}" <<'PY'
import py_compile
import sys

failed = []
for path in sys.argv[1:]:
    try:
        py_compile.compile(path, doraise=True)
        print(f"[OK] {path}")
    except Exception as exc:
        failed.append((path, str(exc)))

if failed:
    print("\n[FAIL] Python compile errors:")
    for path, err in failed:
        print(f"- {path}: {err}")
    raise SystemExit(1)
PY
fi

if [[ ${#YAML_FILES[@]} -gt 0 ]]; then
  echo
  echo "== YAML syntax check =="
  python - "${YAML_FILES[@]}" <<'PY'
import sys

try:
    import yaml
except Exception as exc:
    print(f"[FAIL] PyYAML is required for YAML checks: {exc}")
    raise SystemExit(1)

failed = []
for path in sys.argv[1:]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            yaml.safe_load(f)
        print(f"[OK] {path}")
    except Exception as exc:
        failed.append((path, str(exc)))

if failed:
    print("\n[FAIL] YAML parse errors:")
    for path, err in failed:
        print(f"- {path}: {err}")
    raise SystemExit(1)
PY
fi

if [[ ${#SH_FILES[@]} -gt 0 ]]; then
  echo
  echo "== Shell syntax check =="
  for path in "${SH_FILES[@]}"; do
    bash -n "$path"
    echo "[OK] $path"
  done
fi

if [[ $WITH_TESTS -eq 1 ]]; then
  echo
  echo "== Targeted tests =="
  if ! command -v pytest >/dev/null 2>&1; then
    echo "WARN: pytest is not available; skipping tests"
    exit 0
  fi

  RUN_GLIDE_TESTS=0
  RUN_CACHE_TEST=0
  for rel in "${RESOLVED_FILES[@]}"; do
    case "$rel" in
      allatom_design/eval/glide/*|allatom_design/tests/glide/*)
        RUN_GLIDE_TESTS=1
        ;;
    esac
    case "$rel" in
      allatom_design/tests/test_cache_residue_data.py|allatom_design/data/datasets/*)
        RUN_CACHE_TEST=1
        ;;
    esac
  done

  if [[ $RUN_GLIDE_TESTS -eq 1 ]]; then
    pytest -q allatom_design/tests/glide
  fi
  if [[ $RUN_CACHE_TEST -eq 1 ]]; then
    pytest -q allatom_design/tests/test_cache_residue_data.py
  fi
  if [[ $RUN_GLIDE_TESTS -eq 0 && $RUN_CACHE_TEST -eq 0 ]]; then
    echo "No area-specific test mapping matched; run explicit pytest targets when needed."
  fi
fi

echo
echo "All requested targeted checks completed."
