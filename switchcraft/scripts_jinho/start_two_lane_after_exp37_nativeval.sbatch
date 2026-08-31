#!/bin/bash
#SBATCH --job-name=switchcraft_after_exp37
#SBATCH --partition=normal
#SBATCH --account=possu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%j.out
#SBATCH --error=/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/logs/%x_%j.err

set -euo pipefail

SCRIPT_ROOT=/home/users/zhkim216/code/switchcraft/scripts_jinho
CONTROLLER="$SCRIPT_ROOT/two_lane_generation.py"
RUN_ROOT="${SWITCHCRAFT_RUN_ROOT:-/scratch/users/zhkim216/experiment_result_analysis/benchmarks/state_switching/denovoval/manifests/two_lane_1h2h_v1}"
STATUS_ROOT="${EXP37_STATUS_ROOT:-/scratch/users/zhkim216/job_output/eval_ligand_seq_des/retry_status/exp37_nativeval_roleauto_modes_seq2_upto52500_v1}"
WATCHER_JOB_ID="${EXP37_WATCHER_JOB_ID:?EXP37_WATCHER_JOB_ID must be exported at submission}"
EXPECTED_CHUNKS="${EXP37_EXPECTED_CHUNKS:-144}"

WATCHER_STATE="$(
  sacct -X -n -P -j "$WATCHER_JOB_ID" --format=JobIDRaw,State |
    awk -F'|' -v job_id="$WATCHER_JOB_ID" '$1 == job_id {print $2; exit}'
)"
WATCHER_STATE="${WATCHER_STATE%% *}"
WATCHER_STATE="${WATCHER_STATE%%+}"
if [[ "$WATCHER_STATE" != "COMPLETED" ]]; then
  echo "Refusing SwitchCraft resume: exp37 watcher $WATCHER_JOB_ID state is ${WATCHER_STATE:-missing}." >&2
  exit 1
fi

python3 - "$STATUS_ROOT" "$EXPECTED_CHUNKS" <<'PY'
import json
import sys
from pathlib import Path

status_root = Path(sys.argv[1])
expected_chunks = int(sys.argv[2])
summary_path = status_root / "summary.json"
state_path = status_root / "state.json"
if not summary_path.is_file() or not state_path.is_file():
    raise SystemExit(
        f"Refusing SwitchCraft resume: missing exp37 status artifacts under {status_root}"
    )

summary = json.loads(summary_path.read_text())
state = json.loads(state_path.read_text())
errors = []
expected_summary = {
    "campaign": "exp37",
    "num_sequences_per_input": 2,
    "last_eval_step": 52500,
    "allow_complete_with_metric_errors": False,
    "n_expected_targets": 36,
    "n_discovered_targets": 36,
    "stop_requested": False,
}
for key, expected in expected_summary.items():
    if summary.get(key) != expected:
        errors.append(f"summary.{key}={summary.get(key)!r}, expected {expected!r}")

targets = state.get("targets", {})
chunks = [
    chunk
    for target in targets.values()
    for chunk in target.get("chunks", {}).values()
]
counts = summary.get("counts", {})
if counts != {"complete": expected_chunks}:
    errors.append(f"summary.counts={counts!r}, expected {{'complete': {expected_chunks}}}")
if len(targets) != 36:
    errors.append(f"state target count={len(targets)}, expected 36")
if len(chunks) != expected_chunks:
    errors.append(f"state chunk count={len(chunks)}, expected {expected_chunks}")
noncomplete = [chunk.get("terminal_state") for chunk in chunks if chunk.get("terminal_state") != "complete"]
if noncomplete:
    errors.append(f"state has {len(noncomplete)} non-complete chunks")
if state.get("source_contract_sha256") != summary.get("source_contract_sha256"):
    errors.append("state/summary source contract hashes differ")
if errors:
    raise SystemExit("Refusing SwitchCraft resume: " + "; ".join(errors))

print(
    json.dumps(
        {
            "event": "exp37_nativeval_gate_passed",
            "status_root": str(status_root),
            "targets": len(targets),
            "chunks": len(chunks),
            "source_contract_sha256": summary["source_contract_sha256"],
        },
        sort_keys=True,
    )
)
PY

exec python3 -B "$CONTROLLER" finalize \
  --run-root "$RUN_ROOT" \
  --batch-index 0 \
  --attempt 1 \
  --retry-cancelled \
  --retry-missing-accounting
