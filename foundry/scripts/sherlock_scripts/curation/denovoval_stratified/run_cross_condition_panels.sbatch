#!/usr/bin/env bash
#SBATCH --job-name=dnv_joint_fs_v1
#SBATCH --partition=normal,owners,possu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/logs/%x_%j.out
#SBATCH --error=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/logs/%x_%j.err

set -euo pipefail

expected_venv=/scratch/users/zhkim216/envs/uv/elix-torch280
python_bin="${expected_venv}/bin/python"
script=/home/users/zhkim216/code/elix/foundry/scripts/sherlock_scripts/curation/denovoval_stratified/curate_cross_condition_panels.py
output_root=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1
ss1p0_source_root="${output_root}/work/ss1p0_source"
foldseek_bin=/oak/stanford/groups/possu/jinho/software/foldseek/bin/foldseek

if [[ "${VIRTUAL_ENV:-}" != "${expected_venv}" ]]; then
  echo "Run this job through wrap_sbatch_in_container_elix.sh." >&2
  exit 2
fi
for path in \
  "${python_bin}" \
  "${script}" \
  "${foldseek_bin}" \
  "${ss1p0_source_root}/raw" \
  "${ss1p0_source_root}/curation_steps/step1/selection/candidate_audit.tsv" \
  "${output_root}/smoke/12C_len150/cluster_status.json"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 2
  fi
done
if [[ -e "${output_root}/denovoval_re" || -e "${output_root}/denovoval_subset" ]]; then
  echo "Refusing to overwrite a materialized output under ${output_root}." >&2
  exit 2
fi

date --iso-8601=seconds
hostname
raw_count=$(find "${ss1p0_source_root}/raw" -type f -name '*.cif.gz' -printf '.' | wc -c)
if [[ "${raw_count}" -ne 30800 ]]; then
  echo "Expected 30800 staged ss1p0 raw CIFs, found ${raw_count}." >&2
  exit 2
fi
for path in \
  "${ss1p0_source_root}/curation_steps/step1/ligand_rasa/ligand_rasa.tsv" \
  "${ss1p0_source_root}/curation_steps/step1/selection/candidate_audit.tsv" \
  "${ss1p0_source_root}/curation_steps/step1/selection/selected_samples.tsv"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing staged ss1p0 metadata: ${path}" >&2
    exit 2
  fi
done

"${python_bin}" -B - "${output_root}/smoke/12C_len150" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
status = json.loads((root / "cluster_status.json").read_text())
selection = status.get("selection_canary") or {}
if status.get("status") != "complete" or selection.get("status") != "complete":
    raise SystemExit("Foldseek/selection canary is not complete")
with (root / "foldseek/groups/12C_len150/cluster_assignments.tsv").open(newline="") as handle:
    assignments = list(csv.DictReader(handle, delimiter="\t"))
with (root / "selection_canary.tsv").open(newline="") as handle:
    selected = list(csv.DictReader(handle, delimiter="\t"))
if len(assignments) != 400 or len(selected) != 12:
    raise SystemExit(
        f"Canary cardinality mismatch: assignments={len(assignments)} selected={len(selected)}"
    )
clusters = [row["foldseek_cluster_id"] for row in selected]
if len(set(clusters)) != 12:
    raise SystemExit("Canary selected clusters are not jointly unique")
if any(float(row["rasa_value"]) >= 0.2 for row in selected):
    raise SystemExit("Canary selected an ineligible RASA value")
print("CANARY_READY assignments=400 selected=12 unique_clusters=12 rasa_lt_0.2")
PY

"${python_bin}" -B "${script}" \
  --output-root "${output_root}" \
  --ss1p0-raw-root "${ss1p0_source_root}/raw" \
  --ss1p0-audit "${ss1p0_source_root}/curation_steps/step1/selection/candidate_audit.tsv" \
  --foldseek-bin "${foldseek_bin}" \
  --workers 4 \
  --threads-per-foldseek 4
date --iso-8601=seconds
