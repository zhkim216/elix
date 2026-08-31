#!/usr/bin/env bash
#SBATCH --job-name=dnv_aug_curate
#SBATCH --partition=normal,owners,possu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/runs/seed8_n100_hetero_rasa/logs/%x_%j.out
#SBATCH --error=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/runs/seed8_n100_hetero_rasa/logs/%x_%j.err

set -euo pipefail

expected_venv=/scratch/users/zhkim216/envs/uv/elix-torch280
python_bin="${expected_venv}/bin/python"
scripts_root=/home/users/zhkim216/code/elix/foundry/scripts/sherlock_scripts/curation/denovoval_stratified
curator="${scripts_root}/curate_cross_condition_panels.py"
cache_preparer="${scripts_root}/prepare_augmented_foldseek_cache.py"
run_root=/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_cross_condition_foldseek_v1/runs/seed8_n100_hetero_rasa
rasa_manifest="${run_root}/rasa/input_manifest.tsv"
rasa_results="${run_root}/rasa/results/ligand_rasa.tsv"
rasa_validation="${run_root}/rasa/validation.json"
foldseek_bin=/oak/stanford/groups/possu/jinho/software/foldseek/bin/foldseek

if [[ "${VIRTUAL_ENV:-}" != "${expected_venv}" ]]; then
  echo "Run this job through wrap_sbatch_in_container_elix.sh." >&2
  exit 2
fi
for path in \
  "${python_bin}" \
  "${curator}" \
  "${cache_preparer}" \
  "${rasa_manifest}" \
  "${rasa_results}" \
  "${rasa_validation}" \
  "${foldseek_bin}"; do
  [[ -f "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 2; }
done
for path in \
  "${run_root}/foldseek/groups" \
  "${run_root}/denovoval_re" \
  "${run_root}/denovoval_subset" \
  "${run_root}/curation_manifest.json"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite curation artifact: ${path}" >&2
    exit 2
  fi
done

"${python_bin}" -B - "${rasa_validation}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
if payload.get("status") != "complete" or payload.get("result_rows") != 2100:
    raise SystemExit(f"Corrected RASA validation is incomplete: {payload}")
if payload.get("target_residue_count") != 1:
    raise SystemExit(f"Corrected RASA target contract is invalid: {payload}")
print("RASA_READY rows=2100 target_residues=1")
PY

date --iso-8601=seconds
hostname
"${python_bin}" -B "${cache_preparer}" --run-root "${run_root}"

# Recluster and selection-test only the four groups whose candidate pools grew.
"${python_bin}" -B "${curator}" \
  --output-root "${run_root}" \
  --rasa-override-manifest "${rasa_manifest}" \
  --rasa-override-results "${rasa_results}" \
  --foldseek-bin "${foldseek_bin}" \
  --workers 4 \
  --threads-per-foldseek 4 \
  --groups G2P_len300 GLC_len300 GLU_len150 GLU_len300

# The first call fails closed if any affected group cannot supply 12 joint-unique slots.
# On success, all 308 cached group fingerprints are revalidated before materialization.
"${python_bin}" -B "${curator}" \
  --output-root "${run_root}" \
  --rasa-override-manifest "${rasa_manifest}" \
  --rasa-override-results "${rasa_results}" \
  --foldseek-bin "${foldseek_bin}" \
  --workers 4 \
  --threads-per-foldseek 4
date --iso-8601=seconds
