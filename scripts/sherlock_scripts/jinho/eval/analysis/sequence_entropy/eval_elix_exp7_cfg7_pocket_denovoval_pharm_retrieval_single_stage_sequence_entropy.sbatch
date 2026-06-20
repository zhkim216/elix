#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/analysis/sequence_entropy/eval_elix_exp7_cfg7_pocket_denovoval_pharm_retrieval_single_stage_sequence_entropy.sbatch
# Optional focused runs:
#   PRESET_FILTER=top8 <submit command above>
#   ANALYZE_ONLY=true PRESET_FILTER=top8 SLURM_ARRAY_TASK_ID=0 bash scripts/sherlock_scripts/jinho/eval/analysis/sequence_entropy/eval_elix_exp7_cfg7_pocket_denovoval_pharm_retrieval_single_stage_sequence_entropy.sbatch
#   PRINT_ARRAY_PLAN=true bash scripts/sherlock_scripts/jinho/eval/analysis/sequence_entropy/eval_elix_exp7_cfg7_pocket_denovoval_pharm_retrieval_single_stage_sequence_entropy.sbatch
#
# Array layout for the full run is 4 pharm retrieval presets x 2 chunks = 8 tasks:
#   baseline: query/original only, no pharm-retrieval ensemble
#   top8:     query/original + ranks 0..6
#   middle8:  query/original + ranks 12..18
#   tail8:    query/original + ranks 25..31
#SBATCH --job-name=e7c7_seqent
#SBATCH -p possu
#SBATCH --array=0-7
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/e7c7_seqent_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/e7c7_seqent_%A_%a.err

set -euo pipefail

PRESET_FILTER="${PRESET_FILTER:-all}"
PRINT_ARRAY_PLAN="${PRINT_ARRAY_PLAN:-false}"
ANALYZE_ONLY="${ANALYZE_ONLY:-false}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

PRESET_ROWS=(
  "baseline|baseline_no_pocket_ensemble|false|[]|1"
  "top8|top8|true|[0,1,2,3,4,5,6]|8"
)

MODEL_NAME=elix_exp7_cfg7_context_skip_edge_update_decoder_skip
STEP=102500
PROTEIN_NOISE_STD=0.0
SMALL_MOLECULE_NOISE_STD=0.0
METAL_NOISE_STD=0.0
NUM_CHUNKS=2

BATCH_SIZE=8
NUM_SEQS_PER_PDB=100
SAMPLING_LENGTHS=null
POCKET_DISTANCE_FOR_DOCKING_METRICS=8.0
POCKET_DISTANCES_FOR_SEQ_RECOVERY="[6.0,8.0,10.0,12.0]"
POCKET_DISTANCE_BINS="[[6.0,8.0],[8.0,10.0],[10.0,12.0]]"

POCKET_CUMULATIVE_DISTANCES=(6 8 10 12)
POCKET_SHELL_BINS=(6:8 8:10 10:12)
POCKET_ANNOTATION_METHOD=calpha

matches_preset_filter() {
  local preset_key="$1"
  local preset_label="$2"
  [[ "${PRESET_FILTER}" == "all" || "${PRESET_FILTER}" == "${preset_key}" || "${PRESET_FILTER}" == "${preset_label}" ]]
}

print_valid_filters() {
  echo "Valid PRESET_FILTER values: all" >&2
  for ROW in "${PRESET_ROWS[@]}"; do
    IFS='|' read -r FILTER_PRESET_KEY FILTER_PRESET_LABEL _ <<< "${ROW}"
    echo "  ${FILTER_PRESET_KEY} ${FILTER_PRESET_LABEL}" >&2
  done
}

SELECTED_PRESETS=()
for ROW in "${PRESET_ROWS[@]}"; do
  IFS='|' read -r CANDIDATE_PRESET_KEY CANDIDATE_PRESET_LABEL _ <<< "${ROW}"
  matches_preset_filter "${CANDIDATE_PRESET_KEY}" "${CANDIDATE_PRESET_LABEL}" || continue
  SELECTED_PRESETS+=("${ROW}")
done

NUM_PRESETS=${#SELECTED_PRESETS[@]}
if [[ "${NUM_PRESETS}" -eq 0 ]]; then
  echo "No pharm retrieval presets selected by PRESET_FILTER=${PRESET_FILTER}." >&2
  print_valid_filters
  exit 2
fi

NUM_TASKS=$((NUM_PRESETS * NUM_CHUNKS))

if [[ "${PRINT_ARRAY_PLAN}" == "true" ]]; then
  echo "Selected exp7_cfg7 pocket denovoval pharm-retrieval single-stage sequence entropy run: NUM_PRESETS=${NUM_PRESETS}, NUM_CHUNKS=${NUM_CHUNKS}, NUM_TASKS=${NUM_TASKS}, NUM_SEQS_PER_PDB=${NUM_SEQS_PER_PDB}"
  for PRESET_IDX in "${!SELECTED_PRESETS[@]}"; do
    IFS='|' read -r PLAN_PRESET_KEY PLAN_PRESET_LABEL PLAN_ENABLE_PHARM PLAN_RANK_INDICES PLAN_TOTAL_MEMBERS <<< "${SELECTED_PRESETS[$PRESET_IDX]}"
    echo "preset_idx=${PRESET_IDX} preset=${PLAN_PRESET_KEY} label=${PLAN_PRESET_LABEL} pharm_retrieval=${PLAN_ENABLE_PHARM} rank_indices=${PLAN_RANK_INDICES} total_members=${PLAN_TOTAL_MEMBERS} model=${MODEL_NAME} step=${STEP}"
  done
  exit 0
fi

if [[ "${TASK_ID}" -ge "${NUM_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} is outside selected tasks=${NUM_TASKS}; exiting without work."
  exit 0
fi

PRESET_IDX=$((TASK_ID / NUM_CHUNKS))
CHUNK_IDX=$((TASK_ID % NUM_CHUNKS))
IFS='|' read -r PRESET_KEY PRESET_LABEL ENABLE_PHARM_RETRIEVAL RANK_INDICES TOTAL_MEMBERS <<< "${SELECTED_PRESETS[$PRESET_IDX]}"

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/eval_elix_exp7_cfg7_denovoval_pharm_retrieval_sequence_entropy
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=denovoval_pharm_retrieval_ref_ligand_proteins
PHARM_RETRIEVAL_ROOT=/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs/denovoval_pharm_retrieval/ref_ligand_proteins
PHARM_RETRIEVAL_DATASET_ROOT="$(dirname "${PHARM_RETRIEVAL_ROOT}")"
PDB_DIR="${PHARM_RETRIEVAL_ROOT}/cifs"
SELECTED_QUERIES_TSV=${PHARM_RETRIEVAL_DATASET_ROOT}/selected_queries.tsv

DENOISER_ROOT=/scratch/users/zhkim216/out_dir/ligand_seq_des
DENOISER_TRAIN_DIR=${DENOISER_ROOT}/${MODEL_NAME}
EXP_NAME=${MODEL_NAME}_pocket_${VALSET_NAME}_${PRESET_LABEL}_single_stage_pharm_ens_n${TOTAL_MEMBERS}_seqent_pseqs${NUM_SEQS_PER_PDB}
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}/chunk_${CHUNK_IDX}
PDB_NAME_LIST=${HYDRA_DIR}/pharm_retrieval_query_originals.txt
SAMPLING_INPUTS_CSV=${HYDRA_DIR}/pharm_retrieval_query_sampling_inputs.csv
SAMPLE_METADATA_GLOB="${BASE_OUT_DIR}/${EXP_NAME}/step_${STEP}_epoch_*/samples/sample_metadata_array_${CHUNK_IDX}.pt"
ANALYSIS_OUTPUT_DIR="${BASE_OUT_DIR}/${EXP_NAME}/sequence_entropy/chunk_${CHUNK_IDX}"

checkpoint_exists_for_step() {
  local train_dir="$1"
  local step="$2"
  compgen -G "${train_dir}/checkpoints/ema/sd-step${step}-epoch*-ema*.ckpt" >/dev/null || \
    compgen -G "${train_dir}/checkpoints/sd-step${step}-epoch*.ckpt" >/dev/null
}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

if [[ "${ANALYZE_ONLY}" != "true" ]]; then
  [[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
  [[ -f "${SELECTED_QUERIES_TSV}" ]] || { echo "Missing SELECTED_QUERIES_TSV=${SELECTED_QUERIES_TSV}" >&2; exit 1; }
  [[ -d "${DENOISER_TRAIN_DIR}" ]] || { echo "Missing DENOISER_TRAIN_DIR=${DENOISER_TRAIN_DIR}" >&2; exit 1; }
  checkpoint_exists_for_step "${DENOISER_TRAIN_DIR}" "${STEP}" || {
    echo "Missing checkpoint for MODEL_NAME=${MODEL_NAME}, STEP=${STEP} under ${DENOISER_TRAIN_DIR}/checkpoints or checkpoints/ema" >&2
    exit 1
  }

  python3 - "${PDB_DIR}" "${SELECTED_QUERIES_TSV}" "${PDB_NAME_LIST}" "${SAMPLING_INPUTS_CSV}" "${RANK_INDICES}" <<'PY'
import ast
import csv
import re
import sys
from pathlib import Path

pdb_dir = Path(sys.argv[1])
selected_queries_tsv = Path(sys.argv[2])
pdb_name_list = Path(sys.argv[3])
sampling_inputs_csv = Path(sys.argv[4])
required_ranks_raw = sys.argv[5]
rank_re = re.compile(r"(?:^|_)rank(?P<rank>\d+)(?:_|$)")
required_ranks = set(ast.literal_eval(required_ranks_raw))

with selected_queries_tsv.open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

if not rows:
    raise SystemExit(f"No rows found in {selected_queries_tsv}")

query_list_rows = []
sampling_rows = []
for row in rows:
    query_id = row["query_ccd"]
    query_dir = pdb_dir / query_id
    if not query_dir.is_dir():
        raise SystemExit(f"Missing query directory: {query_dir}")

    query_members = sorted(
        path for path in query_dir.glob("*.cif")
        if rank_re.search(path.stem) is None
    )
    if len(query_members) != 1:
        raise SystemExit(
            f"Expected exactly one rankless query/original CIF in {query_dir}, "
            f"found {len(query_members)}"
        )

    rank_members = {}
    duplicate_ranks = {}
    for path in sorted(query_dir.glob("*.cif")):
        match = rank_re.search(path.stem)
        if match is None:
            continue
        rank = int(match.group("rank"))
        if rank in rank_members:
            duplicate_ranks.setdefault(rank, [rank_members[rank]]).append(path)
            continue
        rank_members[rank] = path
    if duplicate_ranks:
        raise SystemExit(f"Duplicate rank CIFs in {query_dir}: {duplicate_ranks}")
    missing_ranks = sorted(required_ranks - set(rank_members))
    if missing_ranks:
        raise SystemExit(f"Missing ranks in {query_dir}: {missing_ranks}")

    query_member = query_members[0]
    query_list_rows.append(str(query_member.relative_to(pdb_dir)))
    sampling_rows.append(
        {
            "pdb_key": query_member.stem,
            "pdb_id": query_id,
            "query_pn_unit_iids": repr([row["query_pn_unit_iid"]]),
            "ccd_code": query_id,
        }
    )

pdb_name_list.parent.mkdir(parents=True, exist_ok=True)
pdb_name_list.write_text("\n".join(query_list_rows) + "\n")

with sampling_inputs_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["pdb_key", "pdb_id", "query_pn_unit_iids", "ccd_code"],
    )
    writer.writeheader()
    writer.writerows(sampling_rows)

print(
    f"Wrote {len(query_list_rows)} query originals to {pdb_name_list} and "
    f"{len(sampling_rows)} sampling-input rows to {sampling_inputs_csv}"
)
PY

  cd "${BASE_DIR}"

  export PYTHONWARNINGS="ignore"
  WANDB_ID=zhkim216
  PROJECT=eval_ligand_seq_des

  COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
  MAJOR=$(echo "${COMPUTE_CAP}" | cut -d. -f1)

  if [[ "${MAJOR}" -ge 8 ]]; then
      FLASH_ATTN_IMPL="triton"
  else
      FLASH_ATTN_IMPL="xla"
      export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
  fi

  ENSEMBLE_OVERRIDES=()
  if [[ "${ENABLE_PHARM_RETRIEVAL}" == "true" ]]; then
    ENSEMBLE_OVERRIDES+=(
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.enabled=true"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.total_members=${TOTAL_MEMBERS}"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.weights.scheme=mean"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_seed=null"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.mode=gaussian_noise"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.noise_std=${PROTEIN_NOISE_STD}"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.mode=gaussian_noise"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.noise_std=${METAL_NOISE_STD}"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.mode=pharm_retrieval"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.noise_std=${SMALL_MOLECULE_NOISE_STD}"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.pharm_retrieval.cif_root=${PDB_DIR}"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.pharm_retrieval.selected_queries_tsv=${SELECTED_QUERIES_TSV}"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.pharm_retrieval.rank_indices=${RANK_INDICES}"
    )
  else
    ENSEMBLE_OVERRIDES+=(
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.enabled=false"
      "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.total_members=1"
    )
  fi

  echo "Starting exp7_cfg7 pocket denovoval pharm-retrieval single-stage sequence entropy run: task=${TASK_ID}/${NUM_TASKS}, preset=${PRESET_KEY}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}"
  echo "Pocket model: model=${MODEL_NAME}, step=${STEP}, pharm_retrieval=${ENABLE_PHARM_RETRIEVAL}, rank_indices=${RANK_INDICES}, total_members=${TOTAL_MEMBERS}, protein_noise_std=${PROTEIN_NOISE_STD}, small_molecule_noise_std=${SMALL_MOLECULE_NOISE_STD}, metal_noise_std=${METAL_NOISE_STD}"
  echo "Sampling count: num_seqs_per_pdb=${NUM_SEQS_PER_PDB}"
  echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"
  echo "PDB_DIR=${PDB_DIR}"
  echo "PDB_NAME_LIST=${PDB_NAME_LIST}"
  echo "SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}"
  echo "SELECTED_QUERIES_TSV=${SELECTED_QUERIES_TSV}"
  echo "EXP_NAME=${EXP_NAME}"

  COMMON_ARGS=(
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling"
    --config-name run_elix
    wandb.no_wandb=false
    wandb.project=${PROJECT}
    wandb.wandb_id=${WANDB_ID}
    wandb.group=${VALSET_NAME}_single_stage_pharm_retrieval_sequence_entropy
    debug=false
    seed=0
    num_workers=${CPUS_PER_TASK}
    protein_only=false
    run_seq_des=true
    exp_name=${EXP_NAME}
    base_out_dir=${BASE_OUT_DIR}
    model_cfg.model_name=elix_mpnn
    model_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR}
    model_cfg.ckpt_cfg.start_step=${STEP}
    model_cfg.ckpt_cfg.end_step=${STEP}
    model_cfg.ckpt_cfg.eval_every_n_ckpts=1
    model_cfg.ckpt_cfg.eval_last_ckpt=false
    model_cfg.ckpt_cfg.use_ema=true
    sampling_cfg.base_cfg_path="${BASE_DIR}/allatom_design/configs/seq_des/elix_mpnn_inference.yaml"
    sampling_cfg.overrides.batch_size=${BATCH_SIZE}
    sampling_cfg.overrides.num_seqs_per_pdb=${NUM_SEQS_PER_PDB}
    sampling_cfg.overrides.use_potts_sampling=true
    sampling_cfg.overrides.potts_sampling_cfg.potts_sweeps=500
    sampling_cfg.overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=true
    sampling_cfg.overrides.potts_sampling_cfg.rejection_step=false
    sampling_cfg.guidance.enabled=false
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV}
    pos_constraint_csv=null
    pocket_cfg.pocket_distance_for_docking_metrics=${POCKET_DISTANCE_FOR_DOCKING_METRICS}
    "pocket_cfg.pocket_distances_for_seq_recovery=${POCKET_DISTANCES_FOR_SEQ_RECOVERY}"
    "pocket_cfg.pocket_distance_bins=${POCKET_DISTANCE_BINS}"
    pocket_cfg.n_min_ligand_atoms_for_seq_recovery=1
    input_sample_is_all_atom=true
    input_sample_is_designed=true
    cif_cfg.parse.designed_samples.add_missing_atoms=true
    load_from_cache=false
    pdb_cfg.pdb_dir=${PDB_DIR}
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST}
    pdb_cfg.pdb_name_ext=".cif"
    pdb_cfg.recursive=false
    pdb_cfg.split_by_subfolder=false
    pdb_cfg.sample_indices=null
    "pdb_cfg.sample_lengths=${SAMPLING_LENGTHS}"
    pdb_cfg.array_id=${CHUNK_IDX}
    pdb_cfg.num_arrays=${NUM_CHUNKS}
    cif_cfg.save.include_bonds=false
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data"
    struct_pred_cfg.evaluate_self_consistency=false
    struct_pred_cfg.evaluate_docking_consistency=false
    struct_pred_cfg.calculate_metrics_only=false
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL}
    hydra.run.dir=${HYDRA_DIR}
  )

  WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/run_elix.py \
    "${COMMON_ARGS[@]}" \
    "${ENSEMBLE_OVERRIDES[@]}"
else
  cd "${BASE_DIR}"
  echo "ANALYZE_ONLY=true; skipping sampling and analyzing existing metadata."
fi

echo "Running sequence entropy analysis"
echo "SAMPLE_METADATA_GLOB=${SAMPLE_METADATA_GLOB}"
echo "ANALYSIS_OUTPUT_DIR=${ANALYSIS_OUTPUT_DIR}"

python3 -m allatom_design.eval.analysis.sequence_entropy \
  --metadata-path "${SAMPLE_METADATA_GLOB}" \
  --output-dir "${ANALYSIS_OUTPUT_DIR}" \
  --pocket-cumulative-distances "${POCKET_CUMULATIVE_DISTANCES[@]}" \
  --pocket-shell-bins "${POCKET_SHELL_BINS[@]}" \
  --pocket-annotation-method "${POCKET_ANNOTATION_METHOD}"

echo "Sequence entropy complete: preset=${PRESET_KEY}, model=${MODEL_NAME}, step=${STEP}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}, output=${ANALYSIS_OUTPUT_DIR}"
