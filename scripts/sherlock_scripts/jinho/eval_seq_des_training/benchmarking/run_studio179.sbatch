#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval_seq_des_training/benchmarking/run_studio179.sbatch
#SBATCH --job-name=studio179_benchmark
#SBATCH -p possu,bioe
#SBATCH --array=0-19
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/studio179_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/studio179_%A_%a.err

set -euo pipefail

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

MODEL_NAMES=(
  proto_nprot1_no_filter
  proto_nprot1_external_evidence
  proto_allprot_no_filter
  proto_allprot_external_evidence
)
STEPS=(22500 42500 62500 102500 122500)

NUM_MODELS=${#MODEL_NAMES[@]}
NUM_STEPS=${#STEPS[@]}
NUM_TASKS=$((NUM_MODELS * NUM_STEPS))
if (( TASK_ID < 0 || TASK_ID >= NUM_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside model/step tasks=${NUM_TASKS}; exiting without work."
  exit 0
fi

MODEL_IDX=$((TASK_ID / NUM_STEPS))
STEP_IDX=$((TASK_ID % NUM_STEPS))
MODEL_NAME="${MODEL_NAMES[$MODEL_IDX]}"
STEP="${STEPS[$STEP_IDX]}"

if [[ -z "${MODEL_NAME}" ]]; then
  echo "MODEL_NAMES[${MODEL_IDX}] is empty." >&2
  exit 2
fi

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
PYTHON="${PYTHON:-/scratch/users/zhkim216/envs/uv/elix/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python3
fi

DISCO_ROOT=/scratch/users/zhkim216/datasets/val_cifs/DISCO_benchmark_data
STUDIO179_ROOT=${DISCO_ROOT}/disco_inference_benchmarks_release_data/studio-179
OUT_DIR=${STUDIO179_ROOT}/benchmark_runs
PDB_DIR=${STUDIO179_ROOT}/cifs
USER_CCD_MANIFEST=${STUDIO179_ROOT}/conformer_cifs/studio179_userccd_manifest.tsv
USER_CCD_PATH=${STUDIO179_ROOT}/conformer_cifs/studio179_all_components_userccd.cif
TARGETS="CU,MN,NI,ZN,FE,HG,PB"
LENGTHS="150,250"
SAMPLE_INDICES="0,1,2,3,4,5,6,7,8,9"

compact_csv() {
  printf '%s' "$1" | tr -d '[:space:],'
}

count_csv_items() {
  local value="$1"
  awk -v csv="${value}" 'BEGIN {
    n = split(csv, parts, ",")
    count = 0
    for (i = 1; i <= n; i++) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", parts[i])
      if (parts[i] != "") {
        count++
      }
    }
    print count
  }'
}

VALSET_NAME=studio179_metal_ions
LENGTH_TAG="len$(compact_csv "${LENGTHS}")"
NUM_SAMPLE_INDICES="$(count_csv_items "${SAMPLE_INDICES}")"
SAMPLE_INDEX_TAG="num${NUM_SAMPLE_INDICES}"
INPUT_RUN_ID=metal_ions_${LENGTH_TAG}_${SAMPLE_INDEX_TAG}
EXP_NAME=studio179_metal_ions_${MODEL_NAME}_${LENGTH_TAG}_${SAMPLE_INDEX_TAG}
PDB_NAME_LIST=${OUT_DIR}/studio179_${INPUT_RUN_ID}.txt
SELECTED_MANIFEST_TSV=${OUT_DIR}/studio179_${INPUT_RUN_ID}_manifest.tsv
SUMMARY_JSON=${OUT_DIR}/studio179_${INPUT_RUN_ID}_summary.json
PREPARE_LOCK=${OUT_DIR}/studio179_${INPUT_RUN_ID}.prepare.lock

BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/${MODEL_NAME}
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs/studio179/${MODEL_NAME}/step_${STEP}
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des
NUM_SEQS_PER_PDB=1
BATCH_SIZE=8
AF3_RECYCLES=10
AF3_DIFFUSION_SAMPLES=5
POCKET_DISTANCES_FOR_SEQ_RECOVERY="[8.0,10.0]"
POCKET_DISTANCE_FOR_DOCKING_METRICS=10.0

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${OUT_DIR}" "${BASE_OUT_DIR}" "${HYDRA_DIR}"

[[ -d "${BASE_DIR}" ]] || { echo "Missing BASE_DIR=${BASE_DIR}" >&2; exit 1; }
[[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
[[ -f "${USER_CCD_MANIFEST}" ]] || { echo "Missing USER_CCD_MANIFEST=${USER_CCD_MANIFEST}" >&2; exit 1; }
[[ -f "${USER_CCD_PATH}" ]] || { echo "Missing USER_CCD_PATH=${USER_CCD_PATH}" >&2; exit 1; }

cd "${BASE_DIR}"

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "${COMPUTE_CAP}" | cut -d. -f1)
if [[ "${MAJOR}" -ge 8 ]]; then
  FLASH_ATTN_IMPL="triton"
else
  FLASH_ATTN_IMPL="xla"
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi

echo "Starting Studio-179 metal-ion benchmark: model=${MODEL_NAME}, step=${STEP}, task=${TASK_ID}/${NUM_TASKS}"
echo "Selection: targets=${TARGETS} lengths=${LENGTHS} sample_indices=${SAMPLE_INDICES} sample_index_count=${NUM_SAMPLE_INDICES}"
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"
echo "PDB_DIR=${PDB_DIR}"
echo "USER_CCD_MANIFEST=${USER_CCD_MANIFEST}"
echo "USER_CCD_PATH=${USER_CCD_PATH}"
echo "PDB_NAME_LIST=${PDB_NAME_LIST}"
echo "SELECTED_MANIFEST_TSV=${SELECTED_MANIFEST_TSV}"
echo "BASE_OUT_DIR=${BASE_OUT_DIR}"
echo "DENOISER_TRAIN_DIR=${DENOISER_TRAIN_DIR}"
echo "HYDRA_DIR=${HYDRA_DIR}"
echo "PYTHON=${PYTHON}"

echo "Preparing shared Studio-179 lc_seq_des_multi inputs if needed..."
(
  flock 200
  if [[ ! -s "${PDB_NAME_LIST}" || ! -s "${SELECTED_MANIFEST_TSV}" || ! -s "${SUMMARY_JSON}" ]]; then
    "${PYTHON}" -m allatom_design.eval.benchmarking.studio179.prepare_lc_seq_des_inputs \
      --disco-root "${DISCO_ROOT}" \
      --out-dir "${OUT_DIR}" \
      --user-ccd-manifest "${USER_CCD_MANIFEST}" \
      --user-ccd-path "${USER_CCD_PATH}" \
      --target "${TARGETS}" \
      --length "${LENGTHS}" \
      --sample-index "${SAMPLE_INDICES}" \
      --run-id "${INPUT_RUN_ID}"
  else
    echo "Using existing prepared Studio-179 inputs: ${PDB_NAME_LIST}"
  fi
) 200>"${PREPARE_LOCK}"

[[ -f "${PDB_NAME_LIST}" ]] || { echo "Missing prepared PDB_NAME_LIST=${PDB_NAME_LIST}" >&2; exit 1; }
[[ -f "${SELECTED_MANIFEST_TSV}" ]] || { echo "Missing prepared SELECTED_MANIFEST_TSV=${SELECTED_MANIFEST_TSV}" >&2; exit 1; }
[[ -f "${SUMMARY_JSON}" ]] || { echo "Missing prepared SUMMARY_JSON=${SUMMARY_JSON}" >&2; exit 1; }

SAMPLE_COUNT=$(wc -l < "${PDB_NAME_LIST}" | tr -d ' ')
if [[ "${SAMPLE_COUNT}" -le 0 ]]; then
  echo "Prepared Studio-179 sample list is empty: ${PDB_NAME_LIST}" >&2
  exit 1
fi
echo "Prepared Studio-179 samples: ${SAMPLE_COUNT}"

WANDB__SERVICE_WAIT=300 "${PYTHON}" allatom_design/eval/sampling/lc_seq_des_multi.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
    --config-name lc_seq_des_multi_studio179 \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${VALSET_NAME} \
    debug=false \
    seed=0 \
    num_workers=${CPUS_PER_TASK} \
    protein_only=false \
    run_seq_des=true \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    seq_des_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    seq_des_cfg.ckpt_cfg.start_step=${STEP} \
    seq_des_cfg.ckpt_cfg.end_step=${STEP} \
    seq_des_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    seq_des_cfg.ckpt_cfg.eval_last_ckpt=false \
    seq_des_cfg.ckpt_cfg.use_ema=true \
    seq_des_cfg.atom_mpnn.sampling_cfg="${BASE_DIR}/allatom_design/configs/seq_des/lc_atom_mpnn_inference.yaml" \
    seq_des_cfg.atom_mpnn.overrides.batch_size=${BATCH_SIZE} \
    seq_des_cfg.atom_mpnn.overrides.num_seqs_per_pdb=${NUM_SEQS_PER_PDB} \
    seq_des_cfg.atom_mpnn.overrides.num_workers=${CPUS_PER_TASK} \
    seq_des_cfg.atom_mpnn.overrides.use_potts_sampling=true \
    guidance_cfg.enabled=false \
    sampling_inputs_csv=null \
    pos_constraint_csv=null \
    redesign_cfg.use_native_seq=false \
    redesign_cfg.native_cif_dir=null \
    "pocket_cfg.pocket_distances_for_seq_recovery=${POCKET_DISTANCES_FOR_SEQ_RECOVERY}" \
    pocket_cfg.pocket_distance_for_docking_metrics=${POCKET_DISTANCE_FOR_DOCKING_METRICS} \
    pocket_cfg.n_min_ligand_atoms_for_seq_recovery=1 \
    input_sample_is_all_atom=true \
    input_sample_is_designed=true \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.sample_indices=null \
    pdb_cfg.sample_lengths=null \
    pdb_cfg.array_id=null \
    pdb_cfg.num_arrays=null \
    cif_cfg.parse.designed_samples.add_missing_atoms=true \
    cif_cfg.parse.designed_samples.extra_fields=[] \
    cif_cfg.parse.af3_predictions.add_missing_atoms=true \
    cif_cfg.parse.af3_predictions.extra_fields=[] \
    cif_cfg.save.include_bonds=true \
    cif_cfg.save.exclude_field_keys=[] \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.runner_path="${BASE_DIR}/alphafold3/src/alphafold3/run_alphafold.py" \
    struct_pred_cfg.af3.json_config.version=4 \
    struct_pred_cfg.af3.json_config.user_ccd_path=${USER_CCD_PATH} \
    struct_pred_cfg.af3.inference_config.base.model_dir=/scratch/users/zhkim216/model_params/af3 \
    struct_pred_cfg.af3.inference_config.base.db_dir=/scratch/users/zhkim216/af3_databases \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=${AF3_RECYCLES} \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=${AF3_DIFFUSION_SAMPLES} \
    struct_pred_cfg.af3.inference_config.tc.num_recycles=${AF3_RECYCLES} \
    struct_pred_cfg.af3.inference_config.tc.num_diffusion_samples=${AF3_DIFFUSION_SAMPLES} \
    studio179_userccd.enabled=true \
    studio179_userccd.selected_manifest_tsv=${SELECTED_MANIFEST_TSV} \
    studio179_userccd.user_ccd_path=${USER_CCD_PATH} \
    hydra.run.dir=${HYDRA_DIR}

echo "Studio-179 benchmark complete: model=${MODEL_NAME}, step=${STEP}, samples=${SAMPLE_COUNT}"
