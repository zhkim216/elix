#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/denovoval/eval_elix_exp7_denovoval_sm_metal_half_twostage.sbatch
# Optional focused runs:
#   POCKET_MODEL_FILTER=cfg6 <submit command above>
#   POCKET_MODEL_FILTER=cfg6 STEP_FILTER=102500 <submit command above>
# Sweeps best exp7 cfg2-7 pocket model steps x chunks with no pocket ensemble.
# Array layout for the full half-set run is 6 x 4 = 24 tasks.
#SBATCH --job-name=eval_exp7_denovo_ts_no_pens_half
#SBATCH -p possu,bioe
#SBATCH --array=0-23
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-0:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/eval_exp7_denovo_ts_no_pens_half_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/eval_exp7_denovo_ts_no_pens_half_%A_%a.err

set -euo pipefail

POCKET_MODEL_FILTER="${POCKET_MODEL_FILTER:-all}"
STEP_FILTER="${STEP_FILTER:-all}"
PRINT_ARRAY_PLAN="${PRINT_ARRAY_PLAN:-false}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

CONFIGS=(
  "2|elix_exp7_cfg2_node_concat_decoder_skip|102500"
  "3|elix_exp7_cfg3_node_concat_edge_update_decoder_skip|62500"
  "4|elix_exp7_cfg4_context_skip_potts_only|62500"
  "5|elix_exp7_cfg5_context_skip_edge_update|82500"
  "6|elix_exp7_cfg6_context_skip_decoder_skip|102500"
  "7|elix_exp7_cfg7_context_skip_edge_update_decoder_skip|102500"
)

SCAFFOLD_MODEL_NAME=elix_exp1_cfg0
SCAFFOLD_STEP=32500
SCAFFOLD_TOTAL_MEMBERS=16
SCAFFOLD_PROTEIN_NOISE_STD=0.5
SCAFFOLD_SMALL_MOLECULE_NOISE_STD=0.0
SCAFFOLD_METAL_NOISE_STD=0.0

DIRECTION=ps
POCKET_DISTANCE=10.0
POCKET_ANNOTATION_METHOD=calpha
NUM_CHUNKS=4

BATCH_SIZE=8
NUM_POCKET_SEQS_PER_PDB=2
NUM_SCAFFOLD_SEQS_PER_PDB=1
SAMPLING_LENGTHS="[150,300]"
POCKET_DISTANCE_FOR_DOCKING_METRICS=8.0
POCKET_DISTANCES_FOR_SEQ_RECOVERY="[4.0,6.0,8.0,10.0]"
POCKET_DISTANCE_BINS="[[0.0,4.0],[4.0,6.0],[6.0,8.0],[8.0,10.0]]"
AF3_RECYCLES=10
AF3_DIFFUSION_SAMPLES=5

HALF_DATASET_ROOT=/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs/denovoval_sm_metal_half

matches_step_filter() {
  local value="$1"
  [[ "${STEP_FILTER}" == "all" || "${STEP_FILTER}" == "${value}" ]]
}

matches_model_filter() {
  local cfg_idx="$1"
  local model_name="$2"
  [[ "${POCKET_MODEL_FILTER}" == "all" || "${POCKET_MODEL_FILTER}" == "${cfg_idx}" || "${POCKET_MODEL_FILTER}" == "cfg${cfg_idx}" || "${POCKET_MODEL_FILTER}" == "${model_name}" ]]
}

print_valid_filters() {
  echo "Valid selected cfg/step rows:" >&2
  for ROW in "${CONFIGS[@]}"; do
    IFS='|' read -r FILTER_CFG_IDX FILTER_MODEL_NAME FILTER_POCKET_STEP <<< "${ROW}"
    echo "  cfg=${FILTER_CFG_IDX} model=${FILTER_MODEL_NAME} step=${FILTER_POCKET_STEP}" >&2
  done
  echo "Valid STEP_FILTER values: all or one selected step above." >&2
  echo "Valid POCKET_MODEL_FILTER values: all, cfg index 2-7, cfg2-cfg7, or full model name:" >&2
  for ROW in "${CONFIGS[@]}"; do
    IFS='|' read -r FILTER_CFG_IDX FILTER_MODEL_NAME _ <<< "${ROW}"
    echo "  ${FILTER_CFG_IDX} cfg${FILTER_CFG_IDX} ${FILTER_MODEL_NAME}" >&2
  done
}

# Each combo is "cfg_idx|pocket_model|pocket_step".
COMBOS=()
for ROW in "${CONFIGS[@]}"; do
  IFS='|' read -r CANDIDATE_CFG_IDX CANDIDATE_MODEL_NAME CANDIDATE_POCKET_STEP <<< "${ROW}"
  matches_model_filter "${CANDIDATE_CFG_IDX}" "${CANDIDATE_MODEL_NAME}" || continue
  matches_step_filter "${CANDIDATE_POCKET_STEP}" || continue
  COMBOS+=("${CANDIDATE_CFG_IDX}|${CANDIDATE_MODEL_NAME}|${CANDIDATE_POCKET_STEP}")
done

NUM_COMBOS=${#COMBOS[@]}
if [[ "${NUM_COMBOS}" -eq 0 ]]; then
  echo "No selected best-step combos matched POCKET_MODEL_FILTER=${POCKET_MODEL_FILTER}, STEP_FILTER=${STEP_FILTER}." >&2
  print_valid_filters
  exit 2
fi

NUM_TASKS=$((NUM_COMBOS * NUM_CHUNKS))

if [[ "${PRINT_ARRAY_PLAN}" == "true" ]]; then
  echo "Selected exp7 pocket / exp1_cfg0 scaffold denovoval sm-metal-half two-stage no-pocket-ensemble run: NUM_COMBOS=${NUM_COMBOS}, NUM_CHUNKS=${NUM_CHUNKS}, NUM_TASKS=${NUM_TASKS}"
  for COMBO_IDX in "${!COMBOS[@]}"; do
    IFS='|' read -r PLAN_CFG_IDX PLAN_MODEL_NAME PLAN_POCKET_STEP <<< "${COMBOS[$COMBO_IDX]}"
    echo "combo_idx=${COMBO_IDX} cfg=${PLAN_CFG_IDX} pocket_model=${PLAN_MODEL_NAME} pocket_step=${PLAN_POCKET_STEP} pocket_ensemble=false scaffold_model=${SCAFFOLD_MODEL_NAME} scaffold_step=${SCAFFOLD_STEP} scaffold_total_members=${SCAFFOLD_TOTAL_MEMBERS}"
  done
  exit 0
fi

if [[ "${TASK_ID}" -ge "${NUM_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} is outside selected tasks=${NUM_TASKS}; exiting without work."
  exit 0
fi

COMBO_IDX=$((TASK_ID / NUM_CHUNKS))
CHUNK_IDX=$((TASK_ID % NUM_CHUNKS))
IFS='|' read -r POCKET_CFG_IDX POCKET_MODEL_NAME POCKET_STEP <<< "${COMBOS[$COMBO_IDX]}"

SCAFFOLD_PROTEIN_NOISE_LABEL=${SCAFFOLD_PROTEIN_NOISE_STD//./p}

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/eval_elix_exp7_denovoval_sm_metal_half_twostage_pocket_no_ens
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=denovoval_sm_metal_half
PDB_DIR=${HALF_DATASET_ROOT}/cifs
PDB_NAME_LIST=${HALF_DATASET_ROOT}/denovoval_sm_metal_half.txt

POCKET_DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${POCKET_MODEL_NAME}
SCAFFOLD_DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${SCAFFOLD_MODEL_NAME}
EXP_NAME=${POCKET_MODEL_NAME}_pocket_${SCAFFOLD_MODEL_NAME}_scaffold_${VALSET_NAME}_twostage_${DIRECTION}_pocket_no_ens_scaffold_ens_n${SCAFFOLD_TOTAL_MEMBERS}_protein_noise${SCAFFOLD_PROTEIN_NOISE_LABEL}_pseqs${NUM_POCKET_SEQS_PER_PDB}_sseqs${NUM_SCAFFOLD_SEQS_PER_PDB}
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}/pocket_step${POCKET_STEP}/chunk_${CHUNK_IDX}

checkpoint_exists_for_step() {
  local train_dir="$1"
  local step="$2"
  compgen -G "${train_dir}/checkpoints/ema/sd-step${step}-epoch*-ema*.ckpt" >/dev/null || \
    compgen -G "${train_dir}/checkpoints/sd-step${step}-epoch*.ckpt" >/dev/null
}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_ROOT}" "${BASE_OUT_DIR}"

[[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
[[ -f "${PDB_NAME_LIST}" ]] || { echo "Missing PDB_NAME_LIST=${PDB_NAME_LIST}" >&2; exit 1; }
[[ -d "${POCKET_DENOISER_TRAIN_DIR}" ]] || { echo "Missing POCKET_DENOISER_TRAIN_DIR=${POCKET_DENOISER_TRAIN_DIR}" >&2; exit 1; }
[[ -d "${SCAFFOLD_DENOISER_TRAIN_DIR}" ]] || { echo "Missing SCAFFOLD_DENOISER_TRAIN_DIR=${SCAFFOLD_DENOISER_TRAIN_DIR}" >&2; exit 1; }
checkpoint_exists_for_step "${POCKET_DENOISER_TRAIN_DIR}" "${POCKET_STEP}" || {
  echo "Missing pocket checkpoint for MODEL_NAME=${POCKET_MODEL_NAME}, STEP=${POCKET_STEP} under ${POCKET_DENOISER_TRAIN_DIR}/checkpoints or checkpoints/ema" >&2
  exit 1
}
checkpoint_exists_for_step "${SCAFFOLD_DENOISER_TRAIN_DIR}" "${SCAFFOLD_STEP}" || {
  echo "Missing scaffold checkpoint for MODEL_NAME=${SCAFFOLD_MODEL_NAME}, STEP=${SCAFFOLD_STEP} under ${SCAFFOLD_DENOISER_TRAIN_DIR}/checkpoints or checkpoints/ema" >&2
  exit 1
}

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

echo "Starting exp7 pocket / exp1_cfg0 scaffold denovoval sm-metal-half two-stage no-pocket-ensemble run: task=${TASK_ID}/${NUM_TASKS}, combo=${COMBO_IDX}/${NUM_COMBOS}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}"
echo "Stage1 pocket: cfg=${POCKET_CFG_IDX}, model=${POCKET_MODEL_NAME}, step=${POCKET_STEP}, ensemble_conditioning=false"
echo "Stage2 scaffold: model=${SCAFFOLD_MODEL_NAME}, step=${SCAFFOLD_STEP}, total_members=${SCAFFOLD_TOTAL_MEMBERS}, protein_noise_std=${SCAFFOLD_PROTEIN_NOISE_STD}, small_molecule_noise_std=${SCAFFOLD_SMALL_MOLECULE_NOISE_STD}, metal_noise_std=${SCAFFOLD_METAL_NOISE_STD}"
echo "Filters: POCKET_MODEL_FILTER=${POCKET_MODEL_FILTER}, STEP_FILTER=${STEP_FILTER}"
echo "Stage sampling counts: pocket=${NUM_POCKET_SEQS_PER_PDB}, scaffold=${NUM_SCAFFOLD_SEQS_PER_PDB}"
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"
echo "PDB_DIR=${PDB_DIR}"
echo "PDB_NAME_LIST=${PDB_NAME_LIST}"
echo "SAMPLING_LENGTHS=${SAMPLING_LENGTHS}"
echo "POCKET_DISTANCE_BINS=${POCKET_DISTANCE_BINS}"
echo "AF3 SS: recycles=${AF3_RECYCLES}, diffusion_samples=${AF3_DIFFUSION_SAMPLES}"
echo "EXP_NAME=${EXP_NAME}"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/run_elix.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
    --config-name run_elix_two_stage \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${VALSET_NAME}_twostage_exp7_pocket_no_ens \
    debug=false \
    seed=0 \
    num_workers=${CPUS_PER_TASK} \
    protein_only=false \
    run_seq_des=true \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    model_cfg.model_name=elix_mpnn \
    model_cfg.denoiser_train_dir=${SCAFFOLD_DENOISER_TRAIN_DIR} \
    model_cfg.ckpt_cfg.start_step=${SCAFFOLD_STEP} \
    model_cfg.ckpt_cfg.end_step=${SCAFFOLD_STEP} \
    model_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    model_cfg.ckpt_cfg.eval_last_ckpt=false \
    model_cfg.ckpt_cfg.use_ema=true \
    sampling_cfg.base_cfg_path="${BASE_DIR}/allatom_design/configs/seq_des/elix_mpnn_inference.yaml" \
    sampling_cfg.overrides.batch_size=${BATCH_SIZE} \
    sampling_cfg.overrides.num_seqs_per_pdb=${NUM_SCAFFOLD_SEQS_PER_PDB} \
    sampling_cfg.overrides.use_potts_sampling=true \
    sampling_cfg.overrides.potts_sampling_cfg.potts_sweeps=500 \
    sampling_cfg.overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=true \
    sampling_cfg.overrides.potts_sampling_cfg.rejection_step=false \
    sampling_cfg.guidance.enabled=false \
    two_stage.direction=${DIRECTION} \
    two_stage.pocket_distance=${POCKET_DISTANCE} \
    two_stage.pocket_annotation_method=${POCKET_ANNOTATION_METHOD} \
    two_stage.use_calpha_for_pocket_annotation=true \
    two_stage.pocket.label=${POCKET_MODEL_NAME}_pocket_no_ens \
    two_stage.pocket.model_cfg.model_name=elix_mpnn \
    two_stage.pocket.model_cfg.denoiser_train_dir=${POCKET_DENOISER_TRAIN_DIR} \
    two_stage.pocket.model_cfg.ckpt_cfg.start_step=${POCKET_STEP} \
    two_stage.pocket.model_cfg.ckpt_cfg.end_step=${POCKET_STEP} \
    two_stage.pocket.model_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    two_stage.pocket.model_cfg.ckpt_cfg.eval_last_ckpt=false \
    two_stage.pocket.model_cfg.ckpt_cfg.use_ema=true \
    ++two_stage.pocket.sampling_cfg.overrides.batch_size=${BATCH_SIZE} \
    ++two_stage.pocket.sampling_cfg.overrides.num_seqs_per_pdb=${NUM_POCKET_SEQS_PER_PDB} \
    ++two_stage.pocket.sampling_cfg.overrides.use_potts_sampling=true \
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.potts_sweeps=500 \
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=true \
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.rejection_step=false \
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.potts_only_cond=false \
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.enabled=false \
    two_stage.scaffold.label=${SCAFFOLD_MODEL_NAME}_scaffold_ens_p${SCAFFOLD_PROTEIN_NOISE_LABEL} \
    two_stage.scaffold.model_cfg.model_name=elix_mpnn \
    two_stage.scaffold.model_cfg.denoiser_train_dir=${SCAFFOLD_DENOISER_TRAIN_DIR} \
    two_stage.scaffold.model_cfg.ckpt_cfg.start_step=${SCAFFOLD_STEP} \
    two_stage.scaffold.model_cfg.ckpt_cfg.end_step=${SCAFFOLD_STEP} \
    two_stage.scaffold.model_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    two_stage.scaffold.model_cfg.ckpt_cfg.eval_last_ckpt=false \
    two_stage.scaffold.model_cfg.ckpt_cfg.use_ema=true \
    ++two_stage.scaffold.sampling_cfg.overrides.batch_size=${BATCH_SIZE} \
    ++two_stage.scaffold.sampling_cfg.overrides.num_seqs_per_pdb=${NUM_SCAFFOLD_SEQS_PER_PDB} \
    ++two_stage.scaffold.sampling_cfg.overrides.use_potts_sampling=true \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.potts_sweeps=500 \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=true \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.rejection_step=false \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.potts_only_cond=true \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.enabled=true \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.total_members=${SCAFFOLD_TOTAL_MEMBERS} \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.weights.scheme=mean \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_seed=null \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.mode=gaussian_noise \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.noise_std=${SCAFFOLD_PROTEIN_NOISE_STD} \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.mode=gaussian_noise \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.noise_std=${SCAFFOLD_METAL_NOISE_STD} \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.mode=gaussian_noise \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.noise_std=${SCAFFOLD_SMALL_MOLECULE_NOISE_STD} \
    sampling_inputs_csv=null \
    pos_constraint_csv=null \
    pocket_cfg.pocket_distance_for_docking_metrics=${POCKET_DISTANCE_FOR_DOCKING_METRICS} \
    "pocket_cfg.pocket_distances_for_seq_recovery=${POCKET_DISTANCES_FOR_SEQ_RECOVERY}" \
    "pocket_cfg.pocket_distance_bins=${POCKET_DISTANCE_BINS}" \
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
    "pdb_cfg.sample_lengths=${SAMPLING_LENGTHS}" \
    pdb_cfg.array_id=${CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_CHUNKS} \
    cif_cfg.save.include_bonds=false \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.runner_path="${BASE_DIR}/alphafold3/src/alphafold3/run_alphafold.py" \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=${AF3_RECYCLES} \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=${AF3_DIFFUSION_SAMPLES} \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete: pocket_model=${POCKET_MODEL_NAME}, pocket_step=${POCKET_STEP}, scaffold_model=${SCAFFOLD_MODEL_NAME}, scaffold_step=${SCAFFOLD_STEP}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}"
