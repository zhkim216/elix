#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/sequence_recovery/eval_elix_exp5_nativeval_sm_metal_sequence_recovery_ensemble_conditioning.sbatch
#SBATCH --job-name=seqrec_exp5_ens
#SBATCH -p possu,bioe
#SBATCH --array=0-167
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/seqrec_exp5_ens_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/seqrec_exp5_ens_%A_%a.err

set -euo pipefail

MODEL_FILTER="${MODEL_FILTER:-all}"
ENSEMBLE_MODE_FILTER="${ENSEMBLE_MODE_FILTER:-all}"
NUM_ENSEMBLES_FILTER="${NUM_ENSEMBLES_FILTER:-all}"
PROTEIN_NOISE_STD_FILTER="${PROTEIN_NOISE_STD_FILTER:-all}"
PRINT_ARRAY_PLAN="${PRINT_ARRAY_PLAN:-false}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

# Mirrors scripts/sherlock_scripts/jinho/train/elix_exp5.sbatch EXP_NAME values.
MODEL_NAMES=(
  elix_exp5_cfg0_node_concat_base
  elix_exp5_cfg1_node_concat_edge_update
  elix_exp5_cfg2_node_concat_decoder_skip
  elix_exp5_cfg3_node_concat_edge_update_decoder_skip
  elix_exp5_cfg4_context_skip_potts_only
  elix_exp5_cfg5_context_skip_edge_update
  elix_exp5_cfg6_context_skip_decoder_skip
  elix_exp5_cfg7_context_skip_edge_update_decoder_skip
)

START_STEP=2500
END_STEP=122500
EVAL_EVERY_N_CKPTS=4
CHECKPOINT_STEP_INTERVAL=2500
NUM_ENSEMBLES_LIST=(2 4 8 16)
PROTEIN_NOISE_STD_LIST=(0.1 0.2 0.3 0.4 0.5)
NUM_CHUNKS=1

matches_filter() {
  local value="$1"
  local filter="$2"
  [[ "${filter}" == "all" || "${filter}" == "${value}" ]]
}

print_valid_filters() {
  echo "Valid MODEL_FILTER values: all ${MODEL_NAMES[*]}" >&2
  echo "Valid ENSEMBLE_MODE_FILTER values: all baseline ensemble" >&2
  echo "Valid NUM_ENSEMBLES_FILTER values: all ${NUM_ENSEMBLES_LIST[*]}" >&2
  echo "Valid PROTEIN_NOISE_STD_FILTER values: all ${PROTEIN_NOISE_STD_LIST[*]}" >&2
}

expected_eval_steps() {
  local step
  local stride=$((CHECKPOINT_STEP_INTERVAL * EVAL_EVERY_N_CKPTS))
  local steps=()
  for ((step = START_STEP; step <= END_STEP; step += stride)); do
    steps+=("${step}")
  done
  echo "${steps[*]}"
}

SWEEP_COMBOS=()
for CANDIDATE_MODEL in "${MODEL_NAMES[@]}"; do
  matches_filter "${CANDIDATE_MODEL}" "${MODEL_FILTER}" || continue

  if matches_filter "baseline" "${ENSEMBLE_MODE_FILTER}" \
    && [[ "${NUM_ENSEMBLES_FILTER}" == "all" ]] \
    && [[ "${PROTEIN_NOISE_STD_FILTER}" == "all" ]]; then
    SWEEP_COMBOS+=("${CANDIDATE_MODEL}:baseline:1:0.0")
  fi

  if matches_filter "ensemble" "${ENSEMBLE_MODE_FILTER}"; then
    for CANDIDATE_NUM_ENSEMBLES in "${NUM_ENSEMBLES_LIST[@]}"; do
      matches_filter "${CANDIDATE_NUM_ENSEMBLES}" "${NUM_ENSEMBLES_FILTER}" || continue
      for CANDIDATE_PROTEIN_NOISE_STD in "${PROTEIN_NOISE_STD_LIST[@]}"; do
        matches_filter "${CANDIDATE_PROTEIN_NOISE_STD}" "${PROTEIN_NOISE_STD_FILTER}" || continue
        SWEEP_COMBOS+=("${CANDIDATE_MODEL}:ensemble:${CANDIDATE_NUM_ENSEMBLES}:${CANDIDATE_PROTEIN_NOISE_STD}")
      done
    done
  fi
done

NUM_COMBOS=${#SWEEP_COMBOS[@]}
if [[ "${NUM_COMBOS}" -eq 0 ]]; then
  echo "No sweep combinations selected by MODEL_FILTER=${MODEL_FILTER}, ENSEMBLE_MODE_FILTER=${ENSEMBLE_MODE_FILTER}, NUM_ENSEMBLES_FILTER=${NUM_ENSEMBLES_FILTER}, PROTEIN_NOISE_STD_FILTER=${PROTEIN_NOISE_STD_FILTER}." >&2
  print_valid_filters
  exit 2
fi

NUM_TASKS=$((NUM_COMBOS * NUM_CHUNKS))

if [[ "${PRINT_ARRAY_PLAN}" == "true" ]]; then
  echo "Selected exp5 nativeval sm-metal sequence-recovery ensemble/no-ensemble sweep: NUM_COMBOS=${NUM_COMBOS}, NUM_TASKS=${NUM_TASKS}"
  echo "Checkpoint selector: start_step=${START_STEP}, end_step=${END_STEP}, eval_every_n_ckpts=${EVAL_EVERY_N_CKPTS}"
  echo "Expected eval steps for ${CHECKPOINT_STEP_INTERVAL}-step checkpoint cadence: $(expected_eval_steps)"
  for COMBO_IDX in "${!SWEEP_COMBOS[@]}"; do
    IFS=: read -r PLAN_MODEL PLAN_ENSEMBLE_MODE PLAN_NUM_ENSEMBLES PLAN_PROTEIN_NOISE_STD <<< "${SWEEP_COMBOS[$COMBO_IDX]}"
    echo "combo_idx=${COMBO_IDX} model=${PLAN_MODEL} ensemble_mode=${PLAN_ENSEMBLE_MODE} num_ensembles=${PLAN_NUM_ENSEMBLES} protein_noise_std=${PLAN_PROTEIN_NOISE_STD}"
  done
  exit 0
fi

if [[ "${TASK_ID}" -ge "${NUM_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} is outside selected tasks=${NUM_TASKS}; exiting without work."
  exit 0
fi

COMBO_IDX=$((TASK_ID / NUM_CHUNKS))
CHUNK_IDX=$((TASK_ID % NUM_CHUNKS))
IFS=: read -r MODEL_NAME ENSEMBLE_MODE NUM_ENSEMBLES PROTEIN_NOISE_STD <<< "${SWEEP_COMBOS[$COMBO_IDX]}"
NOISE_LABEL=${PROTEIN_NOISE_STD//./p}

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/eval_elix_exp5_nativeval_sm_metal_sequence_recovery_ensemble_conditioning
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=nativeval_sm_metal
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/nativeval_cifs/cifs
SAMPLING_INPUTS_CSV=/scratch/users/zhkim216/datasets/val_cifs/nativeval_cifs/sampling_inputs_sm_metal.csv
PDB_NAME_LIST=/scratch/users/zhkim216/datasets/val_cifs/nativeval_cifs/nativeval_sm_metal.txt

BATCH_SIZE=4
NUM_SEQS_PER_PDB=1
POCKET_DISTANCE_FOR_DOCKING_METRICS=8.0
POCKET_DISTANCES_FOR_SEQ_RECOVERY="[4.0,6.0,8.0,10.0]"
POCKET_DISTANCE_BINS="[[0.0,4.0],[4.0,6.0],[6.0,8.0],[8.0,10.0]]"

DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}
if [[ "${ENSEMBLE_MODE}" == "baseline" ]]; then
  EXP_NAME=${MODEL_NAME}_steps${START_STEP}to${END_STEP}_every${EVAL_EVERY_N_CKPTS}ckpts_${VALSET_NAME}_seqrec_no_ensemble_numseqs${NUM_SEQS_PER_PDB}
  ENSEMBLE_OVERRIDES=(
    "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.enabled=false"
  )
else
  EXP_NAME=${MODEL_NAME}_steps${START_STEP}to${END_STEP}_every${EVAL_EVERY_N_CKPTS}ckpts_${VALSET_NAME}_seqrec_ensemble_n${NUM_ENSEMBLES}_protein_noise${NOISE_LABEL}_numseqs${NUM_SEQS_PER_PDB}
  ENSEMBLE_OVERRIDES=(
    "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.enabled=true"
    "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.num_ensembles=${NUM_ENSEMBLES}"
    "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.reduce=mean"
    "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_seed=null"
    "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_std.protein=${PROTEIN_NOISE_STD}"
    "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_std.metal=0.0"
    "sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_std.nonpolymer=0.0"
  )
fi
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}/chunk_${CHUNK_IDX}

checkpoint_exists_for_step() {
  local step="$1"
  compgen -G "${DENOISER_TRAIN_DIR}/checkpoints/ema/sd-step${step}-epoch*-ema*.ckpt" >/dev/null || \
    compgen -G "${DENOISER_TRAIN_DIR}/checkpoints/sd-step${step}-epoch*.ckpt" >/dev/null
}

validate_checkpoint_cadence() {
  local step
  for ((step = START_STEP; step <= END_STEP; step += CHECKPOINT_STEP_INTERVAL)); do
    checkpoint_exists_for_step "${step}" || {
      echo "Missing checkpoint for MODEL_NAME=${MODEL_NAME}, STEP=${step} under ${DENOISER_TRAIN_DIR}/checkpoints or checkpoints/ema" >&2
      echo "eval_every_n_ckpts=${EVAL_EVERY_N_CKPTS} is index-based over sorted checkpoints, so a complete ${CHECKPOINT_STEP_INTERVAL}-step checkpoint cadence is required." >&2
      return 1
    }
  done
}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_ROOT}" "${BASE_OUT_DIR}"

[[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
[[ -f "${PDB_NAME_LIST}" ]] || { echo "Missing PDB_NAME_LIST=${PDB_NAME_LIST}" >&2; exit 1; }
[[ -f "${SAMPLING_INPUTS_CSV}" ]] || { echo "Missing SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}" >&2; exit 1; }
[[ -d "${DENOISER_TRAIN_DIR}" ]] || { echo "Missing DENOISER_TRAIN_DIR=${DENOISER_TRAIN_DIR}" >&2; exit 1; }
validate_checkpoint_cadence

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"
WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des

echo "Starting exp5 nativeval sm-metal sequence-recovery eval: model=${MODEL_NAME}, ensemble_mode=${ENSEMBLE_MODE}, start_step=${START_STEP}, end_step=${END_STEP}, eval_every_n_ckpts=${EVAL_EVERY_N_CKPTS}, expected_steps=$(expected_eval_steps), num_ensembles=${NUM_ENSEMBLES}, protein_noise_std=${PROTEIN_NOISE_STD}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}, selected_task=${TASK_ID}/${NUM_TASKS}"
echo "Filters: MODEL_FILTER=${MODEL_FILTER}, ENSEMBLE_MODE_FILTER=${ENSEMBLE_MODE_FILTER}, NUM_ENSEMBLES_FILTER=${NUM_ENSEMBLES_FILTER}, PROTEIN_NOISE_STD_FILTER=${PROTEIN_NOISE_STD_FILTER}"
echo "AF3 disabled: evaluate_self_consistency=false, evaluate_docking_consistency=false"
echo "PDB_DIR=${PDB_DIR}"
echo "PDB_NAME_LIST=${PDB_NAME_LIST}"
echo "SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}"
echo "POCKET_DISTANCE_BINS=${POCKET_DISTANCE_BINS}"
echo "EXP_NAME=${EXP_NAME}"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/run_elix.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
    --config-name run_elix \
    wandb.no_wandb=false \
    wandb.project=${PROJECT} \
    wandb.wandb_id=${WANDB_ID} \
    wandb.group=${VALSET_NAME}_seqrec_ens \
    debug=false \
    seed=0 \
    num_workers=${CPUS_PER_TASK} \
    protein_only=false \
    run_seq_des=true \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    model_cfg.model_name=elix_mpnn \
    model_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    model_cfg.ckpt_cfg.start_step=${START_STEP} \
    model_cfg.ckpt_cfg.end_step=${END_STEP} \
    model_cfg.ckpt_cfg.eval_every_n_ckpts=${EVAL_EVERY_N_CKPTS} \
    model_cfg.ckpt_cfg.eval_last_ckpt=false \
    model_cfg.ckpt_cfg.use_ema=true \
    sampling_cfg.base_cfg_path="${BASE_DIR}/allatom_design/configs/seq_des/elix_mpnn_inference.yaml" \
    sampling_cfg.overrides.batch_size=${BATCH_SIZE} \
    sampling_cfg.overrides.num_seqs_per_pdb=${NUM_SEQS_PER_PDB} \
    sampling_cfg.overrides.use_potts_sampling=true \
    "${ENSEMBLE_OVERRIDES[@]}" \
    sampling_cfg.guidance.enabled=false \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=null \
    pocket_cfg.pocket_distance_for_docking_metrics=${POCKET_DISTANCE_FOR_DOCKING_METRICS} \
    "pocket_cfg.pocket_distances_for_seq_recovery=${POCKET_DISTANCES_FOR_SEQ_RECOVERY}" \
    "pocket_cfg.pocket_distance_bins=${POCKET_DISTANCE_BINS}" \
    pocket_cfg.n_min_ligand_atoms_for_seq_recovery=1 \
    input_sample_is_all_atom=true \
    input_sample_is_designed=false \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.sample_indices=null \
    pdb_cfg.sample_lengths=null \
    pdb_cfg.array_id=${CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_CHUNKS} \
    cif_cfg.save.include_bonds=false \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=false \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.calculate_metrics_only=false \
    hydra.run.dir=${HYDRA_DIR}

echo "Sequence-recovery evaluation complete: model=${MODEL_NAME}, ensemble_mode=${ENSEMBLE_MODE}, start_step=${START_STEP}, end_step=${END_STEP}, eval_every_n_ckpts=${EVAL_EVERY_N_CKPTS}, num_ensembles=${NUM_ENSEMBLES}, protein_noise_std=${PROTEIN_NOISE_STD}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}"
