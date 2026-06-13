#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/denovoval/eval_elix_exp6_denovoval_scaffold_sm_metal_ensemble_conditioning.sbatch
#SBATCH --job-name=eval_exp6_denovo_ens
#SBATCH -p possu,bioe
#SBATCH --array=0-17
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/eval_exp6_denovo_ens_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/eval_exp6_denovo_ens_%A_%a.err

set -euo pipefail

MODEL_FILTER="${MODEL_FILTER:-all}"
STEP_FILTER="${STEP_FILTER:-all}"
PRINT_ARRAY_PLAN="${PRINT_ARRAY_PLAN:-false}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

# Mirrors scripts/sherlock_scripts/jinho/train/elix_exp6.sbatch.
MODEL_NAMES=(elix_exp6_cfg0 elix_exp6_cfg1 elix_exp6_cfg2)
STEPS=(2500 12500 22500 32500 42500 52500)
NUM_ENSEMBLES=16
PROTEIN_NOISE_STD=0.5
NUM_CHUNKS=1

matches_filter() {
  local value="$1"
  local filter="$2"
  [[ "${filter}" == "all" || "${filter}" == "${value}" ]]
}

print_valid_filters() {
  echo "Valid MODEL_FILTER values: all ${MODEL_NAMES[*]}" >&2
  echo "Valid STEP_FILTER values: all ${STEPS[*]}" >&2
}

MODEL_STEP_COMBOS=()
for CANDIDATE_MODEL in "${MODEL_NAMES[@]}"; do
  matches_filter "${CANDIDATE_MODEL}" "${MODEL_FILTER}" || continue
  for CANDIDATE_STEP in "${STEPS[@]}"; do
    matches_filter "${CANDIDATE_STEP}" "${STEP_FILTER}" || continue
    MODEL_STEP_COMBOS+=("${CANDIDATE_MODEL}:${CANDIDATE_STEP}")
  done
done

NUM_COMBOS=${#MODEL_STEP_COMBOS[@]}
if [[ "${NUM_COMBOS}" -eq 0 ]]; then
  echo "No model/step combinations selected by MODEL_FILTER=${MODEL_FILTER}, STEP_FILTER=${STEP_FILTER}." >&2
  print_valid_filters
  exit 2
fi

NUM_TASKS=$((NUM_COMBOS * NUM_CHUNKS))

if [[ "${PRINT_ARRAY_PLAN}" == "true" ]]; then
  echo "Selected denovoval scaffold ensemble-conditioning sweep: NUM_COMBOS=${NUM_COMBOS}, NUM_TASKS=${NUM_TASKS}, num_ensembles=${NUM_ENSEMBLES}, protein_noise_std=${PROTEIN_NOISE_STD}"
  for COMBO_IDX in "${!MODEL_STEP_COMBOS[@]}"; do
    IFS=: read -r PLAN_MODEL PLAN_STEP <<< "${MODEL_STEP_COMBOS[$COMBO_IDX]}"
    echo "combo_idx=${COMBO_IDX} model=${PLAN_MODEL} step=${PLAN_STEP} num_ensembles=${NUM_ENSEMBLES} protein_noise_std=${PROTEIN_NOISE_STD}"
  done
  exit 0
fi

if [[ "${TASK_ID}" -ge "${NUM_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} is outside selected tasks=${NUM_TASKS}; exiting without work."
  exit 0
fi

COMBO_IDX=$((TASK_ID / NUM_CHUNKS))
CHUNK_IDX=$((TASK_ID % NUM_CHUNKS))
IFS=: read -r MODEL_NAME STEP <<< "${MODEL_STEP_COMBOS[$COMBO_IDX]}"
NOISE_LABEL=${PROTEIN_NOISE_STD//./p}

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/eval_elix_exp6_denovoval_scaffold_sm_metal_ensemble_conditioning
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=denovoval_scaffold_sm_metal
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs/denovoval_scaffold/cifs
PDB_NAME_LIST=/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs/denovoval_scaffold/denovoval_scaffold_sm_metal.txt

BATCH_SIZE=8
NUM_SEQS_PER_PDB=1
SAMPLING_LENGTHS="[150,300]"
POCKET_DISTANCE_FOR_DOCKING_METRICS=8.0
POCKET_DISTANCES_FOR_SEQ_RECOVERY="[4.0,6.0,8.0,10.0]"
POCKET_DISTANCE_BINS="[[0.0,4.0],[4.0,6.0],[6.0,8.0],[8.0,10.0]]"
AF3_RECYCLES=10
AF3_DIFFUSION_SAMPLES=5

DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}
EXP_NAME=${MODEL_NAME}_${VALSET_NAME}_ensemble_n${NUM_ENSEMBLES}_protein_noise${NOISE_LABEL}_numseqs${NUM_SEQS_PER_PDB}
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}/chunk_${CHUNK_IDX}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_ROOT}" "${BASE_OUT_DIR}"

[[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
[[ -f "${PDB_NAME_LIST}" ]] || { echo "Missing PDB_NAME_LIST=${PDB_NAME_LIST}" >&2; exit 1; }
[[ -d "${DENOISER_TRAIN_DIR}" ]] || { echo "Missing DENOISER_TRAIN_DIR=${DENOISER_TRAIN_DIR}" >&2; exit 1; }

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

echo "Starting denovoval scaffold sm-metal ensemble-conditioning eval: model=${MODEL_NAME}, step=${STEP}, num_ensembles=${NUM_ENSEMBLES}, protein_noise_std=${PROTEIN_NOISE_STD}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}, selected_task=${TASK_ID}/${NUM_TASKS}"
echo "Filters: MODEL_FILTER=${MODEL_FILTER}, STEP_FILTER=${STEP_FILTER}"
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"
echo "PDB_DIR=${PDB_DIR}"
echo "PDB_NAME_LIST=${PDB_NAME_LIST}"
echo "SAMPLING_LENGTHS=${SAMPLING_LENGTHS}"
echo "POCKET_DISTANCE_BINS=${POCKET_DISTANCE_BINS}"
echo "EXP_NAME=${EXP_NAME}"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/run_elix.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
    --config-name run_elix \
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
    model_cfg.model_name=elix_mpnn \
    model_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
    model_cfg.ckpt_cfg.start_step=${STEP} \
    model_cfg.ckpt_cfg.end_step=${STEP} \
    model_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    model_cfg.ckpt_cfg.eval_last_ckpt=false \
    model_cfg.ckpt_cfg.use_ema=true \
    sampling_cfg.base_cfg_path="${BASE_DIR}/allatom_design/configs/seq_des/elix_mpnn_inference.yaml" \
    sampling_cfg.overrides.batch_size=${BATCH_SIZE} \
    sampling_cfg.overrides.num_seqs_per_pdb=${NUM_SEQS_PER_PDB} \
    sampling_cfg.overrides.use_potts_sampling=true \
    sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.enabled=true \
    sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.num_ensembles=${NUM_ENSEMBLES} \
    sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.reduce=mean \
    sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_seed=null \
    sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_std.protein=${PROTEIN_NOISE_STD} \
    sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_std.metal=0.0 \
    sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_std.nonpolymer=0.0 \
    sampling_cfg.guidance.enabled=false \
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

echo "Evaluation complete: model=${MODEL_NAME}, step=${STEP}, num_ensembles=${NUM_ENSEMBLES}, protein_noise_std=${PROTEIN_NOISE_STD}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}"
