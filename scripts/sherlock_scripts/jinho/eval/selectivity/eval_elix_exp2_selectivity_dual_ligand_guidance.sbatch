#!/bin/bash
# Submit from Sherlock:
#   bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/selectivity/eval_elix_exp2_selectivity_dual_ligand_guidance.sbatch
#
# Build the local dataset first, then sync it to SELECTIVITY_DATASET_DIR:
#   python scripts/local_scripts/jinho/preprocessing/stage_selectivity_dual_ligand_tc_inputs.py --overwrite
#
# This job is sharded by target ligand side, guidance scale, and input array.
# Each task writes one target/scale-specific output folder.
#SBATCH --job-name=eval_elix_selectivity_dual_guidance
#SBATCH -p possu,bioe
#SBATCH --array=0-19
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/eval_elix_selectivity_dual_guidance_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/eval_elix_selectivity_dual_guidance_%A_%a.err

set -euo pipefail

# Experiment-defining values.
BASE_DIR=/home/users/zhkim216/code/elix
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/eval_elix_selectivity_dual_ligand_guidance
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=dual_ligand_tc_sel_sm_nhbad2
RUN_LABEL=selectivity_sm_nhbad2
SELECTIVITY_DATASET_DIR=/scratch/users/zhkim216/datasets/val_cifs/selectivity_val_cifs/dual_ligand_tc_sel_sm_nhbad2
MODEL_NAME=elix_exp2_cfg0
STEP=102500
NUM_ARRAYS=2
GUIDANCE_TARGET_LIGAND_SIDES=(1 2)
GUIDANCE_SCALES=(0.0 0.5 1.0 1.5 2.0)
NUM_SEQS_PER_PDB=2
NO_WANDB=false

TASK_ID=${SLURM_ARRAY_TASK_ID}
CPUS_PER_TASK=8
NUM_GUIDANCE_TARGETS=${#GUIDANCE_TARGET_LIGAND_SIDES[@]}
NUM_GUIDANCE_SCALES=${#GUIDANCE_SCALES[@]}
NUM_TASKS=$((NUM_GUIDANCE_TARGETS * NUM_GUIDANCE_SCALES * NUM_ARRAYS))
if [[ "${TASK_ID}" -ge "${NUM_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} outside NUM_TASKS=${NUM_TASKS}; exiting without work."
  exit 0
fi
TARGET_BLOCK_SIZE=$((NUM_GUIDANCE_SCALES * NUM_ARRAYS))
GUIDANCE_TARGET_IDX=$((TASK_ID / TARGET_BLOCK_SIZE))
WITHIN_TARGET_TASK_ID=$((TASK_ID % TARGET_BLOCK_SIZE))
GUIDANCE_SCALE_IDX=$((WITHIN_TARGET_TASK_ID / NUM_ARRAYS))
ARRAY_ID=$((WITHIN_TARGET_TASK_ID % NUM_ARRAYS))
GUIDANCE_TARGET_LIGAND_SIDE=${GUIDANCE_TARGET_LIGAND_SIDES[$GUIDANCE_TARGET_IDX]}
GUIDANCE_TARGET_LABEL=target${GUIDANCE_TARGET_LIGAND_SIDE}
GUIDANCE_SCALE=${GUIDANCE_SCALES[$GUIDANCE_SCALE_IDX]}
GUIDANCE_SCALE_LABEL=${GUIDANCE_SCALE//./p}
INPUT_PDB_DIR=${SELECTIVITY_DATASET_DIR}/cifs
SAMPLING_INPUTS_CSV=${SELECTIVITY_DATASET_DIR}/sampling_inputs_dual_ligand_tc_sel_sm_nhbad2.csv
INPUT_PDB_NAME_LIST=${SELECTIVITY_DATASET_DIR}/dual_ligand_tc_sel_sm_nhbad2.txt
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}
EXP_NAME=${MODEL_NAME}_${RUN_LABEL}_${GUIDANCE_TARGET_LABEL}_guidance${GUIDANCE_SCALE_LABEL}_numseqs${NUM_SEQS_PER_PDB}
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}/array_${ARRAY_ID}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_ROOT}" "${BASE_OUT_DIR}"

[[ -d "${INPUT_PDB_DIR}" ]] || { echo "Missing INPUT_PDB_DIR=${INPUT_PDB_DIR}" >&2; exit 1; }
[[ -f "${INPUT_PDB_NAME_LIST}" ]] || { echo "Missing INPUT_PDB_NAME_LIST=${INPUT_PDB_NAME_LIST}" >&2; exit 1; }
[[ -f "${SAMPLING_INPUTS_CSV}" ]] || { echo "Missing SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}" >&2; exit 1; }
[[ -d "${DENOISER_TRAIN_DIR}" ]] || { echo "Missing DENOISER_TRAIN_DIR=${DENOISER_TRAIN_DIR}" >&2; exit 1; }

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"
WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des

echo "Starting dual-ligand selectivity guidance."
echo "MODEL_NAME=${MODEL_NAME}"
echo "STEP=${STEP}"
echo "TASK_ID=${TASK_ID}/${NUM_TASKS}"
echo "GUIDANCE_TARGET_LIGAND_SIDE=${GUIDANCE_TARGET_LIGAND_SIDE}"
echo "GUIDANCE_SCALE=${GUIDANCE_SCALE}"
echo "ARRAY_ID=${ARRAY_ID}/${NUM_ARRAYS}"
echo "EXP_NAME=${EXP_NAME}"
echo "SELECTIVITY_DATASET_DIR=${SELECTIVITY_DATASET_DIR}"
echo "INPUT_PDB_NAME_LIST=${INPUT_PDB_NAME_LIST}"
echo "SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/run_elix.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
    --config-name guidance/elix_selectivity_guidance \
    wandb.no_wandb=${NO_WANDB} \
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
    sampling_cfg.overrides.num_seqs_per_pdb=${NUM_SEQS_PER_PDB} \
    sampling_cfg.guidance.target_ligand_side=${GUIDANCE_TARGET_LIGAND_SIDE} \
    "sampling_cfg.guidance.guidance_scale_list=[${GUIDANCE_SCALE}]" \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=null \
    input_sample_is_all_atom=true \
    input_sample_is_designed=false \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${INPUT_PDB_DIR} \
    pdb_cfg.pdb_name_list=${INPUT_PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.sample_indices=null \
    pdb_cfg.sample_lengths=null \
    pdb_cfg.array_id=${ARRAY_ID} \
    pdb_cfg.num_arrays=${NUM_ARRAYS} \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    hydra.run.dir=${HYDRA_DIR}

echo "Dual-ligand selectivity guidance complete: EXP_NAME=${EXP_NAME}, GUIDANCE_TARGET_LIGAND_SIDE=${GUIDANCE_TARGET_LIGAND_SIDE}, GUIDANCE_SCALE=${GUIDANCE_SCALE}, ARRAY_ID=${ARRAY_ID}/${NUM_ARRAYS}"
