#!/usr/bin/env bash
#SBATCH --job-name=lc_seq_des
#SBATCH -p possu,bioe
#SBATCH --array=0-19
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/%x_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/%x_%A_%a.err

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
  echo "SLURM_ARRAY_TASK_ID=${TASK_ID} is outside selected model/step tasks=${NUM_TASKS}; exiting without work."
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
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs/eval_proto_metal_nativeval/${MODEL_NAME}/step_${STEP}
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des/${MODEL_NAME}
DATA_DIR=/scratch/users/zhkim216/datasets/val_cifs/nativeval_cifs

PDB_DIR="${PDB_DIR:-${DATA_DIR}/cifs}"
PDB_NAME_LIST="${DATA_DIR}/metal_nativeval.txt"
SAMPLING_INPUTS_CSV="${DATA_DIR}/sampling_inputs_metal_nativeval.csv"
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}

NUM_SEQS_PER_PDB=2
BATCH_SIZE="${BATCH_SIZE:-1}"
VALSET_NAME=metal_nativeval
EXP_NAME=eval_proto_metal_nativeval

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

[[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
[[ -f "${PDB_NAME_LIST}" ]] || { echo "Missing PDB_NAME_LIST=${PDB_NAME_LIST}" >&2; exit 1; }
[[ -f "${SAMPLING_INPUTS_CSV}" ]] || { echo "Missing SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}" >&2; exit 1; }

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
    --config-name lc_seq_des_multi \
    wandb.no_wandb=false \
    wandb.project=eval_ligand_seq_des \
    wandb.wandb_id=zhkim216 \
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
    seq_des_cfg.atom_mpnn.overrides.use_potts_sampling=true \
    guidance_cfg.enabled=false \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=null \
    redesign_cfg.use_native_seq=false \
    redesign_cfg.native_cif_dir=null \
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
    pdb_cfg.array_id=null \
    pdb_cfg.num_arrays=null \
    featurizer_cfg.design.residue_cache_dir="/scratch/users/zhkim216/datasets/atomworks_cached_residue_data" \
    struct_pred_cfg.evaluate_self_consistency=true \
    struct_pred_cfg.evaluate_docking_consistency=false \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.runner_path="${BASE_DIR}/alphafold3/src/alphafold3/run_alphafold.py" \
    struct_pred_cfg.af3.inference_config.ss.num_recycles=10 \
    struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=5 \
    hydra.run.dir=${HYDRA_DIR}

echo "Evaluation complete: model=${MODEL_NAME}, step=${STEP}, exp_name=${EXP_NAME}"
