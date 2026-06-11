#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/structure_prediction/run_tc_af3_nativeval_sm_metal_nativeseq.sbatch
#SBATCH --job-name=tc_af3_native_native
#SBATCH -p possu,bioe
#SBATCH --array=0-0
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/tc_af3_nativeval_nativeseq_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/tc_af3_nativeval_nativeseq_%A_%a.err

set -euo pipefail

RECYCLE_FILTER="${RECYCLE_FILTER:-all}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Default run is native sequence with num_recycles=1 only.
# For a future recycle sweep, expand this list and update the SBATCH array
# range to NUM_RECYCLES_VALUES * NUM_CHUNKS - 1.
RECYCLES=(1)

print_valid_filters() {
  echo "Valid RECYCLE_FILTER values: all ${RECYCLES[*]}" >&2
}

SELECTED_RECYCLES=()
for CANDIDATE_RECYCLES in "${RECYCLES[@]}"; do
  if [[ "${RECYCLE_FILTER}" != "all" && "${RECYCLE_FILTER}" != "${CANDIDATE_RECYCLES}" ]]; then
    continue
  fi
  SELECTED_RECYCLES+=("${CANDIDATE_RECYCLES}")
done

NUM_RECYCLES_VALUES=${#SELECTED_RECYCLES[@]}
if [[ "${NUM_RECYCLES_VALUES}" -eq 0 ]]; then
  echo "No recycle values selected by RECYCLE_FILTER=${RECYCLE_FILTER}." >&2
  print_valid_filters
  exit 2
fi

NUM_CHUNKS=1
NUM_TASKS=$((NUM_RECYCLES_VALUES * NUM_CHUNKS))

if [[ "${TASK_ID}" -ge "${NUM_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} is outside selected tasks=${NUM_TASKS}; exiting without work."
  exit 0
fi

RECYCLE_IDX=$((TASK_ID / NUM_CHUNKS))
CHUNK_IDX=$((TASK_ID % NUM_CHUNKS))
NUM_RECYCLES=${SELECTED_RECYCLES[$RECYCLE_IDX]}

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/tc_af3_nativeval_sm_metal_nativeseq
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=nativeval_sm_metal
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/nativeval_cifs/cifs
SAMPLING_INPUTS_CSV=/scratch/users/zhkim216/datasets/val_cifs/nativeval_cifs/sampling_inputs_sm_metal.csv
PDB_NAME_LIST=/scratch/users/zhkim216/datasets/val_cifs/nativeval_cifs/nativeval_sm_metal.txt

AF3_DIFFUSION_SAMPLES=5
AF3_MAX_TEMPLATES=1
AF3_TEMPLATE_CONDITIONING_MODE=1
AF3_MAX_TEMPLATE_DATE=2026-06-10
POCKET_DISTANCE_FOR_DOCKING_METRICS=5.0

EXP_NAME=tc_af3_${VALSET_NAME}_nativeseq_recycles${NUM_RECYCLES}
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

[[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
[[ -f "${PDB_NAME_LIST}" ]] || { echo "Missing PDB_NAME_LIST=${PDB_NAME_LIST}" >&2; exit 1; }
[[ -f "${SAMPLING_INPUTS_CSV}" ]] || { echo "Missing SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}" >&2; exit 1; }

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "${COMPUTE_CAP}" | cut -d. -f1)

if [[ "${MAJOR}" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi

echo "Starting TC AF3 nativeval sm-metal native-sequence run: num_recycles=${NUM_RECYCLES}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}, task=${TASK_ID}/${NUM_TASKS}"
echo "RECYCLE_FILTER=${RECYCLE_FILTER}"
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"
echo "PDB_DIR=${PDB_DIR}"
echo "PDB_NAME_LIST=${PDB_NAME_LIST}"
echo "SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}"
echo "AF3 TC: recycles=${NUM_RECYCLES}, diffusion_samples=${AF3_DIFFUSION_SAMPLES}, max_templates=${AF3_MAX_TEMPLATES}, ligand_protein_template_conditioning_mode=${AF3_TEMPLATE_CONDITIONING_MODE}"
echo "EXP_NAME=${EXP_NAME}"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/structure_prediction/run_tc_af3.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/structure_prediction" \
    --config-name run_tc_eval_af3 \
    wandb.no_wandb=true \
    debug=false \
    seed=0 \
    input_sample_is_designed=false \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.array_id=${CHUNK_IDX} \
    pdb_cfg.num_arrays=${NUM_CHUNKS} \
    pocket_cfg.pocket_distance_for_docking_metrics=${POCKET_DISTANCE_FOR_DOCKING_METRICS} \
    struct_pred_cfg.evaluate_docking_consistency=true \
    struct_pred_cfg.calculate_metrics_only=false \
    struct_pred_cfg.af3.runner_path="${BASE_DIR}/alphafold3/src/alphafold3/run_alphafold.py" \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.tc.num_recycles=${NUM_RECYCLES} \
    struct_pred_cfg.af3.inference_config.tc.num_diffusion_samples=${AF3_DIFFUSION_SAMPLES} \
    struct_pred_cfg.af3.inference_config.tc.max_templates=${AF3_MAX_TEMPLATES} \
    struct_pred_cfg.af3.inference_config.tc.ligand_protein_template_conditioning_mode=${AF3_TEMPLATE_CONDITIONING_MODE} \
    struct_pred_cfg.af3.inference_config.tc.mask_template_sidechains=true \
    struct_pred_cfg.af3.inference_config.tc.mask_template_sequence=true \
    struct_pred_cfg.af3.inference_config.tc.max_template_date=${AF3_MAX_TEMPLATE_DATE} \
    hydra.run.dir=${HYDRA_DIR}

echo "TC AF3 native-sequence run complete: num_recycles=${NUM_RECYCLES}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}"
