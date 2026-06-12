#!/bin/bash
# Submit this after all guidance array tasks have completed:
#   bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/selectivity/run_selectivity_dual_ligand_tc_af3.sbatch
#
# TC AF3 consumes one target/scale-specific samples_for_af3_tc directory
# produced by the guidance array job. It is intentionally not sharded with
# pdb_cfg.array_id.
#SBATCH --job-name=selectivity_dual_tc_af3
#SBATCH -p possu,bioe
#SBATCH --array=0-9
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/selectivity_dual_tc_af3_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/selectivity_dual_tc_af3_%A_%a.err

set -euo pipefail

# Experiment-defining values.
BASE_DIR=/home/users/zhkim216/code/elix
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/selectivity_dual_ligand_tc_af3
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=dual_ligand_tc_sel_sm_nhbad2
RUN_LABEL=selectivity_sm_nhbad2
SELECTIVITY_DATASET_DIR=/scratch/users/zhkim216/datasets/val_cifs/selectivity_val_cifs/dual_ligand_tc_sel_sm_nhbad2
MODEL_NAME=elix_exp2_cfg0
STEP=102500
NUM_SEQS_PER_PDB=2
GUIDANCE_TARGET_LIGAND_SIDES=(1 2)
GUIDANCE_SCALES=(0.0 0.5 1.0 1.5 2.0)
POCKET_DISTANCE_FOR_DOCKING_METRICS=5.0
AF3_RECYCLES=1
AF3_DIFFUSION_SAMPLES=5
AF3_MAX_TEMPLATES=1
AF3_TEMPLATE_CONDITIONING_MODE=1
AF3_TEMPLATE_PAIR_SCALE=1.0
AF3_MAX_TEMPLATE_DATE=2026-06-10
TC_CALCULATE_METRICS_ONLY=false
TC_GENERATE_INPUTS_ONLY=false

TASK_ID=${SLURM_ARRAY_TASK_ID}
NUM_GUIDANCE_TARGETS=${#GUIDANCE_TARGET_LIGAND_SIDES[@]}
NUM_GUIDANCE_SCALES=${#GUIDANCE_SCALES[@]}
NUM_TASKS=$((NUM_GUIDANCE_TARGETS * NUM_GUIDANCE_SCALES))
if [[ "${TASK_ID}" -ge "${NUM_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} outside NUM_TASKS=${NUM_TASKS}; exiting without work."
  exit 0
fi
GUIDANCE_TARGET_IDX=$((TASK_ID / NUM_GUIDANCE_SCALES))
GUIDANCE_SCALE_IDX=$((TASK_ID % NUM_GUIDANCE_SCALES))
GUIDANCE_TARGET_LIGAND_SIDE=${GUIDANCE_TARGET_LIGAND_SIDES[$GUIDANCE_TARGET_IDX]}
GUIDANCE_TARGET_LABEL=target${GUIDANCE_TARGET_LIGAND_SIDE}
GUIDANCE_SCALE=${GUIDANCE_SCALES[$GUIDANCE_SCALE_IDX]}
GUIDANCE_SCALE_LABEL=${GUIDANCE_SCALE//./p}
SAMPLING_INPUTS_CSV=${SELECTIVITY_DATASET_DIR}/sampling_inputs_dual_ligand_tc_sel_sm_nhbad2.csv
GUIDANCE_EXP_NAME=${MODEL_NAME}_${RUN_LABEL}_${GUIDANCE_TARGET_LABEL}_guidance${GUIDANCE_SCALE_LABEL}_numseqs${NUM_SEQS_PER_PDB}
GUIDANCE_OUT_DIR=${BASE_OUT_DIR}/${GUIDANCE_EXP_NAME}
EXP_NAME=selectivity_dual_ligand_tc_af3_from_${GUIDANCE_EXP_NAME}_recycles${AF3_RECYCLES}_diff${AF3_DIFFUSION_SAMPLES}
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_ROOT}" "${BASE_OUT_DIR}"

[[ -f "${SAMPLING_INPUTS_CSV}" ]] || { echo "Missing SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}" >&2; exit 1; }
[[ -d "${GUIDANCE_OUT_DIR}" ]] || { echo "Missing GUIDANCE_OUT_DIR=${GUIDANCE_OUT_DIR}" >&2; exit 1; }

mapfile -t GUIDANCE_STEP_DIRS < <(find "${GUIDANCE_OUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name "step_${STEP}_epoch_*" | sort)
if [[ "${#GUIDANCE_STEP_DIRS[@]}" -ne 1 ]]; then
  echo "Expected exactly one guidance step dir under ${GUIDANCE_OUT_DIR}; found ${#GUIDANCE_STEP_DIRS[@]}:" >&2
  printf '  %s\n' "${GUIDANCE_STEP_DIRS[@]}" >&2
  exit 1
fi
GUIDANCE_STEP_DIR=${GUIDANCE_STEP_DIRS[0]}
TC_PDB_DIR=${GUIDANCE_STEP_DIR}/samples_for_af3_tc
TC_PDB_NAME_LIST=${GUIDANCE_STEP_DIR}/dual_ligand_tc_af3_names.txt
[[ -d "${TC_PDB_DIR}" ]] || { echo "Missing TC_PDB_DIR=${TC_PDB_DIR}" >&2; exit 1; }

find "${TC_PDB_DIR}" -maxdepth 1 -type f -name "*.cif" -printf "%f\n" | sort > "${TC_PDB_NAME_LIST}"
N_TC_SAMPLES=$(wc -l < "${TC_PDB_NAME_LIST}")
if [[ "${N_TC_SAMPLES}" -eq 0 ]]; then
  echo "No CIFs found for TC AF3 in ${TC_PDB_DIR}" >&2
  exit 1
fi

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

echo "Starting dual-ligand selectivity TC AF3."
echo "TASK_ID=${TASK_ID}/${NUM_TASKS}"
echo "GUIDANCE_TARGET_LIGAND_SIDE=${GUIDANCE_TARGET_LIGAND_SIDE}"
echo "GUIDANCE_SCALE=${GUIDANCE_SCALE}"
echo "GUIDANCE_EXP_NAME=${GUIDANCE_EXP_NAME}"
echo "GUIDANCE_STEP_DIR=${GUIDANCE_STEP_DIR}"
echo "TC_PDB_DIR=${TC_PDB_DIR}"
echo "TC_PDB_NAME_LIST=${TC_PDB_NAME_LIST}"
echo "N_TC_SAMPLES=${N_TC_SAMPLES}"
echo "EXP_NAME=${EXP_NAME}"
echo "POCKET_DISTANCE_FOR_DOCKING_METRICS=${POCKET_DISTANCE_FOR_DOCKING_METRICS}"
echo "AF3 TC: recycles=${AF3_RECYCLES}, diffusion_samples=${AF3_DIFFUSION_SAMPLES}, max_templates=${AF3_MAX_TEMPLATES}, ligand_protein_template_conditioning_mode=${AF3_TEMPLATE_CONDITIONING_MODE}, template_pair_scale=${AF3_TEMPLATE_PAIR_SCALE}"
echo "FLASH_ATTN_IMPL=${FLASH_ATTN_IMPL}"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/structure_prediction/run_selectivity_dual_ligand_tc_af3.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/structure_prediction" \
    --config-name run_tc_eval_af3 \
    wandb.no_wandb=true \
    debug=false \
    seed=0 \
    input_sample_is_designed=true \
    +generate_inputs_only=${TC_GENERATE_INPUTS_ONLY} \
    +guidance_mode=selectivity \
    +guidance_scale=${GUIDANCE_SCALE} \
    +guidance_target_ligand_side=${GUIDANCE_TARGET_LIGAND_SIDE} \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    exp_name=${EXP_NAME} \
    base_out_dir=${BASE_OUT_DIR} \
    pdb_cfg.pdb_dir=${TC_PDB_DIR} \
    pdb_cfg.pdb_name_list=${TC_PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    pdb_cfg.array_id=null \
    pdb_cfg.num_arrays=null \
    pocket_cfg.pocket_distance_for_docking_metrics=${POCKET_DISTANCE_FOR_DOCKING_METRICS} \
    struct_pred_cfg.evaluate_docking_consistency=true \
    struct_pred_cfg.calculate_metrics_only=${TC_CALCULATE_METRICS_ONLY} \
    struct_pred_cfg.af3.runner_path="${BASE_DIR}/alphafold3/src/alphafold3/run_alphafold.py" \
    struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
    struct_pred_cfg.af3.inference_config.tc.num_recycles=${AF3_RECYCLES} \
    struct_pred_cfg.af3.inference_config.tc.num_diffusion_samples=${AF3_DIFFUSION_SAMPLES} \
    struct_pred_cfg.af3.inference_config.tc.max_templates=${AF3_MAX_TEMPLATES} \
    struct_pred_cfg.af3.inference_config.tc.ligand_protein_template_conditioning_mode=${AF3_TEMPLATE_CONDITIONING_MODE} \
    struct_pred_cfg.af3.inference_config.tc.template_pair_scale=${AF3_TEMPLATE_PAIR_SCALE} \
    struct_pred_cfg.af3.inference_config.tc.mask_template_sidechains=true \
    struct_pred_cfg.af3.inference_config.tc.mask_template_sequence=true \
    struct_pred_cfg.af3.inference_config.tc.max_template_date=${AF3_MAX_TEMPLATE_DATE} \
    hydra.run.dir=${HYDRA_DIR}

echo "Dual-ligand selectivity TC AF3 complete: EXP_NAME=${EXP_NAME}"
