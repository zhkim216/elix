#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/denovoval/eval_proto_exp4_metal_ions_two_stage.sbatch
#SBATCH --job-name=eval_proto_exp4_ts_metals
#SBATCH -p possu,bioe
#SBATCH --array=0-23
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/run_elix_denovoval_metals_twostage_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/run_elix_denovoval_metals_twostage_%A_%a.err

set -euo pipefail

STEP_FILTER="${STEP_FILTER:-all}"
POTTS_ONLY_COND_FILTER="${POTTS_ONLY_COND_FILTER:-all}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

POCKET_MODEL_NAME=proto_exp4_cfg9
SCAFFOLD_MODEL_NAME=proto_exp4_cfg8
DIRECTION=ps
STEPS=(2500 22500 42500 62500 82500 102500)
STAGE2_POTTS_ONLY_COND_VALUES=(false true)

print_valid_filters() {
  echo "Valid STEP_FILTER values: all ${STEPS[*]}" >&2
  echo "Valid POTTS_ONLY_COND_FILTER values: all ${STAGE2_POTTS_ONLY_COND_VALUES[*]}" >&2
}

COMBOS=()
for CANDIDATE_STEP in "${STEPS[@]}"; do
  if [[ "${STEP_FILTER}" != "all" && "${STEP_FILTER}" != "${CANDIDATE_STEP}" ]]; then
    continue
  fi
  for CANDIDATE_POTTS_ONLY_COND in "${STAGE2_POTTS_ONLY_COND_VALUES[@]}"; do
    if [[ "${POTTS_ONLY_COND_FILTER}" != "all" && "${POTTS_ONLY_COND_FILTER}" != "${CANDIDATE_POTTS_ONLY_COND}" ]]; then
      continue
    fi
    COMBOS+=("${CANDIDATE_STEP}:${CANDIDATE_POTTS_ONLY_COND}")
  done
done

NUM_COMBOS=${#COMBOS[@]}
if [[ "${NUM_COMBOS}" -eq 0 ]]; then
  echo "No combinations selected by STEP_FILTER=${STEP_FILTER}, POTTS_ONLY_COND_FILTER=${POTTS_ONLY_COND_FILTER}." >&2
  print_valid_filters
  exit 2
fi

NUM_CHUNKS=2
NUM_TASKS=$((NUM_COMBOS * NUM_CHUNKS))

if [[ "${TASK_ID}" -ge "${NUM_TASKS}" ]]; then
  echo "TASK_ID=${TASK_ID} is outside selected tasks=${NUM_TASKS}; exiting without work."
  exit 0
fi

COMBO_IDX=$((TASK_ID / NUM_CHUNKS))
CHUNK_IDX=$((TASK_ID % NUM_CHUNKS))
COMBO=${COMBOS[$COMBO_IDX]}
STEP=${COMBO%%:*}
STAGE2_POTTS_ONLY_COND=${COMBO##*:}
POTTS_ONLY_COND_LABEL=potts_only_cond_false
if [[ "${STAGE2_POTTS_ONLY_COND}" == "true" ]]; then
  POTTS_ONLY_COND_LABEL=potts_only_cond_true
fi

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/eval_proto_exp4_metal_ions_two_stage
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=denovoval_metal_ions
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs
SAMPLING_INPUTS_CSV=${PDB_DIR}/sampling_inputs_${VALSET_NAME}.csv
PDB_NAME_LIST=${PDB_DIR}/${VALSET_NAME}.txt

BATCH_SIZE=8
NUM_POCKET_SEQS_PER_PDB=2
NUM_SCAFFOLD_SEQS_PER_PDB=1
NUM_SAMPLES=50
SAMPLING_INDICES="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]"
SAMPLING_LENGTHS="[150,250]"
POCKET_DISTANCE=8.0
POCKET_DISTANCES_FOR_SEQ_RECOVERY="[6.0,8.0,10.0]"
USE_PSEUDOCB_FOR_POCKET_ANNOTATION="${USE_PSEUDOCB_FOR_POCKET_ANNOTATION:-true}"
AF3_RECYCLES=10
AF3_DIFFUSION_SAMPLES=5

POCKET_DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${POCKET_MODEL_NAME}
SCAFFOLD_DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${SCAFFOLD_MODEL_NAME}
EXP_NAME=${POCKET_MODEL_NAME}_pocket_${SCAFFOLD_MODEL_NAME}_scaffold_${VALSET_NAME}_twostage_${DIRECTION}_${POTTS_ONLY_COND_LABEL}_nsample${NUM_SAMPLES}_pseqs${NUM_POCKET_SEQS_PER_PDB}_sseqs${NUM_SCAFFOLD_SEQS_PER_PDB}
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

[[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
[[ -f "${PDB_NAME_LIST}" ]] || { echo "Missing PDB_NAME_LIST=${PDB_NAME_LIST}" >&2; exit 1; }
[[ -f "${SAMPLING_INPUTS_CSV}" ]] || { echo "Missing SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}" >&2; exit 1; }
[[ -d "${POCKET_DENOISER_TRAIN_DIR}" ]] || { echo "Missing POCKET_DENOISER_TRAIN_DIR=${POCKET_DENOISER_TRAIN_DIR}" >&2; exit 1; }
[[ -d "${SCAFFOLD_DENOISER_TRAIN_DIR}" ]] || { echo "Missing SCAFFOLD_DENOISER_TRAIN_DIR=${SCAFFOLD_DENOISER_TRAIN_DIR}" >&2; exit 1; }

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

echo "Starting two-stage denovoval metal-ion eval: task=${TASK_ID}/${NUM_TASKS}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}"
echo "Stage1 pocket: model=${POCKET_MODEL_NAME}, step=${STEP}"
echo "Stage2 scaffold: model=${SCAFFOLD_MODEL_NAME}, step=${STEP}, potts_only_cond=${STAGE2_POTTS_ONLY_COND}"
echo "Stage sampling counts: pocket=${NUM_POCKET_SEQS_PER_PDB}, scaffold=${NUM_SCAFFOLD_SEQS_PER_PDB}"
echo "Filters: STEP_FILTER=${STEP_FILTER}, POTTS_ONLY_COND_FILTER=${POTTS_ONLY_COND_FILTER}"
echo "Pocket distance=${POCKET_DISTANCE}, use_pseudocb_for_pocket_annotation=${USE_PSEUDOCB_FOR_POCKET_ANNOTATION}"
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"
echo "PDB_DIR=${PDB_DIR}"
echo "PDB_NAME_LIST=${PDB_NAME_LIST}"
echo "SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}"
echo "SAMPLING_INDICES=${SAMPLING_INDICES}"
echo "SAMPLING_LENGTHS=${SAMPLING_LENGTHS}"
echo "EXP_NAME=${EXP_NAME}"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/run_elix.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
    --config-name run_elix_two_stage \
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
    model_cfg.denoiser_train_dir=${SCAFFOLD_DENOISER_TRAIN_DIR} \
    model_cfg.ckpt_cfg.start_step=${STEP} \
    model_cfg.ckpt_cfg.end_step=${STEP} \
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
    two_stage.use_pseudocb_for_pocket_annotation=${USE_PSEUDOCB_FOR_POCKET_ANNOTATION} \
    two_stage.pocket.label=${POCKET_MODEL_NAME}_pocket \
    two_stage.pocket.model_cfg.model_name=elix_mpnn \
    two_stage.pocket.model_cfg.denoiser_train_dir=${POCKET_DENOISER_TRAIN_DIR} \
    two_stage.pocket.model_cfg.ckpt_cfg.start_step=${STEP} \
    two_stage.pocket.model_cfg.ckpt_cfg.end_step=${STEP} \
    two_stage.pocket.model_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    two_stage.pocket.model_cfg.ckpt_cfg.eval_last_ckpt=false \
    two_stage.pocket.model_cfg.ckpt_cfg.use_ema=true \
    ++two_stage.pocket.sampling_cfg.overrides.num_seqs_per_pdb=${NUM_POCKET_SEQS_PER_PDB} \
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.potts_only_cond=false \
    two_stage.scaffold.label=${SCAFFOLD_MODEL_NAME}_scaffold \
    two_stage.scaffold.model_cfg.model_name=elix_mpnn \
    two_stage.scaffold.model_cfg.denoiser_train_dir=${SCAFFOLD_DENOISER_TRAIN_DIR} \
    two_stage.scaffold.model_cfg.ckpt_cfg.start_step=22500 \
    two_stage.scaffold.model_cfg.ckpt_cfg.end_step=22500 \
    two_stage.scaffold.model_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
    two_stage.scaffold.model_cfg.ckpt_cfg.eval_last_ckpt=false \
    two_stage.scaffold.model_cfg.ckpt_cfg.use_ema=true \
    ++two_stage.scaffold.sampling_cfg.overrides.num_seqs_per_pdb=${NUM_SCAFFOLD_SEQS_PER_PDB} \
    ++two_stage.scaffold.sampling_cfg.overrides.potts_sampling_cfg.potts_only_cond=${STAGE2_POTTS_ONLY_COND} \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=null \
    pocket_cfg.pocket_distance_for_docking_metrics=${POCKET_DISTANCE} \
    "pocket_cfg.pocket_distances_for_seq_recovery=${POCKET_DISTANCES_FOR_SEQ_RECOVERY}" \
    pocket_cfg.n_min_ligand_atoms_for_seq_recovery=1 \
    input_sample_is_all_atom=true \
    input_sample_is_designed=true \
    load_from_cache=false \
    pdb_cfg.pdb_dir=${PDB_DIR} \
    pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
    pdb_cfg.pdb_name_ext=".cif" \
    pdb_cfg.recursive=false \
    pdb_cfg.split_by_subfolder=false \
    "pdb_cfg.sample_indices=${SAMPLING_INDICES}" \
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

echo "Evaluation complete: step=${STEP}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}, stage2_potts_only_cond=${STAGE2_POTTS_ONLY_COND}, pocket_seqs=${NUM_POCKET_SEQS_PER_PDB}, scaffold_seqs=${NUM_SCAFFOLD_SEQS_PER_PDB}"
