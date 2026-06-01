#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval_seq_des_training/lc_seq_des_multi/denovoval_metal_ions/lc_seq_des_multi_denovoval_metal_ions_per_step_sweep.sbatch
#SBATCH --job-name=lc_seq_des_denovoval_metals
#SBATCH -p possu,bioe
#SBATCH --array=0-23
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_denovoval_metals_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/lc_seq_des_denovoval_metals_%A_%a.err

set -euo pipefail

MODEL_FILTER="${MODEL_FILTER:-all}"
STEP_FILTER="${STEP_FILTER:-all}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

MODEL_NAMES=(
  proto_nprot1_no_filter
  proto_nprot1_external_evidence
  proto_allprot_no_filter
  proto_allprot_external_evidence
)
STEPS=(2500 22500 42500 62500 102500 122500)

print_valid_filters() {
  echo "Valid MODEL_FILTER values: all ${MODEL_NAMES[*]}" >&2
  echo "Valid STEP_FILTER values: all ${STEPS[*]}" >&2
}

MODEL_STEP_COMBOS=()
for CANDIDATE_MODEL in "${MODEL_NAMES[@]}"; do
  if [[ "${MODEL_FILTER}" != "all" && "${MODEL_FILTER}" != "${CANDIDATE_MODEL}" ]]; then
    continue
  fi
  for CANDIDATE_STEP in "${STEPS[@]}"; do
    if [[ "${STEP_FILTER}" != "all" && "${STEP_FILTER}" != "${CANDIDATE_STEP}" ]]; then
      continue
    fi
    MODEL_STEP_COMBOS+=("${CANDIDATE_MODEL}:${CANDIDATE_STEP}")
  done
done

NUM_COMBOS=${#MODEL_STEP_COMBOS[@]}
if [[ "${NUM_COMBOS}" -eq 0 ]]; then
  echo "No model/step combinations selected by MODEL_FILTER=${MODEL_FILTER}, STEP_FILTER=${STEP_FILTER}." >&2
  print_valid_filters
  exit 2
fi

if [[ "${TASK_ID}" -ge "${NUM_COMBOS}" ]]; then
  echo "TASK_ID=${TASK_ID} is outside selected combinations=${NUM_COMBOS}; exiting without work."
  exit 0
fi

COMBO=${MODEL_STEP_COMBOS[$TASK_ID]}
MODEL_NAME=${COMBO%%:*}
STEP=${COMBO##*:}

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/eval_denovoval_metal_ions_per_step_sweep
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=denovoval_metal_ions
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs
SAMPLING_INPUTS_CSV=${PDB_DIR}/sampling_inputs_${VALSET_NAME}.csv
PDB_NAME_LIST=${PDB_DIR}/${VALSET_NAME}.txt

NUM_CHUNKS=1
CHUNK_IDX=0
BATCH_SIZE=1
NUM_SEQS_PER_PDB=2
NUM_SAMPLES=5
SAMPLING_INDICES="[0,1,2,3,4]"
SAMPLING_LENGTHS="[150,250]"
POCKET_DISTANCE_FOR_DOCKING_METRICS=10.0
POCKET_DISTANCES_FOR_SEQ_RECOVERY="[8.0,10.0]"
AF3_RECYCLES=10
AF3_DIFFUSION_SAMPLES=5

EXP_NAME=${MODEL_NAME}_${VALSET_NAME}_nsample${NUM_SAMPLES}_numseqs${NUM_SEQS_PER_PDB}
DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${MODEL_NAME}
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}/step_${STEP}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_DIR}" "${BASE_OUT_DIR}"

[[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
[[ -f "${PDB_NAME_LIST}" ]] || { echo "Missing PDB_NAME_LIST=${PDB_NAME_LIST}" >&2; exit 1; }
[[ -f "${SAMPLING_INPUTS_CSV}" ]] || { echo "Missing SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}" >&2; exit 1; }

cd "${BASE_DIR}"

export PYTHONWARNINGS="ignore"
WANDB_ID=zhkim216
PROJECT=eval_ligand_seq_des


COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=$(echo "$COMPUTE_CAP" | cut -d. -f1)

if [[ "$MAJOR" -ge 8 ]]; then
    FLASH_ATTN_IMPL="triton"
else
    FLASH_ATTN_IMPL="xla"
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi

echo "Starting denovoval metal-ion eval: model=${MODEL_NAME}, step=${STEP}, selected_task=${TASK_ID}/${NUM_COMBOS}"
echo "Filters: MODEL_FILTER=${MODEL_FILTER}, STEP_FILTER=${STEP_FILTER}"
echo "GPU compute capability: ${COMPUTE_CAP}, using flash_attention_implementation=${FLASH_ATTN_IMPL}"
echo "PDB_DIR=${PDB_DIR}"
echo "PDB_NAME_LIST=${PDB_NAME_LIST}"
echo "SAMPLING_INPUTS_CSV=${SAMPLING_INPUTS_CSV}"
echo "SAMPLING_INDICES=${SAMPLING_INDICES}"
echo "SAMPLING_LENGTHS=${SAMPLING_LENGTHS}"

WANDB__SERVICE_WAIT=300 python3 allatom_design/eval/sampling/lc_seq_des_multi.py \
    --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
    --config-name lc_seq_des_multi \
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
    seq_des_cfg.atom_mpnn.overrides.use_potts_sampling=true \
    guidance_cfg.enabled=false \
    sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
    pos_constraint_csv=null \
    redesign_cfg.use_native_seq=false \
    redesign_cfg.native_cif_dir=null \
    pocket_cfg.pocket_distance_for_docking_metrics=${POCKET_DISTANCE_FOR_DOCKING_METRICS} \
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

echo "Evaluation complete: model=${MODEL_NAME}, step=${STEP}"
