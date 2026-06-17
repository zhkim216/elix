#!/bin/bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/eval/denovoval/eval_elix_exp7_cfg6_pocket_exp1_cfg0_scaffold_denovoval_sm_metal_ensemble_weighted_average_config_sweep.sbatch
# Optional focused run:
#   CONFIG_IDX=4 <submit command above>
# Fixed exp7_cfg6 pocket step 82500 / exp1_cfg0 scaffold step 32500.
# Array layout is CONFIG_IDX x CHUNK_IDX = 7 x 8.
#SBATCH --job-name=eval_exp7c6_exp1c0_ts_wavg
#SBATCH -p possu,bioe
#SBATCH --array=0-55
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/eval_exp7c6_exp1c0_ts_wavg_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/eval_ligand_seq_des/eval_exp7c6_exp1c0_ts_wavg_%A_%a.err

set -euo pipefail

PRINT_ARRAY_PLAN="${PRINT_ARRAY_PLAN:-false}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

POCKET_MODEL_NAME=elix_exp7_cfg6_context_skip_decoder_skip
POCKET_STEP=82500
POCKET_TOTAL_MEMBERS=8
POCKET_REF_MEMBERS=4

SCAFFOLD_MODEL_NAME=elix_exp1_cfg0
SCAFFOLD_STEP=32500
SCAFFOLD_TOTAL_MEMBERS=16
SCAFFOLD_PROTEIN_NOISE_STD=0.5
SCAFFOLD_SMALL_MOLECULE_NOISE_STD=0.0
SCAFFOLD_METAL_NOISE_STD=0.0

DIRECTION=ps
POCKET_DISTANCE=10.0
POCKET_ANNOTATION_METHOD=calpha
NUM_CHUNKS=8
RUN_VARIANT=weighted_average

BATCH_SIZE=8
NUM_POCKET_SEQS_PER_PDB=1
NUM_SCAFFOLD_SEQS_PER_PDB=1
SAMPLING_LENGTHS="[150,300]"
POCKET_DISTANCES_FOR_SEQ_RECOVERY="[4.0,6.0,8.0,10.0]"
POCKET_DISTANCE_BINS="[[0.0,4.0],[4.0,6.0],[6.0,8.0],[8.0,10.0]]"
AF3_RECYCLES=10
AF3_DIFFUSION_SAMPLES=5

CONFIG_LABELS=(
  "gauss_p0p0_sm0p0_m0p0"
  "gauss_p0p0_sm0p0_m0p1"
  "gauss_p0p0_sm0p05_m0p1"
  "gauss_p0p0_sm0p1_m0p1"
  "ligconf_ref${POCKET_REF_MEMBERS}_p0p0_rmsd1p0_m0p1"
  "ligconf_ref${POCKET_REF_MEMBERS}_p0p0_rmsd1p5_m0p1"
  "ligconf_ref${POCKET_REF_MEMBERS}_p0p0_rmsd2p0_m0p1"
)

CONFIGS=(
  "++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.noise_std=0.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.noise_std=0.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.noise_std=0.0"
  "++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.noise_std=0.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.noise_std=0.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.noise_std=0.1"
  "++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.noise_std=0.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.noise_std=0.05 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.noise_std=0.1"
  "++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.noise_std=0.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.noise_std=0.1 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.noise_std=0.1"
  "++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.weights.scheme=weighted_mean ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.weights.ref_members=${POCKET_REF_MEMBERS} ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.noise_std=0.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.mode=ligand_conformer ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.rmsd_cluster_cutoff=1.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.num_conformer_candidates=50 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.num_threads=8 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.uff_optimize=true ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.clash_target_atoms=backbone ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.exclude_clashing_decoys=false ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.noise_std=0.1"
  "++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.weights.scheme=weighted_mean ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.weights.ref_members=${POCKET_REF_MEMBERS} ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.noise_std=0.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.mode=ligand_conformer ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.rmsd_cluster_cutoff=1.5 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.num_conformer_candidates=50 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.num_threads=8 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.uff_optimize=true ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.clash_target_atoms=backbone ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.exclude_clashing_decoys=false ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.noise_std=0.1"
  "++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.weights.scheme=weighted_mean ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.weights.ref_members=${POCKET_REF_MEMBERS} ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.protein.noise_std=0.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.mode=ligand_conformer ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.rmsd_cluster_cutoff=2.0 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.num_conformer_candidates=50 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.num_threads=8 ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.uff_optimize=true ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.clash_target_atoms=backbone ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.small_molecule.exclude_clashing_decoys=false ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.mode=gaussian_noise ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.metal.noise_std=0.1"
)

NUM_CONFIGS=${#CONFIGS[@]}
NUM_TASKS=$((NUM_CONFIGS * NUM_CHUNKS))

USER_CONFIG_IDX_SET=false
if [[ -z "${CONFIG_IDX:-}" ]]; then
  CONFIG_IDX=$((TASK_ID / NUM_CHUNKS))
else
  USER_CONFIG_IDX_SET=true
fi
if [[ -z "${CHUNK_IDX:-}" ]]; then
  if [[ "${USER_CONFIG_IDX_SET}" == "true" ]]; then
    CHUNK_IDX=${TASK_ID}
  else
    CHUNK_IDX=$((TASK_ID % NUM_CHUNKS))
  fi
fi

if [[ "${CONFIG_IDX}" -lt 0 || "${CONFIG_IDX}" -ge "${NUM_CONFIGS}" ]]; then
  echo "CONFIG_IDX=${CONFIG_IDX} is outside valid range 0-$((NUM_CONFIGS - 1))." >&2
  exit 2
fi
if [[ "${CHUNK_IDX}" -lt 0 || "${CHUNK_IDX}" -ge "${NUM_CHUNKS}" ]]; then
  if [[ "${USER_CONFIG_IDX_SET}" == "true" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "TASK_ID=${TASK_ID} is outside focused CONFIG_IDX=${CONFIG_IDX} chunks=${NUM_CHUNKS}; exiting without work."
    exit 0
  fi
  echo "CHUNK_IDX=${CHUNK_IDX} is outside valid range 0-$((NUM_CHUNKS - 1))." >&2
  exit 2
fi

if [[ "${PRINT_ARRAY_PLAN}" == "true" ]]; then
  echo "Selected exp7_cfg6 pocket / exp1_cfg0 scaffold denovoval sm-metal ensemble config sweep (${RUN_VARIANT}): NUM_CONFIGS=${NUM_CONFIGS}, NUM_CHUNKS=${NUM_CHUNKS}, NUM_TASKS=${NUM_TASKS}"
  for PLAN_CONFIG_IDX in "${!CONFIGS[@]}"; do
    echo "config_idx=${PLAN_CONFIG_IDX} label=${CONFIG_LABELS[$PLAN_CONFIG_IDX]} pocket_total_members=${POCKET_TOTAL_MEMBERS} pocket_ref_members=${POCKET_REF_MEMBERS} pocket_step=${POCKET_STEP} scaffold_total_members=${SCAFFOLD_TOTAL_MEMBERS} scaffold_step=${SCAFFOLD_STEP}"
    echo "  ${CONFIGS[$PLAN_CONFIG_IDX]}"
  done
  exit 0
fi

CONFIG_LABEL=${CONFIG_LABELS[$CONFIG_IDX]}
read -r -a POCKET_CONFIG_OVERRIDES <<< "${CONFIGS[$CONFIG_IDX]}"

SCAFFOLD_PROTEIN_NOISE_LABEL=${SCAFFOLD_PROTEIN_NOISE_STD//./p}

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/eval_elix_exp7_cfg6_pocket_exp1_cfg0_scaffold_denovoval_sm_metal_ensemble_weighted_average_config_sweep
BASE_OUT_DIR=/scratch/users/zhkim216/out_dir/eval_ligand_seq_des
VALSET_NAME=denovoval_sm_metal
PDB_DIR=/scratch/users/zhkim216/datasets/val_cifs/denovoval_cifs/denovoval_sm_metal/cifs
PDB_NAME_LIST=null

POCKET_DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${POCKET_MODEL_NAME}
SCAFFOLD_DENOISER_TRAIN_DIR=/scratch/users/zhkim216/out_dir/ligand_seq_des/${SCAFFOLD_MODEL_NAME}
EXP_NAME=${POCKET_MODEL_NAME}_pocket_${SCAFFOLD_MODEL_NAME}_scaffold_${VALSET_NAME}_twostage_${DIRECTION}_${RUN_VARIANT}_pocket_ens_n${POCKET_TOTAL_MEMBERS}_${CONFIG_LABEL}_scaffold_ens_n${SCAFFOLD_TOTAL_MEMBERS}_protein_noise${SCAFFOLD_PROTEIN_NOISE_LABEL}_ss_recycles${AF3_RECYCLES}_diff${AF3_DIFFUSION_SAMPLES}_pseqs${NUM_POCKET_SEQS_PER_PDB}_sseqs${NUM_SCAFFOLD_SEQS_PER_PDB}
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}/chunk_${CHUNK_IDX}

checkpoint_exists_for_step() {
  local train_dir="$1"
  local step="$2"
  compgen -G "${train_dir}/checkpoints/ema/sd-step${step}-epoch*-ema*.ckpt" >/dev/null || \
    compgen -G "${train_dir}/checkpoints/sd-step${step}-epoch*.ckpt" >/dev/null
}

mkdir -p /scratch/users/zhkim216/job_output/eval_ligand_seq_des "${HYDRA_ROOT}" "${BASE_OUT_DIR}"

[[ -d "${PDB_DIR}" ]] || { echo "Missing PDB_DIR=${PDB_DIR}" >&2; exit 1; }
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

echo "Starting exp7_cfg6 pocket / exp1_cfg0 scaffold denovoval sm-metal ensemble config sweep (${RUN_VARIANT}): task=${TASK_ID}/${NUM_TASKS}, config_idx=${CONFIG_IDX}/${NUM_CONFIGS}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}"
echo "Stage1 pocket: model=${POCKET_MODEL_NAME}, step=${POCKET_STEP}, total_members=${POCKET_TOTAL_MEMBERS}, ref_members=${POCKET_REF_MEMBERS}, config_label=${CONFIG_LABEL}"
echo "Stage1 pocket config: ${CONFIGS[$CONFIG_IDX]}"
echo "Stage2 scaffold: model=${SCAFFOLD_MODEL_NAME}, step=${SCAFFOLD_STEP}, total_members=${SCAFFOLD_TOTAL_MEMBERS}, protein_noise_std=${SCAFFOLD_PROTEIN_NOISE_STD}"
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
    wandb.group=${VALSET_NAME}_twostage_ensemble_config_sweep_${RUN_VARIANT} \
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
    two_stage.pocket.label=${POCKET_MODEL_NAME}_pocket_ens_${CONFIG_LABEL} \
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
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.enabled=true \
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.total_members=${POCKET_TOTAL_MEMBERS} \
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.weights.scheme=mean \
    ++two_stage.pocket.sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.noise_seed=null \
    "${POCKET_CONFIG_OVERRIDES[@]}" \
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
    pocket_cfg.pocket_distance_for_docking_metrics=${POCKET_DISTANCE} \
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

echo "Evaluation complete: pocket_model=${POCKET_MODEL_NAME}, pocket_step=${POCKET_STEP}, scaffold_model=${SCAFFOLD_MODEL_NAME}, scaffold_step=${SCAFFOLD_STEP}, config_idx=${CONFIG_IDX}, config_label=${CONFIG_LABEL}, chunk=${CHUNK_IDX}/${NUM_CHUNKS}"
