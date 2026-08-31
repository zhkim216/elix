#!/usr/bin/env bash
# Submit from Sherlock:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/train/elix_exp8.sbatch
#
# NOTE: The #SBATCH lines below (job-name / output / error) cannot use shell
# variables because SLURM parses them before the shell runs. Keep them in sync
# with EXP_NAME_BASE manually.
#SBATCH --job-name=elix_exp8
#SBATCH -p possu
#SBATCH --array=0-4
#SBATCH --gres=gpu:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/elix_exp8_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/elix_exp8_%A_%a.err

set -euo pipefail

# Single source of truth for the experiment name. HYDRA_DIR and the inductor
# cache below are derived from it, so changing this propagates automatically.
# (The #SBATCH lines above still need a manual update.)
EXP_NAME_BASE="elix_exp8"

BASE_DIR=/home/users/zhkim216/code/elix
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs/train_seq_denoiser/${EXP_NAME_BASE}
OUT_DIR=/scratch/users/zhkim216/out_dir
JOB_OUTPUT_DIR=/scratch/users/zhkim216/job_output/train_seq_denoiser

mkdir -p "${JOB_OUTPUT_DIR}" "${HYDRA_DIR}" "${OUT_DIR}"

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"
export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_${EXP_NAME_BASE}_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

cd "${BASE_DIR}"

# Wandb settings
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP=training
DENOISER_TYPE="elix_mpnn"

COMMON_OVERRIDES=(
  "data.cluster_id_col=q_pn_unit_cluster_03_id"
  "data.val_exclusion_cluster_id_col=q_pn_unit_cluster_03_id"
  "data.featurizer_cfg.training_structure_noise=0.1"
  "denoiser.mpnn.potts.parameterization=factor"
  "denoiser.mpnn.potts.norm_potts_inputs=false"
)

CONFIGS=(
  "no_bml_context|data.query_pn_unit_iids_only=true data.bml_context.enabled=false"
  "bml_context|data.query_pn_unit_iids_only=true data.bml_context.enabled=true"
  "bml_context_crop30|data.query_pn_unit_iids_only=true data.bml_context.enabled=true data.featurizer_cfg.crop_center_cutoff_distance=30.0"
  "bml_context_maxtok768|data.query_pn_unit_iids_only=true data.bml_context.enabled=true data.featurizer_cfg.max_tokens=768"  
  "bml_context_dropout02|data.query_pn_unit_iids_only=true data.bml_context.enabled=true denoiser.mpnn.dropout_p=0.2"
  "bml_context_maxtok768|data.query_pn_unit_iids_only=true data.bml_context.enabled=true data.featurizer_cfg.max_tokens=768"  
  "bml_context_dropout02|data.query_pn_unit_iids_only=true data.bml_context.enabled=true denoiser.mpnn.dropout_p=0.2"
)

IDX=$((ARRAY_TASK_ID))
if (( IDX < 0 || IDX >= ${#CONFIGS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID}; expected 0..$((${#CONFIGS[@]} - 1))" >&2
  exit 2
fi

CONFIG_ROW="${CONFIGS[$IDX]}"
IFS='|' read -r CONFIG_NAME OVERRIDES <<< "${CONFIG_ROW}"
if [[ -z "${CONFIG_NAME}" || -z "${OVERRIDES}" ]]; then
  echo "Malformed config row at index ${IDX}: ${CONFIG_ROW}" >&2
  exit 2
fi
EXP_NAME="${EXP_NAME_BASE}_cfg${IDX}_${CONFIG_NAME}"

INTERFACE_TYPE_FILTER="n_prot == 1"
INTERFACE_FILTER_OVERRIDE=("data.train_filters.interface_filter.2=[\"${INTERFACE_TYPE_FILTER}\"]")

# Expand the protein monomer-chain filter resolution cutoff from 3.5 to 9.0.
# Value must be passed as a quoted hydra string because it contains spaces,
# parentheses, and comparison operators.
MONOMER_CHAIN_FILTER="q_pn_unit_is_protein and resolution < 9.0 and 20 <= q_pn_unit_num_resolved_residues < 2048"
MONOMER_FILTER_OVERRIDE=("data.train_filters.protein_monomer_chain_filter.0=\"${MONOMER_CHAIN_FILTER}\"")

# Dataset paths
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data
PDB_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9"
TRAIN_METADATA_PATH="/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9/metadata_ligval_seq_clustered_external_evidence_ver5.parquet"
VALIDATION_IDS_FILE="/scratch/users/zhkim216/datasets/val_cifs/nativeval_cifs/nativeval_sm_metal.txt"
VAL_METADATA_PATH="/scratch/users/zhkim216/datasets/val_cifs/nativeval_cifs/metadata_for_training_nativeval_sm_metal.parquet"

echo "Starting ${EXP_NAME}"
echo "  config_name: ${CONFIG_NAME}"
echo "  config_row: ${CONFIG_ROW}"
echo "  common_overrides: ${COMMON_OVERRIDES[*]}"
echo "  denoiser_type: ${DENOISER_TYPE}"
echo "  external_evidence_policy: external_evidence"
echo "  query_pn_unit_iids_only: true"
echo "  alpha_protein_protein: 0.0"
echo "  k_neighbors: 48"
echo "  interface_filter_override: ${INTERFACE_FILTER_OVERRIDE[*]}"
echo "  monomer_filter_override: ${MONOMER_FILTER_OVERRIDE[*]}"
echo "  base_dir: ${BASE_DIR}"
echo "  out_dir: ${OUT_DIR}"
echo "  hydra_dir: ${HYDRA_DIR}/${EXP_NAME}"
echo "  torchinductor_cache: ${TORCHINDUCTOR_CACHE_DIR}"

DEBUG= HYDRA_FULL_ERROR=1 WANDB__SERVICE_WAIT=300 \
python3 allatom_design/train_seq_denoiser.py \
    --config-path configs/seq_denoiser \
    --config-name elix.yaml \
    wandb.wandb_id="${WANDB_ID}" \
    wandb.project="${PROJECT}" \
    wandb.group="${GROUP}" \
    wandb.no_wandb=false \
    num_workers="${CPUS_PER_TASK}" \
    out_dir="${OUT_DIR}" \
    exp_name="${EXP_NAME}" \
    train.debug=false \
    train.batch_size=16 \
    trainer.accumulate_grad_batches=1 \
    train.batch_size=8 \
    trainer.accumulate_grad_batches=2 \
    train.compile.compile_model=true \
    trainer.strategy="ddp_find_unused_parameters_true" \
    denoiser.name="${DENOISER_TYPE}" \
    resume.use_current_cfg=true \
    resume.ckpt_path=null \
    loss.loss_weights.seq_loss=1.0 \
    loss.main_seq_loss_pocket_only=false \
    loss.main_potts_loss_pocket_only=false \
    loss.seq_loss.per_token_avg=false \
    loss.potts.per_token_avg=false \
    data.pdb_path="${PDB_PATH}" \
    data.train_metadata_path="${TRAIN_METADATA_PATH}" \
    data.validation_ids_file="${VALIDATION_IDS_FILE}" \
    data.val_metadata_path="${VAL_METADATA_PATH}" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    data.exclude_val_cluster=true \
    data.interface_only_training=false \
    data.samples_per_epoch=50000 \
    data.bml_center.metal.external_evidence_policy="external_evidence" \
    data.bml_center.small_molecule.artifact_external_evidence_policy="filter_all_artifacts" \
    data.bml_center.small_molecule.min_contacting_protein_atom_ratio=1.0 \
    data.query_pn_unit_iids_only=true \
    "${INTERFACE_FILTER_OVERRIDE[@]}" \
    "${MONOMER_FILTER_OVERRIDE[@]}" \
    data.sampling_weights.alphas_interface.alpha_protein_protein=0.0 \
    data.sampling_weights.alphas_interface.alpha_protein_small_molecule=1.0 \
    data.sampling_weights.alphas_interface.alpha_protein_metal=1.0 \
    data.sampling_weights.alphas_interface.alpha_protein_nuc=0.0 \
    data.sampling_weights.alphas_interface.alpha_protein_nuc_lig=1.0 \
    data.sampling_weights.alphas_interface.alpha_protein_peptide=1.0 \
    data.sampling_weights.single_protein_context_weight=1.0 \
    data.sampling_weights.multi_protein_context_weight=0.0 \
    data.sampling_weights.k_percentile=90.0 \
    data.sampling_weights.ligand_cluster_col=null \
    data.featurizer_cfg.is_inference=false \
    data.featurizer_cfg.occupancy_threshold_protein_backbone=0.0 \
    data.featurizer_cfg.remove_unresolved_tokens=true \
    data.featurizer_cfg.max_tokens=512 \
    data.featurizer_cfg.max_atoms=4608 \
    data.featurizer_cfg.crop_center_cutoff_distance=15.0 \
    data.featurizer_cfg.crop_spatial_p_protein_monomer_chain=0.25 \
    data.featurizer_cfg.drop_prob_non_protein_chains=0.0 \
    data.featurizer_cfg.pocket_distance=5.0 \
    data.featurizer_cfg.pocket_n_min_ligand_atoms=1 \
    data.featurizer_cfg.asymmetric_noise=false \
    denoiser.mpnn.hidden_dim=128 \
    denoiser.mpnn.k_neighbors=48 \
    denoiser.mpnn.ligand_conditioning=true \
    denoiser.mpnn.token_features.protein_graph_rbf_type="ncacocb" \
    denoiser.mpnn.token_features.ligand_atom_context_num=25 \
    denoiser.mpnn.potts.k_neighbors_potts=48 \
    denoiser.mpnn.potts.multi_head.dim_multi_head=128 \
    denoiser.mpnn.potts.multi_head.num_heads=1 \
    denoiser.mpnn.potts.multi_head.reduce=mean \
    denoiser.mpnn.dropout_p=0.1 \
    denoiser.mpnn.expansion_mode=node_concat_context_skip \
    denoiser.mpnn.lmpnn_module.context_edge_update=true \
    denoiser.mpnn.use_context_skip_connection=false \
    mask_selector.restype_masking_schedule="uniform_cosine_t" \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    trainer.max_steps=200000 \
    "${COMMON_OVERRIDES[@]}" \
    ${OVERRIDES} \
    hydra.run.dir="${HYDRA_DIR}/${EXP_NAME}"
