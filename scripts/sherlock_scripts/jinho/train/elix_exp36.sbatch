#!/usr/bin/env bash
# Submit after review with:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/train/elix_exp36.sbatch
#
# All configs use the same H100 x2 resource shape; decoder selection is an
# experiment-matrix value owned by the array task.
#SBATCH --job-name=elix_exp36
#SBATCH -p possu
#SBATCH --array=0-1
#SBATCH --gres=gpu:2
#SBATCH --mem=164G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --time=1-0:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/elix_exp36_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/elix_exp36_%A_%a.err

set -euo pipefail

EXP_NAME_BASE=elix_exp36
BASE_DIR=/home/users/zhkim216/code/elix
HYDRA_DIR=/scratch/users/zhkim216/hydra_outputs/train_seq_denoiser/${EXP_NAME_BASE}
OUT_DIR=/scratch/users/zhkim216/out_dir
JOB_OUTPUT_DIR=/scratch/users/zhkim216/job_output/train_seq_denoiser
EXPECTED_VENV=/scratch/users/zhkim216/envs/uv/elix-torch280

mkdir -p "${JOB_OUTPUT_DIR}" "${HYDRA_DIR}" "${OUT_DIR}"

PYTHON_PREFIX="$(python3 -c 'import sys; print(sys.prefix)')"
if [[ "${PYTHON_PREFIX}" != "${EXPECTED_VENV}" ]]; then
  echo "Expected Torch 2.8 environment ${EXPECTED_VENV}, got ${PYTHON_PREFIX}" >&2
  echo "Submit through wrap_sbatch_in_container_elix_torch280.sh" >&2
  exit 2
fi

ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
GPU_COUNT=2
NUM_WORKERS=4
BATCH_SIZE=8
ACCUMULATE_GRAD_BATCHES=2
export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_${EXP_NAME_BASE}_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

cd "${BASE_DIR}"

WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP=training
DENOISER_TYPE=elix_mpnn
EXPANSION_MODE=edge_node_context_concat_multi_head_gate_nonlinear
CONTEXT_EDGE_UPDATE=true
UPDATE_FINAL_CONTEXT_PROCESSOR_EDGE=false

# name | decoder input | final encoder edge update | decoder layers
CONFIGS=(
  "ctxedge_lastoff_baseline|legacy_node_add|false|2"
  "ctxedge_lastoff_final_encoder_edge|legacy_node_add|true|2"
)

IDX=$((ARRAY_TASK_ID))
if (( IDX < 0 || IDX >= ${#CONFIGS[@]} )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID}; expected 0..$((${#CONFIGS[@]} - 1))" >&2
  exit 2
fi

CONFIG_ROW="${CONFIGS[$IDX]}"
IFS='|' read -r CONFIG_NAME DECODER_INPUT_MODE UPDATE_FINAL_ENCODER_EDGE NUM_DECODER_LAYERS EXTRA_FIELD <<< "${CONFIG_ROW}"
if [[ -z "${CONFIG_NAME}" || -z "${DECODER_INPUT_MODE}" || \
      -z "${UPDATE_FINAL_ENCODER_EDGE}" || -z "${NUM_DECODER_LAYERS}" || \
      -n "${EXTRA_FIELD}" ]]; then
  echo "Malformed config row at index ${IDX}: ${CONFIG_ROW}" >&2
  exit 2
fi

case "${IDX}|${DECODER_INPUT_MODE}|${UPDATE_FINAL_ENCODER_EDGE}|${NUM_DECODER_LAYERS}" in
  "0|legacy_node_add|false|2"|\
  "1|legacy_node_add|true|2")
    POTTS_EDGE_INPUT_DIM=640
    ;;
  *)
    echo "Unsupported experiment contract at index ${IDX}: ${CONFIG_ROW}" >&2
    exit 2
    ;;
esac

EXP_NAME="${EXP_NAME_BASE}_cfg${IDX}_${CONFIG_NAME}"

COMMON_OVERRIDES=(
  "data.cluster_id_col=q_pn_unit_cluster_id"
  "data.val_exclusion_cluster_id_col=q_pn_unit_cluster_id"
  "data.featurizer_cfg.training_structure_noise=0.1"
  "denoiser.mpnn.potts.parameterization=multi_head_factor"
  "denoiser.mpnn.potts.norm_potts_inputs=false"
)
SAMPLING_OVERRIDES=(
  "data.sampling_weights.fixed_k=3.0"
  "data.sampling_weights.alphas_interface.alpha_protein_small_molecule=8.0"
  "data.sampling_weights.alphas_interface.alpha_protein_metal=1.0"
  "data.sampling_weights.alphas_interface.alpha_protein_nuc_lig=1.0"
  "data.sampling_weights.alphas_interface.alpha_protein_peptide=1.0"
)
FEATURE_OVERRIDES=(
  "denoiser.mpnn.token_features.use_ligand_f_block_features=true"
  "denoiser.mpnn.token_features.use_ligand_asinh_formal_charge=true"
  "denoiser.mpnn.token_features.use_ligand_cached_rdkit_chirality=true"
  "denoiser.mpnn.token_features.use_ligand_bond_order=true"
  "denoiser.mpnn.token_features.use_token_bonds=true"
  "denoiser.mpnn.token_features.add_hydrogenbond_feature=true"
)

INTERFACE_FILTER_OVERRIDE=('data.train_filters.interface_filter.2=["n_prot == 1"]')
MONOMER_FILTER_OVERRIDE=('data.train_filters.protein_monomer_chain_filter.0="q_pn_unit_is_protein and resolution < 9.0 and 20 <= q_pn_unit_num_resolved_residues < 2048"')

RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/cache_residue_runs/ccd_100conf_torch280_20260716/atomworks_cached_residue_data
EXPECTED_RESIDUE_CACHE_FILES=49190
PDB_PATH=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9
TRAIN_METADATA_PATH=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9/metadata_seq_clustered_03_finalized.parquet
RUNTIME_INDEX_PATH=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9/runtime_indices/elix_exp41_exp42_fixedk3.arrow
VALIDATION_IDS_FILE=/scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/nativeval.txt
VAL_METADATA_PATH=/scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/metadata_for_training_nativeval.parquet

if [[ ! -d "${RESIDUE_CACHE_DIR}" ]]; then
  echo "Residue cache directory is missing: ${RESIDUE_CACHE_DIR}" >&2
  exit 2
fi
if [[ ! -f "${RUNTIME_INDEX_PATH}" ]]; then
  echo "Runtime metadata index is missing: ${RUNTIME_INDEX_PATH}" >&2
  echo "Build the config-matched index before submitting training." >&2
  exit 2
fi
RESIDUE_CACHE_FILE_COUNT="$(
  find "${RESIDUE_CACHE_DIR}" -mindepth 2 -maxdepth 2 -type f -name '*.pt' -printf '%P\n' |
    awk -F/ 'NF == 2 {name = $2; sub(/[.]pt$/, "", name); if ($1 == name) count++} END {print count + 0}'
)"
if (( RESIDUE_CACHE_FILE_COUNT < EXPECTED_RESIDUE_CACHE_FILES )); then
  echo "Residue cache is incomplete: found ${RESIDUE_CACHE_FILE_COUNT}; expected at least ${EXPECTED_RESIDUE_CACHE_FILES}" >&2
  exit 2
fi

echo "Starting ${EXP_NAME}"
echo "  config_row: ${CONFIG_ROW}"
echo "  gpu_count: ${GPU_COUNT}"
echo "  gpu_sku: H100_SXM5"
echo "  num_workers_per_rank: ${NUM_WORKERS}"
echo "  batch_size_per_gpu: ${BATCH_SIZE}"
echo "  accumulate_grad_batches: ${ACCUMULATE_GRAD_BATCHES}"
echo "  effective_global_batch_size: $((BATCH_SIZE * ACCUMULATE_GRAD_BATCHES * GPU_COUNT))"
echo "  decoder_input_mode: ${DECODER_INPUT_MODE}"
echo "  decoder_layers: ${NUM_DECODER_LAYERS}"
echo "  expansion_mode: ${EXPANSION_MODE}"
echo "  potts_nodes: 256"
echo "  potts_edge_input_dim: ${POTTS_EDGE_INPUT_DIM}"
echo "  potts_shared_edge_dim: 320"
echo "  potts_num_heads: 2"
echo "  context_edge_update: ${CONTEXT_EDGE_UPDATE}"
echo "  update_final_encoder_edge: ${UPDATE_FINAL_ENCODER_EDGE}"
echo "  update_final_context_processor_edge: ${UPDATE_FINAL_CONTEXT_PROCESSOR_EDGE}"
echo "  cluster_id_col: q_pn_unit_cluster_id"
echo "  val_exclusion_cluster_id_col: q_pn_unit_cluster_id"
echo "  residue_cache_dir: ${RESIDUE_CACHE_DIR}"
echo "  residue_cache_files: ${RESIDUE_CACHE_FILE_COUNT}"
echo "  runtime_index_path: ${RUNTIME_INDEX_PATH}"
echo "  hydra_dir: ${HYDRA_DIR}/${EXP_NAME}"
echo "  torchinductor_cache: ${TORCHINDUCTOR_CACHE_DIR}"

SLURM_JOB_ID_FOR_DDP="${SLURM_JOB_ID:-0}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + (SLURM_JOB_ID_FOR_DDP % 40000) + ARRAY_TASK_ID))
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
echo "[DDP] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[GPU] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

DEBUG= HYDRA_FULL_ERROR=1 WANDB__SERVICE_WAIT=300 \
python3 allatom_design/train_seq_denoiser.py \
    --config-path configs/seq_denoiser \
    --config-name elix.yaml \
    wandb.wandb_id="${WANDB_ID}" \
    wandb.project="${PROJECT}" \
    wandb.group="${GROUP}" \
    wandb.no_wandb=false \
    num_workers="${NUM_WORKERS}" \
    out_dir="${OUT_DIR}" \
    exp_name="${EXP_NAME}" \
    train.debug=false \
    train.batch_size="${BATCH_SIZE}" \
    trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
    train.compile.compile_model=true \
    trainer.strategy=ddp_find_unused_parameters_true \
    denoiser.name="${DENOISER_TYPE}" \
    resume.use_current_cfg=true \
    resume.ckpt_path=auto \
    loss.loss_weights.seq_loss=1.0 \
    loss.main_seq_loss_pocket_only=false \
    loss.main_potts_loss_pocket_only=false \
    loss.seq_loss.per_token_avg=false \
    loss.potts.per_token_avg=false \
    data.pdb_path="${PDB_PATH}" \
    data.train_metadata_path="${TRAIN_METADATA_PATH}" \
    data.runtime_index_path="${RUNTIME_INDEX_PATH}" \
    data.validation_ids_file="${VALIDATION_IDS_FILE}" \
    data.val_metadata_path="${VAL_METADATA_PATH}" \
    data.residue_cache_dir="${RESIDUE_CACHE_DIR}" \
    data.exclude_val_cluster=true \
    data.interface_only_training=false \
    data.samples_per_epoch=50000 \
    data.bml_center.small_molecule.min_contacting_protein_atom_ratio=1.0 \
    data.bml_center.small_molecule.exclude_maybe_covalently_linked_small_molecules=false \
    data.bml_context.enabled=true \
    data.bml_context.scheme=iterative_context_refinement \
    data.clustering.sampling_scheme=fixed_k \
    data.query_pn_unit_iids_only=true \
    "${INTERFACE_FILTER_OVERRIDE[@]}" \
    "${MONOMER_FILTER_OVERRIDE[@]}" \
    data.sampling_weights.alphas_interface.alpha_protein_protein=0.0 \
    data.sampling_weights.alphas_interface.alpha_protein_nuc=0.0 \
    data.sampling_weights.single_protein_context_weight=1.0 \
    data.sampling_weights.multi_protein_context_weight=0.0 \
    data.sampling_weights.ligand_cluster_col=null \
    data.featurizer_cfg.is_inference=false \
    data.featurizer_cfg.occupancy_threshold_protein_backbone=0.0 \
    data.featurizer_cfg.remove_unresolved_tokens=true \
    data.featurizer_cfg.max_tokens=768 \
    data.featurizer_cfg.max_atoms=4608 \
    data.featurizer_cfg.crop_center_cutoff_distance=30.0 \
    data.featurizer_cfg.crop_spatial_p_protein_monomer_chain=0.25 \
    data.featurizer_cfg.drop_prob_non_protein_chains=0.0 \
    data.featurizer_cfg.pocket_distance=5.0 \
    data.featurizer_cfg.pocket_n_min_ligand_atoms=1 \
    data.featurizer_cfg.asymmetric_noise=false \
    denoiser.mpnn.hidden_dim=128 \
    denoiser.mpnn.k_neighbors=48 \
    denoiser.mpnn.ligand_conditioning=true \
    denoiser.mpnn.token_features.protein_graph_rbf_type=ncacocb \
    denoiser.mpnn.token_features.ligand_atom_context_num=25 \
    denoiser.mpnn.potts.k_neighbors_potts=48 \
    denoiser.mpnn.dropout_p=0.2 \
    denoiser.mpnn.decoder_input_mode="${DECODER_INPUT_MODE}" \
    denoiser.mpnn.update_final_encoder_edge="${UPDATE_FINAL_ENCODER_EDGE}" \
    denoiser.mpnn.expansion_mode="${EXPANSION_MODE}" \
    denoiser.mpnn.lmpnn_module.context_edge_update="${CONTEXT_EDGE_UPDATE}" \
    denoiser.mpnn.lmpnn_module.update_final_context_processor_edge="${UPDATE_FINAL_CONTEXT_PROCESSOR_EDGE}" \
    denoiser.mpnn.use_context_skip_connection=false \
    denoiser.mpnn.use_mpnn_decoder=true \
    denoiser.mpnn.num_encoder_layers=2 \
    denoiser.mpnn.num_decoder_layers="${NUM_DECODER_LAYERS}" \
    denoiser.mpnn.lmpnn_module.num_context_feature_processor_layers=2 \
    denoiser.mpnn.lmpnn_module.num_context_feature_aggregator_layers=2 \
    denoiser.mpnn.use_potts_encoding=true \
    denoiser.mpnn.potts.multi_head.num_heads=2 \
    denoiser.mpnn.potts.multi_head.reduce=mean \
    denoiser.mpnn.potts.multi_head.adapter_hidden_dim=640 \
    denoiser.mpnn.potts.multi_head.shared_edge_dim=320 \
    denoiser.mpnn.sidechain_prediction.enabled=true \
    denoiser.mpnn.sidechain_prediction.num_chi=4 \
    denoiser.mpnn.sidechain_prediction.num_bins=72 \
    denoiser.mpnn.sidechain_prediction.use_node_norm=false \
    loss.loss_weights.sidechain_chi_loss=0.125 \
    loss.loss_weights.sidechain_chi_offset_loss=0.0625 \
    mask_selector.restype_masking_schedule=uniform_cosine_t \
    logging.log_every_n_steps=100 \
    logging.wandb_watch_freq=500 \
    checkpointing.save_latest_every_n_steps=2500 \
    trainer.check_val_every_n_epoch=2 \
    trainer.max_steps=200000 \
    "${COMMON_OVERRIDES[@]}" \
    "${SAMPLING_OVERRIDES[@]}" \
    "${FEATURE_OVERRIDES[@]}" \
    hydra.run.dir="${HYDRA_DIR}/${EXP_NAME}"
