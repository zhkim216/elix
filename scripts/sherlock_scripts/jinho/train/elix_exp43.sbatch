#!/usr/bin/env bash
# Submit with:
#   PROJECT_ROOT=/home/users/zhkim216/code/elix \
#     bash scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh \
#     scripts/sherlock_scripts/jinho/train/elix_exp43.sbatch
#
# Array map:
#   0: PP dense,     shared-atom dense,   noise 0.1
#   1: PP dense,     shared-atom dense,   noise 0.3
#   2: no PP update, shared-atom dense,   noise 0.1
#   3: no PP update, shared-atom dense,   noise 0.3
#   4: no PP update, no shared-atom update, noise 0.1
#   5: no PP update, no shared-atom update, noise 0.3
#
# With two GPUs and gradient accumulation 2, per-GPU batch size is derived as
# 32 / (2 * 2) = 8. All tasks start from scratch with seed 0.
#SBATCH --job-name=elix_exp43
#SBATCH --partition=possu
#SBATCH --array=0-5
#SBATCH --gres=gpu:2
#SBATCH --constraint=GPU_SKU:H100_SXM5
#SBATCH --cpus-per-task=8
#SBATCH --mem=164G
#SBATCH --time=1-0:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/users/zhkim216/job_output/train_seq_denoiser/elix_exp43_%A_%a.out
#SBATCH --error=/scratch/users/zhkim216/job_output/train_seq_denoiser/elix_exp43_%A_%a.err

set -euo pipefail

BASE_DIR=/home/users/zhkim216/code/elix
EXPECTED_VENV=/scratch/users/zhkim216/envs/uv/elix-torch280
OUT_DIR=/scratch/users/zhkim216/out_dir
JOB_OUTPUT_DIR=/scratch/users/zhkim216/job_output/train_seq_denoiser
HYDRA_ROOT=/scratch/users/zhkim216/hydra_outputs/train_seq_denoiser/elix_exp43

PDB_PATH=/scratch/users/zhkim216/datasets/atomworks_pdb_full_v9
TRAIN_METADATA_PATH=${PDB_PATH}/metadata_seq_clustered_03_finalized.parquet
RUNTIME_INDEX_PATH=${PDB_PATH}/runtime_indices/elix_exp41_exp42_fixedk3.arrow
VALIDATION_IDS_FILE=/scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/nativeval.txt
VAL_METADATA_PATH=/scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/metadata_for_training_nativeval.parquet
RESIDUE_CACHE_DIR=/scratch/users/zhkim216/datasets/cache_residue_runs/ccd_100conf_torch280_20260716/atomworks_cached_residue_data

ACCUMULATE_GRAD_BATCHES=2
EFFECTIVE_GLOBAL_BATCH_SIZE=32
NUM_WORKERS=4
PREFETCH_FACTOR=1
TRAIN_SEED=0
MAX_TOKENS=768
MAX_ATOMS=4608
MAX_STEPS=200000
WANDB_ID=zhkim216
PROJECT=ligand_seq_des
GROUP=training

ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [[ ! ${ARRAY_TASK_ID} =~ ^[0-5]$ ]]; then
  echo "Expected SLURM_ARRAY_TASK_ID in 0..5, got ${ARRAY_TASK_ID}" >&2
  exit 2
fi
IDX=$((10#${ARRAY_TASK_ID}))

case ${IDX} in
  0|1)
    CONDITION_NAME=pp_dense_shared_dense
    CONDITION_OVERRIDES=(
      'denoiser.mpnn.shared_atom_triangle.mixing_encoder_layers=[0,1]'
      'denoiser.mpnn.protein_pair_triangle.mixing_encoder_layers=[0,1]'
      denoiser.mpnn.lmpnn_module.update_final_context_aggregator_edge=true
    )
    ;;
  2|3)
    CONDITION_NAME=no_pp_shared_dense
    CONDITION_OVERRIDES=(
      'denoiser.mpnn.shared_atom_triangle.mixing_encoder_layers=[0,1]'
      'denoiser.mpnn.protein_pair_triangle.mixing_encoder_layers=[]'
      denoiser.mpnn.lmpnn_module.update_final_context_aggregator_edge=true
    )
    ;;
  4|5)
    CONDITION_NAME=no_pp_no_shared
    CONDITION_OVERRIDES=(
      'denoiser.mpnn.shared_atom_triangle.mixing_encoder_layers=[]'
      'denoiser.mpnn.protein_pair_triangle.mixing_encoder_layers=[]'
      denoiser.mpnn.lmpnn_module.update_final_context_aggregator_edge=false
    )
    ;;
esac

case ${IDX} in
  0|2|4)
    TRAINING_STRUCTURE_NOISE=0.1
    NOISE_NAME=noise0p1
    ;;
  1|3|5)
    TRAINING_STRUCTURE_NOISE=0.3
    NOISE_NAME=noise0p3
    ;;
esac

EXP_NAME="elix_exp43_cfg${IDX}_${CONDITION_NAME}_${NOISE_NAME}_seed${TRAIN_SEED}"
WANDB_RUN_ID="${EXP_NAME}_job${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
HYDRA_DIR=${HYDRA_ROOT}/${EXP_NAME}
export TORCHINDUCTOR_CACHE_DIR="/scratch/users/zhkim216/cache/inductor_cache_${EXP_NAME}"

if [[ "$(python3 -c 'import sys; print(sys.prefix)')" != ${EXPECTED_VENV} ]]; then
  echo "Expected Torch 2.8 environment ${EXPECTED_VENV}" >&2
  echo "Submit through wrap_sbatch_in_container_elix.sh" >&2
  exit 2
fi

for required in \
  "${TRAIN_METADATA_PATH}" \
  "${RUNTIME_INDEX_PATH}" \
  "${VALIDATION_IDS_FILE}" \
  "${VAL_METADATA_PATH}" \
  "${RESIDUE_CACHE_DIR}"; do
  if [[ ! -e ${required} ]]; then
    echo "Required input is missing: ${required}" >&2
    exit 2
  fi
done

if [[ ${SLURM_GPUS_ON_NODE:-} =~ ^[0-9]+$ ]]; then
  GPU_COUNT=${SLURM_GPUS_ON_NODE}
else
  GPU_COUNT=$(python3 -c 'import torch; print(torch.cuda.device_count())')
fi
if (( GPU_COUNT < 1 )); then
  echo "No visible GPU was detected" >&2
  exit 2
fi

batch_denominator=$((GPU_COUNT * ACCUMULATE_GRAD_BATCHES))
if (( EFFECTIVE_GLOBAL_BATCH_SIZE % batch_denominator != 0 )); then
  echo "Effective global batch ${EFFECTIVE_GLOBAL_BATCH_SIZE} is not divisible by GPU count ${GPU_COUNT} * accumulation ${ACCUMULATE_GRAD_BATCHES}" >&2
  exit 2
fi
BATCH_SIZE=$((EFFECTIVE_GLOBAL_BATCH_SIZE / batch_denominator))
calculated_global_batch=$((BATCH_SIZE * batch_denominator))

mkdir -p \
  "${JOB_OUTPUT_DIR}" \
  "${OUT_DIR}" \
  "${HYDRA_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}"
cd "${BASE_DIR}"

WANDB_OVERRIDES=(
  "wandb.wandb_id=${WANDB_ID}"
  "wandb.project=${PROJECT}"
  "wandb.group=${GROUP}"
  wandb.no_wandb=false
  "wandb.run_id=${WANDB_RUN_ID}"
  wandb.resume=allow
)

TRAIN_OVERRIDES=(
  "train.seed=${TRAIN_SEED}"
  train.debug=false
  "train.batch_size=${BATCH_SIZE}"
  train.compile.compile_model=true
  train.compile.fullgraph=false
  train.compile.dynamic=false
  ++trainer.accelerator=gpu
  "++trainer.devices=${GPU_COUNT}"
  ++trainer.num_nodes=1
  trainer.strategy=ddp_find_unused_parameters_true
  trainer.precision=bf16-mixed
  "trainer.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}"
  "trainer.max_steps=${MAX_STEPS}"
)

LOSS_OVERRIDES=(
  loss.loss_weights.seq_loss=1.0
  loss.loss_weights.potts_composite_loss=1.0
  loss.main_seq_loss_pocket_only=false
  loss.main_potts_loss_pocket_only=false
  loss.seq_loss.per_token_avg=false
  loss.potts.per_token_avg=false
  loss.loss_weights.sidechain_chi_loss=0.125
  loss.loss_weights.sidechain_chi_offset_loss=0.0625
)

DATA_OVERRIDES=(
  "data.pdb_path=${PDB_PATH}"
  "data.train_metadata_path=${TRAIN_METADATA_PATH}"
  "data.runtime_index_path=${RUNTIME_INDEX_PATH}"
  "data.validation_ids_file=${VALIDATION_IDS_FILE}"
  "data.val_metadata_path=${VAL_METADATA_PATH}"
  "data.residue_cache_dir=${RESIDUE_CACHE_DIR}"
  "data.random_seed=${TRAIN_SEED}"
  "data.prefetch_factor=${PREFETCH_FACTOR}"
  data.cluster_id_col=q_pn_unit_cluster_id
  data.val_exclusion_cluster_id_col=q_pn_unit_cluster_id
  data.exclude_val_cluster=true
  data.interface_only_training=false
  data.samples_per_epoch=50000
  data.bml_center.small_molecule.min_contacting_protein_atom_ratio=1.0
  data.bml_center.small_molecule.exclude_maybe_covalently_linked_small_molecules=false
  data.bml_context.enabled=true
  data.bml_context.scheme=iterative_context_refinement
  data.clustering.sampling_scheme=fixed_k
  data.query_pn_unit_iids_only=true
)

INTERFACE_FILTER_OVERRIDE=(
  'data.train_filters.interface_filter.2=["n_prot == 1"]'
)

MONOMER_FILTER_OVERRIDE=(
  'data.train_filters.protein_monomer_chain_filter.0="q_pn_unit_is_protein and resolution < 9.0 and 20 <= q_pn_unit_num_resolved_residues < 2048"'
)

DATA_SAMPLING_WEIGHTS_OVERRIDES=(
  data.sampling_weights.alphas_interface.alpha_protein_protein=0.0
  data.sampling_weights.alphas_interface.alpha_protein_small_molecule=8.0
  data.sampling_weights.alphas_interface.alpha_protein_metal=1.0
  data.sampling_weights.alphas_interface.alpha_protein_nuc=0.0
  data.sampling_weights.alphas_interface.alpha_protein_nuc_lig=1.0
  data.sampling_weights.alphas_interface.alpha_protein_peptide=1.0
  data.sampling_weights.single_protein_context_weight=1.0
  data.sampling_weights.multi_protein_context_weight=0.0
  data.sampling_weights.fixed_k=3.0
  data.sampling_weights.ligand_cluster_col=null
)

DATA_FEATURIZER_OVERRIDES=(
  data.featurizer_cfg.is_inference=false
  data.featurizer_cfg.occupancy_threshold_protein_backbone=0.0
  data.featurizer_cfg.remove_unresolved_tokens=true
  "data.featurizer_cfg.training_structure_noise=${TRAINING_STRUCTURE_NOISE}"
  "data.featurizer_cfg.max_tokens=${MAX_TOKENS}"
  "data.featurizer_cfg.max_atoms=${MAX_ATOMS}"
  data.featurizer_cfg.crop_center_cutoff_distance=30.0
  data.featurizer_cfg.crop_spatial_p_protein_monomer_chain=0.25
  data.featurizer_cfg.drop_prob_non_protein_chains=0.0
  data.featurizer_cfg.pocket_distance=5.0
  data.featurizer_cfg.pocket_n_min_ligand_atoms=1
  data.featurizer_cfg.asymmetric_noise=false
)

TOKEN_FEATURE_OVERRIDES=(
  denoiser.mpnn.token_features.protein_graph_rbf_type=ncacocb
  denoiser.mpnn.token_features.ligand_atom_context_num=25
  denoiser.mpnn.token_features.use_ligand_f_block_features=true
  denoiser.mpnn.token_features.use_ligand_asinh_formal_charge=true
  denoiser.mpnn.token_features.use_ligand_cached_rdkit_chirality=true
  denoiser.mpnn.token_features.use_ligand_bond_order=true
  denoiser.mpnn.token_features.use_token_bonds=true
  denoiser.mpnn.token_features.add_hydrogenbond_feature=true
  denoiser.mpnn.token_features.use_ligand_aromatic_atom_feature=true
  denoiser.mpnn.token_features.use_ligand_aromatic_edge_feature=true
)

MPNN_OVERRIDES=(
  denoiser.mpnn.hidden_dim=128
  denoiser.mpnn.k_neighbors=48
  denoiser.mpnn.num_encoder_layers=2
  denoiser.mpnn.num_decoder_layers=2
  denoiser.mpnn.use_mpnn_decoder=true
  denoiser.mpnn.decoder_input_mode=legacy_node_add
  denoiser.mpnn.update_final_encoder_edge=true
  denoiser.mpnn.use_potts_encoding=true
  denoiser.mpnn.dropout_p=0.2
  denoiser.mpnn.ligand_conditioning=true
  denoiser.mpnn.use_context_skip_connection=false
  denoiser.mpnn.expansion_mode=node_concat
  denoiser.mpnn.lmpnn_module.num_context_feature_processor_layers=2
  denoiser.mpnn.lmpnn_module.num_context_feature_aggregator_layers=2
  denoiser.mpnn.lmpnn_module.context_edge_update=true
  ++denoiser.mpnn.lmpnn_module.context_pair_update=true
  ++denoiser.mpnn.lmpnn_module.protein_context_pair_update=true
  denoiser.mpnn.lmpnn_module.update_final_context_processor_edge=false
)

TRIANGLE_UPDATE_OVERRIDES=(
  denoiser.mpnn.shared_atom_triangle.hidden_dim=128
  denoiser.mpnn.shared_atom_triangle.edge_chunk_size=48
  denoiser.mpnn.protein_pair_triangle.hidden_dim=128
  denoiser.mpnn.protein_pair_triangle.target_chunk_size=48
)

POTTS_OVERRIDES=(
  denoiser.mpnn.potts.k_neighbors_potts=48
  denoiser.mpnn.potts.parameterization=factor
  denoiser.mpnn.potts.norm_potts_inputs=false
)

SIDECHAIN_OVERRIDES=(
  denoiser.mpnn.sidechain_prediction.enabled=true
  denoiser.mpnn.sidechain_prediction.num_chi=4
  denoiser.mpnn.sidechain_prediction.num_bins=72
  denoiser.mpnn.sidechain_prediction.use_node_norm=false
)

echo "Starting ${EXP_NAME}"
echo "  wandb_run_id: ${WANDB_RUN_ID}"
echo "  condition: ${CONDITION_NAME}"
echo "  training_structure_noise: ${TRAINING_STRUCTURE_NOISE}"
echo "  seed: ${TRAIN_SEED}"
echo "  gpu_count: ${GPU_COUNT}"
echo "  batch_size_per_gpu: ${BATCH_SIZE}"
echo "  accumulate_grad_batches: ${ACCUMULATE_GRAD_BATCHES}"
echo "  effective_global_batch_size: ${calculated_global_batch}"
echo "  max_tokens: ${MAX_TOKENS}"
echo "  max_atoms: ${MAX_ATOMS}"
printf '  condition_override: %s\n' "${CONDITION_OVERRIDES[@]}"

export MASTER_ADDR=127.0.0.1
unset MASTER_PORT
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

DEBUG= HYDRA_FULL_ERROR=1 WANDB__SERVICE_WAIT=300 \
python3 allatom_design/train_seq_denoiser.py \
  --config-path configs/seq_denoiser \
  --config-name elix \
  "${WANDB_OVERRIDES[@]}" \
  "out_dir=${OUT_DIR}" \
  "exp_name=${EXP_NAME}" \
  "num_workers=${NUM_WORKERS}" \
  resume.use_current_cfg=true \
  resume.ckpt_path=null \
  finetuning.enabled=false \
  finetuning.ckpt_path=null \
  "${TRAIN_OVERRIDES[@]}" \
  "${LOSS_OVERRIDES[@]}" \
  "${DATA_OVERRIDES[@]}" \
  "${INTERFACE_FILTER_OVERRIDE[@]}" \
  "${MONOMER_FILTER_OVERRIDE[@]}" \
  "${DATA_SAMPLING_WEIGHTS_OVERRIDES[@]}" \
  "${DATA_FEATURIZER_OVERRIDES[@]}" \
  "${TOKEN_FEATURE_OVERRIDES[@]}" \
  "${MPNN_OVERRIDES[@]}" \
  "${TRIANGLE_UPDATE_OVERRIDES[@]}" \
  "${POTTS_OVERRIDES[@]}" \
  "${SIDECHAIN_OVERRIDES[@]}" \
  eval.scn_context_ratio=0.0 \
  mask_selector.restype_masking_schedule=uniform_cosine_t \
  logging.log_every_n_steps=100 \
  logging.wandb_watch_freq=500 \
  checkpointing.save_latest_every_n_steps=2500 \
  trainer.check_val_every_n_epoch=2 \
  "${CONDITION_OVERRIDES[@]}" \
  "hydra.run.dir=${HYDRA_DIR}"
