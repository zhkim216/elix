#!/bin/bash
# Nativeval pocket-alignment smoke: sampling -> AF3 -> per-ligand role metrics.
# Submit through scripts/sherlock_scripts/jinho/setup/wrap_sbatch_in_container_elix.sh.
#SBATCH --job-name=native_pocket_smoke
#SBATCH -p possu,bioe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/users/zhkim216/experiment_result_analysis/260710_nativeval_pocket_alignment_smoke/slurm/native_pocket_smoke_%j.out
#SBATCH --error=/scratch/users/zhkim216/experiment_result_analysis/260710_nativeval_pocket_alignment_smoke/slurm/native_pocket_smoke_%j.err

set -euo pipefail

BASE_DIR="${PROJECT_ROOT:-/home/users/zhkim216/code/elix}"
SCRIPT_DIR="${BASE_DIR}/scripts/sherlock_scripts/jinho/application/validation"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/users/zhkim216/experiment_result_analysis/260710_nativeval_pocket_alignment_smoke}"
DENOISER_TRAIN_DIR="/scratch/users/zhkim216/out_dir/ligand_seq_des/elix_exp1_cfg0"
SAMPLING_INPUTS_CSV="${SCRIPT_DIR}/nativeval_pocket_alignment_smoke_sampling_inputs.csv"
PDB_NAME_LIST="${SCRIPT_DIR}/nativeval_pocket_alignment_smoke_pdbs.txt"
PDB_DIR="/scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/cifs"
STEP=32500
NUM_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
POTTS_SWEEPS="${POTTS_SWEEPS:-10}"
AF3_RECYCLES="${AF3_RECYCLES:-1}"
AF3_DIFFUSION_SAMPLES="${AF3_DIFFUSION_SAMPLES:-1}"
RUN_SEQ_DES="${RUN_SEQ_DES:-true}"
CALCULATE_METRICS_ONLY="${CALCULATE_METRICS_ONLY:-false}"
EXP_NAME="nativeval_pocket_alignment_smoke_exp1_cfg0_step${STEP}_seq1_r${AF3_RECYCLES}_d${AF3_DIFFUSION_SAMPLES}"
HYDRA_DIR="${OUTPUT_ROOT}/hydra/${EXP_NAME}"
RUN_DIR="${OUTPUT_ROOT}/${EXP_NAME}/step_32500_epoch_20"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 1; }
}

require_file "${SAMPLING_INPUTS_CSV}"
require_file "${PDB_NAME_LIST}"
require_file "${PDB_DIR}/9bks.cif"
require_file "${PDB_DIR}/8tnb.cif"
require_file "${DENOISER_TRAIN_DIR}/checkpoints/ema/sd-step32500-epoch20-ema0.99.ckpt"
require_file "${BASE_DIR}/alphafold3/src/alphafold3/run_alphafold.py"
command -v nvidia-smi >/dev/null

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/slurm" "${HYDRA_DIR}"
cd "${BASE_DIR}"
export PYTHONWARNINGS=ignore

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
MAJOR=${COMPUTE_CAP%%.*}
if [[ "${MAJOR}" -ge 8 ]]; then
  FLASH_ATTN_IMPL=triton
else
  FLASH_ATTN_IMPL=xla
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
fi

echo "Running nativeval pocket-alignment smoke: 9bks(A_1+B_1), 8tnb(A_1+C_1+D_1)"
echo "Output: ${RUN_DIR}"
python3 allatom_design/eval/sampling/run_elix.py \
  --config-path "${BASE_DIR}/allatom_design/configs/eval/sampling" \
  --config-name run_elix \
  wandb.no_wandb=true \
  debug=false \
  seed=0 \
  num_workers=${NUM_WORKERS} \
  protein_only=false \
  run_seq_des=${RUN_SEQ_DES} \
  exp_name=${EXP_NAME} \
  base_out_dir=${OUTPUT_ROOT} \
  model_cfg.model_name=elix_mpnn \
  model_cfg.denoiser_train_dir=${DENOISER_TRAIN_DIR} \
  model_cfg.ckpt_cfg.start_step=${STEP} \
  model_cfg.ckpt_cfg.end_step=${STEP} \
  model_cfg.ckpt_cfg.eval_every_n_ckpts=1 \
  model_cfg.ckpt_cfg.eval_last_ckpt=false \
  model_cfg.ckpt_cfg.use_ema=true \
  sampling_cfg.base_cfg_path="${BASE_DIR}/allatom_design/configs/seq_des/elix_mpnn_inference.yaml" \
  sampling_cfg.overrides.batch_size=1 \
  sampling_cfg.overrides.num_seqs_per_pdb=1 \
  sampling_cfg.overrides.num_workers=${NUM_WORKERS} \
  sampling_cfg.overrides.use_potts_sampling=true \
  sampling_cfg.overrides.potts_sampling_cfg.potts_sweeps=${POTTS_SWEEPS} \
  ++sampling_cfg.overrides.potts_sampling_cfg.potts_temperature=0.1 \
  sampling_cfg.overrides.potts_sampling_cfg.lcp_expand_edge_idx_fix=true \
  sampling_cfg.overrides.potts_sampling_cfg.rejection_step=false \
  sampling_cfg.overrides.potts_sampling_cfg.ensemble_conditioning.enabled=false \
  sampling_cfg.guidance.enabled=false \
  two_stage.enabled=false \
  sampling_inputs_csv=${SAMPLING_INPUTS_CSV} \
  pos_constraint_csv=null \
  pocket_cfg.n_min_ligand_atoms_for_seq_recovery=1 \
  input_sample_is_all_atom=true \
  input_sample_is_designed=false \
  load_from_cache=false \
  pdb_cfg.pdb_dir=${PDB_DIR} \
  pdb_cfg.pdb_name_list=${PDB_NAME_LIST} \
  pdb_cfg.pdb_name_ext=.cif \
  pdb_cfg.recursive=false \
  pdb_cfg.split_by_subfolder=false \
  pdb_cfg.sample_indices=null \
  pdb_cfg.sample_lengths=null \
  pdb_cfg.array_id=null \
  pdb_cfg.num_arrays=null \
  cif_cfg.save.include_bonds=false \
  featurizer_cfg.design.residue_cache_dir=/scratch/users/zhkim216/datasets/atomworks_cached_residue_data \
  struct_pred_cfg.evaluate_structure_prediction=true \
  struct_pred_cfg.evaluate_self_consistency=false \
  struct_pred_cfg.evaluate_docking_consistency=false \
  struct_pred_cfg.calculate_metrics_only=${CALCULATE_METRICS_ONLY} \
  struct_pred_cfg.af3.require_complete_predictions=true \
  struct_pred_cfg.af3.runner_path="${BASE_DIR}/alphafold3/src/alphafold3/run_alphafold.py" \
  struct_pred_cfg.af3.inference_config.base.flash_attention_implementation=${FLASH_ATTN_IMPL} \
  struct_pred_cfg.af3.inference_config.ss.num_recycles=${AF3_RECYCLES} \
  struct_pred_cfg.af3.inference_config.ss.num_diffusion_samples=${AF3_DIFFUSION_SAMPLES} \
  struct_pred_cfg.af3.inference_config.tc.num_recycles=${AF3_RECYCLES} \
  struct_pred_cfg.af3.inference_config.tc.num_diffusion_samples=${AF3_DIFFUSION_SAMPLES} \
  hydra.run.dir=${HYDRA_DIR}

python3 "${SCRIPT_DIR}/validate_nativeval_pocket_alignment_smoke.py" --run-dir "${RUN_DIR}"
echo "Nativeval pocket-alignment smoke completed: ${RUN_DIR}"
