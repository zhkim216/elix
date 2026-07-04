from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
import yaml
from tqdm import tqdm

from allatom_design.eval.utils.input_files import get_pdb_files
from allatom_design.eval.run_logging import wandb_setup
from allatom_design.eval.structure_prediction.af3_input_utils import (
    filter_sample_paths_by_sampling_inputs,
    load_af3_eval_sample,
    load_sampling_inputs_csv,
)
from allatom_design.eval.structure_prediction.af3_evaluation import (
    evaluate_af3_self_consistency,
)


@hydra.main(config_path="../../configs/eval/structure_prediction", config_name="run_sc_eval_af3", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Run AF3 self-consistency evaluation on pre-designed samples.
    Assumes samples are already designed (e.g. by caliby) and contain ligands.
    """
    ###########################################################
    # Phase 0: Basic setup
    ###########################################################
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Debug mode adjustments
    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"
    
    # Setup wandb logging
    log_dir = Path(wandb_setup(base_out_dir=cfg.base_out_dir, exp_name=cfg.exp_name, cfg_dict=cfg_dict, **cfg.wandb))
    
    # Preserve config
    with open(Path(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg_dict, f)
    
    # Compute CSV suffix for array jobs
    array_id = cfg.pdb_cfg.get("array_id", None)
    csv_suffix = f"_array_{array_id}" if array_id is not None else ""
    
    ###########################################################
    # Phase 1: Load designed samples and extract chain info
    ###########################################################
    print("\n" + "="*80)
    print("Phase 1: Loading designed samples")
    print("="*80 + "\n")
    
    # Get CIF file paths
    sample_paths = get_pdb_files(**cfg.pdb_cfg)

    sampling_inputs_df = load_sampling_inputs_csv(cfg.get("sampling_inputs_csv", None))
    if sampling_inputs_df is not None:
        print(f"Loaded sampling inputs: {len(sampling_inputs_df)} entries from {cfg.sampling_inputs_csv}")

        before_filter = len(sample_paths)
        sample_paths = filter_sample_paths_by_sampling_inputs(sample_paths, sampling_inputs_df)
        print(f"Filtered to {len(sample_paths)} samples matching sampling inputs from {before_filter}")

    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]

    # Select parsing/preprocessing config based on input type
    if cfg.input_sample_is_designed:
        cif_parse_cfg = cfg.cif_cfg.parse.designed_samples
        preprocess_cfg_input = cfg.preprocess_cfg.designed_samples
    else:
        cif_parse_cfg = cfg.cif_cfg.parse.native_samples
        preprocess_cfg_input = cfg.preprocess_cfg.native_samples

    # Build sample_dict in evaluate_af3_self_consistency format
    sample_dict = {}
    desc = "Loading designed samples" if cfg.input_sample_is_designed else "Loading native samples"
    for sample_path in tqdm(sample_paths, desc=desc):
        sample_id = Path(sample_path).stem

        try:
            example = load_af3_eval_sample(
                sample_path=sample_path,
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg_input,
                featurizer_cfg=cfg.featurizer_cfg.prepare_designed_samples,
                sample_is_designed=cfg.input_sample_is_designed,
                sampling_inputs_df=sampling_inputs_df,
            )
        except Exception as e:
            print(f"Failed to load {sample_id}: {e}")
            continue
        
        atom_array = example["atom_array"]
        pdb_chain_info = example["pdb_chain_info"]

        if not pdb_chain_info["protein_pn_unit_iids"]:
            print(f"Warning: No protein chains found in {sample_id}, skipping")
            continue

        # Build entry in evaluate_af3_self_consistency format
        # Each designed sample is both the "input" and the "designed" sample
        sample_dict[sample_id] = {
            "input_sample_path": sample_path,
            "input_sample_id": sample_id,
            "designed_sample_id": [sample_id],
            "designed_sample_atom_array": [atom_array],
            "pdb_chain_info": pdb_chain_info,
        }
    
    print(f"\nSuccessfully loaded {len(sample_dict)} samples")
    if len(sample_dict) == 0:
        raise RuntimeError("No samples were loaded for AF3 self-consistency evaluation")
    
    ###########################################################
    # Phase 2: AF3 Evaluation
    ###########################################################
    if cfg.struct_pred_cfg.evaluate_self_consistency:
        print("\n" + "="*80)
        print("Phase 2: AF3 Self-Consistency Evaluation")
        print("="*80 + "\n")
        
        evaluate_af3_self_consistency(
            sample_dict=sample_dict,
            out_dir=log_dir,
            struct_pred_cfg=cfg.struct_pred_cfg,
            cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
            preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
            featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
            pocket_cfg=cfg.pocket_cfg,
            no_wandb=cfg.wandb.no_wandb,
            ckpt_info=None,
            calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only,
            csv_suffix=csv_suffix,
            input_sample_is_designed=cfg.input_sample_is_designed,
        )
    
    print("\n" + "="*80)
    print("All phases complete!")
    print(f"Results saved to {log_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
