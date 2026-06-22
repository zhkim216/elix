import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.structure_prediction.af3_json import make_af3_json
from allatom_design.eval.structure_prediction.af3_runner import (
    find_pred_sample_path_af3,
    run_af3_single_sequence,
    run_af3_template_conditioned,
)


def _ligand_metric_values_from_chain_info(
    pdb_chain_info: dict,
    keys: tuple[str, ...],
    ligand_pn_unit_iids: list[str],
) -> list[str | None] | None:
    def _clean_metric_value(value: object) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return text

    for key in keys:
        values = pdb_chain_info.get(key)
        if values is None:
            continue
        if isinstance(values, dict):
            result = [_clean_metric_value(values.get(str(iid))) for iid in ligand_pn_unit_iids]
            return result if any(value is not None for value in result) else None
        if isinstance(values, str):
            value = _clean_metric_value(values)
            return [value] if value is not None and len(ligand_pn_unit_iids) == 1 else None
        result = [_clean_metric_value(value) for value in values]
        return result if any(value is not None for value in result) else None
    return None


# ============================================================================
# AF3 Evaluation Functions
# ============================================================================

def evaluate_af3_self_consistency(sample_dict: dict = None,
                                  out_dir: Path = None,
                                  struct_pred_cfg: DictConfig = None,
                                  cif_parse_cfg: DictConfig = None,
                                  preprocess_cfg: DictConfig = None,
                                  featurizer_cfg: DictConfig = None,
                                  pocket_cfg: DictConfig = None,
                                  ckpt_info: dict = None,
                                  no_wandb: bool = False,
                                  calculate_metrics_only: bool = False,
                                  csv_suffix: str = "",
                                  input_sample_is_designed: bool = True,
                                  free_atom_arrays_progressively: bool = False) -> None:
    """
    Run AF3 self-consistency and docking evaluation.

    Args:
        sample_id_list: List of sample IDs.
        pdb_id_list: List of PDB IDs.
        sample_atom_array_list: List of sample atom arrays.
        pdb_chain_info: PDB chain info dictionary.
        out_dir: Output directory.
        cfg: Configuration object.
        ckpt_info: Checkpoint info (optional, for wandb logging).
    """
    # Import here to keep AF3 prediction featurization optional until metrics are requested.
    from allatom_design.eval.structure_prediction.inputs import prepare_af3_prediction
    from allatom_design.eval.metrics import (
        compute_self_consistency_metrics_atomarray,
        compute_docking_metrics_atomarray
    )

    # Make json input directory
    af3_ss_input_dir = Path(out_dir, "af3_ss_inputs")
    af3_ss_input_dir.mkdir(parents=True, exist_ok=True)

    # Make a directory for af3 single-sequence prediction outputs
    af3_ss_pred_dir = Path(out_dir, "af3_ss_preds")
    af3_ss_pred_dir.mkdir(parents=True, exist_ok=True)

    print("Creating AF3 JSON input files...")

    sample_dict = make_af3_json(
        af3_ss_input_dir=af3_ss_input_dir,
        af3_tc_input_dir=None,
        sample_dict=sample_dict,
        metadata=None,
        json_config=struct_pred_cfg.af3.json_config
    )


    # Run AF3 self-consistency and docking evaluation
    af3_runner_path = struct_pred_cfg.af3.runner_path
    af3_inference_config = struct_pred_cfg.af3.inference_config

    designed_sample_id_to_per_pred_sc_metrics = {}
    designed_sample_id_to_per_pred_docking_metrics = {}

    print("\n" + "="*80)
    print("Running AF3 Self-Consistency Evaluation")
    print("="*80 + "\n")

    for input_sample_id in tqdm(sample_dict.keys(), desc="AF3 predictions"):
        subsample_dict = sample_dict[input_sample_id]

        for dsidx, designed_sample_id in enumerate(subsample_dict['designed_sample_id']):
            # Initialize metrics dict for this designed_sample_id (with input_sample_id for reverse lookup)
            designed_sample_id_to_per_pred_sc_metrics[designed_sample_id] = {"input_sample_id": input_sample_id}
            designed_sample_id_to_per_pred_docking_metrics[designed_sample_id] = {"input_sample_id": input_sample_id}

            designed_sample_atom_array = subsample_dict['designed_sample_atom_array'][dsidx]
            reference_sample_atom_array = subsample_dict.get('reference_sample_atom_array', designed_sample_atom_array)
            if isinstance(reference_sample_atom_array, list):
                reference_sample_atom_array = reference_sample_atom_array[dsidx]
            pdb_chain_info = subsample_dict['pdb_chain_info']
            ss_json_path = subsample_dict['af3_ss_json_paths'][dsidx]

            # Get protein and ligand chain ids, because AF3 expects chain ids, not chain iids
            protein_pn_unit_iids = pdb_chain_info['protein_pn_unit_iids']
            ligand_pn_unit_iids = pdb_chain_info['ligand_pn_unit_iids']
            ligand_ccd_codes = pdb_chain_info.get('ligand_ccd_codes', [])
            ligand_smiles = _ligand_metric_values_from_chain_info(
                pdb_chain_info,
                ("ligand_smiles", "ligand_smiles_by_iid"),
                ligand_pn_unit_iids,
            )
            reference_ligand_pn_unit_iids = _ligand_metric_values_from_chain_info(
                pdb_chain_info,
                (
                    "reference_ligand_pn_unit_iids",
                    "original_ligand_pn_unit_iids",
                    "reference_ligand_pn_unit_iids_by_iid",
                ),
                ligand_pn_unit_iids,
            )

            if not calculate_metrics_only:
                # Run AF3 single-sequence prediction
                try:
                    run_af3_single_sequence(str(ss_json_path), str(af3_ss_pred_dir),
                                            runner_path=af3_runner_path,
                                            inference_config=af3_inference_config)
                except Exception as e:
                    print(f"AF3 single sequence prediction failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}: {e}")
                    continue
                gc.collect()

            _, pred_ss_sample_paths = find_pred_sample_path_af3(out_dir=str(af3_ss_pred_dir),
                                                                job_name=designed_sample_id)

            if len(pred_ss_sample_paths) == 0:
                print(f"No AF3 predicted structure found for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}")
                continue

            for pred_idx, pred_ss_sample_path in enumerate(pred_ss_sample_paths):
                pred_example = None
                pred_atom_array = None
                try:
                    try:
                        pred_example = prepare_af3_prediction(
                            pdb_path=pred_ss_sample_path,
                            cif_parse_cfg=cif_parse_cfg,
                            preprocess_cfg=preprocess_cfg,
                            featurizer_cfg=featurizer_cfg,
                        )

                        pred_atom_array = pred_example["atom_array"]
                        per_pred_sc_metrics = compute_self_consistency_metrics_atomarray(
                            pred_atom_array=pred_atom_array,
                            sample_atom_array=designed_sample_atom_array,
                            pred_sample_path=pred_ss_sample_path,
                        )

                    except Exception as e:
                        print(f"Self-consistency metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                        continue
                    else:
                        designed_sample_id_to_per_pred_sc_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_sc_metrics

                    if ligand_pn_unit_iids:
                        try:
                            per_pred_docking_metrics = compute_docking_metrics_atomarray(
                                pred_atom_array=pred_atom_array,
                                sample_atom_array=designed_sample_atom_array,
                                reference_atom_array=reference_sample_atom_array,
                                pred_sample_path=pred_ss_sample_path,
                                pocket_distance_for_docking_metrics=pocket_cfg.pocket_distance_for_docking_metrics,
                                receptor_pn_unit_iids=protein_pn_unit_iids,
                                ligand_pn_unit_iids=ligand_pn_unit_iids,
                                ligand_ccd_codes=ligand_ccd_codes,
                                ligand_smiles=ligand_smiles,
                                reference_ligand_pn_unit_iids=reference_ligand_pn_unit_iids,
                                ref_sample_is_designed=input_sample_is_designed,
                            )

                        except Exception as e:
                            print(f"Docking metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                            continue
                        else:
                            designed_sample_id_to_per_pred_docking_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_docking_metrics
                finally:
                    del pred_example, pred_atom_array

        # Free memory after processing each input sample
        if free_atom_arrays_progressively:
            subsample_dict.pop('designed_sample_atom_array', None)
        gc.collect()

    # Aggregate best metrics per designed_sample_id (best diffusion sample)
    designed_sample_id_best_sc_metrics = _aggregate_best_sc_metrics_per_designed_sample(designed_sample_id_to_per_pred_sc_metrics)
    designed_sample_id_best_docking_metrics = _aggregate_best_docking_metrics_per_designed_sample(designed_sample_id_to_per_pred_docking_metrics)

    # Aggregate best metrics per input_sample_id (best designed sample)
    input_sample_id_best_sc_metrics = _aggregate_best_sc_metrics_per_input_sample(designed_sample_id_best_sc_metrics)
    input_sample_id_best_docking_metrics = _aggregate_best_docking_metrics_per_input_sample(designed_sample_id_best_docking_metrics)

    # Save results
    _save_metrics_results(
        out_dir=out_dir,
        designed_sample_id_to_per_pred_sc_metrics=designed_sample_id_to_per_pred_sc_metrics,
        designed_sample_id_to_per_pred_docking_metrics=designed_sample_id_to_per_pred_docking_metrics,
        designed_sample_id_best_sc_metrics=designed_sample_id_best_sc_metrics,
        designed_sample_id_best_docking_metrics=designed_sample_id_best_docking_metrics,
        input_sample_id_best_sc_metrics=input_sample_id_best_sc_metrics,
        input_sample_id_best_docking_metrics=input_sample_id_best_docking_metrics,
        no_wandb=no_wandb,
        ckpt_info=ckpt_info,
        csv_suffix=csv_suffix
    )

    print("\n" + "="*80)
    print("AF3 Self-Consistency and Docking Evaluation Complete")
    print(f"Results saved to {out_dir}")
    print("="*80 + "\n")


def evaluate_af3_docking_consistency(sample_dict: dict = None,
                                     out_dir: Path = None,
                                     struct_pred_cfg: DictConfig = None,
                                     cif_parse_cfg: DictConfig = None,
                                     preprocess_cfg: DictConfig = None,
                                     featurizer_cfg: DictConfig = None,
                                     pocket_cfg: DictConfig = None,
                                     ckpt_info: dict = None,
                                     no_wandb: bool = False,
                                     calculate_metrics_only: bool = False,
                                     csv_suffix: str = "",
                                     input_sample_is_designed: bool = True,
                                     free_atom_arrays_progressively: bool = False) -> None:
    """
    Run AF3 template-conditioned docking consistency evaluation.

    Uses template-conditioned AF3 predictions (designed backbone as template)
    to evaluate both self-consistency and docking metrics against the designed sample.

    Args:
        sample_dict: Dictionary of sample data (must contain 'designed_sample_path_for_af3_tc').
        out_dir: Output directory.
        struct_pred_cfg: Structure prediction configuration.
        cif_parse_cfg: CIF parsing configuration for AF3 predictions.
        preprocess_cfg: Preprocessing configuration for AF3 predictions.
        featurizer_cfg: Featurizer configuration for AF3 predictions.
        pocket_cfg: Pocket configuration for docking metrics.
        ckpt_info: Checkpoint info (optional, for wandb logging).
        no_wandb: If True, disable wandb logging.
        calculate_metrics_only: If True, skip AF3 prediction and only compute metrics.
        csv_suffix: Optional suffix for CSV filenames (e.g. "_array_0" for array jobs).
    """
    # Import here to keep AF3 prediction featurization optional until metrics are requested.
    from allatom_design.eval.structure_prediction.inputs import prepare_af3_prediction
    from allatom_design.eval.metrics import (
        compute_self_consistency_metrics_atomarray,
        compute_docking_metrics_atomarray,
    )

    # Make JSON input directories
    af3_ss_input_dir = Path(out_dir, "af3_ss_inputs")  # needed by make_af3_json (always creates SS JSONs)
    af3_ss_input_dir.mkdir(parents=True, exist_ok=True)
    af3_tc_input_dir = Path(out_dir, "af3_tc_inputs")
    af3_tc_input_dir.mkdir(parents=True, exist_ok=True)

    # Make a directory for af3 template-conditioned prediction outputs
    af3_tc_pred_dir = Path(out_dir, "af3_tc_preds")
    af3_tc_pred_dir.mkdir(parents=True, exist_ok=True)

    print("Creating AF3 JSON input files (template-conditioned)...")

    sample_dict = make_af3_json(
        af3_ss_input_dir=af3_ss_input_dir,
        af3_tc_input_dir=af3_tc_input_dir,
        sample_dict=sample_dict,
        metadata=None,
        json_config=struct_pred_cfg.af3.json_config,
        make_tc_input=True,
    )


    # Run AF3 template-conditioned docking evaluation
    af3_runner_path = struct_pred_cfg.af3.runner_path
    af3_inference_config = struct_pred_cfg.af3.inference_config

    designed_sample_id_to_per_pred_sc_metrics = {}
    designed_sample_id_to_per_pred_docking_metrics = {}

    print("\n" + "="*80)
    print("Running AF3 Docking Consistency Evaluation (Template-Conditioned)")
    print("="*80 + "\n")

    for input_sample_id in tqdm(sample_dict.keys(), desc="AF3 TC predictions"):
        subsample_dict = sample_dict[input_sample_id]

        for dsidx, designed_sample_id in enumerate(subsample_dict['designed_sample_id']):
            # Initialize metrics dict for this designed_sample_id (with input_sample_id for reverse lookup)
            designed_sample_id_to_per_pred_sc_metrics[designed_sample_id] = {"input_sample_id": input_sample_id}
            designed_sample_id_to_per_pred_docking_metrics[designed_sample_id] = {"input_sample_id": input_sample_id}

            designed_sample_atom_array = subsample_dict['designed_sample_atom_array'][dsidx]
            reference_sample_atom_array = subsample_dict.get('reference_sample_atom_array', designed_sample_atom_array)
            if isinstance(reference_sample_atom_array, list):
                reference_sample_atom_array = reference_sample_atom_array[dsidx]
            pdb_chain_info = subsample_dict['pdb_chain_info']
            tc_json_path = subsample_dict['af3_tc_json_paths'][dsidx]

            # Get protein and ligand chain ids
            protein_pn_unit_iids = pdb_chain_info['protein_pn_unit_iids']
            ligand_pn_unit_iids = pdb_chain_info['ligand_pn_unit_iids']
            ligand_ccd_codes = pdb_chain_info.get('ligand_ccd_codes', [])
            ligand_smiles = _ligand_metric_values_from_chain_info(
                pdb_chain_info,
                ("ligand_smiles", "ligand_smiles_by_iid"),
                ligand_pn_unit_iids,
            )
            reference_ligand_pn_unit_iids = _ligand_metric_values_from_chain_info(
                pdb_chain_info,
                (
                    "reference_ligand_pn_unit_iids",
                    "original_ligand_pn_unit_iids",
                    "reference_ligand_pn_unit_iids_by_iid",
                ),
                ligand_pn_unit_iids,
            )

            if not calculate_metrics_only:
                # Run AF3 template-conditioned prediction
                try:
                    run_af3_template_conditioned(str(tc_json_path), str(af3_tc_pred_dir),
                                            runner_path=af3_runner_path,
                                            inference_config=af3_inference_config)
                except Exception as e:
                    print(f"AF3 template-conditioned prediction failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}: {e}")
                    continue
                gc.collect()

            _, pred_tc_sample_paths = find_pred_sample_path_af3(out_dir=str(af3_tc_pred_dir),
                                                                job_name=designed_sample_id)

            if len(pred_tc_sample_paths) == 0:
                print(f"No AF3 TC predicted structure found for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}")
                continue

            for pred_idx, pred_tc_sample_path in enumerate(pred_tc_sample_paths):
                pred_example = None
                pred_atom_array = None
                try:
                    try:
                        pred_example = prepare_af3_prediction(
                            pdb_path=pred_tc_sample_path,
                            preprocess_cfg=preprocess_cfg,
                            featurizer_cfg=featurizer_cfg,
                        )

                        pred_atom_array = pred_example["atom_array"]
                        per_pred_sc_metrics = compute_self_consistency_metrics_atomarray(
                            pred_atom_array=pred_atom_array,
                            sample_atom_array=designed_sample_atom_array,
                            pred_sample_path=pred_tc_sample_path,
                            save_aligned=False,
                        )

                    except Exception as e:
                        print(f"Self-consistency metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                        continue
                    else:
                        # Store self-consistency metrics
                        designed_sample_id_to_per_pred_sc_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_sc_metrics

                    # Only compute docking metrics if ligand exists
                    if ligand_pn_unit_iids:
                        try:
                            per_pred_docking_metrics = compute_docking_metrics_atomarray(
                                pred_atom_array=pred_atom_array,
                                sample_atom_array=designed_sample_atom_array,
                                reference_atom_array=reference_sample_atom_array,
                                pred_sample_path=pred_tc_sample_path,
                                save_aligned=False,
                                pocket_distance_for_docking_metrics=pocket_cfg.pocket_distance_for_docking_metrics,
                                receptor_pn_unit_iids=protein_pn_unit_iids,
                                ligand_pn_unit_iids=ligand_pn_unit_iids,
                                ligand_ccd_codes=ligand_ccd_codes,
                                ligand_smiles=ligand_smiles,
                                reference_ligand_pn_unit_iids=reference_ligand_pn_unit_iids,
                                ref_sample_is_designed=input_sample_is_designed,
                            )

                        except Exception as e:
                            print(f"Docking metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                            continue
                        else:
                            # Store docking metrics
                            designed_sample_id_to_per_pred_docking_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_docking_metrics
                finally:
                    del pred_example, pred_atom_array

        # Free memory after processing each input sample
        if free_atom_arrays_progressively:
            subsample_dict.pop('designed_sample_atom_array', None)
        gc.collect()

    # Aggregate best metrics per designed_sample_id (best diffusion sample)
    designed_sample_id_best_sc_metrics = _aggregate_best_sc_metrics_per_designed_sample(designed_sample_id_to_per_pred_sc_metrics)
    designed_sample_id_best_docking_metrics = _aggregate_best_docking_metrics_per_designed_sample(designed_sample_id_to_per_pred_docking_metrics)

    # Aggregate best metrics per input_sample_id (best designed sample)
    input_sample_id_best_sc_metrics = _aggregate_best_sc_metrics_per_input_sample(designed_sample_id_best_sc_metrics)
    input_sample_id_best_docking_metrics = _aggregate_best_docking_metrics_per_input_sample(designed_sample_id_best_docking_metrics)

    # Save results with "tc_" prefix to distinguish from self-consistency (SS) results
    _save_metrics_results(
        out_dir=out_dir,
        designed_sample_id_to_per_pred_sc_metrics=designed_sample_id_to_per_pred_sc_metrics,
        designed_sample_id_to_per_pred_docking_metrics=designed_sample_id_to_per_pred_docking_metrics,
        designed_sample_id_best_sc_metrics=designed_sample_id_best_sc_metrics,
        designed_sample_id_best_docking_metrics=designed_sample_id_best_docking_metrics,
        input_sample_id_best_sc_metrics=input_sample_id_best_sc_metrics,
        input_sample_id_best_docking_metrics=input_sample_id_best_docking_metrics,
        no_wandb=no_wandb,
        ckpt_info=ckpt_info,
        csv_suffix=csv_suffix,
        mode_prefix="tc_"
    )

    print("\n" + "="*80)
    print("AF3 Docking Consistency Evaluation (Template-Conditioned) Complete")
    print(f"Results saved to {out_dir}")
    print("="*80 + "\n")


def _aggregate_best_sc_metrics_per_designed_sample(designed_sample_id_to_per_pred_sc_metrics: dict) -> dict:
    """
    Aggregate best self-consistency metrics per designed_sample_id (by max avg_ca_plddt across diffusion samples).

    Returns:
        dict: {designed_sample_id: {"input_sample_id": ..., "avg_ca_plddt": ..., "sc_ca_rmsd": ...}}
    """
    designed_sample_id_best_sc_metrics = {}
    for designed_sample_id, per_pred_sc_metrics in designed_sample_id_to_per_pred_sc_metrics.items():
        input_sample_id = per_pred_sc_metrics.get("input_sample_id")

        # Filter only diffusion predictions (exclude metadata keys like "input_sample_id")
        diffusion_preds = {k: v for k, v in per_pred_sc_metrics.items() if k.startswith("diffusion_")}

        if not diffusion_preds:
            continue

        # Find the prediction with max avg_ca_plddt
        best_pred = max(diffusion_preds.values(), key=lambda x: x["avg_ca_plddt"])
        designed_sample_id_best_sc_metrics[designed_sample_id] = {
            "input_sample_id": input_sample_id,
            "avg_ca_plddt": best_pred["avg_ca_plddt"],
            "sc_ca_rmsd": best_pred["sc_ca_rmsd"]
        }
    return designed_sample_id_best_sc_metrics


_OPTIONAL_DOCKING_METRIC_FIELDS = (
    "ligand_input_type",
    "ligand_rmsd_mode",
    "ligand_pn_unit_iids",
    "reference_ligand_pn_unit_iids",
    "reference_ligand_ccd_code",
    "mcs_num_atoms",
    "mcs_pred_coverage",
    "mcs_reference_coverage",
    "metal_num_predicted_coordinating_protein_residues",
)


def _copy_optional_docking_metric_fields(target: dict, source: dict) -> None:
    for field in _OPTIONAL_DOCKING_METRIC_FIELDS:
        if field in source:
            target[field] = source.get(field)


def _aggregate_best_docking_metrics_per_designed_sample(designed_sample_id_to_per_pred_docking_metrics: dict) -> dict:
    """
    Aggregate best docking metrics per designed_sample_id (by max ligand_plddt across diffusion samples).

    Returns:
        dict: {designed_sample_id: {"input_sample_id": ..., "ligand_rmsd": ..., ...}}
    """
    designed_sample_id_best_docking_metrics = {}
    for designed_sample_id, per_pred_docking_metrics in designed_sample_id_to_per_pred_docking_metrics.items():
        input_sample_id = per_pred_docking_metrics.get("input_sample_id")

        # Filter only diffusion predictions with valid ligand_plddt
        diffusion_preds = {
            k: v for k, v in per_pred_docking_metrics.items()
            if (
                k.startswith("diffusion_")
                and not v.get("error")
                and "ligand_plddt" in v
                and v["ligand_plddt"] is not None
            )
        }

        if not diffusion_preds:
            continue

        # Find the prediction with max ligand_plddt
        best_pred = max(diffusion_preds.values(), key=lambda x: x["ligand_plddt"])
        best_metrics = {
            "input_sample_id": input_sample_id,
            "ligand_rmsd": best_pred["ligand_rmsd"],
            "binding_site_rmsd": best_pred["binding_site_rmsd"],
            "ligand_plddt": best_pred["ligand_plddt"],
            "binding_site_plddt": best_pred["binding_site_plddt"],
            "iptm": best_pred["iptm"],
            "interface_min_pae": best_pred["interface_min_pae"],
            "ligand_ccd_code": best_pred.get("ligand_ccd_code"),
        }
        _copy_optional_docking_metric_fields(best_metrics, best_pred)
        designed_sample_id_best_docking_metrics[designed_sample_id] = best_metrics
    return designed_sample_id_best_docking_metrics


def _aggregate_best_sc_metrics_per_input_sample(designed_sample_id_best_sc_metrics: dict) -> dict:
    """
    Aggregate best self-consistency metrics per input_sample_id (by max avg_ca_plddt across designed samples).

    Returns:
        dict: {input_sample_id: {"best_designed_sample_id": ..., "avg_ca_plddt": ..., "sc_ca_rmsd": ...}}
    """
    # Group by input_sample_id
    input_sample_id_to_designed_samples = defaultdict(list)
    for designed_sample_id, metrics in designed_sample_id_best_sc_metrics.items():
        input_sample_id = metrics["input_sample_id"]
        input_sample_id_to_designed_samples[input_sample_id].append((designed_sample_id, metrics))

    # Find best designed_sample_id per input_sample_id
    input_sample_id_best_sc_metrics = {}
    for input_sample_id, designed_samples in input_sample_id_to_designed_samples.items():
        best_designed_sample_id, best_metrics = max(designed_samples, key=lambda x: x[1]["avg_ca_plddt"])
        input_sample_id_best_sc_metrics[input_sample_id] = {
            "best_designed_sample_id": best_designed_sample_id,
            "avg_ca_plddt": best_metrics["avg_ca_plddt"],
            "sc_ca_rmsd": best_metrics["sc_ca_rmsd"]
        }
    return input_sample_id_best_sc_metrics


def _aggregate_best_docking_metrics_per_input_sample(designed_sample_id_best_docking_metrics: dict) -> dict:
    """
    Aggregate best docking metrics per input_sample_id (by max ligand_plddt across designed samples).

    Returns:
        dict: {input_sample_id: {"best_designed_sample_id": ..., "ligand_rmsd": ..., ...}}
    """
    # Group by input_sample_id
    input_sample_id_to_designed_samples = defaultdict(list)
    for designed_sample_id, metrics in designed_sample_id_best_docking_metrics.items():
        input_sample_id = metrics["input_sample_id"]
        input_sample_id_to_designed_samples[input_sample_id].append((designed_sample_id, metrics))

    # Find best designed_sample_id per input_sample_id
    input_sample_id_best_docking_metrics = {}
    for input_sample_id, designed_samples in input_sample_id_to_designed_samples.items():
        best_designed_sample_id, best_metrics = max(designed_samples, key=lambda x: x[1]["ligand_plddt"])
        best_input_metrics = {
            "best_designed_sample_id": best_designed_sample_id,
            "ligand_rmsd": best_metrics["ligand_rmsd"],
            "binding_site_rmsd": best_metrics["binding_site_rmsd"],
            "ligand_plddt": best_metrics["ligand_plddt"],
            "binding_site_plddt": best_metrics["binding_site_plddt"],
            "iptm": best_metrics["iptm"],
            "interface_min_pae": best_metrics["interface_min_pae"],
            "ligand_ccd_code": best_metrics.get("ligand_ccd_code"),
        }
        _copy_optional_docking_metric_fields(best_input_metrics, best_metrics)
        input_sample_id_best_docking_metrics[input_sample_id] = best_input_metrics
    return input_sample_id_best_docking_metrics


def _save_metrics_results(out_dir: Path = None,
                          designed_sample_id_to_per_pred_sc_metrics: dict = None,
                          designed_sample_id_to_per_pred_docking_metrics: dict = None,
                          designed_sample_id_best_sc_metrics: dict = None,
                          designed_sample_id_best_docking_metrics: dict = None,
                          input_sample_id_best_sc_metrics: dict = None,
                          input_sample_id_best_docking_metrics: dict = None,
                          no_wandb: bool = False,
                          ckpt_info: dict = None,
                          csv_suffix: str = "",
                          mode_prefix: str = "") -> None:
    """Save metrics results to CSV and log to wandb.

    Args:
        csv_suffix: Optional suffix for CSV filenames (e.g. "_array_0" for array jobs).
        mode_prefix: Optional prefix for CSV filenames and wandb keys to distinguish
                     between SS and TC results (e.g. "tc_" for template-conditioned).
    """

    # All self-consistency metrics per designed_sample_id (with all diffusion samples)
    all_sc_metrics_df = pd.DataFrame.from_dict(designed_sample_id_to_per_pred_sc_metrics, orient='index')
    all_sc_metrics_df = all_sc_metrics_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    all_sc_metrics_df.to_csv(Path(out_dir, f"{mode_prefix}all_sc_metrics_per_designed_sample{csv_suffix}.csv"), index=False)

    # All docking metrics per designed_sample_id (with all diffusion samples)
    all_docking_metrics_df = pd.DataFrame.from_dict(designed_sample_id_to_per_pred_docking_metrics, orient='index')
    all_docking_metrics_df = all_docking_metrics_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    all_docking_metrics_df.to_csv(Path(out_dir, f"{mode_prefix}all_docking_metrics_per_designed_sample{csv_suffix}.csv"), index=False)

    # Best self-consistency metrics per designed_sample_id (best diffusion sample)
    best_sc_per_designed_df = pd.DataFrame.from_dict(designed_sample_id_best_sc_metrics, orient='index')
    best_sc_per_designed_df = best_sc_per_designed_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    best_sc_per_designed_df.to_csv(Path(out_dir, f"{mode_prefix}best_sc_metrics_per_designed_sample{csv_suffix}.csv"), index=False)

    # Best docking metrics per designed_sample_id (best diffusion sample)
    best_docking_per_designed_df = pd.DataFrame.from_dict(designed_sample_id_best_docking_metrics, orient='index')
    best_docking_per_designed_df = best_docking_per_designed_df.reset_index().rename(columns={'index': 'designed_sample_id'})
    best_docking_per_designed_df.to_csv(Path(out_dir, f"{mode_prefix}best_docking_metrics_per_designed_sample{csv_suffix}.csv"), index=False)

    # Best self-consistency metrics per input_sample_id (best designed sample)
    best_sc_per_input_df = pd.DataFrame.from_dict(input_sample_id_best_sc_metrics, orient='index')
    best_sc_per_input_df = best_sc_per_input_df.reset_index().rename(columns={'index': 'input_sample_id'})
    best_sc_per_input_df.to_csv(Path(out_dir, f"{mode_prefix}best_sc_metrics_per_input_sample{csv_suffix}.csv"), index=False)

    # Best docking metrics per input_sample_id (best designed sample)
    best_docking_per_input_df = pd.DataFrame.from_dict(input_sample_id_best_docking_metrics, orient='index')
    best_docking_per_input_df = best_docking_per_input_df.reset_index().rename(columns={'index': 'input_sample_id'})
    best_docking_per_input_df.to_csv(Path(out_dir, f"{mode_prefix}best_docking_metrics_per_input_sample{csv_suffix}.csv"), index=False)

    # Log summary metrics to wandb (using input_sample_id level for final reporting)
    if input_sample_id_best_sc_metrics:
        best_sc_ca_rmsds = [m["sc_ca_rmsd"] for m in input_sample_id_best_sc_metrics.values()]
        best_avg_ca_plddts = [m["avg_ca_plddt"] for m in input_sample_id_best_sc_metrics.values()]

        wandb_metrics = {
            f"eval/median/{mode_prefix}sc_ca_rmsd": np.median(best_sc_ca_rmsds),
            f"eval/median/{mode_prefix}avg_ca_plddt": np.median(best_avg_ca_plddts),
        }

        if ckpt_info:
            wandb_metrics["trainer/global_step"] = ckpt_info["global_step"]
            wandb_metrics["trainer/epoch"] = ckpt_info["epoch"]

        if not no_wandb:
            wandb.log(wandb_metrics, commit=True)
            print(f"Logged metrics to wandb: {wandb_metrics}")

    if input_sample_id_best_docking_metrics:
        best_ligand_rmsd = [m["ligand_rmsd"] for m in input_sample_id_best_docking_metrics.values() if m["ligand_rmsd"] is not None]
        best_binding_site_rmsd = [m["binding_site_rmsd"] for m in input_sample_id_best_docking_metrics.values() if m["binding_site_rmsd"] is not None]
        best_ligand_plddt = [m["ligand_plddt"] for m in input_sample_id_best_docking_metrics.values() if m["ligand_plddt"] is not None]
        best_binding_site_plddt = [m["binding_site_plddt"] for m in input_sample_id_best_docking_metrics.values() if m["binding_site_plddt"] is not None]
        best_iptm = [m["iptm"] for m in input_sample_id_best_docking_metrics.values() if m["iptm"] is not None]
        best_interface_min_pae = [m["interface_min_pae"] for m in input_sample_id_best_docking_metrics.values() if m["interface_min_pae"] is not None]

        wandb_metrics = {
            f"eval/median/{mode_prefix}ligand_rmsd": np.median(best_ligand_rmsd) if best_ligand_rmsd else None,
            f"eval/median/{mode_prefix}binding_site_rmsd": np.median(best_binding_site_rmsd) if best_binding_site_rmsd else None,
            f"eval/median/{mode_prefix}ligand_plddt": np.median(best_ligand_plddt) if best_ligand_plddt else None,
            f"eval/median/{mode_prefix}binding_site_plddt": np.median(best_binding_site_plddt) if best_binding_site_plddt else None,
            f"eval/median/{mode_prefix}iptm": np.median(best_iptm) if best_iptm else None,
            f"eval/median/{mode_prefix}interface_min_pae": np.median(best_interface_min_pae) if best_interface_min_pae else None,
        }

        if ckpt_info:
            wandb_metrics["trainer/global_step"] = ckpt_info["global_step"]
            wandb_metrics["trainer/epoch"] = ckpt_info["epoch"]

        if not no_wandb:
            wandb.log(wandb_metrics, commit=True)
            print(f"Logged metrics to wandb: {wandb_metrics}")
