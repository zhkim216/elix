import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.config import config_value_as_bool
from allatom_design.eval.structure_prediction.af3_json import make_af3_json
from allatom_design.eval.structure_prediction.af3_runner import (
    expected_prediction_count_from_json,
    run_af3_single_sequence,
    run_af3_template_conditioned,
    summarize_af3_prediction_outputs,
)
from allatom_design.eval.metrics.sequence_recovery import build_sequence_recovery_metric_config


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


def _pocket_distance_bins_from_cfg(pocket_cfg: DictConfig | dict | None) -> list[tuple[float, float]] | None:
    if pocket_cfg is None:
        return None
    bins_raw = pocket_cfg.get("pocket_distance_bins", None)
    return [tuple(b) for b in bins_raw] if bins_raw is not None else None


def _select_indexed_or_scalar(value: Any, idx: int) -> Any:
    if isinstance(value, list):
        return value[idx] if idx < len(value) else None
    return value


def _require_complete_af3_predictions(struct_pred_cfg: DictConfig | dict | None) -> bool:
    if struct_pred_cfg is None:
        return False
    selectable_cfg = (
        struct_pred_cfg
        if isinstance(struct_pred_cfg, DictConfig)
        else OmegaConf.create(struct_pred_cfg)
    )
    value = OmegaConf.select(
        selectable_cfg,
        "af3.require_complete_predictions",
        default=False,
    )
    return config_value_as_bool(value)


def _strict_af3_input_fingerprint(struct_pred_cfg: DictConfig | dict | None, mode: str) -> bool:
    if struct_pred_cfg is None:
        return False
    selectable_cfg = (
        struct_pred_cfg
        if isinstance(struct_pred_cfg, DictConfig)
        else OmegaConf.create(struct_pred_cfg)
    )
    value = OmegaConf.select(
        selectable_cfg,
        f"af3.inference_config.{mode}.strict_input_fingerprint",
        default=False,
    )
    return config_value_as_bool(value)


def _new_prediction_status_row(
    *,
    input_sample_id: str,
    designed_sample_id: str,
    mode: str,
    n_expected_predictions: int,
) -> dict[str, Any]:
    return {
        "input_sample_id": input_sample_id,
        "designed_sample_id": designed_sample_id,
        "mode": mode,
        "n_expected_predictions": int(n_expected_predictions),
        "n_found_predictions": 0,
        "n_malformed_prediction_dirs": 0,
        "n_surplus_predictions": 0,
        "af3_input_fingerprint_ok": "",
        "af3_input_fingerprint_error": "",
        "n_sc_success": 0,
        "n_docking_success": 0,
        "af3_status": "pending",
        "af3_error": "",
        "sc_errors": "",
        "docking_errors": "",
        "malformed_prediction_dirs": "",
    }


def _append_status_error(row: dict[str, Any], key: str, message: str) -> None:
    if row[key]:
        row[key] = f"{row[key]} | {message}"
    else:
        row[key] = message


def _record_prediction_summary(row: dict[str, Any], summary: dict[str, object]) -> None:
    malformed_dirs = summary.get("malformed_sample_dirs", [])
    row["n_found_predictions"] = int(summary["n_found"])
    row["n_malformed_prediction_dirs"] = int(summary["n_malformed"])
    row["n_surplus_predictions"] = int(summary.get("n_surplus", 0))
    fingerprint_ok = summary.get("input_fingerprint_ok")
    row["af3_input_fingerprint_ok"] = "" if fingerprint_ok is None else bool(fingerprint_ok)
    row["af3_input_fingerprint_error"] = str(summary.get("input_fingerprint_error", ""))
    row["malformed_prediction_dirs"] = ";".join(str(path) for path in malformed_dirs)


def _finalize_prediction_status(row: dict[str, Any], *, has_ligand: bool) -> None:
    if row["af3_status"] == "af3_failed":
        return
    n_expected = int(row["n_expected_predictions"])
    n_found = int(row["n_found_predictions"])
    n_sc_success = int(row["n_sc_success"])
    n_docking_success = int(row["n_docking_success"])
    if n_found == 0:
        row["af3_status"] = "missing_predictions"
    elif n_found < n_expected:
        row["af3_status"] = "incomplete_predictions"
    elif n_found > n_expected:
        row["af3_status"] = "surplus_predictions"
    elif row.get("af3_input_fingerprint_ok") is False:
        row["af3_status"] = "stale_predictions"
    elif n_sc_success < n_found:
        row["af3_status"] = "metric_failed"
    elif has_ligand and n_docking_success < n_found:
        row["af3_status"] = "metric_failed"
    else:
        row["af3_status"] = "complete"


def _raise_if_required_predictions_incomplete(
    *,
    status_rows: list[dict[str, Any]],
    require_complete_predictions: bool,
    mode_label: str,
) -> None:
    if not require_complete_predictions:
        return
    failed_rows = [row for row in status_rows if row["af3_status"] != "complete"]
    if not failed_rows:
        return
    examples = ", ".join(
        f"{row['designed_sample_id']}:{row['af3_status']}"
        for row in failed_rows[:5]
    )
    raise RuntimeError(
        f"{mode_label} AF3 evaluation had {len(failed_rows)} incomplete/failed "
        f"designed samples; first examples: {examples}"
    )


@dataclass(frozen=True)
class BindingSitePlddtContract:
    reference_atom_array: Any
    pocket_distances: Any
    pocket_distance_bins: Any
    pocket_annotation_method: str
    n_min_ligand_atoms: int

    def to_docking_kwargs(self, reference_ligand_pn_unit_iids: list[str] | None) -> dict[str, Any]:
        return {
            "binding_site_plddt_distances": self.pocket_distances,
            "binding_site_plddt_bins": self.pocket_distance_bins,
            "binding_site_plddt_reference_atom_array": self.reference_atom_array,
            "binding_site_plddt_reference_pocket_annotation_method": (
                self.pocket_annotation_method
            ),
            "binding_site_plddt_reference_ligand_pn_unit_iids": (
                reference_ligand_pn_unit_iids
            ),
            "binding_site_plddt_n_min_ligand_atoms": self.n_min_ligand_atoms,
        }


def _binding_site_plddt_contract(
    *,
    subsample_dict: dict,
    dsidx: int,
    pocket_cfg: DictConfig | dict | None,
    input_sample_is_designed: bool,
    fallback_reference_atom_array: Any,
) -> BindingSitePlddtContract:
    reference_atom_array = _select_indexed_or_scalar(
        subsample_dict.get("binding_site_plddt_reference_atom_array"),
        dsidx,
    )
    if reference_atom_array is None:
        reference_atom_array = fallback_reference_atom_array
    metric_config = _select_indexed_or_scalar(
        subsample_dict.get("binding_site_plddt_metric_config"),
        dsidx,
    )
    if metric_config is None:
        metric_config = _select_indexed_or_scalar(
            subsample_dict.get("sequence_recovery_metric_config"),
            dsidx,
        )
    if metric_config is None or not metric_config.enabled:
        metric_config = build_sequence_recovery_metric_config(
            pocket_cfg=pocket_cfg,
            input_sample_is_designed=input_sample_is_designed,
            pocket_distance_bins=_pocket_distance_bins_from_cfg(pocket_cfg),
            enabled=True,
        )
    return BindingSitePlddtContract(
        reference_atom_array=reference_atom_array,
        pocket_distances=metric_config.pocket_distances_for_seq_recovery,
        pocket_distance_bins=metric_config.pocket_distance_bins,
        pocket_annotation_method=metric_config.pocket_annotation_method,
        n_min_ligand_atoms=metric_config.n_min_ligand_atoms,
    )


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
        ckpt_info: Retained for caller compatibility; no best-aggregate logging is written.
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
    require_complete_predictions = _require_complete_af3_predictions(struct_pred_cfg)
    strict_input_fingerprint = _strict_af3_input_fingerprint(struct_pred_cfg, "ss")

    designed_sample_id_to_per_pred_sc_metrics = {}
    designed_sample_id_to_per_pred_docking_metrics = {}
    prediction_status_rows: list[dict[str, Any]] = []

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
            binding_site_plddt_contract = _binding_site_plddt_contract(
                subsample_dict=subsample_dict,
                dsidx=dsidx,
                pocket_cfg=pocket_cfg,
                input_sample_is_designed=input_sample_is_designed,
                fallback_reference_atom_array=reference_sample_atom_array,
            )
            pdb_chain_info = subsample_dict['pdb_chain_info']
            ss_json_path = subsample_dict['af3_ss_json_paths'][dsidx]
            n_expected_predictions = expected_prediction_count_from_json(
                ss_json_path,
                af3_inference_config,
                "ss",
            )
            status_row = _new_prediction_status_row(
                input_sample_id=input_sample_id,
                designed_sample_id=designed_sample_id,
                mode="ss",
                n_expected_predictions=n_expected_predictions,
            )
            prediction_status_rows.append(status_row)

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
                    status_row["af3_status"] = "af3_failed"
                    status_row["af3_error"] = str(e)
                    continue
                gc.collect()

            prediction_summary = summarize_af3_prediction_outputs(
                out_dir=str(af3_ss_pred_dir),
                job_name=designed_sample_id,
                expected_count=n_expected_predictions,
                json_path=ss_json_path,
                inference_config=af3_inference_config,
                mode="ss",
                strict_input_fingerprint=strict_input_fingerprint,
            )
            _record_prediction_summary(status_row, prediction_summary)
            pred_ss_sample_paths = prediction_summary["model_cif_paths"]
            if prediction_summary.get("input_fingerprint_ok") is False:
                _append_status_error(
                    status_row,
                    "af3_error",
                    str(prediction_summary.get("input_fingerprint_error", "")),
                )
                _finalize_prediction_status(status_row, has_ligand=bool(ligand_pn_unit_iids))
                continue

            if len(pred_ss_sample_paths) == 0:
                print(f"No AF3 predicted structure found for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}")
                _finalize_prediction_status(status_row, has_ligand=bool(ligand_pn_unit_iids))
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
                        _append_status_error(status_row, "sc_errors", f"diffusion_{pred_idx}: {e}")
                        continue
                    else:
                        designed_sample_id_to_per_pred_sc_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_sc_metrics
                        status_row["n_sc_success"] += 1

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
                                # Annotate the docking reference pocket with the same method as
                                # binding-site pLDDT (all_atom for native references) instead of
                                # letting it default to calpha when two-stage forces
                                # ref_sample_is_designed=True.
                                reference_pocket_annotation_method=binding_site_plddt_contract.pocket_annotation_method,
                                save_aligned=False,
                                **binding_site_plddt_contract.to_docking_kwargs(
                                    reference_ligand_pn_unit_iids
                                ),
                            )

                        except Exception as e:
                            print(f"Docking metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                            _append_status_error(status_row, "docking_errors", f"diffusion_{pred_idx}: {e}")
                            continue
                        else:
                            designed_sample_id_to_per_pred_docking_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_docking_metrics
                            status_row["n_docking_success"] += 1
                finally:
                    del pred_example, pred_atom_array
            _finalize_prediction_status(status_row, has_ligand=bool(ligand_pn_unit_iids))

        # Free memory after processing each input sample
        if free_atom_arrays_progressively:
            subsample_dict.pop('designed_sample_atom_array', None)
        gc.collect()

    # Save results
    _save_metrics_results(
        out_dir=out_dir,
        designed_sample_id_to_per_pred_sc_metrics=designed_sample_id_to_per_pred_sc_metrics,
        designed_sample_id_to_per_pred_docking_metrics=designed_sample_id_to_per_pred_docking_metrics,
        prediction_status_rows=prediction_status_rows,
        csv_suffix=csv_suffix,
    )
    _raise_if_required_predictions_incomplete(
        status_rows=prediction_status_rows,
        require_complete_predictions=require_complete_predictions,
        mode_label="SS",
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
        ckpt_info: Retained for caller compatibility; no best-aggregate logging is written.
        no_wandb: Retained for caller compatibility.
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
    require_complete_predictions = _require_complete_af3_predictions(struct_pred_cfg)
    strict_input_fingerprint = _strict_af3_input_fingerprint(struct_pred_cfg, "tc")

    designed_sample_id_to_per_pred_sc_metrics = {}
    designed_sample_id_to_per_pred_docking_metrics = {}
    prediction_status_rows: list[dict[str, Any]] = []

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
            binding_site_plddt_contract = _binding_site_plddt_contract(
                subsample_dict=subsample_dict,
                dsidx=dsidx,
                pocket_cfg=pocket_cfg,
                input_sample_is_designed=input_sample_is_designed,
                fallback_reference_atom_array=reference_sample_atom_array,
            )
            pdb_chain_info = subsample_dict['pdb_chain_info']
            tc_json_path = subsample_dict['af3_tc_json_paths'][dsidx]
            n_expected_predictions = expected_prediction_count_from_json(
                tc_json_path,
                af3_inference_config,
                "tc",
            )
            status_row = _new_prediction_status_row(
                input_sample_id=input_sample_id,
                designed_sample_id=designed_sample_id,
                mode="tc",
                n_expected_predictions=n_expected_predictions,
            )
            prediction_status_rows.append(status_row)

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
                    status_row["af3_status"] = "af3_failed"
                    status_row["af3_error"] = str(e)
                    continue
                gc.collect()

            prediction_summary = summarize_af3_prediction_outputs(
                out_dir=str(af3_tc_pred_dir),
                job_name=designed_sample_id,
                expected_count=n_expected_predictions,
                json_path=tc_json_path,
                inference_config=af3_inference_config,
                mode="tc",
                strict_input_fingerprint=strict_input_fingerprint,
            )
            _record_prediction_summary(status_row, prediction_summary)
            pred_tc_sample_paths = prediction_summary["model_cif_paths"]
            if prediction_summary.get("input_fingerprint_ok") is False:
                _append_status_error(
                    status_row,
                    "af3_error",
                    str(prediction_summary.get("input_fingerprint_error", "")),
                )
                _finalize_prediction_status(status_row, has_ligand=bool(ligand_pn_unit_iids))
                continue

            if len(pred_tc_sample_paths) == 0:
                print(f"No AF3 TC predicted structure found for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}")
                _finalize_prediction_status(status_row, has_ligand=bool(ligand_pn_unit_iids))
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
                        _append_status_error(status_row, "sc_errors", f"diffusion_{pred_idx}: {e}")
                        continue
                    else:
                        # Store self-consistency metrics
                        designed_sample_id_to_per_pred_sc_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_sc_metrics
                        status_row["n_sc_success"] += 1

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
                                # Annotate the docking reference pocket with the same method as
                                # binding-site pLDDT (all_atom for native references) instead of
                                # letting it default to calpha when two-stage forces
                                # ref_sample_is_designed=True.
                                reference_pocket_annotation_method=binding_site_plddt_contract.pocket_annotation_method,
                                **binding_site_plddt_contract.to_docking_kwargs(
                                    reference_ligand_pn_unit_iids
                                ),
                            )

                        except Exception as e:
                            print(f"Docking metrics computation failed for input_sample_id: {input_sample_id}, designed_sample_id: {designed_sample_id}, pred_idx: {pred_idx}: {e}")
                            _append_status_error(status_row, "docking_errors", f"diffusion_{pred_idx}: {e}")
                            continue
                        else:
                            # Store docking metrics
                            designed_sample_id_to_per_pred_docking_metrics[designed_sample_id][f"diffusion_{pred_idx}"] = per_pred_docking_metrics
                            status_row["n_docking_success"] += 1
                finally:
                    del pred_example, pred_atom_array
            _finalize_prediction_status(status_row, has_ligand=bool(ligand_pn_unit_iids))

        # Free memory after processing each input sample
        if free_atom_arrays_progressively:
            subsample_dict.pop('designed_sample_atom_array', None)
        gc.collect()

    # Save results with "tc_" prefix to distinguish from self-consistency (SS) results
    _save_metrics_results(
        out_dir=out_dir,
        designed_sample_id_to_per_pred_sc_metrics=designed_sample_id_to_per_pred_sc_metrics,
        designed_sample_id_to_per_pred_docking_metrics=designed_sample_id_to_per_pred_docking_metrics,
        prediction_status_rows=prediction_status_rows,
        csv_suffix=csv_suffix,
        mode_prefix="tc_"
    )
    _raise_if_required_predictions_incomplete(
        status_rows=prediction_status_rows,
        require_complete_predictions=require_complete_predictions,
        mode_label="TC",
    )

    print("\n" + "="*80)
    print("AF3 Docking Consistency Evaluation (Template-Conditioned) Complete")
    print(f"Results saved to {out_dir}")
    print("="*80 + "\n")


def _flatten_per_prediction_metrics(per_designed_sample_metrics: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for designed_sample_id, metrics in per_designed_sample_metrics.items():
        input_sample_id = metrics.get("input_sample_id")
        for prediction_id, prediction_metrics in metrics.items():
            if not str(prediction_id).startswith("diffusion_"):
                continue
            if not isinstance(prediction_metrics, dict):
                continue
            rows.append({
                "designed_sample_id": designed_sample_id,
                "input_sample_id": input_sample_id,
                "prediction_id": prediction_id,
                **prediction_metrics,
            })
    return rows


def _save_metrics_results(out_dir: Path = None,
                          designed_sample_id_to_per_pred_sc_metrics: dict = None,
                          designed_sample_id_to_per_pred_docking_metrics: dict = None,
                          prediction_status_rows: list[dict[str, Any]] | None = None,
                          csv_suffix: str = "",
                          mode_prefix: str = "") -> None:
    """Save per-designed-sample and per-prediction metrics to CSV.

    Args:
        csv_suffix: Optional suffix for CSV filenames (e.g. "_array_0" for array jobs).
        mode_prefix: Optional prefix for CSV filenames to distinguish
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
    pd.DataFrame(
        _flatten_per_prediction_metrics(designed_sample_id_to_per_pred_docking_metrics)
    ).to_csv(
        Path(out_dir, f"{mode_prefix}all_docking_metrics_per_prediction{csv_suffix}.csv"),
        index=False,
    )
    if prediction_status_rows is not None:
        pd.DataFrame(prediction_status_rows).to_csv(
            Path(out_dir, f"{mode_prefix}af3_prediction_status{csv_suffix}.csv"),
            index=False,
        )
