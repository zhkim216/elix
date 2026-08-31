"""ESMFold2 single-sequence prediction and Elix metric evaluation."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io.utils.selection import get_residue_starts
from omegaconf import DictConfig
from tqdm import tqdm

from allatom_design.eval.config import config_value_as_bool
from allatom_design.eval.structure_prediction.evaluation import (
    evaluate_restored_prediction,
    role_contract_error_row,
    save_prediction_metrics,
)

from .inputs import (
    load_esmfold2_input_record,
    prepare_esmfold2_prediction,
    write_esmfold2_inputs,
)
from .runner import (
    ESMFold2RuntimeConfig,
    ESMFold2Session,
    summarize_esmfold2_prediction_outputs,
)


def _append_error(row: dict[str, Any], key: str, message: str) -> None:
    row[key] = f"{row[key]} | {message}" if row[key] else message


def _new_status_row(
    *,
    input_sample_id: str,
    designed_sample_id: str,
    expected_count: int,
) -> dict[str, Any]:
    return {
        "input_sample_id": input_sample_id,
        "designed_sample_id": designed_sample_id,
        "mode": "ss",
        "n_expected_predictions": expected_count,
        "n_found_predictions": 0,
        "n_malformed_prediction_dirs": 0,
        "n_surplus_predictions": 0,
        "esmfold2_input_fingerprint_ok": "",
        "esmfold2_input_fingerprint_error": "",
        "manifest_status": "",
        "n_sc_success": 0,
        "n_role_metric_predictions_success": 0,
        "n_role_metric_rows": 0,
        "n_role_metric_errors": 0,
        "esmfold2_status": "pending",
        "esmfold2_error": "",
        "sc_errors": "",
        "role_metric_errors": "",
        "malformed_prediction_dirs": "",
    }


def _record_summary(
    row: dict[str, Any], summary: dict[str, Any]
) -> None:
    row["n_found_predictions"] = int(summary["n_found"])
    row["n_malformed_prediction_dirs"] = int(summary["n_malformed"])
    row["n_surplus_predictions"] = int(summary["n_surplus"])
    fingerprint_ok = summary["input_fingerprint_ok"]
    row["esmfold2_input_fingerprint_ok"] = (
        "" if fingerprint_ok is None else bool(fingerprint_ok)
    )
    row["esmfold2_input_fingerprint_error"] = str(
        summary["input_fingerprint_error"]
    )
    row["manifest_status"] = str(summary["manifest_status"])
    row["malformed_prediction_dirs"] = ";".join(
        str(path) for path in summary["malformed_prediction_dirs"]
    )


def _record_role_status(
    *,
    row: dict[str, Any],
    prediction_id: str,
    role_rows: list[dict[str, Any]],
) -> None:
    row["n_role_metric_rows"] += len(role_rows)
    error_rows = [
        role_row
        for role_row in role_rows
        if role_row.get("status") not in {"ok", "not_applicable"}
    ]
    row["n_role_metric_errors"] += len(error_rows)
    if error_rows:
        messages = list(
            dict.fromkeys(str(value.get("error", "")) for value in error_rows)
        )
        _append_error(
            row,
            "role_metric_errors",
            f"{prediction_id}: " + " | ".join(messages),
        )
    else:
        row["n_role_metric_predictions_success"] += 1


def _finalize_status(
    row: dict[str, Any], *, has_role_metrics: bool
) -> None:
    if row["esmfold2_status"] == "esmfold2_failed":
        return
    expected = int(row["n_expected_predictions"])
    found = int(row["n_found_predictions"])
    if found == 0:
        row["esmfold2_status"] = "missing_predictions"
    elif found < expected:
        row["esmfold2_status"] = "incomplete_predictions"
    elif found > expected or int(row["n_surplus_predictions"]) > 0:
        row["esmfold2_status"] = "surplus_predictions"
    elif int(row["n_malformed_prediction_dirs"]) > 0:
        row["esmfold2_status"] = "malformed_predictions"
    elif row["esmfold2_input_fingerprint_ok"] is False:
        row["esmfold2_status"] = "stale_predictions"
    elif row["manifest_status"] != "complete":
        row["esmfold2_status"] = "incomplete_predictions"
    elif int(row["n_sc_success"]) < found:
        row["esmfold2_status"] = "metric_failed"
    elif (
        has_role_metrics
        and int(row["n_role_metric_predictions_success"]) < found
    ):
        row["esmfold2_status"] = "metric_failed"
    else:
        row["esmfold2_status"] = "complete"


def restore_esmfold2_prediction_identity(
    pred_atom_array: Any,
    *,
    designed_sample_atom_array: Any,
    input_record: dict[str, Any],
) -> Any:
    """Validate the exact serialized sequence and restore source identity."""
    required_annotations = {"chain_id", "atom_name", "res_name", "res_id"}
    for label, atom_array in (
        ("designed", designed_sample_atom_array),
        ("predicted", pred_atom_array),
    ):
        missing = required_annotations - set(
            atom_array.get_annotation_categories()
        )
        if missing:
            raise ValueError(
                f"{label} atom array lacks identity annotations: "
                f"{sorted(missing)}"
            )
    chain_mapping = {
        str(chain_id): str(pn_unit_iid)
        for chain_id, pn_unit_iid in input_record[
            "chain_id_to_pn_unit_iid"
        ].items()
    }
    observed_chain_ids = set(map(str, pred_atom_array.chain_id))
    expected_chain_ids = set(chain_mapping)
    if observed_chain_ids != expected_chain_ids:
        raise ValueError(
            "ESMFold2 prediction chain IDs do not match the serialized input; "
            f"missing={sorted(expected_chain_ids - observed_chain_ids)}, "
            f"unexpected={sorted(observed_chain_ids - expected_chain_ids)}"
        )
    pred_atom_array.set_annotation(
        "pn_unit_iid",
        np.asarray(
            [
                chain_mapping[str(chain_id)]
                for chain_id in pred_atom_array.chain_id
            ],
            dtype=str,
        ),
    )
    restored_res_ids = np.asarray(pred_atom_array.res_id).copy()
    for entry in input_record["sequences"]:
        chain_id = str(entry["id"])
        pn_unit_iid = str(entry["source_pn_unit_iid"])
        source_res_ids = np.asarray(entry["source_res_ids"], dtype=int)
        source_res_names = np.asarray(entry["source_res_names"], dtype=str)

        designed_chain = designed_sample_atom_array[
            designed_sample_atom_array.pn_unit_iid == pn_unit_iid
        ]
        designed_starts = get_residue_starts(designed_chain)
        designed_res_ids = np.asarray(
            designed_chain.res_id[designed_starts], dtype=int
        )
        designed_res_names = np.asarray(
            designed_chain.res_name[designed_starts], dtype=str
        )
        if not np.array_equal(designed_res_ids, source_res_ids) or not np.array_equal(
            designed_res_names, source_res_names
        ):
            raise ValueError(
                "Persisted ESMFold2 input identity does not match the designed "
                f"sample for PN unit {pn_unit_iid}"
            )

        pred_chain_mask = np.asarray(pred_atom_array.chain_id) == chain_id
        pred_chain = pred_atom_array[pred_chain_mask]
        pred_starts = get_residue_starts(pred_chain)
        pred_names = np.asarray(pred_chain.res_name[pred_starts], dtype=str)
        if len(pred_starts) != len(source_res_ids):
            raise ValueError(
                f"ESMFold2 residue count mismatch for {pn_unit_iid}: "
                f"input={len(source_res_ids)}, predicted={len(pred_starts)}"
            )
        if not np.array_equal(pred_names, source_res_names):
            mismatch = np.where(pred_names != source_res_names)[0]
            examples = ", ".join(
                f"{int(index)}:{source_res_names[index]}!={pred_names[index]}"
                for index in mismatch[:10]
            )
            raise ValueError(
                f"ESMFold2 residue-name mismatch for {pn_unit_iid}: {examples}"
            )
        pred_chain_res_ids = np.asarray(
            pred_chain.res_id[pred_starts], dtype=int
        )
        if len(np.unique(pred_chain_res_ids)) != len(pred_chain_res_ids):
            raise ValueError(
                f"ESMFold2 predicted duplicate residue IDs for {pn_unit_iid}"
            )
        for predicted_res_id, source_res_id in zip(
            pred_chain_res_ids, source_res_ids, strict=True
        ):
            restored_res_ids[
                pred_chain_mask
                & (np.asarray(pred_atom_array.res_id) == predicted_res_id)
            ] = source_res_id
    pred_atom_array.res_id = restored_res_ids
    return pred_atom_array


def evaluate_esmfold2_structure_prediction(
    *,
    sample_dict: dict[str, dict[str, Any]],
    out_dir: str | Path,
    struct_pred_cfg: DictConfig,
    cif_parse_cfg: DictConfig,
    preprocess_cfg: DictConfig,
    featurizer_cfg: DictConfig,
    pocket_cfg: DictConfig,
    calculate_metrics_only: bool,
    csv_suffix: str,
    input_sample_is_designed: bool,
    free_atom_arrays_progressively: bool = True,
) -> list[dict[str, Any]]:
    """Run the locked ESMFold2 no-MSA protocol and existing Elix metrics."""
    out_dir = Path(out_dir)
    input_dir = out_dir / "esmfold2_ss_inputs"
    prediction_dir = out_dir / "esmfold2_ss_preds"
    input_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    sample_dict = write_esmfold2_inputs(
        sample_dict=sample_dict,
        input_dir=input_dir,
    )

    runtime = ESMFold2RuntimeConfig.from_config(
        struct_pred_cfg.esmfold2.inference_config
    )
    require_complete = config_value_as_bool(
        struct_pred_cfg.esmfold2.get(
            "require_complete_predictions", False
        )
    )
    from allatom_design.eval.metrics.role_aware import (
        resolve_role_aware_reference_pocket_distance,
    )

    reference_pocket_distance = resolve_role_aware_reference_pocket_distance(
        (
            pocket_cfg.get("role_aware_reference_pocket_distance", None)
            if pocket_cfg is not None
            else None
        )
    )
    has_any_role_metrics = any(
        entry.get("pn_unit_roles") is not None
        for entry in sample_dict.values()
    )
    role_metric_rows: list[dict[str, Any]] | None = (
        [] if has_any_role_metrics else None
    )
    per_sample_sc_metrics: dict[str, dict[str, Any]] = {}
    status_rows: list[dict[str, Any]] = []
    session = None if calculate_metrics_only else ESMFold2Session(runtime)
    try:
        for input_sample_id in tqdm(
            sample_dict, desc="ESMFold2 SS predictions"
        ):
            subsample_dict = sample_dict[input_sample_id]
            for index, designed_sample_id_raw in enumerate(
                subsample_dict["designed_sample_id"]
            ):
                designed_sample_id = str(designed_sample_id_raw)
                input_path = Path(
                    subsample_dict["esmfold2_input_paths"][index]
                )
                input_record = load_esmfold2_input_record(input_path)
                designed_atom_array = subsample_dict[
                    "designed_sample_atom_array"
                ][index]
                reference_atom_array = subsample_dict.get(
                    "reference_sample_atom_array", designed_atom_array
                )
                if isinstance(reference_atom_array, list):
                    reference_atom_array = reference_atom_array[index]
                pdb_chain_info = subsample_dict["pdb_chain_info"]
                has_role_metrics = (
                    subsample_dict.get("pn_unit_roles") is not None
                )
                per_sample_sc_metrics[designed_sample_id] = {
                    "input_sample_id": input_sample_id
                }
                row = _new_status_row(
                    input_sample_id=str(input_sample_id),
                    designed_sample_id=designed_sample_id,
                    expected_count=runtime.expected_prediction_count,
                )
                status_rows.append(row)
                if session is not None:
                    try:
                        session.predict(
                            input_path=input_path,
                            prediction_dir=prediction_dir,
                        )
                    except Exception as exc:
                        row["esmfold2_status"] = "esmfold2_failed"
                        row["esmfold2_error"] = str(exc)
                        print(
                            f"ESMFold2 prediction failed for "
                            f"{designed_sample_id}: {exc}",
                            flush=True,
                        )
                        continue
                    gc.collect()

                summary = summarize_esmfold2_prediction_outputs(
                    input_path=input_path,
                    prediction_dir=prediction_dir,
                    runtime=runtime,
                )
                _record_summary(row, summary)
                if summary["input_fingerprint_ok"] is False:
                    _append_error(
                        row,
                        "esmfold2_error",
                        summary["input_fingerprint_error"],
                    )
                    _finalize_status(
                        row, has_role_metrics=has_role_metrics
                    )
                    continue

                for artifact in summary["artifacts"]:
                    prediction_id = artifact.prediction_id
                    try:
                        pred_example = prepare_esmfold2_prediction(
                            pdb_path=str(artifact.model_cif_path),
                            cif_parse_cfg=cif_parse_cfg,
                            preprocess_cfg=preprocess_cfg,
                            featurizer_cfg=featurizer_cfg,
                        )
                        pred_atom_array = (
                            restore_esmfold2_prediction_identity(
                                pred_example["atom_array"],
                                designed_sample_atom_array=designed_atom_array,
                                input_record=input_record,
                            )
                        )
                        metrics = evaluate_restored_prediction(
                            input_sample_id=str(input_sample_id),
                            designed_sample_id=designed_sample_id,
                            prediction_id=prediction_id,
                            pred_atom_array=pred_atom_array,
                            designed_sample_atom_array=designed_atom_array,
                            reference_sample_atom_array=reference_atom_array,
                            pdb_chain_info=pdb_chain_info,
                            pn_unit_roles=subsample_dict.get("pn_unit_roles"),
                            pred_sample_path=artifact.model_cif_path,
                            reference_is_designed=input_sample_is_designed,
                            reference_pocket_distance=(
                                reference_pocket_distance
                            ),
                        )
                    except Exception as exc:
                        error = f"prediction parse failed: {exc}"
                        _append_error(
                            row,
                            "sc_errors",
                            f"{prediction_id}: {error}",
                        )
                        if has_role_metrics:
                            failed_role_rows = [
                                role_contract_error_row(
                                    input_sample_id=str(input_sample_id),
                                    designed_sample_id=designed_sample_id,
                                    prediction_id=prediction_id,
                                    error=error,
                                )
                            ]
                            assert role_metric_rows is not None
                            role_metric_rows.extend(failed_role_rows)
                            _record_role_status(
                                row=row,
                                prediction_id=prediction_id,
                                role_rows=failed_role_rows,
                            )
                        continue

                    if has_role_metrics:
                        assert role_metric_rows is not None
                        role_metric_rows.extend(metrics.role_metric_rows)
                        _record_role_status(
                            row=row,
                            prediction_id=prediction_id,
                            role_rows=metrics.role_metric_rows,
                        )
                    if metrics.self_consistency_error:
                        _append_error(
                            row,
                            "sc_errors",
                            f"{prediction_id}: "
                            f"{metrics.self_consistency_error}",
                        )
                    else:
                        assert metrics.self_consistency_metrics is not None
                        per_sample_sc_metrics[designed_sample_id][
                            prediction_id
                        ] = metrics.self_consistency_metrics
                        row["n_sc_success"] += 1
                _finalize_status(row, has_role_metrics=has_role_metrics)

            if free_atom_arrays_progressively:
                subsample_dict.pop("designed_sample_atom_array", None)
            gc.collect()
    finally:
        if session is not None:
            session.close()

    save_prediction_metrics(
        out_dir=out_dir,
        designed_sample_id_to_per_pred_sc_metrics=per_sample_sc_metrics,
        role_metric_rows=role_metric_rows,
        prediction_status_rows=status_rows,
        status_filename="esmfold2_prediction_status",
        csv_suffix=csv_suffix,
    )
    failed = [
        row for row in status_rows if row["esmfold2_status"] != "complete"
    ]
    if require_complete and failed:
        examples = ", ".join(
            f"{row['designed_sample_id']}:{row['esmfold2_status']}"
            for row in failed[:5]
        )
        raise RuntimeError(
            f"ESMFold2 evaluation had {len(failed)} incomplete/failed "
            f"designed samples; first examples: {examples}"
        )
    return status_rows
