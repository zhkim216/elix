import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import atomworks.enums as aw_enums
import numpy as np
import pandas as pd
from atomworks.io.utils.sequence import get_3_from_1_letter_code
from atomworks.io.utils.selection import get_residue_starts

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.config import config_value_as_bool
from allatom_design.eval.structure_prediction.AF3.af3_json import (
    build_af3_chain_id_to_pn_unit_iid,
    make_af3_json,
)
from allatom_design.eval.structure_prediction.AF3.af3_runner import (
    expected_prediction_count_from_json,
    inference_config_with_residue_index_by_chain,
    run_af3_single_sequence,
    run_af3_template_conditioned,
    summarize_af3_prediction_outputs,
)
from allatom_design.eval.utils.sampling_inputs import normalize_pn_unit_roles


ResidueIdLayout = Literal["sparse", "full_span"]


@dataclass(frozen=True)
class CachedAF3Prediction:
    """Inputs needed to evaluate one existing AF3 model CIF."""

    input_sample_id: str
    designed_sample_id: str
    prediction_id: str
    prediction_path: Path
    json_path: Path
    residue_id_layout: ResidueIdLayout
    pdb_chain_info: dict[str, Any]
    pn_unit_roles: dict[str, Any] | None
    af3_chain_id_to_pn_unit_iid: dict[str, str]


@dataclass(frozen=True)
class AF3PredictionMetrics:
    """Role and self-consistency outcomes for one AF3 prediction."""

    role_metric_rows: list[dict[str, Any]]
    self_consistency_metrics: dict[str, Any] | None
    self_consistency_error: str


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
        "n_role_metric_predictions_success": 0,
        "n_role_metric_rows": 0,
        "n_role_metric_errors": 0,
        "af3_status": "pending",
        "af3_error": "",
        "sc_errors": "",
        "role_metric_errors": "",
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


def _finalize_prediction_status(
    row: dict[str, Any],
    *,
    has_role_metrics: bool = False,
) -> None:
    if row["af3_status"] == "af3_failed":
        return
    n_expected = int(row["n_expected_predictions"])
    n_found = int(row["n_found_predictions"])
    n_sc_success = int(row["n_sc_success"])
    n_role_success = int(row["n_role_metric_predictions_success"])
    if n_found == 0:
        row["af3_status"] = "missing_predictions"
    elif n_found < n_expected:
        row["af3_status"] = "incomplete_predictions"
    elif n_found > n_expected:
        row["af3_status"] = "surplus_predictions"
    elif int(row["n_malformed_prediction_dirs"]) > 0:
        row["af3_status"] = "malformed_predictions"
    elif row.get("af3_input_fingerprint_ok") is False:
        row["af3_status"] = "stale_predictions"
    elif n_sc_success < n_found:
        row["af3_status"] = "metric_failed"
    elif has_role_metrics and n_role_success < n_found:
        row["af3_status"] = "metric_failed"
    else:
        row["af3_status"] = "complete"


def _compute_role_metric_rows_for_prediction(
    *,
    input_sample_id: str,
    designed_sample_id: str,
    prediction_id: str,
    subsample_dict: dict,
    pdb_chain_info: dict,
    reference_sample_atom_array: Any,
    designed_sample_atom_array: Any,
    pred_atom_array: Any,
    pred_sample_path: str | Path,
    input_sample_is_designed: bool,
    reference_pocket_distance: float | None,
    save_pocket_aligned: bool = True,
    pocket_aligned_output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    pn_unit_roles = subsample_dict.get("pn_unit_roles")
    if pn_unit_roles is None:
        return []

    from allatom_design.eval.metrics.role_aware import compute_role_aware_metrics_atomarray

    identifiers = {
        "input_sample_id": input_sample_id,
        "designed_sample_id": designed_sample_id,
        "prediction_id": prediction_id,
    }
    try:
        ligand_pn_unit_iids = [
            str(pn_unit_iid)
            for pn_unit_iid in pdb_chain_info.get("ligand_pn_unit_iids", [])
        ]
        ligand_smiles = _ligand_metric_values_from_chain_info(
            pdb_chain_info,
            ("ligand_smiles", "ligand_smiles_by_iid"),
            ligand_pn_unit_iids,
        )
        ligand_smiles_by_iid = (
            dict(zip(ligand_pn_unit_iids, ligand_smiles))
            if ligand_smiles is not None
            else None
        )
        metric_rows = compute_role_aware_metrics_atomarray(
            pred_atom_array=pred_atom_array,
            sample_atom_array=designed_sample_atom_array,
            reference_atom_array=reference_sample_atom_array,
            reference_is_designed=input_sample_is_designed,
            reference_pocket_distance=reference_pocket_distance,
            pn_unit_roles=pn_unit_roles,
            pred_sample_path=pred_sample_path,
            ligand_smiles_by_iid=ligand_smiles_by_iid,
            save_aligned=save_pocket_aligned,
            aligned_output_dir=pocket_aligned_output_dir,
        )
        if not metric_rows:
            raise RuntimeError("Role-aware metric computation returned no operation rows")
    except Exception as exc:
        return [
            _role_contract_error_row(
                **identifiers,
                error=str(exc),
            )
        ]
    return [{**identifiers, **metric_row} for metric_row in metric_rows]


def _role_contract_error_row(
    *,
    input_sample_id: str,
    designed_sample_id: str,
    prediction_id: str,
    error: str,
) -> dict[str, Any]:
    from allatom_design.eval.metrics.role_aware import ROLE_METRIC_COLUMNS

    row = {column: None for column in ROLE_METRIC_COLUMNS}
    row.update(
        input_sample_id=input_sample_id,
        designed_sample_id=designed_sample_id,
        prediction_id=prediction_id,
        metric_family="role_contract",
        operation_scope="prediction",
        operation_id="role_contract",
        status="error",
        error=error,
    )
    return row


def _record_role_metric_status(
    *,
    status_row: dict[str, Any],
    prediction_id: str,
    prediction_role_rows: list[dict[str, Any]],
) -> None:
    """Record one prediction's role-metric row outcomes in the AF3 status row."""
    status_row["n_role_metric_rows"] += len(prediction_role_rows)
    error_rows = [
        row
        for row in prediction_role_rows
        if row.get("status") not in {"ok", "not_applicable"}
    ]
    status_row["n_role_metric_errors"] += len(error_rows)
    if error_rows:
        error_messages = list(
            dict.fromkeys(str(row.get("error", "")) for row in error_rows)
        )
        _append_status_error(
            status_row,
            "role_metric_errors",
            f"{prediction_id}: " + " | ".join(error_messages),
        )
        return
    status_row["n_role_metric_predictions_success"] += 1


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


# ============================================================================
# AF3 Evaluation Functions
# ============================================================================

@dataclass(frozen=True)
class _Af3EvaluationMode:
    mode: str
    input_dirname: str
    prediction_dirname: str
    json_paths_key: str
    residue_index_by_chain_key: str
    residue_id_layout_key: str
    csv_prefix: str
    title: str


_AF3_EVALUATION_MODES = {
    "ss": _Af3EvaluationMode(
        mode="ss",
        input_dirname="af3_ss_inputs",
        prediction_dirname="af3_ss_preds",
        json_paths_key="af3_ss_json_paths",
        residue_index_by_chain_key="af3_ss_residue_index_by_chain",
        residue_id_layout_key="af3_ss_residue_id_layout",
        csv_prefix="",
        title="AF3 Self-Consistency Evaluation",
    ),
    "tc": _Af3EvaluationMode(
        mode="tc",
        input_dirname="af3_tc_inputs",
        prediction_dirname="af3_tc_preds",
        json_paths_key="af3_tc_json_paths",
        residue_index_by_chain_key="af3_tc_residue_index_by_chain",
        residue_id_layout_key="af3_tc_residue_id_layout",
        csv_prefix="tc_",
        title="AF3 Docking Consistency Evaluation (Template-Conditioned)",
    ),
}


def _resolve_af3_mode(subsample_dict: dict) -> str:
    """Resolve SS/TC from one role-bearing runtime sample before any mutation."""
    pn_unit_roles = subsample_dict.get("pn_unit_roles")
    if pn_unit_roles is None:
        raise ValueError(
            "Automatic AF3 evaluation requires pn_unit_roles on every sample entry"
        )
    roles = normalize_pn_unit_roles(pn_unit_roles)
    template_pn_unit_iids = roles["template_pn_unit_iids"]
    if not template_pn_unit_iids:
        return "ss"
    if len(template_pn_unit_iids) == 1:
        pdb_chain_info = subsample_dict.get("pdb_chain_info")
        if pdb_chain_info is not None:
            protein_pn_unit_iids = set(
                str(value)
                for value in pdb_chain_info.get("protein_pn_unit_iids", [])
            )
            template_pn_unit_iid = str(template_pn_unit_iids[0])
            if template_pn_unit_iid not in protein_pn_unit_iids:
                raise ValueError(
                    "template_pn_unit_iids must select exactly one protein PN unit; "
                    f"got {template_pn_unit_iid!r}, proteins={sorted(protein_pn_unit_iids)}"
                )
        return "tc"
    raise ValueError(
        "Automatic AF3 evaluation supports at most one template PN unit; "
        f"got {template_pn_unit_iids}"
    )


def _partition_sample_dict_by_af3_mode(sample_dict: dict) -> dict[str, dict]:
    """Validate and partition all entries before JSON paths are attached."""
    partitions: dict[str, dict] = {"ss": {}, "tc": {}}
    resolved_modes: list[tuple[str, dict, str]] = []
    for input_sample_id, subsample_dict in sample_dict.items():
        resolved_modes.append(
            (
                input_sample_id,
                subsample_dict,
                _resolve_af3_mode(subsample_dict),
            )
        )
    for input_sample_id, subsample_dict, mode in resolved_modes:
        partitions[mode][input_sample_id] = subsample_dict
    return partitions


def _af3_include_ligand(struct_pred_cfg: DictConfig | dict | None) -> bool:
    if struct_pred_cfg is None:
        return True
    selectable_cfg = (
        struct_pred_cfg
        if isinstance(struct_pred_cfg, DictConfig)
        else OmegaConf.create(struct_pred_cfg)
    )
    value = OmegaConf.select(
        selectable_cfg,
        "af3.include_ligand",
        default=True,
    )
    return config_value_as_bool(value)


def _sample_dict_for_af3_ligand_condition(
    sample_dict: dict,
    *,
    include_ligand: bool,
) -> dict:
    """Return a shallow runtime view with ligand AF3 inputs/metrics disabled."""
    if include_ligand:
        return sample_dict

    conditioned: dict = {}
    for input_sample_id, source_entry in sample_dict.items():
        entry = dict(source_entry)
        source_chain_info = source_entry.get("pdb_chain_info")
        if source_chain_info is None:
            raise ValueError(
                "AF3 ligand-free evaluation requires pdb_chain_info for "
                f"{input_sample_id!r}"
            )
        ligand_pn_unit_iids = {
            str(value)
            for value in source_chain_info.get("ligand_pn_unit_iids", [])
        }
        chain_info = dict(source_chain_info)
        chain_info["ligand_pn_unit_iids"] = []
        chain_info["ligand_ccd_codes"] = []
        if "af3_ligand_ccd_codes" in chain_info:
            chain_info["af3_ligand_ccd_codes"] = []
        entry["pdb_chain_info"] = chain_info

        source_roles = source_entry.get("pn_unit_roles")
        if source_roles is not None:
            roles = normalize_pn_unit_roles(source_roles)
            entry["pn_unit_roles"] = {
                role_name: [
                    pn_unit_iid
                    for pn_unit_iid in role_pn_unit_iids
                    if pn_unit_iid not in ligand_pn_unit_iids
                ]
                for role_name, role_pn_unit_iids in roles.items()
            }
            normalize_pn_unit_roles(
                entry["pn_unit_roles"],
                label=f"ligand-free pn_unit_roles for {input_sample_id}",
            )
        conditioned[input_sample_id] = entry
    return conditioned


def _run_af3_prediction_for_mode(
    *,
    mode: str,
    json_path: str | Path,
    prediction_dir: Path,
    runner_path: str,
    inference_config: DictConfig | dict,
) -> None:
    kwargs = {
        "json_path": str(json_path),
        "out_dir": str(prediction_dir),
        "runner_path": runner_path,
        "inference_config": inference_config,
    }
    if mode == "ss":
        run_af3_single_sequence(**kwargs)
    elif mode == "tc":
        run_af3_template_conditioned(**kwargs)
    else:
        raise ValueError(f"Unsupported AF3 evaluation mode: {mode!r}")


def _restore_af3_prediction_pn_unit_iids(
    pred_atom_array: Any,
    *,
    af3_chain_id_to_pn_unit_iid: dict[str, str],
) -> Any:
    """Restore source PN-unit IIDs on one freshly parsed AF3 prediction."""
    if not hasattr(pred_atom_array, "chain_id"):
        raise ValueError("Parsed AF3 prediction is missing the chain_id annotation")

    predicted_chain_ids = [str(chain_id) for chain_id in pred_atom_array.chain_id]
    expected_chain_ids = set(af3_chain_id_to_pn_unit_iid)
    observed_chain_ids = set(predicted_chain_ids)
    missing_chain_ids = sorted(expected_chain_ids - observed_chain_ids)
    unexpected_chain_ids = sorted(observed_chain_ids - expected_chain_ids)
    if missing_chain_ids or unexpected_chain_ids:
        raise ValueError(
            "AF3 prediction chain IDs do not match serialized source chains; "
            f"missing={missing_chain_ids}, unexpected={unexpected_chain_ids}, "
            f"expected={sorted(expected_chain_ids)}, "
            f"observed={sorted(observed_chain_ids)}"
        )

    restored_pn_unit_iids = np.asarray(
        [
            af3_chain_id_to_pn_unit_iid[chain_id]
            for chain_id in predicted_chain_ids
        ],
        dtype=str,
    )
    pred_atom_array.set_annotation("pn_unit_iid", restored_pn_unit_iids)
    return pred_atom_array


def _restore_af3_prediction_protein_res_ids(
    pred_atom_array: Any,
    *,
    designed_sample_atom_array: Any,
    protein_pn_unit_iids: list[str],
    af3_chain_id_to_pn_unit_iid: dict[str, str],
    json_path: str | Path,
    residue_id_layout: ResidueIdLayout,
) -> Any:
    """Restore source label-sequence IDs lost at the AF3 JSON boundary.

    AF3 numbers each serialized protein from one. The producer explicitly records
    whether positions represent observed residues or the inclusive source label-ID
    span. Validate that layout and the exact JSON CCD sequence before restoring IDs.
    """
    if residue_id_layout not in {"sparse", "full_span"}:
        raise ValueError(
            "residue_id_layout must be 'sparse' or 'full_span'; "
            f"got {residue_id_layout!r}"
        )
    required_annotations = {"pn_unit_iid", "atom_name", "res_name", "res_id"}
    for label, atom_array in (
        ("designed", designed_sample_atom_array),
        ("predicted", pred_atom_array),
    ):
        missing = required_annotations - set(atom_array.get_annotation_categories())
        if missing:
            raise ValueError(
                f"{label} atom array is missing annotations required to restore "
                f"AF3 protein label res_ids: {sorted(missing)}"
            )

    with Path(json_path).open() as handle:
        af3_input = json.load(handle)
    expected_res_names_by_chain_id: dict[str, list[str]] = {}
    for sequence_entry in af3_input.get("sequences", []):
        protein = sequence_entry.get("protein")
        if protein is None:
            continue
        chain_id = protein.get("id")
        if not isinstance(chain_id, str):
            raise ValueError(
                "AF3 protein sequence IDs must be scalar strings for source-ID "
                f"restoration; got {chain_id!r} in {json_path}"
            )
        expected_res_names = [
            get_3_from_1_letter_code(
                letter,
                chain_type=aw_enums.ChainType.POLYPEPTIDE_L,
            )
            for letter in str(protein["sequence"])
        ]
        for modification in protein.get("modifications", []):
            position = int(modification["ptmPosition"])
            if position < 1 or position > len(expected_res_names):
                raise ValueError(
                    f"Invalid AF3 protein modification position {position} for "
                    f"chain {chain_id} in {json_path}"
                )
            expected_res_names[position - 1] = str(
                modification["ptmType"]
            ).removeprefix("CCD_")
        if chain_id in expected_res_names_by_chain_id:
            raise ValueError(f"Duplicate AF3 protein chain ID {chain_id!r} in {json_path}")
        expected_res_names_by_chain_id[chain_id] = expected_res_names

    chain_id_by_pn_unit_iid = {
        pn_unit_iid: chain_id
        for chain_id, pn_unit_iid in af3_chain_id_to_pn_unit_iid.items()
    }
    restored_res_ids = np.asarray(pred_atom_array.res_id).copy()
    for pn_unit_iid in protein_pn_unit_iids:
        chain_id = chain_id_by_pn_unit_iid.get(pn_unit_iid)
        if chain_id is None:
            raise ValueError(f"No AF3 chain ID mapping for protein PN unit {pn_unit_iid}")
        expected_res_names = expected_res_names_by_chain_id.get(chain_id)
        if expected_res_names is None:
            raise ValueError(
                f"AF3 JSON {json_path} has no protein sequence for chain {chain_id}"
            )
        designed_chain = designed_sample_atom_array[
            designed_sample_atom_array.pn_unit_iid == pn_unit_iid
        ]
        pred_chain = pred_atom_array[pred_atom_array.pn_unit_iid == pn_unit_iid]
        designed_residue_starts = get_residue_starts(designed_chain)
        pred_residue_starts = get_residue_starts(pred_chain)
        if len(designed_residue_starts) == 0:
            raise ValueError(f"No designed protein residues found for {pn_unit_iid}")
        designed_res_ids = np.asarray(
            designed_chain.res_id[designed_residue_starts], dtype=int
        )
        if np.any(np.diff(designed_res_ids) <= 0):
            raise ValueError(
                f"Designed protein chain {pn_unit_iid} residue res_ids must be "
                "strictly increasing"
            )
        min_res_id = int(np.min(designed_res_ids))
        max_res_id = int(np.max(designed_res_ids))
        full_span_res_ids = np.arange(min_res_id, max_res_id + 1, dtype=int)
        if residue_id_layout == "sparse":
            source_res_ids = designed_res_ids
        else:
            source_res_ids = full_span_res_ids
        if len(expected_res_names) != len(source_res_ids):
            raise ValueError(
                "AF3 JSON protein length does not match residue_id_layout "
                f"{residue_id_layout!r} for {pn_unit_iid}: "
                f"json={len(expected_res_names)}, expected={len(source_res_ids)}"
            )
        if len(pred_residue_starts) != len(expected_res_names):
            raise ValueError(
                "AF3 predicted protein residue count does not match serialized "
                f"chain {pn_unit_iid}: json={len(expected_res_names)}, "
                f"predicted={len(pred_residue_starts)}"
            )
        pred_names = np.asarray(pred_chain.res_name[pred_residue_starts], dtype=str)
        expected_names = np.asarray(expected_res_names, dtype=str)
        mismatch_indices = np.where(expected_names != pred_names)[0]
        if len(mismatch_indices):
            examples = ", ".join(
                f"position {int(index) + 1}:{expected_names[index]}!={pred_names[index]}"
                for index in mismatch_indices[:10]
            )
            raise ValueError(
                "AF3 predicted protein residue names do not match the exact serialized "
                f"JSON sequence for {pn_unit_iid}: {examples}"
            )

        pred_res_ids = np.asarray(pred_chain.res_id[pred_residue_starts])
        if len(np.unique(pred_res_ids)) != len(pred_res_ids):
            raise ValueError(
                f"Predicted protein chain {pn_unit_iid} has duplicate residue res_ids"
            )
        for pred_res_id, source_res_id in zip(pred_res_ids, source_res_ids):
            residue_mask = (
                (pred_atom_array.pn_unit_iid == pn_unit_iid)
                & (pred_atom_array.res_id == pred_res_id)
            )
            restored_res_ids[residue_mask] = source_res_id

    pred_atom_array.res_id = restored_res_ids
    return pred_atom_array


def evaluate_cached_af3_prediction(
    prediction: CachedAF3Prediction,
    *,
    designed_sample_atom_array: Any,
    reference_sample_atom_array: Any,
    cif_parse_cfg: DictConfig | dict[str, Any],
    preprocess_cfg: DictConfig | dict[str, Any],
    featurizer_cfg: DictConfig | dict[str, Any],
    reference_is_designed: bool,
    reference_pocket_distance: float | None,
    save_pocket_aligned: bool = True,
    pocket_aligned_output_dir: str | Path | None = None,
    save_ca_aligned: bool = True,
    compute_tmalign: bool = True,
    save_tmaligned: bool = True,
) -> AF3PredictionMetrics:
    """Evaluate one existing AF3 model CIF without running AF3.

    Parsing and source-identity restoration fail closed. Role-metric rows retain
    their existing row-local error contract, while self-consistency failures are
    returned separately so callers can persist partial metric results.
    """
    from allatom_design.eval.metrics import compute_self_consistency_metrics_atomarray
    from allatom_design.eval.structure_prediction.AF3.inputs import prepare_af3_prediction

    pred_example = prepare_af3_prediction(
        pdb_path=str(prediction.prediction_path),
        cif_parse_cfg=cif_parse_cfg,
        preprocess_cfg=preprocess_cfg,
        featurizer_cfg=featurizer_cfg,
    )
    pred_atom_array = _restore_af3_prediction_pn_unit_iids(
        pred_example["atom_array"],
        af3_chain_id_to_pn_unit_iid=prediction.af3_chain_id_to_pn_unit_iid,
    )
    protein_pn_unit_iids = [
        str(value)
        for value in prediction.pdb_chain_info.get("protein_pn_unit_iids", [])
    ]
    ligand_pn_unit_iids = [
        str(value)
        for value in prediction.pdb_chain_info.get("ligand_pn_unit_iids", [])
    ]
    pred_atom_array = _restore_af3_prediction_protein_res_ids(
        pred_atom_array,
        designed_sample_atom_array=designed_sample_atom_array,
        protein_pn_unit_iids=protein_pn_unit_iids,
        af3_chain_id_to_pn_unit_iid=prediction.af3_chain_id_to_pn_unit_iid,
        json_path=prediction.json_path,
        residue_id_layout=prediction.residue_id_layout,
    )

    role_metric_rows: list[dict[str, Any]] = []
    if prediction.pn_unit_roles is not None:
        role_metric_rows = _compute_role_metric_rows_for_prediction(
            input_sample_id=prediction.input_sample_id,
            designed_sample_id=prediction.designed_sample_id,
            prediction_id=prediction.prediction_id,
            subsample_dict={"pn_unit_roles": prediction.pn_unit_roles},
            pdb_chain_info=prediction.pdb_chain_info,
            reference_sample_atom_array=reference_sample_atom_array,
            designed_sample_atom_array=designed_sample_atom_array,
            pred_atom_array=pred_atom_array,
            pred_sample_path=prediction.prediction_path,
            input_sample_is_designed=reference_is_designed,
            reference_pocket_distance=reference_pocket_distance,
            save_pocket_aligned=save_pocket_aligned,
            pocket_aligned_output_dir=pocket_aligned_output_dir,
        )

    self_consistency_metrics = None
    self_consistency_error = ""
    try:
        self_consistency_metrics = compute_self_consistency_metrics_atomarray(
            pred_atom_array=pred_atom_array,
            sample_atom_array=designed_sample_atom_array,
            pred_sample_path=prediction.prediction_path,
            save_aligned=save_ca_aligned,
            compute_tmalign=compute_tmalign,
            save_tmaligned=save_tmaligned,
            ligand_pn_unit_iids=ligand_pn_unit_iids,
        )
    except Exception as exc:
        self_consistency_error = str(exc)

    return AF3PredictionMetrics(
        role_metric_rows=role_metric_rows,
        self_consistency_metrics=self_consistency_metrics,
        self_consistency_error=self_consistency_error,
    )


def _evaluate_af3_mode(
    *,
    mode: str,
    sample_dict: dict,
    out_dir: Path,
    struct_pred_cfg: DictConfig,
    cif_parse_cfg: DictConfig,
    preprocess_cfg: DictConfig,
    featurizer_cfg: DictConfig,
    pocket_cfg: DictConfig,
    calculate_metrics_only: bool,
    csv_suffix: str,
    input_sample_is_designed: bool,
    free_atom_arrays_progressively: bool,
    enforce_require_complete_predictions: bool,
    save_pocket_aligned: bool = True,
    save_ca_aligned: bool = True,
    compute_tmalign: bool = True,
    save_tmaligned: bool = True,
) -> list[dict[str, Any]]:
    """Run one SS or TC batch through the shared prediction/metric lifecycle."""
    try:
        mode_spec = _AF3_EVALUATION_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported AF3 evaluation mode: {mode!r}") from exc
    sample_dict = _sample_dict_for_af3_ligand_condition(
        sample_dict,
        include_ligand=_af3_include_ligand(struct_pred_cfg),
    )

    from allatom_design.eval.metrics.role_aware import (
        resolve_role_aware_reference_pocket_distance,
    )

    role_aware_reference_pocket_distance = (
        pocket_cfg.get("role_aware_reference_pocket_distance", None)
        if pocket_cfg is not None
        else None
    )
    role_aware_reference_pocket_distance = (
        resolve_role_aware_reference_pocket_distance(
            role_aware_reference_pocket_distance
        )
    )

    out_dir = Path(out_dir)
    input_dir = out_dir / mode_spec.input_dirname
    prediction_dir = out_dir / mode_spec.prediction_dirname
    input_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating AF3 {mode.upper()} JSON input files...")
    sample_dict = make_af3_json(
        af3_ss_input_dir=input_dir if mode == "ss" else None,
        af3_tc_input_dir=input_dir if mode == "tc" else None,
        sample_dict=sample_dict,
        json_config=struct_pred_cfg.af3.json_config,
        make_ss_input=mode == "ss",
        make_tc_input=mode == "tc",
    )

    af3_runner_path = str(struct_pred_cfg.af3.runner_path)
    af3_inference_config = struct_pred_cfg.af3.inference_config
    strict_input_fingerprint = _strict_af3_input_fingerprint(struct_pred_cfg, mode)
    require_complete_predictions = _require_complete_af3_predictions(struct_pred_cfg)
    per_sample_sc_metrics: dict[str, dict[str, Any]] = {}
    has_any_role_metrics = any(
        subsample_dict.get("pn_unit_roles") is not None
        for subsample_dict in sample_dict.values()
    )
    role_metric_rows: list[dict[str, Any]] | None = [] if has_any_role_metrics else None
    prediction_status_rows: list[dict[str, Any]] = []

    print("\n" + "=" * 80)
    print(f"Running {mode_spec.title}")
    print("=" * 80 + "\n")

    for input_sample_id in tqdm(
        sample_dict.keys(),
        desc=f"AF3 {mode.upper()} predictions",
    ):
        subsample_dict = sample_dict[input_sample_id]
        for dsidx, designed_sample_id in enumerate(subsample_dict["designed_sample_id"]):
            per_sample_sc_metrics[designed_sample_id] = {
                "input_sample_id": input_sample_id
            }

            designed_sample_atom_array = subsample_dict["designed_sample_atom_array"][dsidx]
            reference_sample_atom_array = subsample_dict.get(
                "reference_sample_atom_array",
                designed_sample_atom_array,
            )
            if isinstance(reference_sample_atom_array, list):
                reference_sample_atom_array = reference_sample_atom_array[dsidx]

            pdb_chain_info = subsample_dict["pdb_chain_info"]
            protein_pn_unit_iids = pdb_chain_info["protein_pn_unit_iids"]
            ligand_pn_unit_iids = pdb_chain_info["ligand_pn_unit_iids"]
            af3_chain_id_to_pn_unit_iid = build_af3_chain_id_to_pn_unit_iid(
                protein_pn_unit_iids=protein_pn_unit_iids,
                ligand_pn_unit_iids=ligand_pn_unit_iids,
            )
            has_role_metrics = subsample_dict.get("pn_unit_roles") is not None

            json_path = subsample_dict[mode_spec.json_paths_key][dsidx]
            try:
                residue_index_by_chain = subsample_dict[
                    mode_spec.residue_index_by_chain_key
                ][dsidx]
                residue_id_layout = subsample_dict[
                    mode_spec.residue_id_layout_key
                ][dsidx]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"AF3 {mode.upper()} job {designed_sample_id!r} is missing "
                    "per-job residue index/layout metadata"
                ) from exc
            job_inference_config = inference_config_with_residue_index_by_chain(
                af3_inference_config,
                residue_index_by_chain,
                mode=mode,
            )
            n_expected_predictions = expected_prediction_count_from_json(
                json_path,
                job_inference_config,
                mode,
            )
            status_row = _new_prediction_status_row(
                input_sample_id=input_sample_id,
                designed_sample_id=designed_sample_id,
                mode=mode,
                n_expected_predictions=n_expected_predictions,
            )
            prediction_status_rows.append(status_row)

            if not calculate_metrics_only:
                try:
                    _run_af3_prediction_for_mode(
                        mode=mode,
                        json_path=json_path,
                        prediction_dir=prediction_dir,
                        runner_path=af3_runner_path,
                        inference_config=job_inference_config,
                    )
                except Exception as exc:
                    print(
                        f"AF3 {mode.upper()} prediction failed for "
                        f"input_sample_id={input_sample_id}, "
                        f"designed_sample_id={designed_sample_id}: {exc}"
                    )
                    status_row["af3_status"] = "af3_failed"
                    status_row["af3_error"] = str(exc)
                    continue
                gc.collect()

            prediction_summary = summarize_af3_prediction_outputs(
                out_dir=str(prediction_dir),
                job_name=designed_sample_id,
                expected_count=n_expected_predictions,
                json_path=json_path,
                inference_config=job_inference_config,
                mode=mode,
                strict_input_fingerprint=strict_input_fingerprint,
            )
            _record_prediction_summary(status_row, prediction_summary)
            prediction_paths = prediction_summary["model_cif_paths"]

            if prediction_summary.get("input_fingerprint_ok") is False:
                _append_status_error(
                    status_row,
                    "af3_error",
                    str(prediction_summary.get("input_fingerprint_error", "")),
                )
                _finalize_prediction_status(
                    status_row,
                    has_role_metrics=has_role_metrics,
                )
                continue

            if not prediction_paths:
                print(
                    f"No AF3 {mode.upper()} predicted structure found for "
                    f"input_sample_id={input_sample_id}, "
                    f"designed_sample_id={designed_sample_id}"
                )
                _finalize_prediction_status(
                    status_row,
                    has_role_metrics=has_role_metrics,
                )
                continue

            for pred_idx, prediction_path in enumerate(prediction_paths):
                prediction_id = f"diffusion_{pred_idx}"
                try:
                    prediction_metrics = evaluate_cached_af3_prediction(
                        CachedAF3Prediction(
                            input_sample_id=input_sample_id,
                            designed_sample_id=designed_sample_id,
                            prediction_id=prediction_id,
                            prediction_path=Path(prediction_path),
                            json_path=Path(json_path),
                            residue_id_layout=residue_id_layout,
                            pdb_chain_info=pdb_chain_info,
                            pn_unit_roles=subsample_dict.get("pn_unit_roles"),
                            af3_chain_id_to_pn_unit_iid=(
                                af3_chain_id_to_pn_unit_iid
                            ),
                        ),
                        designed_sample_atom_array=designed_sample_atom_array,
                        reference_sample_atom_array=reference_sample_atom_array,
                        cif_parse_cfg=cif_parse_cfg,
                        preprocess_cfg=preprocess_cfg,
                        featurizer_cfg=featurizer_cfg,
                        reference_is_designed=input_sample_is_designed,
                        reference_pocket_distance=(
                            role_aware_reference_pocket_distance
                        ),
                        save_pocket_aligned=save_pocket_aligned,
                        save_ca_aligned=save_ca_aligned and mode != "tc",
                        compute_tmalign=compute_tmalign,
                        save_tmaligned=save_tmaligned,
                    )
                except Exception as exc:
                    parse_error = f"prediction parse failed: {exc}"
                    print(
                        f"AF3 prediction parsing failed for "
                        f"input_sample_id={input_sample_id}, "
                        f"designed_sample_id={designed_sample_id}, "
                        f"prediction_id={prediction_id}: {exc}"
                    )
                    _append_status_error(
                        status_row,
                        "sc_errors",
                        f"{prediction_id}: {parse_error}",
                    )
                    if has_role_metrics:
                        prediction_role_rows = [
                            _role_contract_error_row(
                                input_sample_id=input_sample_id,
                                designed_sample_id=designed_sample_id,
                                prediction_id=prediction_id,
                                error=parse_error,
                            )
                        ]
                        assert role_metric_rows is not None
                        role_metric_rows.extend(prediction_role_rows)
                        _record_role_metric_status(
                            status_row=status_row,
                            prediction_id=prediction_id,
                            prediction_role_rows=prediction_role_rows,
                        )
                    continue

                if has_role_metrics:
                    prediction_role_rows = prediction_metrics.role_metric_rows
                    assert role_metric_rows is not None
                    role_metric_rows.extend(prediction_role_rows)
                    _record_role_metric_status(
                        status_row=status_row,
                        prediction_id=prediction_id,
                        prediction_role_rows=prediction_role_rows,
                    )

                if prediction_metrics.self_consistency_error:
                    error = prediction_metrics.self_consistency_error
                    print(
                        f"Self-consistency metrics computation failed for "
                        f"input_sample_id={input_sample_id}, "
                        f"designed_sample_id={designed_sample_id}, "
                        f"prediction_id={prediction_id}: {error}"
                    )
                    _append_status_error(
                        status_row,
                        "sc_errors",
                        f"{prediction_id}: {error}",
                    )
                else:
                    assert prediction_metrics.self_consistency_metrics is not None
                    per_sample_sc_metrics[designed_sample_id][
                        prediction_id
                    ] = prediction_metrics.self_consistency_metrics
                    status_row["n_sc_success"] += 1

                del prediction_metrics

            _finalize_prediction_status(
                status_row,
                has_role_metrics=has_role_metrics,
            )

        if free_atom_arrays_progressively:
            subsample_dict.pop("designed_sample_atom_array", None)
        gc.collect()

    _save_metrics_results(
        out_dir=out_dir,
        designed_sample_id_to_per_pred_sc_metrics=per_sample_sc_metrics,
        role_metric_rows=role_metric_rows,
        prediction_status_rows=prediction_status_rows,
        csv_suffix=csv_suffix,
        mode_prefix=mode_spec.csv_prefix,
    )
    if enforce_require_complete_predictions:
        _raise_if_required_predictions_incomplete(
            status_rows=prediction_status_rows,
            require_complete_predictions=require_complete_predictions,
            mode_label=mode.upper(),
        )

    print("\n" + "=" * 80)
    print(f"{mode_spec.title} Complete")
    print(f"Results saved to {out_dir}")
    print("=" * 80 + "\n")
    return prediction_status_rows


def evaluate_af3_self_consistency(
    sample_dict: dict = None,
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
    free_atom_arrays_progressively: bool = False,
    save_pocket_aligned: bool = True,
    save_ca_aligned: bool = True,
    compute_tmalign: bool = True,
    save_tmaligned: bool = True,
) -> None:
    """Compatibility adapter for forced single-sequence AF3 evaluation."""
    # (JH) fixed: metrics-only callers can suppress aligned structure artifacts
    # while preserving the legacy defaults for existing evaluation callers.
    _evaluate_af3_mode(
        mode="ss",
        sample_dict=sample_dict,
        out_dir=out_dir,
        struct_pred_cfg=struct_pred_cfg,
        cif_parse_cfg=cif_parse_cfg,
        preprocess_cfg=preprocess_cfg,
        featurizer_cfg=featurizer_cfg,
        pocket_cfg=pocket_cfg,
        calculate_metrics_only=calculate_metrics_only,
        csv_suffix=csv_suffix,
        input_sample_is_designed=input_sample_is_designed,
        free_atom_arrays_progressively=free_atom_arrays_progressively,
        enforce_require_complete_predictions=True,
        save_pocket_aligned=save_pocket_aligned,
        save_ca_aligned=save_ca_aligned,
        compute_tmalign=compute_tmalign,
        save_tmaligned=save_tmaligned,
    )


def evaluate_af3_docking_consistency(
    sample_dict: dict = None,
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
    free_atom_arrays_progressively: bool = False,
) -> None:
    """Compatibility adapter for forced template-conditioned AF3 evaluation."""
    _evaluate_af3_mode(
        mode="tc",
        sample_dict=sample_dict,
        out_dir=out_dir,
        struct_pred_cfg=struct_pred_cfg,
        cif_parse_cfg=cif_parse_cfg,
        preprocess_cfg=preprocess_cfg,
        featurizer_cfg=featurizer_cfg,
        pocket_cfg=pocket_cfg,
        calculate_metrics_only=calculate_metrics_only,
        csv_suffix=csv_suffix,
        input_sample_is_designed=input_sample_is_designed,
        free_atom_arrays_progressively=free_atom_arrays_progressively,
        enforce_require_complete_predictions=True,
    )


def evaluate_af3_structure_prediction(
    *,
    sample_dict: dict,
    out_dir: Path,
    struct_pred_cfg: DictConfig,
    cif_parse_cfg: DictConfig,
    preprocess_cfg: DictConfig,
    featurizer_cfg: DictConfig,
    pocket_cfg: DictConfig,
    calculate_metrics_only: bool = False,
    csv_suffix: str = "",
    input_sample_is_designed: bool = True,
    free_atom_arrays_progressively: bool = True,
) -> list[dict[str, Any]]:
    """Automatically batch role-bearing samples into SS and one-template TC."""
    partitions = _partition_sample_dict_by_af3_mode(sample_dict)
    all_status_rows: list[dict[str, Any]] = []
    for mode in ("ss", "tc"):
        mode_sample_dict = partitions[mode]
        if not mode_sample_dict:
            continue
        all_status_rows.extend(
            _evaluate_af3_mode(
                mode=mode,
                sample_dict=mode_sample_dict,
                out_dir=out_dir,
                struct_pred_cfg=struct_pred_cfg,
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                featurizer_cfg=featurizer_cfg,
                pocket_cfg=pocket_cfg,
                calculate_metrics_only=calculate_metrics_only,
                csv_suffix=csv_suffix,
                input_sample_is_designed=input_sample_is_designed,
                free_atom_arrays_progressively=free_atom_arrays_progressively,
                enforce_require_complete_predictions=False,
            )
        )

    _raise_if_required_predictions_incomplete(
        status_rows=all_status_rows,
        require_complete_predictions=_require_complete_af3_predictions(
            struct_pred_cfg
        ),
        mode_label="AUTO",
    )
    return all_status_rows


def _save_metrics_results(out_dir: Path = None,
                          designed_sample_id_to_per_pred_sc_metrics: dict = None,
                          role_metric_rows: list[dict[str, Any]] | None = None,
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

    if role_metric_rows is not None:
        from allatom_design.eval.metrics.role_aware import ROLE_METRIC_COLUMNS

        pd.DataFrame(role_metric_rows, columns=ROLE_METRIC_COLUMNS).to_csv(
            Path(out_dir, f"{mode_prefix}all_role_metrics_per_prediction{csv_suffix}.csv"),
            index=False,
        )
    if prediction_status_rows is not None:
        pd.DataFrame(prediction_status_rows).to_csv(
            Path(out_dir, f"{mode_prefix}af3_prediction_status{csv_suffix}.csv"),
            index=False,
        )
