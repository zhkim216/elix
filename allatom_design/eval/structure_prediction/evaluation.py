"""Predictor-neutral evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PredictionMetrics:
    role_metric_rows: list[dict[str, Any]]
    self_consistency_metrics: dict[str, Any] | None
    self_consistency_error: str


def _clean_metric_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text and text.lower() != "nan" else None


def _ligand_smiles_by_iid(
    pdb_chain_info: dict[str, Any],
) -> dict[str, str | None] | None:
    ligand_iids = [
        str(value)
        for value in pdb_chain_info.get("ligand_pn_unit_iids", [])
    ]
    for key in ("ligand_smiles", "ligand_smiles_by_iid"):
        values = pdb_chain_info.get(key)
        if values is None:
            continue
        if isinstance(values, dict):
            result = {
                iid: _clean_metric_value(values.get(iid)) for iid in ligand_iids
            }
        elif isinstance(values, str):
            result = (
                {ligand_iids[0]: _clean_metric_value(values)}
                if len(ligand_iids) == 1
                else {}
            )
        else:
            result = {
                iid: _clean_metric_value(value)
                for iid, value in zip(ligand_iids, values)
            }
        if any(value is not None for value in result.values()):
            return result
    return None


def role_contract_error_row(
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


def evaluate_restored_prediction(
    *,
    input_sample_id: str,
    designed_sample_id: str,
    prediction_id: str,
    pred_atom_array: Any,
    designed_sample_atom_array: Any,
    reference_sample_atom_array: Any,
    pdb_chain_info: dict[str, Any],
    pn_unit_roles: dict[str, Any] | None,
    pred_sample_path: str | Path,
    reference_is_designed: bool,
    reference_pocket_distance: float | None,
    save_pocket_aligned: bool = True,
    pocket_aligned_output_dir: str | Path | None = None,
    save_ca_aligned: bool = True,
    compute_tmalign: bool = True,
    save_tmaligned: bool = True,
) -> PredictionMetrics:
    from allatom_design.eval.metrics import (
        compute_self_consistency_metrics_atomarray,
    )
    from allatom_design.eval.metrics.role_aware import (
        compute_role_aware_metrics_atomarray,
    )

    role_rows: list[dict[str, Any]] = []
    if pn_unit_roles is not None:
        identifiers = {
            "input_sample_id": input_sample_id,
            "designed_sample_id": designed_sample_id,
            "prediction_id": prediction_id,
        }
        try:
            role_rows = compute_role_aware_metrics_atomarray(
                pred_atom_array=pred_atom_array,
                sample_atom_array=designed_sample_atom_array,
                reference_atom_array=reference_sample_atom_array,
                reference_is_designed=reference_is_designed,
                reference_pocket_distance=reference_pocket_distance,
                pn_unit_roles=pn_unit_roles,
                pred_sample_path=pred_sample_path,
                ligand_smiles_by_iid=_ligand_smiles_by_iid(pdb_chain_info),
                save_aligned=save_pocket_aligned,
                aligned_output_dir=pocket_aligned_output_dir,
            )
            if not role_rows:
                raise RuntimeError(
                    "Role-aware metric computation returned no operation rows"
                )
            role_rows = [
                {**identifiers, **metric_row} for metric_row in role_rows
            ]
        except Exception as exc:
            role_rows = [role_contract_error_row(**identifiers, error=str(exc))]

    ligand_iids = [
        str(value)
        for value in pdb_chain_info.get("ligand_pn_unit_iids", [])
    ]
    self_consistency_metrics = None
    self_consistency_error = ""
    try:
        self_consistency_metrics = compute_self_consistency_metrics_atomarray(
            pred_atom_array=pred_atom_array,
            sample_atom_array=designed_sample_atom_array,
            pred_sample_path=pred_sample_path,
            save_aligned=save_ca_aligned,
            compute_tmalign=compute_tmalign,
            save_tmaligned=save_tmaligned,
            ligand_pn_unit_iids=ligand_iids,
        )
    except Exception as exc:
        self_consistency_error = str(exc)
    return PredictionMetrics(
        role_metric_rows=role_rows,
        self_consistency_metrics=self_consistency_metrics,
        self_consistency_error=self_consistency_error,
    )


def save_prediction_metrics(
    *,
    out_dir: str | Path,
    designed_sample_id_to_per_pred_sc_metrics: dict[str, dict[str, Any]],
    role_metric_rows: list[dict[str, Any]] | None,
    prediction_status_rows: list[dict[str, Any]],
    status_filename: str,
    csv_suffix: str = "",
) -> None:
    out_dir = Path(out_dir)
    sc_frame = pd.DataFrame.from_dict(
        designed_sample_id_to_per_pred_sc_metrics, orient="index"
    )
    sc_frame.reset_index().rename(
        columns={"index": "designed_sample_id"}
    ).to_csv(
        out_dir / f"all_sc_metrics_per_designed_sample{csv_suffix}.csv",
        index=False,
    )
    if role_metric_rows is not None:
        from allatom_design.eval.metrics.role_aware import ROLE_METRIC_COLUMNS

        pd.DataFrame(
            role_metric_rows, columns=ROLE_METRIC_COLUMNS
        ).to_csv(
            out_dir / f"all_role_metrics_per_prediction{csv_suffix}.csv",
            index=False,
        )
    pd.DataFrame(prediction_status_rows).to_csv(
        out_dir / f"{status_filename}{csv_suffix}.csv",
        index=False,
    )
