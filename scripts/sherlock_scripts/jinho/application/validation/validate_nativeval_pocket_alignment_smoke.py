#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure.io import pdbx


@dataclass(frozen=True)
class ExpectedCase:
    ligand_pn_unit_iid: str
    alignment_pn_unit_iids: str


EXPECTED = {
    "9bks_binder_A_1_context_B_1": ExpectedCase("B_1", "A_1"),
    "8tnb_binder_A_1-C_1_context_D_1": ExpectedCase("D_1", "A_1;C_1"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _atom_count(path: Path) -> int:
    return len(pdbx.get_structure(pdbx.CIFFile.read(str(path)), model=1))


def _original_prediction_path(aligned_path: Path, ligand_pn_unit_iid: str) -> Path:
    suffix = f"_{ligand_pn_unit_iid}_pocket_aligned.cif"
    _require(
        aligned_path.name.endswith(suffix),
        f"Unexpected aligned filename for {ligand_pn_unit_iid}: {aligned_path}",
    )
    return aligned_path.with_name(aligned_path.name[: -len(suffix)] + ".cif")


def validate(run_dir: Path) -> dict[str, object]:
    status_path = run_dir / "af3_prediction_status.csv"
    role_path = run_dir / "all_role_metrics_per_prediction.csv"
    sc_path = run_dir / "all_sc_metrics_per_designed_sample.csv"
    legacy_docking_path = run_dir / "all_docking_metrics_per_prediction.csv"
    _require(status_path.is_file(), f"Missing AF3 status CSV: {status_path}")
    _require(role_path.is_file(), f"Missing role metric CSV: {role_path}")
    _require(sc_path.is_file(), f"Missing self-consistency CSV: {sc_path}")
    _require(
        not legacy_docking_path.exists(),
        f"Generic evaluator unexpectedly wrote legacy docking CSV: {legacy_docking_path}",
    )

    status_df = pd.read_csv(status_path)
    role_df = pd.read_csv(role_path)
    _require(
        set(status_df["input_sample_id"]) == set(EXPECTED),
        "AF3 status input IDs do not match the two requested smoke cases",
    )
    _require(len(status_df) == len(EXPECTED), "Expected one AF3 status row per case")
    _require((status_df["af3_status"] == "complete").all(), "AF3 status is not complete")
    _require((status_df["n_expected_predictions"] == 1).all(), "Expected one prediction per case")
    _require((status_df["n_found_predictions"] == 1).all(), "Did not find one prediction per case")
    _require((status_df["n_malformed_prediction_dirs"] == 0).all(), "Malformed AF3 output found")
    _require((status_df["n_sc_success"] == 1).all(), "Self-consistency parsing failed")
    _require(
        (status_df["n_role_metric_predictions_success"] == 1).all(),
        "Role metric evaluation failed",
    )
    _require((status_df["n_role_metric_errors"] == 0).all(), "Role metric errors were reported")

    duplicate_key = ["designed_sample_id", "prediction_id", "operation_id"]
    _require(not role_df.duplicated(duplicate_key).any(), "Duplicate role-metric operation keys")
    bad_rows = role_df.loc[role_df["status"] != "ok"]
    _require(
        bad_rows.empty,
        "Non-ok role metric rows: "
        + repr(bad_rows[["input_sample_id", "operation_id", "status", "error"]].to_dict("records")),
    )

    case_reports: dict[str, object] = {}
    for sample_id, expected in EXPECTED.items():
        sample_rows = role_df.loc[role_df["input_sample_id"] == sample_id]
        _require(not sample_rows.empty, f"No role metric rows for {sample_id}")
        placement = sample_rows.loc[
            (sample_rows["metric_family"] == "frame_placement")
            & (sample_rows["operation_scope"] == "target")
        ]
        _require(len(placement) == 1, f"Expected one ligand placement row for {sample_id}")
        row = placement.iloc[0]
        _require(
            row["target_pn_unit_iid"] == expected.ligand_pn_unit_iid,
            f"Wrong ligand target for {sample_id}: {row['target_pn_unit_iid']}",
        )
        _require(row["target_kind"] == "small_molecule", f"Wrong target kind for {sample_id}")
        _require(
            row["alignment_pn_unit_iids"] == expected.alignment_pn_unit_iids,
            f"Wrong protein alignment units for {sample_id}: {row['alignment_pn_unit_iids']}",
        )
        _require(row["alignment_mode"] == "all_atom_pocket_ca", f"Wrong alignment mode for {sample_id}")
        _require(row["pocket_annotation_method"] == "all_atom", f"Wrong pocket method for {sample_id}")
        _require(np.isclose(float(row["pocket_distance"]), 10.0), f"Wrong pocket distance for {sample_id}")
        _require(np.isfinite(row["ligand_rmsd"]), f"Nonfinite ligand RMSD for {sample_id}")
        _require(np.isfinite(row["ligand_plddt"]), f"Nonfinite ligand pLDDT for {sample_id}")
        _require(int(row["frame_matched_ca_count"]) >= 3, f"Too few fit CA atoms for {sample_id}")
        _require(
            float(row["frame_designed_match_coverage"]) > 0.0,
            f"No designed-reference residue matches for {sample_id}",
        )

        aligned_path = Path(str(row["aligned_path"]))
        _require(aligned_path.is_file(), f"Missing pocket-aligned CIF for {sample_id}: {aligned_path}")
        prediction_path = _original_prediction_path(aligned_path, expected.ligand_pn_unit_iid)
        _require(prediction_path.is_file(), f"Missing source prediction CIF: {prediction_path}")
        aligned_atom_count = _atom_count(aligned_path)
        prediction_atom_count = _atom_count(prediction_path)
        _require(
            aligned_atom_count == prediction_atom_count,
            f"Pocket-aligned CIF is not the full prediction for {sample_id}: "
            f"aligned={aligned_atom_count}, source={prediction_atom_count}",
        )
        case_reports[sample_id] = {
            "ligand_pn_unit_iid": expected.ligand_pn_unit_iid,
            "alignment_pn_unit_iids": expected.alignment_pn_unit_iids,
            "frame_matched_ca_count": int(row["frame_matched_ca_count"]),
            "frame_reference_ca_count": int(row["frame_reference_ca_count"]),
            "frame_designed_ca_count": int(row["frame_designed_ca_count"]),
            "frame_prediction_ca_count": int(row["frame_prediction_ca_count"]),
            "ligand_rmsd": float(row["ligand_rmsd"]),
            "ligand_plddt": float(row["ligand_plddt"]),
            "aligned_path": str(aligned_path.resolve()),
            "aligned_atom_count": aligned_atom_count,
        }

    report: dict[str, object] = {
        "version": 1,
        "result": "pass",
        "run_dir": str(run_dir.resolve()),
        "n_cases": len(EXPECTED),
        "diagnostics": {
            "missing_cases": 0,
            "unexpected_cases": 0,
            "incomplete_status_rows": 0,
            "malformed_prediction_dirs": 0,
            "role_metric_errors": 0,
        },
        "cases": case_reports,
    }
    report_path = run_dir / "nativeval_pocket_alignment_smoke_validation.json"
    with report_path.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.run_dir)
    print(
        "Validated nativeval pocket-alignment smoke: "
        f"cases={report['n_cases']}, status=complete, aligned_cifs=full_predictions"
    )


if __name__ == "__main__":
    main()
