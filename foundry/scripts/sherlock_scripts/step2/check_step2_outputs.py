#!/usr/bin/env python3
"""Validate denovoval step2 RFD3 output completion and conditioning contract.

This checker is intentionally limited to run-completion and runtime-contract
validation. It does not score whether generated structures are scientifically
good pockets or whether the ligand is geometrically buried after inference.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

from biotite.structure.io.pdbx import CIFFile

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from constants import default_source_data_root  # type: ignore[import-not-found]
else:
    from .constants import default_source_data_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data-root",
        type=Path,
        default=default_source_data_root(),
        help="Step2 source_data root containing inputs/, rfd3_inputs/, and denovoval_step2_ccds.txt.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Generated sample root for one stage. Defaults to <source-data-root>/../generated_samples/<stage>.",
    )
    parser.add_argument("--stage", default="full", help="Stage label used only to derive/report the default output root.")
    parser.add_argument(
        "--lengths",
        default=None,
        help="Comma- or space-separated lengths to validate. Defaults to manifest lengths.",
    )
    parser.add_argument(
        "--diffusion-batch-size",
        type=int,
        default=None,
        help="Expected diffusion batch size. Defaults to the staged manifest value.",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=None,
        help="Expected RFD3 n_batches. Defaults to the staged manifest value.",
    )
    parser.add_argument(
        "--max-ccds",
        type=int,
        default=None,
        help="Validate only the first N CCDs from denovoval_step2_ccds.txt (useful for smoke stages).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="JSON report path. Defaults to <output-root>/step2_validation_report.json.",
    )
    parser.add_argument(
        "--allow-overfull",
        action="store_true",
        help="Report overfull condition directories without failing the validation status.",
    )
    parser.add_argument(
        "--max-details",
        type=int,
        default=50,
        help="Maximum examples retained per diagnostic list in the JSON report.",
    )
    return parser.parse_args()


def parse_lengths(raw: str | None, manifest: dict[str, Any]) -> list[int]:
    if raw is None:
        lengths = manifest.get("lengths")
        if not isinstance(lengths, list) or not lengths:
            raise ValueError("Manifest missing non-empty 'lengths'; pass --lengths explicitly.")
        return [int(length) for length in lengths]
    return [int(part) for part in raw.replace(",", " ").split() if part]


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def read_ccds(path: Path, max_ccds: int | None) -> list[str]:
    ccds = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if max_ccds is not None:
        if max_ccds <= 0:
            raise ValueError("--max-ccds must be positive")
        return ccds[:max_ccds]
    return ccds


def limited_append(items: list[dict[str, Any]], item: dict[str, Any], *, max_details: int) -> None:
    if len(items) < max_details:
        items.append(item)


def expected_input_path(source_data_root: Path, ccd: str) -> str:
    return str(source_data_root / "inputs" / f"{ccd}.cif")


def check_output_json(
    path: Path,
    *,
    source_data_root: Path,
    ccd: str,
    length: int,
) -> list[str]:
    problems: list[str] = []
    try:
        payload = load_json(path)
    except Exception as exc:
        return [f"invalid_json: {exc}"]
    if not isinstance(payload, dict):
        return ["top_level_not_object"]
    spec = payload.get("specification")
    if not isinstance(spec, dict):
        return ["missing_specification"]

    expected_length = f"{length}-{length}"
    expected_fixed = {ccd: ""}
    expected_buried = {ccd: "ALL"}
    expected_input = expected_input_path(source_data_root, ccd)

    if spec.get("ligand") != ccd:
        problems.append(f"ligand={spec.get('ligand')!r}, expected {ccd!r}")
    if spec.get("length") != expected_length:
        problems.append(f"length={spec.get('length')!r}, expected {expected_length!r}")
    if spec.get("select_fixed_atoms") != expected_fixed:
        problems.append(f"select_fixed_atoms={spec.get('select_fixed_atoms')!r}, expected {expected_fixed!r}")
    if spec.get("select_buried") != expected_buried:
        problems.append(f"select_buried={spec.get('select_buried')!r}, expected {expected_buried!r}")
    if spec.get("allow_ligand_on_existing_chain") is not False:
        problems.append(
            "allow_ligand_on_existing_chain="
            f"{spec.get('allow_ligand_on_existing_chain')!r}, expected False"
        )
    if spec.get("input") != expected_input:
        problems.append(f"input={spec.get('input')!r}, expected {expected_input!r}")
    return problems


def check_output_cif(path: Path, *, length: int) -> list[str]:
    """Validate chain/residue identity using the generated CIF's real reader."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            cif = CIFFile.read(handle)
        atom_site = cif.block["atom_site"]
        label_chains = atom_site["label_asym_id"].as_array(str).tolist()
        auth_chains = atom_site["auth_asym_id"].as_array(str).tolist()
        label_residue_ids = atom_site["label_seq_id"].as_array(str).tolist()
        auth_residue_ids = atom_site["auth_seq_id"].as_array(str).tolist()
    except Exception as exc:
        return [f"invalid_cif: {exc}"]

    problems: list[str] = []
    if label_chains != auth_chains:
        problems.append("label/auth chain IDs differ")
    if label_residue_ids != auth_residue_ids:
        problems.append("label/auth residue IDs differ")
    if problems:
        return problems

    try:
        protein_ids = sorted(
            {
                int(residue_id)
                for chain, residue_id in zip(
                    label_chains, label_residue_ids, strict=True
                )
                if chain == "A"
            }
        )
        ligand_ids = {
            int(residue_id)
            for chain, residue_id in zip(
                label_chains, label_residue_ids, strict=True
            )
            if chain == "L"
        }
    except ValueError as exc:
        return [f"non_integer_residue_id: {exc}"]

    expected_protein_ids = list(range(1, length + 1))
    if protein_ids != expected_protein_ids:
        problems.append(
            f"protein chain A residue IDs are {protein_ids!r}, expected 1..{length}"
        )
    if ligand_ids != {1}:
        problems.append(
            f"ligand chain L residue IDs are {sorted(ligand_ids)!r}, expected [1]"
        )
    return problems


def validate_condition(
    *,
    source_data_root: Path,
    output_root: Path,
    ccd: str,
    length: int,
    expected_per_condition: int,
    max_details: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, int]]:
    condition = f"{ccd}_len{length}"
    condition_dir = output_root / condition
    json_files = sorted(condition_dir.glob("*.json")) if condition_dir.is_dir() else []
    cif_files = sorted(condition_dir.glob("*.cif.gz")) if condition_dir.is_dir() else []
    cif_names = {path.name for path in cif_files}

    missing_pairs: list[dict[str, Any]] = []
    bad_contracts: list[dict[str, Any]] = []
    bad_cif_contracts: list[dict[str, Any]] = []
    counts = {
        "missing_dirs": 0,
        "short_conditions": 0,
        "overfull_conditions": 0,
        "unpaired_outputs": 0,
        "bad_json_contracts": 0,
        "cif_contracts_checked": 0,
        "bad_cif_contracts": 0,
    }

    if not condition_dir.is_dir():
        counts["missing_dirs"] = 1
    if len(json_files) < expected_per_condition or len(cif_files) < expected_per_condition:
        counts["short_conditions"] = 1
    if len(json_files) > expected_per_condition or len(cif_files) > expected_per_condition:
        counts["overfull_conditions"] = 1

    paired_outputs = 0
    checked_cif: str | None = None
    for json_path in json_files:
        expected_cif_name = json_path.with_suffix(".cif.gz").name
        if expected_cif_name in cif_names:
            paired_outputs += 1
            if checked_cif is None:
                cif_path = condition_dir / expected_cif_name
                checked_cif = str(cif_path)
                counts["cif_contracts_checked"] = 1
                cif_problems = check_output_cif(cif_path, length=length)
                if cif_problems:
                    counts["bad_cif_contracts"] = 1
                    limited_append(
                        bad_cif_contracts,
                        {
                            "condition": condition,
                            "cif": str(cif_path),
                            "problems": cif_problems,
                        },
                        max_details=max_details,
                    )
        else:
            counts["unpaired_outputs"] += 1
            limited_append(
                missing_pairs,
                {"condition": condition, "json": str(json_path), "expected_cif": expected_cif_name},
                max_details=max_details,
            )

        problems = check_output_json(json_path, source_data_root=source_data_root, ccd=ccd, length=length)
        if problems:
            counts["bad_json_contracts"] += 1
            limited_append(
                bad_contracts,
                {"condition": condition, "json": str(json_path), "problems": problems},
                max_details=max_details,
            )

    row = {
        "condition": condition,
        "ccd_code": ccd,
        "length": length,
        "condition_dir": str(condition_dir),
        "expected_outputs": expected_per_condition,
        "json_outputs": len(json_files),
        "cif_outputs": len(cif_files),
        "paired_outputs": paired_outputs,
        "checked_cif": checked_cif,
        "complete": (
            condition_dir.is_dir()
            and len(json_files) == expected_per_condition
            and len(cif_files) == expected_per_condition
            and paired_outputs == expected_per_condition
            and counts["bad_json_contracts"] == 0
            and counts["cif_contracts_checked"] == 1
            and counts["bad_cif_contracts"] == 0
        ),
    }
    diagnostics = {
        "missing_pairs": missing_pairs,
        "bad_json_contracts": bad_contracts,
        "bad_cif_contracts": bad_cif_contracts,
    }
    return row, diagnostics, counts


def compare_manifest_parameters(
    *,
    manifest: dict[str, Any],
    lengths: list[int],
    diffusion_batch_size: int,
    n_batches: int,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    manifest_lengths = [int(length) for length in manifest.get("lengths", [])]
    missing_lengths = [length for length in lengths if length not in manifest_lengths]
    if missing_lengths:
        mismatches.append(
            {
                "field": "lengths",
                "severity": "error",
                "manifest": manifest.get("lengths"),
                "validation": lengths,
                "missing_lengths": missing_lengths,
            }
        )
    for key, value in {"diffusion_batch_size": diffusion_batch_size, "n_batches": n_batches}.items():
        if manifest.get(key) != value:
            mismatches.append(
                {
                    "field": key,
                    "severity": "warning",
                    "manifest": manifest.get(key),
                    "validation": value,
                }
            )
    return mismatches


def main() -> None:
    args = parse_args()
    source_data_root = args.source_data_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else (source_data_root.parent / "generated_samples" / args.stage).resolve()
    )
    report_path = (
        args.report_path.resolve()
        if args.report_path is not None
        else output_root / "step2_validation_report.json"
    )

    manifest_path = source_data_root / "rfd3_inputs" / "rfd3_input_manifest.json"
    ccd_list_path = source_data_root / "denovoval_step2_ccds.txt"
    missing_required = [str(path) for path in (manifest_path, ccd_list_path) if not path.exists()]
    if missing_required:
        report = {
            "status": "error",
            "missing_required_files": missing_required,
            "source_data_root": str(source_data_root),
            "output_root": str(output_root),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(2)

    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Manifest is not a JSON object: {manifest_path}")

    lengths = parse_lengths(args.lengths, manifest)
    diffusion_batch_size = (
        int(args.diffusion_batch_size)
        if args.diffusion_batch_size is not None
        else int(manifest.get("diffusion_batch_size", 0))
    )
    n_batches = int(args.n_batches) if args.n_batches is not None else int(manifest.get("n_batches", 0))
    if diffusion_batch_size <= 0 or n_batches <= 0:
        raise SystemExit("--diffusion-batch-size and --n-batches must resolve to positive integers")

    ccds = read_ccds(ccd_list_path, args.max_ccds)
    expected_per_condition = diffusion_batch_size * n_batches
    expected_conditions = len(ccds) * len(lengths)

    manifest_mismatches = compare_manifest_parameters(
        manifest=manifest,
        lengths=lengths,
        diffusion_batch_size=diffusion_batch_size,
        n_batches=n_batches,
    )

    condition_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, list[dict[str, Any]]] = {
        "missing_conditions": [],
        "short_conditions": [],
        "overfull_conditions": [],
        "missing_pairs": [],
        "bad_json_contracts": [],
        "bad_cif_contracts": [],
    }
    totals = {
        "conditions_missing": 0,
        "conditions_short": 0,
        "conditions_overfull": 0,
        "unpaired_outputs": 0,
        "bad_json_contracts": 0,
        "cif_contracts_checked": 0,
        "bad_cif_contracts": 0,
        "json_outputs": 0,
        "cif_outputs": 0,
        "paired_outputs": 0,
    }

    for ccd in ccds:
        for length in lengths:
            row, row_diagnostics, row_counts = validate_condition(
                source_data_root=source_data_root,
                output_root=output_root,
                ccd=ccd,
                length=length,
                expected_per_condition=expected_per_condition,
                max_details=args.max_details,
            )
            condition_rows.append(row)
            totals["json_outputs"] += int(row["json_outputs"])
            totals["cif_outputs"] += int(row["cif_outputs"])
            totals["paired_outputs"] += int(row["paired_outputs"])
            totals["conditions_missing"] += row_counts["missing_dirs"]
            totals["conditions_short"] += row_counts["short_conditions"]
            totals["conditions_overfull"] += row_counts["overfull_conditions"]
            totals["unpaired_outputs"] += row_counts["unpaired_outputs"]
            totals["bad_json_contracts"] += row_counts["bad_json_contracts"]
            totals["cif_contracts_checked"] += row_counts["cif_contracts_checked"]
            totals["bad_cif_contracts"] += row_counts["bad_cif_contracts"]

            if row_counts["missing_dirs"]:
                limited_append(diagnostics["missing_conditions"], row, max_details=args.max_details)
            if row_counts["short_conditions"]:
                limited_append(diagnostics["short_conditions"], row, max_details=args.max_details)
            if row_counts["overfull_conditions"]:
                limited_append(diagnostics["overfull_conditions"], row, max_details=args.max_details)
            for key, values in row_diagnostics.items():
                for value in values:
                    limited_append(diagnostics[key], value, max_details=args.max_details)

    complete_conditions = sum(1 for row in condition_rows if row["complete"])
    manifest_failed_ccds = int(manifest.get("num_ccds_failed", 0) or 0)
    expected_outputs = expected_conditions * expected_per_condition

    blocking_manifest_mismatches = [
        mismatch for mismatch in manifest_mismatches if mismatch.get("severity") == "error"
    ]
    blocking_failures = [
        manifest_failed_ccds > 0,
        bool(blocking_manifest_mismatches),
        totals["conditions_missing"] > 0,
        totals["conditions_short"] > 0,
        totals["unpaired_outputs"] > 0,
        totals["bad_json_contracts"] > 0,
        totals["cif_contracts_checked"] != expected_conditions,
        totals["bad_cif_contracts"] > 0,
        totals["conditions_overfull"] > 0 and not args.allow_overfull,
    ]
    status = "ok" if not any(blocking_failures) else "error"

    report = {
        "status": status,
        "config": {
            "source_data_root": str(source_data_root),
            "output_root": str(output_root),
            "stage": args.stage,
            "lengths": lengths,
            "diffusion_batch_size": diffusion_batch_size,
            "n_batches": n_batches,
            "expected_per_condition": expected_per_condition,
            "max_ccds": args.max_ccds,
            "allow_overfull": args.allow_overfull,
        },
        "source_manifest": {
            "path": str(manifest_path),
            "status": manifest.get("status"),
            "num_ccds_requested": manifest.get("num_ccds_requested"),
            "num_ccds_ok": manifest.get("num_ccds_ok"),
            "num_ccds_failed": manifest_failed_ccds,
            "failed_ccds": manifest.get("failed_ccds", []),
            "total_specs": manifest.get("total_specs"),
            "expected_designs": manifest.get("expected_designs"),
        },
        "summary": {
            "ccds_validated": len(ccds),
            "conditions_expected": expected_conditions,
            "conditions_complete": complete_conditions,
            "conditions_missing": totals["conditions_missing"],
            "conditions_short": totals["conditions_short"],
            "conditions_overfull": totals["conditions_overfull"],
            "expected_outputs": expected_outputs,
            "json_outputs": totals["json_outputs"],
            "cif_outputs": totals["cif_outputs"],
            "paired_outputs": totals["paired_outputs"],
            "unpaired_outputs": totals["unpaired_outputs"],
            "bad_json_contracts": totals["bad_json_contracts"],
            "cif_contracts_checked": totals["cif_contracts_checked"],
            "bad_cif_contracts": totals["bad_cif_contracts"],
            "manifest_parameter_mismatches": len(manifest_mismatches),
            "manifest_failed_ccds": manifest_failed_ccds,
        },
        "manifest_parameter_mismatches": manifest_mismatches,
        "diagnostics": diagnostics,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    summary = report["summary"]
    for key in (
        "conditions_expected",
        "conditions_complete",
        "conditions_missing",
        "conditions_short",
        "conditions_overfull",
        "expected_outputs",
        "json_outputs",
        "cif_outputs",
        "paired_outputs",
        "unpaired_outputs",
        "bad_json_contracts",
        "cif_contracts_checked",
        "bad_cif_contracts",
        "manifest_parameter_mismatches",
        "manifest_failed_ccds",
    ):
        print(f"{key}: {summary[key]}")
    print(f"report: {report_path}")
    print(f"status: {status}")
    raise SystemExit(0 if status == "ok" else 1)


if __name__ == "__main__":
    main()
