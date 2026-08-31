#!/usr/bin/env python3
"""Validate Step4 ligand RASA artifacts and write Markdown/JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SPECIAL_METAL_CLUSTER_CCDS = frozenset({"CUA", "F3S", "FES", "SF4", "VO4"})
CATEGORY_LABELS = {
    "metal_free_ligands": "metal-free ligand",
    "metal_containing_ligands_except_special": "metal-containing ligand",
    "CUA_F3S_FES_SF4_VO4": "metal cluster",
    "metal_ions": "metal ion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--ccd-metadata", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[150, 300])
    parser.add_argument("--ccds", nargs="+", default=None)
    parser.add_argument("--expected-ccd-count", type=int, required=True)
    parser.add_argument("--expected-per-condition", type=int, required=True)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def load_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def load_ccd_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"CCD metadata has no header: {path}")
        fieldnames = list(reader.fieldnames)
        rows = [
            {field: row.get(field, "") for field in fieldnames}
            for row in reader
        ]
    return pd.DataFrame(rows, columns=fieldnames)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def classify(ccd: str, ligand_class: str) -> str:
    if ligand_class == "metal_free_ligand":
        return "metal_free_ligands"
    if ligand_class == "single_metal_ion":
        return "metal_ions"
    if ligand_class == "metal_containing_ligand":
        if ccd in SPECIAL_METAL_CLUSTER_CCDS:
            return "CUA_F3S_FES_SF4_VO4"
        return "metal_containing_ligands_except_special"
    raise ValueError(f"Unsupported ligand class for {ccd}: {ligand_class!r}")


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def main() -> int:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: object = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    metadata = load_ccd_metadata(args.ccd_metadata)
    metadata["ccd_code"] = metadata["ccd_code"].astype(str).str.upper()
    if args.ccds:
        requested = {ccd.upper() for ccd in args.ccds}
        metadata = metadata[metadata["ccd_code"].isin(requested)].copy()
    metadata["category"] = [
        classify(str(row.ccd_code), str(row.ligand_class))
        for row in metadata.itertuples(index=False)
    ]
    ccds = set(metadata["ccd_code"])
    expected_groups = {(ccd, length) for ccd in ccds for length in args.lengths}

    input_df = load_tsv(args.input_manifest)
    rasa_path = args.output_root / "ligand_rasa.tsv"
    summary_path = args.output_root / "ligand_rasa_summary.tsv"
    manifest_path = args.output_root / "ligand_rasa_manifest.json"
    rasa_df = load_tsv(rasa_path)
    summary_df = load_tsv(summary_path)
    manifest = load_json(manifest_path)

    for frame in (input_df, rasa_df):
        frame["ccd"] = frame["ccd"].astype(str).str.upper()
        frame["length"] = pd.to_numeric(frame["length"], errors="raise").astype(int)

    input_ids = set(input_df["staged_id"].astype(str))
    rasa_ids = set(rasa_df["staged_id"].astype(str))
    group_counts = input_df.groupby(["ccd", "length"]).size()
    input_groups = set(group_counts.index)
    expected_rows = len(expected_groups) * args.expected_per_condition
    status_counts = {
        str(key): int(value)
        for key, value in rasa_df["status"].value_counts().sort_index().items()
    }
    rasa_values = pd.to_numeric(rasa_df["rasa_value"], errors="coerce")

    check("metadata_ccd_count", len(metadata) == args.expected_ccd_count, len(metadata))
    check("hec_absent_from_metadata", "HEC" not in ccds)
    check("input_groups_exact", input_groups == expected_groups)
    check(
        "input_rows_per_condition",
        len(group_counts) == len(expected_groups)
        and bool((group_counts == args.expected_per_condition).all()),
        group_counts.value_counts().sort_index().to_dict(),
    )
    check("input_row_count", len(input_df) == expected_rows, len(input_df))
    check("input_ids_unique", input_df["staged_id"].is_unique)
    check("rasa_row_count", len(rasa_df) == expected_rows, len(rasa_df))
    check("rasa_ids_unique", rasa_df["staged_id"].is_unique)
    check("rasa_id_set_matches_input", rasa_ids == input_ids)
    check("rasa_all_ok", status_counts == {"ok": expected_rows}, status_counts)
    check("rasa_unknown_category_absent", not bool((rasa_df["category"] == "unknown").any()))
    check("hec_absent_from_rasa", not bool((rasa_df["ccd"] == "HEC").any()))
    check("rasa_values_finite", bool(np.isfinite(rasa_values.to_numpy(dtype=float)).all()))
    check("rasa_values_nonnegative", bool((rasa_values >= 0.0).all()))
    check("rasa_values_lte_one", bool((rasa_values <= 1.0 + 1e-8).all()))
    check("step4_manifest_complete", manifest.get("status") == "complete")
    check(
        "step4_manifest_row_count",
        int(manifest.get("counts", {}).get("result_row_count", -1)) == expected_rows,
        manifest.get("counts", {}).get("result_row_count"),
    )
    check("step4_summary_nonempty", len(summary_df) > 0, len(summary_df))

    figure_paths = [
        args.output_root / "figures" / "rasa_histograms.png",
        args.output_root / "figures" / "rasa_histograms.pdf",
    ]
    if not args.skip_figures:
        check(
            "histogram_figures_exist",
            all(path.is_file() and path.stat().st_size > 0 for path in figure_paths),
            [str(path) for path in figure_paths],
        )

    expected_category_counts = {
        category: int(
            len(metadata[metadata["category"] == category])
            * len(args.lengths)
            * args.expected_per_condition
        )
        for category in sorted(set(metadata["category"]))
    }
    actual_category_counts = {
        str(key): int(value)
        for key, value in rasa_df["category"].value_counts().sort_index().items()
    }
    check(
        "category_counts_exact",
        actual_category_counts == expected_category_counts,
        actual_category_counts,
    )

    stats_rows: list[list[object]] = []
    for category, group in rasa_df.assign(rasa_numeric=rasa_values).groupby(
        "category", sort=True
    ):
        values = group["rasa_numeric"]
        stats_rows.append(
            [
                CATEGORY_LABELS.get(str(category), str(category)),
                len(group),
                f"{values.min():.4f}",
                f"{values.quantile(0.05):.4f}",
                f"{values.median():.4f}",
                f"{values.mean():.4f}",
                f"{values.quantile(0.95):.4f}",
                f"{values.max():.4f}",
            ]
        )

    status = "complete" if all(item["passed"] for item in checks) else "failed"
    payload = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inputs": {
            "input_manifest": str(args.input_manifest),
            "ccd_metadata": str(args.ccd_metadata),
        },
        "outputs": {
            "output_root": str(args.output_root),
            "ligand_rasa_tsv": str(rasa_path),
            "summary_tsv": str(summary_path),
            "manifest_json": str(manifest_path),
            "figures": [str(path) for path in figure_paths],
        },
        "counts": {
            "ccd_count": len(metadata),
            "condition_count": len(expected_groups),
            "input_rows": len(input_df),
            "rasa_rows": len(rasa_df),
            "status_counts": status_counts,
            "category_counts": actual_category_counts,
        },
        "checks": checks,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Denovoval ss1p0 gamma1p0 Step4 RASA report",
        "",
        f"- Status: **{status}**",
        f"- Generated: {payload['generated_at']}",
        f"- Slurm job: `{payload['slurm_job_id'] or 'interactive'}`",
        f"- CCDs: {len(metadata)}",
        f"- Conditions: {len(expected_groups)}",
        f"- Input rows: {len(input_df):,}",
        f"- RASA rows: {len(rasa_df):,}",
        f"- HEC present: {'yes' if 'HEC' in set(rasa_df['ccd']) else 'no'}",
        "",
        "## Method",
        "",
        "- Target: all finite heavy atoms whose residue name matches the CCD",
        "- Metric: complex SASA / free target SASA",
        "- Probe radius: 1.4 Å",
        "- Surface points: 1,000",
        "",
        "## Category summary",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Category", "Rows", "Min", "Q05", "Median", "Mean", "Q95", "Max"],
            stats_rows,
        )
    )
    lines.extend(["", "## Validation", ""])
    lines.extend(
        markdown_table(
            ["Check", "Passed", "Detail"],
            [
                [item["name"], "yes" if item["passed"] else "NO", item["detail"]]
                for item in checks
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- RASA rows: `{rasa_path}`",
            f"- Summary: `{summary_path}`",
            f"- Manifest: `{manifest_path}`",
            f"- Machine-readable validation: `{args.report_json}`",
            "",
            "Step6 was not run by this job.",
            "",
        ]
    )
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
