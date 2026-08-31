#!/usr/bin/env python3
"""Compute denovoval ligand RASA values and histogram reports."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biotite.structure import filter_amino_acids, sasa
from biotite.structure.info import vdw_radius_single
from biotite.structure.io.pdbx import CIFFile, get_structure


DEFAULT_STEP_ROOT = Path("/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval")
DEFAULT_MANIFEST = DEFAULT_STEP_ROOT / "step3" / "foldseek_clustering" / "input_manifest.tsv"
DEFAULT_CCD_METADATA = DEFAULT_STEP_ROOT / "step1" / "denovoval_ccd_list.tsv"
DEFAULT_OUTPUT_ROOT = DEFAULT_STEP_ROOT / "step4"

SPECIAL_METAL_CLUSTER_CCDS = frozenset({"CUA", "F3S", "FES", "SF4", "VO4"})
CATEGORY_ORDER = [
    "metal_free_ligands",
    "metal_containing_ligands_except_special",
    "CUA_F3S_FES_SF4_VO4",
    "metal_ions",
]
CATEGORY_LABELS = {
    "all": "All",
    "metal_free_ligands": "Metal-free ligands",
    "metal_containing_ligands_except_special": "Metal-containing ligands\n(excluding CUA/F3S/FES/SF4/VO4)",
    "CUA_F3S_FES_SF4_VO4": "CUA, F3S, FES, SF4, VO4",
    "metal_ions": "Metal ions",
    "unknown": "Unknown",
}
CATEGORY_COLORS = {
    "all": "#4a5568",
    "metal_free_ligands": "#2b6cb0",
    "metal_containing_ligands_except_special": "#2f855a",
    "CUA_F3S_FES_SF4_VO4": "#b7791f",
    "metal_ions": "#c53030",
}
OUTPUT_FILES = [
    "ligand_rasa.tsv",
    "ligand_rasa_summary.tsv",
    "ligand_rasa_manifest.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ccd-metadata", type=Path, default=DEFAULT_CCD_METADATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--probe-radius", type=float, default=1.4)
    parser.add_argument("--point-number", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N selected manifest rows.")
    parser.add_argument(
        "--smoke-per-category",
        type=int,
        default=None,
        help="Process up to N rows per plotted category for a fast smoke run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                columns.append(key)
                seen.add(key)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    tmp_path.replace(path)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def ensure_output_root(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in OUTPUT_FILES if (path / name).exists()]
    figure_dir = path / "figures"
    existing.extend(p for p in figure_dir.glob("rasa_histograms.*") if figure_dir.is_dir())
    if existing and not overwrite:
        joined = "\n".join(str(p) for p in existing)
        raise FileExistsError(f"Output files exist; pass --overwrite to replace:\n{joined}")
    if overwrite:
        for file_path in existing:
            file_path.unlink()


def load_ccd_metadata(path: Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(path)
    metadata: dict[str, dict[str, str]] = {}
    required = {"ccd_code", "ligand_class"}
    if not rows:
        raise ValueError(f"CCD metadata is empty: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"CCD metadata missing columns {sorted(missing)}: {path}")
    for row in rows:
        ccd = row["ccd_code"].strip().upper()
        if ccd:
            metadata[ccd] = row
    return metadata


def classify_ccd(ccd: str, metadata: dict[str, dict[str, str]]) -> tuple[str, str]:
    normalized = ccd.strip().upper()
    row = metadata.get(normalized)
    if row is None:
        return "unknown", ""
    ligand_class = row["ligand_class"]
    if ligand_class == "metal_free_ligand":
        return "metal_free_ligands", ligand_class
    if ligand_class == "single_metal_ion":
        return "metal_ions", ligand_class
    if ligand_class == "metal_containing_ligand":
        if normalized in SPECIAL_METAL_CLUSTER_CCDS:
            return "CUA_F3S_FES_SF4_VO4", ligand_class
        return "metal_containing_ligands_except_special", ligand_class
    return "unknown", ligand_class


def load_manifest_rows(
    manifest_path: Path,
    metadata: dict[str, dict[str, str]],
    *,
    limit: int | None,
    smoke_per_category: int | None,
) -> list[dict[str, Any]]:
    rows = read_tsv(manifest_path)
    required = {"staged_id", "condition", "ccd", "length", "source_path", "json_path"}
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Manifest missing columns {sorted(missing)}: {manifest_path}")

    enriched: list[dict[str, Any]] = []
    for row in rows:
        ccd = row["ccd"].strip().upper()
        category, ligand_class = classify_ccd(ccd, metadata)
        enriched.append(
            {
                "staged_id": row["staged_id"],
                "condition": row["condition"],
                "ccd": ccd,
                "length": int(row["length"]),
                "source_path": row["source_path"],
                "json_path": row["json_path"],
                "category": category,
                "ligand_class": ligand_class,
            }
        )

    if smoke_per_category is not None:
        selected: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for row in enriched:
            category = row["category"]
            if category not in CATEGORY_ORDER:
                continue
            if counts[category] >= smoke_per_category:
                continue
            selected.append(row)
            counts[category] += 1
            if all(counts[category] >= smoke_per_category for category in CATEGORY_ORDER):
                break
        enriched = selected

    if limit is not None:
        enriched = enriched[:limit]

    return enriched


def read_cif_atom_array(path: Path) -> Any:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            cif = CIFFile.read(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            cif = CIFFile.read(handle)
    return get_structure(
        cif,
        model=1,
        extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
    )


def finite_coord_mask(atom_array: Any) -> np.ndarray:
    return np.isfinite(atom_array.coord).all(axis=1)


def heavy_atom_mask(atom_array: Any) -> np.ndarray:
    element = np.char.upper(atom_array.element.astype(str))
    return ~np.isin(element, ["H", "D", "T", ""])


def protein_heavy_mask(atom_array: Any) -> np.ndarray:
    amino_acid_mask = filter_amino_acids(atom_array)
    if not bool(np.any(amino_acid_mask)) and "hetero" in atom_array.get_annotation_categories():
        amino_acid_mask = ~atom_array.hetero
    return amino_acid_mask & heavy_atom_mask(atom_array) & finite_coord_mask(atom_array)


def target_ligand_heavy_mask(atom_array: Any, ccd: str) -> np.ndarray:
    """Select the non-polymer ligand residue, not same-named protein residues."""
    return (
        finite_coord_mask(atom_array)
        & heavy_atom_mask(atom_array)
        & np.asarray(atom_array.hetero, dtype=bool)
        & (np.char.upper(atom_array.res_name.astype(str)) == ccd.strip().upper())
    )


def vdw_radius(element: str) -> tuple[float, str | None]:
    normalized = element.strip().upper()
    radius = vdw_radius_single(normalized)
    if radius is None:
        return 1.7, normalized or "<blank>"
    return float(radius), None


def make_vdw_radii(atom_array: Any) -> tuple[np.ndarray, dict[str, int]]:
    radii = np.empty(len(atom_array), dtype=float)
    fallback_counts: Counter[str] = Counter()
    for index, element in enumerate(atom_array.element.astype(str)):
        radius, fallback_element = vdw_radius(element)
        radii[index] = radius
        if fallback_element is not None:
            fallback_counts[fallback_element] += 1
    return radii, dict(fallback_counts)


def target_residue_count(atom_array: Any, indices: np.ndarray) -> int:
    residues = {
        (
            str(atom_array.chain_id[index]),
            int(atom_array.res_id[index]),
            str(atom_array.res_name[index]),
        )
        for index in indices
    }
    return len(residues)


def base_output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "staged_id": row["staged_id"],
        "condition": row["condition"],
        "ccd": row["ccd"],
        "length": row["length"],
        "category": row["category"],
        "ligand_class": row["ligand_class"],
        "source_path": row["source_path"],
        "json_path": row["json_path"],
        "status": "",
        "detail": "",
        "n_atoms": 0,
        "n_target_heavy_atoms": 0,
        "n_target_residues": 0,
        "n_protein_heavy_atoms": 0,
        "free_sasa": "",
        "complex_sasa": "",
        "rasa_value": "",
        "radius_fallback_counts_json": "{}",
    }


def process_manifest_row(args: tuple[dict[str, Any], float, int]) -> dict[str, Any]:
    row, probe_radius, point_number = args
    out = base_output_row(row)

    if row["category"] == "unknown":
        out["status"] = "metadata_missing"
        out["detail"] = f"CCD metadata missing or unsupported ligand_class for {row['ccd']}"
        return out

    source_path = Path(row["source_path"])
    if not source_path.is_file():
        out["status"] = "source_missing"
        out["detail"] = str(source_path)
        return out

    try:
        atom_array = read_cif_atom_array(source_path)
    except Exception as exc:  # noqa: BLE001 - report per-row failures in large batch.
        out["status"] = "parse_failed"
        out["detail"] = f"{type(exc).__name__}: {exc}"
        return out

    out["n_atoms"] = int(len(atom_array))
    target_mask = target_ligand_heavy_mask(atom_array, str(row["ccd"]))
    target_indices = np.where(target_mask)[0]
    protein_mask = protein_heavy_mask(atom_array) & ~target_mask
    protein_indices = np.where(protein_mask)[0]

    out["n_target_heavy_atoms"] = int(len(target_indices))
    out["n_target_residues"] = target_residue_count(atom_array, target_indices) if len(target_indices) else 0
    out["n_protein_heavy_atoms"] = int(len(protein_indices))
    if len(target_indices) == 0:
        out["status"] = "no_target_atoms"
        out["detail"] = f"No finite heavy atoms found for target CCD {row['ccd']}"
        return out

    radii, fallback_counts = make_vdw_radii(atom_array)
    out["radius_fallback_counts_json"] = json.dumps(fallback_counts, sort_keys=True)

    try:
        ligand = atom_array[target_indices]
        ligand_filter = np.ones(len(target_indices), dtype=bool)
        free_by_atom = sasa(
            ligand,
            probe_radius=probe_radius,
            atom_filter=ligand_filter,
            ignore_ions=False,
            point_number=point_number,
            vdw_radii=radii[target_indices],
        )
        free_sasa = float(np.nansum(free_by_atom))
        if math.isnan(free_sasa) or free_sasa <= 0.0:
            out["status"] = "free_sasa_invalid"
            out["detail"] = f"free_sasa={free_sasa}"
            return out

        keep_indices = np.concatenate([target_indices, protein_indices])
        complex_filter = np.zeros(len(keep_indices), dtype=bool)
        complex_filter[: len(target_indices)] = True
        complex_by_atom = sasa(
            atom_array[keep_indices],
            probe_radius=probe_radius,
            atom_filter=complex_filter,
            ignore_ions=False,
            point_number=point_number,
            vdw_radii=radii[keep_indices],
        )
        complex_sasa = float(np.nansum(complex_by_atom[: len(target_indices)]))
    except Exception as exc:  # noqa: BLE001 - report per-row failures in large batch.
        out["status"] = "sasa_failed"
        out["detail"] = f"{type(exc).__name__}: {exc}"
        return out

    if math.isnan(complex_sasa):
        out["status"] = "sasa_nan"
        out["detail"] = "complex_sasa is NaN"
        return out

    out["status"] = "ok"
    out["free_sasa"] = f"{free_sasa:.6f}"
    out["complex_sasa"] = f"{complex_sasa:.6f}"
    out["rasa_value"] = f"{complex_sasa / free_sasa:.8f}"
    return out


def compute_rows(
    rows: list[dict[str, Any]],
    *,
    probe_radius: float,
    point_number: int,
    workers: int,
    chunksize: int,
    progress_interval: int,
) -> list[dict[str, Any]]:
    tasks = [(row, probe_radius, point_number) for row in rows]
    total = len(tasks)
    results: list[dict[str, Any]] = []
    last_log = time.monotonic()

    def maybe_log(index: int, *, force: bool = False) -> None:
        nonlocal last_log
        elapsed_since_log = time.monotonic() - last_log
        if not force and index != total and index % progress_interval != 0 and elapsed_since_log < 30:
            return
        last_log = time.monotonic()
        print(f"Computed RASA for {index:,}/{total:,} rows", flush=True)

    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            results.append(process_manifest_row(task))
            maybe_log(index)
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(process_manifest_row, tasks, chunksize=chunksize), start=1):
            results.append(result)
            maybe_log(index)
    return results


def numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def summary_record(df: pd.DataFrame, *, level: str, category: str, ccd: str = "", length: str = "") -> dict[str, Any]:
    rasa = numeric_series(df.loc[df["status"] == "ok", "rasa_value"]).dropna()
    record: dict[str, Any] = {
        "level": level,
        "category": category,
        "ccd": ccd,
        "length": length,
        "input_count": int(len(df)),
        "ok_count": int(len(rasa)),
        "failed_count": int(len(df) - len(rasa)),
        "zero_rasa_count": int((rasa == 0.0).sum()),
        "rasa_gt_1_count": int((rasa > 1.0).sum()),
    }
    if len(rasa) == 0:
        for key in ("min", "q05", "median", "mean", "q95", "max"):
            record[key] = ""
        return record

    record.update(
        {
            "min": f"{float(rasa.min()):.8f}",
            "q05": f"{float(rasa.quantile(0.05)):.8f}",
            "median": f"{float(rasa.median()):.8f}",
            "mean": f"{float(rasa.mean()):.8f}",
            "q95": f"{float(rasa.quantile(0.95)):.8f}",
            "max": f"{float(rasa.max()):.8f}",
        }
    )
    return record


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    summary = [summary_record(df, level="all", category="all")]

    for category in CATEGORY_ORDER + ["unknown"]:
        subset = df[df["category"] == category]
        if len(subset) > 0:
            summary.append(summary_record(subset, level="category", category=category))

    for (category, ccd), subset in df.groupby(["category", "ccd"], sort=True):
        summary.append(summary_record(subset, level="category_ccd", category=str(category), ccd=str(ccd)))

    for (category, ccd, length), subset in df.groupby(["category", "ccd", "length"], sort=True):
        summary.append(
            summary_record(
                subset,
                level="category_ccd_length",
                category=str(category),
                ccd=str(ccd),
                length=str(length),
            )
        )

    return summary


def plot_histograms(rows: list[dict[str, Any]], output_root: Path) -> dict[str, str]:
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].copy()
    ok["rasa_numeric"] = numeric_series(ok["rasa_value"])
    ok = ok[np.isfinite(ok["rasa_numeric"])]
    if len(ok) == 0:
        raise RuntimeError("No successful RASA rows available for plotting.")

    max_rasa = float(ok["rasa_numeric"].max())
    bin_max = max(1.0, math.ceil(max_rasa * 20.0) / 20.0)
    bins = np.linspace(0.0, bin_max, 61)

    figure_dir = output_root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png_path = figure_dir / "rasa_histograms.png"
    pdf_path = figure_dir / "rasa_histograms.pdf"

    panels = ["all"] + CATEGORY_ORDER
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    axes_flat = list(axes.ravel())

    for ax, category in zip(axes_flat, panels):
        if category == "all":
            values = ok["rasa_numeric"]
            failed = int((df["status"] != "ok").sum())
        else:
            values = ok.loc[ok["category"] == category, "rasa_numeric"]
            failed = int(((df["category"] == category) & (df["status"] != "ok")).sum())

        color = CATEGORY_COLORS.get(category, "#4a5568")
        ax.hist(values, bins=bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.35)
        median_text = "NA" if len(values) == 0 else f"{float(values.median()):.3f}"
        zero_count = int((values == 0.0).sum())
        ax.set_title(
            f"{CATEGORY_LABELS[category]}\n"
            f"n={len(values):,}, failed={failed:,}, zero={zero_count:,}, median={median_text}",
            fontsize=10,
        )
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.22)

    axes_flat[-1].axis("off")
    for ax in axes_flat[:-1]:
        ax.set_xlabel("RASA")
    fig.suptitle("denovoval ligand relative solvent accessible surface area", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def aggregate_radius_fallback_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        payload = row.get("radius_fallback_counts_json") or "{}"
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        counts.update({str(key): int(value) for key, value in parsed.items()})
    return dict(sorted(counts.items()))


def build_manifest(
    *,
    args: argparse.Namespace,
    input_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    figure_paths: dict[str, str],
    started_at: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    status_counts = Counter(row["status"] for row in result_rows)
    category_input_counts = Counter(row["category"] for row in input_rows)
    category_ok_counts = Counter(row["category"] for row in result_rows if row["status"] == "ok")
    ccd_counts = Counter(row["ccd"] for row in input_rows)
    return {
        "status": "complete",
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "command": " ".join(sys.argv),
        "inputs": {
            "manifest": str(args.manifest),
            "ccd_metadata": str(args.ccd_metadata),
            "input_row_count": len(input_rows),
            "unique_ccd_count": len(ccd_counts),
        },
        "parameters": {
            "probe_radius": args.probe_radius,
            "point_number": args.point_number,
            "workers": args.workers,
            "chunksize": args.chunksize,
            "progress_interval": args.progress_interval,
            "limit": args.limit,
            "smoke_per_category": args.smoke_per_category,
            "special_metal_cluster_ccds": sorted(SPECIAL_METAL_CLUSTER_CCDS),
        },
        "outputs": {
            "ligand_rasa_tsv": str(args.output_root / "ligand_rasa.tsv"),
            "summary_tsv": str(args.output_root / "ligand_rasa_summary.tsv"),
            "manifest_json": str(args.output_root / "ligand_rasa_manifest.json"),
            "figures": figure_paths,
        },
        "counts": {
            "result_row_count": len(result_rows),
            "summary_row_count": len(summary_rows),
            "status_counts": dict(sorted(status_counts.items())),
            "category_input_counts": dict(sorted(category_input_counts.items())),
            "category_ok_counts": dict(sorted(category_ok_counts.items())),
            "zero_rasa_count": int(
                sum(
                    1
                    for row in result_rows
                    if row["status"] == "ok" and float(row["rasa_value"]) == 0.0
                )
            ),
            "rasa_gt_1_count": int(
                sum(
                    1
                    for row in result_rows
                    if row["status"] == "ok" and float(row["rasa_value"]) > 1.0
                )
            ),
            "radius_fallback_counts": aggregate_radius_fallback_counts(result_rows),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def main() -> None:
    args = parse_args()
    started_at = utc_now()
    start = time.time()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be >= 1")
    if args.progress_interval < 1:
        raise ValueError("--progress-interval must be >= 1")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.smoke_per_category is not None and args.smoke_per_category < 1:
        raise ValueError("--smoke-per-category must be >= 1")

    ensure_output_root(args.output_root, args.overwrite)
    metadata = load_ccd_metadata(args.ccd_metadata)
    input_rows = load_manifest_rows(
        args.manifest,
        metadata,
        limit=args.limit,
        smoke_per_category=args.smoke_per_category,
    )
    if not input_rows:
        raise RuntimeError("No manifest rows selected.")

    print(
        f"Selected {len(input_rows):,} rows; "
        f"workers={args.workers}, chunksize={args.chunksize}, point_number={args.point_number}",
        flush=True,
    )
    print(f"Category counts: {dict(Counter(row['category'] for row in input_rows))}", flush=True)

    result_rows = compute_rows(
        input_rows,
        probe_radius=args.probe_radius,
        point_number=args.point_number,
        workers=args.workers,
        chunksize=args.chunksize,
        progress_interval=args.progress_interval,
    )
    summary_rows = build_summary(result_rows)
    write_tsv(result_rows, args.output_root / "ligand_rasa.tsv")
    write_tsv(summary_rows, args.output_root / "ligand_rasa_summary.tsv")

    figure_paths: dict[str, str] = {}
    if not args.skip_plots:
        figure_paths = plot_histograms(result_rows, args.output_root)

    manifest = build_manifest(
        args=args,
        input_rows=input_rows,
        result_rows=result_rows,
        summary_rows=summary_rows,
        figure_paths=figure_paths,
        started_at=started_at,
        elapsed_seconds=time.time() - start,
    )
    write_json(manifest, args.output_root / "ligand_rasa_manifest.json")

    print(f"Status counts: {manifest['counts']['status_counts']}", flush=True)
    print(f"Wrote {args.output_root / 'ligand_rasa.tsv'}", flush=True)
    if figure_paths:
        print(f"Wrote {figure_paths['png']}", flush=True)


if __name__ == "__main__":
    main()
