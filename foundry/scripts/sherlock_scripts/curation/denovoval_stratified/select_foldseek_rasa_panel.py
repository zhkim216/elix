#!/usr/bin/env python3
"""Select Foldseek-diverse denovoval samples with absolute RASA tiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DENOVOVAL_ROOT = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/curation/ver2/outputs/denovoval"
)
DEFAULT_STEP3_ROOT = DENOVOVAL_ROOT / "step3" / "foldseek_clustering"
DEFAULT_RASA_TSV = DENOVOVAL_ROOT / "step4" / "ligand_rasa.tsv"
DEFAULT_OUTPUT_ROOT = DENOVOVAL_ROOT / "step6"

SMALL_MOLECULE_CATEGORIES = frozenset(
    {
        "metal_free_ligands",
        "metal_containing_ligands_except_special",
    }
)
METAL_OR_CLUSTER_CATEGORIES = frozenset(
    {
        "metal_ions",
        "CUA_F3S_FES_SF4_VO4",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step3-root", type=Path, default=DEFAULT_STEP3_ROOT)
    parser.add_argument("--rasa-tsv", type=Path, default=DEFAULT_RASA_TSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--low-rasa-cutoff", type=float, default=0.06)
    parser.add_argument("--max-rasa-cutoff", type=float, default=0.20)
    parser.add_argument("--small-molecule-low-quota", type=int, default=5)
    parser.add_argument("--small-molecule-total-quota", type=int, default=10)
    parser.add_argument("--metal-or-cluster-low-quota", type=int, default=10)
    parser.add_argument("--metal-or-cluster-total-quota", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=216)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--ccds", nargs="+", default=None)
    parser.add_argument("--lengths", type=int, nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-stage-files", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "selected": output_root / "selected_samples.tsv",
        "summary": output_root / "selection_summary.tsv",
        "tiers": output_root / "rasa_tier_counts.tsv",
        "audit": output_root / "candidate_audit.tsv",
        "manifest": output_root / "manifest.json",
        "validation": output_root / "validation_report.json",
        "selected_files": output_root / "selected_samples",
    }


def prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    paths = output_paths(output_root)
    stale_paths = [output_root / "rasa_q90_cutoffs.tsv"]
    existing = [path for path in list(paths.values()) + stale_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"step6 outputs already exist under {output_root}; pass --overwrite")
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def read_tsv(path: Path, *, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return df


def require_unique(df: pd.DataFrame, column: str, label: str) -> None:
    if not df[column].is_unique:
        duplicated = sorted(df.loc[df[column].duplicated(), column].head(10).tolist())
        raise ValueError(f"{label} has duplicate {column} values; examples={duplicated}")


def quota_pair_for_category(
    category: str,
    *,
    small_low_quota: int,
    small_total_quota: int,
    metal_low_quota: int,
    metal_total_quota: int,
) -> tuple[int, int]:
    if category in SMALL_MOLECULE_CATEGORIES:
        return small_low_quota, small_total_quota
    if category in METAL_OR_CLUSTER_CATEGORIES:
        return metal_low_quota, metal_total_quota
    raise ValueError(f"unknown category for quota assignment: {category}")


def stable_group_seed(seed: int, ccd: str, length: str, stage: str = "group") -> int:
    digest = hashlib.sha256(f"{seed}|{ccd}|{length}|{stage}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def clean_float(value: float | int | np.floating[Any]) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.8f}"


def validate_args(args: argparse.Namespace) -> None:
    if args.low_rasa_cutoff < 0.0:
        raise ValueError("--low-rasa-cutoff must be non-negative")
    if args.max_rasa_cutoff <= args.low_rasa_cutoff:
        raise ValueError("--max-rasa-cutoff must be greater than --low-rasa-cutoff")
    quota_pairs = [
        ("small molecule", args.small_molecule_low_quota, args.small_molecule_total_quota),
        ("metal or cluster", args.metal_or_cluster_low_quota, args.metal_or_cluster_total_quota),
    ]
    for label, low_quota, total_quota in quota_pairs:
        if low_quota < 0:
            raise ValueError(f"{label} low quota must be non-negative")
        if total_quota < 1:
            raise ValueError(f"{label} total quota must be positive")
        if low_quota > total_quota:
            raise ValueError(f"{label} low quota cannot exceed total quota")


def validate_inputs(
    input_manifest: pd.DataFrame,
    clusters: pd.DataFrame,
    rasa: pd.DataFrame,
) -> dict[str, Any]:
    require_unique(input_manifest, "staged_id", "input_manifest")
    require_unique(clusters, "member_id", "foldseek_clusters")
    require_unique(rasa, "staged_id", "ligand_rasa")

    input_ids = set(input_manifest["staged_id"])
    cluster_ids = set(clusters["member_id"])
    rasa_ids = set(rasa["staged_id"])
    diagnostics = {
        "input_manifest_rows": int(len(input_manifest)),
        "foldseek_cluster_rows": int(len(clusters)),
        "rasa_rows": int(len(rasa)),
        "input_missing_from_clusters": sorted(input_ids - cluster_ids)[:20],
        "cluster_missing_from_input": sorted(cluster_ids - input_ids)[:20],
        "input_missing_from_rasa": sorted(input_ids - rasa_ids)[:20],
        "rasa_missing_from_input": sorted(rasa_ids - input_ids)[:20],
    }
    if input_ids != cluster_ids:
        raise ValueError("input_manifest staged_id set does not match foldseek member_id set")
    if input_ids != rasa_ids:
        raise ValueError("input_manifest staged_id set does not match ligand_rasa staged_id set")
    return diagnostics


def build_candidates(
    *,
    input_manifest: pd.DataFrame,
    clusters: pd.DataFrame,
    rasa: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    cluster_map = clusters[["member_id", "representative_id"]].rename(
        columns={"member_id": "staged_id", "representative_id": "foldseek_cluster_id"}
    )
    input_cols = input_manifest[
        [
            "staged_id",
            "condition",
            "ccd",
            "length",
            "source_path",
            "json_path",
            "source_size",
            "source_mtime",
        ]
    ].rename(
        columns={
            "condition": "input_condition",
            "ccd": "input_ccd",
            "length": "input_length",
            "source_path": "input_source_path",
            "json_path": "input_json_path",
        }
    )

    candidates = rasa.merge(input_cols, on="staged_id", how="inner", validate="one_to_one")
    candidates = candidates.merge(cluster_map, on="staged_id", how="inner", validate="one_to_one")

    mismatches = candidates[
        (candidates["condition"] != candidates["input_condition"])
        | (candidates["ccd"] != candidates["input_ccd"])
        | (candidates["length"] != candidates["input_length"])
        | (candidates["source_path"] != candidates["input_source_path"])
    ]
    if len(mismatches):
        examples = mismatches["staged_id"].head(10).tolist()
        raise ValueError(f"step3/step4 metadata mismatch for staged_id examples={examples}")

    if args.ccds:
        ccds = set(args.ccds)
        candidates = candidates[candidates["ccd"].isin(ccds)].copy()
    if args.lengths:
        lengths = {str(length) for length in args.lengths}
        candidates = candidates[candidates["length"].isin(lengths)].copy()

    if args.max_groups is not None:
        if args.max_groups <= 0:
            raise ValueError("--max-groups must be positive when provided")
        groups = (
            candidates[["ccd", "length"]]
            .drop_duplicates()
            .sort_values(["ccd", "length"], kind="mergesort")
            .head(args.max_groups)
        )
        candidates = candidates.merge(groups, on=["ccd", "length"], how="inner")

    if candidates.empty:
        raise ValueError("no candidates remain after filters")

    candidates["rasa_numeric"] = pd.to_numeric(candidates["rasa_value"], errors="coerce")
    rasa_numeric = candidates["rasa_numeric"].to_numpy(dtype=float)
    status_ok = candidates["status"].eq("ok").to_numpy()
    finite_rasa = np.isfinite(rasa_numeric)

    candidates["eligible_low_rasa"] = status_ok & finite_rasa & (rasa_numeric <= args.low_rasa_cutoff)
    candidates["eligible_backfill_band"] = (
        status_ok
        & finite_rasa
        & (rasa_numeric > args.low_rasa_cutoff)
        & (rasa_numeric <= args.max_rasa_cutoff)
    )
    candidates["eligible_after_rasa"] = status_ok & finite_rasa & (rasa_numeric <= args.max_rasa_cutoff)

    reasons = np.full(len(candidates), "eligible_low_rasa", dtype=object)
    reasons[candidates["eligible_backfill_band"].to_numpy()] = "eligible_backfill_band"
    reasons[status_ok & finite_rasa & (rasa_numeric > args.max_rasa_cutoff)] = "rasa_gt_max_cutoff"
    reasons[status_ok & ~finite_rasa] = "missing_or_nonfinite_rasa"
    reasons[~status_ok] = "rasa_status_not_ok"
    candidates["eligibility_reason"] = reasons
    candidates["low_rasa_cutoff"] = args.low_rasa_cutoff
    candidates["max_rasa_cutoff"] = args.max_rasa_cutoff
    return candidates


def empty_selection(group: pd.DataFrame, *, low_quota: int, total_quota: int, seed: int) -> pd.DataFrame:
    selected = group.head(0).copy()
    selected["selection_tier"] = ""
    selected["selection_tier_rank"] = []
    selected["selection_rank"] = []
    selected["selection_quota"] = total_quota
    selected["low_selection_quota"] = low_quota
    selected["total_selection_quota"] = total_quota
    selected["group_random_seed"] = str(seed)
    return selected


def representative_rows_by_cluster(pool: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    if pool.empty:
        return pool.head(0).copy()
    cluster_rows: list[pd.Series] = []
    for _, cluster_group in pool.sort_values("staged_id", kind="mergesort").groupby(
        "foldseek_cluster_id", sort=True
    ):
        position = int(rng.integers(0, len(cluster_group)))
        cluster_rows.append(cluster_group.iloc[position])
    return pd.DataFrame(cluster_rows).reset_index(drop=True)


def choose_cluster_rows(
    pool: pd.DataFrame,
    *,
    quota: int,
    rng: np.random.Generator,
    selection_tier: str,
) -> pd.DataFrame:
    if quota <= 0 or pool.empty:
        out = pool.head(0).copy()
        out["selection_tier"] = ""
        out["selection_tier_rank"] = []
        return out
    cluster_df = representative_rows_by_cluster(pool, rng)
    if cluster_df.empty:
        out = pool.head(0).copy()
        out["selection_tier"] = ""
        out["selection_tier_rank"] = []
        return out
    order = rng.permutation(len(cluster_df))
    selected = cluster_df.iloc[order[: min(quota, len(cluster_df))]].copy().reset_index(drop=True)
    selected["selection_tier"] = selection_tier
    selected["selection_tier_rank"] = np.arange(1, len(selected) + 1)
    return selected


def select_one_group(
    group: pd.DataFrame,
    *,
    args: argparse.Namespace,
) -> pd.DataFrame:
    ccd = str(group["ccd"].iloc[0])
    length = str(group["length"].iloc[0])
    categories = sorted(group["category"].unique())
    if len(categories) != 1:
        raise ValueError(f"multiple categories in group {(ccd, length)}: {categories}")
    low_quota, total_quota = quota_pair_for_category(
        categories[0],
        small_low_quota=args.small_molecule_low_quota,
        small_total_quota=args.small_molecule_total_quota,
        metal_low_quota=args.metal_or_cluster_low_quota,
        metal_total_quota=args.metal_or_cluster_total_quota,
    )
    group_seed = stable_group_seed(args.random_seed, ccd, length)
    if total_quota <= 0:
        return empty_selection(group, low_quota=low_quota, total_quota=total_quota, seed=group_seed)

    low_pool = group[group["eligible_low_rasa"]].copy()
    low_selected = choose_cluster_rows(
        low_pool,
        quota=low_quota,
        rng=np.random.default_rng(stable_group_seed(args.random_seed, ccd, length, "low")),
        selection_tier="low",
    )

    used_clusters = set(low_selected["foldseek_cluster_id"])
    backfill_needed = max(0, total_quota - len(low_selected))
    band_pool = group[
        group["eligible_backfill_band"] & ~group["foldseek_cluster_id"].isin(used_clusters)
    ].copy()
    band_selected = choose_cluster_rows(
        band_pool,
        quota=backfill_needed,
        rng=np.random.default_rng(stable_group_seed(args.random_seed, ccd, length, "backfill_band")),
        selection_tier="backfill_band",
    )

    used_clusters.update(band_selected["foldseek_cluster_id"])
    remaining_needed = max(0, total_quota - len(low_selected) - len(band_selected))
    leftover_low_pool = group[
        group["eligible_low_rasa"] & ~group["foldseek_cluster_id"].isin(used_clusters)
    ].copy()
    leftover_low_selected = choose_cluster_rows(
        leftover_low_pool,
        quota=remaining_needed,
        rng=np.random.default_rng(stable_group_seed(args.random_seed, ccd, length, "backfill_low_leftover")),
        selection_tier="backfill_low_leftover",
    )

    selected_parts = [low_selected, band_selected, leftover_low_selected]
    nonempty = [part for part in selected_parts if len(part)]
    if not nonempty:
        return empty_selection(group, low_quota=low_quota, total_quota=total_quota, seed=group_seed)

    selected = pd.concat(nonempty, ignore_index=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    selected["selection_quota"] = total_quota
    selected["low_selection_quota"] = low_quota
    selected["total_selection_quota"] = total_quota
    selected["group_random_seed"] = str(group_seed)
    return selected


def select_panel(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected_groups: list[pd.DataFrame] = []
    for _, group in candidates.groupby(["ccd", "length"], sort=True):
        selected_groups.append(select_one_group(group, args=args))
    if not selected_groups:
        return candidates.head(0).copy()
    selected = pd.concat(selected_groups, ignore_index=True)
    selected = selected.sort_values(["ccd", "length", "selection_rank"], kind="mergesort").reset_index(drop=True)
    return selected


def build_summary(candidates: pd.DataFrame, selected: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (ccd, length), group in candidates.groupby(["ccd", "length"], sort=True):
        categories = sorted(group["category"].unique())
        if len(categories) != 1:
            raise ValueError(f"multiple categories in group {(ccd, length)}: {categories}")
        category = categories[0]
        low_quota, total_quota = quota_pair_for_category(
            category,
            small_low_quota=args.small_molecule_low_quota,
            small_total_quota=args.small_molecule_total_quota,
            metal_low_quota=args.metal_or_cluster_low_quota,
            metal_total_quota=args.metal_or_cluster_total_quota,
        )
        if selected.empty:
            selected_group = selected.head(0)
        else:
            selected_group = selected[(selected["ccd"] == ccd) & (selected["length"] == length)]
        ok = group["status"].eq("ok") & np.isfinite(group["rasa_numeric"])
        low_eligible = group[group["eligible_low_rasa"]]
        backfill_band = group[group["eligible_backfill_band"]]
        max_eligible = group[group["eligible_after_rasa"]]
        n_selected = int(len(selected_group))
        n_selected_low = int((selected_group["selection_tier"] == "low").sum()) if n_selected else 0
        n_selected_backfill_band = (
            int((selected_group["selection_tier"] == "backfill_band").sum()) if n_selected else 0
        )
        n_selected_backfill_low_leftover = (
            int((selected_group["selection_tier"] == "backfill_low_leftover").sum()) if n_selected else 0
        )
        rows.append(
            {
                "category": category,
                "ccd": ccd,
                "length": length,
                "low_quota": low_quota,
                "total_quota": total_quota,
                "low_rasa_cutoff": clean_float(args.low_rasa_cutoff),
                "max_rasa_cutoff": clean_float(args.max_rasa_cutoff),
                "n_candidates": int(len(group)),
                "n_rasa_ok": int(ok.sum()),
                "n_low_eligible": int(len(low_eligible)),
                "n_low_unique_clusters": int(low_eligible["foldseek_cluster_id"].nunique()),
                "n_backfill_band_eligible": int(len(backfill_band)),
                "n_backfill_band_unique_clusters": int(backfill_band["foldseek_cluster_id"].nunique()),
                "n_max_eligible": int(len(max_eligible)),
                "n_max_unique_clusters": int(max_eligible["foldseek_cluster_id"].nunique()),
                "n_foldseek_clusters": int(group["foldseek_cluster_id"].nunique()),
                "n_selected_low": n_selected_low,
                "n_selected_backfill_band": n_selected_backfill_band,
                "n_selected_backfill_low_leftover": n_selected_backfill_low_leftover,
                "n_selected": n_selected,
                "selected_unique_clusters": int(selected_group["foldseek_cluster_id"].nunique()) if n_selected else 0,
                "low_shortfall": max(0, low_quota - n_selected_low),
                "shortfall": max(0, total_quota - n_selected),
            }
        )
    return pd.DataFrame(rows)


def add_audit_selection_columns(candidates: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_cols = selected[
        [
            "staged_id",
            "selection_rank",
            "selection_tier",
            "selection_tier_rank",
            "selection_quota",
            "low_selection_quota",
            "total_selection_quota",
            "group_random_seed",
        ]
    ].copy()
    audit = candidates.merge(selected_cols, on="staged_id", how="left", validate="one_to_one")
    audit["is_selected"] = audit["selection_rank"].notna()
    for column in (
        "selection_rank",
        "selection_tier",
        "selection_tier_rank",
        "selection_quota",
        "low_selection_quota",
        "total_selection_quota",
        "group_random_seed",
    ):
        audit[column] = audit[column].fillna("")
    return audit


def stage_selected_files(selected: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    selected = selected.copy()
    cif_root = output_root / "selected_samples" / "cifs"
    json_root = output_root / "selected_samples" / "jsons"
    cif_root.mkdir(parents=True, exist_ok=True)
    json_root.mkdir(parents=True, exist_ok=True)

    cif_targets: list[str] = []
    json_targets: list[str] = []
    for row in selected.itertuples(index=False):
        condition = str(row.condition)
        staged_id = str(row.staged_id)

        cif_source = Path(str(row.source_path))
        cif_dir = cif_root / condition
        cif_dir.mkdir(parents=True, exist_ok=True)
        cif_target = cif_dir / f"{staged_id}.cif.gz"
        if cif_target.exists() or cif_target.is_symlink():
            cif_target.unlink()
        cif_target.symlink_to(cif_source)
        cif_targets.append(str(cif_target))

        json_source_text = str(row.json_path)
        json_target_text = ""
        if json_source_text:
            json_source = Path(json_source_text)
            if json_source.exists():
                json_dir = json_root / condition
                json_dir.mkdir(parents=True, exist_ok=True)
                json_target = json_dir / f"{staged_id}.json"
                if json_target.exists() or json_target.is_symlink():
                    json_target.unlink()
                json_target.symlink_to(json_source)
                json_target_text = str(json_target)
        json_targets.append(json_target_text)

    selected["selected_cif_path"] = cif_targets
    selected["selected_json_path"] = json_targets
    return selected


def selected_output_columns() -> list[str]:
    return [
        "staged_id",
        "condition",
        "ccd",
        "length",
        "category",
        "ligand_class",
        "source_path",
        "json_path",
        "foldseek_cluster_id",
        "selection_rank",
        "selection_tier",
        "selection_tier_rank",
        "selection_quota",
        "low_selection_quota",
        "total_selection_quota",
        "group_random_seed",
        "rasa_value",
        "rasa_numeric",
        "low_rasa_cutoff",
        "max_rasa_cutoff",
        "selected_cif_path",
        "selected_json_path",
    ]


def audit_output_columns() -> list[str]:
    return [
        "staged_id",
        "condition",
        "ccd",
        "length",
        "category",
        "ligand_class",
        "source_path",
        "json_path",
        "foldseek_cluster_id",
        "status",
        "rasa_value",
        "rasa_numeric",
        "low_rasa_cutoff",
        "max_rasa_cutoff",
        "eligible_low_rasa",
        "eligible_backfill_band",
        "eligible_after_rasa",
        "eligibility_reason",
        "is_selected",
        "selection_rank",
        "selection_tier",
        "selection_tier_rank",
        "selection_quota",
        "low_selection_quota",
        "total_selection_quota",
        "group_random_seed",
    ]


def validate_outputs(
    *,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    input_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: object = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add_check("input_sets_match", True, json.dumps(input_diagnostics, sort_keys=True))
    add_check("selected_rows_lte_max_rasa_cutoff", bool((selected["rasa_numeric"] <= args.max_rasa_cutoff).all()))
    low_selected = selected[selected["selection_tier"] == "low"]
    add_check("low_selected_rows_lte_low_cutoff", bool((low_selected["rasa_numeric"] <= args.low_rasa_cutoff).all()))
    band_selected = selected[selected["selection_tier"] == "backfill_band"]
    add_check(
        "backfill_band_rows_between_cutoffs",
        bool(
            (
                (band_selected["rasa_numeric"] > args.low_rasa_cutoff)
                & (band_selected["rasa_numeric"] <= args.max_rasa_cutoff)
            ).all()
        ),
    )
    duplicate_selected_clusters = (
        selected.groupby(["ccd", "length", "foldseek_cluster_id"]).size().reset_index(name="n").query("n > 1")
    )
    add_check(
        "selected_foldseek_clusters_unique_per_group",
        duplicate_selected_clusters.empty,
        duplicate_selected_clusters.head(10).to_json(orient="records"),
    )
    over_total = summary[pd.to_numeric(summary["n_selected"]) > pd.to_numeric(summary["total_quota"])]
    add_check("selected_counts_do_not_exceed_total_quota", over_total.empty, over_total.head(10).to_json(orient="records"))
    over_low = summary[pd.to_numeric(summary["n_selected_low"]) > pd.to_numeric(summary["low_quota"])]
    add_check("low_selected_counts_do_not_exceed_low_quota", over_low.empty, over_low.head(10).to_json(orient="records"))
    add_check(
        "selected_count_matches_summary",
        int(summary["n_selected"].astype(int).sum()) == int(len(selected)),
    )
    add_check(
        "selected_low_count_matches_summary",
        int(summary["n_selected_low"].astype(int).sum()) == int((selected["selection_tier"] == "low").sum()),
    )
    add_check(
        "all_selected_have_source_files",
        bool(selected["source_path"].map(lambda value: Path(str(value)).is_file()).all()),
    )
    if not args.no_stage_files:
        add_check(
            "all_selected_cif_symlinks_exist",
            bool(selected["selected_cif_path"].map(lambda value: Path(str(value)).is_symlink()).all()),
        )

    status = "complete" if all(check["passed"] for check in checks) else "failed"
    return {
        "status": status,
        "checks": checks,
        "counts": {
            "candidate_rows": int(len(candidates)),
            "eligible_low_rasa_rows": int(candidates["eligible_low_rasa"].sum()),
            "eligible_after_rasa_rows": int(candidates["eligible_after_rasa"].sum()),
            "selected_rows": int(len(selected)),
            "selected_low_rows": int((selected["selection_tier"] == "low").sum()),
            "selected_backfill_band_rows": int((selected["selection_tier"] == "backfill_band").sum()),
            "selected_backfill_low_leftover_rows": int(
                (selected["selection_tier"] == "backfill_low_leftover").sum()
            ),
            "groups": int(len(summary)),
            "groups_with_low_shortfall": int((summary["low_shortfall"].astype(int) > 0).sum()),
            "total_low_shortfall": int(summary["low_shortfall"].astype(int).sum()),
            "groups_with_total_shortfall": int((summary["shortfall"].astype(int) > 0).sum()),
            "total_shortfall": int(summary["shortfall"].astype(int).sum()),
        },
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def build_manifest(
    *,
    args: argparse.Namespace,
    started_at: str,
    elapsed_seconds: float,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    summary: pd.DataFrame,
    validation: dict[str, Any],
) -> dict[str, Any]:
    paths = output_paths(args.output_root)
    reason_counts = Counter(candidates["eligibility_reason"])
    category_selected_counts = Counter(selected["category"])
    category_candidate_counts = Counter(candidates["category"])
    tier_selected_counts = Counter(selected["selection_tier"])
    return {
        "status": validation["status"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "command": " ".join(sys.argv),
        "source": {
            "step3_root": str(args.step3_root),
            "input_manifest": str(args.step3_root / "input_manifest.tsv"),
            "foldseek_clusters": str(args.step3_root / "foldseek_clusters.tsv"),
            "rasa_tsv": str(args.rasa_tsv),
        },
        "parameters": {
            "rasa_policy": "absolute_low_then_max_backfill",
            "rasa_group_keys": ["ccd", "length"],
            "low_rasa_cutoff": args.low_rasa_cutoff,
            "max_rasa_cutoff": args.max_rasa_cutoff,
            "small_molecule_categories": sorted(SMALL_MOLECULE_CATEGORIES),
            "metal_or_cluster_categories": sorted(METAL_OR_CLUSTER_CATEGORIES),
            "small_molecule_low_quota": args.small_molecule_low_quota,
            "small_molecule_total_quota": args.small_molecule_total_quota,
            "metal_or_cluster_low_quota": args.metal_or_cluster_low_quota,
            "metal_or_cluster_total_quota": args.metal_or_cluster_total_quota,
            "random_seed": args.random_seed,
            "selection_policy": (
                "For each (ccd, length), choose fixed-seed random unique Foldseek clusters "
                "from RASA <= low cutoff first. Fill remaining total quota with fixed-seed "
                "random unique clusters from low < RASA <= max cutoff. If that band is "
                "insufficient, use remaining RASA <= low cutoff clusters as an explicitly "
                "diagnosed low-leftover fallback."
            ),
            "max_groups": args.max_groups,
            "ccds": args.ccds,
            "lengths": args.lengths,
            "stage_files": not args.no_stage_files,
        },
        "counts": {
            "candidate_rows": int(len(candidates)),
            "eligible_low_rasa_rows": int(candidates["eligible_low_rasa"].sum()),
            "eligible_after_rasa_rows": int(candidates["eligible_after_rasa"].sum()),
            "selected_rows": int(len(selected)),
            "group_count": int(len(summary)),
            "selection_summary_low_shortfall_groups": int((summary["low_shortfall"].astype(int) > 0).sum()),
            "selection_summary_total_low_shortfall": int(summary["low_shortfall"].astype(int).sum()),
            "selection_summary_total_shortfall_groups": int((summary["shortfall"].astype(int) > 0).sum()),
            "selection_summary_total_shortfall": int(summary["shortfall"].astype(int).sum()),
            "candidate_counts_by_category": dict(sorted(category_candidate_counts.items())),
            "selected_counts_by_category": dict(sorted(category_selected_counts.items())),
            "selected_counts_by_tier": dict(sorted(tier_selected_counts.items())),
            "eligibility_reason_counts": dict(sorted(reason_counts.items())),
        },
        "outputs": {
            "selected_samples_tsv": str(paths["selected"]),
            "selection_summary_tsv": str(paths["summary"]),
            "rasa_tier_counts_tsv": str(paths["tiers"]),
            "candidate_audit_tsv": str(paths["audit"]),
            "manifest_json": str(paths["manifest"]),
            "validation_report_json": str(paths["validation"]),
            "selected_files_root": str(paths["selected_files"]),
        },
        "environment": {
            "python": sys.version,
            "platform": sys.platform,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    prepare_output_root(args.output_root, overwrite=args.overwrite)
    started_at = utc_now()
    start = time.time()

    input_manifest = read_tsv(
        args.step3_root / "input_manifest.tsv",
        required_columns={"staged_id", "condition", "ccd", "length", "source_path", "json_path"},
    )
    clusters = read_tsv(
        args.step3_root / "foldseek_clusters.tsv",
        required_columns={"representative_id", "member_id"},
    )
    rasa = read_tsv(
        args.rasa_tsv,
        required_columns={
            "staged_id",
            "condition",
            "ccd",
            "length",
            "category",
            "ligand_class",
            "source_path",
            "json_path",
            "status",
            "rasa_value",
        },
    )

    input_diagnostics = validate_inputs(input_manifest, clusters, rasa)
    candidates = build_candidates(
        input_manifest=input_manifest,
        clusters=clusters,
        rasa=rasa,
        args=args,
    )
    selected = select_panel(candidates, args)
    summary = build_summary(candidates, selected, args)

    if not args.no_stage_files:
        selected = stage_selected_files(selected, args.output_root)
    else:
        selected = selected.copy()
        selected["selected_cif_path"] = ""
        selected["selected_json_path"] = ""

    audit = add_audit_selection_columns(candidates, selected)
    validation = validate_outputs(
        candidates=candidates,
        selected=selected,
        summary=summary,
        args=args,
        input_diagnostics=input_diagnostics,
    )
    elapsed = time.time() - start
    manifest = build_manifest(
        args=args,
        started_at=started_at,
        elapsed_seconds=elapsed,
        candidates=candidates,
        selected=selected,
        summary=summary,
        validation=validation,
    )

    paths = output_paths(args.output_root)
    selected[selected_output_columns()].to_csv(paths["selected"], sep="\t", index=False)
    summary.to_csv(paths["summary"], sep="\t", index=False)
    summary.to_csv(paths["tiers"], sep="\t", index=False)
    audit[audit_output_columns()].to_csv(paths["audit"], sep="\t", index=False)
    write_json(paths["validation"], validation)
    write_json(paths["manifest"], manifest)

    print(
        f"Selected {len(selected):,} samples from {len(candidates):,} candidates "
        f"across {len(summary):,} groups",
        flush=True,
    )
    print(f"Eligibility reason counts: {dict(sorted(Counter(candidates['eligibility_reason']).items()))}", flush=True)
    print(f"Selection tier counts: {dict(sorted(Counter(selected['selection_tier']).items()))}", flush=True)
    print(f"Validation status: {validation['status']}", flush=True)
    print(f"Wrote {paths['selected']}", flush=True)
    print(f"Wrote {paths['manifest']}", flush=True)
    if validation["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
