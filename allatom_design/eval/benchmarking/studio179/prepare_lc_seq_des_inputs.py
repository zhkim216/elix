from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from allatom_design.eval.benchmarking.studio179.studio179_ligand_map import (
    build_studio179_manifest,
    filter_studio179_manifest,
    normalize_target_signatures,
)


def resolve_studio179_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.studio179_root is not None:
        studio179_root = Path(args.studio179_root).expanduser()
    else:
        studio179_root = (
            Path(args.disco_root).expanduser()
            / "disco_inference_benchmarks_release_data/studio-179"
        )

    converted_root = (
        Path(args.converted_root).expanduser()
        if args.converted_root is not None
        else studio179_root
    )
    cif_dir = (
        Path(args.cif_dir).expanduser()
        if args.cif_dir is not None
        else converted_root / "cifs"
    )
    legacy_nested_cif_dir = studio179_root / "cifs" / "cifs"
    if cif_dir.exists() and not any(cif_dir.glob("*.cif")) and legacy_nested_cif_dir.exists():
        cif_dir = legacy_nested_cif_dir

    return {
        "studio179_root": studio179_root,
        "converted_root": converted_root,
        "cif_dir": cif_dir,
        "sample_id_list": (
            Path(args.sample_id_list).expanduser()
            if args.sample_id_list is not None
            else converted_root / "studio179_all.txt"
        ),
        "metadata_csv": (
            Path(args.metadata_csv).expanduser()
            if args.metadata_csv is not None
            else studio179_root / "all_diversity_results.csv"
        ),
        "userccd_manifest_tsv": (
            Path(args.userccd_manifest_tsv).expanduser()
            if args.userccd_manifest_tsv is not None
            else converted_root / "conformer_cifs/studio179_userccd_manifest.tsv"
        ),
        "userccd_path": (
            Path(args.userccd_path).expanduser()
            if args.userccd_path is not None
            else converted_root / "conformer_cifs/studio179_all_components_userccd.cif"
        ),
    }


def _parse_target_filter(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    targets: list[str] = []
    for raw_part in value.split(","):
        target = raw_part.strip()
        if target and target not in targets:
            targets.append(target)
    return targets


def _parse_int_filter(value: str | None, *, option_name: str) -> list[int]:
    if value is None or not value.strip():
        return []

    values: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        try:
            parsed = int(part)
        except ValueError as exc:
            raise ValueError(f"{option_name} must be a comma-separated integer list") from exc
        if parsed not in values:
            values.append(parsed)
    return values


def _sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "all"


def make_run_id(
    *,
    targets: Sequence[str],
    lengths: Sequence[int],
    sample_indices: Sequence[int],
    smoke_n: int | None,
    limit: int | None,
) -> str:
    parts: list[str] = ["target_" + "_".join(targets) if targets else "all"]
    if lengths:
        parts.append("len" + "_".join(str(length) for length in lengths))
    if sample_indices:
        parts.append("idx" + "_".join(str(sample_index) for sample_index in sample_indices))
    if limit is not None:
        parts.append(f"n{limit}")
    if smoke_n is not None:
        parts.append(f"smoke{smoke_n}")
    return _sanitize_label("_".join(parts))


def make_artifact_stem(
    *,
    run_id: str,
    array_id: int | None,
    num_arrays: int | None,
) -> str:
    stem = f"studio179_{run_id}"
    if array_id is not None:
        if num_arrays is None:
            raise ValueError("array_id requires num_arrays")
        stem = f"{stem}_array_{array_id}_of_{num_arrays}"
    return stem


def _select_rows(
    manifest_df: pd.DataFrame,
    *,
    targets: Sequence[str],
    lengths: Sequence[int],
    sample_indices: Sequence[int],
    limit: int | None,
    smoke_n: int | None,
) -> pd.DataFrame:
    selected_df = filter_studio179_manifest(
        manifest_df,
        targets=targets,
        lengths=lengths or None,
        sample_indices=sample_indices or None,
    )
    selected_df = selected_df.sort_values(["disco_target", "length", "sample_index"]).reset_index(drop=True)
    if limit is not None:
        selected_df = selected_df.head(limit).copy()
    if smoke_n is not None:
        selected_df = selected_df.head(smoke_n).copy()
    if selected_df.empty:
        requested = ",".join(targets) if targets else "all"
        requested_lengths = ",".join(str(length) for length in lengths) if lengths else "all"
        requested_indices = ",".join(str(index) for index in sample_indices) if sample_indices else "all"
        raise ValueError(
            "No Studio-179 samples selected for "
            f"target={requested}, length={requested_lengths}, sample_index={requested_indices}"
        )
    return selected_df


def _load_config_defaults(config_path: str | None, valid_dests: set[str]) -> tuple[dict[str, object], list[str]]:
    if config_path is None:
        return {}, []

    path = Path(config_path).expanduser()
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"--config must point to a YAML mapping: {path}")

    aliases = {
        "user_ccd_manifest": "userccd_manifest_tsv",
        "userccd_manifest": "userccd_manifest_tsv",
        "user_ccd_path": "userccd_path",
    }
    defaults: dict[str, object] = {}
    ignored_keys: list[str] = []
    for key, value in config.items():
        normalized_key = str(key).replace("-", "_")
        dest = aliases.get(normalized_key, normalized_key)
        if dest in valid_dests:
            defaults[dest] = value
        else:
            ignored_keys.append(str(key))
    return defaults, sorted(ignored_keys)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Studio-179 name lists and manifests for lc_seq_des_multi."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--disco-root",
        default=str(Path.home() / "model-dev/datasets/val_cifs/DISCO_benchmark_data"),
    )
    parser.add_argument("--studio179-root", default=None)
    parser.add_argument("--converted-root", default=None)
    parser.add_argument("--cif-dir", default=None)
    parser.add_argument("--sample-id-list", default=None)
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--user-ccd-manifest", "--userccd-manifest-tsv", dest="userccd_manifest_tsv", default=None)
    parser.add_argument("--user-ccd-path", "--userccd-path", dest="userccd_path", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--length", default=None)
    parser.add_argument("--sample-index", default=None)
    parser.add_argument("--smoke-n", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--array-id", type=int, default=None)
    parser.add_argument("--num-arrays", type=int, default=None)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = _build_parser()
    valid_dests = {action.dest for action in parser._actions}
    config_defaults, ignored_keys = _load_config_defaults(pre_args.config, valid_dests)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    args.ignored_config_keys = ignored_keys
    return args


def _validate_args(args: argparse.Namespace) -> None:
    lengths = _parse_int_filter(args.length, option_name="--length")
    sample_indices = _parse_int_filter(args.sample_index, option_name="--sample-index")

    if any(length <= 0 for length in lengths):
        raise ValueError("--length values must be positive")
    if any(sample_index < 0 for sample_index in sample_indices):
        raise ValueError("--sample-index values must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.smoke_n is not None and args.smoke_n <= 0:
        raise ValueError("--smoke-n must be positive")
    if args.num_arrays is not None and args.num_arrays <= 0:
        raise ValueError("--num-arrays must be positive")
    if args.array_id is not None and args.num_arrays is None:
        raise ValueError("--array-id requires --num-arrays")
    if args.array_id is not None and args.array_id < 0:
        raise ValueError("--array-id must be non-negative")
    if args.array_id is not None and args.array_id >= args.num_arrays:
        raise ValueError("--array-id must be smaller than --num-arrays")


def prepare_lc_seq_des_inputs(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    paths = resolve_studio179_paths(args)
    targets = _parse_target_filter(args.target)
    lengths = _parse_int_filter(args.length, option_name="--length")
    sample_indices = _parse_int_filter(args.sample_index, option_name="--sample-index")

    manifest_df = build_studio179_manifest(
        sample_id_list=paths["sample_id_list"],
        metadata_csv=paths["metadata_csv"],
        cif_dir=paths["cif_dir"],
        require_cifs=True,
        userccd_manifest_tsv=paths["userccd_manifest_tsv"],
        userccd_path=paths["userccd_path"],
    )
    target_signatures = normalize_target_signatures(manifest_df, targets)
    selected_df = _select_rows(
        manifest_df,
        targets=targets,
        lengths=lengths,
        sample_indices=sample_indices,
        limit=args.limit,
        smoke_n=args.smoke_n,
    )

    run_id = args.run_id or make_run_id(
        targets=targets,
        lengths=lengths,
        sample_indices=sample_indices,
        smoke_n=args.smoke_n,
        limit=args.limit,
    )
    out_dir = (
        Path(args.out_dir).expanduser()
        if args.out_dir
        else paths["converted_root"] / "benchmark_runs"
    )
    artifact_stem = make_artifact_stem(
        run_id=run_id,
        array_id=args.array_id,
        num_arrays=args.num_arrays,
    )
    list_path = out_dir / f"{artifact_stem}.txt"
    manifest_path = out_dir / f"{artifact_stem}_manifest.tsv"
    summary_path = out_dir / f"{artifact_stem}_summary.json"

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "run_id": run_id,
        "config": args.config,
        "ignored_config_keys": args.ignored_config_keys,
        "artifact_stem": artifact_stem,
        "array_id": args.array_id,
        "num_arrays": args.num_arrays,
        "target_filter": targets,
        "target_signatures": target_signatures,
        "missing_target_signatures": sorted(
            set(target_signatures) - set(selected_df["target_signature"].astype(str))
        ),
        "length_filter": lengths,
        "sample_index_filter": sample_indices,
        "source_sample_count": int(len(manifest_df)),
        "selected_sample_count": int(len(selected_df)),
        "selected_unique_targets": int(selected_df["disco_target"].nunique()),
        "selected_af3_component_ids": sorted(
            {
                component_id
                for value in selected_df["af3_ligand_ccd_codes"].astype(str)
                for component_id in value.split(";")
                if component_id
            }
        ),
        "paths": {
            "studio179_root": str(paths["studio179_root"]),
            "converted_root": str(paths["converted_root"]),
            "cif_dir": str(paths["cif_dir"]),
            "source_sample_id_list": str(paths["sample_id_list"]),
            "metadata_csv": str(paths["metadata_csv"]),
            "userccd_manifest_tsv": str(paths["userccd_manifest_tsv"]),
            "userccd_path": str(paths["userccd_path"]),
            "sample_id_list": str(list_path),
            "manifest_tsv": str(manifest_path),
            "summary_json": str(summary_path),
        },
    }

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        list_path.write_text("\n".join(selected_df["sample_id"].astype(str).tolist()) + "\n")
        selected_df.to_csv(manifest_path, sep="\t", index=False)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = prepare_lc_seq_des_inputs(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
