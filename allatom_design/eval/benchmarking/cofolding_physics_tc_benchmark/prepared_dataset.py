from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATASET_ROOT = Path("/home/yjhk/model-dev/datasets/cofolding_physics_tc_benchmark")
DEFAULT_CONDITION = "holo_protein_only_template"
DATASET_PATH_KEYS = {"mmcifPath", "userCCDPath"}


@dataclass(frozen=True)
class ChunkPlan:
    total_rows: int
    array_id: int | None
    num_arrays: int
    start: int
    end: int
    chunk_size: int

    @property
    def selected_rows(self) -> int:
        return max(0, self.end - self.start)

    @property
    def sbatch_array(self) -> str:
        return f"0-{max(self.num_arrays - 1, 0)}"


def prepared_root(dataset_root: Path) -> Path:
    return Path(dataset_root).expanduser().resolve() / "prepared_af3_template_conditioning"


def default_output_dir(dataset_root: Path) -> Path:
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(dataset_root).expanduser().resolve() / "tc_benchmark_runs" / run_id


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field, "")) for field in fieldnames})


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def parse_list(values: list[str] | None, *, default: list[str] | None = None) -> list[str]:
    if not values:
        return list(default or [])
    parsed: list[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                parsed.append(item)
    return parsed


def load_expected_experiments(root: Path) -> dict[str, dict[str, str]]:
    path = root / "manifests" / "expected_experiments.csv"
    if not path.exists():
        return {}
    return {row["experiment_id"]: row for row in read_csv(path)}


def load_run_plan(root: Path) -> list[dict[str, str]]:
    path = root / "manifests" / "run_plan.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing run_plan.csv: {path}")
    return read_csv(path)


def select_run_plan_rows(
    *,
    root: Path,
    conditions: list[str],
    experiment_ids: list[str] | None = None,
    challenge_groups: list[str] | None = None,
    variants: list[str] | None = None,
    ready_only: bool = True,
) -> list[dict[str, str]]:
    run_rows = load_run_plan(root)
    expected_by_id = load_expected_experiments(root)
    condition_set = set(conditions)
    experiment_set = set(experiment_ids or [])
    challenge_set = set(challenge_groups or [])
    variant_set = set(variants or [])

    selected = []
    for row in run_rows:
        expected = expected_by_id.get(row["experiment_id"], {})
        merged = {**expected, **row}
        if condition_set and "all" not in condition_set and row["af3_condition"] not in condition_set:
            continue
        if ready_only and str(row.get("ready_to_run", "")).lower() != "true":
            continue
        if experiment_set and row["experiment_id"] not in experiment_set:
            continue
        if challenge_set and expected.get("challenge_group", "") not in challenge_set:
            continue
        if variant_set and row.get("variant", "") not in variant_set:
            continue
        selected.append(merged)

    return sorted(selected, key=lambda r: (r["af3_condition"], r["experiment_id"]))


def resolve_array_id(array_id: int | None, *, ignore_env: bool = False) -> int | None:
    if array_id is not None:
        return array_id
    if ignore_env:
        return None
    env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(env_id) if env_id is not None else None


def plan_chunk(
    total_rows: int,
    *,
    array_id: int | None = None,
    num_arrays: int | None = None,
    chunk_size: int | None = None,
    ignore_env: bool = False,
) -> ChunkPlan:
    resolved_array_id = resolve_array_id(array_id, ignore_env=ignore_env)
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if num_arrays is not None and num_arrays <= 0:
        raise ValueError("--num-arrays must be positive")

    if chunk_size is not None:
        resolved_num_arrays = max(1, math.ceil(total_rows / chunk_size))
        resolved_chunk_size = chunk_size
    else:
        if num_arrays is None:
            env_count = None if ignore_env else os.environ.get("SLURM_ARRAY_TASK_COUNT")
            num_arrays = int(env_count) if env_count else 1
        resolved_num_arrays = max(1, int(num_arrays))
        resolved_chunk_size = max(1, math.ceil(total_rows / resolved_num_arrays))

    if resolved_array_id is None:
        start = 0
        end = total_rows
    else:
        start = resolved_array_id * resolved_chunk_size
        end = min(start + resolved_chunk_size, total_rows)
        if start > total_rows:
            start = total_rows
            end = total_rows

    return ChunkPlan(
        total_rows=total_rows,
        array_id=resolved_array_id,
        num_arrays=resolved_num_arrays,
        start=start,
        end=end,
        chunk_size=resolved_chunk_size,
    )


def rows_for_chunk(rows: list[dict[str, str]], plan: ChunkPlan) -> list[dict[str, str]]:
    return rows[plan.start : plan.end]


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def metadata_paths(root: Path) -> list[Path]:
    systems_dir = root / "systems"
    if not systems_dir.exists():
        return []
    return sorted(systems_dir.glob("*/metadata.yaml"))


def internal_reference_source_path(root: Path, source_path: Path) -> Path:
    parts = source_path.parts
    if "pdb_mirror" in parts:
        mirror_idx = parts.index("pdb_mirror")
        relative = Path(*parts[mirror_idx:])
    else:
        relative_parts = [part for part in parts if part not in {"", source_path.anchor}]
        relative = Path("external", *relative_parts)
    return root / "reference_sources" / relative


def resolve_dataset_path(dataset_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False)
    return (Path(dataset_root).expanduser().resolve() / path).resolve(strict=False)


def dataset_relative_path(dataset_root: Path, path: str | Path) -> str:
    dataset_root = Path(dataset_root).expanduser().resolve()
    resolved = resolve_dataset_path(dataset_root, path)
    try:
        return resolved.relative_to(dataset_root).as_posix()
    except ValueError:
        return str(path)


def resolve_metadata_paths(metadata: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    out = dict(metadata)
    if out.get("source_structure"):
        out["source_structure"] = str(resolve_dataset_path(dataset_root, out["source_structure"]))
    if out.get("ligand_sources"):
        out["ligand_sources"] = [
            str(resolve_dataset_path(dataset_root, path)) for path in out.get("ligand_sources", [])
        ]
    return out


def absolutize_af3_input_paths(data: Any, dataset_root: Path) -> Any:
    if isinstance(data, list):
        return [absolutize_af3_input_paths(item, dataset_root) for item in data]
    if isinstance(data, dict):
        out = {}
        for key, value in data.items():
            if key in DATASET_PATH_KEYS and isinstance(value, str):
                out[key] = str(resolve_dataset_path(dataset_root, value))
            else:
                out[key] = absolutize_af3_input_paths(value, dataset_root)
        return out
    return data


def relativize_af3_input_paths(data: Any, dataset_root: Path) -> tuple[Any, int]:
    if isinstance(data, list):
        changed = 0
        out = []
        for item in data:
            new_item, item_changed = relativize_af3_input_paths(item, dataset_root)
            out.append(new_item)
            changed += item_changed
        return out, changed
    if isinstance(data, dict):
        changed = 0
        out = {}
        for key, value in data.items():
            if key in DATASET_PATH_KEYS and isinstance(value, str):
                new_value = dataset_relative_path(dataset_root, value)
                out[key] = new_value
                changed += int(new_value != value)
            else:
                new_value, item_changed = relativize_af3_input_paths(value, dataset_root)
                out[key] = new_value
                changed += item_changed
        return out, changed
    return data, 0


def normalize_metadata_paths(
    *,
    dataset_root: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    root = prepared_root(dataset_root)
    reports: list[dict[str, Any]] = []
    for metadata_path in metadata_paths(root):
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        changed = False
        if metadata.get("source_structure"):
            old_value = str(metadata["source_structure"])
            new_value = dataset_relative_path(dataset_root, old_value)
            if new_value != old_value:
                metadata["source_structure"] = new_value
                changed = True
                reports.append(
                    {
                        "path": metadata_path,
                        "field": "source_structure",
                        "old_value": old_value,
                        "new_value": new_value,
                        "status": "dry_run" if dry_run else "updated",
                    }
                )
        if "source_structure_original" in metadata:
            old_value = str(metadata["source_structure_original"])
            metadata.pop("source_structure_original", None)
            changed = True
            reports.append(
                {
                    "path": metadata_path,
                    "field": "source_structure_original",
                    "old_value": old_value,
                    "new_value": "",
                    "status": "dry_run" if dry_run else "removed",
                }
            )
        ligand_sources = metadata.get("ligand_sources")
        if isinstance(ligand_sources, list):
            new_sources = [dataset_relative_path(dataset_root, value) for value in ligand_sources]
            if new_sources != ligand_sources:
                metadata["ligand_sources"] = new_sources
                changed = True
                reports.append(
                    {
                        "path": metadata_path,
                        "field": "ligand_sources",
                        "old_value": ligand_sources,
                        "new_value": new_sources,
                        "status": "dry_run" if dry_run else "updated",
                    }
                )
        if changed and not dry_run:
            metadata_path.write_text(
                yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
    return reports


def normalize_af3_json_paths(
    *,
    dataset_root: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    root = prepared_root(dataset_root)
    reports: list[dict[str, Any]] = []
    for json_path in sorted((root / "af3_inputs").glob("*/*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        new_data, changed_count = relativize_af3_input_paths(data, dataset_root)
        if changed_count == 0:
            continue
        if not dry_run:
            json_path.write_text(json.dumps(new_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        reports.append(
            {
                "path": json_path,
                "field": "af3_path_fields",
                "old_value": "",
                "new_value": "",
                "changed_count": changed_count,
                "status": "dry_run" if dry_run else "updated",
            }
        )
    return reports


def normalize_prepared_dataset_paths(
    *,
    dataset_root: Path,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = prepared_root(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing prepared dataset root: {root}")
    reports = []
    reports.extend(normalize_metadata_paths(dataset_root=dataset_root, dry_run=dry_run))
    reports.extend(normalize_af3_json_paths(dataset_root=dataset_root, dry_run=dry_run))
    summary = {
        "prepared_root": str(root),
        "dry_run": dry_run,
        "report_rows": len(reports),
        "metadata_report_rows": sum(1 for row in reports if str(row.get("path", "")).endswith("metadata.yaml")),
        "af3_json_report_rows": sum(1 for row in reports if str(row.get("path", "")).endswith(".json")),
    }
    if not dry_run and reports:
        write_csv(root / "manifests" / "relative_path_report.csv", reports)
        (root / "manifests" / "relative_path_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return reports, summary


def rewrite_old_root_text(
    *,
    root: Path,
    old_root: Path,
    new_root: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    old_text = str(old_root)
    new_text = str(new_root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if old_text not in text:
            continue
        count = text.count(old_text)
        if not dry_run:
            path.write_text(text.replace(old_text, new_text), encoding="utf-8")
        reports.append(
            {
                "path": path,
                "action": "rewrite_text",
                "old_root_hits": count,
                "status": "dry_run" if dry_run else "updated",
            }
        )
    return reports


def rewrite_old_root_symlinks(
    *,
    root: Path,
    old_root: Path,
    new_root: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    old_root = old_root.resolve(strict=False)
    new_root = new_root.resolve(strict=False)
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        target = Path(os.readlink(path))
        resolved_target = target if target.is_absolute() else (path.parent / target)
        try:
            relative = resolved_target.resolve(strict=False).relative_to(old_root)
        except ValueError:
            continue
        new_target = new_root / relative
        if not dry_run:
            path.unlink()
            path.symlink_to(new_target)
        reports.append(
            {
                "path": path,
                "action": "rewrite_symlink",
                "old_target": str(target),
                "new_target": str(new_target),
                "status": "dry_run" if dry_run else "updated",
            }
        )
    return reports


def count_old_root_hits(root: Path, old_root: Path) -> int:
    old_text = str(old_root)
    hits = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            hits += path.read_text(encoding="utf-8").count(old_text)
        except UnicodeDecodeError:
            continue
    return hits


def count_broken_symlinks(root: Path) -> int:
    broken = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and not path.exists():
            broken += 1
    return broken


def cleanup_prepared_dataset(
    *,
    dataset_root: Path,
    old_root: Path,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    old_root = Path(old_root).expanduser().resolve()
    root = prepared_root(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing prepared dataset root: {root}")

    old_root_hits_before = count_old_root_hits(root, old_root)
    broken_symlinks_before = count_broken_symlinks(root)
    reports = []
    reports.extend(
        rewrite_old_root_text(root=root, old_root=old_root, new_root=dataset_root, dry_run=dry_run)
    )
    reports.extend(
        rewrite_old_root_symlinks(root=root, old_root=old_root, new_root=dataset_root, dry_run=dry_run)
    )
    old_root_hits_after = count_old_root_hits(root, old_root)
    broken_symlinks_after = count_broken_symlinks(root)
    summary = {
        "prepared_root": str(root),
        "old_root": str(old_root),
        "new_root": str(dataset_root),
        "dry_run": dry_run,
        "report_rows": len(reports),
        "old_root_hits_before": old_root_hits_before,
        "old_root_hits_after": old_root_hits_after,
        "broken_symlinks_before": broken_symlinks_before,
        "broken_symlinks_after": broken_symlinks_after,
    }
    return reports, summary


def stage_reference_sources(
    *,
    dataset_root: Path,
    dry_run: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    root = prepared_root(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing prepared dataset root: {root}")

    reports: list[dict[str, Any]] = []
    updated_metadata = 0
    missing_sources = 0
    copied_sources: set[Path] = set()

    for metadata_path in metadata_paths(root):
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        source_value = str(metadata.get("source_structure") or "")
        if not source_value:
            continue
        source_path = resolve_dataset_path(dataset_root, source_value)
        if is_inside(source_path, dataset_root):
            continue

        target_path = internal_reference_source_path(root, source_path)
        report = {
            "metadata_path": metadata_path,
            "system_id": metadata.get("system_id", metadata_path.parent.name),
            "old_source_structure": str(source_path),
            "new_source_structure": str(target_path),
            "action": "stage_reference_source",
            "status": "dry_run" if dry_run else "updated",
        }
        if not source_path.exists():
            report["status"] = "missing_source"
            missing_sources += 1
            reports.append(report)
            continue

        if not dry_run:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path not in copied_sources:
                if not target_path.exists() or target_path.stat().st_size != source_path.stat().st_size:
                    shutil.copy2(source_path, target_path)
                    report["copy_status"] = "copied"
                else:
                    report["copy_status"] = "already_present"
                copied_sources.add(target_path)
            else:
                report["copy_status"] = "reused"

            metadata["source_structure"] = dataset_relative_path(dataset_root, target_path)
            metadata_path.write_text(
                yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        reports.append(report)
        updated_metadata += 1

    summary = {
        "prepared_root": str(root),
        "dry_run": dry_run,
        "metadata_rows_updated": 0 if dry_run else updated_metadata,
        "metadata_rows_would_update": updated_metadata,
        "unique_sources_staged": 0 if dry_run else len(copied_sources),
        "missing_sources": missing_sources,
        "report_rows": len(reports),
    }
    if not dry_run and reports:
        write_csv(root / "manifests" / "reference_source_report.csv", reports)
        (root / "manifests" / "reference_source_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return reports, summary


def load_system_metadata(root: Path, system_id: str) -> dict[str, Any]:
    metadata_path = root / "systems" / system_id / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.yaml for {system_id}: {metadata_path}")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    return resolve_metadata_paths(metadata, root.parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and inspect prepared AF3 TC benchmark data.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stage-reference-sources",
        action="store_true",
        help="Copy external reference source structures into the prepared dataset and rewrite metadata.",
    )
    parser.add_argument(
        "--normalize-relative-paths",
        action="store_true",
        help="Rewrite prepared metadata and AF3 JSON path fields to be dataset-root-relative.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage_reference_sources:
        _, reference_summary = stage_reference_sources(
            dataset_root=args.dataset_root,
            dry_run=args.dry_run,
        )
        for key, value in reference_summary.items():
            print(f"reference_{key}={value}")
    if args.normalize_relative_paths:
        _, relative_summary = normalize_prepared_dataset_paths(
            dataset_root=args.dataset_root,
            dry_run=args.dry_run,
        )
        for key, value in relative_summary.items():
            print(f"relative_{key}={value}")


if __name__ == "__main__":
    main()
