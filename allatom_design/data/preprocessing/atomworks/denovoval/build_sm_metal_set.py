#!/usr/bin/env python3
"""Build the denovoval small-molecule/metal validation CIF set."""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.data.preprocessing.atomworks.denovoval.foldseek import (
    assign_foldseek_clusters,
    parse_cluster_tsv,
    run_foldseek_easy_cluster,
    stage_foldseek_inputs,
)
from allatom_design.data.preprocessing.atomworks.denovoval.metal_rasa import (
    process_metal_record,
)
from allatom_design.data.preprocessing.atomworks.denovoval.records import (
    METAL,
    SourceRecord,
    apply_debug_sample_cap,
    discover_source_records,
)
from allatom_design.data.preprocessing.atomworks.denovoval.selection import (
    assign_final_names,
    build_selection_summary,
    select_metal_rows,
    select_small_molecule_rows,
)


logger = logging.getLogger(__name__)


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def optional_int_set(values: Any) -> set[int] | None:
    if values is None:
        return None
    parsed = {int(value) for value in values}
    return parsed or None


def optional_str_set(values: Any) -> set[str] | None:
    if values is None:
        return None
    parsed = {str(value).upper() for value in values}
    return parsed or None


def cfg_list(values: Any) -> list[str]:
    if values is None:
        return []
    return [str(value) for value in values]


def compute_metal_rasa_df(
    metal_records: list[SourceRecord],
    *,
    point_number: int,
    probe_radius: float,
    workers: int,
) -> pd.DataFrame:
    if not metal_records:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    if workers <= 1:
        for index, record in enumerate(metal_records, start=1):
            rows.append(
                process_metal_record(
                    record,
                    point_number=point_number,
                    probe_radius=probe_radius,
                )
            )
            if index % 100 == 0 or index == len(metal_records):
                logger.info("Computed metal RASA for %d/%d CIFs", index, len(metal_records))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_metal_record,
                    record,
                    point_number=point_number,
                    probe_radius=probe_radius,
                ): record
                for record in metal_records
            }
            for index, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if index % 100 == 0 or index == len(futures):
                    logger.info("Computed metal RASA for %d/%d CIFs", index, len(futures))

    return pd.DataFrame(rows).sort_values(["length", "ccd_code", "sample_id"]).reset_index(drop=True)


def enrich_records_with_foldseek(
    records_df: pd.DataFrame,
    foldseek_by_sample: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    foldseek_rows = []
    for sample_id, row in foldseek_by_sample.items():
        out = {"sample_id": sample_id}
        out.update(row)
        foldseek_rows.append(out)
    foldseek_df = pd.DataFrame(foldseek_rows)
    if foldseek_df.empty:
        for column in (
            "foldseek_member_id",
            "foldseek_representative_id",
            "foldseek_cluster",
            "foldseek_cluster_size",
        ):
            records_df[column] = ""
        return records_df
    return records_df.merge(foldseek_df, on="sample_id", how="left")


def enrich_records_with_metal_rasa(
    records_df: pd.DataFrame,
    metal_rasa_df: pd.DataFrame,
) -> pd.DataFrame:
    if metal_rasa_df.empty:
        records_df["metal_rasa_status"] = ""
        records_df["rasa_value"] = pd.NA
        return records_df

    metric_columns = [
        column
        for column in metal_rasa_df.columns
        if column == "sample_id" or column not in records_df.columns
    ]
    out = records_df.merge(metal_rasa_df[metric_columns], on="sample_id", how="left")
    if "metal_rasa_status" not in out.columns:
        out["metal_rasa_status"] = ""
    if "rasa_value" not in out.columns:
        out["rasa_value"] = pd.NA
    return out


def prepare_dataset_outputs(
    *,
    selected_df: pd.DataFrame,
    final_cif_dir: Path,
    dataset_name: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    dataset_dir = final_cif_dir.parent
    key_list_path = dataset_dir / f"{dataset_name}.txt"
    manifest_path = dataset_dir / f"{dataset_name}_manifest.tsv"

    if final_cif_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Final CIF dir exists; set overwrite=true to replace: {final_cif_dir}")
        shutil.rmtree(final_cif_dir)
    final_cif_dir.mkdir(parents=True, exist_ok=True)

    for path in (key_list_path, manifest_path, dataset_dir / f"{dataset_name}_summary.json"):
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"Output file exists; set overwrite=true to replace: {path}")
            path.unlink()

    key_list: list[str] = []
    for row in selected_df.itertuples(index=False):
        target_path = final_cif_dir / row.final_filename
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with gzip.open(row.source_path, "rb") as src, tmp_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp_path.replace(target_path)
        key_list.append(str(row.final_id))

    key_list_path.write_text("\n".join(key_list) + ("\n" if key_list else ""))
    return key_list_path, manifest_path


def run(cfg: DictConfig) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    overwrite = bool(cfg.overwrite)
    work_dir = as_path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    debug_num_samples = cfg.get("debug_num_samples")
    if debug_num_samples is not None:
        debug_num_samples = int(debug_num_samples)
    if debug_num_samples is not None and debug_num_samples <= 0:
        debug_num_samples = None

    lengths = optional_int_set(cfg.filters.lengths)
    ccd_codes = optional_str_set(cfg.filters.ccd_codes)
    all_records = discover_source_records(
        small_molecule_root=as_path(cfg.source_roots.small_molecule),
        metal_root=as_path(cfg.source_roots.metal),
        lengths=lengths,
        ccd_codes=ccd_codes,
    )
    records = apply_debug_sample_cap(all_records, debug_num_samples)
    if not records:
        raise RuntimeError("No source records selected for processing.")

    records_df = pd.DataFrame([record.to_row() for record in records])
    logger.info(
        "Selected %d source records for Foldseek (%s)",
        len(records),
        records_df["kind"].value_counts().to_dict(),
    )
    write_tsv(records_df, work_dir / "source_records.tsv")

    id_lookup = stage_foldseek_inputs(
        records,
        work_dir / "foldseek_input",
        overwrite=overwrite,
    )
    foldseek_result = run_foldseek_easy_cluster(
        foldseek_bin=str(cfg.foldseek.bin),
        input_dir=work_dir / "foldseek_input",
        cluster_prefix=work_dir / "foldseek" / "denovoval_sm_metal_c09",
        tmp_dir=work_dir / "foldseek_tmp",
        threads=int(cfg.foldseek.threads),
        coverage=float(cfg.foldseek.coverage),
        use_gpu=bool(cfg.foldseek.use_gpu),
        overwrite=overwrite,
        extra_args=cfg_list(cfg.foldseek.extra_args),
    )
    representative_by_member, cluster_sizes, cluster_conflicts = parse_cluster_tsv(
        foldseek_result.cluster_tsv
    )
    foldseek_by_sample, unresolved_foldseek_ids = assign_foldseek_clusters(
        records=records,
        representative_by_member=representative_by_member,
        cluster_sizes=cluster_sizes,
        id_lookup=id_lookup,
    )
    enriched_df = enrich_records_with_foldseek(records_df, foldseek_by_sample)
    write_tsv(enriched_df, work_dir / "foldseek_assignments.tsv")
    write_json(
        {
            "command": foldseek_result.command,
            "cluster_tsv": str(foldseek_result.cluster_tsv),
            "elapsed_seconds": foldseek_result.elapsed_seconds,
            "stdout_tail": foldseek_result.stdout_tail,
            "stderr_tail": foldseek_result.stderr_tail,
            "cluster_conflicts": cluster_conflicts,
            "unresolved_foldseek_ids": unresolved_foldseek_ids,
            "n_clusters": len(cluster_sizes),
        },
        work_dir / "foldseek_run.json",
    )

    metal_records = [record for record in records if record.kind == METAL]
    metal_rasa_df = compute_metal_rasa_df(
        metal_records,
        point_number=int(cfg.metal_rasa.point_number),
        probe_radius=float(cfg.metal_rasa.probe_radius),
        workers=int(cfg.metal_rasa.workers),
    )
    if not metal_rasa_df.empty:
        write_tsv(metal_rasa_df, work_dir / "metal_rasa.tsv")
    enriched_df = enrich_records_with_metal_rasa(enriched_df, metal_rasa_df)
    write_tsv(enriched_df, work_dir / "records_with_foldseek_and_rasa.tsv")

    allow_underfilled = bool(cfg.selection.allow_underfilled) or debug_num_samples is not None
    sm_selected, sm_warnings = select_small_molecule_rows(
        enriched_df,
        target_per_length_ccd=int(cfg.selection.small_molecule.target_per_length_ccd),
        allow_underfilled=allow_underfilled,
    )
    metal_selected, metal_warnings = select_metal_rows(
        enriched_df,
        target_per_length_ccd=int(cfg.selection.metal.target_per_length_ccd),
        max_rasa=float(cfg.selection.metal.max_rasa),
        allow_underfilled=allow_underfilled,
    )
    selected_df = assign_final_names(pd.concat([sm_selected, metal_selected], ignore_index=True))
    write_tsv(selected_df, work_dir / "selected_records.tsv")

    final_cif_dir = as_path(cfg.final_cif_dir)
    key_list_path, manifest_path = prepare_dataset_outputs(
        selected_df=selected_df,
        final_cif_dir=final_cif_dir,
        dataset_name=str(cfg.dataset_name),
        overwrite=overwrite,
    )
    manifest_df = selected_df.copy()
    manifest_df["new_cif"] = manifest_df["final_filename"].map(lambda name: f"cifs/{name}")
    write_tsv(manifest_df, manifest_path)

    summary = build_selection_summary(
        all_records_df=enriched_df,
        selected_df=selected_df,
        warning_messages=sm_warnings + metal_warnings,
    )
    summary.update(
        {
            "dataset_name": str(cfg.dataset_name),
            "debug_num_samples": debug_num_samples,
            "work_dir": str(work_dir),
            "final_cif_dir": str(final_cif_dir),
            "key_list_path": str(key_list_path),
            "manifest_path": str(manifest_path),
            "foldseek": {
                "coverage": float(cfg.foldseek.coverage),
                "use_gpu": bool(cfg.foldseek.use_gpu),
                "command": foldseek_result.command,
                "cluster_tsv": str(foldseek_result.cluster_tsv),
                "n_clusters": len(cluster_sizes),
            },
            "metal_rasa": {
                "max_rasa": float(cfg.selection.metal.max_rasa),
                "point_number": int(cfg.metal_rasa.point_number),
                "workers": int(cfg.metal_rasa.workers),
            },
            "config": OmegaConf.to_container(cfg, resolve=True),
            "pid": os.getpid(),
        }
    )
    write_json(summary, final_cif_dir.parent / f"{cfg.dataset_name}_summary.json")
    logger.info("Wrote %d final CIFs to %s", len(selected_df), final_cif_dir)
    return summary


@hydra.main(
    config_path="../../../../configs_local/preprocessing/atomworks/denovoval",
    config_name="build_sm_metal_debug",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
