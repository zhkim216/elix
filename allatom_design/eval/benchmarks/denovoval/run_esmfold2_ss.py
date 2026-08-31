"""Run ESMFold2 on a validated denovoval sequence-design manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.denovoval.ligandmpnn.run_af3_ss import (
    cfg_path,
    chunk_label,
    load_reference,
    plan_chunk,
    validate_backmapped_cif_digest,
    validate_design_manifest,
)
from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    sha256_file,
)
from allatom_design.eval.structure_prediction.ESMFold2.evaluation import (
    evaluate_esmfold2_structure_prediction,
)
from allatom_design.eval.structure_prediction.ESMFold2.inputs import (
    write_esmfold2_inputs,
)
from allatom_design.eval.structure_prediction.ESMFold2.runner import (
    ESMFold2RuntimeConfig,
)


DEFAULT_CONFIG = Path(
    "/scratch/users/zhkim216/out_dir/benchmarks/"
    "denovoval_re_ligand_residue_1/configs/"
    "ss1p0_gamma1p0/ligandmpnn.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--array-id", type=int, default=None)
    parser.add_argument("--num-arrays", type=int, default=None)
    parser.add_argument("--designed-sample-id", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--print-array-plan", action="store_true")
    modes.add_argument("--generate-inputs-only", action="store_true")
    modes.add_argument("--run-esmfold2", action="store_true")
    modes.add_argument("--metrics-only", action="store_true")
    return parser.parse_args()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _selected_records(
    *, cfg: DictConfig, args: argparse.Namespace, manifest_path: Path
) -> tuple[list[dict[str, Any]], Any]:
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    required = {
        "designed_sample_id",
        "source_sample_id",
        "model_name",
        "design_index",
        "ccd_code",
        "protein_sequence",
        "protein_chain_id",
        "ligand_chain_id",
        "backmapped_cif_path",
        "backmapped_cif_sha256",
        "status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing required columns: {missing}")
    if not frame["status"].isin(["backmapped", "reused"]).all():
        raise ValueError(f"Design manifest contains failed rows: {manifest_path}")
    if frame["designed_sample_id"].duplicated().any():
        raise ValueError(f"Duplicate designed_sample_id in {manifest_path}")
    validate_design_manifest(frame, cfg, smoke=args.smoke)

    if args.designed_sample_id:
        requested = list(dict.fromkeys(map(str, args.designed_sample_id)))
        observed = set(map(str, frame["designed_sample_id"]))
        missing_ids = sorted(set(requested) - observed)
        if missing_ids:
            raise ValueError(f"Requested designed sample IDs are absent: {missing_ids}")
        order = {sample_id: index for index, sample_id in enumerate(requested)}
        frame = frame[frame["designed_sample_id"].isin(requested)].copy()
        frame["_requested_order"] = frame["designed_sample_id"].map(order)
        frame = frame.sort_values("_requested_order").drop(columns="_requested_order")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        frame = frame.head(args.limit)
    records = frame.to_dict(orient="records")
    plan = plan_chunk(
        len(records), num_arrays=args.num_arrays, array_id=args.array_id
    )
    return records[plan.start : plan.end], plan


def _sample_dict(
    records: list[dict[str, Any]], cfg: DictConfig
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    condition = str(OmegaConf.select(cfg, "dataset.condition", default=""))
    sample_dict: dict[str, dict[str, Any]] = {}
    identity_by_design: dict[str, dict[str, Any]] = {}
    for record in records:
        validate_backmapped_cif_digest(record)
        loaded = load_reference(record, cfg)
        chain_info = loaded["pdb_chain_info"]
        protein_iids = list(map(str, chain_info["protein_pn_unit_iids"]))
        ligand_iids = list(map(str, chain_info["ligand_pn_unit_iids"]))
        input_sample_id = (
            f"{condition}__{record['source_sample_id']}"
            if condition
            else str(record["source_sample_id"])
        )
        entry = sample_dict.setdefault(
            input_sample_id,
            {
                "designed_sample_id": [],
                "designed_sample_atom_array": [],
                "reference_sample_atom_array": [],
                "pdb_chain_info": chain_info,
                "pn_unit_roles": {
                    "binder_pn_unit_iids": protein_iids,
                    "context_pn_unit_iids": ligand_iids,
                    "frame_pn_unit_iids": protein_iids,
                    "template_pn_unit_iids": [],
                },
            },
        )
        entry_chain_info = entry["pdb_chain_info"]
        entry_signature = (
            tuple(map(str, entry_chain_info["protein_pn_unit_iids"])),
            tuple(map(str, entry_chain_info["ligand_pn_unit_iids"])),
            tuple(map(str, entry_chain_info["ligand_ccd_codes"])),
        )
        observed_signature = (
            tuple(map(str, chain_info["protein_pn_unit_iids"])),
            tuple(map(str, chain_info["ligand_pn_unit_iids"])),
            tuple(map(str, chain_info["ligand_ccd_codes"])),
        )
        if entry_signature != observed_signature:
            raise ValueError(
                f"Chain info differs within source {record['source_sample_id']}"
            )
        designed_id = str(record["designed_sample_id"])
        atom_array = loaded["atom_array"]
        entry["designed_sample_id"].append(designed_id)
        entry["designed_sample_atom_array"].append(atom_array)
        entry["reference_sample_atom_array"].append(atom_array)
        identity_by_design[designed_id] = {
            "condition": record.get("condition", condition),
            "source_sample_id": record["source_sample_id"],
            "partial_diffusion_source_index": record.get(
                "partial_diffusion_source_index", ""
            ),
            "model_name": record["model_name"],
            "design_index": record["design_index"],
            "output_slot": record.get("output_slot", ""),
            "ccd_code": record["ccd_code"],
        }
    return sample_dict, identity_by_design


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    mode = "smoke" if args.smoke else "full"
    manifest_path = cfg_path(
        cfg,
        "backmapping.smoke_manifest_csv"
        if args.smoke
        else "backmapping.full_manifest_csv",
    )
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else cfg_path(
            cfg,
            "esmfold2.smoke_output_dir"
            if args.smoke
            else "esmfold2.full_output_dir",
        )
    )
    runtime = ESMFold2RuntimeConfig.from_config(cfg.esmfold2.inference_config)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    selected, plan = _selected_records(
        cfg=cfg, args=args, manifest_path=manifest_path
    )
    if args.print_array_plan:
        print(
            json.dumps(
                {
                    "mode": mode,
                    "condition": str(cfg.dataset.condition),
                    "manifest": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "output_dir": str(output_dir),
                    "total_rows": plan.total_rows,
                    "num_arrays": plan.num_arrays,
                    "array": f"0-{plan.num_arrays - 1}",
                    "array_id": plan.array_id,
                    "start": plan.start,
                    "end": plan.end,
                    "chunk_size": plan.chunk_size,
                    "seeds": list(runtime.seeds),
                    "num_loops": runtime.num_loops,
                    "num_sampling_steps": runtime.num_sampling_steps,
                    "num_diffusion_samples": runtime.num_diffusion_samples,
                    "lm_dropout": runtime.lm_dropout,
                    "expected_predictions_per_design": runtime.expected_prediction_count,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dict, identity_by_design = _sample_dict(selected, cfg)
    label = chunk_label(plan)
    reports_dir = output_dir / "reports" / "chunks"
    if args.generate_inputs_only:
        write_esmfold2_inputs(
            sample_dict=sample_dict,
            input_dir=output_dir / "esmfold2_ss_inputs",
        )
        rows = [
            {
                **identity,
                "designed_sample_id": designed_id,
                "status": "input_ready",
                "n_predictions_expected": runtime.expected_prediction_count,
            }
            for designed_id, identity in identity_by_design.items()
        ]
    else:
        runtime_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        runtime_cfg.esmfold2.require_complete_predictions = False
        statuses = evaluate_esmfold2_structure_prediction(
            sample_dict=sample_dict,
            out_dir=output_dir,
            struct_pred_cfg=runtime_cfg,
            cif_parse_cfg=cfg.cif_parse_cfg.af3_predictions,
            preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
            featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
            pocket_cfg=cfg.pocket_cfg,
            calculate_metrics_only=args.metrics_only,
            csv_suffix=f"_{label}",
            input_sample_is_designed=True,
        )
        rows = []
        for status in statuses:
            designed_id = str(status["designed_sample_id"])
            rows.append(
                {
                    **identity_by_design[designed_id],
                    **status,
                    "status": (
                        "complete"
                        if status["esmfold2_status"] == "complete"
                        else "error"
                    ),
                }
            )

    status_csv = reports_dir / f"status_{label}.csv"
    status_json = reports_dir / f"status_{label}.json"
    _atomic_write_csv(status_csv, rows)
    status_counts = pd.Series(
        [row["status"] for row in rows], dtype=str
    ).value_counts().to_dict()
    payload = {
        "mode": mode,
        "condition": str(cfg.dataset.condition),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "plan": plan.__dict__,
        "runtime": runtime.fingerprint_payload(),
        "status_counts": status_counts,
        "expected_predictions": len(rows) * runtime.expected_prediction_count,
        "status_csv": str(status_csv),
    }
    _atomic_write_json(status_json, payload)
    print(status_json)
    errors = [row for row in rows if row["status"] == "error"]
    if errors:
        raise RuntimeError(
            f"ESMFold2 processing failed for {len(errors)} rows; see {status_json}"
        )


if __name__ == "__main__":
    main()
