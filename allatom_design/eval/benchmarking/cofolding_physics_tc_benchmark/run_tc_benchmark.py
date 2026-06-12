from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from allatom_design.eval.benchmarking.cofolding_physics_tc_benchmark.prepared_dataset import (
    DEFAULT_CONDITION,
    DEFAULT_DATASET_ROOT,
    absolutize_af3_input_paths,
    default_output_dir,
    load_system_metadata,
    parse_list,
    plan_chunk,
    prepared_root,
    rows_for_chunk,
    select_run_plan_rows,
    write_csv,
)
from allatom_design.eval.utils.data_utils import prepare_af3_prediction
from allatom_design.eval.utils.folding_utils import (
    find_pred_sample_path_af3,
    run_af3_single_sequence,
    run_af3_template_conditioned,
)


DEFAULT_RUNNER_PATH = Path(
    "/home/yjhk/model-dev/allatom-design/alphafold3/src/alphafold3/run_alphafold.py"
)
DEFAULT_MODEL_DIR = Path("/home/yjhk/model-dev/model_params/af3")
DEFAULT_DB_DIR = Path("/home/yjhk/model-dev/af3_databases")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the cofolding physics AF3 TC benchmark.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--condition", action="append", default=None)
    parser.add_argument("--experiment-id", action="append", default=None)
    parser.add_argument("--challenge-group", action="append", default=None)
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--include-not-ready", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--array-id", type=int, default=None)
    parser.add_argument("--num-arrays", type=int, default=None)
    parser.add_argument("--print-array-plan", action="store_true")
    parser.add_argument("--generate-inputs-only", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--run-af3", action="store_true")
    parser.add_argument("--predictions-root", type=Path, default=None)
    parser.add_argument("--runner-path", type=Path, default=DEFAULT_RUNNER_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--num-recycles", type=int, default=1)
    parser.add_argument("--num-diffusion-samples", type=int, default=1)
    parser.add_argument("--template-pair-scale", type=float, default=1.0)
    parser.add_argument("--flash-attention-implementation", default="xla")
    parser.add_argument("--use-subprocess", action="store_true")
    parser.add_argument("--pocket-distance-for-docking-metrics", type=float, default=8.0)
    return parser.parse_args()


def validate_mode(args: argparse.Namespace) -> None:
    modes = [args.generate_inputs_only, args.metrics_only, args.run_af3, args.print_array_plan]
    if not any(modes):
        raise ValueError(
            "Specify one of --print-array-plan, --generate-inputs-only, --metrics-only, or --run-af3"
        )
    if args.generate_inputs_only and (args.metrics_only or args.run_af3):
        raise ValueError("--generate-inputs-only cannot be combined with --metrics-only or --run-af3")
    if args.metrics_only and args.run_af3:
        raise ValueError("--metrics-only cannot be combined with --run-af3")


def af3_inference_config(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "base": {
                "model_dir": str(args.model_dir),
                "db_dir": str(args.db_dir),
                "flash_attention_implementation": args.flash_attention_implementation,
            },
            "ss": {
                "num_recycles": args.num_recycles,
                "num_diffusion_samples": args.num_diffusion_samples,
                "max_templates": 0,
                "ligand_protein_template_conditioning_mode": 0,
                "template_pair_scale": args.template_pair_scale,
            },
            "tc": {
                "num_recycles": args.num_recycles,
                "num_diffusion_samples": args.num_diffusion_samples,
                "max_templates": 1,
                "ligand_protein_template_conditioning_mode": 1,
                "mask_template_sidechains": True,
                "mask_template_sequence": True,
                "template_pair_scale": args.template_pair_scale,
                "max_template_date": "2026-06-10",
            },
        }
    )


def copy_af3_input(source_json: Path, staged_json: Path, dataset_root: Path) -> dict[str, Any]:
    data = json.loads(source_json.read_text(encoding="utf-8"))
    data = absolutize_af3_input_paths(data, dataset_root)
    staged_json.parent.mkdir(parents=True, exist_ok=True)
    staged_json.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def validate_af3_json(json_path: Path) -> str:
    try:
        from alphafold3.common import folding_input

        list(folding_input.load_fold_inputs_from_path(json_path))
    except Exception as exc:
        return f"invalid:{type(exc).__name__}:{exc}"
    return "valid"


def prediction_root_for_condition(args: argparse.Namespace, output_dir: Path, condition: str) -> Path:
    if args.predictions_root is None:
        return output_dir / "af3_predictions" / condition
    root = args.predictions_root.expanduser().resolve()
    return root / condition if (root / condition).exists() else root


def find_prediction_cifs(predictions_dir: Path, experiment_id: str) -> list[Path]:
    try:
        _, paths = find_pred_sample_path_af3(out_dir=str(predictions_dir), job_name=experiment_id)
        if paths:
            return [Path(path) for path in paths]
    except Exception:
        pass
    job_dir = predictions_dir / experiment_id
    if not job_dir.exists():
        return []
    model_cifs = sorted(path for path in job_dir.rglob("*.cif") if path.stem.endswith("model"))
    return model_cifs or sorted(job_dir.rglob("*.cif"))


def af3_mode_for_condition(condition: str) -> str:
    return "ss" if condition == "no_template_control" else "tc"


def run_af3(json_path: Path, predictions_dir: Path, condition: str, args: argparse.Namespace) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    runner = (
        run_af3_single_sequence
        if af3_mode_for_condition(condition) == "ss"
        else run_af3_template_conditioned
    )
    runner(
        str(json_path),
        str(predictions_dir),
        runner_path=str(args.runner_path),
        inference_config=af3_inference_config(args),
        use_subprocess=args.use_subprocess,
    )


def scalar_prediction_status(prediction_cifs: list[Path]) -> str:
    return "found" if prediction_cifs else "missing"


def process_row(
    *,
    row: dict[str, str],
    args: argparse.Namespace,
    root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    experiment_id = row["experiment_id"]
    condition = row["af3_condition"]
    source_json = root / "af3_inputs" / condition / f"{experiment_id}.json"
    staged_json = output_dir / "staged_af3_inputs" / condition / f"{experiment_id}.json"
    predictions_dir = prediction_root_for_condition(args, output_dir, condition)

    status: dict[str, Any] = {
        "experiment_id": experiment_id,
        "system_id": row["system_id"],
        "pdb_id": row.get("pdb_id", ""),
        "variant": row.get("variant", ""),
        "challenge_group": row.get("challenge_group", ""),
        "af3_condition": condition,
        "source_json": str(source_json),
        "staged_json": str(staged_json),
        "predictions_dir": str(predictions_dir),
    }
    docking_metric_rows: list[dict[str, Any]] = []
    sc_metric_rows: list[dict[str, Any]] = []

    if not source_json.exists():
        status["status"] = "missing_af3_input_json"
        return status, docking_metric_rows, sc_metric_rows

    copy_af3_input(source_json, staged_json, root.parent)
    schema_status = validate_af3_json(staged_json)
    status["schema_status"] = schema_status
    if not schema_status.startswith("valid"):
        status["status"] = "invalid_af3_input_json"
        return status, docking_metric_rows, sc_metric_rows

    if args.generate_inputs_only:
        status["status"] = "input_prepared"
        return status, docking_metric_rows, sc_metric_rows

    if args.run_af3:
        run_af3(staged_json, predictions_dir, condition, args)

    prediction_cifs = find_prediction_cifs(predictions_dir, experiment_id)
    status["prediction_status"] = scalar_prediction_status(prediction_cifs)
    status["prediction_cif_count"] = len(prediction_cifs)
    status["prediction_cifs"] = ";".join(str(path) for path in prediction_cifs)
    if not prediction_cifs:
        status["status"] = "no_prediction_cif"
        return status, docking_metric_rows, sc_metric_rows

    from allatom_design.eval.benchmarking.cofolding_physics_tc_benchmark.metric_references import (
        metric_rows_for_prediction,
        self_consistency_row_for_prediction,
    )

    metadata = load_system_metadata(root, row["system_id"])
    for diffusion_idx, pred_cif in enumerate(prediction_cifs):
        row_context = {
            "experiment_id": experiment_id,
            "system_id": row["system_id"],
            "pdb_id": row.get("pdb_id", ""),
            "variant": row.get("variant", ""),
            "challenge_group": row.get("challenge_group", ""),
            "af3_condition": condition,
            "diffusion_idx": diffusion_idx,
        }
        try:
            # Keep the existing AF3 parser path exercised before metric code runs.
            prepare_af3_prediction(
                pdb_path=pred_cif,
                preprocess_cfg=OmegaConf.create(
                    {
                        "undesired_res_names": [],
                        "b_factor_min": None,
                        "b_factor_max": None,
                        "min_residues_for_polymers": 0,
                        "remove_terminal_oxygen_protein": False,
                        "remove_terminal_oxygen_nucleic_acid": False,
                    }
                ),
                featurizer_cfg=OmegaConf.create(
                    {
                        "max_tokens": None,
                        "max_atoms": None,
                        "remove_keys": [],
                        "remove_unresolved_tokens": True,
                    }
                ),
            )
            per_target_rows = metric_rows_for_prediction(
                pred_cif=pred_cif,
                metadata=metadata,
                pocket_distance_for_docking_metrics=args.pocket_distance_for_docking_metrics,
            )
            sc_row = self_consistency_row_for_prediction(
                pred_cif=pred_cif,
                metadata=metadata,
            )
        except Exception as exc:
            per_target_rows = [
                {
                    "metric_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "pred_cif": str(pred_cif),
                }
            ]
            sc_row = {
                "metric_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "pred_cif": str(pred_cif),
            }
        sc_row.update(row_context)
        sc_metric_rows.append(sc_row)
        for metric_row in per_target_rows:
            metric_row.update(row_context)
            docking_metric_rows.append(metric_row)
    status["docking_metric_rows"] = len(docking_metric_rows)
    status["sc_metric_rows"] = len(sc_metric_rows)
    status["metric_rows"] = len(docking_metric_rows) + len(sc_metric_rows)
    status["docking_metric_error_rows"] = sum(
        1 for metric_row in docking_metric_rows if metric_row.get("metric_status") != "ok"
    )
    status["sc_metric_error_rows"] = sum(
        1 for metric_row in sc_metric_rows if metric_row.get("metric_status") != "ok"
    )
    status["metric_error_rows"] = status["docking_metric_error_rows"] + status["sc_metric_error_rows"]
    status["status"] = "ok" if status["metric_error_rows"] == 0 else "metric_errors"
    return status, docking_metric_rows, sc_metric_rows


def write_summary(
    output_dir: Path,
    statuses: list[dict[str, Any]],
    docking_metric_rows: list[dict[str, Any]],
    sc_metric_rows: list[dict[str, Any]],
) -> None:
    summary = {
        "run_status_rows": len(statuses),
        "docking_metric_rows": len(docking_metric_rows),
        "sc_metric_rows": len(sc_metric_rows),
        "metric_rows": len(docking_metric_rows) + len(sc_metric_rows),
        "ok_status_rows": sum(1 for row in statuses if row.get("status") == "ok"),
        "input_prepared_status_rows": sum(
            1 for row in statuses if row.get("status") == "input_prepared"
        ),
        "error_status_rows": sum(1 for row in statuses if row.get("status") not in {"ok", "input_prepared"}),
        "docking_metric_error_rows": sum(
            1 for row in docking_metric_rows if row.get("metric_status") != "ok"
        ),
        "sc_metric_error_rows": sum(1 for row in sc_metric_rows if row.get("metric_status") != "ok"),
        "metric_error_rows": sum(
            1 for row in [*docking_metric_rows, *sc_metric_rows] if row.get("metric_status") != "ok"
        ),
    }
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    validate_mode(args)
    dataset_root = args.dataset_root.expanduser().resolve()
    root = prepared_root(dataset_root)

    conditions = parse_list(args.condition, default=[DEFAULT_CONDITION])
    rows = select_run_plan_rows(
        root=root,
        conditions=conditions,
        experiment_ids=parse_list(args.experiment_id),
        challenge_groups=parse_list(args.challenge_group),
        variants=parse_list(args.variant),
        ready_only=not args.include_not_ready,
    )
    if args.limit is not None:
        rows = rows[: args.limit]
    chunk_plan = plan_chunk(
        len(rows),
        array_id=args.array_id,
        num_arrays=args.num_arrays,
        chunk_size=args.chunk_size,
    )
    if args.print_array_plan:
        print(f"selected_rows={chunk_plan.total_rows}")
        print(f"chunk_size={chunk_plan.chunk_size}")
        print(f"num_arrays={chunk_plan.num_arrays}")
        print(f"sbatch_array={chunk_plan.sbatch_array}")
        return

    chunk_rows = rows_for_chunk(rows, chunk_plan)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else default_output_dir(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_dir / "manifests"
    metrics_dir = output_dir / "metrics"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(manifests_dir / "selected_run_plan.csv", chunk_rows)
    (manifests_dir / "chunk_plan.json").write_text(
        json.dumps(chunk_plan.__dict__, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    statuses: list[dict[str, Any]] = []
    docking_metric_rows: list[dict[str, Any]] = []
    sc_metric_rows: list[dict[str, Any]] = []
    for row in chunk_rows:
        status, row_docking_metrics, row_sc_metrics = process_row(
            row=row,
            args=args,
            root=root,
            output_dir=output_dir,
        )
        statuses.append(status)
        docking_metric_rows.extend(row_docking_metrics)
        sc_metric_rows.extend(row_sc_metrics)
        print(
            f"{status['experiment_id']}: status={status.get('status')} "
            f"predictions={status.get('prediction_cif_count', 0)} "
            f"docking_metric_rows={status.get('docking_metric_rows', 0)} "
            f"sc_metric_rows={status.get('sc_metric_rows', 0)}"
        )

    write_csv(manifests_dir / "run_status.csv", statuses)
    write_csv(metrics_dir / "per_prediction_docking_metrics.csv", docking_metric_rows)
    write_csv(metrics_dir / "per_prediction_self_consistency_metrics.csv", sc_metric_rows)
    write_summary(output_dir, statuses, docking_metric_rows, sc_metric_rows)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
