from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    protein_sequence,
    read_structure,
    sha256_file,
    unique_ligand_ccd,
)
from allatom_design.eval.metrics.docking import compute_docking_metrics_atomarray
from allatom_design.eval.metrics.self_consistency import (
    compute_self_consistency_metrics_atomarray,
)
from allatom_design.eval.structure_prediction.AF3.af3_input_utils import load_af3_eval_sample
from allatom_design.eval.structure_prediction.AF3.af3_json import (
    make_af3_protein_sequence_entry,
)
from allatom_design.eval.structure_prediction.AF3.af3_runner import (
    expected_prediction_count_from_json,
    run_af3_single_sequence,
    summarize_af3_prediction_outputs,
)
from allatom_design.eval.structure_prediction.AF3.inputs import prepare_af3_prediction


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/ligandmpnn/default.yaml"
)
PREDICTION_DIR_PATTERN = re.compile(
    r"^seed-(?P<seed>[0-9]+)_sample-(?P<sample>[0-9]+)$"
)


@dataclass(frozen=True)
class ChunkPlan:
    total_rows: int
    num_arrays: int
    array_id: int | None
    chunk_size: int
    start: int
    end: int


def parse_args(default_config: Path = DEFAULT_CONFIG) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AF3 single-sequence self-consistency for denovoval MPNN designs."
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--array-id", type=int, default=None)
    parser.add_argument("--num-arrays", type=int, default=None)
    parser.add_argument("--designed-sample-id", action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-seed", type=int, action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-recycles", type=int, default=None)
    parser.add_argument("--num-diffusion-samples", type=int, default=None)
    parser.add_argument("--use-subprocess", action="store_true")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--print-array-plan", action="store_true")
    modes.add_argument("--generate-inputs-only", action="store_true")
    modes.add_argument("--run-af3", action="store_true")
    modes.add_argument("--metrics-only", action="store_true")
    modes.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def cfg_path(cfg: DictConfig, key: str) -> Path:
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ValueError(f"Missing required config path: {key}")
    return Path(str(value)).expanduser()


def expected_design_rows_from_config(cfg: DictConfig) -> int:
    """Return the pre-manifest design count from the method-neutral contract."""
    model_names = OmegaConf.select(cfg, "sequence_design.model_names")
    designs_per_sample = OmegaConf.select(
        cfg, "sequence_design.expected_designs_per_sample"
    )
    if model_names is None or designs_per_sample is None:
        raise ValueError(
            "Config must define sequence_design.model_names and "
            "sequence_design.expected_designs_per_sample"
        )
    return (
        int(cfg.dataset.expected_count)
        * len(list(model_names))
        * int(designs_per_sample)
    )


def expected_manifest_rows(cfg: DictConfig, *, smoke: bool) -> int:
    source_count = (
        len(list(cfg.dataset.smoke_sample_ids))
        if smoke
        else int(cfg.dataset.expected_count)
    )
    return (
        source_count
        * len(list(cfg.sequence_design.model_names))
        * int(cfg.sequence_design.expected_designs_per_sample)
    )


def configured_design_indices(cfg: DictConfig) -> list[int]:
    """Return the method-owned design indices, preserving the legacy default."""
    expected_designs = int(cfg.sequence_design.expected_designs_per_sample)
    configured = OmegaConf.select(cfg, "sequence_design.design_indices")
    indices = (
        list(range(1, expected_designs + 1))
        if configured is None
        else list(map(int, configured))
    )
    if len(indices) != expected_designs or len(set(indices)) != len(indices):
        raise ValueError(
            "sequence_design.design_indices must contain exactly "
            f"{expected_designs} unique integers, got {indices}"
        )
    return indices


def validate_design_manifest(frame: pd.DataFrame, cfg: DictConfig, *, smoke: bool) -> None:
    expected_rows = expected_manifest_rows(cfg, smoke=smoke)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Design manifest has {len(frame)} rows, expected {expected_rows}"
        )
    expected_models = list(map(str, cfg.sequence_design.model_names))
    expected_indices = configured_design_indices(cfg)
    observed_models = sorted(set(map(str, frame["model_name"])))
    if observed_models != sorted(expected_models):
        raise ValueError(
            f"Design manifest models {observed_models} != {sorted(expected_models)}"
        )
    if smoke:
        observed_sources = set(map(str, frame["source_sample_id"]))
        expected_sources = set(map(str, cfg.dataset.smoke_sample_ids))
        if observed_sources != expected_sources:
            raise ValueError(
                f"Smoke source IDs {sorted(observed_sources)} != {sorted(expected_sources)}"
            )
    expected_pairs = {
        (model_name, design_index)
        for model_name in expected_models
        for design_index in expected_indices
    }
    condition = str(OmegaConf.select(cfg, "dataset.condition", default=""))
    if condition:
        required_identity = {
            "condition",
            "partial_diffusion_source_index",
            "output_slot",
        }
        missing_identity = sorted(required_identity - set(frame.columns))
        if missing_identity:
            raise ValueError(
                f"Design manifest is missing denovoval_re identity columns: "
                f"{missing_identity}"
            )
        observed_conditions = set(map(str, frame["condition"]))
        if observed_conditions != {condition}:
            raise ValueError(
                f"Design manifest conditions {sorted(observed_conditions)} != "
                f"{[condition]}"
            )
        output_slots = list(
            map(
                int,
                OmegaConf.select(
                    cfg,
                    "sequence_design.output_slots",
                    default=list(range(len(expected_indices))),
                ),
            )
        )
        if output_slots != list(range(len(expected_indices))):
            raise ValueError(
                "sequence_design.output_slots must be contiguous 0..N-1, got "
                f"{output_slots}"
            )
        slot_by_index = dict(zip(expected_indices, output_slots, strict=True))
        invalid_slots = frame.loc[
            frame.apply(
                lambda row: int(row["output_slot"])
                != slot_by_index[int(row["design_index"])],
                axis=1,
            )
        ]
        if not invalid_slots.empty:
            raise ValueError(
                "Design manifest raw design_index/output_slot mapping differs; "
                f"examples={invalid_slots['designed_sample_id'].head(20).tolist()}"
            )
        source_indices = frame.groupby("source_sample_id", sort=False)[
            "partial_diffusion_source_index"
        ].nunique()
        if not source_indices.eq(1).all():
            raise ValueError(
                "A source maps to multiple partial_diffusion_source_index values"
            )
        unique_source_indices = frame.drop_duplicates("source_sample_id")[
            "partial_diffusion_source_index"
        ]
        if unique_source_indices.duplicated().any():
            raise ValueError(
                "Multiple sources share a partial_diffusion_source_index"
            )
    invalid_sources: list[str] = []
    for source_id, group in frame.groupby("source_sample_id", sort=False):
        observed_pairs = set(
            zip(map(str, group["model_name"]), map(int, group["design_index"]))
        )
        if observed_pairs != expected_pairs or len(group) != len(expected_pairs):
            invalid_sources.append(str(source_id))
    expected_source_count = (
        len(list(cfg.dataset.smoke_sample_ids))
        if smoke
        else int(cfg.dataset.expected_count)
    )
    if frame["source_sample_id"].nunique() != expected_source_count or invalid_sources:
        raise ValueError(
            "Design manifest source/model/design matrix is incomplete: "
            f"source_count={frame['source_sample_id'].nunique()} expected="
            f"{expected_source_count}, invalid={invalid_sources[:20]}"
        )
def resolve_array_id(value: int | None) -> int | None:
    if value is not None:
        return value
    env_value = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(env_value) if env_value is not None else None


def plan_chunk(total_rows: int, *, num_arrays: int | None, array_id: int | None) -> ChunkPlan:
    resolved_array_id = resolve_array_id(array_id)
    if num_arrays is None:
        env_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        num_arrays = int(env_count) if env_count else 1
    if num_arrays < 1:
        raise ValueError("num_arrays must be positive")
    if resolved_array_id is not None and not 0 <= resolved_array_id < num_arrays:
        raise ValueError(
            f"array_id {resolved_array_id} is outside 0..{num_arrays - 1}"
        )
    chunk_size = max(1, math.ceil(total_rows / num_arrays))
    if resolved_array_id is None:
        start, end = 0, total_rows
    else:
        start = min(total_rows, resolved_array_id * chunk_size)
        end = min(total_rows, start + chunk_size)
    return ChunkPlan(total_rows, num_arrays, resolved_array_id, chunk_size, start, end)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(tmp_path, path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def validate_backmapped_cif_digest(record: dict[str, Any]) -> str:
    path = Path(str(record["backmapped_cif_path"]))
    observed = sha256_file(path)
    expected = str(record["backmapped_cif_sha256"])
    if observed != expected:
        raise ValueError(
            f"Backmapped CIF digest mismatch for {record['designed_sample_id']}: "
            f"{observed} != {expected}"
        )
    return observed


def prediction_model_cif_cardinality_errors(prediction_dir: Path) -> list[str]:
    if not prediction_dir.exists():
        return []
    return [
        str(path)
        for path in sorted(prediction_dir.iterdir())
        if path.is_dir()
        and len(
            [
                cif_path
                for cif_path in path.glob("*.cif")
                if cif_path.stem.endswith("model")
            ]
        )
        != 1
    ]


def prediction_seed_sample(path: Path) -> tuple[int, int]:
    match = PREDICTION_DIR_PATTERN.fullmatch(path.parent.name)
    if match is None:
        raise ValueError(
            "AF3 prediction CIF parent must be seed-<seed>_sample-<index>, got "
            f"{path.parent.name!r}: {path}"
        )
    return int(match.group("seed")), int(match.group("sample"))


def af3_json_for_record(
    record: dict[str, Any],
    *,
    protein_chain_id: str,
    ligand_chain_id: str,
    json_config: DictConfig,
) -> dict[str, Any]:
    structure = read_structure(record["backmapped_cif_path"])
    sequence = protein_sequence(structure, protein_chain_id)
    ligand_ccd = unique_ligand_ccd(structure, ligand_chain_id)
    if sequence != record["protein_sequence"]:
        raise ValueError(
            f"Manifest/CIF sequence mismatch for {record['designed_sample_id']}"
        )
    if ligand_ccd != record["ccd_code"]:
        raise ValueError(
            f"Manifest/CIF ligand mismatch for {record['designed_sample_id']}: "
            f"{record['ccd_code']} != {ligand_ccd}"
        )
    config = OmegaConf.to_container(json_config, resolve=True)
    payload: dict[str, Any] = {
        "name": record["designed_sample_id"],
        "sequences": [
            make_af3_protein_sequence_entry(
                chain_id=protein_chain_id,
                sequence=sequence,
                modifications=[],
                templates=[],
            ),
            {"ligand": {"id": ligand_chain_id, "ccdCodes": [ligand_ccd]}},
        ],
        "modelSeeds": list(config.get("model_seeds", [42])),
        "dialect": "alphafold3",
        "version": int(config.get("version", 2)),
    }
    user_ccd_path = config.get("user_ccd_path") or config.get("userCCDPath")
    if user_ccd_path:
        payload["userCCDPath"] = str(user_ccd_path)
        payload["version"] = max(3, payload["version"])
    return payload


def validate_af3_json(path: Path) -> None:
    from alphafold3.common import folding_input

    list(folding_input.load_fold_inputs_from_path(path))


def resolved_inference_config(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    inference = OmegaConf.create(
        OmegaConf.to_container(cfg.af3.inference_config, resolve=True)
    )
    if args.num_recycles is not None:
        inference.ss.num_recycles = args.num_recycles
    if args.num_diffusion_samples is not None:
        inference.ss.num_diffusion_samples = args.num_diffusion_samples
    return inference


def load_reference(record: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    loaded = load_af3_eval_sample(
        sample_path=str(record["backmapped_cif_path"]),
        cif_parse_cfg=cfg.cif_parse_cfg.designed_samples,
        preprocess_cfg=cfg.preprocess_cfg.designed_samples,
        featurizer_cfg=cfg.featurizer_cfg.prepare_designed_samples,
        sample_is_designed=True,
    )
    chain_info = loaded["pdb_chain_info"]
    expected_protein = [f"{record['protein_chain_id']}_1"]
    expected_ligand = [f"{record['ligand_chain_id']}_1"]
    observed_protein = list(map(str, chain_info["protein_pn_unit_iids"]))
    observed_ligand = list(map(str, chain_info["ligand_pn_unit_iids"]))
    observed_ccd = list(map(str, chain_info["ligand_ccd_codes"]))
    if observed_protein != expected_protein:
        raise ValueError(f"Protein entity mismatch: {observed_protein} != {expected_protein}")
    if observed_ligand != expected_ligand:
        raise ValueError(f"Ligand entity mismatch: {observed_ligand} != {expected_ligand}")
    if observed_ccd != [record["ccd_code"]]:
        raise ValueError(f"Ligand CCD mismatch: {observed_ccd} != {[record['ccd_code']]}")
    return loaded


def metric_rows(
    record: dict[str, Any],
    *,
    prediction_cifs: list[Path],
    cfg: DictConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = load_reference(record, cfg)
    reference_atom_array = reference["atom_array"]
    protein_iids = list(map(str, reference["pdb_chain_info"]["protein_pn_unit_iids"]))
    ligand_iids = list(map(str, reference["pdb_chain_info"]["ligand_pn_unit_iids"]))
    ligand_ccds = list(map(str, reference["pdb_chain_info"]["ligand_ccd_codes"]))
    sc_rows: list[dict[str, Any]] = []
    docking_rows: list[dict[str, Any]] = []
    for diffusion_index, prediction_cif in enumerate(prediction_cifs):
        try:
            model_seed, diffusion_sample_index = prediction_seed_sample(prediction_cif)
        except ValueError:
            if record.get("condition"):
                raise
            # Historical manifests predate explicit AF3 seed/sample identity.
            model_seed = ""
            diffusion_sample_index = diffusion_index
        base = {
            "condition": record.get("condition", ""),
            "designed_sample_id": record["designed_sample_id"],
            "source_sample_id": record["source_sample_id"],
            "partial_diffusion_source_index": record.get(
                "partial_diffusion_source_index", ""
            ),
            "model_name": record["model_name"],
            "design_index": record["design_index"],
            "output_slot": record.get("output_slot", ""),
            "ccd_code": record["ccd_code"],
            "model_seed": model_seed,
            "diffusion_sample_index": diffusion_sample_index,
            "diffusion_index": diffusion_index,
            "prediction_cif": str(prediction_cif),
        }
        try:
            prediction = prepare_af3_prediction(
                pdb_path=str(prediction_cif),
                cif_parse_cfg=cfg.cif_parse_cfg.af3_predictions,
                preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
                featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
            )
            pred_atom_array = prediction["atom_array"]
            metrics = compute_self_consistency_metrics_atomarray(
                pred_atom_array=pred_atom_array,
                sample_atom_array=reference_atom_array,
                pred_sample_path=prediction_cif,
                save_aligned=bool(cfg.metrics.save_aligned),
                compute_tmalign=bool(
                    OmegaConf.select(cfg, "metrics.compute_tmalign", default=False)
                ),
                save_tmaligned=bool(
                    OmegaConf.select(cfg, "metrics.save_tmaligned", default=False)
                ),
            )
            sc_rows.append({**base, **metrics, "metric_status": "ok", "error": ""})
        except Exception as exc:
            sc_rows.append(
                {
                    **base,
                    "metric_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if not bool(cfg.metrics.docking.enabled):
            continue
        try:
            docking = compute_docking_metrics_atomarray(
                pred_atom_array=pred_atom_array,
                sample_atom_array=reference_atom_array,
                reference_atom_array=reference_atom_array,
                pred_sample_path=prediction_cif,
                pocket_distance_for_docking_metrics=float(
                    cfg.metrics.docking.pocket_distance_for_docking_metrics
                ),
                receptor_pn_unit_iids=protein_iids,
                ligand_pn_unit_iids=ligand_iids,
                ligand_ccd_codes=ligand_ccds,
                ligand_smiles=None,
                reference_ligand_pn_unit_iids=ligand_iids,
                ref_sample_is_designed=True,
                reference_pocket_annotation_method=str(
                    cfg.metrics.docking.reference_pocket_annotation_method
                ),
                save_aligned=bool(cfg.metrics.docking.save_aligned),
            )
            docking_error = str(docking.get("error") or "")
            docking_rows.append(
                {
                    **base,
                    **docking,
                    "metric_status": "error" if docking_error else "ok",
                    "error": docking_error,
                }
            )
        except Exception as exc:
            docking_rows.append(
                {
                    **base,
                    "metric_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return sc_rows, docking_rows


def chunk_label(plan: ChunkPlan) -> str:
    return "all" if plan.array_id is None else f"{plan.array_id:04d}"


def main(default_config: Path = DEFAULT_CONFIG) -> None:
    args = parse_args(default_config)
    cfg = OmegaConf.load(args.config)
    mode = "smoke" if args.smoke else "full"
    manifest_path = cfg_path(
        cfg,
        "backmapping.smoke_manifest_csv" if args.smoke else "backmapping.full_manifest_csv",
    )
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else cfg_path(
            cfg,
            "af3.smoke_output_dir" if args.smoke else "af3.full_output_dir",
        )
    )
    json_config = OmegaConf.create(
        OmegaConf.to_container(cfg.af3.json_config, resolve=True)
    )
    if args.model_seed is not None:
        model_seeds = list(map(int, args.model_seed))
        if any(seed < 0 for seed in model_seeds) or len(set(model_seeds)) != len(
            model_seeds
        ):
            raise ValueError(
                f"--model-seed values must be unique nonnegative integers: {model_seeds}"
            )
        json_config.model_seeds = model_seeds
    if not manifest_path.exists():
        if not args.print_array_plan:
            raise FileNotFoundError(manifest_path)
        expected_total = expected_manifest_rows(cfg, smoke=args.smoke)
        if args.limit is not None:
            expected_total = min(expected_total, args.limit)
        plan = plan_chunk(
            expected_total,
            num_arrays=args.num_arrays,
            array_id=args.array_id,
        )
        print(
            json.dumps(
                {
                    "mode": mode,
                    "manifest": str(manifest_path),
                    "manifest_exists": False,
                    "manifest_sha256": "",
                    "output_dir": str(output_dir),
                    "model_seeds": list(map(int, json_config.model_seeds)),
                    "total_rows": plan.total_rows,
                    "total_rows_source": "expected_contract",
                    "num_arrays": plan.num_arrays,
                    "array": f"0-{plan.num_arrays - 1}",
                    "chunk_size": plan.chunk_size,
                    "array_id": plan.array_id,
                    "start": plan.start,
                    "end": plan.end,
                    "num_recycles": args.num_recycles
                    if args.num_recycles is not None
                    else int(cfg.af3.inference_config.ss.num_recycles),
                    "num_diffusion_samples": args.num_diffusion_samples
                    if args.num_diffusion_samples is not None
                    else int(cfg.af3.inference_config.ss.num_diffusion_samples),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
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
        requested_ids = list(dict.fromkeys(map(str, args.designed_sample_id)))
        observed_ids = set(map(str, frame["designed_sample_id"]))
        missing_ids = sorted(set(requested_ids) - observed_ids)
        if missing_ids:
            raise ValueError(
                f"Requested designed sample IDs are absent from {manifest_path}: "
                f"{missing_ids}"
            )
        order = {sample_id: index for index, sample_id in enumerate(requested_ids)}
        frame = frame[frame["designed_sample_id"].isin(requested_ids)].copy()
        frame["_requested_order"] = frame["designed_sample_id"].map(order)
        frame = frame.sort_values("_requested_order").drop(columns="_requested_order")
    if args.limit is not None:
        frame = frame.head(args.limit)
    records = frame.to_dict(orient="records")
    plan = plan_chunk(
        len(records),
        num_arrays=args.num_arrays,
        array_id=args.array_id,
    )
    if args.print_array_plan:
        print(
            json.dumps(
                {
                    "mode": mode,
                    "manifest": str(manifest_path),
                    "manifest_exists": True,
                    "manifest_sha256": sha256_file(manifest_path),
                    "output_dir": str(output_dir),
                    "model_seeds": list(map(int, json_config.model_seeds)),
                    "total_rows": plan.total_rows,
                    "total_rows_source": "observed_manifest",
                    "num_arrays": plan.num_arrays,
                    "array": f"0-{plan.num_arrays - 1}",
                    "chunk_size": plan.chunk_size,
                    "array_id": plan.array_id,
                    "start": plan.start,
                    "end": plan.end,
                    "num_recycles": args.num_recycles
                    if args.num_recycles is not None
                    else int(cfg.af3.inference_config.ss.num_recycles),
                    "num_diffusion_samples": args.num_diffusion_samples
                    if args.num_diffusion_samples is not None
                    else int(cfg.af3.inference_config.ss.num_diffusion_samples),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    selected = records[plan.start : plan.end]
    input_dir = output_dir / "inputs"
    predictions_dir = output_dir / "predictions"
    reports_dir = output_dir / "reports" / "chunks"
    metrics_dir = output_dir / "metrics" / "chunks"
    for directory in (input_dir, predictions_dir, reports_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    inference_config = resolved_inference_config(cfg, args)
    runner_path = cfg_path(cfg, "af3.runner_path")
    use_subprocess = bool(cfg.af3.use_subprocess) or args.use_subprocess

    status_rows: list[dict[str, Any]] = []
    all_sc_rows: list[dict[str, Any]] = []
    all_docking_rows: list[dict[str, Any]] = []
    for record in selected:
        job_name = str(record["designed_sample_id"])
        json_path = input_dir / f"{job_name}.json"
        status: dict[str, Any] = {
            "mode": mode,
            "condition": record.get("condition", ""),
            "designed_sample_id": job_name,
            "source_sample_id": record["source_sample_id"],
            "partial_diffusion_source_index": record.get(
                "partial_diffusion_source_index", ""
            ),
            "model_name": record["model_name"],
            "design_index": record["design_index"],
            "output_slot": record.get("output_slot", ""),
            "ccd_code": record["ccd_code"],
            "json_path": str(json_path),
            "status": "error",
            "error": "",
        }
        try:
            observed_cif_sha256 = validate_backmapped_cif_digest(record)
            payload = af3_json_for_record(
                record,
                protein_chain_id=str(record["protein_chain_id"]),
                ligand_chain_id=str(record["ligand_chain_id"]),
                json_config=json_config,
            )
            atomic_write_json(json_path, payload)
            if bool(cfg.af3.validate_json):
                validate_af3_json(json_path)
            status.update(
                {
                    "json_status": "valid",
                    "json_sha256": sha256_file(json_path),
                    "protein_sequence": record["protein_sequence"],
                    "protein_chain_id": record["protein_chain_id"],
                    "ligand_chain_id": record["ligand_chain_id"],
                    "ligand_ccd_code": record["ccd_code"],
                    "backmapped_cif_sha256": observed_cif_sha256,
                }
            )
            expected_predictions = expected_prediction_count_from_json(
                json_path, inference_config, "ss"
            )
            status["n_predictions_expected"] = expected_predictions
            if args.generate_inputs_only:
                status["status"] = "input_ready"
                status_rows.append(status)
                continue
            if args.run_af3:
                run_af3_single_sequence(
                    json_path=str(json_path),
                    out_dir=str(predictions_dir),
                    runner_path=str(runner_path),
                    inference_config=inference_config,
                    use_subprocess=use_subprocess,
                )
            summary = summarize_af3_prediction_outputs(
                out_dir=predictions_dir,
                job_name=job_name,
                expected_count=expected_predictions,
                json_path=json_path,
                inference_config=inference_config,
                mode="ss",
                strict_input_fingerprint=bool(
                    inference_config.ss.strict_input_fingerprint
                ),
            )
            prediction_dir = Path(summary["prediction_dir"])
            bad_model_cif_dirs = prediction_model_cif_cardinality_errors(
                prediction_dir
            )
            if bad_model_cif_dirs:
                raise RuntimeError(
                    "Prediction sample directories must contain exactly one model CIF: "
                    f"{bad_model_cif_dirs[:20]}"
                )
            status.update(
                {
                    "n_predictions_found": summary["n_found"],
                    "n_malformed": summary["n_malformed"],
                    "n_surplus": summary["n_surplus"],
                    "input_fingerprint_ok": summary["input_fingerprint_ok"],
                    "input_fingerprint_error": summary["input_fingerprint_error"],
                }
            )
            if args.status_only:
                status["status"] = "complete" if summary["complete"] else "incomplete"
                status_rows.append(status)
                continue
            if not summary["complete"]:
                raise RuntimeError(
                    f"Prediction outputs incomplete: {summary['n_found']}/"
                    f"{expected_predictions}, malformed={summary['n_malformed']}"
                )
            sc_rows, docking_rows = metric_rows(
                record,
                prediction_cifs=list(summary["model_cif_paths"]),
                cfg=cfg,
            )
            all_sc_rows.extend(sc_rows)
            all_docking_rows.extend(docking_rows)
            metric_errors = sum(row["metric_status"] != "ok" for row in sc_rows)
            metric_errors += sum(
                row["metric_status"] != "ok" for row in docking_rows
            )
            status.update(
                {
                    "sc_metric_rows": len(sc_rows),
                    "docking_metric_rows": len(docking_rows),
                    "metric_error_rows": metric_errors,
                }
            )
            if metric_errors:
                raise RuntimeError(f"Metric computation produced {metric_errors} errors")
            status["status"] = "complete"
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
        status_rows.append(status)

    label = chunk_label(plan)
    status_csv = reports_dir / f"status_{label}.csv"
    status_json = reports_dir / f"status_{label}.json"
    atomic_write_csv(status_csv, status_rows)
    atomic_write_json(
        status_json,
        {
            "mode": mode,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "plan": plan.__dict__,
            "status_counts": pd.Series(
                [row["status"] for row in status_rows], dtype=str
            ).value_counts().to_dict(),
            "diagnostics": {
                "errors": sum(row["status"] == "error" for row in status_rows),
                "incomplete": sum(row["status"] == "incomplete" for row in status_rows),
                "missing": sum(
                    "FileNotFoundError" in str(row.get("error", ""))
                    for row in status_rows
                ),
                "stale": sum(
                    bool(row.get("input_fingerprint_error")) for row in status_rows
                ),
                "skipped": 0,
                "unmapped": 0,
                "metric_errors": sum(
                    int(row.get("metric_error_rows", 0) or 0) for row in status_rows
                ),
            },
            "status_csv": str(status_csv),
        },
    )
    if all_sc_rows:
        atomic_write_csv(metrics_dir / f"self_consistency_{label}.csv", all_sc_rows)
    if all_docking_rows:
        atomic_write_csv(metrics_dir / f"docking_{label}.csv", all_docking_rows)
    errors = [row for row in status_rows if row["status"] == "error"]
    print(status_json)
    if errors and not args.status_only:
        raise RuntimeError(f"AF3 processing failed for {len(errors)} rows; see {status_json}")


if __name__ == "__main__":
    main()
