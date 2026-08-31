"""Materialize pinned per-condition configs for the denovoval_re campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.denovoval.sequence_design_inputs import (
    cif_tree_sha256,
    load_canonical_source,
)
from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    sha256_file,
)


DEFAULT_CAMPAIGN_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval_re/campaign.yaml"
)
DEFAULT_OUTPUT_DIR = Path(
    "/scratch/users/zhkim216/out_dir/benchmarks/"
    "denovoval_re_ligand_residue_1/configs"
)
OLD_OUTPUT_ROOT = "/scratch/users/zhkim216/out_dir/benchmarks/denovoval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", type=Path, default=DEFAULT_CAMPAIGN_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition", action="append", default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _replace_output_root(value: Any, *, output_root: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_output_root(item, output_root=output_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_output_root(item, output_root=output_root) for item in value]
    if isinstance(value, str) and value.startswith(OLD_OUTPUT_ROOT):
        return output_root + value[len(OLD_OUTPUT_ROOT) :]
    return value


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _method_smoke_ids(
    *,
    method: str,
    source: Any,
    campaign: DictConfig,
    baseline_smoke_ids: list[str],
) -> list[str]:
    if method == "ADFLIP":
        missing = sorted(set(baseline_smoke_ids) - set(map(str, source["sample_id"])))
        if missing:
            raise ValueError(f"ADFLIP predictor smoke IDs are absent: {missing}")
        return baseline_smoke_ids
    if method == "abacus-t":
        return ["A1L3W_len150_0", "NA_len150_0"]
    return list(map(str, campaign.dataset.canary_sample_ids))


def _resolved_method_config(
    *,
    campaign: DictConfig,
    condition: str,
    method: str,
    baseline_path: Path,
) -> tuple[DictConfig, dict[str, Any]]:
    baseline = OmegaConf.load(baseline_path)
    baseline_smoke_ids = list(map(str, baseline.dataset.smoke_sample_ids))
    output_root = str(Path(str(campaign.output_root)) / condition)
    raw = OmegaConf.to_container(baseline, resolve=True)
    assert isinstance(raw, dict)
    cfg = OmegaConf.create(_replace_output_root(raw, output_root=output_root))

    condition_cfg = campaign.dataset.conditions[condition]
    dataset_values = {
        "condition": condition,
        "sampling_inputs_csv": str(condition_cfg.sampling_inputs_csv),
        "sampling_inputs_sha256": str(condition_cfg.sampling_inputs_sha256),
        "sample_id_list": str(condition_cfg.sample_id_list),
        "sample_id_list_sha256": str(condition_cfg.sample_id_list_sha256),
        "source_cif_dir": str(condition_cfg.source_cif_dir),
        "cif_tree_sha256": str(condition_cfg.cif_tree_sha256),
        "expected_count": int(campaign.dataset.expected_count),
        "expected_unique_ccd_count": int(
            campaign.dataset.expected_unique_ccd_count
        ),
        "metal_ccd_codes": list(map(str, campaign.dataset.metal_ccd_codes)),
        "selection_tsv": str(campaign.dataset.selection_tsv),
        "selection_sha256": str(campaign.dataset.selection_sha256),
        "partial_diffusion_source_index_required": True,
    }
    cfg.dataset = OmegaConf.create(dataset_values)

    source, source_diagnostics = load_canonical_source(cfg)
    if int(source["ccd_code"].nunique()) != int(
        campaign.dataset.expected_unique_ccd_count
    ):
        raise ValueError(
            f"{condition} has {source['ccd_code'].nunique()} CCDs, expected "
            f"{campaign.dataset.expected_unique_ccd_count}"
        )
    cfg.dataset.smoke_sample_ids = _method_smoke_ids(
        method=method,
        source=source,
        campaign=campaign,
        baseline_smoke_ids=baseline_smoke_ids,
    )
    ligand_residue_id = OmegaConf.select(
        condition_cfg, "ligand_residue_id", default=None
    )
    if ligand_residue_id is None:
        cfg.structure.pop("ligand_residue_id", None)
    else:
        cfg.structure.ligand_residue_id = int(ligand_residue_id)
    cfg.sequence_design.output_slots = [0, 1]

    if method == "lasermpnn":
        cfg.chemistry.target_component_set_sha256 = str(
            campaign.chemistry.lasermpnn_target_component_set_sha256
        )
        condition_fallback_sample_ids = OmegaConf.select(
            condition_cfg,
            "lasermpnn_atomworks_fallback_sample_ids",
            default=[],
        )
        cfg.chemistry.atomworks_fallback_sample_ids = sorted(
            set(map(str, cfg.chemistry.atomworks_fallback_sample_ids))
            | set(map(str, condition_fallback_sample_ids))
        )

    af3 = campaign.prediction.af3
    cfg.af3.json_config.model_seeds = list(map(int, af3.model_seeds))
    cfg.af3.inference_config.ss.num_recycles = int(af3.num_recycles)
    cfg.af3.inference_config.ss.num_diffusion_samples = int(
        af3.num_diffusion_samples
    )
    cfg.af3.inference_config.ss.max_templates = int(af3.max_templates)

    method_root = Path(output_root) / method
    esm = OmegaConf.to_container(campaign.prediction.esmfold2, resolve=True)
    assert isinstance(esm, dict)
    cfg.esmfold2 = OmegaConf.create(
        {
            "full_output_dir": str(method_root / "esmfold2_ss" / "full"),
            "smoke_output_dir": str(method_root / "esmfold2_ss" / "smoke"),
            "require_complete_predictions": True,
            "inference_config": esm,
        }
    )
    cfg.pocket_cfg = OmegaConf.create(
        {"role_aware_reference_pocket_distance": 8.0}
    )

    if method == "abacus-t":
        cfg.sampling.full.input_count = int(campaign.dataset.expected_count)
        cfg.sampling.full.inputs_per_chunk = 154
        cfg.sampling.full.chunk_count = 10

    OmegaConf.resolve(cfg)
    selection_path = Path(str(cfg.dataset.selection_tsv))
    if sha256_file(selection_path) != str(cfg.dataset.selection_sha256):
        raise ValueError(f"selection.tsv digest mismatch: {selection_path}")
    cif_paths = sorted(
        Path(str(cfg.dataset.source_cif_dir)).glob("*.cif"), key=lambda path: path.name
    )
    observed_tree = cif_tree_sha256(cif_paths)
    if observed_tree != str(cfg.dataset.cif_tree_sha256):
        raise ValueError(
            f"{condition} CIF tree digest mismatch: {observed_tree} != "
            f"{cfg.dataset.cif_tree_sha256}"
        )

    expected_designs = (
        int(cfg.dataset.expected_count)
        * len(list(cfg.sequence_design.model_names))
        * int(cfg.sequence_design.expected_designs_per_sample)
    )
    metadata = {
        "condition": condition,
        "method_family": method,
        "model_names": list(map(str, cfg.sequence_design.model_names)),
        "source_count": int(cfg.dataset.expected_count),
        "expected_designs": expected_designs,
        "smoke_source_count": len(list(cfg.dataset.smoke_sample_ids)),
        "source_diagnostics": source_diagnostics,
        "baseline_config": str(baseline_path),
        "baseline_config_sha256": sha256_file(baseline_path),
    }
    return cfg, metadata


def materialize(
    *,
    campaign_path: Path,
    output_dir: Path,
    requested_conditions: list[str] | None,
    check_only: bool,
) -> dict[str, Any]:
    campaign = OmegaConf.load(campaign_path)
    OmegaConf.resolve(campaign)
    available_conditions = list(map(str, campaign.dataset.conditions.keys()))
    conditions = requested_conditions or available_conditions
    if len(set(conditions)) != len(conditions):
        raise ValueError(f"Duplicate conditions requested: {conditions}")
    missing = sorted(set(conditions) - set(available_conditions))
    if missing:
        raise ValueError(f"Unknown conditions: {missing}")

    records: list[dict[str, Any]] = []
    for condition in conditions:
        for method, baseline_raw in campaign.baseline_configs.items():
            baseline_path = Path(str(baseline_raw))
            cfg, metadata = _resolved_method_config(
                campaign=campaign,
                condition=condition,
                method=str(method),
                baseline_path=baseline_path,
            )
            text = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False)
            config_path = output_dir / condition / f"{method}.yaml"
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if not check_only:
                _atomic_write(config_path, text)
                if sha256_file(config_path) != digest:
                    raise RuntimeError(f"Config write digest mismatch: {config_path}")
            records.append(
                {
                    **metadata,
                    "config_path": str(config_path),
                    "config_sha256": digest,
                }
            )

    report = {
        "status": "validated" if check_only else "complete",
        "campaign_config": str(campaign_path),
        "campaign_config_sha256": sha256_file(campaign_path),
        "selection_tsv": str(campaign.dataset.selection_tsv),
        "selection_sha256": str(campaign.dataset.selection_sha256),
        "conditions": conditions,
        "records": records,
        "total_expected_designs": sum(record["expected_designs"] for record in records),
    }
    if not check_only:
        _atomic_write(
            output_dir / "config_manifest.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
    return report


def main() -> None:
    args = parse_args()
    report = materialize(
        campaign_path=args.campaign_config,
        output_dir=args.output_dir,
        requested_conditions=args.condition,
        check_only=args.check_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
