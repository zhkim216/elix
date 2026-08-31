"""Run AF3 single-sequence self-consistency from a saved design bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.run_logging import print_phase, wandb_setup
from allatom_design.eval.structure_prediction.AF3.af3_evaluation import (
    evaluate_af3_self_consistency,
)
from allatom_design.eval.structure_prediction.cached_designs import (
    designed_sample_ids,
    input_ids_for_chunk,
    load_validated_bundle,
    subset_bundle,
    validate_bundle_shape,
)


def _count_designed_sequences(sample_dict: dict[str, dict[str, Any]]) -> int:
    return sum(len(entry["designed_sample_id"]) for entry in sample_dict.values())


@hydra.main(
    config_path="../../../configs/eval/structure_prediction",
    config_name="run_cached_sc_eval_af3",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    source_dir = Path(cfg.cache_cfg.source_dir)
    csv_suffix = str(cfg.cache_cfg.get("csv_suffix", ""))
    chunk_index = int(cfg.cache_cfg.chunk_index)
    num_chunks = int(cfg.cache_cfg.num_chunks)
    max_designed_sequences_raw = cfg.cache_cfg.get("max_designed_sequences", None)
    max_designed_sequences = (
        None
        if max_designed_sequences_raw is None
        else int(max_designed_sequences_raw)
    )

    log_dir = Path(
        wandb_setup(
            base_out_dir=cfg.base_out_dir,
            exp_name=cfg.exp_name,
            cfg_dict=cfg_dict,
            **cfg.wandb,
        )
    )
    with (log_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(cfg_dict, handle)

    print_phase("Phase 1: Loading cached sequence-design outputs")
    sample_dict, marker = load_validated_bundle(
        source_dir=source_dir,
        csv_suffix=csv_suffix,
    )
    validate_bundle_shape(
        sample_dict,
        expected_input_count=int(cfg.cache_cfg.expected_input_count),
        expected_sequences_per_input=int(
            cfg.cache_cfg.expected_sequences_per_input
        ),
    )
    selected_input_ids = input_ids_for_chunk(
        list(sample_dict),
        chunk_index=chunk_index,
        num_chunks=num_chunks,
    )
    chunk_dict = subset_bundle(
        sample_dict,
        selected_input_ids=selected_input_ids,
        max_designed_sequences=max_designed_sequences,
    )
    if not chunk_dict:
        raise RuntimeError(
            f"Chunk {chunk_index}/{num_chunks} selected no input samples"
        )

    run_manifest = {
        "source_dir": str(source_dir),
        "source_bundle": marker["bundle_path"],
        "source_bundle_sha256": marker["bundle_sha256"],
        "source_input_count": len(sample_dict),
        "source_designed_sequence_count": _count_designed_sequences(sample_dict),
        "chunk_index": chunk_index,
        "num_chunks": num_chunks,
        "selected_input_ids": list(chunk_dict),
        "selected_designed_sample_ids": designed_sample_ids(chunk_dict),
        "selected_input_count": len(chunk_dict),
        "selected_designed_sequence_count": _count_designed_sequences(chunk_dict),
    }
    with (log_dir / "cache_selection_manifest.json").open("w") as handle:
        json.dump(run_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"Validated source bundle: inputs={len(sample_dict)}, "
        f"designs={_count_designed_sequences(sample_dict)}"
    )
    print(
        f"Selected chunk {chunk_index}/{num_chunks}: "
        f"inputs={len(chunk_dict)}, designs={_count_designed_sequences(chunk_dict)}"
    )

    print_phase("Phase 2: AF3 single-sequence self-consistency")
    evaluate_af3_self_consistency(
        sample_dict=chunk_dict,
        out_dir=log_dir,
        struct_pred_cfg=cfg.struct_pred_cfg,
        cif_parse_cfg=cfg.cif_cfg.parse.af3_predictions,
        preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
        featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
        pocket_cfg=cfg.pocket_cfg,
        no_wandb=cfg.wandb.no_wandb,
        ckpt_info=marker["ckpt_info"],
        calculate_metrics_only=cfg.struct_pred_cfg.calculate_metrics_only,
        input_sample_is_designed=cfg.input_sample_is_designed,
        free_atom_arrays_progressively=True,
    )
    print_phase(f"AF3 self-consistency complete: {log_dir}")


if __name__ == "__main__":
    main()
