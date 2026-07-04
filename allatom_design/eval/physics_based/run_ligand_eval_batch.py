"""Hydra runner for integrated physics-based ligand evaluation.

Usage:
    python -m allatom_design.eval.physics_based.run_ligand_eval_batch

Retry failed samples:
    python -m allatom_design.eval.physics_based.run_ligand_eval_batch \
        retry_csv=path/to/failures.csv
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.physics_based.composer import EvalSettings, evaluate_batch
from allatom_design.eval.physics_based.glide.schrodinger_runner import find_schrodinger
from allatom_design.eval.physics_based.results import (
    resolve_run_output_dir,
    write_ligand_eval_outputs,
)
from allatom_design.eval.physics_based.sample_selection import select_samples_for_eval

logger = logging.getLogger(__name__)


def _array_info(cfg_dict: dict[str, Any]) -> tuple[int | None, int | None]:
    array_id = cfg_dict.get("array_id")
    num_arrays = cfg_dict.get("num_arrays")

    env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_id is None and env_id not in (None, ""):
        array_id = int(env_id)
    env_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if num_arrays is None and env_count not in (None, ""):
        num_arrays = int(env_count)

    if array_id is not None:
        array_id = int(array_id)
        num_arrays = int(num_arrays or 1)
    return array_id, num_arrays


def split_for_array_job(
    df: pd.DataFrame,
    *,
    array_id: int | None,
    num_arrays: int | None,
) -> pd.DataFrame:
    """Slice selected rows for one SLURM array task."""
    if array_id is None:
        return df
    num_arrays = int(num_arrays or 1)
    chunk_size = math.ceil(len(df) / num_arrays) if len(df) else 0
    start = array_id * chunk_size
    end = min(start + chunk_size, len(df))
    return df.iloc[start:end].reset_index(drop=True)


def _resolve_schrodinger_path(cfg_dict: dict[str, Any]) -> str | None:
    glide_cfg = cfg_dict.get("glide", {}) or {}
    if not glide_cfg.get("enabled", True):
        return None

    raw_path = (cfg_dict.get("schrodinger", {}) or {}).get("schrodinger_path")
    if raw_path in (None, ""):
        return None
    try:
        return find_schrodinger(raw_path)
    except Exception as exc:
        logger.warning("Schrodinger unavailable; Glide phases will fail: %s", exc)
        return None


def _require_run_paths(cfg_dict: dict[str, Any]) -> None:
    missing = [
        key
        for key in ("af3_eval_dir", "output_dir")
        if cfg_dict.get(key) in (None, "")
    ]
    if missing:
        raise ValueError("Missing required config value(s): " + ", ".join(missing))


def _input_csv_paths(
    *,
    af3_eval_dir: Path,
    retry_csv: str | None,
) -> dict[str, str | None]:
    if retry_csv:
        return {"retry_csv": str(retry_csv)}
    return {
        "docking_csv": str(af3_eval_dir / "all_docking_metrics_per_designed_sample.csv"),
        "sc_csv": str(af3_eval_dir / "all_sc_metrics_per_designed_sample.csv"),
    }


@hydra.main(
    config_path="../../configs/eval/physics_based",
    config_name="run_ligand_eval_batch",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    """Run integrated PB + optional Glide ligand evaluation."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    _require_run_paths(cfg_dict)

    af3_eval_dir = Path(cfg_dict["af3_eval_dir"])
    af3_preds_dir = cfg_dict.get("af3_preds_dir") or str(af3_eval_dir / "af3_ss_preds")
    retry_csv = cfg_dict.get("retry_csv")
    array_id, num_arrays = _array_info(cfg_dict)
    root = resolve_run_output_dir(cfg_dict["output_dir"], array_id=array_id)
    selection_cfg = cfg_dict.get("selection", {}) or {}

    selected = select_samples_for_eval(
        af3_eval_dir=af3_eval_dir,
        af3_preds_dir=af3_preds_dir,
        selection_cfg=selection_cfg,
        retry_csv=retry_csv,
    )
    selection_summary = dict(selected.attrs.get("selection_summary", {}))

    if cfg_dict.get("debug", False):
        limit = int(cfg_dict.get("num_debug_samples", 10))
        selected = selected.head(limit).reset_index(drop=True)

    selected = split_for_array_job(
        selected,
        array_id=array_id,
        num_arrays=num_arrays,
    )

    settings = EvalSettings(
        work_dir=str(root / "work"),
        sample_dir=cfg_dict.get("sample_dir"),
        schrodinger_path=_resolve_schrodinger_path(cfg_dict),
        schrodinger_cfg=cfg_dict.get("schrodinger", {}) or {},
        glide_cfg=cfg_dict.get("glide", {}) or {},
        pb_cfg=cfg_dict.get("posebusters", {}) or {},
        cif_parse_cfg=cfg_dict.get("cif_parse_cfg"),
        structure_prep_cfg=cfg_dict.get("structure_prep", {}) or {},
        receptor_pn_unit_iids=cfg_dict.get("receptor_pn_unit_iids"),
        ligand_pn_unit_iids=cfg_dict.get("ligand_pn_unit_iids"),
        ref_sample_is_designed=bool(cfg_dict.get("ref_sample_is_designed", False)),
        num_workers=int(cfg_dict.get("num_workers", 1)),
    )

    if selected.empty:
        results_df = pd.DataFrame()
    else:
        results_df = evaluate_batch(selected, settings)

    manifest = write_ligand_eval_outputs(
        output_dir=cfg_dict["output_dir"],
        cfg_dict=cfg_dict,
        selected_df=selected,
        results_df=results_df,
        config_path="allatom_design/configs/eval/physics_based",
        config_name="run_ligand_eval_batch",
        input_csv_paths=_input_csv_paths(af3_eval_dir=af3_eval_dir, retry_csv=retry_csv),
        selection_summary=selection_summary,
        array_id=array_id,
        array_count=num_arrays,
    )

    logger.info(
        "Wrote ligand eval outputs to %s (%s results, %s failures)",
        manifest["output_dir"],
        manifest["row_counts"]["result_rows"],
        manifest["row_counts"]["failure_rows"],
    )


if __name__ == "__main__":
    main()
