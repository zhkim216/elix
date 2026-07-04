"""Hydra entrypoint for PoseBusters evaluation of AF3 predictions.

Usage:
    python -m allatom_design.eval.physics_based.posebusters.run_pb_eval \
        af3_pred_dir=/path/to/af3_ss_preds \
        out_dir=/path/to/pb_output \
        num_workers=4

SLURM array jobs can pass ``num_arrays`` and rely on auto-detected
``SLURM_ARRAY_TASK_ID``.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.physics_based.posebusters.core import (
    discover_af3_cif_paths,
    evaluate_batch,
    split_entries_for_array_job,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../../configs/eval/physics_based",
    config_name="posebusters/run_pb_eval",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    missing = [
        key for key in ("af3_pred_dir", "out_dir")
        if cfg_dict.get(key) in (None, "")
    ]
    if missing:
        raise ValueError(
            "Missing required config value(s): " + ", ".join(missing)
        )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Discovering CIF files in {cfg.af3_pred_dir}")
    entries = discover_af3_cif_paths(cfg.af3_pred_dir, cfg.cif_pattern)
    logger.info(f"Found {len(entries)} CIF files")

    if not entries:
        logger.warning("No CIF files found. Exiting.")
        sys.exit(0)

    entries = split_entries_for_array_job(
        entries,
        array_id=cfg.array_id,
        num_arrays=cfg.num_arrays,
    )
    logger.info(f"Processing {len(entries)} entries in this task")

    work_dir = str(out_dir / "work")
    df = evaluate_batch(
        entries=entries,
        out_dir=work_dir,
        config=cfg_dict.get("config", "dock"),
        cif_parse_cfg=cfg_dict.get("cif_parse_cfg"),
        structure_prep_cfg=cfg_dict.get("structure_prep", {}),
        num_workers=cfg_dict.get("num_workers", 1),
        full_report=cfg_dict.get("full_report", False),
    )

    array_id = cfg.array_id
    if array_id is None:
        env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_id is not None:
            array_id = int(env_id)

    if array_id is not None:
        csv_name = f"pb_metrics_array_{array_id}.csv"
    else:
        csv_name = "pb_metrics.csv"

    csv_path = out_dir / csv_name
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} results to {csv_path}")

    if "pb_valid" in df.columns:
        valid = df["pb_valid"].dropna()
        if len(valid) > 0:
            n_valid = valid.sum()
            logger.info(
                f"pb_valid: {n_valid}/{len(valid)} "
                f"({100 * n_valid / len(valid):.1f}%)"
            )

    errors = df.get("error")
    if errors is not None:
        n_err = errors.notna().sum()
        if n_err > 0:
            logger.warning(f"Errors: {n_err}/{len(df)}")


if __name__ == "__main__":
    main()
