"""Run Glide docking modes and parse docking metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from allatom_design.eval.physics_based.glide.result_parser import (
    extract_best_scores,
    parse_glide_csv,
)
from allatom_design.eval.physics_based.glide.schrodinger_runner import (
    run_glide,
    write_docking_input,
)


def run_docking_mode(
    grid_file: str,
    ligand_file: str,
    work_dir: str,
    schrodinger_path: str,
    glide_cfg: dict[str, Any],
    mode: str,
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run one Glide docking mode (mininplace or redocking) and return metrics."""
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if mode == "mininplace":
        cfg = glide_cfg.get("inplace", {})
        jobname = "dock_inplace"
        docking_method = cfg.get("docking_method", "mininplace")
        num_poses = 1
    elif mode == "redocking":
        cfg = glide_cfg.get("redocking", {})
        jobname = "dock_redock"
        docking_method = cfg.get("docking_method", "confgen")
        num_poses = cfg.get("num_poses", 1)
    else:
        raise ValueError(f"Unknown docking mode: {mode}")

    dock_input = write_docking_input(
        gridfile=grid_file,
        ligandfile=ligand_file,
        out_dir=work_dir,
        jobname=jobname,
        docking_method=docking_method,
        precision=cfg.get("precision", "SP"),
        num_poses=num_poses,
        write_csv=True,
        pose_outtype="ligandlib_sd",
        compress_poses=False,
        forcefield=cfg.get("forcefield", "OPLS4"),
    )

    outputs = run_glide(
        input_file=dock_input,
        schrodinger_path=schrodinger_path,
        timeout=timeout,
    )

    metrics: dict[str, Any] = {}
    if outputs["csv_path"]:
        df = parse_glide_csv(outputs["csv_path"])
        if not df.empty:
            metrics.update(extract_best_scores(df))
            metrics["num_poses"] = len(df)
        else:
            metrics["error"] = "empty_csv"
    else:
        metrics["error"] = "no_csv_output"

    if outputs.get("sdf_path"):
        metrics["sdf_path"] = outputs["sdf_path"]

    return metrics
