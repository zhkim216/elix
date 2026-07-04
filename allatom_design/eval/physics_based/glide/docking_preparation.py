"""Prepare receptor, ligand, and grid artifacts for Glide docking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure import AtomArray

from allatom_design.eval.physics_based.glide.schrodinger_runner import (
    run_grid_generation,
    run_ligprep,
    run_prepwizard,
    write_gridgen_input,
)

logger = logging.getLogger(__name__)


def compute_dynamic_outerbox(
    ligand_array: AtomArray,
    padding: float = 20.0,
) -> list[float]:
    """Compute OUTERBOX from ligand coordinate range + padding per axis.

    Per PoseX protocol: (x_range + padding, y_range + padding, z_range + padding).
    """
    heavy_mask = ligand_array.element != "H"
    coords = ligand_array[heavy_mask].coord if heavy_mask.any() else ligand_array.coord
    ranges = coords.max(axis=0) - coords.min(axis=0)
    return (ranges + padding).tolist()


def prepare_receptor_and_grid(
    protein_pdb: str,
    ligand_sdf: str,
    ligand_atom_array: AtomArray,
    ligand_centroid: np.ndarray,
    out_dir: str,
    sample_name: str,
    schrodinger_path: str,
    schrodinger_cfg: dict[str, Any],
    glide_cfg: dict[str, Any],
    timeout: int,
    designed_id: str | None = None,
) -> dict[str, str]:
    """Run PrepWizard, optional LigPrep, and Glide grid generation."""
    log_id = designed_id or sample_name
    prep_dir = Path(out_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{log_id}] Phase 2: PrepWizard")
    receptor_mae = str(prep_dir / f"{sample_name}_protein_prepared.mae")
    run_prepwizard(
        input_file=protein_pdb,
        output_file=receptor_mae,
        schrodinger_path=schrodinger_path,
        options=schrodinger_cfg.get("prepwizard", {}),
        timeout=timeout,
        log_dir=schrodinger_cfg.get("log_dir"),
    )

    ligand_file = ligand_sdf
    if glide_cfg.get("use_ligprep", True):
        try:
            logger.info(f"[{log_id}] Phase 2: LigPrep")
            prepared_ligand = str(prep_dir / f"{sample_name}_ligand_prepared.maegz")
            ligand_file = run_ligprep(
                input_sdf=ligand_sdf,
                output_file=prepared_ligand,
                schrodinger_path=schrodinger_path,
                options=schrodinger_cfg.get("ligprep", {}),
                timeout=timeout,
                log_dir=schrodinger_cfg.get("log_dir"),
            )
        except Exception as e:
            logger.warning(f"[{log_id}] LigPrep failed, using raw SDF: {e}")

    logger.info(f"[{log_id}] Phase 2: Grid generation")
    grid_cfg = glide_cfg.get("grid", {})
    outer_box = grid_cfg.get("outer_box")
    if outer_box is None:
        outer_box = compute_dynamic_outerbox(ligand_atom_array)

    gridgen_input = write_gridgen_input(
        receptor_mae=receptor_mae,
        grid_center=ligand_centroid.tolist(),
        out_dir=str(prep_dir),
        jobname="gridgen",
        inner_box=grid_cfg.get("inner_box", [10, 10, 10]),
        outer_box=outer_box,
        forcefield=grid_cfg.get("forcefield", "OPLS4"),
    )
    grid_file = run_grid_generation(
        input_file=gridgen_input,
        schrodinger_path=schrodinger_path,
        timeout=timeout,
    )

    return {
        "receptor_mae": receptor_mae,
        "ligand_file": ligand_file,
        "grid_file": grid_file,
    }
