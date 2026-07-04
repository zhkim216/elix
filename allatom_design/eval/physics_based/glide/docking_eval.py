"""Glide-only evaluation phases for integrated ligand evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from biotite.structure import AtomArray

from allatom_design.eval.physics_based.glide.docking import run_docking_mode
from allatom_design.eval.physics_based.glide.docking_preparation import (
    prepare_receptor_and_grid,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlideSampleInput:
    """Prepared per-sample inputs consumed by Glide phases."""

    designed_sample_id: str
    sample_name: str
    protein_pdb: str
    ligand_sdf: str
    ligand_atom_array: AtomArray
    ligand_centroid: np.ndarray
    prep_dir: str
    mininplace_dir: str
    redocking_dir: str


@dataclass(frozen=True)
class GlideEvalSettings:
    """Glide runtime settings shared across samples."""

    schrodinger_path: str
    schrodinger_cfg: dict[str, Any]
    glide_cfg: dict[str, Any]
    timeout: int = 3600


@dataclass(frozen=True)
class GlideEvalResult:
    """Glide phase metrics plus pose files for downstream PB/RMSD phases."""

    metrics: dict[str, Any]
    inplace_sdf: str | None = None
    redock_sdf: str | None = None


def _prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def evaluate_glide_sample(
    sample: GlideSampleInput,
    settings: GlideEvalSettings,
) -> GlideEvalResult:
    """Run Glide prep, mininplace, and redocking for one prepared sample."""
    metrics: dict[str, Any] = {}
    inplace_sdf: str | None = None
    redock_sdf: str | None = None

    prep = prepare_receptor_and_grid(
        protein_pdb=sample.protein_pdb,
        ligand_sdf=sample.ligand_sdf,
        ligand_atom_array=sample.ligand_atom_array,
        ligand_centroid=sample.ligand_centroid,
        out_dir=sample.prep_dir,
        sample_name=sample.sample_name,
        schrodinger_path=settings.schrodinger_path,
        schrodinger_cfg=settings.schrodinger_cfg,
        glide_cfg=settings.glide_cfg,
        timeout=settings.timeout,
        designed_id=sample.designed_sample_id,
    )

    modes = settings.glide_cfg.get("modes", {})
    if modes.get("inplace_scoring", True):
        try:
            logger.info("[%s] Glide mininplace", sample.designed_sample_id)
            inplace_metrics = run_docking_mode(
                grid_file=prep["grid_file"],
                ligand_file=prep["ligand_file"],
                work_dir=sample.mininplace_dir,
                schrodinger_path=settings.schrodinger_path,
                glide_cfg=settings.glide_cfg,
                mode="mininplace",
                timeout=settings.timeout,
            )
            inplace_sdf = inplace_metrics.get("sdf_path")
            metrics.update(_prefix_metrics("glide_inplace", inplace_metrics))
        except Exception as exc:
            logger.error("[%s] Glide mininplace failed: %s", sample.designed_sample_id, exc)
            metrics["glide_inplace_error"] = str(exc)

    if modes.get("redocking", True):
        try:
            logger.info("[%s] Glide redocking", sample.designed_sample_id)
            redock_metrics = run_docking_mode(
                grid_file=prep["grid_file"],
                ligand_file=prep["ligand_file"],
                work_dir=sample.redocking_dir,
                schrodinger_path=settings.schrodinger_path,
                glide_cfg=settings.glide_cfg,
                mode="redocking",
                timeout=settings.timeout,
            )
            redock_sdf = redock_metrics.get("sdf_path")
            metrics.update(_prefix_metrics("glide_redock", redock_metrics))
        except Exception as exc:
            logger.error("[%s] Glide redocking failed: %s", sample.designed_sample_id, exc)
            metrics["glide_redock_error"] = str(exc)

    return GlideEvalResult(
        metrics=metrics,
        inplace_sdf=inplace_sdf,
        redock_sdf=redock_sdf,
    )
