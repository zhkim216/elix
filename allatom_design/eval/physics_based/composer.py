"""Compose resolved ligand-eval rows through PB, Glide, and PB phases."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.physics_based.glide.docking_eval import (
    GlideEvalSettings,
    GlideSampleInput,
    evaluate_glide_sample,
)
from allatom_design.eval.physics_based.glide.result_parser import (
    compute_redock_vs_reference_rmsd,
)
from allatom_design.eval.physics_based.posebusters.core import run_pb_and_summarize
from allatom_design.eval.physics_based.structure_prep import (
    get_ligand_pn_unit_iids,
    preprocess_structure,
)
from allatom_design.utils.sample_io_utils import load_example_with_parse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalSettings:
    """Integrated ligand-eval settings shared across selected rows."""

    work_dir: str
    sample_dir: str | None
    schrodinger_path: str | None = None
    schrodinger_cfg: dict[str, Any] = field(default_factory=dict)
    glide_cfg: dict[str, Any] = field(default_factory=dict)
    pb_cfg: dict[str, Any] = field(default_factory=dict)
    cif_parse_cfg: dict[str, Any] | None = None
    structure_prep_cfg: dict[str, Any] = field(default_factory=dict)
    receptor_pn_unit_iids: list[str] | None = None
    ligand_pn_unit_iids: list[str] | None = None
    ref_sample_is_designed: bool = False
    num_workers: int = 1

    @property
    def glide_enabled(self) -> bool:
        return bool(self.glide_cfg.get("enabled", True))

    @property
    def timeout(self) -> int:
        return int(self.schrodinger_cfg.get("timeout", 3600))


def _value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _base_result(row: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in row.items()
        if not str(key).startswith("_")
    }
    result.setdefault("designed_sample_id", row.get("designed_sample_id"))
    result.setdefault("input_sample_id", row.get("input_sample_id", ""))
    result.setdefault("diffusion_idx", row.get("diffusion_idx"))
    result.setdefault("cif_path", row.get("cif_path"))
    result.setdefault("status", pd.NA)
    result.setdefault("error", pd.NA)
    return result


def _finalize_status(result: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if _value_present(result.get("error")):
        errors.append(str(result["error"]))
    for key, value in sorted(result.items()):
        if key.endswith("_error") and _value_present(value):
            errors.append(f"{key}: {value}")

    if errors:
        result["status"] = "failed"
        result["error"] = "; ".join(dict.fromkeys(errors))
    else:
        result["status"] = "success"
        result["error"] = pd.NA
    return result


def _as_dict_config(cfg: dict[str, Any] | DictConfig | None) -> dict[str, Any] | DictConfig | None:
    if cfg is not None and not isinstance(cfg, DictConfig):
        return OmegaConf.create(cfg)
    return cfg


def _compute_redock_vs_ref_rmsd(
    *,
    sample_cif: Path,
    cif_parse_cfg: dict[str, Any] | None,
    redock_sdf_path: str,
    designed_id: str,
) -> dict[str, Any]:
    if not sample_cif.exists():
        return {
            "redock_vs_ref_error": f"sample_cif_not_found: {sample_cif}",
        }

    try:
        ref_example = load_example_with_parse(
            str(sample_cif),
            cif_parse_cfg=_as_dict_config(cif_parse_cfg),
        )
        ref_array = ref_example["atom_array"]
        ref_lig_ids = get_ligand_pn_unit_iids(ref_array)
        if not ref_lig_ids:
            return {"redock_vs_ref_error": "no_ligand_in_sample_cif"}
        ref_lig = ref_array[np.isin(ref_array.pn_unit_iid, ref_lig_ids)]
        rmsd_result = compute_redock_vs_reference_rmsd(
            redock_sdf_path=redock_sdf_path,
            ref_ligand_array=ref_lig,
        )
        if rmsd_result.get("error"):
            return {"redock_vs_ref_error": rmsd_result["error"]}
        return {
            "redock_vs_ref_ligand_rmsd": rmsd_result.get(
                "redock_vs_ref_ligand_rmsd"
            )
        }
    except Exception as exc:
        logger.warning("[%s] Redock vs ref RMSD failed: %s", designed_id, exc)
        return {"redock_vs_ref_error": str(exc)}


def process_single_sample(
    row: dict[str, Any],
    settings: EvalSettings,
) -> dict[str, Any]:
    """Run one resolved selection row through the integrated eval phases."""
    result = _base_result(row)
    designed_id = str(result.get("designed_sample_id", ""))
    input_id = str(result.get("input_sample_id", ""))
    cif_path = result.get("cif_path")
    if not _value_present(cif_path):
        result["error"] = "missing_cif_path"
        return _finalize_status(result)

    sample_name = Path(str(cif_path)).stem
    work_dir = Path(settings.work_dir)
    prep_dir = work_dir / "prep" / sample_name
    mininplace_dir = work_dir / "mininplace" / sample_name
    redocking_dir = work_dir / "redocking" / sample_name

    try:
        prep_dir.mkdir(parents=True, exist_ok=True)
        sample_info = preprocess_structure(
            cif_path=str(cif_path),
            out_dir=str(prep_dir),
            sample_id=sample_name,
            cif_parse_cfg=settings.cif_parse_cfg,
            receptor_pn_unit_iids=settings.receptor_pn_unit_iids,
            ligand_pn_unit_iids=settings.ligand_pn_unit_iids,
            ligand_sdf_add_missing_atoms=settings.structure_prep_cfg.get(
                "ligand_sdf_add_missing_atoms", True
            ),
        )
    except Exception as exc:
        logger.error("[%s] Preprocessing failed: %s", designed_id, exc)
        result["error"] = f"preprocessing_failed: {exc}"
        return _finalize_status(result)

    protein_pdb = sample_info["protein_pdb_path"]
    ligand_sdf = sample_info["ligand_sdf_path"]

    try:
        result.update(
            run_pb_and_summarize(
                mol_pred=ligand_sdf,
                mol_cond=protein_pdb,
                prefix="pb_af3",
                pb_cfg=settings.pb_cfg,
            )
        )
    except Exception as exc:
        logger.error("[%s] PB AF3 failed: %s", designed_id, exc)
        result["pb_af3_error"] = str(exc)

    glide_result = None
    if settings.glide_enabled:
        if not settings.schrodinger_path:
            result["glide_error"] = "schrodinger_unavailable"
        else:
            try:
                glide_result = evaluate_glide_sample(
                    GlideSampleInput(
                        designed_sample_id=designed_id,
                        sample_name=sample_name,
                        protein_pdb=protein_pdb,
                        ligand_sdf=ligand_sdf,
                        ligand_atom_array=sample_info["ligand_atom_array"],
                        ligand_centroid=sample_info["ligand_centroid"],
                        prep_dir=str(prep_dir),
                        mininplace_dir=str(mininplace_dir),
                        redocking_dir=str(redocking_dir),
                    ),
                    GlideEvalSettings(
                        schrodinger_path=settings.schrodinger_path,
                        schrodinger_cfg=settings.schrodinger_cfg,
                        glide_cfg=settings.glide_cfg,
                        timeout=settings.timeout,
                    ),
                )
                result.update(glide_result.metrics)
            except Exception as exc:
                logger.error("[%s] Glide prep failed: %s", designed_id, exc)
                result["glide_prep_error"] = str(exc)

    if glide_result and glide_result.inplace_sdf:
        try:
            result.update(
                run_pb_and_summarize(
                    mol_pred=glide_result.inplace_sdf,
                    mol_cond=protein_pdb,
                    prefix="pb_mininplace",
                    pb_cfg=settings.pb_cfg,
                )
            )
        except Exception as exc:
            logger.error("[%s] PB mininplace failed: %s", designed_id, exc)
            result["pb_mininplace_error"] = str(exc)

    if glide_result and glide_result.redock_sdf:
        try:
            result.update(
                run_pb_and_summarize(
                    mol_pred=glide_result.redock_sdf,
                    mol_cond=protein_pdb,
                    prefix="pb_redocking",
                    pb_cfg=settings.pb_cfg,
                )
            )
        except Exception as exc:
            logger.error("[%s] PB redocking failed: %s", designed_id, exc)
            result["pb_redocking_error"] = str(exc)

        if settings.sample_dir:
            ref_stem = designed_id if settings.ref_sample_is_designed else input_id
            result.update(
                _compute_redock_vs_ref_rmsd(
                    sample_cif=Path(settings.sample_dir) / f"{ref_stem}.cif",
                    cif_parse_cfg=settings.cif_parse_cfg,
                    redock_sdf_path=glide_result.redock_sdf,
                    designed_id=designed_id,
                )
            )
        else:
            result["redock_vs_ref_error"] = "sample_dir_not_configured"

    return _finalize_status(result)


def evaluate_batch(
    selected_df: pd.DataFrame,
    settings: EvalSettings,
) -> pd.DataFrame:
    """Evaluate selected, already-resolved rows."""
    work_dir = Path(settings.work_dir)
    for name in ("prep", "mininplace", "redocking"):
        (work_dir / name).mkdir(parents=True, exist_ok=True)

    rows = selected_df.to_dict("records")
    if not rows:
        return pd.DataFrame()

    results: list[dict[str, Any]] = []
    if settings.num_workers <= 1:
        for i, row in enumerate(rows):
            logger.info(
                "[%s/%s] %s diffusion_%s",
                i + 1,
                len(rows),
                row.get("designed_sample_id"),
                row.get("diffusion_idx"),
            )
            results.append(process_single_sample(row, settings))
    else:
        with ProcessPoolExecutor(max_workers=settings.num_workers) as executor:
            futures = {
                executor.submit(process_single_sample, row, settings): row
                for row in rows
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failed = _base_result(row)
                    failed["error"] = f"worker_failed: {exc}"
                    results.append(_finalize_status(failed))

    return pd.DataFrame(results)
