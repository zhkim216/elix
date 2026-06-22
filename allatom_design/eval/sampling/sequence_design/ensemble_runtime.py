from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from allatom_design.eval.sampling.sequence_design.ensemble.conditioning import (
    ligand_conformer_conditioning_enabled,
    pharm_retrieval_conditioning_enabled,
)
from allatom_design.eval.sampling.sequence_design.ensemble.staging import (
    EnsembleStagingResult,
)
from allatom_design.eval.sampling.sequence_design.guidance import guidance_is_enabled
from allatom_design.eval.sampling.sequence_design.ensemble.ligand_conformer import (
    stage_ligand_conformer_ensembles,
)
from allatom_design.eval.sampling.sequence_design.ensemble.pharm_retrieval import (
    stage_pharm_retrieval_ensembles,
)


MODE_LIGAND_CONFORMER_ENSEMBLE = "ligand_conformer_ensemble"
MODE_PHARM_RETRIEVAL_ENSEMBLE = "pharm_retrieval_ensemble"


@dataclass(frozen=True)
class EnsembleRuntimeStaging:
    mode: str
    ensemble_staging: EnsembleStagingResult
    staging_message: str


def build_ensemble_runtime_staging(
    *,
    pdb_paths: list[str],
    sampling_inputs_for_setup: dict[str, Any],
    sampling_inputs_df: pd.DataFrame | None,
    guidance_cfg: DictConfig | dict | None,
    out_dir: str,
    cif_parse_cfg: DictConfig | dict | None,
    cif_save_cfg: DictConfig | dict | None,
    csv_suffix: str = "",
) -> EnsembleRuntimeStaging | None:
    if ligand_conformer_conditioning_enabled(sampling_inputs_for_setup):
        if guidance_is_enabled(guidance_cfg):
            raise NotImplementedError(
                "ligand_conformer ensemble_conditioning is not supported "
                "together with Potts guidance"
            )
        ensemble_cfg = sampling_inputs_for_setup["potts_sampling_cfg"]["ensemble_conditioning"]
        ensemble_staging = stage_ligand_conformer_ensembles(
            pdb_paths=pdb_paths,
            out_dir=out_dir,
            ensemble_cfg=ensemble_cfg,
            sampling_inputs_df=sampling_inputs_df,
            cif_parse_cfg=cif_parse_cfg,
            cif_save_cfg=cif_save_cfg,
            csv_suffix=csv_suffix,
        )
        return EnsembleRuntimeStaging(
            mode=MODE_LIGAND_CONFORMER_ENSEMBLE,
            ensemble_staging=ensemble_staging,
            staging_message=_staging_message("ligand conformer", ensemble_staging),
        )

    if pharm_retrieval_conditioning_enabled(sampling_inputs_for_setup):
        if guidance_is_enabled(guidance_cfg):
            raise NotImplementedError(
                "pharm_retrieval ensemble_conditioning is not supported "
                "together with Potts guidance"
            )
        ensemble_cfg = sampling_inputs_for_setup["potts_sampling_cfg"]["ensemble_conditioning"]
        ensemble_staging = stage_pharm_retrieval_ensembles(
            pdb_paths=pdb_paths,
            out_dir=out_dir,
            ensemble_cfg=ensemble_cfg,
            sampling_inputs_df=sampling_inputs_df,
            csv_suffix=csv_suffix,
        )
        return EnsembleRuntimeStaging(
            mode=MODE_PHARM_RETRIEVAL_ENSEMBLE,
            ensemble_staging=ensemble_staging,
            staging_message=_staging_message("pharm retrieval", ensemble_staging),
        )

    return None


def _staging_message(label: str, ensemble_staging: EnsembleStagingResult) -> str:
    return (
        f"Staged {label} ensemble members in "
        f"{ensemble_staging.root_dir}; manifest: {ensemble_staging.manifest_path}"
    )
