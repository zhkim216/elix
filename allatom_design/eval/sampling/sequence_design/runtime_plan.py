from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.sampling.sequence_design.ensemble_runtime import (
    MODE_LIGAND_CONFORMER_ENSEMBLE,
    MODE_PHARM_RETRIEVAL_ENSEMBLE,
    build_ensemble_runtime_staging,
)
from allatom_design.eval.sampling.sequence_design.constraints import (
    POS_CONSTRAINT_COLUMNS,
    parse_fixed_pos_info,
    parse_pos_restrict_aatype_info,
)
from allatom_design.eval.sampling.sequence_design.guidance import (
    guidance_mode,
    inject_guidance_inputs,
)
from allatom_design.eval.sampling.sequence_design.ensemble.staging import (
    EnsembleStagingResult,
)
from allatom_design.eval.sampling.sequence_design.masks import initialize_sampling_masks


MODE_DEFAULT = "default"


@dataclass(frozen=True)
class SamplingRuntimeBatch:
    batch: dict[str, Any]
    sampling_inputs: dict[str, Any]
    guidance_cfg_dict: dict[str, Any] | None
    guidance_metadata_by_example_id: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SamplingRuntimePlan:
    mode: str
    pdb_paths: list[str]
    sampling_inputs_df: pd.DataFrame | None
    pos_constraint_df: pd.DataFrame | None
    guidance_cfg: DictConfig | dict | None = None
    selectivity_residue_alignment_df: pd.DataFrame | None = None
    ensemble_staging: EnsembleStagingResult | None = None
    staging_message: str | None = None

    def iter_batches(self, *, batch_size: int) -> Iterator[list[str]]:
        if self.ensemble_staging is not None:
            yield from self.ensemble_staging.iter_member_batches(max_members=int(batch_size))
        else:
            yield from _iter_default_sampling_batches(self.pdb_paths, batch_size)

    def target_count(self, batch_pdb_paths: list[str] | None = None) -> int:
        if self.ensemble_staging is not None:
            return self.ensemble_staging.target_count(batch_pdb_paths)
        if batch_pdb_paths is None:
            return len(self.pdb_paths)
        return len(batch_pdb_paths)

    def annotate_batch(
        self,
        batch: dict[str, Any],
        *,
        batch_pdb_paths: list[str],
        device: str | torch.device | None,
    ) -> dict[str, Any]:
        if self.ensemble_staging is None:
            return batch
        return self.ensemble_staging.annotate_batch(
            batch,
            batch_pdb_paths=batch_pdb_paths,
            device=device,
        )

    def prepare_batch(
        self,
        *,
        batch: dict[str, Any],
        batch_pdb_paths: list[str],
        sampling_cfg: DictConfig,
        protein_only: bool,
        device: str | torch.device | None,
    ) -> SamplingRuntimeBatch:
        batch = self.annotate_batch(
            batch,
            batch_pdb_paths=batch_pdb_paths,
            device=device,
        )
        batch = initialize_sampling_masks(batch, protein_only=protein_only)
        batch = parse_fixed_pos_info(
            batch,
            self.pos_constraint_df,
            verbose=sampling_cfg.verbose,
        )

        sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
        sampling_inputs.pop("sample_token_prefix", None)
        sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(
            batch,
            self.pos_constraint_df,
            verbose=sampling_cfg.verbose,
        )

        guidance_cfg_dict: dict[str, Any] | None = None
        guidance_metadata_by_example_id: dict[str, dict[str, Any]] = {}
        if self.guidance_cfg is not None:
            guidance_cfg_dict, guidance_metadata_by_example_id = inject_guidance_inputs(
                guidance_cfg=self.guidance_cfg,
                batch=batch,
                sampling_inputs_df=self.sampling_inputs_df,
                selectivity_residue_alignment_df=self.selectivity_residue_alignment_df,
            )
            sampling_inputs.setdefault("potts_sampling_cfg", {})
            sampling_inputs["potts_sampling_cfg"]["guidance_cfg"] = guidance_cfg_dict

        return SamplingRuntimeBatch(
            batch=batch,
            sampling_inputs=sampling_inputs,
            guidance_cfg_dict=guidance_cfg_dict,
            guidance_metadata_by_example_id=guidance_metadata_by_example_id,
        )


def build_sampling_runtime_plan(
    *,
    pdb_paths: list[str],
    sampling_cfg: DictConfig,
    sampling_inputs_df: pd.DataFrame | None,
    selectivity_residue_alignment_df: pd.DataFrame | None,
    pos_constraint_df: pd.DataFrame | None,
    guidance_cfg: DictConfig | dict | None,
    out_dir: str,
    cif_parse_cfg: DictConfig | dict | None,
    cif_save_cfg: DictConfig | dict | None,
    csv_suffix: str = "",
) -> SamplingRuntimePlan:
    if _selectivity_guidance_requested(
        guidance_cfg=guidance_cfg,
        selectivity_residue_alignment_df=selectivity_residue_alignment_df,
    ):
        raise NotImplementedError(
            "guidance.mode=selectivity is owned by "
            "allatom_design.eval.sampling.run_selectivity_elix; "
            "generic run_elix/design_sequence only supports standard and ensemble sampling"
        )

    sampling_inputs_for_setup = OmegaConf.to_container(sampling_cfg, resolve=True)
    sampling_inputs_for_setup.pop("sample_token_prefix", None)

    ensemble_runtime = build_ensemble_runtime_staging(
        pdb_paths=pdb_paths,
        sampling_inputs_for_setup=sampling_inputs_for_setup,
        sampling_inputs_df=sampling_inputs_df,
        guidance_cfg=guidance_cfg,
        out_dir=out_dir,
        cif_parse_cfg=cif_parse_cfg,
        cif_save_cfg=cif_save_cfg,
        csv_suffix=csv_suffix,
    )
    if ensemble_runtime is not None:
        return _plan_from_ensemble_staging(
            mode=ensemble_runtime.mode,
            ensemble_staging=ensemble_runtime.ensemble_staging,
            pos_constraint_df=pos_constraint_df,
            guidance_cfg=guidance_cfg,
            staging_message=ensemble_runtime.staging_message,
        )

    return SamplingRuntimePlan(
        mode=MODE_DEFAULT,
        pdb_paths=pdb_paths,
        sampling_inputs_df=sampling_inputs_df,
        pos_constraint_df=normalize_pos_constraint_df(pos_constraint_df),
        guidance_cfg=guidance_cfg,
    )


def _plan_from_ensemble_staging(
    *,
    mode: str,
    ensemble_staging: EnsembleStagingResult,
    pos_constraint_df: pd.DataFrame | None,
    guidance_cfg: DictConfig | dict | None,
    staging_message: str,
) -> SamplingRuntimePlan:
    return SamplingRuntimePlan(
        mode=mode,
        pdb_paths=ensemble_staging.pdb_paths,
        sampling_inputs_df=ensemble_staging.sampling_inputs_df,
        pos_constraint_df=normalize_pos_constraint_df(
            ensemble_staging.expand_pos_constraints(pos_constraint_df)
        ),
        guidance_cfg=guidance_cfg,
        ensemble_staging=ensemble_staging,
        staging_message=staging_message,
    )


def _selectivity_guidance_requested(
    *,
    guidance_cfg: DictConfig | dict | None,
    selectivity_residue_alignment_df: pd.DataFrame | None,
) -> bool:
    if selectivity_residue_alignment_df is not None:
        return True
    if guidance_cfg is None:
        return False
    return bool(guidance_cfg.get("enabled", False)) and guidance_mode(guidance_cfg) == "selectivity"


def _iter_default_sampling_batches(
    pdb_paths: list[str],
    batch_size: int,
) -> Iterator[list[str]]:
    for bi in range(0, len(pdb_paths), batch_size):
        yield pdb_paths[bi : bi + batch_size]


def normalize_pos_constraint_df(
    pos_constraint_df: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if pos_constraint_df is None:
        return None

    if not set(pos_constraint_df.columns).issubset(POS_CONSTRAINT_COLUMNS):
        print(
            "Invalid columns in pos_constraint_df. Expected subset of "
            f"{list(POS_CONSTRAINT_COLUMNS)}. Found: {pos_constraint_df.columns}"
        )
        cols_to_keep = [c for c in POS_CONSTRAINT_COLUMNS if c in pos_constraint_df.columns]
        print(f"Keeping columns: {cols_to_keep}")
        pos_constraint_df = pos_constraint_df[cols_to_keep]

    return pos_constraint_df.set_index("pdb_key").replace("", np.nan)
