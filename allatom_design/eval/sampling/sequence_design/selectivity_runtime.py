from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.sampling.sequence_design.ensemble.conditioning import (
    ligand_conformer_conditioning_enabled,
    pharm_retrieval_conditioning_enabled,
)
from allatom_design.eval.sampling.sequence_design.guidance import (
    guidance_is_enabled,
    guidance_mode,
)
from allatom_design.eval.sampling.sequence_design.runtime_plan import (
    SamplingRuntimeInputBatch,
    SamplingRuntimePlan,
    normalize_pos_constraint_df,
)
from allatom_design.eval.selectivity import (
    normalize_target_ligand_side,
)


MODE_SELECTIVITY_PAIR = "selectivity_pair"


@dataclass(frozen=True)
class SelectivitySamplingRuntimePlan(SamplingRuntimePlan):
    """Runtime plan for paired selectivity sequence design."""

    def iter_input_batches(self, *, batch_size: int) -> Iterator[SamplingRuntimeInputBatch]:
        del batch_size
        for pdb_paths in iter_selectivity_pair_batches(
            pdb_paths=self.pdb_paths,
            sampling_inputs_df=self.sampling_inputs_df,
        ):
            yield SamplingRuntimeInputBatch(
                pdb_paths=pdb_paths,
                sample_ids=[Path(pdb_path).stem for pdb_path in pdb_paths],
            )


def build_selectivity_sampling_runtime_plan(
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
) -> SelectivitySamplingRuntimePlan:
    del out_dir, cif_parse_cfg, cif_save_cfg, csv_suffix
    if sampling_inputs_df is None:
        raise ValueError("selectivity sequence design requires paired sampling_inputs_csv")
    if selectivity_residue_alignment_df is None:
        raise ValueError("selectivity sequence design requires pair_residue_alignment.csv")
    if not guidance_is_enabled(guidance_cfg) or guidance_mode(guidance_cfg) != "selectivity":
        raise ValueError("selectivity sequence design requires sampling_cfg.guidance.mode=selectivity")
    if normalize_target_ligand_side(guidance_cfg.get("target_ligand_side", None)) is None:
        raise ValueError("selectivity sequence design requires explicit target_ligand_side 1 or 2")

    sampling_inputs_for_setup = OmegaConf.to_container(sampling_cfg, resolve=True)
    sampling_inputs_for_setup.pop("sample_token_prefix", None)
    if ligand_conformer_conditioning_enabled(sampling_inputs_for_setup):
        raise NotImplementedError("selectivity guidance is not supported with ligand_conformer ensemble_conditioning")
    if pharm_retrieval_conditioning_enabled(sampling_inputs_for_setup):
        raise NotImplementedError("selectivity guidance is not supported with pharm_retrieval ensemble_conditioning")

    return SelectivitySamplingRuntimePlan(
        mode=MODE_SELECTIVITY_PAIR,
        pdb_paths=pdb_paths,
        sampling_inputs_df=sampling_inputs_df,
        pos_constraint_df=normalize_pos_constraint_df(pos_constraint_df),
        guidance_cfg=guidance_cfg,
        selectivity_residue_alignment_df=selectivity_residue_alignment_df,
    )


def iter_selectivity_pair_batches(
    *,
    pdb_paths: list[str],
    sampling_inputs_df: pd.DataFrame | None,
) -> Iterator[list[str]]:
    if sampling_inputs_df is None:
        raise ValueError("selectivity pair sampling requires sampling_inputs_df")

    path_by_stem = {Path(path).stem: path for path in pdb_paths}
    present_rows = sampling_inputs_df[
        sampling_inputs_df["pdb_key"].astype(str).isin(path_by_stem.keys())
    ].copy()
    missing_keys = sorted(set(path_by_stem) - set(present_rows["pdb_key"].astype(str)))
    if missing_keys:
        raise ValueError(
            "selectivity pair sampling received input CIFs not present in sampling_inputs_csv: "
            f"{missing_keys[:10]}"
        )

    emitted: set[str] = set()
    for pair_id, group in present_rows.groupby("selectivity_pair_id", sort=False):
        side_rows = group.set_index(group["selectivity_side"].astype(int))
        if set(side_rows.index) != {1, 2}:
            raise ValueError(
                "selectivity pair sampling requires both sides in the same shard; "
                f"selectivity_pair_id={pair_id!r} has sides {sorted(side_rows.index.tolist())}"
            )
        side1_key = str(side_rows.loc[1, "pdb_key"])
        side2_key = str(side_rows.loc[2, "pdb_key"])
        emitted.update({side1_key, side2_key})
        yield [path_by_stem[side1_key], path_by_stem[side2_key]]

    if emitted != set(path_by_stem):
        missing = sorted(set(path_by_stem) - emitted)
        raise ValueError(f"selectivity pair sampling did not emit paths for keys: {missing[:10]}")
