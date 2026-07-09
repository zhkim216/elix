from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import atomworks.enums as aw_enums
import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf

from allatom_design.data.datasets.atomworks_sd import sd_collator
from allatom_design.data.transform.sd_featurizer import sd_featurizer_for_design
from allatom_design.eval.utils.input_files import get_pdb_files
from allatom_design.eval.utils.input_preprocessing import preprocess_input
from allatom_design.eval.utils.sampling_inputs import (
    ROLE_DERIVED_SAMPLE_ID_COLUMN,
    matched_role_sampling_input_row,
    normalize_role_sampling_inputs_df,
    resolve_query_pn_unit_iids,
)
from allatom_design.utils.atom_array_utils import get_res_name_by_chain_res_id
from allatom_design.utils.sample_io_utils import load_example_with_parse
from allatom_design.utils.tensor_utils import to


def create_sample_dict(
    *,
    sample_paths: list[str] | None = None,
    sample_ids: list[str] | None = None,
    prefix: str = "input",
) -> dict[str, dict[str, str]]:
    """Build a sample dictionary keyed by sample ID."""
    if sample_paths is None:
        sample_paths = []

    if sample_ids is None:
        sample_ids = [Path(sample_path).stem for sample_path in sample_paths]

    sample_dict: dict[str, dict[str, str]] = defaultdict(dict)
    for i, sample_id in enumerate(sample_ids):
        sample_dict[sample_id][f"{prefix}_sample_path"] = sample_paths[i]
        sample_dict[sample_id][f"{prefix}_sample_id"] = sample_ids[i]
    return sample_dict


def _pdb_cfg_without_array_split(pdb_cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    cfg_dict = (
        OmegaConf.to_container(pdb_cfg, resolve=True)
        if isinstance(pdb_cfg, DictConfig)
        else dict(pdb_cfg)
    )
    cfg_dict["array_id"] = None
    cfg_dict["num_arrays"] = None
    return cfg_dict


def _slice_role_rows_for_array(
    role_df: pd.DataFrame,
    *,
    array_id: int | None,
    num_arrays: int | None,
) -> pd.DataFrame:
    if array_id is None:
        return role_df
    if num_arrays is None:
        raise ValueError("pdb_cfg.num_arrays is required when pdb_cfg.array_id is set")

    array_id = int(array_id)
    num_arrays = int(num_arrays)
    if array_id < 0 or array_id >= num_arrays:
        raise ValueError(f"pdb_cfg.array_id must be in [0, {num_arrays}), got {array_id}")

    chunk_size = math.ceil(len(role_df) / num_arrays) if len(role_df) else 0
    start_idx = array_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(role_df))
    selected = role_df.iloc[start_idx:end_idx].reset_index(drop=True)
    print(f"Role sampling array {array_id}/{num_arrays}: using {len(selected)} rows")
    return selected


def prepare_sample_dict(
    cfg: DictConfig | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,
    prefix: str = "input",
) -> dict[str, dict[str, str]]:
    """Resolve input structure files from ``cfg.pdb_cfg`` into ``sample_dict``."""
    if cfg is None:
        raise ValueError("cfg must be provided")

    if sampling_inputs_df is not None:
        role_df = normalize_role_sampling_inputs_df(sampling_inputs_df)
        sample_paths = get_pdb_files(**_pdb_cfg_without_array_split(cfg.pdb_cfg))

        path_by_pdb_key = {Path(sample_path).stem: sample_path for sample_path in sample_paths}
        missing_pdb_keys = sorted(
            set(role_df["pdb_key"].astype(str)) - set(path_by_pdb_key)
        )
        if missing_pdb_keys:
            raise FileNotFoundError(
                "sampling_inputs_csv refers to source pdb_key values not found in "
                f"cfg.pdb_cfg inputs: {missing_pdb_keys[:10]}"
            )

        if cfg.debug:
            role_df = role_df.head(cfg.num_debug_samples).copy()

        role_df = _slice_role_rows_for_array(
            role_df,
            array_id=cfg.pdb_cfg.get("array_id", None),
            num_arrays=cfg.pdb_cfg.get("num_arrays", None),
        )
        role_sample_paths = [
            path_by_pdb_key[str(row["pdb_key"])]
            for _, row in role_df.iterrows()
        ]
        role_sample_ids = [
            str(row[ROLE_DERIVED_SAMPLE_ID_COLUMN])
            for _, row in role_df.iterrows()
        ]
        return create_sample_dict(
            sample_paths=role_sample_paths,
            sample_ids=role_sample_ids,
            prefix=prefix,
        )

    sample_paths = get_pdb_files(**cfg.pdb_cfg)
    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]

    return create_sample_dict(sample_paths=sample_paths, prefix=prefix)


def get_sd_batch(
    pdb_paths: list[str] | None = None,
    *,
    sample_ids: list[str] | None = None,
    sample_is_designed: bool = False,
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,
    device: str | None = None,
    parallel_pool: Parallel | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return a sequence-design model feature batch for structure paths."""
    if pdb_paths is None:
        return {}
    if sample_ids is None:
        sample_ids = [Path(pdb_path).stem for pdb_path in pdb_paths]
    if len(sample_ids) != len(pdb_paths):
        raise ValueError(
            f"sample_ids length ({len(sample_ids)}) must match pdb_paths length ({len(pdb_paths)})"
        )

    if parallel_pool is None:
        batch_examples = [
            get_sd_example(
                pdb_path=pdb_path,
                sample_id=sample_ids[i],
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                featurizer_cfg=featurizer_cfg,
                sampling_inputs_df=sampling_inputs_df,
                sample_is_designed=sample_is_designed,
            )
            for i, pdb_path in enumerate(pdb_paths)
        ]
    else:
        batch_examples = parallel_pool(
            delayed(get_sd_example)(
                pdb_path=pdb_path,
                sample_id=sample_ids[i],
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                featurizer_cfg=featurizer_cfg,
                sampling_inputs_df=sampling_inputs_df,
                sample_is_designed=sample_is_designed,
            )
            for i, pdb_path in enumerate(pdb_paths)
        )

    native_res_name_by_chain_res_id = [
        example.pop("native_res_name_by_chain_res_id")
        for example in batch_examples
    ]

    batch = sd_collator(batch_examples)
    batch["native_res_name_by_chain_res_id"] = native_res_name_by_chain_res_id
    batch = to(batch, device)
    return batch


def get_sd_example(
    pdb_path: str | None = None,
    *,
    sample_id: str | None = None,
    sample_is_designed: bool = False,
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return one sequence-design model feature example for a structure path."""
    if pdb_path is None:
        raise ValueError("pdb_path must be provided")

    example = load_example_with_parse(pdb_path, cif_parse_cfg)
    example = preprocess_input(
        example=example,
        preprocess_cfg=preprocess_cfg,
        sample_is_designed=sample_is_designed,
    )
    native_res_name_by_chain_res_id = get_res_name_by_chain_res_id(example["atom_array"])

    pdb_key = Path(pdb_path).stem
    sample_id = sample_id or pdb_key
    example["example_id"] = sample_id
    pdb_id = pdb_key.split("_")[0]
    example["query_pn_unit_iids"] = resolve_query_pn_unit_iids(
        atom_array=example["atom_array"],
        sampling_inputs_df=sampling_inputs_df,
        pdb_id=pdb_id,
        pdb_key=pdb_key,
        sample_id=sample_id,
    )
    sampling_row = matched_role_sampling_input_row(
        sampling_inputs_df,
        sample_id=str(sample_id),
    )
    if sampling_row is not None:
        example["crop_center_pn_unit_iids"] = list(sampling_row["frame_pn_unit_iids"])
    if sample_is_designed and _is_protein_nonpolymer_complex(example["atom_array"]):
        example.setdefault("data_category", "interface")
        example.setdefault("phase", "train")

    featurizer_cfg = OmegaConf.to_container(featurizer_cfg, resolve=True)
    featurizer = sd_featurizer_for_design(**featurizer_cfg, sample_is_designed=sample_is_designed)

    featurized = featurizer(example)
    featurized["native_res_name_by_chain_res_id"] = native_res_name_by_chain_res_id
    return featurized


def _is_protein_nonpolymer_complex(atom_array: AtomArray) -> bool:
    if "chain_type" not in atom_array.get_annotation_categories():
        return False
    chain_types = atom_array.chain_type
    protein_values = [chain_type.value for chain_type in aw_enums.ChainTypeInfo.PROTEINS]
    nonpolymer_values = [chain_type.value for chain_type in aw_enums.ChainTypeInfo.NON_POLYMERS]
    has_protein = np.isin(chain_types, protein_values).any()
    has_nonpolymer = np.isin(chain_types, nonpolymer_values).any()
    return bool(has_protein and has_nonpolymer)
