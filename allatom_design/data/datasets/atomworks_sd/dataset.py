"""``AtomworksSDDataset`` -- the torch dataset that serves cached examples.

The class is intentionally thin: ``__init__`` orchestrates the index-building
pipeline (delegated to :mod:`metadata`, :mod:`interface` and :mod:`sampling`)
and the remaining methods implement the ``MolecularDataset`` interface.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from atomworks.ml.datasets import MolecularDataset
from omegaconf import DictConfig
from typing_extensions import override

from allatom_design.data.sampler import Sampler
from allatom_design.data.transform import sd_featurizer

from allatom_design.data.datasets.atomworks_sd import metadata as md
from allatom_design.data.datasets.atomworks_sd.sampling import (
    add_sampling_weights,
    validate_sampling_weights,
)
from allatom_design.data.datasets.atomworks_sd.runtime_index import (
    RuntimeMetadataIndex,
)

logger = logging.getLogger(__name__)

TRAIN_EXAMPLE_MODES = {"both", "monomer_only"}


def build_train_parsed_index(cfg: DictConfig) -> tuple[pd.Series | pd.DataFrame, list]:
    """Run the legacy Pandas pipeline once for an explicit index build."""

    train_example_mode = str(cfg.get("train_example_mode", "both"))
    if train_example_mode not in TRAIN_EXAMPLE_MODES:
        raise ValueError(
            f"Invalid data.train_example_mode={train_example_mode!r}; "
            f"expected one of {sorted(TRAIN_EXAMPLE_MODES)}"
        )
    metadata_path = cfg.train_metadata_path
    metadata_df, val_cluster_ids = md.process_train_metadata_df(
        metadata_path,
        cfg=cfg,
        phase="train",
    )
    dataset_name = cfg.get("dataset_name", Path(metadata_path).parent.name)
    monomer_df = md.build_monomer_chain_df(
        metadata_df,
        cfg,
        dataset_name,
        val_cluster_ids,
    )
    if train_example_mode == "both":
        interface_df = md.build_train_interface_df(
            metadata_df,
            cfg,
            dataset_name,
            val_cluster_ids,
        )
    else:
        interface_df = pd.DataFrame(
            {"protein_cluster_multiset": pd.Series(dtype=object)}
        )
    monomer_df, interface_df = add_sampling_weights(
        monomer_df=monomer_df,
        interface_df=interface_df,
        alphas_interface=cfg.sampling_weights["alphas_interface"],
        cluster_col="q_pn_unit_cluster_id",
        fixed_k=cfg.sampling_weights.get("fixed_k"),
        single_protein_context_weight=cfg.sampling_weights.get(
            "single_protein_context_weight", 1.0
        ),
        multi_protein_context_weight=cfg.sampling_weights.get(
            "multi_protein_context_weight", 1.0
        ),
        clustering_cfg=cfg.get("clustering", {}),
    )
    validate_sampling_weights(monomer_df, interface_df)
    return md.parse_train_dfs(monomer_df, interface_df, cfg), val_cluster_ids


def build_val_parsed_index(cfg: DictConfig) -> pd.Series | pd.DataFrame:
    """Build the final validation records for an explicit index build."""

    metadata_df = md.process_val_metadata_df(cfg.val_metadata_path)
    return md.parse_val_df(metadata_df)


class AtomworksSDDataset(MolecularDataset):
    def __init__(self, cfg: DictConfig, phase: Literal["train", "val"]):
        super().__init__(name=f"sd::{phase}", transform=None)
        self.cfg = cfg
        self.phase = phase
        self.save_failed_examples_to_dir = cfg.save_failed_examples_to_dir
        self.val_cluster_ids: list = []
        self._runtime_index: RuntimeMetadataIndex | None = None

        self.featurizer = sd_featurizer.sd_featurizer(
            **cfg.featurizer_cfg,
            is_validation=self.phase == "val",
            remove_keys=sd_featurizer.INFERENCE_ONLY_KEYS,
        )

        runtime_index_path = cfg.get("runtime_index_path")
        if runtime_index_path:
            self.metadata_path = (
                cfg.train_metadata_path
                if self.phase == "train"
                else cfg.val_metadata_path
            )
            self._runtime_index = RuntimeMetadataIndex(
                runtime_index_path,
                cfg=cfg,
                phase=phase,
            )
            if self.phase == "train":
                self._sampler = Sampler(self.get_sampling_weights())
                self._rng, self._samples = None, None
        elif cfg.get("debug", False):
            if self.phase == "train":
                self._build_train_index()
            else:
                self._build_val_index()
        else:
            raise ValueError(
                "Full-data atomworks_sd training requires "
                "data.runtime_index_path. Build the matching Arrow index "
                "explicitly with `python -m "
                "allatom_design.data.datasets.atomworks_sd.build_runtime_index`. "
                "The dynamic Pandas path is restricted to data.debug=true."
            )

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    def _build_train_index(self) -> None:
        self.metadata_path = self.cfg.train_metadata_path
        self.parsed_df, self.val_cluster_ids = build_train_parsed_index(
            self.cfg
        )
        self._sampler = Sampler(self.get_sampling_weights())
        self._rng, self._samples = None, None

    def _build_val_index(self) -> None:
        self.metadata_path = self.cfg.val_metadata_path
        self.parsed_df = build_val_parsed_index(self.cfg)

    # ------------------------------------------------------------------ #
    # MolecularDataset interface
    # ------------------------------------------------------------------ #
    @override
    def __getitem__(self, idx: int):
        if self.phase == "train":
            self._ensure_worker_rng()
            idx = next(self._samples)

        runtime_index = getattr(self, "_runtime_index", None)
        if runtime_index is not None:
            parsed_row = runtime_index[idx]
            example_id = parsed_row["example_id"]
        else:
            example_id = self.idx_to_id(idx)
            parsed_row = self.parsed_df.loc[example_id]

        try:
            example = self._load_cached_example(parsed_row["extra_info"]["pdb_id"])
        except FileNotFoundError:
            logger.warning(
                "Cached example for %s not found in %s/cached_examples in %s dataset, skipping...",
                parsed_row["extra_info"]["pdb_id"],
                self.cfg.pdb_path,
                self.phase,
            )
            if self.phase == "val":
                raise
            if self._metadata_index_size() == 0:
                raise
            return self.__getitem__((idx + 1) % self._metadata_index_size())

        example.update(parsed_row)
        example["phase"] = self.phase

        try:
            return self.featurizer(example)
        except Exception as exc:
            logger.error(
                "Error applying transforms to example %s in %s dataset: %s",
                example_id,
                self.phase,
                exc,
            )
            if self.phase == "val":
                raise
            if self._metadata_index_size() == 0:
                raise
            return self.__getitem__((idx + 1) % self._metadata_index_size())

    def _ensure_worker_rng(self) -> None:
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            self._samples = self._sampler.sample(self._rng)

    def get_sampling_weights(self) -> np.ndarray:
        runtime_index = getattr(self, "_runtime_index", None)
        if runtime_index is not None:
            return runtime_index.sampling_weights()
        return self.parsed_df.apply(lambda row: row["extra_info"]["sampling_weight"]).to_numpy()

    def _metadata_index_size(self) -> int:
        runtime_index = getattr(self, "_runtime_index", None)
        if runtime_index is not None:
            return len(runtime_index)
        return len(self.parsed_df)

    @override
    def __len__(self) -> int:
        if self.phase == "train":
            return self.cfg.samples_per_epoch
        return self._metadata_index_size()

    @override
    def __contains__(self, example_id: str) -> bool:
        runtime_index = getattr(self, "_runtime_index", None)
        if runtime_index is not None:
            return runtime_index.contains(example_id)
        return example_id in self.parsed_df.index

    @override
    def id_to_idx(self, example_id: str) -> int:
        runtime_index = getattr(self, "_runtime_index", None)
        if runtime_index is not None:
            return runtime_index.index_of(example_id)
        return self.parsed_df.index.get_loc(example_id)

    @override
    def idx_to_id(self, idx: int) -> str:
        runtime_index = getattr(self, "_runtime_index", None)
        if runtime_index is not None:
            return runtime_index.example_id(idx)
        return self.parsed_df.index[idx]

    def _load_cached_example(self, pdb_id: str) -> dict[str, torch.Tensor]:
        cached_example_path = f"{self.cfg.pdb_path}/cached_examples/{pdb_id}.pt"
        return torch.load(cached_example_path, map_location="cpu", weights_only=False)
