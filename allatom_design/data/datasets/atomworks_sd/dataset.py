"""``AtomworksSDDataset`` -- the torch dataset that serves cached examples.

The class is intentionally thin: ``__init__`` orchestrates the index-building
pipeline (delegated to :mod:`metadata`, :mod:`interface` and :mod:`sampling`)
and the remaining methods implement the ``MolecularDataset`` interface.
"""

import logging
from pathlib import Path
from typing import Literal

import numpy as np
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

logger = logging.getLogger(__name__)


class AtomworksSDDataset(MolecularDataset):
    def __init__(self, cfg: DictConfig, phase: Literal["train", "val"]):
        super().__init__(name=f"sd::{phase}", transform=None)
        self.cfg = cfg
        self.phase = phase
        self.save_failed_examples_to_dir = cfg.save_failed_examples_to_dir
        self.val_cluster_ids: list = []

        self.featurizer = sd_featurizer.sd_featurizer(
            **cfg.featurizer_cfg,
            remove_keys=sd_featurizer.INFERENCE_ONLY_KEYS,
        )

        if self.phase == "train":
            self._build_train_index()
        else:
            self._build_val_index()

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    def _build_train_index(self) -> None:
        self.metadata_path = self.cfg.train_metadata_path
        metadata_df, self.val_cluster_ids = md.process_train_metadata_df(
            self.metadata_path,
            cfg=self.cfg,
            phase=self.phase,
        )
        dataset_name = self.cfg.get("dataset_name", Path(self.metadata_path).parent.name)
        monomer_df = md.build_monomer_chain_df(
            metadata_df,
            self.cfg,
            dataset_name,
            self.val_cluster_ids,
        )
        interface_df = md.build_train_interface_df(
            metadata_df,
            self.cfg,
            dataset_name,
            self.val_cluster_ids,
        )
        monomer_df, interface_df = add_sampling_weights(
            monomer_df=monomer_df,
            interface_df=interface_df,
            alphas_interface=self.cfg.sampling_weights["alphas_interface"],
            cluster_col="q_pn_unit_cluster_id",
            k_percentile=self.cfg.sampling_weights["k_percentile"],
            single_protein_context_weight=self.cfg.sampling_weights.get("single_protein_context_weight", 1.0),
            multi_protein_context_weight=self.cfg.sampling_weights.get("multi_protein_context_weight", 1.0),
            clustering_cfg=self.cfg.get("clustering", {}),
        )
        validate_sampling_weights(monomer_df, interface_df)
        self.parsed_df = md.parse_train_dfs(
            monomer_df,
            interface_df,
            self.cfg,
        )
        self._sampler = Sampler(self.get_sampling_weights())
        self._rng, self._samples = None, None

    def _build_val_index(self) -> None:
        self.metadata_path = self.cfg.val_metadata_path
        metadata_df = md.process_val_metadata_df(self.metadata_path)
        self.parsed_df = md.parse_val_df(metadata_df)

    # ------------------------------------------------------------------ #
    # MolecularDataset interface
    # ------------------------------------------------------------------ #
    @override
    def __getitem__(self, idx: int):
        if self.phase == "train":
            self._ensure_worker_rng()
            idx = next(self._samples)

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
            if len(self.parsed_df) == 0:
                raise
            return self.__getitem__((idx + 1) % len(self.parsed_df))

        example.update(parsed_row)
        example["phase"] = self.phase

        try:
            return self.featurizer(example)
        except Exception as exc:
            logger.error(
                "Error applying train-time transforms to example %s in %s dataset: %s",
                example_id,
                self.phase,
                exc,
            )
            if len(self.parsed_df) == 0:
                raise
            return self.__getitem__((idx + 1) % len(self.parsed_df))

    def _ensure_worker_rng(self) -> None:
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            self._samples = self._sampler.sample(self._rng)

    def get_sampling_weights(self) -> np.ndarray:
        return self.parsed_df.apply(lambda row: row["extra_info"]["sampling_weight"]).to_numpy()

    @override
    def __len__(self) -> int:
        if self.phase == "train":
            return self.cfg.samples_per_epoch
        return len(self.parsed_df)

    @override
    def __contains__(self, example_id: str) -> bool:
        return example_id in self.parsed_df.index

    @override
    def id_to_idx(self, example_id: str) -> int:
        return self.parsed_df.index.get_loc(example_id)

    @override
    def idx_to_id(self, idx: int) -> str:
        return self.parsed_df.index[idx]

    def _load_cached_example(self, pdb_id: str) -> dict[str, torch.Tensor]:
        cached_example_path = f"{self.cfg.pdb_path}/cached_examples/{pdb_id}.pt"
        return torch.load(cached_example_path, map_location="cpu", weights_only=False)
