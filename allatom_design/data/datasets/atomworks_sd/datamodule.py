"""Lightning DataModule, collator and worker init for the SD dataset."""

import random

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from allatom_design.data.transform import sd_featurizer
from allatom_design.data.transform.pad import pad_to_max

from allatom_design.data.datasets.atomworks_sd.dataset import AtomworksSDDataset


class AtomworksSDDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.pdb_path = cfg.pdb_path
        self._train_set = AtomworksSDDataset(cfg, phase="train")
        self._val_set = AtomworksSDDataset(cfg, phase="val")

    def train_dataloader(self) -> DataLoader:
        num_workers = self.cfg.get("num_workers", 0)
        train_loader = DataLoader(
            dataset=self._train_set,
            batch_size=self.cfg.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=sd_collator,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        num_workers = self.cfg.get("num_workers", 0)
        val_loader = DataLoader(
            dataset=self._val_set,
            batch_size=self.cfg.batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=sd_collator,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
            worker_init_fn=worker_init_fn,
        )
        return val_loader


def sd_collator(data: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = data[0].keys()
    collated = {}
    for key in keys:
        values = [d[key] for d in data]
        if key not in ["example_id", *sd_featurizer.INFERENCE_ONLY_KEYS]:
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)
        collated[key] = values
    return collated


def worker_init_fn(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
