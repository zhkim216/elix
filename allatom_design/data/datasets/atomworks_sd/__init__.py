"""All-atom SD dataset package.

Public entry points:

- :class:`AtomworksSDDataModule` -- the Lightning DataModule used in training.
- :class:`AtomworksSDDataset` -- the underlying torch dataset.

The remaining modules (``metadata``, ``interface``, ``sampling``,
``selectors``) hold the index-building pipeline and are imported on demand.
"""

from allatom_design.data.datasets.atomworks_sd.datamodule import (
    AtomworksSDDataModule,
    sd_collator,
    worker_init_fn,
)
from allatom_design.data.datasets.atomworks_sd.dataset import AtomworksSDDataset
from allatom_design.data.datasets.atomworks_sd.interface import build_interface_df
from allatom_design.data.datasets.atomworks_sd.metadata import add_chain_counts_info
from allatom_design.data.datasets.atomworks_sd.sampling import (
    add_sampling_weights,
    validate_sampling_weights,
)

__all__ = [
    "AtomworksSDDataModule",
    "AtomworksSDDataset",
    "sd_collator",
    "worker_init_fn",
    "build_interface_df",
    "add_chain_counts_info",
    "add_sampling_weights",
    "validate_sampling_weights",
]
