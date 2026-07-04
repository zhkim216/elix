from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from allatom_design.data.transform.preprocess import (
    preprocess_transform,
    preprocess_transform_designed_samples,
)


def preprocess_input(
    example: dict[str, Any],
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    sample_is_designed: bool = False,
) -> dict[str, Any]:
    """Preprocess an already-loaded eval example with the SD preprocess transforms."""
    preprocess_cfg = OmegaConf.to_container(preprocess_cfg, resolve=True)

    if sample_is_designed:
        pipeline = preprocess_transform_designed_samples(**preprocess_cfg)
    else:
        pipeline = preprocess_transform(**preprocess_cfg)

    return pipeline(example)
