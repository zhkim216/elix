from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from allatom_design.data.transform.sd_featurizer import featurizer_af3_prediction
from allatom_design.eval.utils.input_preprocessing import preprocess_input
from allatom_design.utils.sample_io_utils import load_example_with_parse


def prepare_af3_prediction(
    pdb_path: str | None = None,
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return sequence-design model features for an AF3 prediction structure."""
    example = load_example_with_parse(pdb_path, cif_parse_cfg)

    example = preprocess_input(
        example=example,
        preprocess_cfg=preprocess_cfg,
        sample_is_designed=True,
    )

    featurizer_cfg = OmegaConf.to_container(featurizer_cfg, resolve=True)
    featurizer = featurizer_af3_prediction(**featurizer_cfg)
    return featurizer(example)
