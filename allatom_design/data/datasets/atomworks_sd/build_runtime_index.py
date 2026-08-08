"""Build the final Arrow runtime index used by atomworks_sd training."""

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from allatom_design.data.datasets.atomworks_sd.dataset import (
    build_train_parsed_index,
    build_val_parsed_index,
)
from allatom_design.data.datasets.atomworks_sd.runtime_index import (
    RuntimeMetadataIndex,
    write_runtime_index,
)


@hydra.main(
    config_path="../../../configs/seq_denoiser",
    config_name="elix",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    data_cfg = cfg.data
    output_value = data_cfg.get("runtime_index_path")
    if not output_value:
        raise ValueError(
            "data.runtime_index_path must name the Arrow file to build"
        )
    output = Path(output_value).expanduser().resolve()
    overwrite = bool(
        cfg.get("runtime_index_builder", {}).get("overwrite", False)
    )

    if output.exists() and not overwrite:
        # An exact existing artifact is an idempotent success. Any stale or
        # malformed artifact fails with the reader's contract error.
        train_index = RuntimeMetadataIndex(output, cfg=data_cfg, phase="train")
        val_index = RuntimeMetadataIndex(output, cfg=data_cfg, phase="val")
        print(
            f"Runtime index already matches the resolved data contract: {output} "
            f"(train={len(train_index)}, val={len(val_index)})"
        )
        return

    np.random.seed(int(data_cfg.get("random_seed", cfg.train.seed)))
    train_records, _ = build_train_parsed_index(data_cfg)
    val_records = build_val_parsed_index(data_cfg)
    written = write_runtime_index(
        output,
        cfg=data_cfg,
        train_records=train_records,
        val_records=val_records,
        overwrite=overwrite,
    )
    print(
        f"Wrote runtime index: {written} "
        f"(train={len(train_records)}, val={len(val_records)})"
    )


if __name__ == "__main__":
    main()
