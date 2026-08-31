from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from allatom_design.eval.benchmarks.denovoval.sequence_design_inputs import (
    prepare_inputs,
)


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/ligandmpnn/default.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage canonical denovoval CIFs as PDB sequence-design inputs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_inputs(
        OmegaConf.load(args.config),
        smoke=args.smoke,
        sample_ids=args.sample_id,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
