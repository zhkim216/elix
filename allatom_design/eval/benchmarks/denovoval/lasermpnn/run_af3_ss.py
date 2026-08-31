"""LASErMPNN entrypoint for the shared denovoval AF3 single-sequence runner."""

from pathlib import Path

from allatom_design.eval.benchmarks.denovoval.ligandmpnn.run_af3_ss import main


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/lasermpnn/default.yaml"
)


if __name__ == "__main__":
    main(DEFAULT_CONFIG)
