#!/usr/bin/env python3
"""Run RFD3 design on a single prepared input JSON (denovoval step2).

Naming-agnostic runner: the caller (sbatch array / local smoke) computes the input JSON
path and the output directory, and this invokes ``rfd3 design`` with the diffused-ligand
RASA-buried classifier-free-guidance overrides. ``--rfd3-cmd`` is shlex-split so callers
can pass an installed console script (``rfd3``) or a module invocation
(``python -m rfd3.cli``) for environments without the console script.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from constants import CFG_FEATURES, CFG_SCALE  # type: ignore[import-not-found]
else:
    from .constants import CFG_FEATURES, CFG_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True, help="Prepared RFD3 input JSON (dict of specs).")
    parser.add_argument("--out-dir", type=Path, required=True, help="RFD3 design output directory for this JSON.")
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--diffusion-batch-size", type=int, default=10)
    parser.add_argument("--n-batches", type=int, default=30)
    parser.add_argument(
        "--step-scale",
        type=float,
        default=1.5,
        help="RFD3 diffusion step scale (higher is less diverse; default: 1.5).",
    )
    parser.add_argument(
        "--gamma-0",
        type=float,
        default=0.6,
        help="RFD3 stochastic churn strength (higher is more diverse; default: 0.6).",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.003,
        help="RFD3 Gaussian noise multiplier (default: 1.003).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional global RFD3 seed for matched sampler comparisons.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=CFG_SCALE,
        help="Classifier-free guidance scale for RASA-buried conditioning.",
    )
    parser.add_argument(
        "--cfg-features",
        default=",".join(CFG_FEATURES),
        help="Comma-separated CFG feature names guided (default: RASA only).",
    )
    parser.add_argument(
        "--no-cfg",
        action="store_true",
        help="Disable classifier-free guidance (debug only; drops RASA burial guidance).",
    )
    parser.add_argument(
        "--rfd3-cmd",
        default="rfd3",
        help="Command that runs RFD3, shlex-split (e.g. 'rfd3' or 'python -m rfd3.cli').",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        *shlex.split(args.rfd3_cmd),
        "design",
        f"ckpt_path={args.ckpt_path}",
        f"inputs={args.input_json}",
        f"out_dir={args.out_dir}",
        f"diffusion_batch_size={args.diffusion_batch_size}",
        f"n_batches={args.n_batches}",
        "skip_existing=True",
        "prevalidate_inputs=True",
        f"inference_sampler.step_scale={args.step_scale}",
        f"inference_sampler.gamma_0={args.gamma_0}",
        f"inference_sampler.noise_scale={args.noise_scale}",
    ]
    if args.seed is not None:
        command.append(f"seed={args.seed}")
    if not args.no_cfg:
        features = [feature.strip() for feature in args.cfg_features.split(",") if feature.strip()]
        command += [
            "inference_sampler.use_classifier_free_guidance=True",
            f"inference_sampler.cfg_scale={args.cfg_scale}",
            f"inference_sampler.cfg_features=[{','.join(features)}]",
        ]
    return command


def main() -> None:
    args = parse_args()
    if not args.input_json.exists():
        raise FileNotFoundError(f"Missing RFD3 input JSON: {args.input_json}")
    specs = json.loads(args.input_json.read_text())
    if not specs:
        raise ValueError(f"RFD3 input JSON contains no specs: {args.input_json}")
    if args.diffusion_batch_size <= 0:
        raise ValueError("--diffusion-batch-size must be positive")
    if args.n_batches <= 0:
        raise ValueError("--n-batches must be positive")
    if args.step_scale <= 0:
        raise ValueError("--step-scale must be positive")
    if args.gamma_0 < 0:
        raise ValueError("--gamma-0 must be non-negative")
    if args.noise_scale <= 0:
        raise ValueError("--noise-scale must be positive")

    command = build_command(args)
    print(shlex.join(str(part) for part in command), flush=True)  # copy-paste-safe echo
    if args.dry_run:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
