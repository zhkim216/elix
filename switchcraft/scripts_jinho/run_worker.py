#!/usr/bin/env python3
"""Run one resumable SwitchCraft work item through the existing CLI."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from generation_artifacts import inspect_design


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--switchcraft-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outpath", type=Path, required=True)
    parser.add_argument("--design-root", type=Path, required=True)
    parser.add_argument("--num-designs", type=int, required=True)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument(
        "--design-indices",
        help="Comma-separated explicit design indices for a two-lane work item.",
    )
    parser.add_argument("--num-states", type=int, required=True)
    parser.add_argument(
        "--motifs",
        default="-",
        help="Comma-separated motif stems, or '-' for a motif-free task.",
    )
    return parser.parse_args()


def parse_design_indices(raw, num_designs):
    if raw is None:
        return None
    try:
        indices = [int(value) for value in raw.split(",") if value != ""]
    except ValueError as exc:
        raise ValueError("--design-indices must contain only integers") from exc
    if not indices:
        raise ValueError("--design-indices must not be empty")
    if len(indices) != len(set(indices)):
        raise ValueError("--design-indices must not contain duplicates")
    if any(index < 0 or index >= num_designs for index in indices):
        raise ValueError("--design-indices entries must be in [0, num-designs)")
    return indices


def main():
    args = parse_args()
    if args.num_designs <= 0:
        raise ValueError("--num-designs must be positive")
    if args.num_states <= 0:
        raise ValueError("--num-states must be positive")

    explicit_indices = parse_design_indices(args.design_indices, args.num_designs)
    if explicit_indices is None:
        if args.num_workers is None or args.worker_id is None:
            raise ValueError(
                "provide --design-indices or both --num-workers and --worker-id"
            )
        if args.num_workers <= 0:
            raise ValueError("--num-workers must be positive")
        if args.worker_id < 0 or args.worker_id >= args.num_workers:
            raise ValueError("--worker-id must be in [0, num-workers)")

    switchcraft_root = args.switchcraft_root.resolve()
    config = args.config.resolve()
    outpath = args.outpath.resolve()
    design_root = args.design_root.resolve()
    motifs = [] if args.motifs == "-" else args.motifs.split(",")
    assigned = (
        explicit_indices
        if explicit_indices is not None
        else list(range(args.worker_id, args.num_designs, args.num_workers))
    )
    completed = 0
    skipped = 0

    print(
        json.dumps(
            {
                "event": "worker_start",
                "worker_id": args.worker_id,
                "num_workers": args.num_workers,
                "assigned_designs": assigned,
                "design_root": str(design_root),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for design_index in assigned:
        design_dir = design_root / "design{}".format(design_index)
        before = inspect_design(design_dir, args.num_states, motifs)
        if before["valid"]:
            skipped += 1
            print(
                json.dumps(
                    {"event": "design_skip_complete", "design_index": design_index},
                    sort_keys=True,
                ),
                flush=True,
            )
            continue

        command = [
            sys.executable,
            "-u",
            str(switchcraft_root / "switchcraft.py"),
            "--config",
            str(config),
            "--num_designs",
            str(args.num_designs),
            "--num_workers",
            str(args.num_designs),
            "--worker_id",
            str(design_index),
            "--outpath",
            str(outpath),
        ]
        print(
            json.dumps(
                {
                    "event": "design_start",
                    "design_index": design_index,
                    "integrity_before": before,
                    "command": command,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        subprocess.run(command, cwd=str(switchcraft_root), check=True)
        after = inspect_design(design_dir, args.num_states, motifs)
        if not after["valid"]:
            raise RuntimeError(
                "Design {} remained incomplete: {}".format(design_index, after)
            )
        completed += 1
        print(
            json.dumps(
                {"event": "design_complete", "design_index": design_index},
                sort_keys=True,
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                "event": "worker_complete",
                "assigned": len(assigned),
                "completed_now": completed,
                "skipped_complete": skipped,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
