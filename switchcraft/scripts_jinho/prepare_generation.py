#!/usr/bin/env python3
"""Freeze SwitchCraft generation configs, cases, and optional Slurm work items."""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/users/zhkim216/experiment_result_analysis/benchmarks/"
    "state_switching/denovoval"
)
DEFAULT_SWITCHCRAFT_ROOT = Path("/scratch/users/zhkim216/code/switchcraft")
DEFAULT_ENV_ROOT = Path("/scratch/users/zhkim216/envs/uv/switchcraft-py311-cu126")
DEFAULT_CONTAINER = Path("/scratch/users/zhkim216/containers/elix.sif")
PAPER_PATH = Path("/home/users/zhkim216/code/switchcraft/2026_switchcraft.pdf")

ALLOSTERY_DESIGNS_PER_CASE = 100
MOTIF_SWITCHING_DESIGNS = 1000
LIGAND_MODIFICATION_DESIGNS = 558
LIGAND_DISCRIMINATION_DESIGNS = 465
EXPECTED_TOTAL_DESIGNS = 26023
FINAL_DIFFUSION_SAMPLES_PER_STATE = 5
DEFAULT_DESIGNS_PER_WORKER_BY_NUM_STATES = {2: 20, 3: 15}

MOTIFS = (
    "1bcf",
    "1prw",
    "1qjg",
    "1ycr",
    "2kl8",
    "3ixt",
    "4jhw",
    "4zyp",
    "5ius",
    "5tpn",
    "5trv_long",
    "5trv_med",
    "5trv_short",
    "5wn9",
    "5yui",
    "6e6r_long",
    "6e6r_med",
    "6e6r_short",
    "6exz_long",
    "6exz_med",
    "6exz_short",
    "7mrx_128",
    "7mrx_85",
    "7mrx_60",
)

LIGANDS = (
    ("oqo", "ccd:OQO", "OQO"),
    ("fad", "ccd:FAD", "FAD"),
    ("zn", "ccd:ZN", "Zn2+"),
    ("mg", "ccd:MG", "Mg2+"),
    ("dna_GAATTC", "dna:GAATTC", "dsDNA GAATTC"),
)

CASE_FIELDS = (
    "case_index",
    "case_id",
    "family",
    "motif_key",
    "ligand_key",
    "num_designs",
    "num_states",
    "config_path",
    "outpath",
    "design_root",
    "config_sha256",
)

WORK_FIELDS = (
    "work_index",
    "case_id",
    "family",
    "worker_id",
    "num_workers",
    "num_designs",
    "num_states",
    "motif_key",
    "config_path",
    "outpath",
    "design_root",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--switchcraft-root", type=Path, default=DEFAULT_SWITCHCRAFT_ROOT
    )
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--container", type=Path, default=DEFAULT_CONTAINER)
    parser.add_argument(
        "--designs-per-worker",
        type=int,
        help=(
            "Override the scheduler-safe defaults for every case. Without this "
            "override, two-state cases use 20 designs and three-state cases use 15."
        ),
    )
    parser.add_argument(
        "--replace-work-plan",
        action="store_true",
        help="Replace an existing work plan when its worker sizing differs.",
    )
    parser.add_argument(
        "--replace-generation-plan",
        action="store_true",
        help=(
            "Replace frozen configs, cases, manifest, and work plan after an "
            "explicit scientific-contract change."
        ),
    )
    return parser.parse_args()


def canonical_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_commit(repo):
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_immutable(path, text):
    if path.exists():
        current = path.read_text()
        if current != text:
            raise RuntimeError(
                "Refusing to change frozen scientific artifact: {}".format(path)
            )
        return "unchanged"
    atomic_write(path, text)
    return "created"


def write_replaceable(path, text, allow_replace):
    existed = path.exists()
    if existed:
        current = path.read_text()
        if current == text:
            return "unchanged"
        if not allow_replace:
            raise RuntimeError(
                "Generated artifact differs from {}. Re-run with the appropriate "
                "explicit replacement flag."
                .format(path)
            )
    atomic_write(path, text)
    return "replaced" if existed else "created"


def allostery_config(direction, motif, ligand_state):
    if direction == "positive_allostery":
        motif_losses = [
            {"type": "AntiMotifLoss", "motif": 0, "state": 0},
            {"type": "MotifLoss", "motif": 0, "state": 1},
        ]
    elif direction == "negative_allostery":
        motif_losses = [
            {"type": "MotifLoss", "motif": 0, "state": 0},
            {"type": "AntiMotifLoss", "motif": 0, "state": 1},
        ]
    else:
        raise ValueError("Unknown allostery direction: {}".format(direction))
    return {
        "num_states": 2,
        "motifs": [motif],
        "states": [[], [ligand_state]],
        "losses": motif_losses + [{"type": "LigandContactLoss", "state": 1}],
    }


def motif_switching_config():
    return {
        "num_states": 2,
        "motifs": ["3ixt", "1ycr"],
        "states": [[], ["ccd:OQO"]],
        "losses": [
            {"type": "MotifLoss", "motif": 0, "state": 0},
            {"type": "AntiMotifLoss", "motif": 0, "state": 1},
            {"type": "AntiMotifLoss", "motif": 1, "state": 0},
            {"type": "MotifLoss", "motif": 1, "state": 1},
            {"type": "ConfChangeLoss", "state": [0, 1], "strength": 10},
            {"type": "LigandContactLoss", "state": 1},
        ],
    }


def ligand_modification_config():
    return {
        "num_states": 2,
        "motifs": [],
        "length": 50,
        "states": [["ccd:HEM"], ["ccd:HEM", "ccd:OXY"]],
        "losses": [
            {"type": "LigandContactLoss", "state": 0, "idx": 1},
            {"type": "LigandContactLoss", "state": 1, "idx": 1},
            {"type": "LigandContactLoss", "state": 1, "idx": 2},
            {"type": "ConfChangeLoss", "state": [0, 1], "strength": 10},
        ],
    }


def ligand_discrimination_config():
    return {
        "num_states": 3,
        "motifs": [],
        "length": 50,
        "states": [[], ["ccd:OQO"], ["ccd:CA"]],
        "losses": [
            {"type": "LigandContactLoss", "state": 1},
            {"type": "LigandContactLoss", "state": 2},
            {"type": "ConfChangeLoss", "state": [0, 1], "strength": 10},
            {"type": "ConfChangeLoss", "state": [0, 2], "strength": 10},
            {"type": "ConfChangeLoss", "state": [1, 2], "strength": 10},
        ],
    }


def add_case(cases, config_dir, output_root, family, case_suffix, config, count,
             allow_replace, ligand_key="-", motif_key="-"):
    case_id = "{}__{}".format(family, case_suffix)
    config_path = config_dir / family / (case_suffix + ".yaml")
    config_text = canonical_json(config)
    if allow_replace:
        write_replaceable(config_path, config_text, allow_replace=True)
    else:
        write_immutable(config_path, config_text)

    outpath = output_root / family
    motifs = config.get("motifs") or []
    if family in ("positive_allostery", "negative_allostery"):
        outpath = outpath / ligand_key
    design_root = outpath / "_".join(motifs) if motifs else outpath

    case = {
        "case_index": len(cases),
        "case_id": case_id,
        "family": family,
        "motif_key": motif_key,
        "ligand_key": ligand_key,
        "num_designs": count,
        "num_states": config["num_states"],
        "motifs": motifs,
        "config_path": str(config_path),
        "outpath": str(outpath),
        "design_root": str(design_root),
        "config_sha256": sha256_text(config_text),
    }
    cases.append(case)


def build_cases(output_root, allow_replace=False):
    config_dir = output_root / "manifests" / "configs"
    cases = []
    for family in ("positive_allostery", "negative_allostery"):
        for motif in MOTIFS:
            for ligand_key, ligand_state, _ in LIGANDS:
                suffix = "{}__{}".format(motif, ligand_key)
                add_case(
                    cases,
                    config_dir,
                    output_root,
                    family,
                    suffix,
                    allostery_config(family, motif, ligand_state),
                    ALLOSTERY_DESIGNS_PER_CASE,
                    allow_replace,
                    ligand_key=ligand_key,
                    motif_key=motif,
                )

    add_case(
        cases,
        config_dir,
        output_root,
        "motif_switching",
        "3ixt__1ycr__oqo",
        motif_switching_config(),
        MOTIF_SWITCHING_DESIGNS,
        allow_replace,
        ligand_key="oqo",
        motif_key="3ixt,1ycr",
    )
    add_case(
        cases,
        config_dir,
        output_root,
        "ligand_modification",
        "hem__oxy",
        ligand_modification_config(),
        LIGAND_MODIFICATION_DESIGNS,
        allow_replace,
        ligand_key="hem,oxy",
    )
    add_case(
        cases,
        config_dir,
        output_root,
        "ligand_discrimination",
        "apo__oqo__ca",
        ligand_discrimination_config(),
        LIGAND_DISCRIMINATION_DESIGNS,
        allow_replace,
        ligand_key="apo,oqo,ca",
    )
    return cases


def tsv_text(rows, fields):
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    return handle.getvalue()


def validate_runtime_paths(switchcraft_root, env_root, container):
    required = [
        switchcraft_root / "switchcraft.py",
        switchcraft_root / "boltz" / "ccd.pkl",
        switchcraft_root / "boltz" / "boltz1_conf.ckpt",
        env_root / "bin" / "python",
        container,
        PAPER_PATH,
    ]
    required.extend(switchcraft_root / "motifs" / (motif + ".pdb") for motif in MOTIFS)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required runtime paths:\n" + "\n".join(missing))


def preserved_created_at(manifest_path):
    if not manifest_path.exists():
        return utc_now()
    try:
        return json.loads(manifest_path.read_text())["created_at"]
    except (KeyError, ValueError, TypeError):
        raise RuntimeError("Cannot preserve created_at from {}".format(manifest_path))


def build_work_items(cases, designs_per_worker_override=None):
    if designs_per_worker_override is not None and designs_per_worker_override <= 0:
        raise ValueError("--designs-per-worker must be positive")
    rows = []
    max_assigned = 0
    worker_sizes = {}
    for case in cases:
        designs_per_worker = designs_per_worker_override
        if designs_per_worker is None:
            try:
                designs_per_worker = DEFAULT_DESIGNS_PER_WORKER_BY_NUM_STATES[
                    case["num_states"]
                ]
            except KeyError:
                raise ValueError(
                    "No default worker size for {} states in {}".format(
                        case["num_states"], case["case_id"]
                    )
                )
        worker_sizes[str(case["num_states"])] = designs_per_worker
        num_workers = int(math.ceil(case["num_designs"] / designs_per_worker))
        for worker_id in range(num_workers):
            assigned = len(range(worker_id, case["num_designs"], num_workers))
            max_assigned = max(max_assigned, assigned)
            rows.append(
                {
                    "work_index": len(rows),
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "worker_id": worker_id,
                    "num_workers": num_workers,
                    "num_designs": case["num_designs"],
                    "num_states": case["num_states"],
                    "motif_key": case["motif_key"],
                    "config_path": case["config_path"],
                    "outpath": case["outpath"],
                    "design_root": case["design_root"],
                }
            )
    return rows, max_assigned, worker_sizes


def main():
    args = parse_args()
    output_root = args.output_root.resolve()
    switchcraft_root = args.switchcraft_root.resolve()
    env_root = args.env_root.resolve()
    container = args.container.resolve()
    validate_runtime_paths(switchcraft_root, env_root, container)

    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(
        output_root, allow_replace=args.replace_generation_plan
    )

    if len(cases) != 243:
        raise AssertionError("Expected 243 cases, found {}".format(len(cases)))
    total_designs = sum(case["num_designs"] for case in cases)
    if total_designs != EXPECTED_TOTAL_DESIGNS:
        raise AssertionError(
            "Expected {} designs, found {}".format(
                EXPECTED_TOTAL_DESIGNS, total_designs
            )
        )
    expected_design_files = sum(
        case["num_designs"]
        * (
            case["num_states"]
            * (1 + 2 * FINAL_DIFFUSION_SAMPLES_PER_STATE)
            + len(case["motifs"])
        )
        for case in cases
    )

    cases_path = manifests_dir / "cases.tsv"
    cases_text = tsv_text(cases, CASE_FIELDS)
    if args.replace_generation_plan:
        write_replaceable(cases_path, cases_text, allow_replace=True)
    else:
        write_immutable(cases_path, cases_text)

    manifest_path = manifests_dir / "manifest.json"
    manifest = {
        "schema_version": 2,
        "created_at": preserved_created_at(manifest_path),
        "output_root": str(output_root),
        "case_count": len(cases),
        "total_designs": total_designs,
        "allostery_contract": {
            "motif_count": len(MOTIFS),
            "ligand_count": len(LIGANDS),
            "direction_count": 2,
            "case_count": 240,
            "designs_per_case": ALLOSTERY_DESIGNS_PER_CASE,
            "total_designs": 24000,
        },
        "motifs": list(MOTIFS),
        "ligands": [
            {"key": key, "state_entry": state, "paper_label": label}
            for key, state, label in LIGANDS
        ],
        "scientific_sources": {
            "paper_path": str(PAPER_PATH),
            "paper_sha256": sha256_file(PAPER_PATH),
            "motif_specs_root": str(switchcraft_root / "motifs"),
        },
        "runtime": {
            "switchcraft_root": str(switchcraft_root),
            "switchcraft_git_commit": git_commit(switchcraft_root),
            "environment_root": str(env_root),
            "container_path": str(container),
            "final_diffusion_samples_per_state": FINAL_DIFFUSION_SAMPLES_PER_STATE,
            "settings_policy": (
                "Paper task specifications with SwitchCraft runtime defaults; "
                "only requested trajectory counts changed"
            ),
        },
        "output_projection": {
            "expected_design_files": expected_design_files,
            "expected_design_directories": total_designs,
            "expected_design_output_inodes": expected_design_files + total_designs,
        },
        "cases_tsv": str(cases_path),
        "cases_tsv_sha256": sha256_text(cases_text),
        "cases": cases,
    }
    if args.replace_generation_plan:
        write_replaceable(
            manifest_path, canonical_json(manifest), allow_replace=True
        )
    else:
        write_immutable(manifest_path, canonical_json(manifest))

    summary = {
        "manifest": str(manifest_path),
        "cases": len(cases),
        "total_designs": total_designs,
        "work_plan": None,
    }
    work_rows, max_assigned, worker_sizes = build_work_items(
        cases, args.designs_per_worker
    )
    work_path = manifests_dir / "work_items.tsv"
    work_text = tsv_text(work_rows, WORK_FIELDS)
    allow_work_replace = args.replace_work_plan or args.replace_generation_plan
    work_status = write_replaceable(
        work_path, work_text, allow_replace=allow_work_replace
    )
    work_plan_path = manifests_dir / "work_plan.json"
    work_plan_created_at = (
        preserved_created_at(work_plan_path)
        if work_status == "unchanged" and work_plan_path.exists()
        else utc_now()
    )
    work_plan = {
        "schema_version": 2,
        "created_at": work_plan_created_at,
        "designs_per_worker_override": args.designs_per_worker,
        "designs_per_worker_by_num_states": worker_sizes,
        "max_designs_assigned_to_worker": max_assigned,
        "work_item_count": len(work_rows),
        "total_designs": total_designs,
        "walltime": "04:00:00",
        "array_chunk_size": 1000,
        "submission_policy": [
            {
                "global_work_range": "0-999",
                "partition": "owners",
                "max_concurrent_tasks": None,
            },
            {
                "global_work_range": "1000-1308",
                "partition": "possu",
                "max_concurrent_tasks": 8,
            },
        ],
        "serial_array_dependencies": False,
        "expected_array_submissions": int(math.ceil(len(work_rows) / 1000)),
        "expected_slurm_log_files": 2 * len(work_rows),
        "work_items_tsv": str(work_path),
        "work_items_tsv_sha256": sha256_text(work_text),
    }
    work_plan_status = write_replaceable(
        work_plan_path,
        canonical_json(work_plan),
        allow_replace=allow_work_replace,
    )
    summary["work_plan"] = {
        "path": str(work_plan_path),
        "work_items": len(work_rows),
        "designs_per_worker_override": args.designs_per_worker,
        "designs_per_worker_by_num_states": worker_sizes,
        "max_designs_assigned_to_worker": max_assigned,
        "work_items_status": work_status,
        "work_plan_status": work_plan_status,
    }

    print(canonical_json(summary), end="")


if __name__ == "__main__":
    main()
