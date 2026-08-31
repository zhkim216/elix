#!/usr/bin/env python3
"""Prepare, submit, finalize, and inspect resumable two-lane SwitchCraft batches."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

from generation_artifacts import inspect_design


OUTPUT_ROOT = Path(
    "/scratch/users/zhkim216/experiment_result_analysis/benchmarks/"
    "state_switching/denovoval"
)
DEFAULT_MANIFEST = OUTPUT_ROOT / "manifests" / "manifest.json"
DEFAULT_RUN_ROOT = OUTPUT_ROOT / "manifests" / "two_lane_1h2h_v1"
SCRIPT_ROOT = Path(__file__).resolve().parent

SLOW_TASK_LIMIT = 450
FAST_TASK_LIMIT = 550
SLOW_DESIGNS_PER_TASK = 2
FAST_TWO_STATE_DESIGNS_PER_TASK = 10
FAST_THREE_STATE_DESIGNS_PER_TASK = 5
SLOW_DESIGN_FRACTION = 0.145
SLOW_CONCURRENCY = 4
FAST_CONCURRENCY = 10
MAX_ATTEMPTS = 5
GPU_QOS_SUBMIT_LIMIT = 100
WAVE_BASE_TASKS_PER_LANE = 50

TASK_FIELDS = (
    "task_id",
    "lane",
    "case_id",
    "family",
    "num_states",
    "design_indices",
    "motifs",
    "config_path",
    "outpath",
    "design_root",
    "num_designs",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def atomic_json(path: Path, data: object) -> None:
    atomic_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_tsv(path: Path, rows: list[dict[str, object]], fields=TASK_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row[field] for field in fields})
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@contextlib.contextmanager
def run_lock(run_root: Path):
    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / ".controller.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def load_run(run_root: Path) -> dict:
    path = run_root / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"Run is not prepared: {path}")
    return json.loads(path.read_text())


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if "cases" not in manifest or "total_designs" not in manifest:
        raise ValueError(f"Invalid SwitchCraft manifest: {path}")
    return manifest


def motifs_for_case(case: dict) -> list[str]:
    return list(case.get("motifs") or [])


def target_pairs(manifest: dict):
    for case in manifest["cases"]:
        for design_index in range(int(case["num_designs"])):
            yield case, design_index


def inspect_case_design(case: dict, design_index: int) -> dict[str, object]:
    design_dir = Path(case["design_root"]) / f"design{design_index}"
    return inspect_design(
        design_dir, int(case["num_states"]), motifs_for_case(case)
    )


def prepare_run(manifest_path: Path, run_root: Path) -> dict:
    manifest_path = manifest_path.resolve()
    run_root = run_root.resolve()
    manifest = load_manifest(manifest_path)
    run_path = run_root / "run.json"
    manifest_sha = sha256_file(manifest_path)

    if run_path.exists():
        existing = json.loads(run_path.read_text())
        if existing["manifest_sha256"] != manifest_sha:
            raise RuntimeError("Prepared run points to a different scientific manifest")
        if not (run_root / "batches" / "batch_000" / "batch.json").exists():
            create_batch(run_root, 0)
        return existing

    run_root.mkdir(parents=True, exist_ok=True)
    complete_rows: list[dict[str, object]] = []
    invalid_details: list[dict[str, object]] = []
    valid_count = 0
    total = int(manifest["total_designs"])
    for position, (case, design_index) in enumerate(target_pairs(manifest), start=1):
        report = inspect_case_design(case, design_index)
        if report["valid"]:
            valid_count += 1
            complete_rows.append(
                {"case_id": case["case_id"], "design_index": design_index}
            )
        elif len(invalid_details) < 200 and (
            report["invalid_files"] or Path(case["design_root"], f"design{design_index}").exists()
        ):
            invalid_details.append(
                {
                    "case_id": case["case_id"],
                    "design_index": design_index,
                    "report": report,
                }
            )
        if position % 1000 == 0 or position == total:
            print(
                json.dumps(
                    {
                        "event": "initial_scan_progress",
                        "scanned": position,
                        "total": total,
                        "valid": valid_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    write_tsv(
        run_root / "initial_complete.tsv",
        complete_rows,
        fields=("case_id", "design_index"),
    )
    atomic_json(
        run_root / "initial_validation.json",
        {
            "created_at": utc_now(),
            "target_designs": total,
            "initially_complete": valid_count,
            "initially_incomplete": total - valid_count,
            "invalid_or_partial_examples": invalid_details,
            "max_examples": 200,
        },
    )
    run = {
        "schema_version": 1,
        "created_at": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "run_root": str(run_root),
        "initial_complete_tsv": str(run_root / "initial_complete.tsv"),
        "settings": {
            "slow": {
                "gpu_skus": [
                    "A40",
                    "A100_SXM4",
                    "A100_PCIE",
                    "H100_SXM5",
                    "H200_SXM5",
                ],
                "walltime": "01:00:00",
                "tasks_per_batch": SLOW_TASK_LIMIT,
                "designs_per_two_state_task": SLOW_DESIGNS_PER_TASK,
                "concurrency": SLOW_CONCURRENCY,
                "cpus": 2,
                "memory": "16G",
            },
            "fast": {
                "gpu_skus": [
                    "A40",
                    "A100_SXM4",
                    "A100_PCIE",
                    "H100_SXM5",
                    "H200_SXM5",
                ],
                "walltime": "02:00:00",
                "tasks_per_batch": FAST_TASK_LIMIT,
                "designs_per_two_state_task": FAST_TWO_STATE_DESIGNS_PER_TASK,
                "designs_per_three_state_task": FAST_THREE_STATE_DESIGNS_PER_TASK,
                "concurrency": FAST_CONCURRENCY,
                "cpus": 2,
                "memory": "32G",
            },
            "max_attempts": MAX_ATTEMPTS,
        },
    }
    atomic_json(run_path, run)
    create_batch(run_root, 0)
    return run


def initial_complete_pairs(run_root: Path) -> set[tuple[str, int]]:
    return {
        (row["case_id"], int(row["design_index"]))
        for row in read_tsv(run_root / "initial_complete.tsv")
    }


def design_indices(row: dict[str, str]) -> list[int]:
    return [int(value) for value in row["design_indices"].split(",") if value]


def batch_dirs(run_root: Path) -> list[Path]:
    return sorted((run_root / "batches").glob("batch_[0-9][0-9][0-9]"))


def assigned_pairs(run_root: Path) -> set[tuple[str, int]]:
    assigned: set[tuple[str, int]] = set()
    for batch_dir in batch_dirs(run_root):
        for lane in ("slow", "fast"):
            path = batch_dir / f"{lane}.tsv"
            if not path.exists():
                continue
            for row in read_tsv(path):
                for index in design_indices(row):
                    pair = (row["case_id"], index)
                    if pair in assigned:
                        raise RuntimeError(f"Design assigned more than once: {pair}")
                    assigned.add(pair)
    return assigned


def make_task_row(
    task_id: int, lane: str, case: dict, indices: list[int]
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "lane": lane,
        "case_id": case["case_id"],
        "family": case["family"],
        "num_states": int(case["num_states"]),
        "design_indices": ",".join(str(index) for index in indices),
        "motifs": ",".join(motifs_for_case(case)) or "-",
        "config_path": case["config_path"],
        "outpath": case["outpath"],
        "design_root": case["design_root"],
        "num_designs": int(case["num_designs"]),
    }


def take_round_robin(
    queues: dict[str, deque[int]],
    cases: list[dict],
    lane: str,
    group_size: int,
    task_limit: int,
    design_budget: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    taken = 0
    while len(rows) < task_limit:
        progressed = False
        for case in cases:
            queue = queues.get(case["case_id"])
            if not queue or len(rows) >= task_limit:
                continue
            remaining_budget = None if design_budget is None else design_budget - taken
            if remaining_budget is not None and remaining_budget <= 0:
                return rows
            count = min(group_size, len(queue))
            if remaining_budget is not None:
                count = min(count, remaining_budget)
            if count <= 0:
                return rows
            indices = [queue.popleft() for _ in range(count)]
            rows.append(make_task_row(len(rows), lane, case, indices))
            taken += count
            progressed = True
        if not progressed:
            break
    return rows


def create_batch(run_root: Path, batch_index: int) -> dict | None:
    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    batch_json = batch_dir / "batch.json"
    if batch_json.exists():
        return json.loads(batch_json.read_text())

    run = load_run(run_root)
    manifest = load_manifest(Path(run["manifest"]))
    complete = initial_complete_pairs(run_root)
    assigned = assigned_pairs(run_root)
    overlap = complete & assigned
    if overlap:
        raise RuntimeError(f"Initially complete designs were assigned: {sorted(overlap)[:5]}")

    cases = list(manifest["cases"])
    pending_by_case: dict[str, deque[int]] = {}
    case_map = {case["case_id"]: case for case in cases}
    for case in cases:
        pending = [
            index
            for index in range(int(case["num_designs"]))
            if (case["case_id"], index) not in complete
            and (case["case_id"], index) not in assigned
        ]
        pending_by_case[case["case_id"]] = deque(pending)

    pending_total = sum(len(queue) for queue in pending_by_case.values())
    if pending_total == 0:
        return None

    two_state_cases = [case for case in cases if int(case["num_states"]) == 2]
    three_state_cases = [case for case in cases if int(case["num_states"]) == 3]
    two_state_remaining = sum(
        len(pending_by_case[case["case_id"]]) for case in two_state_cases
    )
    slow_budget = min(
        SLOW_TASK_LIMIT * SLOW_DESIGNS_PER_TASK,
        int(two_state_remaining * SLOW_DESIGN_FRACTION),
    )
    slow_budget -= slow_budget % SLOW_DESIGNS_PER_TASK
    if two_state_remaining >= 2 and slow_budget == 0:
        slow_budget = 2

    slow_rows = take_round_robin(
        pending_by_case,
        two_state_cases,
        "slow",
        SLOW_DESIGNS_PER_TASK,
        SLOW_TASK_LIMIT,
        design_budget=slow_budget,
    )
    fast_three = take_round_robin(
        pending_by_case,
        three_state_cases,
        "fast",
        FAST_THREE_STATE_DESIGNS_PER_TASK,
        FAST_TASK_LIMIT,
    )
    fast_remaining = FAST_TASK_LIMIT - len(fast_three)
    fast_two = take_round_robin(
        pending_by_case,
        two_state_cases,
        "fast",
        FAST_TWO_STATE_DESIGNS_PER_TASK,
        fast_remaining,
    )
    fast_rows = fast_three + fast_two
    for task_id, row in enumerate(fast_rows):
        row["task_id"] = task_id

    if not slow_rows and not fast_rows:
        raise RuntimeError("Pending designs exist but no batch tasks were created")

    batch_dir.mkdir(parents=True, exist_ok=False)
    write_tsv(batch_dir / "slow.tsv", slow_rows)
    write_tsv(batch_dir / "fast.tsv", fast_rows)
    batch_pairs: set[tuple[str, int]] = set()
    family_counts: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    for row in slow_rows + fast_rows:
        indices = [int(value) for value in str(row["design_indices"]).split(",")]
        family_counts[str(row["family"])] = family_counts.get(str(row["family"]), 0) + len(indices)
        state_key = str(row["num_states"])
        state_counts[state_key] = state_counts.get(state_key, 0) + len(indices)
        for index in indices:
            pair = (str(row["case_id"]), index)
            if pair in batch_pairs or pair in complete or pair in assigned:
                raise RuntimeError(f"Invalid duplicate batch assignment: {pair}")
            if pair[0] not in case_map:
                raise RuntimeError(f"Unknown case in batch: {pair[0]}")
            batch_pairs.add(pair)

    batch = {
        "schema_version": 1,
        "created_at": utc_now(),
        "batch_index": batch_index,
        "pending_before_assignment": pending_total,
        "slow_tasks": len(slow_rows),
        "fast_tasks": len(fast_rows),
        "total_tasks": len(slow_rows) + len(fast_rows),
        "assigned_designs": len(batch_pairs),
        "family_design_counts": family_counts,
        "state_design_counts": state_counts,
        "slow_manifest": str(batch_dir / "slow.tsv"),
        "fast_manifest": str(batch_dir / "fast.tsv"),
        "slow_manifest_sha256": sha256_file(batch_dir / "slow.tsv"),
        "fast_manifest_sha256": sha256_file(batch_dir / "fast.tsv"),
    }
    atomic_json(batch_json, batch)
    return batch


def compress_ids(task_ids: list[int]) -> str:
    values = sorted(set(task_ids))
    if not values:
        raise ValueError("Cannot submit an empty task list")
    parts: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    parts.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(parts)


def run_token(run_root: Path) -> str:
    return hashlib.sha256(str(run_root.resolve()).encode()).hexdigest()[:7]


def parse_job_id(output: str) -> str:
    job_id = output.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job ID from {output!r}")
    return job_id


def lookup_job(name: str, created_at: str) -> str | None:
    user = os.environ.get("USER", "zhkim216")
    active = subprocess.run(
        ["squeue", "-h", "-u", user, "-n", name, "-o", "%A"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    roots = {line.strip() for line in active.splitlines() if line.strip().isdigit()}
    if not roots:
        start_date = created_at[:10]
        result = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-S",
                start_date,
                "--name",
                name,
                "--format=JobID,JobName",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        for line in result.splitlines():
            fields = line.split("|")
            if len(fields) < 2 or fields[1] != name:
                continue
            match = re.match(r"^(\d+)", fields[0])
            if match:
                roots.add(match.group(1))
    if len(roots) > 1:
        raise RuntimeError(f"Multiple Slurm jobs have unique name {name}: {sorted(roots)}")
    return next(iter(roots)) if roots else None


def submit_or_recover(
    name: str, command: list[str], created_at: str, dry_run: bool
) -> str:
    existing = lookup_job(name, created_at)
    if existing:
        print(f"recovered_job name={name} job_id={existing}")
        return existing
    if dry_run:
        print("DRY_RUN " + " ".join(command))
        return f"DRYRUN_{name}"
    output = subprocess.check_output(command, text=True)
    job_id = parse_job_id(output)
    print(f"submitted_job name={name} job_id={job_id}")
    return job_id


def submission_path(batch_dir: Path) -> Path:
    return batch_dir / "submission.json"


def wave_plan_path(batch_dir: Path) -> Path:
    return batch_dir / "waves.json"


def wave_submission_path(batch_dir: Path) -> Path:
    return batch_dir / "wave_submission.json"


def ensure_wave_plan(run_root: Path, batch_index: int) -> dict:
    """Freeze submission windows that obey the gpu partition's 100-job QOS cap."""

    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    batch = load_json(batch_dir / "batch.json")
    if batch is None:
        raise FileNotFoundError(f"Missing batch: {batch_dir}")
    path = wave_plan_path(batch_dir)
    manifest_hashes = {
        lane: sha256_file(batch_dir / f"{lane}.tsv") for lane in ("slow", "fast")
    }
    if path.exists():
        plan = load_json(path)
        if plan["lane_manifest_sha256"] != manifest_hashes:
            raise RuntimeError("Frozen wave plan points to changed lane manifests")
        return plan

    remaining = {
        lane: deque(int(row["task_id"]) for row in read_tsv(batch_dir / f"{lane}.tsv"))
        for lane in ("slow", "fast")
    }
    waves: list[dict[str, object]] = []
    while remaining["slow"] or remaining["fast"]:
        selected: dict[str, list[int]] = {"slow": [], "fast": []}
        for lane in ("slow", "fast"):
            count = min(WAVE_BASE_TASKS_PER_LANE, len(remaining[lane]))
            selected[lane].extend(remaining[lane].popleft() for _ in range(count))
        capacity = GPU_QOS_SUBMIT_LIMIT - sum(map(len, selected.values()))
        while capacity > 0 and (remaining["slow"] or remaining["fast"]):
            progressed = False
            for lane in ("slow", "fast"):
                if capacity == 0 or not remaining[lane]:
                    continue
                selected[lane].append(remaining[lane].popleft())
                capacity -= 1
                progressed = True
            if not progressed:
                break
        waves.append(
            {
                "wave_index": len(waves),
                "task_count": sum(map(len, selected.values())),
                "lanes": selected,
            }
        )

    expected = {
        lane: {int(row["task_id"]) for row in read_tsv(batch_dir / f"{lane}.tsv")}
        for lane in ("slow", "fast")
    }
    observed: dict[str, list[int]] = {"slow": [], "fast": []}
    for wave in waves:
        if int(wave["task_count"]) > GPU_QOS_SUBMIT_LIMIT:
            raise RuntimeError(f"Wave exceeds GPU QOS limit: {wave}")
        for lane in ("slow", "fast"):
            observed[lane].extend(wave["lanes"][lane])
    for lane in ("slow", "fast"):
        if len(observed[lane]) != len(set(observed[lane])):
            raise RuntimeError(f"Duplicate {lane} task in wave plan")
        if set(observed[lane]) != expected[lane]:
            raise RuntimeError(f"Wave plan does not cover the frozen {lane} manifest")

    plan = {
        "schema_version": 1,
        "created_at": utc_now(),
        "batch_index": batch_index,
        "gpu_qos_submit_limit": GPU_QOS_SUBMIT_LIMIT,
        "lane_manifest_sha256": manifest_hashes,
        "wave_count": len(waves),
        "waves": waves,
    }
    atomic_json(path, plan)
    return plan


def submit_wave_attempt(
    run_root: Path,
    batch_index: int,
    wave_index: int,
    attempt: int,
    lane_task_ids: dict[str, list[int]] | None = None,
    dry_run: bool = False,
) -> dict:
    run = load_run(run_root)
    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    plan = ensure_wave_plan(run_root, batch_index)
    if wave_index < 0 or wave_index >= len(plan["waves"]):
        raise ValueError(f"Wave index {wave_index} is outside the frozen wave plan")
    planned = plan["waves"][wave_index]["lanes"]
    selected = {
        lane: list(planned[lane] if lane_task_ids is None else lane_task_ids.get(lane, []))
        for lane in ("slow", "fast")
    }
    if sum(map(len, selected.values())) > GPU_QOS_SUBMIT_LIMIT:
        raise ValueError("Selected wave exceeds the gpu QOS 100-task submit limit")
    for lane in ("slow", "fast"):
        if not set(selected[lane]) <= set(planned[lane]):
            raise ValueError(f"Retry selected tasks outside wave {wave_index} {lane}")

    submissions = load_json(
        wave_submission_path(batch_dir),
        {"schema_version": 1, "batch_index": batch_index, "waves": {}},
    )
    wave_record = submissions["waves"].setdefault(
        str(wave_index), {"planned_task_ids": planned, "attempts": {}}
    )
    if wave_record["planned_task_ids"] != planned:
        raise RuntimeError(f"Frozen task selection changed for wave {wave_index}")
    attempt_record = wave_record["attempts"].setdefault(
        str(attempt), {"created_at": utc_now(), "lanes": {}}
    )
    token = run_token(run_root)
    job_ids: list[str] = []

    for lane in ("slow", "fast"):
        task_ids = selected[lane]
        if not task_ids:
            continue
        previous = attempt_record["lanes"].get(lane)
        if previous and previous["task_ids"] != task_ids:
            raise RuntimeError(
                f"Wave {wave_index} attempt {attempt} {lane} task set changed"
            )
        name = f"sc{token}_{lane[0]}b{batch_index:03d}w{wave_index:02d}a{attempt}"
        concurrency = min(
            len(task_ids), SLOW_CONCURRENCY if lane == "slow" else FAST_CONCURRENCY
        )
        array_spec = f"{compress_ids(task_ids)}%{concurrency}"
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={name}",
            (
                f"--comment=switchcraft:{token}:batch{batch_index:03d}:"
                f"wave{wave_index:02d}:{lane}:attempt{attempt}"
            ),
            f"--array={array_spec}",
            (
                "--export=ALL,"
                f"LANE_MANIFEST={batch_dir / f'{lane}.tsv'},"
                f"RUN_ROOT={run_root}"
            ),
            str(SCRIPT_ROOT / f"run_generation_{lane}_array.sbatch"),
        ]
        if previous:
            job_id = previous["job_id"]
        else:
            job_id = submit_or_recover(name, command, run["created_at"], dry_run)
            if not dry_run:
                attempt_record["lanes"][lane] = {
                    "job_id": job_id,
                    "job_name": name,
                    "array_spec": array_spec,
                    "task_ids": task_ids,
                    "submitted_at": utc_now(),
                }
                atomic_json(wave_submission_path(batch_dir), submissions)
        job_ids.append(job_id)

    if not job_ids:
        raise RuntimeError("No wave tasks were selected for submission")
    finalizer_name = (
        f"sc{token}_zb{batch_index:03d}w{wave_index:02d}a{attempt}"
    )
    command = [
        "sbatch",
        "--parsable",
        f"--job-name={finalizer_name}",
        (
            f"--comment=switchcraft:{token}:batch{batch_index:03d}:"
            f"wave{wave_index:02d}:finalizer:attempt{attempt}"
        ),
        f"--dependency=afterany:{':'.join(job_ids)}",
        (
            "--export=ALL,"
            f"RUN_ROOT={run_root},BATCH_INDEX={batch_index},"
            f"WAVE_INDEX={wave_index},ATTEMPT={attempt}"
        ),
        str(SCRIPT_ROOT / "finalize_generation_wave.sbatch"),
    ]
    if "finalizer" not in attempt_record:
        finalizer_id = submit_or_recover(
            finalizer_name, command, run["created_at"], dry_run
        )
        if not dry_run:
            attempt_record["finalizer"] = {
                "job_id": finalizer_id,
                "job_name": finalizer_name,
                "dependency_job_ids": job_ids,
                "submitted_at": utc_now(),
            }
            atomic_json(wave_submission_path(batch_dir), submissions)
    return submissions


def submit_attempt(
    run_root: Path,
    batch_index: int,
    attempt: int,
    lane_task_ids: dict[str, list[int]] | None = None,
    dry_run: bool = False,
) -> dict:
    run = load_run(run_root)
    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    batch = load_json(batch_dir / "batch.json")
    if batch is None:
        raise FileNotFoundError(f"Missing batch: {batch_dir}")
    submissions = load_json(
        submission_path(batch_dir),
        {"schema_version": 1, "batch_index": batch_index, "attempts": {}},
    )
    attempt_key = str(attempt)
    attempt_record = submissions["attempts"].setdefault(
        attempt_key, {"created_at": utc_now(), "lanes": {}}
    )
    token = run_token(run_root)
    job_ids: list[str] = []

    for lane in ("slow", "fast"):
        rows = read_tsv(batch_dir / f"{lane}.tsv")
        all_ids = [int(row["task_id"]) for row in rows]
        selected = all_ids if lane_task_ids is None else lane_task_ids.get(lane, [])
        if not selected:
            continue
        if not set(selected) <= set(all_ids):
            raise ValueError(f"Unknown {lane} task IDs: {sorted(set(selected) - set(all_ids))}")
        previous = attempt_record["lanes"].get(lane)
        if previous and previous["task_ids"] != selected:
            raise RuntimeError(f"Attempt {attempt} {lane} task set changed")
        name = f"sc{token}_{lane[0]}b{batch_index:03d}a{attempt}"
        concurrency = SLOW_CONCURRENCY if lane == "slow" else FAST_CONCURRENCY
        array_spec = f"{compress_ids(selected)}%{concurrency}"
        sbatch_file = SCRIPT_ROOT / f"run_generation_{lane}_array.sbatch"
        command = [
            "sbatch",
            "--parsable",
            f"--job-name={name}",
            f"--comment=switchcraft:{token}:batch{batch_index:03d}:{lane}:attempt{attempt}",
            f"--array={array_spec}",
            (
                "--export=ALL,"
                f"LANE_MANIFEST={batch_dir / f'{lane}.tsv'},"
                f"RUN_ROOT={run_root}"
            ),
            str(sbatch_file),
        ]
        if previous:
            job_id = previous["job_id"]
        else:
            job_id = submit_or_recover(name, command, run["created_at"], dry_run)
            if not dry_run:
                attempt_record["lanes"][lane] = {
                    "job_id": job_id,
                    "job_name": name,
                    "array_spec": array_spec,
                    "task_ids": selected,
                    "submitted_at": utc_now(),
                }
                atomic_json(submission_path(batch_dir), submissions)
        job_ids.append(job_id)

    if not job_ids:
        raise RuntimeError("No lane tasks were selected for submission")
    finalizer_name = f"sc{token}_zb{batch_index:03d}a{attempt}"
    dependency = ":".join(job_ids)
    finalizer_command = [
        "sbatch",
        "--parsable",
        f"--job-name={finalizer_name}",
        f"--comment=switchcraft:{token}:batch{batch_index:03d}:finalizer:attempt{attempt}",
        f"--dependency=afterany:{dependency}",
        (
            "--export=ALL,"
            f"RUN_ROOT={run_root},BATCH_INDEX={batch_index},ATTEMPT={attempt}"
        ),
        str(SCRIPT_ROOT / "finalize_generation_batch.sbatch"),
    ]
    if "finalizer" not in attempt_record:
        finalizer_id = submit_or_recover(
            finalizer_name, finalizer_command, run["created_at"], dry_run
        )
        if not dry_run:
            attempt_record["finalizer"] = {
                "job_id": finalizer_id,
                "job_name": finalizer_name,
                "dependency_job_ids": job_ids,
                "submitted_at": utc_now(),
            }
            atomic_json(submission_path(batch_dir), submissions)
    return submissions


def validate_batch(run_root: Path, batch_index: int, attempt: int) -> tuple[dict, dict[str, list[int]], list[dict]]:
    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    incomplete: dict[str, list[int]] = {"slow": [], "fast": []}
    partial_designs: list[dict] = []
    lane_reports: dict[str, dict] = {}
    total_designs = valid_designs = 0
    for lane in ("slow", "fast"):
        rows = read_tsv(batch_dir / f"{lane}.tsv")
        task_details: list[dict] = []
        complete_tasks = 0
        for row in rows:
            task_valid = True
            design_details: list[dict] = []
            motifs = [] if row["motifs"] == "-" else row["motifs"].split(",")
            for index in design_indices(row):
                total_designs += 1
                design_dir = Path(row["design_root"]) / f"design{index}"
                report = inspect_design(design_dir, int(row["num_states"]), motifs)
                if report["valid"]:
                    valid_designs += 1
                    continue
                task_valid = False
                detail = {
                    "case_id": row["case_id"],
                    "design_index": index,
                    "design_dir": str(design_dir),
                    "report": report,
                }
                design_details.append(detail)
                if design_dir.exists():
                    partial_designs.append(detail)
            if task_valid:
                complete_tasks += 1
            else:
                task_id = int(row["task_id"])
                incomplete[lane].append(task_id)
                task_details.append(
                    {"task_id": task_id, "incomplete_designs": design_details}
                )
        lane_reports[lane] = {
            "tasks": len(rows),
            "complete_tasks": complete_tasks,
            "incomplete_tasks": len(incomplete[lane]),
            "details": task_details,
        }
    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "batch_index": batch_index,
        "attempt": attempt,
        "complete": not incomplete["slow"] and not incomplete["fast"],
        "total_designs": total_designs,
        "valid_designs": valid_designs,
        "incomplete_designs": total_designs - valid_designs,
        "lanes": lane_reports,
    }
    atomic_json(
        batch_dir / f"artifact_validation_attempt_{attempt:02d}.json", report
    )
    return report, incomplete, partial_designs


def validate_wave(
    run_root: Path, batch_index: int, wave_index: int, attempt: int
) -> tuple[dict, dict[str, list[int]], list[dict]]:
    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    plan = ensure_wave_plan(run_root, batch_index)
    selected = plan["waves"][wave_index]["lanes"]
    incomplete: dict[str, list[int]] = {"slow": [], "fast": []}
    partial_designs: list[dict] = []
    lane_reports: dict[str, dict] = {}
    total_designs = valid_designs = 0

    for lane in ("slow", "fast"):
        by_task_id = {
            int(row["task_id"]): row for row in read_tsv(batch_dir / f"{lane}.tsv")
        }
        details: list[dict] = []
        complete_tasks = 0
        for task_id in selected[lane]:
            row = by_task_id[int(task_id)]
            task_valid = True
            design_details: list[dict] = []
            motifs = [] if row["motifs"] == "-" else row["motifs"].split(",")
            for index in design_indices(row):
                total_designs += 1
                design_dir = Path(row["design_root"]) / f"design{index}"
                report = inspect_design(
                    design_dir, int(row["num_states"]), motifs
                )
                if report["valid"]:
                    valid_designs += 1
                    continue
                task_valid = False
                detail = {
                    "case_id": row["case_id"],
                    "design_index": index,
                    "design_dir": str(design_dir),
                    "report": report,
                }
                design_details.append(detail)
                if design_dir.exists():
                    partial_designs.append(detail)
            if task_valid:
                complete_tasks += 1
            else:
                incomplete[lane].append(int(task_id))
                details.append(
                    {"task_id": int(task_id), "incomplete_designs": design_details}
                )
        lane_reports[lane] = {
            "tasks": len(selected[lane]),
            "complete_tasks": complete_tasks,
            "incomplete_tasks": len(incomplete[lane]),
            "details": details,
        }

    report = {
        "schema_version": 1,
        "created_at": utc_now(),
        "batch_index": batch_index,
        "wave_index": wave_index,
        "attempt": attempt,
        "complete": not incomplete["slow"] and not incomplete["fast"],
        "total_designs": total_designs,
        "valid_designs": valid_designs,
        "incomplete_designs": total_designs - valid_designs,
        "lanes": lane_reports,
    }
    atomic_json(
        batch_dir
        / f"artifact_validation_wave_{wave_index:03d}_attempt_{attempt:02d}.json",
        report,
    )
    return report, incomplete, partial_designs


def collect_slurm_rows(job_id: str, lane: str) -> list[dict[str, str]]:
    output = ""
    for retry in range(6):
        output = subprocess.run(
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-j",
                job_id,
                "--format=JobID,State,ExitCode,ElapsedRaw,NodeList,Reason",
            ],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        if re.search(rf"^{re.escape(job_id)}_[0-9]+\|", output, re.MULTILINE):
            break
        if retry < 5:
            time.sleep(5)
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) < 6:
            continue
        match = re.match(rf"^{re.escape(job_id)}_(\d+)$", fields[0])
        if not match:
            continue
        rows.append(
            {
                "lane": lane,
                "task_id": match.group(1),
                "state": fields[1].split()[0].rstrip("+"),
                "exit_code": fields[2],
                "elapsed_seconds": fields[3],
                "node": fields[4],
                "reason": fields[5],
            }
        )
    return rows


def slurm_attempt_report(run_root: Path, batch_index: int, attempt: int) -> dict[tuple[str, int], dict[str, str]]:
    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    submissions = load_json(submission_path(batch_dir))
    if not submissions or str(attempt) not in submissions["attempts"]:
        raise RuntimeError(f"Missing submission receipt for batch {batch_index} attempt {attempt}")
    attempt_record = submissions["attempts"][str(attempt)]
    rows: list[dict[str, str]] = []
    for lane, record in attempt_record["lanes"].items():
        rows.extend(collect_slurm_rows(record["job_id"], lane))
    write_tsv(
        batch_dir / f"slurm_attempt_{attempt:02d}.tsv",
        rows,
        fields=("lane", "task_id", "state", "exit_code", "elapsed_seconds", "node", "reason"),
    )
    return {(row["lane"], int(row["task_id"])): row for row in rows}


def slurm_wave_report(
    run_root: Path, batch_index: int, wave_index: int, attempt: int
) -> dict[tuple[str, int], dict[str, str]]:
    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    submissions = load_json(wave_submission_path(batch_dir))
    try:
        attempt_record = submissions["waves"][str(wave_index)]["attempts"][str(attempt)]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Missing receipt for batch {batch_index} wave {wave_index} attempt {attempt}"
        ) from exc
    rows: list[dict[str, str]] = []
    for lane, record in attempt_record["lanes"].items():
        rows.extend(collect_slurm_rows(record["job_id"], lane))
    write_tsv(
        batch_dir / f"slurm_wave_{wave_index:03d}_attempt_{attempt:02d}.tsv",
        rows,
        fields=(
            "lane",
            "task_id",
            "state",
            "exit_code",
            "elapsed_seconds",
            "node",
            "reason",
        ),
    )
    return {(row["lane"], int(row["task_id"])): row for row in rows}


def quarantine_partial(
    run_root: Path,
    batch_index: int,
    attempt: int,
    partial_designs: list[dict],
    wave_index: int | None = None,
) -> list[dict]:
    moved: list[dict] = []
    seen: set[str] = set()
    for detail in partial_designs:
        source = Path(detail["design_dir"])
        if str(source) in seen or not source.exists():
            continue
        seen.add(str(source))
        destination = (
            run_root
            / "quarantine"
            / f"batch_{batch_index:03d}"
            / (f"wave_{wave_index:03d}" if wave_index is not None else "whole_batch")
            / f"attempt_{attempt:02d}"
            / detail["case_id"]
            / source.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"Quarantine destination already exists: {destination}")
        shutil.move(str(source), str(destination))
        moved.append(
            {
                "case_id": detail["case_id"],
                "design_index": detail["design_index"],
                "source": str(source),
                "destination": str(destination),
            }
        )
    if moved:
        scope = (
            f"wave_{wave_index:03d}_attempt_{attempt:02d}"
            if wave_index is not None
            else f"attempt_{attempt:02d}"
        )
        atomic_json(
            run_root
            / "batches"
            / f"batch_{batch_index:03d}"
            / f"quarantine_{scope}.json",
            {"created_at": utc_now(), "moved": moved},
        )
    return moved


def validate_all(run_root: Path) -> dict:
    run = load_run(run_root)
    manifest = load_manifest(Path(run["manifest"]))
    invalid: list[dict] = []
    valid = 0
    total = int(manifest["total_designs"])
    for position, (case, index) in enumerate(target_pairs(manifest), start=1):
        report = inspect_case_design(case, index)
        if report["valid"]:
            valid += 1
        elif len(invalid) < 500:
            invalid.append(
                {"case_id": case["case_id"], "design_index": index, "report": report}
            )
        if position % 2000 == 0:
            print(f"final_validation scanned={position}/{total} valid={valid}", flush=True)
    result = {
        "created_at": utc_now(),
        "complete": valid == total,
        "target_designs": total,
        "valid_designs": valid,
        "invalid_designs": total - valid,
        "invalid_examples": invalid,
        "max_examples": 500,
    }
    atomic_json(run_root / "final_validation.json", result)
    return result


def finalize_attempt(
    run_root: Path,
    batch_index: int,
    attempt: int,
    *,
    retry_cancelled: bool = False,
    retry_missing_accounting: bool = False,
    dry_run: bool = False,
) -> dict:
    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    artifact_report, incomplete, partial_designs = validate_batch(
        run_root, batch_index, attempt
    )
    slurm = slurm_attempt_report(run_root, batch_index, attempt)
    summary = {
        "created_at": utc_now(),
        "batch_index": batch_index,
        "attempt": attempt,
        "artifact_complete": artifact_report["complete"],
        "retry_cancelled": retry_cancelled,
        "retry_missing_accounting": retry_missing_accounting,
        "dry_run": dry_run,
        "status": None,
    }

    if not artifact_report["complete"]:
        blocked: list[dict] = []
        submissions = load_json(submission_path(batch_dir))
        current = submissions["attempts"][str(attempt)]["lanes"]
        for lane in ("slow", "fast"):
            submitted_ids = set(current.get(lane, {}).get("task_ids", []))
            for task_id in incomplete[lane]:
                status = slurm.get((lane, task_id))
                if task_id not in submitted_ids:
                    blocked.append(
                        {"lane": lane, "task_id": task_id, "reason": "not_in_current_attempt"}
                    )
                elif status is None and not retry_missing_accounting:
                    blocked.append(
                        {"lane": lane, "task_id": task_id, "reason": "missing_sacct_state"}
                    )
                elif status is not None and (
                    status["state"] == "OUT_OF_MEMORY"
                    or (
                        status["state"] == "CANCELLED"
                        and not retry_cancelled
                    )
                ):
                    blocked.append(
                        {
                            "lane": lane,
                            "task_id": task_id,
                            "reason": status["state"],
                        }
                    )
        if not dry_run:
            quarantine_partial(run_root, batch_index, attempt, partial_designs)
        if attempt >= MAX_ATTEMPTS:
            blocked.append({"reason": "max_attempts_reached", "attempt": attempt})
        if blocked:
            summary["status"] = "blocked"
            summary["blocked"] = blocked
            summary["incomplete_task_ids"] = incomplete
            if not dry_run:
                atomic_json(batch_dir / "summary.json", summary)
                atomic_json(run_root / "status.json", summary)
            return summary

        summary["status"] = "retry_planned" if dry_run else "retry_submitted"
        summary["next_attempt"] = attempt + 1
        summary["incomplete_task_ids"] = incomplete
        if not dry_run:
            atomic_json(batch_dir / "summary.json", summary)
        submit_attempt(
            run_root,
            batch_index,
            attempt + 1,
            lane_task_ids=incomplete,
            dry_run=dry_run,
        )
        return summary

    summary["status"] = "complete"
    if dry_run:
        summary["status"] = "complete_dry_run"
        return summary
    atomic_json(batch_dir / "summary.json", summary)
    next_index = batch_index + 1
    next_batch = create_batch(run_root, next_index)
    if next_batch is not None:
        summary["status"] = "next_batch_submitted"
        summary["next_batch_index"] = next_index
        atomic_json(batch_dir / "summary.json", summary)
        submit_attempt(run_root, next_index, 1, dry_run=False)
        return summary

    final = validate_all(run_root)
    if final["complete"]:
        summary["status"] = "run_complete"
    else:
        summary["status"] = "blocked_final_validation"
        summary["invalid_designs"] = final["invalid_designs"]
    atomic_json(run_root / "status.json", summary)
    atomic_json(batch_dir / "summary.json", summary)
    return summary


def finalize_wave_attempt(
    run_root: Path, batch_index: int, wave_index: int, attempt: int
) -> dict:
    batch_dir = run_root / "batches" / f"batch_{batch_index:03d}"
    plan = ensure_wave_plan(run_root, batch_index)
    artifact_report, incomplete, partial_designs = validate_wave(
        run_root, batch_index, wave_index, attempt
    )
    slurm = slurm_wave_report(run_root, batch_index, wave_index, attempt)
    summary = {
        "created_at": utc_now(),
        "batch_index": batch_index,
        "wave_index": wave_index,
        "attempt": attempt,
        "artifact_complete": artifact_report["complete"],
        "status": None,
    }
    summary_path = batch_dir / f"wave_{wave_index:03d}_summary.json"

    if not artifact_report["complete"]:
        blocked: list[dict] = []
        submissions = load_json(wave_submission_path(batch_dir))
        current = submissions["waves"][str(wave_index)]["attempts"][str(attempt)][
            "lanes"
        ]
        for lane in ("slow", "fast"):
            submitted_ids = set(current.get(lane, {}).get("task_ids", []))
            for task_id in incomplete[lane]:
                status = slurm.get((lane, task_id))
                if task_id not in submitted_ids:
                    blocked.append(
                        {
                            "lane": lane,
                            "task_id": task_id,
                            "reason": "not_in_current_attempt",
                        }
                    )
                elif status is None:
                    blocked.append(
                        {
                            "lane": lane,
                            "task_id": task_id,
                            "reason": "missing_sacct_state",
                        }
                    )
                elif status["state"] in {"OUT_OF_MEMORY", "CANCELLED"}:
                    blocked.append(
                        {
                            "lane": lane,
                            "task_id": task_id,
                            "reason": status["state"],
                        }
                    )
        quarantine_partial(
            run_root,
            batch_index,
            attempt,
            partial_designs,
            wave_index=wave_index,
        )
        if attempt >= MAX_ATTEMPTS:
            blocked.append({"reason": "max_attempts_reached", "attempt": attempt})
        if blocked:
            summary["status"] = "blocked"
            summary["blocked"] = blocked
            summary["incomplete_task_ids"] = incomplete
            atomic_json(summary_path, summary)
            atomic_json(run_root / "status.json", summary)
            return summary

        summary["status"] = "retry_submission_pending"
        summary["next_attempt"] = attempt + 1
        summary["incomplete_task_ids"] = incomplete
        atomic_json(summary_path, summary)
        submit_wave_attempt(
            run_root,
            batch_index,
            wave_index,
            attempt + 1,
            lane_task_ids=incomplete,
            dry_run=False,
        )
        summary["status"] = "retry_submitted"
        atomic_json(summary_path, summary)
        return summary

    summary["status"] = "complete"
    atomic_json(summary_path, summary)
    next_wave_index = wave_index + 1
    if next_wave_index < int(plan["wave_count"]):
        summary["status"] = "next_wave_submission_pending"
        summary["next_wave_index"] = next_wave_index
        atomic_json(summary_path, summary)
        submit_wave_attempt(
            run_root, batch_index, next_wave_index, 1, dry_run=False
        )
        summary["status"] = "next_wave_submitted"
        atomic_json(summary_path, summary)
        return summary

    batch_report, batch_incomplete, _partial = validate_batch(
        run_root, batch_index, attempt
    )
    if not batch_report["complete"]:
        summary["status"] = "blocked_batch_validation"
        summary["incomplete_task_ids"] = batch_incomplete
        atomic_json(summary_path, summary)
        atomic_json(batch_dir / "summary.json", summary)
        atomic_json(run_root / "status.json", summary)
        return summary

    summary["status"] = "batch_complete"
    atomic_json(batch_dir / "summary.json", summary)
    next_batch_index = batch_index + 1
    next_batch = create_batch(run_root, next_batch_index)
    if next_batch is not None:
        ensure_wave_plan(run_root, next_batch_index)
        summary["status"] = "next_batch_submission_pending"
        summary["next_batch_index"] = next_batch_index
        atomic_json(batch_dir / "summary.json", summary)
        submit_wave_attempt(
            run_root, next_batch_index, 0, 1, dry_run=False
        )
        summary["status"] = "next_batch_submitted"
        atomic_json(batch_dir / "summary.json", summary)
        return summary

    final = validate_all(run_root)
    if final["complete"]:
        summary["status"] = "run_complete"
    else:
        summary["status"] = "blocked_final_validation"
        summary["invalid_designs"] = final["invalid_designs"]
    atomic_json(run_root / "status.json", summary)
    atomic_json(batch_dir / "summary.json", summary)
    return summary


def status_report(run_root: Path) -> dict:
    run = load_run(run_root)
    batches = []
    job_ids: list[str] = []
    for batch_dir in batch_dirs(run_root):
        submission = load_json(submission_path(batch_dir), {})
        for attempt in submission.get("attempts", {}).values():
            job_ids.extend(
                record["job_id"] for record in attempt.get("lanes", {}).values()
            )
            if attempt.get("finalizer"):
                job_ids.append(attempt["finalizer"]["job_id"])
        wave_submission = load_json(wave_submission_path(batch_dir), {})
        for wave in wave_submission.get("waves", {}).values():
            for attempt in wave.get("attempts", {}).values():
                job_ids.extend(
                    record["job_id"] for record in attempt.get("lanes", {}).values()
                )
                if attempt.get("finalizer"):
                    job_ids.append(attempt["finalizer"]["job_id"])
        batches.append(
            {
                "batch": load_json(batch_dir / "batch.json"),
                "summary": load_json(batch_dir / "summary.json"),
                "submission": submission,
                "wave_plan": load_json(wave_plan_path(batch_dir)),
                "wave_submission": wave_submission,
            }
        )
    scheduler = []
    numeric_ids = [job_id for job_id in job_ids if str(job_id).isdigit()]
    if numeric_ids:
        output = subprocess.run(
            ["squeue", "-h", "-j", ",".join(sorted(set(numeric_ids))), "-o", "%A|%F|%K|%T|%M|%R"],
            check=False,
            text=True,
            capture_output=True,
        ).stdout
        scheduler = [line for line in output.splitlines() if line]
    return {
        "run": run,
        "status": load_json(run_root / "status.json"),
        "batches": batches,
        "active_scheduler_rows": scheduler,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    submit.add_argument("--batch-index", type=int, default=0)
    submit.add_argument("--attempt", type=int, default=1)
    submit.add_argument("--dry-run", action="store_true")

    submit_wave = subparsers.add_parser("submit-wave")
    submit_wave.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    submit_wave.add_argument("--batch-index", type=int, default=0)
    submit_wave.add_argument("--wave-index", type=int, default=0)
    submit_wave.add_argument("--attempt", type=int, default=1)
    submit_wave.add_argument("--dry-run", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run-root", type=Path, required=True)
    finalize.add_argument("--batch-index", type=int, required=True)
    finalize.add_argument("--attempt", type=int, required=True)
    finalize.add_argument(
        "--retry-cancelled",
        action="store_true",
        help=(
            "Explicitly resume incomplete task IDs from a user-cancelled attempt; "
            "the default finalizer remains fail-closed on CANCELLED tasks"
        ),
    )
    finalize.add_argument(
        "--retry-missing-accounting",
        action="store_true",
        help=(
            "Explicitly resume artifact-incomplete task IDs whose individual Slurm "
            "accounting rows are absent; the default remains fail-closed"
        ),
    )
    finalize.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and render the next-attempt submissions without submitting jobs",
    )

    finalize_wave = subparsers.add_parser("finalize-wave")
    finalize_wave.add_argument("--run-root", type=Path, required=True)
    finalize_wave.add_argument("--batch-index", type=int, required=True)
    finalize_wave.add_argument("--wave-index", type=int, required=True)
    finalize_wave.add_argument("--attempt", type=int, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        with run_lock(args.run_root.resolve()):
            result = prepare_run(args.manifest, args.run_root)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "submit":
        with run_lock(args.run_root.resolve()):
            result = submit_attempt(
                args.run_root.resolve(),
                args.batch_index,
                args.attempt,
                dry_run=args.dry_run,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "submit-wave":
        with run_lock(args.run_root.resolve()):
            result = submit_wave_attempt(
                args.run_root.resolve(),
                args.batch_index,
                args.wave_index,
                args.attempt,
                dry_run=args.dry_run,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "finalize":
        with run_lock(args.run_root.resolve()):
            result = finalize_attempt(
                args.run_root.resolve(),
                args.batch_index,
                args.attempt,
                retry_cancelled=args.retry_cancelled,
                retry_missing_accounting=args.retry_missing_accounting,
                dry_run=args.dry_run,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] in {"blocked", "blocked_final_validation"}:
            raise SystemExit(1)
    elif args.command == "finalize-wave":
        with run_lock(args.run_root.resolve()):
            result = finalize_wave_attempt(
                args.run_root.resolve(),
                args.batch_index,
                args.wave_index,
                args.attempt,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"].startswith("blocked"):
            raise SystemExit(1)
    elif args.command == "status":
        print(json.dumps(status_report(args.run_root.resolve()), indent=2, sort_keys=True))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
