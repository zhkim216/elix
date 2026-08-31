#!/usr/bin/env python3
"""Run one prepared RFD3 partial-diffusion JSON and save CA-aligned CIFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from atomworks.io.utils.io_utils import to_cif_file
from rfd3.constants import SAVED_CONDITIONING_ANNOTATIONS
from rfd3.inference.input_parsing import preserve_explicit_nonpolymer_identity
from rfd3.utils.inference import inference_load_


EXPECTED_FOUNDRY_VERSION = "0.2.0"
RAW_OUTPUT_PATTERN = re.compile(r"_(\d+)_model_(\d+)\.cif\.gz$")
SAFE_ROLE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing file without --overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def load_structure(path: Path):
    atom_array = inference_load_(str(path))["atom_array"]
    if len(atom_array) == 0:
        raise ValueError(f"No atoms parsed from {path}")
    if not np.isfinite(np.asarray(atom_array.coord, dtype=float)).all():
        raise ValueError(f"Non-finite coordinates in {path}")
    return atom_array


def annotation_as_str(atom_array, name: str) -> np.ndarray:
    if name not in atom_array.get_annotation_categories():
        return np.full(len(atom_array), "", dtype=str)
    return np.asarray(atom_array.get_annotation(name), dtype=str)


def protein_ca_indices(atom_array, chain_id: str) -> dict[tuple[str, int, str], int]:
    insertion_codes = annotation_as_str(atom_array, "ins_code")
    mask = (
        (atom_array.chain_id.astype(str) == chain_id)
        & (atom_array.atom_name.astype(str) == "CA")
    )
    indices: dict[tuple[str, int, str], int] = {}
    for index in np.flatnonzero(mask):
        key = (chain_id, int(atom_array.res_id[index]), str(insertion_codes[index]))
        if key in indices:
            raise ValueError(f"Duplicate protein CA key: {key}")
        indices[key] = int(index)
    if len(indices) < 3:
        raise ValueError(f"Need at least three CA atoms on protein chain {chain_id!r}")
    return indices


def protein_residue_names(atom_array, chain_id: str) -> dict[tuple[str, int, str], str]:
    insertion_codes = annotation_as_str(atom_array, "ins_code")
    mask = atom_array.chain_id.astype(str) == chain_id
    residue_names: dict[tuple[str, int, str], str] = {}
    for index in np.flatnonzero(mask):
        key = (chain_id, int(atom_array.res_id[index]), str(insertion_codes[index]))
        value = str(atom_array.res_name[index])
        previous = residue_names.setdefault(key, value)
        if previous != value:
            raise ValueError(f"Multiple residue names for protein key {key}: {previous}, {value}")
    if not residue_names:
        raise ValueError(f"No protein residues found on chain {chain_id!r}")
    return residue_names


def context_mask(atom_array, chain_id: str) -> np.ndarray:
    mask = atom_array.chain_id.astype(str) == chain_id
    if not np.any(mask):
        raise ValueError(f"No ligand atoms found on context chain {chain_id!r}")
    residue_keys = set(
        zip(
            atom_array.res_id[mask].astype(int),
            annotation_as_str(atom_array, "ins_code")[mask].astype(str),
            strict=True,
        )
    )
    if len(residue_keys) != 1:
        raise ValueError(
            f"Context chain {chain_id!r} must contain one residue, got {residue_keys}"
        )
    return mask


def ligand_heavy_atom_coords(atom_array, chain_id: str) -> dict[tuple[str, str], np.ndarray]:
    mask = context_mask(atom_array, chain_id)
    mask &= np.char.upper(atom_array.element.astype(str)) != "H"
    coordinates: dict[tuple[str, str], np.ndarray] = {}
    for atom_name, element, coordinate in zip(
        atom_array.atom_name[mask].astype(str),
        atom_array.element[mask].astype(str),
        atom_array.coord[mask],
        strict=True,
    ):
        key = (atom_name, element.upper())
        if key in coordinates:
            raise ValueError(f"Duplicate ligand atom identity: {key}")
        coordinates[key] = np.asarray(coordinate, dtype=float)
    if not coordinates:
        raise ValueError(f"No ligand heavy atoms found on context chain {chain_id!r}")
    return coordinates


def distance_matrix_max_error(
    reference: dict[tuple[str, str], np.ndarray],
    observed: dict[tuple[str, str], np.ndarray],
) -> float:
    if set(reference) != set(observed):
        raise ValueError(
            "Fixed-ligand atom identity mismatch: "
            f"missing={sorted(set(reference) - set(observed))}, "
            f"extra={sorted(set(observed) - set(reference))}"
        )
    keys = sorted(reference)
    reference_coord = np.stack([reference[key] for key in keys])
    observed_coord = np.stack([observed[key] for key in keys])
    reference_dist = np.linalg.norm(
        reference_coord[:, None, :] - reference_coord[None, :, :], axis=-1
    )
    observed_dist = np.linalg.norm(
        observed_coord[:, None, :] - observed_coord[None, :, :], axis=-1
    )
    return float(np.max(np.abs(reference_dist - observed_dist)))


def kabsch_align(mobile: np.ndarray, target: np.ndarray, full: np.ndarray):
    mobile_centroid = mobile.mean(axis=0)
    target_centroid = target.mean(axis=0)
    mobile_centered = mobile - mobile_centroid
    target_centered = target - target_centroid
    u, _, vt = np.linalg.svd(mobile_centered.T @ target_centered)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    aligned_full = (full - mobile_centroid) @ rotation + target_centroid
    aligned_mobile = (mobile - mobile_centroid) @ rotation + target_centroid
    rmsd = float(np.sqrt(np.mean(np.sum((aligned_mobile - target) ** 2, axis=1))))
    return aligned_full, rmsd


def restore_context_identity(prediction, reference, chain_id: str) -> None:
    prediction_mask = context_mask(prediction, chain_id)
    reference_mask = context_mask(reference, chain_id)
    reference_res_ids = set(map(int, reference.res_id[reference_mask]))
    reference_res_names = set(reference.res_name[reference_mask].astype(str))
    reference_ins_codes = set(annotation_as_str(reference, "ins_code")[reference_mask])
    if len(reference_res_ids) != 1 or len(reference_res_names) != 1 or len(reference_ins_codes) != 1:
        raise ValueError("Reference context identity is not unique")
    prediction.res_id[prediction_mask] = next(iter(reference_res_ids))
    prediction.res_name[prediction_mask] = next(iter(reference_res_names))
    if "ins_code" in prediction.get_annotation_categories():
        prediction.ins_code[prediction_mask] = next(iter(reference_ins_codes))
    prediction.chain_type[prediction_mask] = reference.chain_type[reference_mask][0]
    prediction.is_polymer[prediction_mask] = reference.is_polymer[reference_mask][0]
    prediction.hetero[prediction_mask] = reference.hetero[reference_mask][0]


def partial_t_token(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def expected_condition(spec: dict[str, Any]) -> str:
    return f"partialt{partial_t_token(float(spec['partial_t']))}"


def sample_identity(spec: dict[str, Any]) -> str:
    extra = spec.get("extra", {})
    return str(extra.get("sample_id") or extra.get("role_sample_id") or "")


def validate_preparation_contract(
    output_root: Path, input_json: Path, spec: dict[str, Any]
) -> dict[str, str]:
    preparation_manifest = output_root / "preparation_manifest.json"
    prepared_manifest = output_root / "prepared_inputs_manifest.csv"
    if not preparation_manifest.is_file() or not prepared_manifest.is_file():
        raise FileNotFoundError("Missing preparation manifests; prepare inputs first")
    preparation = json.loads(preparation_manifest.read_text(encoding="utf-8"))
    allowed_statuses = {"complete"}
    if bool(preparation.get("allow_partial_preparation")):
        allowed_statuses.add("complete_with_failures")
    if preparation.get("status") not in allowed_statuses:
        raise ValueError("Preparation manifest is not complete")
    diagnostics = preparation.get("diagnostics", {})
    nonzero = {key: value for key, value in diagnostics.items() if int(value) != 0}
    if nonzero and not bool(preparation.get("allow_partial_preparation")):
        raise ValueError(f"Preparation manifest has nonzero diagnostics: {nonzero}")

    matches: list[dict[str, str]] = []
    with prepared_manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if Path(row["input_json"]).resolve() == input_json:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one prepared manifest row for {input_json}, got {len(matches)}"
        )
    row = matches[0]
    if row["input_json_sha256"] != sha256_file(input_json):
        raise ValueError("Prepared input JSON digest does not match manifest")
    extra = spec["extra"]
    identity_key = "sample_id" if "sample_id" in extra else "role_sample_id"
    checks = {
        identity_key: str(extra[identity_key]),
        "condition": str(extra["condition"]),
        "original_cif_sha256": str(extra["original_cif_sha256"]),
        "rfd3_input_cif_sha256": str(extra["rfd3_input_cif_sha256"]),
    }
    for key, expected in checks.items():
        if row[key] != expected:
            raise ValueError(f"Prepared manifest mismatch for {key}: {row[key]!r} != {expected!r}")
    return row


def complete_output_exists(
    *,
    manifest_path: Path,
    input_json: Path,
    identity: str,
    condition: str,
    num_samples: int,
    final_paths: list[Path],
) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    manifest_identity = str(
        manifest.get("sample_id") or manifest.get("role_sample_id") or ""
    )
    if (
        manifest.get("status") != "complete"
        or manifest_identity != identity
        or manifest.get("condition") != condition
        or int(manifest.get("num_samples_requested", -1)) != num_samples
        or int(manifest.get("num_samples_generated", -1)) != num_samples
        or manifest.get("input_json_sha256") != sha256_file(input_json)
    ):
        return False
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != num_samples:
        return False
    observed_indices = [record.get("sample_index") for record in samples]
    if observed_indices != list(range(num_samples)):
        return False
    observed_paths = [Path(str(record.get("output_cif", ""))).resolve() for record in samples]
    expected_paths = [path.resolve() for path in final_paths]
    return observed_paths == expected_paths and all(path.is_file() for path in expected_paths)


def record_failure(args: argparse.Namespace, error: BaseException) -> None:
    try:
        input_json = args.input_json.resolve()
        payload = json.loads(input_json.read_text(encoding="utf-8"))
        _, spec = next(iter(payload.items()))
        identity = sample_identity(spec) or input_json.stem
        condition = expected_condition(spec)
        if not SAFE_ROLE_ID.fullmatch(identity):
            identity = input_json.stem
        failure_dir = args.output_root.resolve() / "failure_attempts" / condition / identity
        now = datetime.now(timezone.utc)
        job_token = os.environ.get("SLURM_JOB_ID", "local")
        restart_token = os.environ.get("SLURM_RESTART_COUNT", "0")
        failure_path = failure_dir / (
            f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}_job{job_token}_"
            f"restart{restart_token}_pid{os.getpid()}.json"
        )
        failure = {
            "schema_version": 1,
            "status": "failed",
            "failed_at_utc": now.isoformat(),
            "sample_id": identity,
            "condition": condition,
            "input_json": str(input_json),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_restart_count": int(restart_token),
        }
        atomic_write_json(failure_path, failure, overwrite=False)
        print(f"Failure status written: {failure_path}", flush=True)
    except Exception as record_error:  # pragma: no cover - best-effort error path
        print(f"Could not write failure status: {record_error}", flush=True)


def runtime_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import rfd3

    package_version = version("rc-foundry")
    if package_version != EXPECTED_FOUNDRY_VERSION:
        raise RuntimeError(
            f"Expected rc-foundry {EXPECTED_FOUNDRY_VERSION}, got {package_version}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Foundry container")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source = Path(inspect.getfile(rfd3)).resolve()
    container_path = args.container.resolve() if args.container else None
    container_sha256 = args.container_sha256
    if container_path is not None:
        if not container_path.is_file():
            raise FileNotFoundError(container_path)
        if container_sha256 is None:
            container_sha256 = sha256_file(container_path)
    return {
        "rc_foundry_version": package_version,
        "rfd3_source": str(source),
        "cuda_device": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256 or sha256_file(checkpoint),
        "container": str(container_path) if container_path else None,
        "container_sha256": container_sha256,
    }


def run_sampling(command: list[str], raw_dir: Path) -> None:
    log_path = raw_dir / "sampling.log"
    print(shlex.join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def discover_raw_outputs(
    raw_dir: Path,
    expected_count: int,
    diffusion_batch_size: int,
) -> list[tuple[int, Path, Path]]:
    outputs: list[tuple[int, Path, Path]] = []
    for cif_path in raw_dir.glob("*.cif.gz"):
        match = RAW_OUTPUT_PATTERN.search(cif_path.name)
        if match is None:
            continue
        batch_index, model_index = map(int, match.groups())
        if model_index >= diffusion_batch_size:
            raise ValueError(f"Unexpected RFD3 model index in {cif_path.name}")
        sample_index = batch_index * diffusion_batch_size + model_index
        metadata_path = Path(str(cif_path)[: -len(".cif.gz")] + ".json")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing raw metadata for {cif_path}")
        outputs.append((sample_index, cif_path, metadata_path))
    outputs.sort(key=lambda item: item[0])
    observed_indices = [item[0] for item in outputs]
    if observed_indices != list(range(expected_count)):
        raise ValueError(
            f"Raw sample indices mismatch: observed={observed_indices}, "
            f"expected={list(range(expected_count))}"
        )
    return outputs


def validate_raw_specification(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("input", "ligand", "partial_t", "select_fixed_atoms", "select_unfixed_sequence"):
        if observed.get(key) != expected.get(key):
            raise ValueError(
                f"Raw output specification mismatch for {key}: "
                f"{observed.get(key)!r} != {expected.get(key)!r}"
            )
    observed_extra = observed.get("extra", {})
    for key, value in expected["extra"].items():
        if observed_extra.get(key) != value:
            raise ValueError(f"Raw output extra metadata mismatch for {key}")


def align_and_write_sample(
    *,
    reference,
    raw_cif: Path,
    raw_metadata: Path,
    expected_spec: dict[str, Any],
    output_cif: Path,
    binder_chain: str,
    context_chain: str,
    sequence_unfixed: bool,
) -> dict[str, Any]:
    metadata = json.loads(raw_metadata.read_text(encoding="utf-8"))
    validate_raw_specification(metadata.get("specification", {}), expected_spec)
    prediction = load_structure(raw_cif)
    restore_context_identity(prediction, reference, context_chain)
    prediction = preserve_explicit_nonpolymer_identity(prediction)

    reference_ca = protein_ca_indices(reference, binder_chain)
    prediction_ca = protein_ca_indices(prediction, binder_chain)
    if set(reference_ca) != set(prediction_ca):
        raise ValueError(
            "Protein CA key mismatch: "
            f"missing={sorted(set(reference_ca) - set(prediction_ca))}, "
            f"extra={sorted(set(prediction_ca) - set(reference_ca))}"
        )
    # (JH) fixed: key-set equality alone cannot detect residue reordering across an internal gap.
    if list(reference_ca) != list(prediction_ca):
        raise ValueError("Protein CA residue order changed in RFD3 output")
    keys = sorted(reference_ca)
    reference_indices = np.asarray([reference_ca[key] for key in keys], dtype=int)
    prediction_indices = np.asarray([prediction_ca[key] for key in keys], dtype=int)
    if not sequence_unfixed:
        reference_names = protein_residue_names(reference, binder_chain)
        prediction_names = protein_residue_names(prediction, binder_chain)
        if prediction_names != reference_names:
            raise ValueError("Sequence-fixed output changed binder residue identities")

    reference_ligand = ligand_heavy_atom_coords(reference, context_chain)
    prediction_ligand = ligand_heavy_atom_coords(prediction, context_chain)
    # (JH) fixed: retain ligand geometry drift as a diagnostic; no authoritative hard cutoff exists.
    ligand_distance_error = distance_matrix_max_error(reference_ligand, prediction_ligand)

    aligned_coordinates, fitted_ca_rmsd = kabsch_align(
        np.asarray(prediction.coord[prediction_indices], dtype=float),
        np.asarray(reference.coord[reference_indices], dtype=float),
        np.asarray(prediction.coord, dtype=float),
    )
    prediction.coord = aligned_coordinates
    output_cif.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = [
        field
        for field in SAVED_CONDITIONING_ANNOTATIONS
        if field in prediction.get_annotation_categories()
    ]
    to_cif_file(
        prediction,
        output_cif,
        file_type="cif",
        include_entity_poly=False,
        extra_fields=extra_fields,
    )
    roundtrip = load_structure(output_cif)
    if len(roundtrip) != len(prediction):
        raise ValueError("Aligned CIF roundtrip changed atom count")
    roundtrip_ca = protein_ca_indices(roundtrip, binder_chain)
    if set(roundtrip_ca) != set(reference_ca):
        raise ValueError("Aligned CIF roundtrip changed protein CA keys")
    # (JH) fixed: the final written CIF must preserve the same ordered residue identifiers.
    if list(roundtrip_ca) != list(reference_ca):
        raise ValueError("Aligned CIF roundtrip changed protein CA residue order")
    roundtrip_coordinates = np.stack([roundtrip.coord[roundtrip_ca[key]] for key in keys])
    reference_coordinates = np.stack([reference.coord[reference_ca[key]] for key in keys])
    direct_ca_rmsd = float(
        np.sqrt(
            np.mean(np.sum((roundtrip_coordinates - reference_coordinates) ** 2, axis=1))
        )
    )
    centroid_offset = float(
        np.linalg.norm(roundtrip_coordinates.mean(axis=0) - reference_coordinates.mean(axis=0))
    )
    if not math.isfinite(direct_ca_rmsd) or abs(direct_ca_rmsd - fitted_ca_rmsd) > 2e-3:
        raise ValueError(
            f"Aligned CIF CA RMSD changed on write: fitted={fitted_ca_rmsd}, "
            f"roundtrip={direct_ca_rmsd}"
        )
    if centroid_offset > 2e-3:
        raise ValueError(f"Aligned CA centroid offset too large: {centroid_offset}")
    roundtrip_ligand = ligand_heavy_atom_coords(roundtrip, context_chain)
    roundtrip_ligand_error = distance_matrix_max_error(reference_ligand, roundtrip_ligand)
    return {
        "raw_cif": str(raw_cif),
        "raw_cif_sha256": sha256_file(raw_cif),
        "raw_metadata": str(raw_metadata),
        "matched_ca_count": len(keys),
        "protein_ca_rmsd_to_reference": fitted_ca_rmsd,
        "roundtrip_direct_ca_rmsd": direct_ca_rmsd,
        "roundtrip_ca_centroid_offset": centroid_offset,
        "fixed_ligand_heavy_atom_count": len(reference_ligand),
        "raw_fixed_ligand_distance_matrix_max_error": ligand_distance_error,
        "fixed_ligand_distance_matrix_max_error": roundtrip_ligand_error,
        "sequence_identity_checked": not sequence_unfixed,
        "output_atom_count": len(roundtrip),
    }


def run(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.diffusion_batch_size <= 0:
        raise ValueError("--diffusion-batch-size must be positive")
    n_batches = (
        args.n_batches
        if args.n_batches is not None
        else args.num_samples // args.diffusion_batch_size
    )
    if n_batches <= 0:
        raise ValueError("--n-batches must be positive")
    if args.diffusion_batch_size * n_batches != args.num_samples:
        raise ValueError(
            "--num-samples must equal --diffusion-batch-size * --n-batches"
        )
    if args.step_scale <= 0 or args.noise_scale <= 0 or args.gamma_0 < 0:
        raise ValueError("Invalid sampler scale: step/noise must be positive and gamma non-negative")

    input_json = args.input_json.resolve()
    output_root = args.output_root.resolve()
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    if len(payload) != 1:
        raise ValueError(f"Expected exactly one RFD3 specification in {input_json}")
    spec_name, spec = next(iter(payload.items()))
    if spec_name != "partial_diffusion":
        raise ValueError(f"Unexpected RFD3 specification key: {spec_name!r}")
    extra = spec.get("extra", {})
    role_id = sample_identity(spec)
    if not SAFE_ROLE_ID.fullmatch(role_id):
        raise ValueError(f"Unsafe or missing sample identity: {role_id!r}")
    condition = expected_condition(spec)
    if extra.get("condition") != condition:
        raise ValueError(f"Condition metadata mismatch: {extra.get('condition')!r} != {condition!r}")
    if not bool(extra.get("ligand_fixed")):
        raise ValueError("This runner currently requires ligand_fixed=true")
    if spec.get("select_fixed_atoms") != {spec.get("ligand"): "ALL"}:
        raise ValueError("Prepared spec does not fix exactly the selected ligand residue")
    prepared_row = validate_preparation_contract(output_root, input_json, spec)

    original_cif = Path(extra["original_cif"]).resolve()
    expected_original = output_root / "original_samples" / f"{role_id}.cif"
    if original_cif != expected_original:
        raise ValueError(f"Original CIF path is outside the prepared source contract: {original_cif}")
    if sha256_file(original_cif) != extra["original_cif_sha256"]:
        raise ValueError("Original CIF digest mismatch")
    rfd3_input = Path(spec["input"]).resolve()
    if sha256_file(rfd3_input) != extra["rfd3_input_cif_sha256"]:
        raise ValueError("RFD3 input CIF digest mismatch")

    binder_chain = str(extra["binder_chain_id"])
    context_chain = str(extra["context_chain_id"])
    sequence_unfixed = bool(extra["sequence_unfixed"])
    sequence_token = "sequnfix" if sequence_unfixed else "seqfix"
    ligand_token = "ligfix"
    partial_token = partial_t_token(float(spec["partial_t"]))
    sample_dir = output_root / condition / "samples" / role_id
    manifest_path = sample_dir / "manifest.json"
    original_copy = sample_dir / f"{role_id}.cif"
    final_paths = [
        sample_dir
        / f"{role_id}_{sequence_token}_{ligand_token}_partialt{partial_token}_sample{index}.cif"
        for index in range(args.num_samples)
    ]
    if args.resume and complete_output_exists(
        manifest_path=manifest_path,
        input_json=input_json,
        identity=role_id,
        condition=condition,
        num_samples=args.num_samples,
        final_paths=final_paths,
    ):
        print(
            json.dumps(
                {
                    "status": "skipped_complete",
                    "sample_id": role_id,
                    "condition": condition,
                    "num_samples": args.num_samples,
                    "manifest": str(manifest_path),
                },
                sort_keys=True,
            )
        )
        return
    effective_overwrite = args.overwrite or args.resume
    existing = [path for path in [manifest_path, original_copy, *final_paths] if path.exists()]
    if existing and not effective_overwrite:
        raise FileExistsError(
            "Output files already exist; use --overwrite to replace them: "
            + ", ".join(map(str, existing))
        )

    runtime = runtime_preflight(args)
    raw_parent = output_root / ".partial_diffusion_raw" / condition
    raw_parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(tempfile.mkdtemp(prefix=f"{role_id}.", dir=raw_parent))
    success = False
    try:
        command = [
            *shlex.split(args.rfd3_cmd),
            "design",
            f"ckpt_path={args.checkpoint.resolve()}",
            f"inputs={input_json}",
            f"out_dir={raw_dir}",
            f"diffusion_batch_size={args.diffusion_batch_size}",
            f"n_batches={n_batches}",
            f"seed={args.seed}",
            "skip_existing=False",
            "prevalidate_inputs=True",
            f"inference_sampler.step_scale={args.step_scale}",
            f"inference_sampler.gamma_0={args.gamma_0}",
            f"inference_sampler.noise_scale={args.noise_scale}",
            "inference_sampler.use_classifier_free_guidance=False",
            "dump_prediction_metadata_json=True",
            "output_full_json=True",
        ]
        run_sampling(command, raw_dir)
        raw_outputs = discover_raw_outputs(
            raw_dir,
            args.num_samples,
            args.diffusion_batch_size,
        )
        reference = load_structure(original_cif)
        postprocess_dir = raw_dir / "postprocessed"
        postprocess_dir.mkdir()
        sample_records: list[dict[str, Any]] = []
        staged_outputs: list[tuple[Path, Path]] = []
        for expected_index, (raw_index, raw_cif, raw_metadata) in enumerate(raw_outputs):
            if raw_index != expected_index:
                raise ValueError(f"Unexpected raw sample order: {raw_index} != {expected_index}")
            staged_cif = postprocess_dir / final_paths[expected_index].name
            record = align_and_write_sample(
                reference=reference,
                raw_cif=raw_cif,
                raw_metadata=raw_metadata,
                expected_spec=spec,
                output_cif=staged_cif,
                binder_chain=binder_chain,
                context_chain=context_chain,
                sequence_unfixed=sequence_unfixed,
            )
            record.update(
                {
                    "sample_index": expected_index,
                    "output_cif": str(final_paths[expected_index]),
                    "output_cif_sha256": sha256_file(staged_cif),
                }
            )
            sample_records.append(record)
            staged_outputs.append((staged_cif, final_paths[expected_index]))

        sample_dir.mkdir(parents=True, exist_ok=True)
        for staged_cif, final_cif in staged_outputs:
            os.replace(staged_cif, final_cif)
        atomic_copy(original_cif, original_copy, overwrite=effective_overwrite)
        if sha256_file(original_copy) != extra["original_cif_sha256"]:
            raise ValueError("Per-condition original copy digest mismatch")
        identity_fields = (
            {"sample_id": role_id}
            if "sample_id" in extra
            else {"role_sample_id": role_id}
        )
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **identity_fields,
            "condition": condition,
            "partial_t": float(spec["partial_t"]),
            "sequence_unfixed": sequence_unfixed,
            "ligand_fixed": True,
            "num_samples_requested": args.num_samples,
            "num_samples_generated": len(sample_records),
            "seed": args.seed,
            "sampler": {
                "diffusion_batch_size": args.diffusion_batch_size,
                "n_batches": n_batches,
                "step_scale": args.step_scale,
                "gamma_0": args.gamma_0,
                "noise_scale": args.noise_scale,
                "classifier_free_guidance": False,
            },
            "input_json": str(input_json),
            "input_json_sha256": sha256_file(input_json),
            "prepared_manifest_row": prepared_row,
            "original_cif": str(original_copy),
            "original_cif_sha256": sha256_file(original_copy),
            "rfd3_input_cif": str(rfd3_input),
            "rfd3_input_cif_sha256": sha256_file(rfd3_input),
            "runtime": runtime,
            "samples": sample_records,
            "diagnostics": {
                "missing": 0,
                "skipped": 0,
                "unmapped": 0,
                "unmatched": 0,
                "failed_validation": 0,
            },
        }
        atomic_write_json(manifest_path, manifest, overwrite=effective_overwrite)
        success = True
        print(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        if success and not args.keep_raw:
            shutil.rmtree(raw_dir)
        else:
            print(f"Raw RFD3 directory preserved: {raw_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--diffusion-batch-size", type=int, default=1)
    parser.add_argument("--n-batches", type=int)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-scale", type=float, default=1.5)
    parser.add_argument("--gamma-0", type=float, default=0.6)
    parser.add_argument("--noise-scale", type=float, default=1.003)
    parser.add_argument("--rfd3-cmd", default="rfd3")
    parser.add_argument("--container", type=Path)
    parser.add_argument("--container-sha256")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except BaseException as error:
        if not isinstance(error, KeyboardInterrupt):
            record_failure(args, error)
        raise


if __name__ == "__main__":
    main()
