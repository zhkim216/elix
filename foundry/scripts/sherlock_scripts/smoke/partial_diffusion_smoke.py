#!/usr/bin/env python3
"""Prepare, align, and validate the ATP partial-diffusion smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.utils.geometry import align_atom_arrays
from biotite.structure import AtomArray
from rfd3.constants import SAVED_CONDITIONING_ANNOTATIONS
from rfd3.model.inference_sampler import SampleDiffusionWithMotif
from rfd3.utils.inference import inference_load_


PARTIAL_T = 5.0
EXPECTED_SOURCE_SHA256 = (
    "42dd64c63413d4a4fc793dea2abc24de8370011147cccd27f2c088aa590f642a"
)
EXPECTED_CHECKPOINT_SIZE = 2_690_316_669
EXPECTED_PROTEIN_CA_COUNT = 165
EXPECTED_ATP_HEAVY_ATOM_COUNT = 31
ATP_FIXED_DISTANCE_TOLERANCE = 0.1

CONDITIONS: dict[str, dict[str, Any]] = {
    "protein_diffusion_sequence_fixed": {
        "select_fixed_atoms": {"ATP": "ALL"},
        "select_unfixed_sequence": False,
        "diffuse_context": False,
        "sequence_unfixed": False,
    },
    "protein_atp_diffusion_sequence_fixed": {
        "select_fixed_atoms": False,
        "select_unfixed_sequence": False,
        "diffuse_context": True,
        "sequence_unfixed": False,
    },
    "protein_diffusion_sequence_unfixed": {
        "select_fixed_atoms": {"ATP": "ALL"},
        "select_unfixed_sequence": "A1-165",
        "diffuse_context": False,
        "sequence_unfixed": True,
    },
    "protein_atp_diffusion_sequence_unfixed": {
        "select_fixed_atoms": False,
        "select_unfixed_sequence": "A1-165",
        "diffuse_context": True,
        "sequence_unfixed": True,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise FileExistsError(f"Existing JSON differs; set overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def require_partial_t_suffix(path: Path, suffix: str) -> None:
    name = path.name
    if suffix == ".cif.gz":
        if not name.endswith(suffix):
            raise ValueError(f"Expected {suffix} output: {path}")
        stem = name[: -len(suffix)]
    else:
        if path.suffix != suffix:
            raise ValueError(f"Expected {suffix} output: {path}")
        stem = path.stem
    if not stem.endswith("_partialt5"):
        raise ValueError(f"Output stem must end in _partialt5: {path}")


def load_structure(path: Path) -> AtomArray:
    structure = inference_load_(str(path))["atom_array"]
    if len(structure) == 0:
        raise ValueError(f"No atoms parsed from {path}")
    if not np.isfinite(np.asarray(structure.coord, dtype=float)).all():
        raise ValueError(f"Non-finite coordinates in {path}")
    return structure


def _annotation_as_str(structure: AtomArray, name: str) -> np.ndarray:
    if name not in structure.get_annotation_categories():
        return np.full(len(structure), "", dtype=str)
    return np.asarray(structure.get_annotation(name), dtype=str)


def atom_identity_counter(structure: AtomArray) -> Counter[tuple[Any, ...]]:
    ins_code = _annotation_as_str(structure, "ins_code")
    return Counter(
        zip(
            structure.chain_id.astype(str),
            structure.res_id.astype(int),
            ins_code,
            structure.res_name.astype(str),
            structure.atom_name.astype(str),
            structure.element.astype(str),
            structure.hetero.astype(bool),
            strict=True,
        )
    )


def protein_ca_indices(
    structure: AtomArray,
    chain_id: str,
) -> dict[tuple[str, int, str], int]:
    ins_code = _annotation_as_str(structure, "ins_code")
    mask = (
        (structure.chain_id.astype(str) == chain_id)
        & (structure.atom_name.astype(str) == "CA")
    )
    indices: dict[tuple[str, int, str], int] = {}
    for index in np.flatnonzero(mask):
        key = (chain_id, int(structure.res_id[index]), str(ins_code[index]))
        if key in indices:
            raise ValueError(f"Duplicate protein CA key: {key}")
        indices[key] = int(index)
    if not indices:
        raise ValueError(f"No CA atoms found for protein chain {chain_id!r}")
    return indices


def atp_heavy_atom_coords(structure: AtomArray) -> dict[str, np.ndarray]:
    mask = (
        (structure.res_name.astype(str) == "ATP")
        & (np.char.upper(structure.element.astype(str)) != "H")
    )
    coords: dict[str, np.ndarray] = {}
    for name, coord in zip(
        structure.atom_name[mask].astype(str),
        structure.coord[mask],
        strict=True,
    ):
        if name in coords:
            raise ValueError(f"Duplicate ATP heavy atom name: {name}")
        coords[name] = np.asarray(coord, dtype=float)
    if len(coords) != EXPECTED_ATP_HEAVY_ATOM_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ATP_HEAVY_ATOM_COUNT} ATP heavy atoms, got {len(coords)}"
        )
    return coords


def distance_matrix_max_error(
    reference: dict[str, np.ndarray],
    observed: dict[str, np.ndarray],
) -> float:
    if set(reference) != set(observed):
        raise ValueError(
            "ATP heavy atom identity mismatch: "
            f"missing={sorted(set(reference) - set(observed))}, "
            f"extra={sorted(set(observed) - set(reference))}"
        )
    names = sorted(reference)
    reference_coords = np.stack([reference[name] for name in names])
    observed_coords = np.stack([observed[name] for name in names])
    reference_distances = np.linalg.norm(
        reference_coords[:, None, :] - reference_coords[None, :, :], axis=-1
    )
    observed_distances = np.linalg.norm(
        observed_coords[:, None, :] - observed_coords[None, :, :], axis=-1
    )
    return float(np.max(np.abs(reference_distances - observed_distances)))


def prepare_specs(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    require_partial_t_suffix(output, ".json")
    observed_sha256 = sha256_file(source)
    if observed_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"Semantic CIF SHA256 mismatch: {observed_sha256} != {EXPECTED_SOURCE_SHA256}"
        )

    payload = {}
    for condition, contract in CONDITIONS.items():
        payload[condition] = {
            "input": str(source),
            "ligand": "ATP",
            "partial_t": PARTIAL_T,
            "select_fixed_atoms": contract["select_fixed_atoms"],
            "select_unfixed_sequence": contract["select_unfixed_sequence"],
            "extra": {
                "condition": condition,
                "diffuse_context": contract["diffuse_context"],
                "sequence_unfixed": contract["sequence_unfixed"],
            },
        }
    atomic_write_json(output, payload, overwrite=args.overwrite)
    print(json.dumps({"status": "prepared", "input_json": str(output), "conditions": list(payload)}))


def runtime_preflight(args: argparse.Namespace) -> None:
    import rfd3

    checkpoint = args.checkpoint.resolve()
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_SIZE:
        raise ValueError(
            f"Checkpoint size mismatch: {checkpoint.stat().st_size} != {EXPECTED_CHECKPOINT_SIZE}"
        )
    package_version = version("rc-foundry")
    source = Path(inspect.getfile(rfd3)).resolve()
    sampler = SampleDiffusionWithMotif()
    full = sampler._construct_inference_noise_schedule(torch.device("cpu"))
    partial = sampler._construct_inference_noise_schedule(
        torch.device("cpu"), torch.tensor(PARTIAL_T)
    )
    if package_version != "0.2.0":
        raise ValueError(f"Expected rc-foundry 0.2.0, got {package_version}")
    if not str(source).startswith("/app/foundry/.venv/"):
        raise ValueError(f"Unexpected rfd3 import source: {source}")
    if len(full) != 200 or len(partial) != 60:
        raise ValueError(f"Unexpected schedules: full={len(full)}, partial={len(partial)}")
    if not math.isclose(float(partial[0]), 4.8102483749, rel_tol=1e-6):
        raise ValueError(f"Unexpected partial_t=5 initial sigma: {float(partial[0])}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the Foundry container")

    report = {
        "status": "ok",
        "rc_foundry_version": package_version,
        "rfd3_source": str(source),
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "cuda_device": torch.cuda.get_device_name(0),
        "full_schedule_points": len(full),
        "partial_schedule_points": len(partial),
        "partial_updates": len(partial) - 1,
        "partial_initial_sigma": float(partial[0]),
        "partial_final_sigma": float(partial[-1]),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def align_prediction(args: argparse.Namespace) -> None:
    contract = CONDITIONS[args.condition]
    reference_path = args.reference.resolve()
    prediction_path = args.prediction.resolve()
    metadata_path = args.prediction_metadata.resolve()
    output_cif = args.output_cif.resolve()
    output_json = args.output_json.resolve()
    require_partial_t_suffix(output_cif, ".cif.gz")
    require_partial_t_suffix(output_json, ".json")
    if output_cif.name[: -len(".cif.gz")] != output_json.stem:
        raise ValueError("Aligned CIF and JSON must have the same stem")
    if output_cif.exists() and not args.overwrite:
        raise FileExistsError(output_cif)
    if output_json.exists() and not args.overwrite:
        raise FileExistsError(output_json)

    if sha256_file(reference_path) != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Unexpected reference CIF: {reference_path}")
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    specification = raw_metadata.get("specification", {})
    if float(specification.get("partial_t", -1.0)) != PARTIAL_T:
        raise ValueError(f"Raw output partial_t mismatch: {specification.get('partial_t')!r}")
    if specification.get("select_fixed_atoms") != contract["select_fixed_atoms"]:
        raise ValueError("Raw output select_fixed_atoms mismatch")
    if specification.get("select_unfixed_sequence") != contract["select_unfixed_sequence"]:
        raise ValueError("Raw output select_unfixed_sequence mismatch")

    reference = load_structure(reference_path)
    prediction = load_structure(prediction_path)
    reference_ca = protein_ca_indices(reference, args.protein_chain)
    prediction_ca = protein_ca_indices(prediction, args.protein_chain)
    if set(reference_ca) != set(prediction_ca):
        raise ValueError(
            "Protein CA key mismatch: "
            f"missing={sorted(set(reference_ca) - set(prediction_ca))}, "
            f"extra={sorted(set(prediction_ca) - set(reference_ca))}"
        )
    keys = sorted(reference_ca)
    if len(keys) != EXPECTED_PROTEIN_CA_COUNT:
        raise ValueError(f"Expected {EXPECTED_PROTEIN_CA_COUNT} CA pairs, got {len(keys)}")
    reference_indices = np.asarray([reference_ca[key] for key in keys], dtype=int)
    prediction_indices = np.asarray([prediction_ca[key] for key in keys], dtype=int)

    if not contract["sequence_unfixed"]:
        reference_names = reference.res_name[reference_indices].astype(str).tolist()
        prediction_names = prediction.res_name[prediction_indices].astype(str).tolist()
        if prediction_names != reference_names:
            raise ValueError("Sequence-fixed output changed protein residue identities")

    reference_atp = atp_heavy_atom_coords(reference)
    prediction_atp = atp_heavy_atom_coords(prediction)
    atp_distance_error = distance_matrix_max_error(reference_atp, prediction_atp)
    if not contract["diffuse_context"] and atp_distance_error > ATP_FIXED_DISTANCE_TOLERANCE:
        raise ValueError(
            f"Fixed ATP geometry changed by {atp_distance_error:.6f} A "
            f"(tolerance {ATP_FIXED_DISTANCE_TOLERANCE})"
        )

    aligned, fitted_ca_rmsd = align_atom_arrays(
        mbl_sele=prediction[prediction_indices],
        tgt_sele=reference[reference_indices],
        mbl_full=prediction,
    )
    fitted_ca_rmsd = float(np.asarray(fitted_ca_rmsd).reshape(-1)[0])
    if not math.isfinite(fitted_ca_rmsd):
        raise ValueError("Non-finite fitted CA RMSD")

    output_cif.parent.mkdir(parents=True, exist_ok=True)
    temporary_cif = output_cif.with_name(
        f".{output_cif.name[: -len('.cif.gz')]}.tmp.{os.getpid()}.cif.gz"
    )
    to_cif_file(
        aligned,
        temporary_cif,
        file_type="cif.gz",
        include_entity_poly=False,
        extra_fields=SAVED_CONDITIONING_ANNOTATIONS,
    )
    roundtrip = load_structure(temporary_cif)
    if atom_identity_counter(roundtrip) != atom_identity_counter(prediction):
        raise ValueError("Aligned CIF writer roundtrip changed atom identities")
    roundtrip_ca = protein_ca_indices(roundtrip, args.protein_chain)
    if set(roundtrip_ca) != set(reference_ca):
        raise ValueError("Aligned CIF roundtrip changed protein CA keys")
    roundtrip_coords = np.stack([roundtrip.coord[roundtrip_ca[key]] for key in keys])
    reference_coords = np.stack([reference.coord[reference_ca[key]] for key in keys])
    direct_ca_rmsd = float(
        np.sqrt(np.mean(np.sum((roundtrip_coords - reference_coords) ** 2, axis=1)))
    )
    centroid_offset = float(
        np.linalg.norm(roundtrip_coords.mean(axis=0) - reference_coords.mean(axis=0))
    )
    if abs(direct_ca_rmsd - fitted_ca_rmsd) > 2e-3:
        raise ValueError(
            f"Aligned CIF RMSD changed on write: fitted={fitted_ca_rmsd}, "
            f"roundtrip={direct_ca_rmsd}"
        )
    if centroid_offset > 2e-3:
        raise ValueError(f"Aligned CA centroid offset too large: {centroid_offset}")
    os.replace(temporary_cif, output_cif)

    raw_metadata["postprocessing"] = {
        "status": "ok",
        "condition": args.condition,
        "partial_t": PARTIAL_T,
        "reference_cif": str(reference_path),
        "reference_sha256": EXPECTED_SOURCE_SHA256,
        "raw_prediction_sha256": sha256_file(prediction_path),
        "aligned_cif": str(output_cif),
        "aligned_cif_sha256": sha256_file(output_cif),
        "container": str(args.container.resolve()),
        "container_sha256": args.container_sha256,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": args.checkpoint_sha256,
        "protein_chain": args.protein_chain,
        "matched_ca_count": len(keys),
        "protein_ca_rmsd_to_reference": fitted_ca_rmsd,
        "roundtrip_direct_ca_rmsd": direct_ca_rmsd,
        "roundtrip_ca_centroid_offset": centroid_offset,
        "prediction_atom_count": len(prediction),
        "roundtrip_atom_count": len(roundtrip),
        "atp_heavy_atom_count": len(prediction_atp),
        "atp_distance_matrix_max_error": atp_distance_error,
        "sequence_unfixed": contract["sequence_unfixed"],
        "diffuse_context": contract["diffuse_context"],
        "atom_identity_roundtrip_ok": True,
    }
    atomic_write_json(output_json, raw_metadata, overwrite=args.overwrite)
    print(json.dumps(raw_metadata["postprocessing"], indent=2, sort_keys=True))


def summarize(args: argparse.Namespace) -> None:
    aligned_dir = args.aligned_dir.resolve()
    preflight = json.loads(args.preflight_json.resolve().read_text(encoding="utf-8"))
    if preflight.get("status") != "ok":
        raise ValueError("Runtime preflight did not pass")
    if preflight.get("full_schedule_points") != 200:
        raise ValueError("Runtime preflight did not observe the 200-point full schedule")
    if preflight.get("partial_schedule_points") != 60:
        raise ValueError("Runtime preflight did not observe the 60-point partial_t=5 schedule")
    if float(preflight.get("partial_initial_sigma", math.inf)) > PARTIAL_T:
        raise ValueError("Runtime preflight partial schedule exceeds partial_t=5")
    records = []
    for condition in CONDITIONS:
        stem = f"{args.sample_id}__{condition}_ca_aligned_partialt5"
        cif_path = aligned_dir / f"{stem}.cif.gz"
        json_path = aligned_dir / f"{stem}.json"
        if not cif_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"Missing aligned pair for {condition}")
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        postprocessing = metadata.get("postprocessing", {})
        if postprocessing.get("status") != "ok":
            raise ValueError(f"Invalid postprocessing status for {condition}")
        if postprocessing.get("aligned_cif_sha256") != sha256_file(cif_path):
            raise ValueError(f"Aligned CIF digest mismatch for {condition}")
        records.append(postprocessing)

    payload = {
        "schema_version": 1,
        "status": "complete",
        "sample_id": args.sample_id,
        "partial_t": PARTIAL_T,
        "n_conditions": len(records),
        "runtime_preflight": preflight,
        "conditions": records,
    }
    atomic_write_json(args.output.resolve(), payload, overwrite=args.overwrite)
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--checkpoint", type=Path, required=True)
    preflight.set_defaults(func=runtime_preflight)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")
    prepare.set_defaults(func=prepare_specs)

    align = subparsers.add_parser("align")
    align.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    align.add_argument("--reference", type=Path, required=True)
    align.add_argument("--prediction", type=Path, required=True)
    align.add_argument("--prediction-metadata", type=Path, required=True)
    align.add_argument("--output-cif", type=Path, required=True)
    align.add_argument("--output-json", type=Path, required=True)
    align.add_argument("--protein-chain", default="A")
    align.add_argument("--container", type=Path, required=True)
    align.add_argument("--container-sha256", required=True)
    align.add_argument("--checkpoint", type=Path, required=True)
    align.add_argument("--checkpoint-sha256", required=True)
    align.add_argument("--overwrite", action="store_true")
    align.set_defaults(func=align_prediction)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--aligned-dir", type=Path, required=True)
    summary.add_argument("--sample-id", required=True)
    summary.add_argument("--preflight-json", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--overwrite", action="store_true")
    summary.set_defaults(func=summarize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
