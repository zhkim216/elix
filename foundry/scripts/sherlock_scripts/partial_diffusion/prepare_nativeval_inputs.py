#!/usr/bin/env python3
"""Stage nativeval role CIFs and prepare one RFD3 partial-diffusion JSON per condition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from rfd3.inference.input_parsing import DesignInputSpecification
from rfd3.inference.parsing import InputSelection
from rfd3.transforms.util_transforms import assign_types_
from rfd3.utils.inference import inference_load_


EXPECTED_FOUNDRY_VERSION = "0.2.0"
DEFAULT_SAMPLING_INPUTS = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/sampling_inputs.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/nativeval/ensembles"
)
EXPECTED_SEMANTIC_CONTRACT = "elix_native_stage1_structure_v1"
SEQUENCE_MODES = ("fixed", "unfixed")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        overwrite=overwrite,
    )


def atomic_copy(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to replace existing file without --overwrite: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def derive_role_sample_id(row: dict[str, str]) -> str:
    binder = json.loads(row["binder_pn_unit_iids"])
    context = json.loads(row["context_pn_unit_iids"])
    binder_token = "-".join(map(str, binder))
    context_token = "-".join(map(str, context)) if context else "none"
    return f"{row['pdb_key']}_binder_{binder_token}_context_{context_token}"


def annotation_as_str(atom_array, name: str) -> np.ndarray:
    if name not in atom_array.get_annotation_categories():
        return np.full(len(atom_array), "", dtype=str)
    return np.asarray(atom_array.get_annotation(name), dtype=str)


def load_structure(path: Path):
    atom_array = inference_load_(str(path))["atom_array"]
    if len(atom_array) == 0:
        raise ValueError(f"No atoms parsed from {path}")
    if not np.isfinite(np.asarray(atom_array.coord, dtype=float)).all():
        raise ValueError(f"Non-finite coordinates in {path}")
    return atom_array


def residue_ids_for_chain(atom_array, chain_id: str) -> list[int]:
    chain_mask = atom_array.chain_id.astype(str) == chain_id
    if not np.any(chain_mask):
        raise ValueError(f"No protein atoms found for binder chain {chain_id!r}")
    residue_keys = set(
        zip(
            atom_array.res_id[chain_mask].astype(int),
            annotation_as_str(atom_array, "ins_code")[chain_mask].astype(str),
            strict=True,
        )
    )
    if any(code for _, code in residue_keys):
        raise ValueError(
            f"RFD3 selections do not support insertion codes on binder chain {chain_id!r}"
        )
    residue_ids = [residue_id for residue_id, _ in residue_keys]
    if len(residue_ids) != len(set(residue_ids)):
        raise ValueError(f"Duplicate residue IDs on binder chain {chain_id!r}")
    if min(residue_ids) < 0:
        raise ValueError(f"RFD3 selections require non-negative residue IDs: {residue_ids}")
    return sorted(residue_ids)


def selection_from_residue_ids(chain_id: str, residue_ids: list[int]) -> str:
    ranges: list[tuple[int, int]] = []
    start = previous = residue_ids[0]
    for residue_id in residue_ids[1:]:
        if residue_id == previous + 1:
            previous = residue_id
            continue
        ranges.append((start, previous))
        start = previous = residue_id
    ranges.append((start, previous))
    return ",".join(
        f"{chain_id}{start}" if start == stop else f"{chain_id}{start}-{stop}"
        for start, stop in ranges
    )


def context_identity(atom_array, chain_id: str) -> tuple[int, str, int]:
    mask = atom_array.chain_id.astype(str) == chain_id
    if not np.any(mask):
        raise ValueError(f"No context atoms found for chain {chain_id!r}")
    residue_ids = set(map(int, atom_array.res_id[mask]))
    insertion_codes = set(annotation_as_str(atom_array, "ins_code")[mask].astype(str))
    residue_names = set(atom_array.res_name[mask].astype(str))
    if len(residue_ids) != 1 or len(insertion_codes) != 1 or len(residue_names) != 1:
        raise ValueError(
            "Context must contain exactly one residue: "
            f"chain={chain_id!r}, res_ids={sorted(residue_ids)}, "
            f"ins_codes={sorted(insertion_codes)}, res_names={sorted(residue_names)}"
        )
    insertion_code = next(iter(insertion_codes))
    if insertion_code:
        raise ValueError(
            f"RFD3 selections do not support context insertion code {insertion_code!r}"
        )
    return next(iter(residue_ids)), next(iter(residue_names)), int(mask.sum())


def validate_spec(
    spec: dict[str, Any],
    *,
    atom_array_input,
    binder_chain: str,
    context_chain: str,
    sequence_unfixed: bool,
    expected_context_atoms: int,
) -> None:
    validation_spec = dict(spec)
    validation_spec["atom_array_input"] = atom_array_input
    validation_spec["select_fixed_atoms"] = InputSelection.from_any(
        validation_spec["select_fixed_atoms"], atom_array=atom_array_input
    )
    validation_spec["select_unfixed_sequence"] = InputSelection.from_any(
        validation_spec["select_unfixed_sequence"], atom_array=atom_array_input
    )
    model = DesignInputSpecification.model_construct(**validation_spec)
    model._assign_types_to_input()
    built = model.build()
    binder_mask = built.chain_id.astype(str) == binder_chain
    context_mask = built.chain_id.astype(str) == context_chain
    if not np.any(binder_mask):
        raise ValueError(f"Built RFD3 input lost binder chain {binder_chain!r}")
    if int(context_mask.sum()) != expected_context_atoms:
        raise ValueError(
            f"Built RFD3 input context atom count mismatch: "
            f"{int(context_mask.sum())} != {expected_context_atoms}"
        )
    fixed_coord = built.is_motif_atom_with_fixed_coord.astype(bool)
    if np.any(fixed_coord[binder_mask]) or not np.all(fixed_coord[context_mask]):
        raise ValueError("Ligand-fixed coordinate mask does not match binder/context roles")
    fixed_sequence = built.is_motif_atom_with_fixed_seq.astype(bool)
    if sequence_unfixed and np.any(fixed_sequence[binder_mask]):
        raise ValueError("Sequence-unfixed spec left binder atoms sequence-fixed")
    if not sequence_unfixed and not np.all(fixed_sequence[binder_mask]):
        raise ValueError("Sequence-fixed spec left binder atoms sequence-unfixed")
    if not np.all(fixed_sequence[context_mask]):
        raise ValueError("Ligand sequence must remain fixed")


def condition_name(partial_t: int) -> str:
    return f"partialt{partial_t}"


def prepare(args: argparse.Namespace) -> None:
    if version("rc-foundry") != EXPECTED_FOUNDRY_VERSION:
        raise RuntimeError(
            f"Expected rc-foundry {EXPECTED_FOUNDRY_VERSION}, got {version('rc-foundry')}"
        )
    partial_ts = sorted(set(args.partial_t))
    if not partial_ts or any(value <= 0 for value in partial_ts):
        raise ValueError("--partial-t values must be positive integers")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")

    sampling_rows = load_csv(args.sampling_inputs.resolve())
    staging_rows = load_csv(args.staging_manifest.resolve())
    if not sampling_rows or not staging_rows:
        raise ValueError("No nativeval rows found")

    selected_rows: list[tuple[int, dict[str, str], dict[str, str]]] = []
    selected_source_indices: set[int] = set()
    for staging_row in staging_rows:
        if staging_row.get("semantic_contract") != EXPECTED_SEMANTIC_CONTRACT:
            raise ValueError(
                "Nativeval RFD3 staging requires semantic_contract="
                f"{EXPECTED_SEMANTIC_CONTRACT!r}"
            )
        source_index = int(staging_row["source_index"])
        if source_index < 0 or source_index >= len(sampling_rows):
            raise IndexError(
                f"Staging source_index is outside sampling inputs: {source_index}"
            )
        if source_index in selected_source_indices:
            raise ValueError(f"Duplicate staging source_index: {source_index}")
        selected_source_indices.add(source_index)
        selected_rows.append((source_index, sampling_rows[source_index], staging_row))

    role_ids = [str(row["role_sample_id"]) for row in staging_rows]
    if len(role_ids) != len(set(role_ids)):
        raise ValueError("Staging manifest contains duplicate role_sample_id values")

    output_root = args.output_root.resolve()
    original_dir = output_root / "original_samples"
    manifest_csv = output_root / "prepared_inputs_manifest.csv"
    manifest_json = output_root / "preparation_manifest.json"
    index_path = output_root / "input_index.txt"
    array_index_path = output_root / "array_index.txt"

    managed_outputs = [manifest_csv, manifest_json, index_path, array_index_path]
    if not args.overwrite:
        existing = [path for path in managed_outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Preparation outputs already exist; use --overwrite to replace them: "
                + ", ".join(map(str, existing))
            )

    records: list[dict[str, Any]] = []
    index_lines: list[str] = []
    inputs_by_partial_t: dict[int, list[str]] = {partial_t: [] for partial_t in partial_ts}
    for source_index, sampling_row, staging_row in selected_rows:
        role_id = str(staging_row["role_sample_id"])
        expected_role_id = derive_role_sample_id(sampling_row)
        if staging_row["source_role_sample_id"] != expected_role_id:
            raise ValueError(
                f"Role identity mismatch at row {source_index}: "
                f"sampling={expected_role_id!r}, "
                f"staging_source={staging_row['source_role_sample_id']!r}"
            )
        for sampling_key, staging_key in (
            ("pdb_key", "source_pdb_key"),
            ("binder_pn_unit_iids", "source_binder_pn_unit_iids"),
            ("context_pn_unit_iids", "source_context_pn_unit_iids"),
        ):
            if sampling_row[sampling_key] != staging_row[staging_key]:
                raise ValueError(
                    f"Sampling/staging mismatch for {role_id}: "
                    f"{sampling_key} != {staging_key}"
                )
        if staging_row["validation_ok"] != "True":
            raise ValueError(f"Upstream semantic staging did not validate: {role_id}")

        semantic_source = Path(staging_row["semantic_cif_path"]).resolve()
        semantic_sha256 = sha256_file(semantic_source)
        if semantic_sha256 != staging_row["semantic_cif_sha256"]:
            raise ValueError(f"Semantic CIF digest mismatch for {role_id}")
        original_path = original_dir / f"{role_id}.cif"
        if not original_path.exists() or sha256_file(original_path) != semantic_sha256:
            atomic_copy(semantic_source, original_path, overwrite=args.overwrite)
        if sha256_file(original_path) != semantic_sha256:
            raise ValueError(f"Staged original digest mismatch for {role_id}")

        structure = load_structure(original_path)
        binder_chain = str(staging_row["binder_chain_id"])
        context_chain = str(staging_row["context_chain_id"])
        if len(binder_chain) != 1 or len(context_chain) != 1:
            raise ValueError(f"RFD3 requires one-character chain IDs: {role_id}")
        if binder_chain == context_chain:
            raise ValueError(f"Binder/context chains overlap for {role_id}")
        binder_residue_ids = residue_ids_for_chain(structure, binder_chain)
        if len(binder_residue_ids) != int(staging_row["protein_length"]):
            raise ValueError(f"Protein CA count does not match staging manifest for {role_id}")
        binder_selector = selection_from_residue_ids(binder_chain, binder_residue_ids)
        binder_ca_count = int(
            np.sum(
                (structure.chain_id.astype(str) == binder_chain)
                & (structure.atom_name.astype(str) == "CA")
            )
        )
        if binder_ca_count < 3:
            raise ValueError(f"Need at least three binder CA atoms for alignment: {role_id}")
        context_res_id, context_ccd, context_atom_count = context_identity(
            structure, context_chain
        )
        context_ccds = json.loads(staging_row["context_ccd_codes"])
        if context_ccds != [context_ccd]:
            raise ValueError(
                f"Context CCD mismatch for {role_id}: parsed={context_ccd}, "
                f"manifest={context_ccds}"
            )
        context_selector = f"{context_chain}{context_res_id}"

        typed_structure = assign_types_(structure.copy())
        typed_context_mask = typed_structure.chain_id.astype(str) == context_chain
        context_is_polymer = bool(
            np.any(
                typed_structure.is_protein[typed_context_mask]
                | typed_structure.is_dna[typed_context_mask]
                | typed_structure.is_rna[typed_context_mask]
            )
        )
        # (JH) fixed: The shared RFD3 classifier now honors AtomWorks
        # chain_type=NON_POLYMER. Fail closed if a stale runtime still promotes a
        # context ligand to a polymer instead of changing its real CCD identity.
        if context_is_polymer:
            raise ValueError(
                f"RFD3 classified context {role_id}:{context_selector} ({context_ccd}) "
                "as a polymer despite its AtomWorks non-polymer annotation"
            )
        rfd3_input_path = original_path
        rfd3_context_ccd = context_ccd
        rfd3_input_sha256 = sha256_file(rfd3_input_path)
        rfd3_structure = load_structure(rfd3_input_path)

        sequence_mode = args.sequence_mode
        sequence_unfixed = sequence_mode == "unfixed"
        validated_spec = False
        for partial_t in partial_ts:
            condition = condition_name(partial_t)
            input_json = output_root / condition / "inputs" / f"{role_id}.json"
            spec = {
                "input": str(rfd3_input_path),
                "ligand": context_selector,
                "partial_t": float(partial_t),
                "select_fixed_atoms": {context_selector: "ALL"},
                "select_unfixed_sequence": binder_selector if sequence_unfixed else False,
                "extra": {
                    "schema_version": 1,
                    "role_sample_id": role_id,
                    "condition": condition,
                    "sequence_unfixed": sequence_unfixed,
                    "ligand_fixed": True,
                    "binder_chain_id": binder_chain,
                    "context_chain_id": context_chain,
                    "context_res_id": context_res_id,
                    "context_ccd": context_ccd,
                    "rfd3_context_ccd": rfd3_context_ccd,
                    "ccd_alias": None,
                    "original_cif": str(original_path),
                    "original_cif_sha256": semantic_sha256,
                    "rfd3_input_cif_sha256": rfd3_input_sha256,
                },
            }
            if not validated_spec:
                validate_spec(
                    spec,
                    atom_array_input=rfd3_structure,
                    binder_chain=binder_chain,
                    context_chain=context_chain,
                    sequence_unfixed=sequence_unfixed,
                    expected_context_atoms=context_atom_count,
                )
                validated_spec = True
            payload = {"partial_diffusion": spec}
            atomic_write_json(input_json, payload, overwrite=args.overwrite)
            input_json_path = str(input_json.resolve())
            input_json_sha256 = sha256_file(input_json)
            index_lines.append(input_json_path)
            inputs_by_partial_t[partial_t].append(input_json_path)
            records.append(
                {
                    "source_index": source_index,
                    "role_sample_id": role_id,
                    "source_pdb_key": staging_row["source_pdb_key"],
                    "condition": condition,
                    "partial_t": partial_t,
                    "sequence_mode": sequence_mode,
                    "sequence_unfixed": sequence_unfixed,
                    "ligand_fixed": True,
                    "binder_chain_id": binder_chain,
                    "context_chain_id": context_chain,
                    "context_res_id": context_res_id,
                    "context_ccd": context_ccd,
                    "rfd3_context_ccd": rfd3_context_ccd,
                    "ccd_alias": "",
                    "binder_selector": binder_selector,
                    "context_selector": context_selector,
                    "protein_residue_count": len(binder_residue_ids),
                    "protein_ca_count": binder_ca_count,
                    "context_atom_count": context_atom_count,
                    "original_cif": str(original_path),
                    "original_cif_sha256": semantic_sha256,
                    "rfd3_input_cif": str(rfd3_input_path),
                    "rfd3_input_cif_sha256": rfd3_input_sha256,
                    "input_json": input_json_path,
                    "input_json_sha256": input_json_sha256,
                    "consumer_validation_ok": True,
                }
            )

    fieldnames = list(records[0])
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    atomic_write_text(manifest_csv, buffer.getvalue(), overwrite=args.overwrite)
    atomic_write_text(index_path, "\n".join(index_lines) + "\n", overwrite=args.overwrite)

    expected_record_count = len(selected_rows) * len(partial_ts)
    if len(records) != expected_record_count:
        raise ValueError(f"Prepared record count mismatch: {len(records)} != {expected_record_count}")
    if args.num_shards > len(selected_rows):
        raise ValueError(
            f"Cannot split {len(selected_rows)} selected rows into {args.num_shards} non-empty shards"
        )
    array_index_lines: list[str] = []
    shard_sizes: list[int] = []
    for partial_t in partial_ts:
        partial_inputs = inputs_by_partial_t[partial_t]
        if len(partial_inputs) != len(selected_rows):
            raise ValueError(f"Unexpected input count for partial_t={partial_t}")
        shards_dir = output_root / condition_name(partial_t) / "inputs" / "shards"
        base_size, remainder = divmod(len(partial_inputs), args.num_shards)
        start = 0
        for shard_index in range(args.num_shards):
            shard_size = base_size + int(shard_index < remainder)
            stop = start + shard_size
            shard_path = shards_dir / f"shard{shard_index:02d}.txt"
            atomic_write_text(
                shard_path,
                "\n".join(partial_inputs[start:stop]) + "\n",
                overwrite=args.overwrite,
            )
            array_index_lines.append(str(shard_path.resolve()))
            shard_sizes.append(shard_size)
            start = stop
    atomic_write_text(
        array_index_path,
        "\n".join(array_index_lines) + "\n",
        overwrite=args.overwrite,
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rc_foundry_version": version("rc-foundry"),
        "sampling_inputs": str(args.sampling_inputs.resolve()),
        "sampling_inputs_sha256": sha256_file(args.sampling_inputs.resolve()),
        "staging_manifest": str(args.staging_manifest.resolve()),
        "staging_manifest_sha256": sha256_file(args.staging_manifest.resolve()),
        "output_root": str(output_root),
        "original_samples_dir": str(original_dir),
        "prepared_inputs_index": str(index_path),
        "array_index": str(array_index_path),
        "prepared_inputs_manifest": str(manifest_csv),
        "partial_t_values": partial_ts,
        "sequence_modes": [args.sequence_mode],
        "ligand_fixed": True,
        "n_sampling_rows": len(sampling_rows),
        "n_source_rows": len(selected_rows),
        "n_role_sample_ids": len(role_ids),
        "n_original_samples": len(role_ids),
        "n_alias_inputs": 0,
        "n_prepared_inputs": len(records),
        "n_shards_per_partial_t": args.num_shards,
        "n_inputs_per_shard": sorted(set(shard_sizes)),
        "n_array_tasks": len(array_index_lines),
        "diagnostics": {
            "missing": 0,
            "skipped": 0,
            "unmatched": 0,
            "invalid": 0,
            "duplicate_role_ids": 0,
        },
    }
    atomic_write_json(manifest_json, summary, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-inputs", type=Path, default=DEFAULT_SAMPLING_INPUTS)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--partial-t", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--sequence-mode", choices=SEQUENCE_MODES, default="unfixed")
    parser.add_argument("--num-shards", type=int, default=18)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    prepare(parse_args())


if __name__ == "__main__":
    main()
