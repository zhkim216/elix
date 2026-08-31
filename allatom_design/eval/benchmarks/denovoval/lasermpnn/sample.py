#!/usr/bin/env python3
"""Deterministic, manifest-producing LASErMPNN sampler for denovoval.

This file intentionally depends only on the NISE environment (ProDy, NumPy,
PyTorch).  It can therefore be launched directly with the LASErMPNN Python.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import platform
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import prody as pr
import torch


NISE_ROOT = Path("/home/users/zhkim216/code/NISE")
DEFAULT_MANIFEST = Path(
    "/scratch/users/zhkim216/out_dir/benchmarks/denovoval/lasermpnn/"
    "staging/staging_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/scratch/users/zhkim216/out_dir/benchmarks/denovoval/lasermpnn/sampling"
)
DEFAULT_WEIGHTS = NISE_ROOT / "LASErMPNN/model_weights/laser_weights_0p1A_nothing_heldout.pt"
EXPECTED_WEIGHTS_SHA256 = "304fe02a4807c310bdd9d68c988ae87619da3cf2025d5c223fb31030aa411173"
SEQUENCE_TEMP = 1.0e-6
FIRST_SHELL_SEQUENCE_TEMP = 1.0e-6
CHI_TEMP = 1.0e-6
SEQ_MIN_P = 0.0
CHI_MIN_P = 0.0
DISABLED_RESIDUES = ("X",)
ALA_BUDGET = 4
GLY_BUDGET = 0
FS_CA_DISTANCE = 10.0
FS_BURIAL_ALPHA = 9.0


def _prepare_cache() -> None:
    user = os.environ.get("USER", "user")
    cache = Path(os.environ.setdefault("KEOPS_CACHE_FOLDER", f"/tmp/{user}_keops"))
    cache.mkdir(parents=True, exist_ok=True)
    build = "_".join(platform.uname()[:3]) + f"_p{sys.version.split()[0]}"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        build += f"_CUDA_VISIBLE_DEVICES_{visible_devices.replace(',', '_')}"
    (cache / build).mkdir(parents=True, exist_ok=True)
    Path(os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/{user}_xdg")).mkdir(
        parents=True, exist_ok=True
    )


_prepare_cache()
if str(NISE_ROOT) not in sys.path:
    sys.path.insert(0, str(NISE_ROOT))

from LASErMPNN.run_batch_inference import _run_inference, output_protein_structure  # noqa: E402
from LASErMPNN.run_inference import load_model_from_parameter_dict  # noqa: E402
from allatom_design.eval.benchmarks.denovoval.lasermpnn.sampling_provenance import (  # noqa: E402
    decoding_order_sha256,
    sample_seed,
    serialize_decoding_order,
)


OUTPUT_COLUMNS = [
    "source_index", "source_sample_id", "kind", "protein_length", "ccd_code",
    "pdb_ccd_code", "protein_chain_id", "ligand_chain_id",
    "transport_ligand_chain_id", "transport_ligand_res_id",
    "transport_ligand_res_name", "atom_mapping_path", "atom_mapping_sha256",
    "input_pdb_path", "input_pdb_sha256", "design_index", "designed_sample_id", "sampled_pdb_path",
    "sampled_pdb_sha256", "protein_sequence", "laser_nll", "laser_bs_nll",
    "decoding_order_local", "decoding_order_sha256", "decoding_order_length",
    "model_ligand_resnames", "model_ligand_atom_count", "model_cap_residue_count",
    "output_ligand_atom_count", "output_ligand_coordinate_max_error",
    "model_weights_path", "model_weights_sha256", "sample_seed", "sequence_temp",
    "first_shell_sequence_temp", "chi_temp", "seq_min_p", "chi_min_p",
    "disabled_residues", "ala_budget", "gly_budget", "status", "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample denovoval with LASErMPNN.")
    parser.add_argument("--staging-manifest-csv", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-dir", type=Path, default=None)
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--model-weights-path", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-seqs-per-pdb", type=int, default=2)
    parser.add_argument("--designs-per-batch", type=int, default=2)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--ccd-code", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_shard_id(value: int | None) -> int:
    if value is not None:
        return value
    return int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "source_index", "sample_id", "kind", "protein_length", "ccd_code",
            "pdb_ccd_code", "protein_chain_id", "ligand_chain_id",
            "transport_ligand_chain_id", "transport_ligand_res_id",
            "transport_ligand_res_name", "atom_mapping_path", "atom_mapping_sha256",
            "protonated_pdb_path", "protonated_pdb_sha256", "status",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        rows = list(reader)
    if any(row["status"] != "prepared" for row in rows):
        raise ValueError(f"{path} contains unprepared rows")
    duplicates = [key for key, count in Counter(row["sample_id"] for row in rows).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sample IDs: {duplicates[:20]}")
    return rows


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, str]], int]:
    shard_id = resolve_shard_id(args.shard_id)
    if args.num_shards < 1 or not 0 <= shard_id < args.num_shards:
        raise ValueError(f"Invalid shard {shard_id}/{args.num_shards}")
    sample_ids = set(args.sample_id or [])
    ccd_codes = set(args.ccd_code or [])
    selected = [
        row for row in rows
        if (not sample_ids or row["sample_id"] in sample_ids)
        and (not ccd_codes or row["ccd_code"] in ccd_codes)
    ]
    missing_ids = sample_ids - {row["sample_id"] for row in selected}
    missing_ccds = ccd_codes - {row["ccd_code"] for row in selected}
    if missing_ids or missing_ccds:
        raise ValueError(f"Missing sample IDs/CCDs: {sorted(missing_ids)}, {sorted(missing_ccds)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]
    return [row for index, row in enumerate(selected) if index % args.num_shards == shard_id], shard_id


def atomic_write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def mean_neg_log10(values: torch.Tensor) -> str:
    if values.numel() == 0:
        return "nan"
    return f"{float((-torch.log10(values)).detach().cpu().mean()):.8g}"


def model_hierview(input_path: Path, ligand_chain_id: str) -> pr.HierView:
    """Hide the N/CA/C/O naming signature from LASEr's protein-track heuristic.

    Only the in-memory model copy is renamed.  Elements and coordinates are
    unchanged, and the serialized sampled PDB receives the immutable original
    ligand.  This prevents synthetic CAP creation without modifying NISE.
    """
    atoms = pr.parsePDB(str(input_path))
    if not isinstance(atoms, pr.AtomGroup):
        raise ValueError(f"ProDy failed to parse {input_path}")
    atoms = atoms.copy()
    ligand = atoms.select(f"chain {ligand_chain_id}")
    if ligand is None:
        raise ValueError(f"No ligand chain {ligand_chain_id} in {input_path}")
    signature = {"N", "CA", "C", "O"}
    names = list(map(str, atoms.getNames()))
    ligand_indices = list(map(int, ligand.getIndices()))
    if signature.issubset({names[index] for index in ligand_indices}):
        for local_index, atom_index in enumerate(ligand_indices):
            names[atom_index] = f"Q{local_index:03d}"
        atoms.setNames(names)
    # NISE's get_all_gly_protein() accesses the cached ProDy protein flag
    # directly, so force its computation after the in-memory rename.
    if atoms.protein is None:
        raise ValueError(f"No protein atoms in {input_path}")
    return atoms.getHierView()


def base_row(
    source: dict[str, str],
    design_index: int,
    output: Path,
    weights_path: Path,
    weights_sha: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "source_index": source["source_index"],
        "source_sample_id": source["sample_id"],
        "kind": source["kind"],
        "protein_length": source["protein_length"],
        "ccd_code": source["ccd_code"],
        "pdb_ccd_code": source["pdb_ccd_code"],
        "protein_chain_id": source["protein_chain_id"],
        "ligand_chain_id": source["ligand_chain_id"],
        "transport_ligand_chain_id": source["transport_ligand_chain_id"],
        "transport_ligand_res_id": source["transport_ligand_res_id"],
        "transport_ligand_res_name": source["transport_ligand_res_name"],
        "atom_mapping_path": source["atom_mapping_path"],
        "atom_mapping_sha256": source["atom_mapping_sha256"],
        "input_pdb_path": source["protonated_pdb_path"],
        "input_pdb_sha256": source["protonated_pdb_sha256"],
        "design_index": design_index,
        "designed_sample_id": f"lasermpnn_{source['sample_id']}_{design_index}",
        "sampled_pdb_path": str(output),
        "sampled_pdb_sha256": "", "protein_sequence": "", "laser_nll": "nan",
        "laser_bs_nll": "nan", "decoding_order_local": "",
        "decoding_order_sha256": "", "decoding_order_length": "",
        "model_ligand_resnames": "",
        "model_ligand_atom_count": "", "model_cap_residue_count": "",
        "output_ligand_atom_count": "", "output_ligand_coordinate_max_error": "",
        "model_weights_path": str(weights_path), "model_weights_sha256": weights_sha,
        "sample_seed": seed, "sequence_temp": SEQUENCE_TEMP,
        "first_shell_sequence_temp": FIRST_SHELL_SEQUENCE_TEMP, "chi_temp": CHI_TEMP,
        "seq_min_p": SEQ_MIN_P, "chi_min_p": CHI_MIN_P,
        "disabled_residues": ",".join(DISABLED_RESIDUES), "ala_budget": ALA_BUDGET,
        "gly_budget": GLY_BUDGET, "status": "failed", "error": "",
    }


def normalized_decoding_order(batch_data: Any, batch_index: int) -> list[int]:
    """Return one design's decoding order as local zero-based residue indices."""
    if batch_data.decoding_order is None:
        raise ValueError("LASErMPNN did not retain a decoding order")
    batch_global_indices = (
        (batch_data.batch_indices == batch_index).nonzero().flatten().detach().cpu().tolist()
    )
    global_to_local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(batch_global_indices)
    }
    raw_order = batch_data.decoding_order[batch_index]
    finite_order = raw_order[~raw_order.isnan()].long().detach().cpu().tolist()
    if len(finite_order) != len(batch_global_indices):
        raise ValueError(
            "Decoding order length differs from the design residue count: "
            f"{len(finite_order)} != {len(batch_global_indices)}"
        )
    if set(map(int, finite_order)) != set(map(int, batch_global_indices)):
        raise ValueError("Decoding order is not a permutation of design residues")
    return [global_to_local[int(global_index)] for global_index in finite_order]


def write_sample(
    *, out_protein: pr.Atomic, input_ligand: pr.Atomic, output: Path,
    protein_chain: str, ligand_chain: str, pdb_ccd: str,
) -> dict[str, Any]:
    combined = out_protein + input_ligand.copy()
    combined.setSerials(np.arange(1, len(combined) + 1, dtype=int))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.{os.getpid()}{output.suffix}")
    pr.writePDB(str(temporary), combined)
    os.replace(temporary, output)
    observed = pr.parsePDB(str(output))
    protein = observed.select(f"chain {protein_chain}")
    ligand = observed.select(f"chain {ligand_chain}")
    if protein is None or ligand is None:
        raise ValueError("Sampled PDB lost protein or ligand")
    codes = sorted(set(map(str, ligand.getResnames())))
    if codes != [pdb_ccd] or "CAP" in codes:
        raise ValueError(f"Sampled ligand codes {codes} != {[pdb_ccd]}")
    if len(ligand) != len(input_ligand):
        raise ValueError("Sampled PDB ligand atom count differs from input")
    error = float(np.max(np.abs(ligand.getCoords() - input_ligand.getCoords())))
    if error > 0.00051:
        raise ValueError(f"Sampled ligand coordinate error {error}")
    ca = protein.select("name CA")
    if ca is None:
        raise ValueError("Sampled protein has no CA atoms")
    return {
        "sampled_pdb_sha256": sha256_file(output),
        "protein_sequence": str(ca.getSequence()),
        "output_ligand_atom_count": len(ligand),
        "output_ligand_coordinate_max_error": error,
    }


def sample_one(source: dict[str, str], model: Any, params: dict[str, Any], args: argparse.Namespace, samples_dir: Path, weights_sha: str) -> list[dict[str, Any]]:
    input_path = Path(source["protonated_pdb_path"])
    if sha256_file(input_path) != source["protonated_pdb_sha256"]:
        raise ValueError(f"Prepared input digest mismatch: {input_path}")
    mapping_path = Path(source["atom_mapping_path"])
    if sha256_file(mapping_path) != source["atom_mapping_sha256"]:
        raise ValueError(f"Atom-mapping sidecar digest mismatch: {mapping_path}")
    transport_chain = source["transport_ligand_chain_id"]
    parsed = pr.parsePDB(str(input_path))
    input_ligand = parsed.select(f"chain {transport_chain}")
    if input_ligand is None:
        raise ValueError(f"No ligand chain in {input_path}")
    input_ligand = input_ligand.copy()
    seed = sample_seed(source["sample_id"], args.base_seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hv = model_hierview(input_path, transport_chain)
    output_rows: list[dict[str, Any]] = []
    first = 1
    while first <= args.num_seqs_per_pdb:
        batch_size = min(args.designs_per_batch, args.num_seqs_per_pdb - first + 1)
        sampled, coords, nh_coords, probs, batch_data, data_list = _run_inference(
            model, params, hv, batch_size,
            sequence_temp=SEQUENCE_TEMP, first_shell_sequence_temp=FIRST_SHELL_SEQUENCE_TEMP,
            chi_temp=CHI_TEMP, seq_min_p=SEQ_MIN_P, chi_min_p=CHI_MIN_P,
            disable_pbar=True, disabled_residues_list=list(DISABLED_RESIDUES),
            noncanonical_aa_ligand=False, fs_calc_ca_distance=FS_CA_DISTANCE,
            fs_calc_burial_hull_alpha_value=FS_BURIAL_ALPHA,
            fs_no_calc_burial=False, ala_budget=ALA_BUDGET, gly_budget=GLY_BUDGET,
        )
        data = data_list[0]
        model_resnames = [str(identifier.resname) for identifier in data.ligand_info.residue_identifiers]
        if model_resnames != [source["transport_ligand_res_name"]] or "CAP" in model_resnames:
            raise ValueError(
                "Model ligand track "
                f"{model_resnames} != {[source['transport_ligand_res_name']]}"
            )
        model_ligand_atoms = len(data.ligand_info.atom_coords)
        if model_ligand_atoms != len(input_ligand):
            raise ValueError(
                f"Model ligand atoms {model_ligand_atoms} != input {len(input_ligand)}"
            )
        for batch_index in range(batch_size):
            design_index = first + batch_index
            batch_mask = batch_data.batch_indices == batch_index
            batch_probs = probs[batch_mask]
            binding_mask = batch_data.first_shell_ligand_contact_mask[batch_mask]
            out_protein = output_protein_structure(
                coords[batch_mask], sampled.sampled_sequence_indices[batch_mask],
                data.residue_identifiers, nh_coords[batch_mask], batch_probs,
            )
            output = samples_dir / f"{source['sample_id']}_{design_index}.pdb"
            row = base_row(
                source, design_index, output, args.model_weights_path, weights_sha, seed
            )
            decoding_order = normalized_decoding_order(batch_data, batch_index)
            serialized_order = serialize_decoding_order(decoding_order)
            stats = write_sample(
                out_protein=out_protein, input_ligand=input_ligand, output=output,
                protein_chain=source["protein_chain_id"], ligand_chain=transport_chain,
                pdb_ccd=source["transport_ligand_res_name"],
            )
            if len(stats["protein_sequence"]) != int(source["protein_length"]):
                raise ValueError("Sampled protein sequence length differs")
            row.update(stats)
            row.update(
                {
                    "laser_nll": mean_neg_log10(batch_probs),
                    "laser_bs_nll": mean_neg_log10(batch_probs[binding_mask]),
                    "decoding_order_local": serialized_order,
                    "decoding_order_sha256": decoding_order_sha256(serialized_order),
                    "decoding_order_length": len(decoding_order),
                    "model_ligand_resnames": ";".join(model_resnames),
                    "model_ligand_atom_count": model_ligand_atoms,
                    "model_cap_residue_count": 0,
                    "status": "success",
                }
            )
            output_rows.append(row)
        first += batch_size
    order_digests = [str(row["decoding_order_sha256"]) for row in output_rows]
    if len(order_digests) > 1 and len(set(order_digests)) != len(order_digests):
        raise ValueError(
            "Multiple designs for one input reused the same autoregressive decoding order"
        )
    return output_rows


def main() -> None:
    args = parse_args()
    if args.num_seqs_per_pdb < 1 or args.designs_per_batch < 1:
        raise ValueError("Design counts must be positive")
    rows, shard_id = select_rows(read_rows(args.staging_manifest_csv), args)
    samples_dir = args.samples_dir or (args.output_root / "pdbs")
    manifest = args.manifest_csv or (args.output_root / "manifests" / f"shard_{shard_id:02d}.csv")
    print(f"rows={len(rows)} shard={shard_id}/{args.num_shards} designs={args.num_seqs_per_pdb}")
    print(f"sampling inputs={args.staging_manifest_csv}; outputs={samples_dir}")
    print(
        f"weights={args.model_weights_path}; seq/fs/chi temp={SEQUENCE_TEMP}; "
        f"disabled={','.join(DISABLED_RESIDUES)}; min-p={SEQ_MIN_P}/{CHI_MIN_P}"
    )
    if args.dry_run:
        return
    weights_sha = sha256_file(args.model_weights_path)
    if weights_sha != EXPECTED_WEIGHTS_SHA256:
        raise ValueError(f"Weights digest {weights_sha} != {EXPECTED_WEIGHTS_SHA256}")
    model, params = load_model_from_parameter_dict(str(args.model_weights_path), args.device)
    output_rows: list[dict[str, Any]] = []
    for index, source in enumerate(rows, start=1):
        print(f"[{index}/{len(rows)}] {source['sample_id']} {source['ccd_code']}")
        try:
            existing = [samples_dir / f"{source['sample_id']}_{i}.pdb" for i in range(1, args.num_seqs_per_pdb + 1)]
            if not args.overwrite and any(path.exists() for path in existing):
                raise ValueError("Existing partial/complete outputs require --overwrite")
            output_rows.extend(sample_one(source, model, params, args, samples_dir, weights_sha))
        except Exception as exc:
            traceback.print_exc()
            for design_index in range(1, args.num_seqs_per_pdb + 1):
                output = samples_dir / f"{source['sample_id']}_{design_index}.pdb"
                row = base_row(
                    source,
                    design_index,
                    output,
                    args.model_weights_path,
                    weights_sha,
                    sample_seed(source["sample_id"], args.base_seed),
                )
                row["error"] = f"{type(exc).__name__}: {exc}"
                output_rows.append(row)
        atomic_write_manifest(manifest, output_rows)
    failed = [row for row in output_rows if row["status"] != "success"]
    expected = len(rows) * args.num_seqs_per_pdb
    if len(output_rows) != expected or failed:
        raise RuntimeError(
            f"Sampling incomplete: rows={len(output_rows)}/{expected}, failed={len(failed)}; {manifest}"
        )


if __name__ == "__main__":
    main()
