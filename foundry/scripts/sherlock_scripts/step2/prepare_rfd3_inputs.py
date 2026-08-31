#!/usr/bin/env python3
"""Stage RFD3 diffused-ligand + all-buried inputs for denovoval step2.

For every CCD in the step1 list this writes a ligand CIF and, per requested length, a
single-spec RFD3 JSON that co-diffuses the ligand (``select_fixed_atoms`` empty) and marks
every ligand atom buried (``select_buried`` ALL). The CIF writer branches on atom count:
single-atom metal ions get zeroed coordinates and no bonds (their ideal coord is NaN),
multi-atom ligands keep their CCD ideal coordinates and bonds.

Per-CCD failures are recorded in the manifest and skipped; they never abort the batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.io.utils.ccd import atom_array_from_ccd_code
from atomworks.io.utils.io_utils import to_cif_file

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from constants import (  # type: ignore[import-not-found]
        DEFAULT_LENGTHS,
        FULL_DIFFUSION_BATCH_SIZE,
        FULL_N_BATCHES,
        METAL_FORMAL_CHARGES,
        default_ccd_mirror,
        default_source_data_root,
        default_step1_ccd_list,
    )
else:
    from .constants import (
        DEFAULT_LENGTHS,
        FULL_DIFFUSION_BATCH_SIZE,
        FULL_N_BATCHES,
        METAL_FORMAL_CHARGES,
        default_ccd_mirror,
        default_source_data_root,
        default_step1_ccd_list,
    )

CCD_COLUMN = "ccd_code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ccd-list",
        type=Path,
        default=default_step1_ccd_list(),
        help="Step1 denovoval_ccd_list.tsv (source of the CCD set).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_source_data_root(),
        help="Output root for inputs/ CIFs and rfd3_inputs/ JSONs.",
    )
    parser.add_argument(
        "--ccd-mirror",
        type=Path,
        default=default_ccd_mirror(),
        help="CCD mirror directory used by atomworks.",
    )
    parser.add_argument(
        "--ccd",
        action="append",
        default=None,
        help="Restrict to a CCD code. May be repeated. Defaults to the full step1 list.",
    )
    parser.add_argument(
        "--lengths",
        default=",".join(str(length) for length in DEFAULT_LENGTHS),
        help="Comma-separated fixed lengths, e.g. 150,300.",
    )
    parser.add_argument(
        "--select-buried",
        choices=("all", "enumerate"),
        default="all",
        help="How to mark ligand atoms buried: 'all' shorthand or explicit atom-name list.",
    )
    parser.add_argument(
        "--diffusion-batch-size",
        type=int,
        default=FULL_DIFFUSION_BATCH_SIZE,
        help="Manifest-only expected samples per RFD3 spec (execution passes its own).",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=FULL_N_BATCHES,
        help="Manifest-only expected RFD3 batches per spec.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite ligand CIFs even if they already exist.",
    )
    return parser.parse_args()


def parse_lengths(raw: str) -> list[int]:
    lengths = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not lengths:
        raise ValueError("--lengths must contain at least one integer")
    if any(length <= 0 for length in lengths):
        raise ValueError("--lengths must be positive")
    return lengths


def read_ccd_codes(ccd_list_tsv: Path) -> list[str]:
    """Return unique CCD codes from the step1 TSV, preserving file order."""
    with ccd_list_tsv.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or CCD_COLUMN not in reader.fieldnames:
            raise ValueError(f"{ccd_list_tsv} missing '{CCD_COLUMN}' column")
        seen: set[str] = set()
        codes: list[str] = []
        for row in reader:
            code = str(row[CCD_COLUMN]).strip().upper()
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_ligand_cif(ccd: str, ccd_mirror: Path, cif_path: Path) -> dict[str, Any]:
    """Write a ligand CIF for RFD3; return metadata (atom count, buried atom names, charge)."""
    atom_array = atom_array_from_ccd_code(ccd, ccd_mirror_path=ccd_mirror)
    n_atoms = int(atom_array.array_length())
    single_atom = n_atoms == 1

    charge: int | None = None
    if single_atom:
        # Single-atom metal CCDs carry a NaN ideal coordinate; the ligand is diffused
        # (not fixed), so the position is only an initialization -> pin to the origin.
        atom_array.coord = np.zeros((n_atoms, 3), dtype=np.float32)
        charge = int(atom_array.charge[0])
        expected = METAL_FORMAL_CHARGES.get(ccd)
        if expected is not None and charge != expected:
            raise ValueError(
                f"CCD formal charge for {ccd} is {charge}, expected {expected}. "
                "Update METAL_FORMAL_CHARGES if the CCD definition changed."
            )
    elif not np.isfinite(np.asarray(atom_array.coord)).all():
        # A multi-atom ligand seeds the diffusion with its ideal conformer; non-finite
        # coords would break RFD3, so attribute the failure at stage time.
        raise ValueError(f"{ccd}: multi-atom CCD has non-finite ideal coordinates")

    atom_names = [str(name) for name in atom_array.atom_name]
    atom_array.chain_id = np.full(n_atoms, "L", dtype=atom_array.chain_id.dtype)
    atom_array.res_id = np.full(n_atoms, 1, dtype=atom_array.res_id.dtype)
    atom_array.res_name = np.full(n_atoms, ccd, dtype=atom_array.res_name.dtype)
    atom_array.hetero = np.full(n_atoms, True, dtype=bool)
    atom_array.set_annotation("is_polymer", np.full(n_atoms, False, dtype=bool))
    atom_array.set_annotation("chain_entity", np.full(n_atoms, "1"))

    cif_path.parent.mkdir(parents=True, exist_ok=True)
    to_cif_file(
        atom_array,
        cif_path,
        file_type="cif",
        id=ccd,
        include_entity_poly=False,
        include_bonds=not single_atom,
    )
    return {"n_atoms": n_atoms, "single_atom": single_atom, "charge": charge, "atom_names": atom_names}


def build_spec(
    ccd: str,
    cif_path: Path,
    *,
    length: int,
    buried_selector: str,
) -> dict[str, Any]:
    """Diffused-ligand + fully-buried RFD3 spec for one (ccd, length)."""
    return {
        "input": str(cif_path),
        "ligand": ccd,
        "length": f"{length}-{length}",
        "select_fixed_atoms": {ccd: ""},  # empty selection -> ligand coordinates diffused
        "select_buried": {ccd: buried_selector},
        "allow_ligand_on_existing_chain": False,
        "cif_parser_args": {"add_missing_atoms": False},
    }


def buried_selector_for(select_buried: str, atom_names: list[str]) -> str:
    if select_buried == "all":
        return "ALL"
    if not atom_names:
        raise ValueError("enumerate mode requires at least one atom name")
    return ",".join(atom_names)


def main() -> None:
    args = parse_args()
    if not args.ccd_mirror.exists():
        raise FileNotFoundError(f"CCD mirror does not exist: {args.ccd_mirror}")

    lengths = parse_lengths(args.lengths)
    if args.ccd:
        ccd_codes = [code.upper() for code in args.ccd]
    else:
        ccd_codes = read_ccd_codes(args.ccd_list)
    if not ccd_codes:
        raise SystemExit("No CCD codes selected.")

    inputs_dir = args.source_dir / "inputs"
    rfd3_inputs_dir = args.source_dir / "rfd3_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    rfd3_inputs_dir.mkdir(parents=True, exist_ok=True)

    ccd_records: list[dict[str, Any]] = []
    json_outputs: list[dict[str, Any]] = []
    ok_ccds: list[str] = []
    failed_ccds: list[str] = []

    for ccd in ccd_codes:
        cif_path = inputs_dir / f"{ccd}.cif"
        try:
            if args.overwrite or not cif_path.exists():
                meta = write_ligand_cif(ccd, args.ccd_mirror, cif_path)
            else:
                # Re-derive atom names from the mirror without rewriting the CIF.
                atom_array = atom_array_from_ccd_code(ccd, ccd_mirror_path=args.ccd_mirror)
                meta = {
                    "n_atoms": int(atom_array.array_length()),
                    "single_atom": int(atom_array.array_length()) == 1,
                    "charge": METAL_FORMAL_CHARGES.get(ccd),
                    "atom_names": [str(name) for name in atom_array.atom_name],
                }
        except Exception as exc:  # atomworks/RDKit raise varied types on malformed CCDs.
            failed_ccds.append(ccd)
            ccd_records.append({"ccd_code": ccd, "status": "error", "error": str(exc)})
            continue

        buried_selector = buried_selector_for(args.select_buried, meta["atom_names"])
        for length in lengths:
            spec_name = f"{ccd}_len{length}"
            specs = {spec_name: build_spec(ccd, cif_path, length=length, buried_selector=buried_selector)}
            output_path = rfd3_inputs_dir / f"{spec_name}.json"
            write_json(output_path, specs)
            json_outputs.append({"ccd_code": ccd, "length": length, "path": str(output_path)})

        ok_ccds.append(ccd)
        ccd_records.append(
            {
                "ccd_code": ccd,
                "status": "ok",
                "cif": str(cif_path),
                "n_atoms": meta["n_atoms"],
                "single_atom": meta["single_atom"],
                "charge": meta["charge"],
            }
        )

    # Persist the canonical CCD list (the sbatch array's source of truth) only on a FULL
    # stage. A --ccd subset run (smoke / re-stage of a few failures) must not clobber it.
    if not args.ccd:
        (args.source_dir / "denovoval_step2_ccds.txt").write_text("\n".join(ok_ccds) + "\n")

    total_specs = len(json_outputs)
    manifest = {
        "status": "ok" if not failed_ccds else "ok_partial",
        "source_dir": str(args.source_dir),
        "inputs_dir": str(inputs_dir),
        "rfd3_inputs_dir": str(rfd3_inputs_dir),
        "ccd_mirror": str(args.ccd_mirror),
        "ccd_list": str(args.ccd_list),
        "lengths": lengths,
        "select_buried": args.select_buried,
        "diffusion_batch_size": args.diffusion_batch_size,
        "n_batches": args.n_batches,
        "num_ccds_requested": len(ccd_codes),
        "num_ccds_ok": len(ok_ccds),
        "num_ccds_failed": len(failed_ccds),
        "failed_ccds": failed_ccds,
        "total_specs": total_specs,
        "expected_designs": total_specs * args.diffusion_batch_size * args.n_batches,
        "ccd_records": ccd_records,
        "json_outputs": json_outputs,
    }
    write_json(rfd3_inputs_dir / "rfd3_input_manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in (
        "status", "num_ccds_requested", "num_ccds_ok", "num_ccds_failed",
        "failed_ccds", "total_specs", "expected_designs",
    )}, indent=2))


if __name__ == "__main__":
    main()
