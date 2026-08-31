#!/usr/bin/env python3
"""Shared SwitchCraft design-output integrity checks."""

from __future__ import annotations

import pickletools
from pathlib import Path
from typing import Iterable


def expected_files(num_states: int, motifs: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for state in range(num_states):
        names.add(f"state{state}.pkl")
        for sample in range(5):
            stem = f"state{state}_sample{sample}"
            names.add(stem + ".pdb")
            names.add(stem + ".cif")
    for motif in motifs:
        names.add(f"{motif}_spec.pkl")
    return names


def _validate_pickle(path: Path) -> str | None:
    try:
        last_opcode = None
        with path.open("rb") as handle:
            for opcode, _argument, _position in pickletools.genops(handle):
                last_opcode = opcode.name
        if last_opcode != "STOP":
            return f"pickle stream ended with {last_opcode!r}, not 'STOP'"
    except Exception as exc:  # pickletools reports several truncation exception types
        return f"invalid pickle stream: {type(exc).__name__}: {exc}"
    return None


def _validate_pdb(path: Path) -> str | None:
    try:
        with path.open("r", errors="replace") as handle:
            if any(line.startswith(("ATOM  ", "HETATM")) for line in handle):
                return None
    except OSError as exc:
        return f"could not read PDB: {exc}"
    return "PDB has no ATOM/HETATM records"


def _validate_cif(path: Path) -> str | None:
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return f"could not read CIF: {exc}"
    if not text.lstrip().startswith("data_"):
        return "CIF does not start with a data_ block"
    if "_atom_site." not in text:
        return "CIF has no _atom_site loop"
    return None


def inspect_design(
    design_dir: Path, num_states: int, motifs: Iterable[str]
) -> dict[str, object]:
    """Return a JSON-serializable integrity report for one design directory."""

    expected = expected_files(num_states, motifs)
    if not design_dir.is_dir():
        return {
            "valid": False,
            "missing_files": sorted(expected),
            "invalid_files": [],
            "unexpected_files": [],
        }

    present = {entry.name for entry in design_dir.iterdir() if entry.is_file()}
    missing = sorted(expected - present)
    invalid: list[dict[str, str]] = []
    for name in sorted(expected & present):
        path = design_dir / name
        if path.stat().st_size == 0:
            error = "empty file"
        elif path.suffix == ".pkl":
            error = _validate_pickle(path)
        elif path.suffix == ".pdb":
            error = _validate_pdb(path)
        elif path.suffix == ".cif":
            error = _validate_cif(path)
        else:
            error = None
        if error:
            invalid.append({"file": name, "error": error})

    return {
        "valid": not missing and not invalid,
        "missing_files": missing,
        "invalid_files": invalid,
        "unexpected_files": sorted(present - expected),
    }
