"""Standardize denovoval_re ss1p6 ligand residue IDs without reserializing CIFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path(
    "/scratch/users/zhkim216/datasets/evaluation_datasets/denovoval_re"
)
CONDITION = "ss1p6_gamma0p6"
ACTIVE_COUNT = 1_540
ENSEMBLE_COUNT = 246_400
TOKEN = re.compile(rb"\S+")
LENGTH = re.compile(r"_len(?P<length>[0-9]+)_")


@dataclass(frozen=True)
class TransformResult:
    data: bytes
    old_residue_id: int
    ligand_atom_rows: int
    atom_site_tokens_changed: int
    nonpoly_tokens_changed: int

    @property
    def changed(self) -> bool:
        return self.atom_site_tokens_changed + self.nonpoly_tokens_changed > 0


def _token_matches(line: bytes) -> list[re.Match[bytes]]:
    return list(TOKEN.finditer(line))


def _token_value(match: re.Match[bytes]) -> bytes:
    value = match.group(0)
    if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {b"'", b'"'}:
        return value[1:-1]
    return value


def _replace_tokens(
    line: bytes,
    matches: list[re.Match[bytes]],
    replacements: dict[int, bytes],
) -> bytes:
    if not replacements:
        return line
    pieces: list[bytes] = []
    cursor = 0
    for index, match in enumerate(matches):
        pieces.append(line[cursor : match.start()])
        pieces.append(replacements.get(index, match.group(0)))
        cursor = match.end()
    pieces.append(line[cursor:])
    return b"".join(pieces)


def _parse_residue_id(match: re.Match[bytes], *, field: str) -> int:
    value = _token_value(match)
    try:
        residue_id = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not an integer: {value!r}") from exc
    if residue_id < 1:
        raise ValueError(f"{field} must be positive, got {residue_id}")
    return residue_id


def _loop_end(lines: list[bytes], start: int) -> int:
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        if (
            not stripped
            or stripped.startswith(b"#")
            or stripped == b"loop_"
            or stripped.startswith(b"_")
            or stripped.startswith(b"data_")
            or stripped.startswith(b"save_")
        ):
            break
        index += 1
    return index


def transform_cif_bytes(
    data: bytes,
    *,
    expected_old_residue_id: int | None = None,
) -> TransformResult:
    """Return a byte-preserving chain-L residue-ID normalization."""

    lines = data.splitlines(keepends=True)
    observed_ids: set[int] = set()
    ligand_atom_rows = 0
    atom_site_tokens_changed = 0
    nonpoly_tokens_changed = 0
    atom_site_loop_count = 0
    nonpoly_scalar_items: dict[str, tuple[int, list[re.Match[bytes]]]] = {}

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == b"loop_":
            header_index = index + 1
            headers: list[str] = []
            while header_index < len(lines):
                matches = _token_matches(lines[header_index])
                if len(matches) != 1 or not matches[0].group(0).startswith(b"_"):
                    break
                headers.append(matches[0].group(0).decode("ascii"))
                header_index += 1
            if not headers:
                raise ValueError(f"loop_ at line {index + 1} has no headers")
            data_end = _loop_end(lines, header_index)

            if all(header.startswith("_atom_site.") for header in headers):
                atom_site_loop_count += 1
                required = {
                    "_atom_site.label_asym_id",
                    "_atom_site.auth_asym_id",
                    "_atom_site.label_seq_id",
                    "_atom_site.auth_seq_id",
                }
                missing = sorted(required - set(headers))
                if missing:
                    raise ValueError(f"_atom_site loop is missing fields: {missing}")
                column = {header: position for position, header in enumerate(headers)}
                for row_index in range(header_index, data_end):
                    matches = _token_matches(lines[row_index])
                    if len(matches) != len(headers):
                        raise ValueError(
                            f"_atom_site row {row_index + 1} has {len(matches)} "
                            f"tokens, expected {len(headers)}"
                        )
                    label_chain = _token_value(
                        matches[column["_atom_site.label_asym_id"]]
                    )
                    auth_chain = _token_value(
                        matches[column["_atom_site.auth_asym_id"]]
                    )
                    if (label_chain == b"L") != (auth_chain == b"L"):
                        raise ValueError(
                            f"_atom_site row {row_index + 1} has inconsistent "
                            f"label/auth ligand chains"
                        )
                    if label_chain != b"L":
                        continue
                    label_position = column["_atom_site.label_seq_id"]
                    auth_position = column["_atom_site.auth_seq_id"]
                    label_id = _parse_residue_id(
                        matches[label_position], field="_atom_site.label_seq_id"
                    )
                    auth_id = _parse_residue_id(
                        matches[auth_position], field="_atom_site.auth_seq_id"
                    )
                    if label_id != auth_id:
                        raise ValueError(
                            f"_atom_site row {row_index + 1} has label/auth "
                            f"residue IDs {label_id}/{auth_id}"
                        )
                    observed_ids.add(label_id)
                    ligand_atom_rows += 1
                    replacements: dict[int, bytes] = {}
                    if label_id != 1:
                        replacements[label_position] = b"1"
                        replacements[auth_position] = b"1"
                    lines[row_index] = _replace_tokens(
                        lines[row_index], matches, replacements
                    )
                    atom_site_tokens_changed += len(replacements)

            elif all(
                header.startswith("_pdbx_nonpoly_scheme.") for header in headers
            ):
                column = {header: position for position, header in enumerate(headers)}
                required = {
                    "_pdbx_nonpoly_scheme.asym_id",
                    "_pdbx_nonpoly_scheme.ndb_seq_num",
                    "_pdbx_nonpoly_scheme.pdb_seq_num",
                    "_pdbx_nonpoly_scheme.auth_seq_num",
                }
                missing = sorted(required - set(headers))
                if missing:
                    raise ValueError(
                        f"_pdbx_nonpoly_scheme loop is missing fields: {missing}"
                    )
                for row_index in range(header_index, data_end):
                    matches = _token_matches(lines[row_index])
                    if len(matches) != len(headers):
                        raise ValueError(
                            f"_pdbx_nonpoly_scheme row {row_index + 1} has "
                            f"{len(matches)} tokens, expected {len(headers)}"
                        )
                    if (
                        _token_value(
                            matches[column["_pdbx_nonpoly_scheme.asym_id"]]
                        )
                        != b"L"
                    ):
                        continue
                    ndb_id = _parse_residue_id(
                        matches[column["_pdbx_nonpoly_scheme.ndb_seq_num"]],
                        field="_pdbx_nonpoly_scheme.ndb_seq_num",
                    )
                    if ndb_id != 1:
                        raise ValueError(
                            "_pdbx_nonpoly_scheme.ndb_seq_num must already be 1"
                        )
                    pdb_position = column["_pdbx_nonpoly_scheme.pdb_seq_num"]
                    auth_position = column["_pdbx_nonpoly_scheme.auth_seq_num"]
                    pdb_id = _parse_residue_id(
                        matches[pdb_position],
                        field="_pdbx_nonpoly_scheme.pdb_seq_num",
                    )
                    auth_id = _parse_residue_id(
                        matches[auth_position],
                        field="_pdbx_nonpoly_scheme.auth_seq_num",
                    )
                    if pdb_id != auth_id:
                        raise ValueError(
                            "_pdbx_nonpoly_scheme pdb/auth residue IDs differ"
                        )
                    observed_ids.add(pdb_id)
                    replacements = {}
                    if pdb_id != 1:
                        replacements[pdb_position] = b"1"
                        replacements[auth_position] = b"1"
                    lines[row_index] = _replace_tokens(
                        lines[row_index], matches, replacements
                    )
                    nonpoly_tokens_changed += len(replacements)
            index = data_end
            continue

        if stripped.startswith(b"_pdbx_nonpoly_scheme."):
            matches = _token_matches(lines[index])
            if len(matches) != 2:
                raise ValueError(
                    f"scalar _pdbx_nonpoly_scheme item at line {index + 1} "
                    f"has {len(matches)} tokens, expected 2"
                )
            key = matches[0].group(0).decode("ascii")
            if key in nonpoly_scalar_items:
                raise ValueError(f"duplicate scalar item: {key}")
            nonpoly_scalar_items[key] = (index, matches)
        index += 1

    if atom_site_loop_count != 1:
        raise ValueError(
            f"expected exactly one _atom_site loop, found {atom_site_loop_count}"
        )
    if ligand_atom_rows < 1:
        raise ValueError("_atom_site contains no chain L atoms")

    if nonpoly_scalar_items:
        required = {
            "_pdbx_nonpoly_scheme.asym_id",
            "_pdbx_nonpoly_scheme.ndb_seq_num",
            "_pdbx_nonpoly_scheme.pdb_seq_num",
            "_pdbx_nonpoly_scheme.auth_seq_num",
        }
        missing = sorted(required - set(nonpoly_scalar_items))
        if missing:
            raise ValueError(
                f"scalar _pdbx_nonpoly_scheme is missing fields: {missing}"
            )
        asym_line, asym_matches = nonpoly_scalar_items[
            "_pdbx_nonpoly_scheme.asym_id"
        ]
        if _token_value(asym_matches[1]) != b"L":
            raise ValueError(
                f"scalar _pdbx_nonpoly_scheme asym_id at line {asym_line + 1} "
                "is not L"
            )
        ndb_line, ndb_matches = nonpoly_scalar_items[
            "_pdbx_nonpoly_scheme.ndb_seq_num"
        ]
        if (
            _parse_residue_id(
                ndb_matches[1], field="_pdbx_nonpoly_scheme.ndb_seq_num"
            )
            != 1
        ):
            raise ValueError(
                f"_pdbx_nonpoly_scheme.ndb_seq_num at line {ndb_line + 1} "
                "must already be 1"
            )
        pdb_line, pdb_matches = nonpoly_scalar_items[
            "_pdbx_nonpoly_scheme.pdb_seq_num"
        ]
        auth_line, auth_matches = nonpoly_scalar_items[
            "_pdbx_nonpoly_scheme.auth_seq_num"
        ]
        pdb_id = _parse_residue_id(
            pdb_matches[1], field="_pdbx_nonpoly_scheme.pdb_seq_num"
        )
        auth_id = _parse_residue_id(
            auth_matches[1], field="_pdbx_nonpoly_scheme.auth_seq_num"
        )
        if pdb_id != auth_id:
            raise ValueError("_pdbx_nonpoly_scheme pdb/auth residue IDs differ")
        observed_ids.add(pdb_id)
        if pdb_id != 1:
            lines[pdb_line] = _replace_tokens(
                lines[pdb_line], pdb_matches, {1: b"1"}
            )
            lines[auth_line] = _replace_tokens(
                lines[auth_line], auth_matches, {1: b"1"}
            )
            nonpoly_tokens_changed += 2

    if len(observed_ids) != 1:
        raise ValueError(
            f"chain L uses inconsistent residue IDs: {sorted(observed_ids)}"
        )
    old_residue_id = next(iter(observed_ids))
    if (
        expected_old_residue_id is not None
        and old_residue_id not in {1, expected_old_residue_id}
    ):
        raise ValueError(
            f"chain L residue ID is {old_residue_id}, expected 1 or "
            f"{expected_old_residue_id}"
        )
    return TransformResult(
        data=b"".join(lines),
        old_residue_id=old_residue_id,
        ligand_atom_rows=ligand_atom_rows,
        atom_site_tokens_changed=atom_site_tokens_changed,
        nonpoly_tokens_changed=nonpoly_tokens_changed,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _expected_old_residue_id(path: Path) -> int:
    match = LENGTH.search(path.name)
    if match is None:
        raise ValueError(f"cannot derive protein length from CIF name: {path}")
    length = int(match.group("length"))
    if length not in {150, 300}:
        raise ValueError(f"unexpected denovoval protein length {length}: {path}")
    return length + 1


def _atomic_replace(path: Path, data: bytes) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    try:
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _discover_paths(dataset_root: Path) -> tuple[list[Path], list[Path]]:
    condition_root = dataset_root / CONDITION
    active_root = condition_root / "cifs"
    ensemble_root = condition_root / "ensembles_m32"
    active = sorted(active_root.glob("*.cif"), key=lambda path: path.name)
    ensemble = sorted(ensemble_root.rglob("*.cif"), key=lambda path: str(path))
    return active, ensemble


def _replace_selection_field(
    line: bytes,
    *,
    field_index: int,
    value: str,
) -> bytes:
    newline = b""
    body = line
    if body.endswith(b"\r\n"):
        body, newline = body[:-2], b"\r\n"
    elif body.endswith(b"\n"):
        body, newline = body[:-1], b"\n"
    fields = body.split(b"\t")
    fields[field_index] = value.encode("ascii")
    return b"\t".join(fields) + newline


def update_selection_tsv(
    *,
    dataset_root: Path,
    active_sha256: dict[str, str],
    apply: bool,
) -> dict[str, Any]:
    selection_path = dataset_root / "selection.tsv"
    original = selection_path.read_bytes()
    lines = original.splitlines(keepends=True)
    if not lines:
        raise ValueError(f"empty selection file: {selection_path}")
    header = lines[0].rstrip(b"\r\n").split(b"\t")
    required = {
        b"condition",
        b"pdb_key",
        b"source_cif_sha256",
        b"materialized_cif_sha256",
    }
    missing = sorted(value.decode() for value in required - set(header))
    if missing:
        raise ValueError(f"selection.tsv is missing fields: {missing}")
    column = {name.decode(): index for index, name in enumerate(header)}
    ss1p6_rows = 0
    authoritative_ala_rows = 0
    changed_rows = 0
    updated = list(lines)
    for row_index, line in enumerate(lines[1:], start=1):
        fields = line.rstrip(b"\r\n").split(b"\t")
        if len(fields) != len(header):
            raise ValueError(
                f"selection.tsv row {row_index + 1} has {len(fields)} fields, "
                f"expected {len(header)}"
            )
        condition = fields[column["condition"]].decode()
        sample_id = fields[column["pdb_key"]].decode()
        expected_hash: str | None = None
        if condition == CONDITION:
            ss1p6_rows += 1
            try:
                expected_hash = active_sha256[sample_id]
            except KeyError as exc:
                raise ValueError(
                    f"selection.tsv ss1p6 sample is absent from active CIFs: "
                    f"{sample_id}"
                ) from exc
        elif condition == "ss1p0_gamma1p0" and sample_id == "1SY_len300_0":
            authoritative_ala_rows += 1
            expected_hash = _sha256(
                (
                    dataset_root
                    / "ss1p0_gamma1p0"
                    / "cifs"
                    / f"{sample_id}.cif"
                ).read_bytes()
            )
        if expected_hash is None:
            continue
        current_hash = fields[column["materialized_cif_sha256"]].decode()
        if current_hash != expected_hash:
            updated[row_index] = _replace_selection_field(
                line,
                field_index=column["materialized_cif_sha256"],
                value=expected_hash,
            )
            changed_rows += 1
    if ss1p6_rows != ACTIVE_COUNT or len(active_sha256) != ACTIVE_COUNT:
        raise ValueError(
            f"selection/active ss1p6 cardinality mismatch: "
            f"{ss1p6_rows}/{len(active_sha256)}"
        )
    if authoritative_ala_rows != 1:
        raise ValueError(
            f"expected one authoritative ss1p0 1SY_len300_0 row, found "
            f"{authoritative_ala_rows}"
        )
    output = b"".join(updated)
    if apply and output != original:
        _atomic_replace(selection_path, output)
    return {
        "path": str(selection_path),
        "changed_rows": changed_rows,
        "old_sha256": _sha256(original),
        "new_sha256": _sha256(output),
        "applied": bool(apply and output != original),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically replace changed CIFs and update selection.tsv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Bounded dry-run/development limit; live partial apply is forbidden.",
    )
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    if dataset_root.name != "denovoval_re":
        raise ValueError(
            f"dataset root must be named denovoval_re, got {dataset_root}"
        )
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if (
        args.apply
        and args.limit is not None
        and dataset_root == DEFAULT_DATASET_ROOT.resolve()
    ):
        raise ValueError("partial --apply is forbidden for the live dataset")

    active, ensemble = _discover_paths(dataset_root)
    if args.limit is None:
        if len(active) != ACTIVE_COUNT or len(ensemble) != ENSEMBLE_COUNT:
            raise ValueError(
                f"unexpected CIF counts: active={len(active)} "
                f"ensemble={len(ensemble)}"
            )
        selected = active + ensemble
    else:
        selected = (active + ensemble)[: args.limit]

    active_root = dataset_root / CONDITION / "cifs"
    active_sha256: dict[str, str] = {}
    changed_files = 0
    unchanged_files = 0
    atom_rows = 0
    atom_tokens = 0
    nonpoly_tokens = 0
    for processed, path in enumerate(selected, start=1):
        try:
            path.resolve().relative_to((dataset_root / CONDITION).resolve())
        except ValueError as exc:
            raise ValueError(f"CIF path escapes the allowed ss1p6 root: {path}") from exc
        if "ss1p0_gamma1p0" in path.parts:
            raise ValueError(f"ss1p0 mutation is forbidden: {path}")
        original = path.read_bytes()
        result = transform_cif_bytes(
            original,
            expected_old_residue_id=_expected_old_residue_id(path),
        )
        if result.changed:
            changed_files += 1
            if args.apply:
                _atomic_replace(path, result.data)
        else:
            unchanged_files += 1
        atom_rows += result.ligand_atom_rows
        atom_tokens += result.atom_site_tokens_changed
        nonpoly_tokens += result.nonpoly_tokens_changed
        if path.parent == active_root:
            active_sha256[path.stem] = _sha256(result.data)
        if processed % args.progress_every == 0:
            print(
                f"processed={processed} changed={changed_files} "
                f"unchanged={unchanged_files}",
                file=sys.stderr,
                flush=True,
            )

    selection: dict[str, Any] | None = None
    if args.limit is None:
        selection = update_selection_tsv(
            dataset_root=dataset_root,
            active_sha256=active_sha256,
            apply=args.apply,
        )
    return {
        "status": "complete",
        "mode": "apply" if args.apply else "dry_run",
        "dataset_root": str(dataset_root),
        "selected_files": len(selected),
        "active_files": len(active),
        "ensemble_files": len(ensemble),
        "changed_files": changed_files,
        "unchanged_files": unchanged_files,
        "ligand_atom_rows": atom_rows,
        "atom_site_tokens_changed": atom_tokens,
        "nonpoly_tokens_changed": nonpoly_tokens,
        "selection": selection,
    }


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
