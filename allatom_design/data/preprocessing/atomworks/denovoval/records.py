"""Source-record discovery for denovoval small-molecule/metal CIF pools."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SMALL_MOLECULE = "small_molecule"
METAL = "metal"

_NATURAL_PART_RE = re.compile(r"(\d+)")
_SMALL_STEM_RE = re.compile(
    r"^chunk_(?P<chunk>\d+)_len(?P<chunk_length>\d+)_(?P<ccd>.+?)_conf"
    r"(?P<conf_cluster>\d+)_len(?P<name_length>\d+)_(?P<input_index>\d+)_model_"
    r"(?P<model_index>\d+)$"
)
_METAL_STEM_RE = re.compile(
    r"^(?P<ccd>[A-Za-z0-9]+)_len(?P<condition_length>\d+)_"
    r"(?P=ccd)_len(?P<name_length>\d+)_(?P<input_index>\d+)_model_"
    r"(?P<model_index>\d+)$"
)


@dataclass(frozen=True)
class SourceRecord:
    """One source ``.cif.gz`` with parsed denovoval grouping metadata."""

    sample_id: str
    kind: str
    ccd_code: str
    length: int
    source_path: Path
    json_path: Path
    source_root: Path
    source_condition: str
    conf_cluster: str | None
    input_index: int
    model_index: int

    def to_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "kind": self.kind,
            "ccd_code": self.ccd_code,
            "length": self.length,
            "source_path": str(self.source_path),
            "json_path": str(self.json_path),
            "source_root": str(self.source_root),
            "source_condition": self.source_condition,
            "conf_cluster": self.conf_cluster or "",
            "input_index": self.input_index,
            "model_index": self.model_index,
            "json_exists": self.json_path.exists(),
        }


def natural_key(value: str | Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    for part in _NATURAL_PART_RE.split(str(value)):
        parts.append(int(part) if part.isdigit() else part)
    return tuple(parts)


def strip_cif_gz(path: Path) -> str:
    if path.name.endswith(".cif.gz"):
        return path.name[: -len(".cif.gz")]
    return path.stem


def paired_json_path(cif_path: Path) -> Path:
    if not cif_path.name.endswith(".cif.gz"):
        raise ValueError(f"Expected .cif.gz source path, got: {cif_path}")
    return cif_path.with_name(f"{strip_cif_gz(cif_path)}.json")


def parse_small_molecule_record(path: Path, source_root: Path) -> SourceRecord:
    stem = strip_cif_gz(path)
    match = _SMALL_STEM_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"Cannot parse small-molecule source filename: {path}")

    chunk_length = int(match.group("chunk_length"))
    name_length = int(match.group("name_length"))
    if chunk_length != name_length:
        raise ValueError(f"Length mismatch in small-molecule filename: {path}")

    parent_match = re.fullmatch(r"chunk_\d+_len(?P<length>\d+)", path.parent.name)
    if parent_match is not None and int(parent_match.group("length")) != chunk_length:
        raise ValueError(f"Parent length mismatch for small-molecule source: {path}")

    return SourceRecord(
        sample_id=stem,
        kind=SMALL_MOLECULE,
        ccd_code=match.group("ccd").upper(),
        length=chunk_length,
        source_path=path,
        json_path=paired_json_path(path),
        source_root=source_root,
        source_condition=path.parent.name,
        conf_cluster=f"conf{int(match.group('conf_cluster')):03d}",
        input_index=int(match.group("input_index")),
        model_index=int(match.group("model_index")),
    )


def parse_metal_record(path: Path, source_root: Path) -> SourceRecord:
    stem = strip_cif_gz(path)
    match = _METAL_STEM_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"Cannot parse metal source filename: {path}")

    condition_length = int(match.group("condition_length"))
    name_length = int(match.group("name_length"))
    if condition_length != name_length:
        raise ValueError(f"Length mismatch in metal filename: {path}")

    parent_match = re.fullmatch(
        rf"{re.escape(match.group('ccd'))}_len(?P<length>\d+)",
        path.parent.name,
        flags=re.IGNORECASE,
    )
    if parent_match is not None and int(parent_match.group("length")) != condition_length:
        raise ValueError(f"Parent length mismatch for metal source: {path}")

    return SourceRecord(
        sample_id=stem,
        kind=METAL,
        ccd_code=match.group("ccd").upper(),
        length=condition_length,
        source_path=path,
        json_path=paired_json_path(path),
        source_root=source_root,
        source_condition=path.parent.name,
        conf_cluster=None,
        input_index=int(match.group("input_index")),
        model_index=int(match.group("model_index")),
    )


def discover_source_records(
    *,
    small_molecule_root: Path,
    metal_root: Path,
    lengths: set[int] | None = None,
    ccd_codes: set[str] | None = None,
) -> list[SourceRecord]:
    """Discover and parse all source CIFs in the two denovoval roots."""
    records: list[SourceRecord] = []
    for root, parser in (
        (small_molecule_root, parse_small_molecule_record),
        (metal_root, parse_metal_record),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing source root: {root}")
        for path in sorted(root.glob("*/*.cif.gz"), key=natural_key):
            record = parser(path, root)
            if lengths is not None and record.length not in lengths:
                continue
            if ccd_codes is not None and record.ccd_code not in ccd_codes:
                continue
            records.append(record)

    return sorted(records, key=record_sort_key)


def record_sort_key(record: SourceRecord) -> tuple[Any, ...]:
    kind_rank = 0 if record.kind == SMALL_MOLECULE else 1
    return (
        kind_rank,
        record.length,
        record.ccd_code,
        record.conf_cluster or "",
        record.input_index,
        record.model_index,
        natural_key(record.sample_id),
    )


def apply_debug_sample_cap(
    records: list[SourceRecord],
    debug_num_samples: int | None,
) -> list[SourceRecord]:
    """Select up to ``debug_num_samples`` per kind across diverse groups."""
    if debug_num_samples is None or int(debug_num_samples) <= 0:
        return sorted(records, key=record_sort_key)

    capped: list[SourceRecord] = []
    for kind in (SMALL_MOLECULE, METAL):
        kind_records = [record for record in records if record.kind == kind]
        capped.extend(_round_robin_by_group(kind_records, int(debug_num_samples)))
    return sorted(capped, key=record_sort_key)


def _round_robin_by_group(records: list[SourceRecord], limit: int) -> list[SourceRecord]:
    grouped: dict[tuple[int, str], list[SourceRecord]] = defaultdict(list)
    for record in sorted(records, key=record_sort_key):
        grouped[(record.length, record.ccd_code)].append(record)

    selected: list[SourceRecord] = []
    group_keys = sorted(grouped, key=lambda key: (key[1], key[0]))
    group_offsets = {key: 0 for key in group_keys}
    while len(selected) < limit:
        made_progress = False
        for key in group_keys:
            offset = group_offsets[key]
            group = grouped[key]
            if offset >= len(group):
                continue
            selected.append(group[offset])
            group_offsets[key] = offset + 1
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    return selected
