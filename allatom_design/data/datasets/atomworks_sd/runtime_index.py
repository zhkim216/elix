"""Memory-mapped runtime metadata index for :mod:`atomworks_sd`.

The expensive Pandas metadata pipeline is intentionally kept out of training
processes.  A separate builder writes the final, sampled train records followed
by the validation records to one Arrow IPC file.  Training ranks and their
DataLoader workers then share the file-backed pages and materialize only the
record being accessed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
from omegaconf import DictConfig, OmegaConf

FORMAT_VERSION = 1
_METADATA_PREFIX = b"elix.atomworks_sd.runtime_index."
_BATCH_SIZE = 8192

# These settings affect loading/featurization, not which metadata records or
# sampling weights are written to the runtime index.
_NON_INDEX_CONFIG_KEYS = {
    "batch_size",
    "cif_parser_args",
    "featurizer_cfg",
    "num_workers",
    "pdb_path",
    "prefetch_factor",
    "residue_cache_dir",
    "runtime_index_path",
    "samples_per_epoch",
    "save_failed_examples_to_dir",
    "task",
    "val_batch_size",
}

_SCHEMA = pa.schema(
    [
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("pdb_id", pa.string(), nullable=False),
        pa.field("path", pa.string(), nullable=False),
        pa.field("assembly_id", pa.string(), nullable=False),
        pa.field("query_pn_unit_iids", pa.list_(pa.string()), nullable=False),
        pa.field("target_ligand_iids", pa.list_(pa.string()), nullable=False),
        pa.field("sampling_weight", pa.float64()),
        pa.field("ligand_pn_unit_iids", pa.list_(pa.string())),
        pa.field("protein_pn_unit_iids", pa.list_(pa.string())),
        pa.field("crop_center_pn_unit_iids", pa.list_(pa.string())),
        pa.field("query_pn_unit_iids_only", pa.bool_()),
    ]
)


def _file_identity(value: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def runtime_index_contract(cfg: DictConfig) -> tuple[str, str]:
    """Return the canonical JSON contract and its SHA-256 digest."""

    resolved = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    if not isinstance(resolved, dict):
        raise TypeError("The atomworks_sd data config must resolve to a mapping")

    index_cfg = {
        key: value
        for key, value in resolved.items()
        if key not in _NON_INDEX_CONFIG_KEYS
    }
    source_files = {
        key: _file_identity(resolved.get(key))
        for key in (
            "train_metadata_path",
            "val_metadata_path",
            "validation_ids_file",
        )
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "data_config": index_cfg,
        "source_files": source_files,
    }
    contract_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    return contract_json, digest


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        value = [value]
    return [str(item) for item in value]


def _as_optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    return _as_string_list(value)


def _iter_parsed_records(
    parsed_records: pd.Series | pd.DataFrame,
) -> Iterator[dict[str, Any]]:
    if isinstance(parsed_records, pd.Series):
        values: Iterable[Any] = parsed_records.values
    elif isinstance(parsed_records, pd.DataFrame):
        values = (row for _, row in parsed_records.iterrows())
    else:
        raise TypeError(
            "Parsed metadata must be a Pandas Series or DataFrame, got "
            f"{type(parsed_records).__name__}"
        )

    for value in values:
        if isinstance(value, pd.Series):
            yield value.to_dict()
        elif isinstance(value, dict):
            yield value
        else:
            raise TypeError(
                "Each parsed metadata record must be a mapping, got "
                f"{type(value).__name__}"
            )


def _compact_record(record: dict[str, Any], *, phase: str) -> dict[str, Any]:
    extra_info = record.get("extra_info") or {}
    pdb_id = extra_info.get("pdb_id")
    if pdb_id in (None, ""):
        raise ValueError(
            f"Runtime metadata record {record.get('example_id')!r} has no pdb_id"
        )

    sampling_weight = extra_info.get("sampling_weight")
    if phase == "train":
        if sampling_weight is None or not np.isfinite(float(sampling_weight)):
            raise ValueError(
                f"Train record {record.get('example_id')!r} has an invalid "
                f"sampling weight: {sampling_weight!r}"
            )
        sampling_weight = float(sampling_weight)
    else:
        sampling_weight = None

    query_only = record.get("query_pn_unit_iids_only")
    if isinstance(query_only, (float, np.floating)) and np.isnan(query_only):
        query_only = None
    elif query_only is not None:
        query_only = bool(query_only)

    return {
        "example_id": str(record["example_id"]),
        "pdb_id": str(pdb_id),
        "path": str(record["path"]),
        "assembly_id": str(record["assembly_id"]),
        "query_pn_unit_iids": _as_string_list(record["query_pn_unit_iids"]),
        "target_ligand_iids": _as_string_list(record["target_ligand_iids"]),
        "sampling_weight": sampling_weight,
        "ligand_pn_unit_iids": _as_optional_string_list(
            record.get("ligand_pn_unit_iids")
        ),
        "protein_pn_unit_iids": _as_optional_string_list(
            record.get("protein_pn_unit_iids")
        ),
        "crop_center_pn_unit_iids": _as_optional_string_list(
            record.get("crop_center_pn_unit_iids")
        ),
        "query_pn_unit_iids_only": query_only,
    }


def _record_batches(
    parsed_records: pd.Series | pd.DataFrame,
    *,
    phase: str,
) -> Iterator[pa.RecordBatch]:
    columns = {field.name: [] for field in _SCHEMA}
    for record in _iter_parsed_records(parsed_records):
        compact = _compact_record(record, phase=phase)
        for key, value in compact.items():
            columns[key].append(value)
        if len(columns["example_id"]) == _BATCH_SIZE:
            yield pa.RecordBatch.from_pydict(columns, schema=_SCHEMA)
            columns = {field.name: [] for field in _SCHEMA}

    if columns["example_id"]:
        yield pa.RecordBatch.from_pydict(columns, schema=_SCHEMA)


def write_runtime_index(
    output_path: str | os.PathLike[str],
    *,
    cfg: DictConfig,
    train_records: pd.Series | pd.DataFrame,
    val_records: pd.Series | pd.DataFrame,
    overwrite: bool = False,
) -> Path:
    """Atomically write final train and validation records to Arrow IPC."""

    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Runtime metadata index already exists: {output}. "
            "Set +runtime_index_builder.overwrite=true only after resolving "
            "the exact target."
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    contract_json, contract_digest = runtime_index_contract(cfg)
    train_rows = len(train_records)
    val_rows = len(val_records)
    schema = _SCHEMA.with_metadata(
        {
            _METADATA_PREFIX + b"format_version": str(FORMAT_VERSION),
            _METADATA_PREFIX + b"contract_digest": contract_digest,
            _METADATA_PREFIX + b"contract_json": contract_json,
            _METADATA_PREFIX + b"train_rows": str(train_rows),
            _METADATA_PREFIX + b"val_rows": str(val_rows),
        }
    )

    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    try:
        with pa.OSFile(str(temporary), "wb") as sink:
            with ipc.new_file(sink, schema) as writer:
                for phase, records in (
                    ("train", train_records),
                    ("val", val_records),
                ):
                    for batch in _record_batches(records, phase=phase):
                        writer.write_batch(batch)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


class RuntimeMetadataIndex:
    """Read-only phase view over a memory-mapped runtime index."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        cfg: DictConfig,
        phase: Literal["train", "val"],
    ):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Runtime metadata index is missing: {self.path}. "
                "Build it explicitly with `python -m "
                "allatom_design.data.datasets.atomworks_sd.build_runtime_index`."
            )

        self._source = pa.memory_map(str(self.path), "r")
        self._reader = ipc.open_file(self._source)
        self._table = self._reader.read_all()
        metadata = self._table.schema.metadata or {}

        def metadata_value(key: bytes) -> str:
            full_key = _METADATA_PREFIX + key
            if full_key not in metadata:
                raise ValueError(
                    f"Runtime metadata index {self.path} is missing schema "
                    f"metadata {full_key.decode()!r}"
                )
            return metadata[full_key].decode("utf-8")

        format_version = int(metadata_value(b"format_version"))
        if format_version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported runtime metadata index version {format_version}; "
                f"expected {FORMAT_VERSION}: {self.path}"
            )

        _, expected_digest = runtime_index_contract(cfg)
        actual_digest = metadata_value(b"contract_digest")
        if actual_digest != expected_digest:
            raise ValueError(
                "Runtime metadata index contract mismatch. "
                f"Expected {expected_digest}, found {actual_digest}: {self.path}. "
                "Rebuild the index for the resolved data config."
            )

        train_rows = int(metadata_value(b"train_rows"))
        val_rows = int(metadata_value(b"val_rows"))
        if train_rows + val_rows != self._table.num_rows:
            raise ValueError(
                f"Runtime metadata index row counts are inconsistent: "
                f"train={train_rows}, val={val_rows}, "
                f"table={self._table.num_rows}: {self.path}"
            )

        self.phase = phase
        self._offset = 0 if phase == "train" else train_rows
        self._length = train_rows if phase == "train" else val_rows
        self._example_ids = self._table["example_id"].slice(
            self._offset, self._length
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        absolute_idx = self._offset + idx

        def value(column: str) -> Any:
            return self._table[column][absolute_idx].as_py()

        sampling_weight = value("sampling_weight")
        record: dict[str, Any] = {
            "example_id": value("example_id"),
            "path": Path(value("path")),
            "assembly_id": value("assembly_id"),
            "query_pn_unit_iids": value("query_pn_unit_iids"),
            "target_ligand_iids": np.asarray(
                value("target_ligand_iids"), dtype=str
            ),
            "extra_info": {"pdb_id": value("pdb_id")},
        }
        if sampling_weight is not None:
            record["extra_info"]["sampling_weight"] = float(sampling_weight)

        for key in (
            "ligand_pn_unit_iids",
            "protein_pn_unit_iids",
            "crop_center_pn_unit_iids",
        ):
            item = value(key)
            if item is not None:
                record[key] = item
        if value("query_pn_unit_iids_only"):
            record["query_pn_unit_iids_only"] = True
        return record

    def sampling_weights(self) -> np.ndarray:
        if self.phase != "train":
            raise RuntimeError("Validation runtime indices have no sampling weights")
        weights = self._table["sampling_weight"].slice(
            self._offset, self._length
        ).to_numpy(zero_copy_only=False)
        weights = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"Invalid sampling weights in {self.path}")
        return weights

    def index_of(self, example_id: str) -> int:
        idx = pc.index(self._example_ids, pa.scalar(str(example_id))).as_py()
        if idx == -1:
            raise KeyError(example_id)
        return int(idx)

    def contains(self, example_id: str) -> bool:
        try:
            self.index_of(example_id)
        except KeyError:
            return False
        return True

    def example_id(self, idx: int) -> str:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        return str(self._example_ids[idx].as_py())
