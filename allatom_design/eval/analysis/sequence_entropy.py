"""Compute sequence entropy from run_elix sample metadata."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import atomworks.enums as aw_enums
from atomworks.constants import DICT_THREE_TO_ONE
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from biotite.structure import AtomArray
from omegaconf import OmegaConf

from allatom_design.data.transform.custom_transforms import (
    annotate_ligand_pockets,
    annotate_ligand_pockets_calpha,
    annotate_ligand_pockets_pseudocb,
)
from allatom_design.eval.utils.constraint_utils import resolve_pocket_annotation_method
from allatom_design.eval.utils.sequence_recovery import DESIGNED_CIF_PARSE_CFG
from allatom_design.utils.atom_array_utils import get_valid_standard_aa_residue_mask
from allatom_design.utils.sample_io_utils import load_example_with_parse

AA20 = tuple("ACDEFGHIKLMNPQRSTVWY")
NORMALIZATION_DENOMINATOR = math.log2(len(AA20))
FAILURE_COLUMNS = [
    "metadata_path",
    "example_id",
    "designed_sample_id",
    "designed_sample_path",
    "reason",
]


@dataclass(frozen=True)
class MetadataRecord:
    metadata_path: Path
    example_id: str
    designed_sample_id: str
    designed_sample_seq: str | None
    designed_sample_path: Path | None


@dataclass(frozen=True)
class RegionSpec:
    name: str
    lo: float | None = None
    hi: float | None = None

    @property
    def requires_structure(self) -> bool:
        return self.hi is not None


@dataclass(frozen=True)
class ResidueSequence:
    keys: tuple[tuple[str, int, str, int], ...]
    letters: tuple[str, ...]
    region_masks: dict[str, tuple[bool, ...]]


@dataclass(frozen=True)
class AnalysisResult:
    summary_df: pd.DataFrame
    per_position_df: pd.DataFrame
    failures_df: pd.DataFrame


def region_name_for_cumulative(distance: float) -> str:
    return f"pocket_le_{_format_distance_label(distance)}A"


def region_name_for_shell(lo: float, hi: float) -> str:
    return f"pocket_{_format_distance_label(lo)}_to_{_format_distance_label(hi)}A"


def default_region_specs(
    cumulative_distances: Sequence[float] = (6.0, 8.0, 10.0, 12.0),
    shell_bins: Sequence[tuple[float, float]] = ((6.0, 8.0), (8.0, 10.0), (10.0, 12.0)),
) -> list[RegionSpec]:
    specs = [RegionSpec(name="all")]
    specs.extend(
        RegionSpec(name=region_name_for_cumulative(float(distance)), hi=float(distance))
        for distance in cumulative_distances
    )
    specs.extend(
        RegionSpec(name=region_name_for_shell(float(lo), float(hi)), lo=float(lo), hi=float(hi))
        for lo, hi in shell_bins
    )
    return specs


def parse_shell_bin(raw: str) -> tuple[float, float]:
    for separator in (":", "-"):
        if separator in raw:
            lo_raw, hi_raw = raw.split(separator, 1)
            lo, hi = float(lo_raw), float(hi_raw)
            if not lo < hi:
                raise ValueError(f"Shell bin must satisfy lo < hi: {raw!r}")
            return lo, hi
    raise ValueError(f"Shell bin must be formatted as lo:hi or lo-hi: {raw!r}")


def expand_metadata_paths(path_patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in path_patterns:
        raw = str(raw_path)
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.rglob("sample_metadata*.pt")))
        elif path.is_file():
            paths.append(path)
        else:
            matches = sorted(Path(match) for match in glob.glob(raw))
            paths.extend(match for match in matches if match.is_file())

    unique_paths = sorted(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise FileNotFoundError(f"No sample_metadata*.pt files matched: {list(map(str, path_patterns))}")
    return unique_paths


def load_metadata_records(metadata_paths: Sequence[Path]) -> tuple[list[MetadataRecord], list[dict[str, Any]]]:
    records: list[MetadataRecord] = []
    failures: list[dict[str, Any]] = []

    for metadata_path in metadata_paths:
        try:
            metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            failures.append(_failure_row(metadata_path=metadata_path, reason=f"metadata_load_failed: {exc}"))
            continue
        if not isinstance(metadata, dict):
            failures.append(_failure_row(metadata_path=metadata_path, reason="metadata_is_not_dict"))
            continue

        for fallback_sample_id, entry in metadata.items():
            if not isinstance(entry, dict):
                failures.append(
                    _failure_row(
                        metadata_path=metadata_path,
                        designed_sample_id=str(fallback_sample_id),
                        reason="metadata_entry_is_not_dict",
                    )
                )
                continue
            example_id = entry.get("example_id")
            designed_sample_id = entry.get("designed_sample_id", fallback_sample_id)
            if example_id is None or designed_sample_id is None:
                failures.append(
                    _failure_row(
                        metadata_path=metadata_path,
                        designed_sample_id=str(fallback_sample_id),
                        reason="missing_example_or_sample_id",
                    )
                )
                continue

            raw_sample_path = entry.get("designed_sample_path")
            sample_path = Path(raw_sample_path) if raw_sample_path else None
            records.append(
                MetadataRecord(
                    metadata_path=metadata_path,
                    example_id=str(example_id),
                    designed_sample_id=str(designed_sample_id),
                    designed_sample_seq=entry.get("designed_sample_seq"),
                    designed_sample_path=sample_path,
                )
            )

    return records, failures


def analyze_metadata(
    metadata_paths: Sequence[Path],
    *,
    region_specs: Sequence[RegionSpec],
    pocket_annotation_method: str = "calpha",
    cif_parse_cfg: dict[str, Any] | None = None,
) -> AnalysisResult:
    records, failures = load_metadata_records(metadata_paths)
    load_residue_sequence = _build_cif_loader(
        region_specs=region_specs,
        pocket_annotation_method=pocket_annotation_method,
        cif_parse_cfg=cif_parse_cfg,
    )
    result = analyze_records(
        records,
        region_specs=region_specs,
        load_residue_sequence=load_residue_sequence,
        initial_failures=failures,
    )
    return result


def analyze_records(
    records: Sequence[MetadataRecord],
    *,
    region_specs: Sequence[RegionSpec],
    load_residue_sequence: Callable[[MetadataRecord], ResidueSequence],
    initial_failures: Sequence[dict[str, Any]] | None = None,
) -> AnalysisResult:
    failures = list(initial_failures or [])
    records_by_example: dict[str, list[MetadataRecord]] = defaultdict(list)
    for record in records:
        records_by_example[record.example_id].append(record)

    per_position_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for example_id, group_records in sorted(records_by_example.items()):
        loaded: list[tuple[MetadataRecord, ResidueSequence]] = []
        reference: ResidueSequence | None = None
        for record in group_records:
            try:
                residue_sequence = load_residue_sequence(record)
            except Exception as exc:
                failures.append(_failure_from_record(record, f"sample_load_failed: {exc}"))
                continue
            if len(residue_sequence.letters) == 0:
                failures.append(_failure_from_record(record, "empty_protein_sequence"))
                continue
            if reference is None:
                reference = residue_sequence
            elif residue_sequence.keys != reference.keys:
                failures.append(_failure_from_record(record, "residue_keys_do_not_match_group_reference"))
                continue
            loaded.append((record, residue_sequence))

        if reference is None or not loaded:
            failures.append({"example_id": example_id, "reason": "no_valid_samples_for_group"})
            continue

        rows = _entropy_rows_for_group(
            example_id=example_id,
            loaded=loaded,
            reference=reference,
            region_specs=region_specs,
        )
        per_position_rows.extend(rows)
        summary_rows.extend(_summarize_position_rows(example_id=example_id, rows=rows, n_samples=len(loaded)))

    aggregate_rows = _summarize_position_rows(
        example_id="__all__",
        rows=per_position_rows,
        n_samples=None,
    )
    summary_rows.extend(aggregate_rows)

    return AnalysisResult(
        summary_df=pd.DataFrame(summary_rows),
        per_position_df=pd.DataFrame(per_position_rows),
        failures_df=pd.DataFrame(failures, columns=FAILURE_COLUMNS),
    )


def write_analysis_result(result: AnalysisResult, output_dir: str | Path, manifest: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary_df.to_csv(output_dir / "sequence_entropy_summary.csv", index=False)
    result.per_position_df.to_csv(output_dir / "sequence_entropy_per_position.csv", index=False)
    result.failures_df.to_csv(output_dir / "sequence_entropy_failures.csv", index=False)
    with (output_dir / "sequence_entropy_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _entropy_rows_for_group(
    *,
    example_id: str,
    loaded: Sequence[tuple[MetadataRecord, ResidueSequence]],
    reference: ResidueSequence,
    region_specs: Sequence[RegionSpec],
) -> list[dict[str, Any]]:
    rows = []
    letters_by_position = list(zip(*(residue_sequence.letters for _, residue_sequence in loaded), strict=True))
    for region in region_specs:
        region_mask = reference.region_masks.get(region.name)
        if region_mask is None:
            region_mask = tuple(True for _ in reference.letters)
        for position_index, in_region in enumerate(region_mask):
            if not in_region:
                continue
            letters = tuple(letters_by_position[position_index])
            entropy = sequence_entropy_bits(letters)
            chain_id, res_id, insertion_code, within_poly_res_idx = reference.keys[position_index]
            rows.append(
                {
                    "example_id": example_id,
                    "region": region.name,
                    "position_index": position_index,
                    "chain_id": chain_id,
                    "res_id": res_id,
                    "insertion_code": insertion_code,
                    "within_poly_res_idx": within_poly_res_idx,
                    "reference_letter": reference.letters[position_index],
                    "n_samples": len(letters),
                    "n_unique_aas": len(set(letters)),
                    "entropy_bits": entropy,
                    "normalized_entropy": entropy / NORMALIZATION_DENOMINATOR,
                    "aa_counts": _format_counts(letters),
                }
            )
    return rows


def _summarize_position_rows(
    *,
    example_id: str,
    rows: Sequence[dict[str, Any]],
    n_samples: int | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    rows_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_region[str(row["region"])].append(row)

    summary_rows = []
    for region, region_rows in sorted(rows_by_region.items()):
        entropies = np.asarray([row["entropy_bits"] for row in region_rows], dtype=float)
        normalized = np.asarray([row["normalized_entropy"] for row in region_rows], dtype=float)
        summary_rows.append(
            {
                "example_id": example_id,
                "region": region,
                "n_positions": int(len(region_rows)),
                "n_samples": n_samples if n_samples is not None else "",
                "mean_entropy_bits": float(np.mean(entropies)),
                "sum_entropy_bits": float(np.sum(entropies)),
                "max_entropy_bits": float(np.max(entropies)),
                "mean_normalized_entropy": float(np.mean(normalized)),
            }
        )
    return summary_rows


def sequence_entropy_bits(letters: Iterable[str]) -> float:
    counts = Counter(letter for letter in letters if letter)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return 0.0 if entropy == 0.0 else float(entropy)


def _build_cif_loader(
    *,
    region_specs: Sequence[RegionSpec],
    pocket_annotation_method: str,
    cif_parse_cfg: dict[str, Any] | None,
) -> Callable[[MetadataRecord], ResidueSequence]:
    parse_cfg = OmegaConf.create(cif_parse_cfg or DESIGNED_CIF_PARSE_CFG)
    resolved_pocket_annotation_method = resolve_pocket_annotation_method(
        pocket_annotation_method=pocket_annotation_method,
        use_calpha_for_pocket_annotation=pocket_annotation_method == "calpha",
    )
    needs_structure = any(region.requires_structure for region in region_specs)

    def load_residue_sequence(record: MetadataRecord) -> ResidueSequence:
        if record.designed_sample_path is None:
            if needs_structure:
                raise ValueError("designed_sample_path is required for pocket regions")
            return _residue_sequence_from_metadata(record)
        if not record.designed_sample_path.exists():
            raise FileNotFoundError(record.designed_sample_path)
        example = load_example_with_parse(str(record.designed_sample_path), parse_cfg)
        atom_array = _ensure_analysis_annotations(example["atom_array"])
        return residue_sequence_from_atom_array(
            atom_array,
            region_specs=region_specs,
            pocket_annotation_method=resolved_pocket_annotation_method,
        )

    return load_residue_sequence


def _ensure_analysis_annotations(atom_array: AtomArray) -> AtomArray:
    annotations = atom_array.get_annotation_categories()
    if "atomize" not in annotations:
        atom_array.set_annotation("atomize", np.zeros(len(atom_array), dtype=bool))
    if "is_covalent_modification" not in annotations:
        atom_array.set_annotation("is_covalent_modification", np.zeros(len(atom_array), dtype=bool))
    if "atom_is_protein_chain" not in annotations and "chain_type" in annotations:
        atom_array.set_annotation(
            "atom_is_protein_chain",
            atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L,
        )
    return atom_array


def residue_sequence_from_atom_array(
    atom_array: AtomArray,
    *,
    region_specs: Sequence[RegionSpec],
    pocket_annotation_method: str = "calpha",
) -> ResidueSequence:
    ca_mask = get_valid_standard_aa_residue_mask(atom_array) & (atom_array.atom_name == "CA")
    ca_indices = np.where(ca_mask)[0]
    keys = tuple(_residue_key(atom_array, atom_index, seq_index) for seq_index, atom_index in enumerate(ca_indices))
    letters = tuple(DICT_THREE_TO_ONE.get(str(atom_array.res_name[atom_index]), "X") for atom_index in ca_indices)
    region_masks = _region_masks_for_atom_array(
        atom_array,
        ca_mask=ca_mask,
        region_specs=region_specs,
        pocket_annotation_method=pocket_annotation_method,
    )
    return ResidueSequence(keys=keys, letters=letters, region_masks=region_masks)


def _region_masks_for_atom_array(
    atom_array: AtomArray,
    *,
    ca_mask: np.ndarray,
    region_specs: Sequence[RegionSpec],
    pocket_annotation_method: str,
) -> dict[str, tuple[bool, ...]]:
    ca_count = int(np.sum(ca_mask))
    region_masks: dict[str, tuple[bool, ...]] = {
        "all": tuple(True for _ in range(ca_count)),
    }
    distances = sorted(
        {
            distance
            for spec in region_specs
            for distance in (spec.lo, spec.hi)
            if distance is not None and float(distance) > 0.0
        }
    )
    edge_masks = {
        distance: _pocket_residue_mask_for_distance(
            atom_array,
            pocket_distance=float(distance),
            pocket_annotation_method=pocket_annotation_method,
        )[ca_mask]
        for distance in distances
    }
    for spec in region_specs:
        if not spec.requires_structure:
            continue
        hi_mask = edge_masks[float(spec.hi)]
        if spec.lo is None or float(spec.lo) == 0.0:
            region_mask = hi_mask
        else:
            region_mask = hi_mask & ~edge_masks[float(spec.lo)]
        region_masks[spec.name] = tuple(bool(value) for value in region_mask)
    return region_masks


def _pocket_residue_mask_for_distance(
    atom_array: AtomArray,
    *,
    pocket_distance: float,
    pocket_annotation_method: str,
) -> np.ndarray:
    annotation_name = f"sequence_entropy_pocket_{_format_distance_label(pocket_distance)}"
    atom_array = atom_array.copy()
    if pocket_annotation_method == "calpha":
        atom_array = annotate_ligand_pockets_calpha(
            atom_array=atom_array,
            pocket_distance=pocket_distance,
            n_min_ligand_atoms=1,
            annotation_name=annotation_name,
        )
    elif pocket_annotation_method == "pseudocb":
        atom_array = annotate_ligand_pockets_pseudocb(
            atom_array=atom_array,
            pocket_distance=pocket_distance,
            n_min_ligand_atoms=1,
            annotation_name=annotation_name,
        )
    else:
        atom_array = annotate_ligand_pockets(
            atom_array=atom_array,
            pocket_distance=pocket_distance,
            n_min_ligand_atoms=1,
            annotation_name=annotation_name,
        )
    return apply_and_spread_residue_wise(
        atom_array,
        atom_array.get_annotation(annotation_name),
        function=np.any,
    )


def _residue_sequence_from_metadata(record: MetadataRecord) -> ResidueSequence:
    if record.designed_sample_seq is None:
        raise ValueError("designed_sample_seq is missing")
    letters = tuple(letter for letter in record.designed_sample_seq.replace(":", "") if letter in AA20)
    keys = tuple(("", idx + 1, "", idx) for idx in range(len(letters)))
    return ResidueSequence(keys=keys, letters=letters, region_masks={"all": tuple(True for _ in letters)})


def _residue_key(atom_array: AtomArray, atom_index: int, sequence_index: int) -> tuple[str, int, str, int]:
    chain_id = str(atom_array.chain_id[atom_index])
    res_id = int(atom_array.res_id[atom_index])
    insertion_code = ""
    if "ins_code" in atom_array.get_annotation_categories():
        insertion_code = str(atom_array.ins_code[atom_index])
    within_poly_res_idx = sequence_index
    if "within_poly_res_idx" in atom_array.get_annotation_categories():
        within_poly_res_idx = int(atom_array.within_poly_res_idx[atom_index])
    return chain_id, res_id, insertion_code, within_poly_res_idx


def _format_counts(letters: Iterable[str]) -> str:
    counts = Counter(letters)
    return ";".join(f"{letter}:{counts[letter]}" for letter in sorted(counts))


def _format_distance_label(distance: float) -> str:
    value = float(distance)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def _failure_from_record(record: MetadataRecord, reason: str) -> dict[str, Any]:
    return _failure_row(
        metadata_path=record.metadata_path,
        example_id=record.example_id,
        designed_sample_id=record.designed_sample_id,
        designed_sample_path=record.designed_sample_path,
        reason=reason,
    )


def _failure_row(
    *,
    metadata_path: Path,
    reason: str,
    example_id: str | None = None,
    designed_sample_id: str | None = None,
    designed_sample_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "metadata_path": str(metadata_path),
        "example_id": example_id or "",
        "designed_sample_id": designed_sample_id or "",
        "designed_sample_path": str(designed_sample_path) if designed_sample_path is not None else "",
        "reason": reason,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-path",
        action="append",
        nargs="+",
        required=True,
        help="sample_metadata*.pt file, directory, or glob. Can be provided more than once.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--pocket-cumulative-distances",
        nargs="*",
        type=float,
        default=[6.0, 8.0, 10.0, 12.0],
    )
    parser.add_argument(
        "--pocket-shell-bins",
        nargs="*",
        default=["6:8", "8:10", "10:12"],
        help="Shell bins formatted as lo:hi or lo-hi.",
    )
    parser.add_argument(
        "--pocket-annotation-method",
        default="calpha",
        choices=["calpha", "pseudocb", "all_atom"],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    metadata_inputs = [item for group in args.metadata_path for item in group]
    metadata_paths = expand_metadata_paths(metadata_inputs)
    shell_bins = [parse_shell_bin(raw) for raw in args.pocket_shell_bins]
    region_specs = default_region_specs(
        cumulative_distances=args.pocket_cumulative_distances,
        shell_bins=shell_bins,
    )
    result = analyze_metadata(
        metadata_paths,
        region_specs=region_specs,
        pocket_annotation_method=args.pocket_annotation_method,
    )
    manifest = {
        "metadata_paths": [str(path) for path in metadata_paths],
        "region_specs": [spec.__dict__ for spec in region_specs],
        "pocket_annotation_method": args.pocket_annotation_method,
        "n_summary_rows": int(len(result.summary_df)),
        "n_per_position_rows": int(len(result.per_position_df)),
        "n_failures": int(len(result.failures_df)),
    }
    write_analysis_result(result, args.output_dir, manifest)
    if result.summary_df.empty:
        raise ValueError(
            "No valid sequence entropy rows were produced; "
            f"failure details were written to {Path(args.output_dir) / 'sequence_entropy_failures.csv'}"
        )
    print(f"Wrote sequence entropy analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
