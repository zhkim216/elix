"""Global integrity gates for denovoval LASErMPNN sampling and AF3 outputs."""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.denovoval.lasermpnn.backmap_designs import (
    compare_ligands,
)
from allatom_design.eval.benchmarks.denovoval.lasermpnn.sampling_provenance import (
    decoding_order_sha256,
    sample_seed,
)
from allatom_design.eval.benchmarks.denovoval.sequence_design_inputs import (
    atomic_write_json,
    cfg_path,
)
from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    protein_sequence,
    read_structure,
    sha256_file,
)


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/lasermpnn/default.yaml"
)
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sampling", "af3-inputs", "af3-complete"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--num-arrays", type=int, default=None)
    parser.add_argument("--predictions-per-design", type=int, default=None)
    return parser.parse_args()


class Diagnostics:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, str]] = []

    def add(self, category: str, key: str, message: str) -> None:
        self.counts[category] += 1
        if len(self.examples) < 50:
            self.examples.append(
                {"category": category, "key": str(key), "error": str(message)}
            )

    @property
    def failed(self) -> bool:
        return bool(sum(self.counts.values()))

    def payload(self) -> dict[str, Any]:
        return {
            "total_errors": int(sum(self.counts.values())),
            **{key: int(value) for key, value in sorted(self.counts.items())},
        }


def _finish(report_path: Path, report: dict[str, Any], diagnostics: Diagnostics) -> None:
    report["status"] = "failed" if diagnostics.failed else "complete"
    report["diagnostics"] = diagnostics.payload()
    report["error_examples"] = diagnostics.examples
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if diagnostics.failed:
        raise RuntimeError(f"Validation failed; see {report_path}")


def _staging_frame(cfg: DictConfig, *, smoke: bool) -> pd.DataFrame:
    path = cfg_path(cfg, "staging.full_manifest_csv")
    frame = pd.read_csv(path, keep_default_na=False)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id in {path}")
    if not frame["status"].eq("prepared").all():
        raise ValueError(f"Unprepared rows in {path}")
    if smoke:
        selected = set(map(str, cfg.dataset.smoke_sample_ids))
        frame = frame[frame["sample_id"].astype(str).isin(selected)].copy()
        observed = set(map(str, frame["sample_id"]))
        if observed != selected:
            raise ValueError(f"Smoke staging IDs {observed} != {selected}")
    return frame.reset_index(drop=True)


def _expected_manifest_paths(pattern: str, count: int) -> list[Path]:
    parent = Path(pattern).parent
    return [parent / f"shard_{index:02d}.csv" for index in range(count)]


def _load_exact_csv_shards(
    pattern: str,
    *,
    count: int,
    diagnostics: Diagnostics,
) -> tuple[pd.DataFrame, list[Path]]:
    expected = _expected_manifest_paths(pattern, count)
    observed = [Path(path) for path in sorted(glob.glob(pattern))]
    for path in sorted(set(expected) - set(observed)):
        diagnostics.add("missing_shard", path.name, str(path))
    for path in sorted(set(observed) - set(expected)):
        diagnostics.add("extra_shard", path.name, str(path))
    frames: list[pd.DataFrame] = []
    for path in observed:
        try:
            frame = pd.read_csv(path, keep_default_na=False)
        except Exception as exc:
            diagnostics.add("unreadable_shard", path.name, f"{type(exc).__name__}: {exc}")
            continue
        frame["_manifest_path"] = str(path)
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
        observed,
    )


def _backbone_coordinates(structure: Any, chain_id: str) -> tuple[list[tuple[int, str]], np.ndarray]:
    mask = (structure.chain_id.astype(str) == chain_id) & np.isin(
        structure.atom_name.astype(str), ["N", "CA", "C", "O"]
    )
    selected = structure[mask]
    keys = list(
        zip(
            map(int, selected.res_id),
            map(str, selected.atom_name),
            strict=True,
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate protein backbone atom keys")
    order = sorted(range(len(keys)), key=keys.__getitem__)
    return [keys[index] for index in order], selected.coord[order]


def _validate_sample_structure(
    record: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    sampled_path = Path(str(record["sampled_pdb_path"]))
    input_path = Path(str(record["input_pdb_path"]))
    if not sampled_path.is_file():
        raise FileNotFoundError(sampled_path)
    if sha256_file(sampled_path) != str(record["sampled_pdb_sha256"]):
        raise ValueError("sampled PDB SHA256 mismatch")
    if sha256_file(input_path) != str(record["input_pdb_sha256"]):
        raise ValueError("input PDB SHA256 mismatch")
    sampled = read_structure(sampled_path)
    source_structure = read_structure(input_path)
    if not np.all(np.isfinite(sampled.coord)):
        raise ValueError("sampled PDB has nonfinite coordinates")
    if "atom_id" in sampled.get_annotation_categories():
        serials = np.asarray(sampled.atom_id, dtype=int)
        if not np.array_equal(serials, np.arange(1, len(sampled) + 1)):
            raise ValueError("sampled PDB atom serials are not contiguous from 1")
    protein_chain = str(source["protein_chain_id"])
    ligand_chain = str(source["transport_ligand_chain_id"])
    observed_sequence = protein_sequence(sampled, protein_chain)
    if observed_sequence != str(record["protein_sequence"]):
        raise ValueError("sampled PDB/manifest protein sequence mismatch")
    if len(observed_sequence) != int(source["protein_length"]):
        raise ValueError("sampled protein length mismatch")
    source_keys, source_coords = _backbone_coordinates(source_structure, protein_chain)
    sampled_keys, sampled_coords = _backbone_coordinates(sampled, protein_chain)
    if source_keys != sampled_keys:
        raise ValueError("protein N/CA/C/O identity changed during sampling")
    expected_backbone_atoms = 4 * int(source["protein_length"])
    if len(sampled_keys) != expected_backbone_atoms:
        raise ValueError(
            f"protein backbone atom count {len(sampled_keys)} != {expected_backbone_atoms}"
        )
    coordinate_delta = sampled_coords - source_coords
    atom_errors = {
        atom_name: float(
            np.max(
                np.abs(
                    coordinate_delta[
                        [index for index, key in enumerate(source_keys) if key[1] == atom_name]
                    ]
                )
            )
        )
        for atom_name in ("N", "CA", "C", "O")
    }
    ca_delta = coordinate_delta[
        [index for index, key in enumerate(source_keys) if key[1] == "CA"]
    ]
    ca_rmsd = float(np.sqrt(np.mean(np.sum(ca_delta**2, axis=1))))
    n_ca_c_error = max(atom_errors[name] for name in ("N", "CA", "C"))
    # Stock NISE deliberately idealizes the N/CA/C frame and rebuilds O from
    # psi before writing rotamers.  The 154-CCD baseline has CA RMSD <=0.0243 A,
    # N/CA/C max error <=0.083 A, and O max error <=2.085 A.  Gate gross frame
    # movement by CA RMSD while reporting (not rejecting) the expected N/CA/C
    # idealization displacement and O reconstruction.
    if ca_rmsd > 0.05:
        raise ValueError(
            "protein frame moved beyond the stock-NISE idealization envelope: "
            f"CA_RMSD={ca_rmsd}"
        )
    ligand_stats = compare_ligands(source_structure, sampled, ligand_chain)
    source_ligand = source_structure[source_structure.chain_id.astype(str) == ligand_chain]
    sampled_ligand = sampled[sampled.chain_id.astype(str) == ligand_chain]
    for annotation in ("chain_id", "res_id", "res_name", "atom_name", "element", "hetero"):
        if not np.array_equal(
            np.asarray(getattr(source_ligand, annotation)).astype(str),
            np.asarray(getattr(sampled_ligand, annotation)).astype(str),
        ):
            raise ValueError(f"ligand {annotation} changed during sampling")
    if "CAP" in set(map(str, sampled.res_name)):
        raise ValueError("synthetic CAP residue present in sampled PDB")
    return {
        **ligand_stats,
        "input_ca_rmsd": ca_rmsd,
        "input_n_ca_c_coordinate_max_abs_error": n_ca_c_error,
        "input_oxygen_coordinate_max_abs_error": atom_errors["O"],
        "sampled_atom_count": len(sampled),
        "_backbone_keys": source_keys,
        "_backbone_coordinates": sampled_coords,
    }


def validate_sampling(cfg: DictConfig, *, smoke: bool, num_shards: int) -> None:
    diagnostics = Diagnostics()
    staging = _staging_frame(cfg, smoke=smoke)
    pattern = str(
        cfg.sampling.smoke_manifest_glob if smoke else cfg.sampling.manifest_glob
    )
    frame, manifest_paths = _load_exact_csv_shards(
        pattern, count=num_shards, diagnostics=diagnostics
    )
    required = {
        "source_index", "source_sample_id", "kind", "protein_length", "ccd_code",
        "pdb_ccd_code", "protein_chain_id", "ligand_chain_id",
        "transport_ligand_chain_id", "transport_ligand_res_name",
        "atom_mapping_path", "atom_mapping_sha256", "input_pdb_path",
        "input_pdb_sha256", "design_index", "designed_sample_id",
        "sampled_pdb_path", "sampled_pdb_sha256", "protein_sequence",
        "laser_nll", "laser_bs_nll", "decoding_order_local",
        "decoding_order_sha256", "decoding_order_length", "model_ligand_resnames",
        "model_ligand_atom_count", "model_cap_residue_count",
        "output_ligand_atom_count", "output_ligand_coordinate_max_error",
        "model_weights_sha256", "sample_seed", "sequence_temp",
        "first_shell_sequence_temp", "chi_temp", "seq_min_p", "chi_min_p",
        "disabled_residues", "ala_budget", "gly_budget", "status", "error",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        diagnostics.add("missing_columns", "sampling_manifests", ",".join(missing_columns))

    designs = int(cfg.sequence_design.expected_designs_per_sample)
    expected_keys = {
        (str(sample_id), design_index)
        for sample_id in staging["sample_id"]
        for design_index in range(1, designs + 1)
    }
    observed_keys: list[tuple[str, int]] = []
    if {"source_sample_id", "design_index"}.issubset(frame.columns):
        observed_keys = [
            (str(row.source_sample_id), int(row.design_index))
            for row in frame.itertuples(index=False)
        ]
        for key, count in Counter(observed_keys).items():
            if count != 1:
                diagnostics.add("duplicate_design_key", str(key), str(count))
        for key in sorted(expected_keys - set(observed_keys))[:100]:
            diagnostics.add("missing_design_key", str(key), "not present")
        for key in sorted(set(observed_keys) - expected_keys)[:100]:
            diagnostics.add("extra_design_key", str(key), "unexpected")

    expected_by_shard = {
        index: {
            str(row.sample_id)
            for source_position, row in enumerate(staging.itertuples(index=False))
            if source_position % num_shards == index
        }
        for index in range(num_shards)
    }
    if not frame.empty and "source_sample_id" in frame.columns:
        for shard_id, path in enumerate(_expected_manifest_paths(pattern, num_shards)):
            if path not in manifest_paths:
                continue
            shard = frame[frame["_manifest_path"] == str(path)]
            observed_sources = set(map(str, shard["source_sample_id"]))
            if observed_sources != expected_by_shard[shard_id]:
                diagnostics.add(
                    "shard_membership_mismatch",
                    path.name,
                    f"observed={len(observed_sources)} expected={len(expected_by_shard[shard_id])}",
                )
            expected_rows = len(expected_by_shard[shard_id]) * designs
            if len(shard) != expected_rows:
                diagnostics.add(
                    "shard_row_count_mismatch", path.name, f"{len(shard)} != {expected_rows}"
                )

    source_by_id = {
        str(row["sample_id"]): row.to_dict() for _, row in staging.iterrows()
    }
    expected_weight = str(cfg.sampling.model_weights_sha256)
    expected_temps = {
        "sequence_temp": float(cfg.sampling.defaults.sequence_temp),
        "first_shell_sequence_temp": float(
            cfg.sampling.defaults.first_shell_sequence_temp
        ),
        "chi_temp": float(cfg.sampling.defaults.chi_temp),
        "seq_min_p": float(cfg.sampling.defaults.seq_min_p),
        "chi_min_p": float(cfg.sampling.defaults.chi_min_p),
    }
    structure_maxima: dict[str, float] = {
        "validated_rows": 0,
        "ligand_coordinate_error": 0.0,
        "input_ca_rmsd": 0.0,
        "input_n_ca_c_error": 0.0,
        "input_oxygen_error": 0.0,
        "paired_design_backbone_error": 0.0,
    }
    backbones_by_source: dict[str, list[tuple[int, list[tuple[int, str]], np.ndarray]]] = {}
    if not missing_columns:
        for raw_record in frame.to_dict(orient="records"):
            key = str(raw_record["designed_sample_id"])
            try:
                sample_id = str(raw_record["source_sample_id"])
                source = source_by_id[sample_id]
                design_index = int(raw_record["design_index"])
                if key != f"lasermpnn_{sample_id}_{design_index}":
                    raise ValueError("designed_sample_id does not match source/design index")
                if str(raw_record["status"]) != "success" or str(raw_record["error"]):
                    raise ValueError(f"sampling status={raw_record['status']}: {raw_record['error']}")
                for column, source_column in (
                    ("source_index", "source_index"),
                    ("kind", "kind"),
                    ("protein_length", "protein_length"),
                    ("ccd_code", "ccd_code"),
                    ("pdb_ccd_code", "pdb_ccd_code"),
                    ("protein_chain_id", "protein_chain_id"),
                    ("ligand_chain_id", "ligand_chain_id"),
                    ("transport_ligand_chain_id", "transport_ligand_chain_id"),
                    ("transport_ligand_res_name", "transport_ligand_res_name"),
                    ("atom_mapping_path", "atom_mapping_path"),
                    ("atom_mapping_sha256", "atom_mapping_sha256"),
                    ("input_pdb_path", "protonated_pdb_path"),
                    ("input_pdb_sha256", "protonated_pdb_sha256"),
                ):
                    if str(raw_record[column]) != str(source[source_column]):
                        raise ValueError(f"sampling/staging mismatch for {column}")
                sequence = str(raw_record["protein_sequence"])
                if len(sequence) != int(source["protein_length"]):
                    raise ValueError("manifest protein sequence length mismatch")
                if set(sequence) - CANONICAL_AA:
                    raise ValueError(f"noncanonical sampled residues: {sorted(set(sequence) - CANONICAL_AA)}")
                if str(raw_record["model_weights_sha256"]) != expected_weight:
                    raise ValueError("model weights SHA256 mismatch")
                if int(raw_record["sample_seed"]) != sample_seed(sample_id, 0):
                    raise ValueError("sample_seed mismatch")
                for column, expected in expected_temps.items():
                    if float(raw_record[column]) != expected:
                        raise ValueError(f"{column}={raw_record[column]} != {expected}")
                if str(raw_record["disabled_residues"]) != str(
                    cfg.sampling.defaults.disabled_residues
                ):
                    raise ValueError("disabled_residues mismatch")
                if int(raw_record["model_cap_residue_count"]) != 0:
                    raise ValueError("model CAP count is nonzero")
                if str(raw_record["model_ligand_resnames"]) != str(
                    source["transport_ligand_res_name"]
                ):
                    raise ValueError("model ligand track identity mismatch")
                for column in ("laser_nll", "laser_bs_nll"):
                    if not math.isfinite(float(raw_record[column])):
                        raise ValueError(f"{column} is nonfinite")
                serialized_order = str(raw_record["decoding_order_local"])
                order = [int(value) for value in serialized_order.split()]
                expected_length = int(source["protein_length"])
                if len(order) != expected_length or set(order) != set(range(expected_length)):
                    raise ValueError("decoding order is not a full local residue permutation")
                if int(raw_record["decoding_order_length"]) != expected_length:
                    raise ValueError("decoding_order_length mismatch")
                if decoding_order_sha256(serialized_order) != str(
                    raw_record["decoding_order_sha256"]
                ):
                    raise ValueError("decoding order SHA256 mismatch")
                stats = _validate_sample_structure(raw_record, source)
                if int(raw_record["model_ligand_atom_count"]) != int(stats["ligand_atom_count"]):
                    raise ValueError("model ligand atom count mismatch")
                if int(raw_record["output_ligand_atom_count"]) != int(stats["ligand_atom_count"]):
                    raise ValueError("output ligand atom count mismatch")
                if float(raw_record["output_ligand_coordinate_max_error"]) != float(
                    stats["ligand_coordinate_max_abs_error"]
                ):
                    raise ValueError("output ligand coordinate error provenance mismatch")
                mapping_path = Path(str(raw_record["atom_mapping_path"]))
                if sha256_file(mapping_path) != str(raw_record["atom_mapping_sha256"]):
                    raise ValueError("atom-mapping sidecar SHA256 mismatch")
                if int(raw_record["ala_budget"]) != int(cfg.sampling.defaults.ala_budget):
                    raise ValueError("ala_budget mismatch")
                if int(raw_record["gly_budget"]) != int(cfg.sampling.defaults.gly_budget):
                    raise ValueError("gly_budget mismatch")
                structure_maxima["validated_rows"] += 1
                structure_maxima["ligand_coordinate_error"] = max(
                    structure_maxima["ligand_coordinate_error"],
                    float(stats["ligand_coordinate_max_abs_error"]),
                )
                structure_maxima["input_ca_rmsd"] = max(
                    structure_maxima["input_ca_rmsd"], float(stats["input_ca_rmsd"])
                )
                structure_maxima["input_n_ca_c_error"] = max(
                    structure_maxima["input_n_ca_c_error"],
                    float(stats["input_n_ca_c_coordinate_max_abs_error"]),
                )
                structure_maxima["input_oxygen_error"] = max(
                    structure_maxima["input_oxygen_error"],
                    float(stats["input_oxygen_coordinate_max_abs_error"]),
                )
                backbones_by_source.setdefault(sample_id, []).append(
                    (
                        design_index,
                        list(stats["_backbone_keys"]),
                        np.asarray(stats["_backbone_coordinates"]),
                    )
                )
            except Exception as exc:
                diagnostics.add("invalid_design", key, f"{type(exc).__name__}: {exc}")

    for sample_id, entries in backbones_by_source.items():
        entries = sorted(entries, key=lambda item: item[0])
        if len(entries) != designs:
            continue
        reference_keys = entries[0][1]
        reference_coordinates = entries[0][2]
        for design_index, keys, coordinates in entries[1:]:
            if keys != reference_keys:
                diagnostics.add(
                    "paired_design_backbone_identity",
                    sample_id,
                    f"design {design_index} backbone keys differ",
                )
                continue
            pair_error = float(np.max(np.abs(coordinates - reference_coordinates)))
            structure_maxima["paired_design_backbone_error"] = max(
                structure_maxima["paired_design_backbone_error"], pair_error
            )
            if pair_error > 0.00051:
                diagnostics.add(
                    "paired_design_backbone_coordinates",
                    sample_id,
                    f"design {design_index} max error {pair_error}",
                )

    if not frame.empty and {"source_sample_id", "decoding_order_sha256"}.issubset(frame.columns):
        for sample_id, group in frame.groupby("source_sample_id", sort=False):
            if group["decoding_order_sha256"].nunique() != len(group):
                diagnostics.add(
                    "duplicate_decoding_order",
                    str(sample_id),
                    f"{group['decoding_order_sha256'].nunique()} unique for {len(group)} designs",
                )

    samples_dir = Path(pattern).parent.parent / "pdbs"
    observed_pdbs = set(samples_dir.glob("*.pdb")) if samples_dir.is_dir() else set()
    expected_pdbs = {
        samples_dir / f"{sample_id}_{design_index}.pdb"
        for sample_id, design_index in expected_keys
    }
    for path in sorted(expected_pdbs - observed_pdbs)[:100]:
        diagnostics.add("missing_sampled_pdb", path.name, str(path))
    for path in sorted(observed_pdbs - expected_pdbs)[:100]:
        diagnostics.add("extra_sampled_pdb", path.name, str(path))

    report_path = cfg_path(
        cfg, "sampling.smoke_validation_json" if smoke else "sampling.validation_json"
    )
    report = {
        "stage": "sampling",
        "mode": "smoke" if smoke else "full",
        "staging_manifest": str(cfg_path(cfg, "staging.full_manifest_csv")),
        "staging_manifest_sha256": sha256_file(cfg_path(cfg, "staging.full_manifest_csv")),
        "sampling_manifests": [str(path) for path in manifest_paths],
        "num_shards": num_shards,
        "expected_source_rows": len(staging),
        "expected_design_rows": len(expected_keys),
        "observed_design_rows": len(frame),
        "expected_sampled_pdbs": len(expected_pdbs),
        "observed_sampled_pdbs": len(observed_pdbs),
        "unique_ccd_codes": int(staging["ccd_code"].nunique()),
        "kind_counts": staging["kind"].value_counts().to_dict(),
        "protein_length_counts": staging["protein_length"].value_counts().to_dict(),
        "validated_structure_rows": int(structure_maxima["validated_rows"]),
        "max_ligand_coordinate_abs_error": structure_maxima["ligand_coordinate_error"],
        "max_input_ca_rmsd": structure_maxima["input_ca_rmsd"],
        "max_input_n_ca_c_coordinate_abs_error": structure_maxima["input_n_ca_c_error"],
        "max_input_oxygen_coordinate_abs_error": structure_maxima["input_oxygen_error"],
        "max_paired_design_backbone_coordinate_abs_error": structure_maxima[
            "paired_design_backbone_error"
        ],
        "af3_submission_allowed": not diagnostics.failed,
    }
    _finish(report_path, report, diagnostics)


def _af3_paths(cfg: DictConfig, *, smoke: bool) -> tuple[Path, Path]:
    output_root = cfg_path(cfg, "af3.smoke_output_dir" if smoke else "af3.full_output_dir")
    manifest = cfg_path(
        cfg, "backmapping.smoke_manifest_csv" if smoke else "backmapping.full_manifest_csv"
    )
    return output_root, manifest


def _expected_chunk_slice(total: int, count: int, array_id: int) -> tuple[int, int]:
    chunk_size = max(1, math.ceil(total / count))
    start = min(total, array_id * chunk_size)
    return start, min(total, start + chunk_size)


def _validate_af3_input_artifacts(
    cfg: DictConfig,
    *,
    smoke: bool,
    num_arrays: int,
    allowed_statuses: set[str],
    diagnostics: Diagnostics,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    output_root, manifest_path = _af3_paths(cfg, smoke=smoke)
    if not manifest_path.is_file():
        diagnostics.add("missing_design_manifest", manifest_path.name, str(manifest_path))
        return pd.DataFrame(), pd.DataFrame(), output_root, manifest_path
    design = pd.read_csv(manifest_path, keep_default_na=False)
    expected_designs = (
        len(list(cfg.dataset.smoke_sample_ids))
        if smoke
        else int(cfg.dataset.expected_count)
    ) * int(cfg.sequence_design.expected_designs_per_sample)
    if len(design) != expected_designs:
        diagnostics.add("design_manifest_row_count", manifest_path.name, f"{len(design)} != {expected_designs}")
    if design.get("designed_sample_id", pd.Series(dtype=str)).duplicated().any():
        diagnostics.add("duplicate_design_id", manifest_path.name, "duplicate designed_sample_id")
    if "status" in design and not design["status"].isin(["backmapped", "reused"]).all():
        diagnostics.add("invalid_backmapping_status", manifest_path.name, "failed design rows")
    chunk_order = design
    if smoke and {"source_sample_id", "design_index"}.issubset(design.columns):
        requested_order = {
            (str(source_id), design_index): order_index
            for order_index, (source_id, design_index) in enumerate(
                (
                    (source_id, design_index)
                    for source_id in map(str, cfg.dataset.smoke_sample_ids)
                    for design_index in range(
                        1, int(cfg.sequence_design.expected_designs_per_sample) + 1
                    )
                )
            )
        }
        chunk_order = design.copy()
        chunk_order["_requested_order"] = [
            requested_order.get((str(source_id), int(design_index)), -1)
            for source_id, design_index in zip(
                chunk_order["source_sample_id"],
                chunk_order["design_index"],
                strict=True,
            )
        ]
        if (chunk_order["_requested_order"] < 0).any():
            diagnostics.add(
                "smoke_requested_order",
                manifest_path.name,
                "design manifest contains IDs outside the configured smoke matrix",
            )
        chunk_order = chunk_order.sort_values("_requested_order").drop(
            columns="_requested_order"
        )

    reports_dir = output_root / "reports" / "chunks"
    expected_csvs = [reports_dir / f"status_{index:04d}.csv" for index in range(num_arrays)]
    expected_jsons = [reports_dir / f"status_{index:04d}.json" for index in range(num_arrays)]
    observed_csvs = set(reports_dir.glob("status_*.csv")) if reports_dir.is_dir() else set()
    observed_jsons = set(reports_dir.glob("status_*.json")) if reports_dir.is_dir() else set()
    for path in sorted(set(expected_csvs) - observed_csvs):
        diagnostics.add("missing_status_csv", path.name, str(path))
    for path in sorted(observed_csvs - set(expected_csvs)):
        diagnostics.add("extra_status_csv", path.name, str(path))
    for path in sorted(set(expected_jsons) - observed_jsons):
        diagnostics.add("missing_status_json", path.name, str(path))
    for path in sorted(observed_jsons - set(expected_jsons)):
        diagnostics.add("extra_status_json", path.name, str(path))

    status_frames: list[pd.DataFrame] = []
    manifest_sha = sha256_file(manifest_path)
    for array_id, (csv_path, json_path) in enumerate(zip(expected_csvs, expected_jsons, strict=True)):
        if not csv_path.is_file() or not json_path.is_file():
            continue
        try:
            chunk = pd.read_csv(csv_path, keep_default_na=False)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            start, end = _expected_chunk_slice(len(design), num_arrays, array_id)
            expected_ids = list(
                map(str, chunk_order.iloc[start:end]["designed_sample_id"])
            )
            observed_ids = list(map(str, chunk["designed_sample_id"]))
            if observed_ids != expected_ids:
                raise ValueError("status chunk IDs/order do not match design manifest slice")
            if payload.get("manifest_sha256") != manifest_sha:
                raise ValueError("status JSON references a stale design manifest")
            plan = payload.get("plan", {})
            if int(plan.get("array_id", -1)) != array_id or int(plan.get("start", -1)) != start or int(plan.get("end", -1)) != end:
                raise ValueError("status JSON chunk plan mismatch")
            if not chunk["status"].isin(allowed_statuses).all():
                raise ValueError(f"unexpected statuses: {chunk['status'].value_counts().to_dict()}")
            if chunk.get("error", pd.Series(dtype=str)).astype(str).str.len().sum():
                raise ValueError("status chunk contains nonempty errors")
            chunk["_status_path"] = str(csv_path)
            status_frames.append(chunk)
        except Exception as exc:
            diagnostics.add("invalid_status_chunk", csv_path.name, f"{type(exc).__name__}: {exc}")
    status = pd.concat(status_frames, ignore_index=True) if status_frames else pd.DataFrame()
    if len(status) != len(design):
        diagnostics.add("status_row_count", "all_chunks", f"{len(status)} != {len(design)}")

    input_dir = output_root / "inputs"
    observed_inputs = set(input_dir.glob("*.json")) if input_dir.is_dir() else set()
    expected_inputs = {
        input_dir / f"{sample_id}.json"
        for sample_id in design.get("designed_sample_id", pd.Series(dtype=str)).astype(str)
    }
    for path in sorted(expected_inputs - observed_inputs)[:100]:
        diagnostics.add("missing_af3_json", path.name, str(path))
    for path in sorted(observed_inputs - expected_inputs)[:100]:
        diagnostics.add("extra_af3_json", path.name, str(path))

    design_by_id = {
        str(row["designed_sample_id"]): row
        for row in design.to_dict(orient="records")
    }
    status_by_id = {
        str(row["designed_sample_id"]): row
        for row in status.to_dict(orient="records")
    }
    for path in sorted(expected_inputs & observed_inputs):
        sample_id = path.stem
        try:
            record = design_by_id[sample_id]
            status_record = status_by_id[sample_id]
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("name") != sample_id:
                raise ValueError("JSON name mismatch")
            if payload.get("dialect") != "alphafold3" or int(payload.get("version", -1)) != int(cfg.af3.json_config.version):
                raise ValueError("AF3 dialect/version mismatch")
            if list(payload.get("modelSeeds", [])) != list(cfg.af3.json_config.model_seeds):
                raise ValueError("AF3 model seeds mismatch")
            sequences = payload.get("sequences", [])
            if len(sequences) != 2 or "protein" not in sequences[0] or "ligand" not in sequences[1]:
                raise ValueError("AF3 JSON must contain one protein then one ligand")
            protein = sequences[0]["protein"]
            ligand = sequences[1]["ligand"]
            if protein.get("id") != str(record["protein_chain_id"]) or protein.get("sequence") != str(record["protein_sequence"]):
                raise ValueError("AF3 protein chain/sequence mismatch")
            if protein.get("unpairedMsa") != "" or protein.get("pairedMsa") != "" or protein.get("templates") != []:
                raise ValueError("AF3 input is not single-sequence/no-template mode")
            if ligand.get("id") != str(record["ligand_chain_id"]) or ligand.get("ccdCodes") != [str(record["ccd_code"])]:
                raise ValueError("AF3 ligand chain/CCD mismatch")
            if str(status_record.get("json_sha256", "")) != sha256_file(path):
                raise ValueError("AF3 JSON/status SHA256 mismatch")
            for field in ("protein_sequence", "protein_chain_id", "ligand_chain_id", "ligand_ccd_code", "backmapped_cif_sha256"):
                expected_field = record["ccd_code"] if field == "ligand_ccd_code" else record.get(field)
                if str(status_record.get(field, "")) != str(expected_field):
                    raise ValueError(f"AF3 status/design mismatch for {field}")
        except Exception as exc:
            diagnostics.add("invalid_af3_json", sample_id, f"{type(exc).__name__}: {exc}")
    return design, status, output_root, manifest_path


def validate_af3_inputs(cfg: DictConfig, *, smoke: bool, num_arrays: int) -> None:
    diagnostics = Diagnostics()
    design, status, output_root, manifest_path = _validate_af3_input_artifacts(
        cfg,
        smoke=smoke,
        num_arrays=num_arrays,
        allowed_statuses={"input_ready", "complete"},
        diagnostics=diagnostics,
    )
    expected_ccds = 2 if smoke else int(cfg.dataset.expected_unique_ccd_count)
    observed_ccds = int(design["ccd_code"].nunique()) if "ccd_code" in design else 0
    if observed_ccds != expected_ccds:
        diagnostics.add("ccd_coverage", "design_manifest", f"{observed_ccds} != {expected_ccds}")
    report_path = output_root / "reports" / "validation_inputs.json"
    report = {
        "stage": "af3-inputs",
        "mode": "smoke" if smoke else "full",
        "design_manifest": str(manifest_path),
        "design_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "",
        "num_arrays": num_arrays,
        "design_rows": len(design),
        "status_rows": len(status),
        "unique_ccd_codes": observed_ccds,
        "protein_mode": "single_sequence_no_templates",
        "af3_submission_allowed": not diagnostics.failed,
    }
    _finish(report_path, report, diagnostics)


def _metric_files(
    directory: Path,
    stem: str,
    count: int,
    diagnostics: Diagnostics,
) -> list[Path]:
    expected = [directory / f"{stem}_{index:04d}.csv" for index in range(count)]
    observed = set(directory.glob(f"{stem}_*.csv")) if directory.is_dir() else set()
    for path in sorted(set(expected) - observed):
        diagnostics.add("missing_metric_shard", path.name, str(path))
    for path in sorted(observed - set(expected)):
        diagnostics.add("extra_metric_shard", path.name, str(path))
    return [path for path in expected if path.is_file()]


def validate_af3_complete(
    cfg: DictConfig,
    *,
    smoke: bool,
    num_arrays: int,
    predictions_per_design: int,
) -> None:
    diagnostics = Diagnostics()
    design, status, output_root, manifest_path = _validate_af3_input_artifacts(
        cfg,
        smoke=smoke,
        num_arrays=num_arrays,
        allowed_statuses={"complete"},
        diagnostics=diagnostics,
    )
    if not status.empty:
        numeric_expectations = {
            "n_predictions_expected": predictions_per_design,
            "n_predictions_found": predictions_per_design,
            "n_malformed": 0,
            "n_surplus": 0,
            "sc_metric_rows": predictions_per_design,
            "docking_metric_rows": predictions_per_design,
            "metric_error_rows": 0,
        }
        for column, expected in numeric_expectations.items():
            if column not in status or not np.all(pd.to_numeric(status[column]) == expected):
                diagnostics.add("invalid_status_metric_count", column, f"expected every row == {expected}")
        if "input_fingerprint_ok" not in status or not status["input_fingerprint_ok"].astype(str).str.lower().eq("true").all():
            diagnostics.add("invalid_input_fingerprint", "status", "not all fingerprints are true")
        if "input_fingerprint_error" in status and status["input_fingerprint_error"].astype(str).str.len().sum():
            diagnostics.add("stale_input_fingerprint", "status", "nonempty fingerprint errors")

    metrics_dir = output_root / "metrics" / "chunks"
    sc_paths = _metric_files(metrics_dir, "self_consistency", num_arrays, diagnostics)
    docking_paths = _metric_files(metrics_dir, "docking", num_arrays, diagnostics)
    try:
        sc = pd.concat([pd.read_csv(path, keep_default_na=False) for path in sc_paths], ignore_index=True)
    except Exception as exc:
        diagnostics.add("unreadable_sc_metrics", "self_consistency", f"{type(exc).__name__}: {exc}")
        sc = pd.DataFrame()
    try:
        docking = pd.concat([pd.read_csv(path, keep_default_na=False) for path in docking_paths], ignore_index=True)
    except Exception as exc:
        diagnostics.add("unreadable_docking_metrics", "docking", f"{type(exc).__name__}: {exc}")
        docking = pd.DataFrame()
    expected_metric_rows = len(design) * predictions_per_design
    for name, frame in (("self_consistency", sc), ("docking", docking)):
        if len(frame) != expected_metric_rows:
            diagnostics.add("metric_row_count", name, f"{len(frame)} != {expected_metric_rows}")
        if "metric_status" not in frame or not frame["metric_status"].eq("ok").all():
            diagnostics.add("metric_status_error", name, "not all metric_status values are ok")
        if "error" in frame and frame["error"].astype(str).str.len().sum():
            diagnostics.add("nonempty_metric_error", name, "metric error column is nonempty")
        if {"designed_sample_id", "diffusion_index"}.issubset(frame.columns):
            observed = Counter(
                zip(
                    map(str, frame["designed_sample_id"]),
                    map(int, frame["diffusion_index"]),
                )
            )
            expected = {
                (str(sample_id), index)
                for sample_id in design.get("designed_sample_id", pd.Series(dtype=str))
                for index in range(predictions_per_design)
            }
            if set(observed) != expected or any(value != 1 for value in observed.values()):
                diagnostics.add("metric_key_matrix", name, "design/diffusion matrix is incomplete or duplicated")
    finite_metric_columns = {
        "self_consistency": (sc, ["sc_ca_rmsd", "avg_ca_plddt"]),
        "docking": (
            docking,
            [
                "ligand_rmsd", "binding_site_rmsd", "num_bs_residues",
                "ligand_plddt", "binding_site_plddt", "iptm",
                "interface_min_pae",
            ],
        ),
    }
    for metric_name, (frame, columns) in finite_metric_columns.items():
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            diagnostics.add(
                "missing_metric_columns", metric_name, ",".join(missing)
            )
            continue
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
            diagnostics.add(
                "nonfinite_metric_values", metric_name, ",".join(columns)
            )
    if {"sc_ca_rmsd", "avg_ca_plddt"}.issubset(sc.columns):
        sc_rmsd = pd.to_numeric(sc["sc_ca_rmsd"], errors="coerce").to_numpy()
        sc_plddt = pd.to_numeric(sc["avg_ca_plddt"], errors="coerce").to_numpy()
        if np.any(sc_rmsd < 0.0) or np.any((sc_plddt < 0.0) | (sc_plddt > 100.0)):
            diagnostics.add("invalid_sc_metric_range", "self_consistency", "RMSD or pLDDT outside valid range")
    if {"ligand_rmsd", "binding_site_rmsd", "ligand_plddt", "binding_site_plddt", "iptm", "interface_min_pae"}.issubset(docking.columns):
        ligand_rmsd = pd.to_numeric(docking["ligand_rmsd"], errors="coerce").to_numpy()
        binding_rmsd = pd.to_numeric(docking["binding_site_rmsd"], errors="coerce").to_numpy()
        ligand_plddt = pd.to_numeric(docking["ligand_plddt"], errors="coerce").to_numpy()
        binding_plddt = pd.to_numeric(docking["binding_site_plddt"], errors="coerce").to_numpy()
        iptm = pd.to_numeric(docking["iptm"], errors="coerce").to_numpy()
        pae = pd.to_numeric(docking["interface_min_pae"], errors="coerce").to_numpy()
        if (
            np.any(ligand_rmsd < 0.0)
            or np.any(binding_rmsd < 0.0)
            or np.any((ligand_plddt < 0.0) | (ligand_plddt > 100.0))
            or np.any((binding_plddt < 0.0) | (binding_plddt > 100.0))
            or np.any((iptm < 0.0) | (iptm > 1.0))
            or np.any(pae < 0.0)
        ):
            diagnostics.add("invalid_docking_metric_range", "docking", "one or more metrics outside valid range")
    tmalign_columns = {
        "tmalign_score", "tmalign_rmsd", "tmalign_matched_ca_count",
        "tmalign_sample_ca_count", "tmalign_pred_ca_count",
        "tmalign_sample_ca_coverage", "tmalign_pred_ca_coverage",
        "tmalign_label_span_count", "tmalign_excluded_label_positions",
        "tmalign_modified_ca_count",
    }
    missing_tmalign = sorted(tmalign_columns - set(sc.columns))
    if missing_tmalign:
        diagnostics.add("missing_tmalign_columns", "self_consistency", ",".join(missing_tmalign))
    else:
        numeric = sc[list(tmalign_columns)].apply(pd.to_numeric, errors="coerce")
        if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
            diagnostics.add("nonfinite_tmalign", "self_consistency", "TM-align fields contain nonfinite values")
        scores = numeric["tmalign_score"].to_numpy(dtype=float)
        if np.any((scores < 0.0) | (scores > 1.00001)):
            diagnostics.add("invalid_tmalign_score", "self_consistency", "TM-score outside [0,1]")
        if np.any(numeric["tmalign_matched_ca_count"].to_numpy(dtype=float) < 3):
            diagnostics.add("insufficient_tmalign_matches", "self_consistency", "fewer than three matched CA atoms")
        for column in ("tmalign_sample_ca_coverage", "tmalign_pred_ca_coverage"):
            values = numeric[column].to_numpy(dtype=float)
            if np.any((values < 0.0) | (values > 1.0)):
                diagnostics.add("invalid_tmalign_coverage", column, "coverage outside [0,1]")

    prediction_root = output_root / "predictions"
    observed_prediction_dirs = {
        path.name for path in prediction_root.iterdir() if path.is_dir()
    } if prediction_root.is_dir() else set()
    expected_prediction_dirs = set(
        map(str, design.get("designed_sample_id", pd.Series(dtype=str)))
    )
    if observed_prediction_dirs != expected_prediction_dirs:
        diagnostics.add(
            "prediction_directory_matrix",
            "predictions",
            f"observed={len(observed_prediction_dirs)} expected={len(expected_prediction_dirs)}",
        )

    report_path = output_root / "reports" / "validation_complete.json"
    report = {
        "stage": "af3-complete",
        "mode": "smoke" if smoke else "full",
        "design_manifest": str(manifest_path),
        "design_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else "",
        "num_arrays": num_arrays,
        "design_rows": len(design),
        "predictions_per_design": predictions_per_design,
        "expected_prediction_rows": expected_metric_rows,
        "status_rows": len(status),
        "self_consistency_rows": len(sc),
        "docking_rows": len(docking),
        "prediction_directories": len(observed_prediction_dirs),
        "tmalign_score_min": float(pd.to_numeric(sc["tmalign_score"], errors="coerce").min()) if "tmalign_score" in sc and len(sc) else None,
        "tmalign_score_max": float(pd.to_numeric(sc["tmalign_score"], errors="coerce").max()) if "tmalign_score" in sc and len(sc) else None,
    }
    _finish(report_path, report, diagnostics)


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.stage == "sampling":
        num_shards = args.num_shards or (2 if args.smoke else 20)
        validate_sampling(cfg, smoke=args.smoke, num_shards=num_shards)
    elif args.stage == "af3-inputs":
        num_arrays = args.num_arrays or (2 if args.smoke else 20)
        validate_af3_inputs(cfg, smoke=args.smoke, num_arrays=num_arrays)
    else:
        num_arrays = args.num_arrays or (2 if args.smoke else 20)
        predictions = args.predictions_per_design
        if predictions is None:
            predictions = 1 if args.smoke else int(cfg.af3.inference_config.ss.num_diffusion_samples)
        validate_af3_complete(
            cfg,
            smoke=args.smoke,
            num_arrays=num_arrays,
            predictions_per_design=predictions,
        )


if __name__ == "__main__":
    main()
