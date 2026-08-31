"""Merge protein-only ABACUS-T designs with canonical denovoval ligands."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from biotite.structure import AtomArray, get_residue_starts
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    prepare_pdb_structure,
    protein_sequence,
    read_structure,
    restore_ligand_ccd_and_roles,
    sha256_file,
    unique_ligand_ccd,
    validate_pdb_roundtrip,
    validate_semantic_cif_entities,
    write_semantic_cif,
)


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/abacus-t/default.yaml"
)
DESIGN_MANIFEST_COLUMNS = {
    "source_sample_id",
    "designed_sample_id",
    "model_name",
    "design_index",
    "protein_sequence",
    "sampled_pdb_path",
    "sampled_pdb_sha256",
    "fasta_path",
    "fasta_sha256",
    "iteration",
    "source_staging_manifest_sha256",
    "source_preparation_manifest_sha256",
    "run_fingerprint",
    "status",
}
STAGING_COLUMNS = {
    "sample_id",
    "source_cif_path",
    "source_cif_sha256",
    "staged_pdb_path",
    "staged_pdb_sha256",
    "original_ccd_code",
    "pdb_ccd_code",
    "protein_chain_id",
    "ligand_chain_id",
    "protein_length",
    "kind",
    "status",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def cfg_path(cfg: DictConfig, key: str) -> Path:
    value = OmegaConf.select(cfg, key)
    if value is None:
        raise ValueError(f"Missing required config path: {key}")
    return Path(str(value)).expanduser()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _annotation_or_default(
    structure: AtomArray,
    name: str,
    default: Any,
    *,
    dtype: Any,
) -> np.ndarray:
    if name in structure.get_annotation_categories():
        return np.asarray(getattr(structure, name), dtype=dtype)
    if callable(default):
        return np.asarray(default(structure), dtype=dtype)
    return np.full(len(structure), default, dtype=dtype)


def _residue_keys(structure: AtomArray) -> list[tuple[int, str]]:
    starts = get_residue_starts(structure)
    return [
        (int(structure.res_id[index]), str(structure.ins_code[index]))
        for index in starts
    ]


def _backbone(structure: AtomArray) -> tuple[list[tuple[int, str, str]], np.ndarray]:
    mask = np.isin(structure.atom_name.astype(str), ["N", "CA", "C", "O"])
    selected = structure[mask]
    keys = list(
        zip(
            map(int, selected.res_id),
            map(str, selected.ins_code),
            map(str, selected.atom_name),
            strict=True,
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate protein backbone atom keys")
    order = sorted(range(len(keys)), key=keys.__getitem__)
    return [keys[index] for index in order], selected.coord[order]


def validate_sampled_protein(
    sampled: AtomArray,
    source: AtomArray,
    *,
    protein_chain_id: str,
    expected_sequence: str,
) -> dict[str, float | int]:
    observed_chains = set(map(str, sampled.chain_id))
    if observed_chains != {protein_chain_id}:
        raise ValueError(
            f"ABACUS-T sampled PDB must contain only chain {protein_chain_id!r}, "
            f"got {sorted(observed_chains)}"
        )
    if np.any(sampled.hetero):
        raise ValueError("ABACUS-T sampled protein contains HETATM records")
    if not np.all(np.isfinite(sampled.coord)):
        raise ValueError("ABACUS-T sampled protein has nonfinite coordinates")
    sequence = protein_sequence(sampled, protein_chain_id)
    if sequence != expected_sequence:
        raise ValueError("Sampled PDB sequence differs from the design manifest")

    source_protein = source[source.chain_id.astype(str) == protein_chain_id]
    if _residue_keys(sampled) != _residue_keys(source_protein):
        raise ValueError("Sampled/source protein residue identifiers differ")
    source_keys, source_coordinates = _backbone(source_protein)
    sampled_keys, sampled_coordinates = _backbone(sampled)
    if sampled_keys != source_keys:
        raise ValueError("Sampled/source N/CA/C/O atom topology differs")
    if len(sampled_keys) != 4 * len(expected_sequence):
        raise ValueError(
            f"Expected four backbone atoms for each of {len(expected_sequence)} residues"
        )
    coordinate_error = float(
        np.max(np.abs(sampled_coordinates - source_coordinates))
    )
    # ABACUS-T may update backbone coordinates during sampling. Preserve that
    # sampled backbone in the semantic CIF; the subsequent roundtrip validator
    # proves that writing the CIF does not change it.
    return {
        "protein_atom_count": len(sampled),
        "protein_backbone_atom_count": len(sampled_keys),
        "protein_backbone_coordinate_max_abs_error": coordinate_error,
    }


def merge_sampled_protein_with_source_ligand(
    sampled_protein: AtomArray,
    source_structure: AtomArray,
    *,
    protein_chain_id: str,
    ligand_chain_id: str,
    original_ccd_code: str,
    pdb_ccd_code: str,
) -> AtomArray:
    """Create a complex while retaining canonical ligand atom annotations."""
    ligand = source_structure[
        source_structure.chain_id.astype(str) == ligand_chain_id
    ]
    if len(ligand) == 0:
        raise ValueError(f"Canonical source has no ligand chain {ligand_chain_id!r}")
    if unique_ligand_ccd(ligand, ligand_chain_id) != original_ccd_code:
        raise ValueError("Canonical source ligand CCD differs from staging manifest")
    aliased_ligand = prepare_pdb_structure(
        ligand,
        ligand_chain_id=ligand_chain_id,
        original_ccd_code=original_ccd_code,
        pdb_ccd_code=pdb_ccd_code,
    )
    combined = sampled_protein + aliased_ligand

    for name, default, dtype in (
        ("occupancy", 1.0, float),
        ("b_factor", 0.0, float),
        ("charge", 0, int),
        ("auth_asym_id", lambda atoms: atoms.chain_id, str),
        ("auth_seq_id", lambda atoms: atoms.res_id.astype(str), str),
    ):
        values = np.concatenate(
            [
                _annotation_or_default(
                    sampled_protein, name, default, dtype=dtype
                ),
                _annotation_or_default(aliased_ligand, name, default, dtype=dtype),
            ]
        )
        combined.set_annotation(name, values)
    combined.set_annotation("atom_id", np.arange(1, len(combined) + 1, dtype=int))
    return restore_ligand_ccd_and_roles(
        combined,
        protein_chain_id=protein_chain_id,
        ligand_chain_id=ligand_chain_id,
        pdb_ccd_code=pdb_ccd_code,
        original_ccd_code=original_ccd_code,
    )


def validate_ligand_immutable(
    source: AtomArray,
    observed: AtomArray,
    *,
    ligand_chain_id: str,
) -> dict[str, float | int]:
    left = source[source.chain_id.astype(str) == ligand_chain_id]
    right = observed[observed.chain_id.astype(str) == ligand_chain_id]
    if len(left) != len(right):
        raise ValueError(f"Ligand atom count changed: {len(left)} != {len(right)}")
    categorical = (
        "chain_id",
        "res_id",
        "ins_code",
        "res_name",
        "atom_name",
        "element",
        "hetero",
    )
    for annotation in categorical:
        if not np.array_equal(
            np.asarray(getattr(left, annotation)).astype(str),
            np.asarray(getattr(right, annotation)).astype(str),
        ):
            raise ValueError(f"Canonical ligand {annotation} changed")
    coordinate_error = float(np.max(np.abs(left.coord - right.coord)))
    if coordinate_error > 0.00051:
        raise ValueError(f"Canonical ligand coordinates changed: {coordinate_error}")
    numeric_errors: dict[str, float] = {}
    for annotation, default, tolerance in (
        ("occupancy", 1.0, 0.0051),
        ("b_factor", 0.0, 0.0051),
        ("charge", 0, 0.0),
    ):
        left_values = _annotation_or_default(left, annotation, default, dtype=float)
        right_values = _annotation_or_default(right, annotation, default, dtype=float)
        if not np.array_equal(np.isfinite(left_values), np.isfinite(right_values)):
            raise ValueError(f"Canonical ligand {annotation} finite mask changed")
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        error = (
            float(np.max(np.abs(left_values[finite] - right_values[finite])))
            if np.any(finite)
            else 0.0
        )
        numeric_errors[annotation] = error
        if error > tolerance:
            raise ValueError(f"Canonical ligand {annotation} changed: {error}")
    return {
        "ligand_atom_count": len(right),
        "ligand_coordinate_max_abs_error": coordinate_error,
        **{
            f"ligand_{name}_max_abs_error": value
            for name, value in numeric_errors.items()
        },
    }


def validate_sampled_protein_roundtrip(
    sampled: AtomArray,
    observed_complex: AtomArray,
    *,
    protein_chain_id: str,
) -> dict[str, Any]:
    """Prove semantic CIF writing preserved every sampled protein atom."""
    observed = observed_complex[
        observed_complex.chain_id.astype(str) == protein_chain_id
    ]
    expected = sampled.copy()
    if (
        "atom_id" in expected.get_annotation_categories()
        and "atom_id" in observed.get_annotation_categories()
    ):
        expected.atom_id = observed.atom_id.copy()
    validation = validate_pdb_roundtrip(expected, observed)
    if not validation["ok"]:
        raise ValueError(
            "Sampled protein fields changed during semantic CIF writing: "
            + "; ".join(map(str, validation["errors"]))
        )
    return validation


def _load_design_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sampling design manifest: {path}")
    output = pd.read_csv(path, keep_default_na=False)
    missing = sorted(DESIGN_MANIFEST_COLUMNS - set(output.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    output["sampling_manifest_path"] = str(path)
    if output["designed_sample_id"].duplicated().any():
        duplicates = output.loc[
            output["designed_sample_id"].duplicated(keep=False),
            "designed_sample_id",
        ].head(20)
        raise ValueError(f"Duplicate designed sample IDs: {duplicates.tolist()}")
    return output


def _validate_matrix(
    staging: pd.DataFrame,
    sampling: pd.DataFrame,
    *,
    design_indices: list[int],
) -> None:
    expected = {
        (str(sample_id), design_index)
        for sample_id in staging["sample_id"]
        for design_index in design_indices
    }
    observed_counter = Counter(
        zip(
            map(str, sampling["source_sample_id"]),
            map(int, sampling["design_index"]),
            strict=True,
        )
    )
    observed = set(observed_counter)
    duplicates = sorted(key for key, count in observed_counter.items() if count != 1)
    if observed != expected or duplicates:
        raise ValueError(
            "Sampling design matrix differs: "
            f"missing={sorted(expected - observed)[:20]}, "
            f"extra={sorted(observed - expected)[:20]}, duplicates={duplicates[:20]}"
        )


def run(cfg: DictConfig, *, smoke: bool, overwrite: bool) -> dict[str, Any]:
    mode = "smoke" if smoke else "full"
    staging_manifest = cfg_path(
        cfg, "staging.smoke_manifest_csv" if smoke else "staging.full_manifest_csv"
    )
    staging_digest = sha256_file(staging_manifest)
    staging = pd.read_csv(staging_manifest, keep_default_na=False)
    missing_staging = sorted(STAGING_COLUMNS - set(staging.columns))
    if missing_staging:
        raise ValueError(f"{staging_manifest} is missing columns: {missing_staging}")
    if not staging["status"].eq("complete").all():
        raise ValueError("Staging manifest contains incomplete rows")
    if staging["sample_id"].duplicated().any():
        raise ValueError("Staging manifest contains duplicate sample IDs")

    mode_cfg = cfg.sampling[mode]
    sampling_manifest = Path(str(mode_cfg.design_manifest_csv)).expanduser()
    sampling = _load_design_manifest(sampling_manifest)
    design_indices = list(map(int, cfg.sequence_design.design_indices))
    output_slot_by_index = {
        design_index: output_slot
        for output_slot, design_index in enumerate(design_indices)
    }
    condition = str(OmegaConf.select(cfg, "dataset.condition", default=""))
    _validate_matrix(staging, sampling, design_indices=design_indices)
    if not sampling["model_name"].eq("abacus-t").all():
        raise ValueError("Sampling manifests contain a non-ABACUS-T model name")
    if not sampling["status"].isin(["complete", "reused"]).all():
        raise ValueError("Sampling manifests contain incomplete rows")
    if not pd.to_numeric(sampling["iteration"]).eq(19).all():
        raise ValueError("Sampling manifests contain a non-final iteration")
    if not sampling["source_staging_manifest_sha256"].eq(staging_digest).all():
        raise ValueError("Sampling manifests reference a stale staging manifest")
    preparation_manifest = cfg_path(cfg, f"preparation.{mode}.manifest_csv")
    preparation_digest = sha256_file(preparation_manifest)
    if not sampling["source_preparation_manifest_sha256"].eq(
        preparation_digest
    ).all():
        raise ValueError("Sampling manifest references a stale preparation manifest")

    output_root = cfg_path(cfg, f"backmapping.{mode}.root")
    cif_root = cfg_path(cfg, f"backmapping.{mode}.cif_dir")
    manifest_path = cfg_path(cfg, f"backmapping.{mode}.manifest_csv")
    validation_path = cfg_path(cfg, f"backmapping.{mode}.validation_json")
    source_by_id = staging.set_index("sample_id", drop=False)
    rows: list[dict[str, Any]] = []
    error_examples: list[dict[str, str]] = []

    for design in sampling.sort_values(
        ["source_sample_id", "design_index"], kind="stable"
    ).itertuples(index=False):
        source_id = str(design.source_sample_id)
        design_index = int(design.design_index)
        designed_id = str(design.designed_sample_id)
        output_cif = cif_root / f"{designed_id}.cif"
        row: dict[str, Any] = {
            "mode": mode,
            "condition": condition,
            "source_sample_id": source_id,
            "designed_sample_id": designed_id,
            "model_name": "abacus-t",
            "design_index": design_index,
            "output_slot": output_slot_by_index[design_index],
            "partial_diffusion_source_index": "",
            "iteration": int(design.iteration),
            "protein_sequence": str(design.protein_sequence),
            "protein_chain_id": "",
            "ligand_chain_id": "",
            "ccd_code": "",
            "pdb_ccd_code": "",
            "kind": "",
            "protein_length": "",
            "source_cif_path": "",
            "source_cif_sha256": "",
            "staged_pdb_path": "",
            "staged_pdb_sha256": "",
            "sampling_manifest_path": str(design.sampling_manifest_path),
            "sampled_pdb_path": str(design.sampled_pdb_path),
            "sampled_pdb_sha256": str(design.sampled_pdb_sha256),
            "fasta_path": str(design.fasta_path),
            "fasta_sha256": str(design.fasta_sha256),
            "run_fingerprint": str(design.run_fingerprint),
            "source_staging_manifest_sha256": staging_digest,
            "backmapped_cif_path": str(output_cif),
            "backmapped_cif_sha256": "",
            "status": "failed",
            "error": "",
        }
        try:
            if source_id not in source_by_id.index:
                raise KeyError(f"Unknown source sample {source_id}")
            source = source_by_id.loc[source_id]
            row["partial_diffusion_source_index"] = int(
                source.get("partial_diffusion_source_index", source["source_index"])
            )
            expected_id = f"{source_id}_design_{design_index}_19"
            if designed_id != expected_id:
                raise ValueError(f"Designed sample ID {designed_id} != {expected_id}")
            source_cif = Path(str(source["source_cif_path"]))
            staged_pdb = Path(str(source["staged_pdb_path"]))
            sampled_pdb = Path(str(design.sampled_pdb_path))
            fasta = Path(str(design.fasta_path))
            for path, expected_digest, label in (
                (source_cif, str(source["source_cif_sha256"]), "source CIF"),
                (staged_pdb, str(source["staged_pdb_sha256"]), "staged PDB"),
                (sampled_pdb, str(design.sampled_pdb_sha256), "sampled PDB"),
                (fasta, str(design.fasta_sha256), "FASTA"),
            ):
                if not path.is_file() or sha256_file(path) != expected_digest:
                    raise ValueError(f"Missing or stale {label}: {path}")
            protein_chain = str(source["protein_chain_id"])
            ligand_chain = str(source["ligand_chain_id"])
            source_structure = read_structure(source_cif)
            sampled_structure = read_structure(sampled_pdb)
            protein_stats = validate_sampled_protein(
                sampled_structure,
                source_structure,
                protein_chain_id=protein_chain,
                expected_sequence=str(design.protein_sequence),
            )
            if len(str(design.protein_sequence)) != int(source["protein_length"]):
                raise ValueError("Design sequence length differs from staging manifest")
            restored = merge_sampled_protein_with_source_ligand(
                sampled_structure,
                source_structure,
                protein_chain_id=protein_chain,
                ligand_chain_id=ligand_chain,
                original_ccd_code=str(source["original_ccd_code"]),
                pdb_ccd_code=str(source["pdb_ccd_code"]),
            )
            status = "reused"
            if overwrite or not output_cif.is_file():
                write_semantic_cif(output_cif, restored)
                status = "backmapped"
            reparsed = read_structure(output_cif)
            protein_roundtrip = validate_sampled_protein_roundtrip(
                sampled_structure,
                reparsed,
                protein_chain_id=protein_chain,
            )
            if protein_sequence(reparsed, protein_chain) != str(design.protein_sequence):
                raise ValueError("Protein sequence changed during semantic CIF writing")
            if unique_ligand_ccd(reparsed, ligand_chain) != str(
                source["original_ccd_code"]
            ):
                raise ValueError("Semantic CIF did not restore the original CCD")
            ligand_stats = validate_ligand_immutable(
                source_structure, reparsed, ligand_chain_id=ligand_chain
            )
            validate_semantic_cif_entities(
                output_cif,
                protein_chain_id=protein_chain,
                ligand_chain_id=ligand_chain,
                original_ccd_code=str(source["original_ccd_code"]),
                pdb_ccd_code=str(source["pdb_ccd_code"]),
            )
            row.update(
                {
                    "protein_chain_id": protein_chain,
                    "ligand_chain_id": ligand_chain,
                    "ccd_code": str(source["original_ccd_code"]),
                    "pdb_ccd_code": str(source["pdb_ccd_code"]),
                    "kind": str(source["kind"]),
                    "protein_length": int(source["protein_length"]),
                    "source_cif_path": str(source_cif),
                    "source_cif_sha256": str(source["source_cif_sha256"]),
                    "staged_pdb_path": str(staged_pdb),
                    "staged_pdb_sha256": str(source["staged_pdb_sha256"]),
                    "backmapped_cif_sha256": sha256_file(output_cif),
                    "status": status,
                    **protein_stats,
                    "protein_cif_coordinate_max_abs_error": protein_roundtrip[
                        "coordinate_max_abs_error"
                    ],
                    **ligand_stats,
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if len(error_examples) < 50:
                error_examples.append(
                    {"designed_sample_id": designed_id, "error": row["error"]}
                )
        rows.append(row)

    output = pd.DataFrame(rows)
    failed = int((output["status"] == "failed").sum())
    report = {
        "stage": "backmapping",
        "mode": mode,
        "status": "failed" if failed else "complete",
        "staging_manifest": str(staging_manifest),
        "staging_manifest_sha256": staging_digest,
        "sampling_manifest": str(sampling_manifest),
        "sampling_manifest_sha256": sha256_file(sampling_manifest),
        "preparation_manifest": str(preparation_manifest),
        "preparation_manifest_sha256": preparation_digest,
        "expected_rows": len(staging) * len(design_indices),
        "observed_rows": len(output),
        "failed_rows": failed,
        "missing_rows": len(staging) * len(design_indices) - len(output),
        "error_examples": error_examples,
        "output_root": str(output_root),
        "manifest_csv": str(manifest_path),
    }
    atomic_write_json(validation_path, report)
    if failed:
        raise RuntimeError(f"Backmapping failed; see {validation_path}")
    atomic_write_csv(manifest_path, output)
    return report


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    report = run(cfg, smoke=args.smoke, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
