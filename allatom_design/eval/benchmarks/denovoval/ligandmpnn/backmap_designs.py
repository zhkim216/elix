from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    protein_sequence,
    read_structure,
    restore_ligand_ccd_and_roles,
    sha256_file,
    unique_ligand_ccd,
    validate_semantic_cif_entities,
    validate_pdb_roundtrip,
    write_semantic_cif,
)


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/ligandmpnn/default.yaml"
)
SEED_PATTERN = re.compile(r"(?:^|,\s*)seed=(?P<seed>\d+)(?:,|$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backmap ProteinMPNN/LigandMPNN PDB designs to semantic CIFs."
    )
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
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def fasta_seed(path: Path) -> int | None:
    if not path.exists():
        return None
    seeds = {
        int(match.group("seed"))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(">")
        for match in [SEED_PATTERN.search(line)]
        if match is not None
    }
    if len(seeds) > 1:
        raise ValueError(f"Multiple seeds recorded in {path}: {sorted(seeds)}")
    return next(iter(seeds)) if seeds else None


def fasta_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence_parts)))
            header = line[1:]
            sequence_parts = []
        elif header is None:
            raise ValueError(f"FASTA sequence precedes header in {path}")
        else:
            sequence_parts.append(line)
    if header is not None:
        records.append((header, "".join(sequence_parts)))
    if not records:
        raise ValueError(f"No FASTA records found in {path}")
    return records


def validate_ligand_context_preserved(
    staged_structure,
    sampled_structure,
    *,
    ligand_chain_id: str,
) -> dict[str, Any]:
    """Compare ligand context while ignoring writer-owned serial/B-factor fields.

    LigandMPNN renumbers atom serials when it combines designed protein atoms
    with the ligand Selection, and ProDy zeroes some ligand B-factors. Neither
    field is consumed as ligand identity or geometry. All categorical fields,
    coordinates, occupancy, and formal charge remain strict.
    """
    staged_ligand = staged_structure[
        staged_structure.chain_id.astype(str) == ligand_chain_id
    ].copy()
    sampled_ligand = sampled_structure[
        sampled_structure.chain_id.astype(str) == ligand_chain_id
    ]
    for annotation in ("atom_id", "b_factor"):
        if (
            annotation in staged_ligand.get_annotation_categories()
            and annotation in sampled_ligand.get_annotation_categories()
        ):
            setattr(
                staged_ligand,
                annotation,
                getattr(sampled_ligand, annotation).copy(),
            )
    return validate_pdb_roundtrip(staged_ligand, sampled_ligand)


def expected_sample_outputs(
    staging: pd.DataFrame,
    *,
    sampling_root: Path,
    model_names: list[str],
    expected_designs: int,
) -> tuple[set[Path], set[Path]]:
    pdb_paths: set[Path] = set()
    fasta_paths: set[Path] = set()
    for model_name in model_names:
        model_root = sampling_root / model_name
        for sample_id in staging["sample_id"]:
            fasta_paths.add(model_root / "seqs" / f"{sample_id}.fa")
            for design_index in range(1, expected_designs + 1):
                pdb_paths.add(
                    model_root / "backbones" / f"{sample_id}_{design_index}.pdb"
                )
    return pdb_paths, fasta_paths


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    mode = "smoke" if args.smoke else "full"
    staging_manifest = cfg_path(
        cfg,
        "staging.smoke_manifest_csv" if args.smoke else "staging.full_manifest_csv",
    )
    sampling_root = cfg_path(
        cfg,
        "ligandmpnn.smoke_sampling_root" if args.smoke else "ligandmpnn.full_sampling_root",
    )
    cif_root = cfg_path(
        cfg,
        "backmapping.smoke_cif_root" if args.smoke else "backmapping.full_cif_root",
    )
    design_manifest = cfg_path(
        cfg,
        "backmapping.smoke_manifest_csv" if args.smoke else "backmapping.full_manifest_csv",
    )
    validation_json = cfg_path(
        cfg,
        "backmapping.smoke_validation_json" if args.smoke else "backmapping.full_validation_json",
    )
    expected_designs = int(cfg.ligandmpnn.expected_designs_per_sample)
    protein_chain_id = str(cfg.structure.protein_chain_id)
    ligand_chain_id = str(cfg.structure.ligand_chain_id)
    model_names = list(map(str, cfg.sequence_design.model_names))
    missing_model_configs = sorted(set(model_names) - set(cfg.ligandmpnn.models.keys()))
    if missing_model_configs:
        raise ValueError(f"Missing LigandMPNN model configs: {missing_model_configs}")
    design_indices = list(
        map(
            int,
            OmegaConf.select(
                cfg,
                "sequence_design.design_indices",
                default=list(range(1, expected_designs + 1)),
            ),
        )
    )
    output_slot_by_index = {
        design_index: output_slot
        for output_slot, design_index in enumerate(design_indices)
    }
    condition = str(OmegaConf.select(cfg, "dataset.condition", default=""))

    staging = pd.read_csv(staging_manifest, keep_default_na=False)
    required = {
        "sample_id",
        "kind",
        "protein_length",
        "ccd_code",
        "pdb_ccd_code",
        "source_cif_path",
        "source_cif_sha256",
        "staged_pdb_path",
        "staged_pdb_sha256",
        "status",
    }
    missing_columns = sorted(required - set(staging.columns))
    if missing_columns:
        raise ValueError(f"{staging_manifest} is missing columns: {missing_columns}")
    if not staging["status"].isin(["staged", "reused"]).all():
        raise ValueError(f"Staging manifest contains failed rows: {staging_manifest}")
    if staging["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {staging_manifest}")

    expected_pdbs, expected_fastas = expected_sample_outputs(
        staging,
        sampling_root=sampling_root,
        model_names=model_names,
        expected_designs=expected_designs,
    )
    observed_pdbs = {
        path
        for model_name in model_names
        for path in (sampling_root / model_name / "backbones").glob("*.pdb")
    }
    observed_fastas = {
        path
        for model_name in model_names
        for path in (sampling_root / model_name / "seqs").glob("*.fa")
    }
    missing_pdbs = sorted(map(str, expected_pdbs - observed_pdbs))
    extra_pdbs = sorted(map(str, observed_pdbs - expected_pdbs))
    missing_fastas = sorted(map(str, expected_fastas - observed_fastas))
    extra_fastas = sorted(map(str, observed_fastas - expected_fastas))

    rows: list[dict[str, Any]] = []
    for _, source_row in staging.iterrows():
        sample_id = str(source_row["sample_id"])
        source_cif_path = Path(str(source_row["source_cif_path"]))
        staged_pdb_path = Path(str(source_row["staged_pdb_path"]))
        if sha256_file(source_cif_path) != str(source_row["source_cif_sha256"]):
            raise ValueError(f"Source CIF digest mismatch: {source_cif_path}")
        if sha256_file(staged_pdb_path) != str(source_row["staged_pdb_sha256"]):
            raise ValueError(f"Staged PDB digest mismatch: {staged_pdb_path}")
        staged_structure = read_structure(staged_pdb_path)
        staged_sequence = protein_sequence(staged_structure, protein_chain_id)
        for model_name in model_names:
            model_root = sampling_root / model_name
            fasta_path = model_root / "seqs" / f"{sample_id}.fa"
            seed_error = ""
            try:
                seed = fasta_seed(fasta_path)
                records = fasta_records(fasta_path)
                if len(records) != expected_designs + 1:
                    raise ValueError(
                        f"Expected native+{expected_designs} FASTA records, "
                        f"found {len(records)}"
                    )
                if records[0][1] != staged_sequence:
                    raise ValueError("Native FASTA sequence differs from staged PDB")
            except Exception as exc:
                seed = None
                records = []
                seed_error = f"{type(exc).__name__}: {exc}"
            for design_index in range(1, expected_designs + 1):
                sampled_pdb = model_root / "backbones" / f"{sample_id}_{design_index}.pdb"
                designed_sample_id = f"{model_name}_{sample_id}_{design_index}"
                backmapped_cif = cif_root / model_name / f"{sample_id}_{design_index}.cif"
                row: dict[str, Any] = {
                    "mode": mode,
                    "condition": condition,
                    "model_name": model_name,
                    "source_sample_id": sample_id,
                    "partial_diffusion_source_index": int(
                        source_row.get(
                            "partial_diffusion_source_index",
                            source_row["source_index"],
                        )
                    ),
                    "design_index": design_index,
                    "output_slot": output_slot_by_index[design_index],
                    "designed_sample_id": designed_sample_id,
                    "kind": str(source_row["kind"]),
                    "protein_length": int(source_row["protein_length"]),
                    "ccd_code": str(source_row["ccd_code"]),
                    "pdb_ccd_code": str(source_row["pdb_ccd_code"]),
                    "protein_chain_id": protein_chain_id,
                    "ligand_chain_id": ligand_chain_id,
                    "source_cif_path": str(source_row["source_cif_path"]),
                    "staged_pdb_path": str(source_row["staged_pdb_path"]),
                    "fasta_path": str(fasta_path),
                    "sampled_pdb_path": str(sampled_pdb),
                    "backmapped_cif_path": str(backmapped_cif),
                    "sampling_seed": seed if seed is not None else "",
                    "status": "failed",
                    "error": "",
                }
                try:
                    if seed_error:
                        raise ValueError(f"Invalid FASTA seed metadata: {seed_error}")
                    if not sampled_pdb.exists():
                        raise FileNotFoundError(sampled_pdb)
                    sampled_structure = read_structure(sampled_pdb)
                    expected_header, expected_sequence = records[design_index]
                    if not re.search(rf"(?:^|,\s*)id={design_index}(?:,|$)", expected_header):
                        raise ValueError(
                            f"FASTA record {design_index} lacks matching id metadata"
                        )
                    sampled_sequence = protein_sequence(
                        sampled_structure, protein_chain_id
                    )
                    if sampled_sequence != expected_sequence:
                        raise ValueError(
                            f"FASTA/PDB sequence mismatch for {designed_sample_id}"
                        )
                    ligand_validation = validate_ligand_context_preserved(
                        staged_structure,
                        sampled_structure,
                        ligand_chain_id=ligand_chain_id,
                    )
                    if not ligand_validation["ok"]:
                        raise ValueError(
                            "Sampled PDB changed ligand atoms: "
                            + "; ".join(map(str, ligand_validation["errors"]))
                        )
                    restored_structure = restore_ligand_ccd_and_roles(
                        sampled_structure,
                        protein_chain_id=protein_chain_id,
                        ligand_chain_id=ligand_chain_id,
                        pdb_ccd_code=str(source_row["pdb_ccd_code"]),
                        original_ccd_code=str(source_row["ccd_code"]),
                    )
                    status = "reused"
                    if args.overwrite or not backmapped_cif.exists():
                        write_semantic_cif(backmapped_cif, restored_structure)
                        status = "backmapped"
                    entity_validation = validate_semantic_cif_entities(
                        backmapped_cif,
                        protein_chain_id=protein_chain_id,
                        ligand_chain_id=ligand_chain_id,
                        original_ccd_code=str(source_row["ccd_code"]),
                        pdb_ccd_code=str(source_row["pdb_ccd_code"]),
                    )
                    observed_cif = read_structure(backmapped_cif)
                    validation = validate_pdb_roundtrip(restored_structure, observed_cif)
                    if not validation["ok"]:
                        raise ValueError("; ".join(map(str, validation["errors"])))
                    observed_ccd = unique_ligand_ccd(observed_cif, ligand_chain_id)
                    if observed_ccd != str(source_row["ccd_code"]):
                        raise ValueError(
                            f"Backmapped ligand CCD {observed_ccd!r} != {source_row['ccd_code']!r}"
                        )
                    row.update(
                        {
                            "status": status,
                            "protein_sequence": protein_sequence(
                                observed_cif, protein_chain_id
                            ),
                            "sampled_pdb_sha256": sha256_file(sampled_pdb),
                            "backmapped_cif_sha256": sha256_file(backmapped_cif),
                            "n_atoms": len(observed_cif),
                            "coordinate_max_abs_error": validation[
                                "coordinate_max_abs_error"
                            ],
                            "categorical_mismatch_count": int(
                                sum(
                                    validation[
                                        "categorical_mismatch_counts"
                                    ].values()
                                )
                            ),
                            "ligand_coordinate_max_abs_error": ligand_validation[
                                "coordinate_max_abs_error"
                            ],
                            "protein_entity_id": entity_validation[
                                "protein_entity_id"
                            ],
                            "ligand_entity_id": entity_validation[
                                "ligand_entity_id"
                            ],
                        }
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)

    frame = pd.DataFrame(rows)
    atomic_write_csv(design_manifest, frame)
    failed = frame[frame["status"] == "failed"]
    status_counts = Counter(frame["status"])
    expected_rows = len(staging) * len(model_names) * expected_designs
    report = {
        "status": (
            "complete"
            if failed.empty
            and not missing_pdbs
            and not extra_pdbs
            and not missing_fastas
            and not extra_fastas
            and len(frame) == expected_rows
            else "failed"
        ),
        "mode": mode,
        "staging_manifest": str(staging_manifest),
        "staging_manifest_sha256": sha256_file(staging_manifest),
        "sampling_root": str(sampling_root),
        "design_manifest": str(design_manifest),
        "design_manifest_sha256": sha256_file(design_manifest),
        "source_count": len(staging),
        "model_count": len(model_names),
        "expected_designs_per_sample": expected_designs,
        "expected_design_rows": expected_rows,
        "observed_design_rows": len(frame),
        "status_counts": dict(status_counts),
        "diagnostics": {
            "failed": len(failed),
            "missing_pdb": len(missing_pdbs),
            "extra_pdb": len(extra_pdbs),
            "missing_fasta": len(missing_fastas),
            "extra_fasta": len(extra_fastas),
            "duplicate_design_id": int(frame["designed_sample_id"].duplicated().sum()),
            "unmapped": 0,
            "skipped": 0,
            "categorical_mismatches": int(
                frame.get("categorical_mismatch_count", pd.Series(dtype=int)).fillna(0).sum()
            ),
        },
        "examples": {
            "failed": failed[["designed_sample_id", "error"]]
            .head(20)
            .to_dict(orient="records"),
            "missing_pdb": missing_pdbs[:20],
            "extra_pdb": extra_pdbs[:20],
            "missing_fasta": missing_fastas[:20],
            "extra_fasta": extra_fastas[:20],
        },
        "contract": {
            "loader": "atomworks.read_any -> get_structure",
            "include_bonds": False,
            "ccd_backmapping": "sampled PDB alias -> source manifest CCD",
            "protein_entity": f"{protein_chain_id}_1 polypeptide(L)",
            "ligand_entity": f"{ligand_chain_id}_1 non-polymer",
            "staged_to_sampled_ligand_ignored_fields": ["atom_id", "b_factor"],
        },
    }
    atomic_write_json(validation_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "complete":
        raise RuntimeError(f"Backmapping validation failed; see {validation_json}")


if __name__ == "__main__":
    main()
