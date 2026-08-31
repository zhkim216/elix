from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.denovoval.ligandmpnn.backmap_designs import (
    validate_ligand_context_preserved,
)
from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
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
    "denovoval/ADFLIP/default.yaml"
)
MODES = ("per_ccd_smoke", "full")
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backmap ADFLIP backbone PDB designs to semantic CIFs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=MODES, default="per_ccd_smoke")
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


def load_sampling_shards(manifest_dir: Path, expected_shards: int) -> tuple[pd.DataFrame, list[Path]]:
    expected_paths = [manifest_dir / f"shard_{index:03d}.csv" for index in range(expected_shards)]
    missing = [str(path) for path in expected_paths if not path.is_file()]
    extras = sorted(
        str(path) for path in manifest_dir.glob("shard_*.csv") if path not in expected_paths
    )
    if missing or extras:
        raise ValueError(
            f"Sampling shard inventory mismatch: missing={missing[:20]}, extras={extras[:20]}"
        )
    frames = []
    for path in expected_paths:
        frame = pd.read_csv(path, keep_default_na=False)
        frame["_sampling_manifest_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), expected_paths


def _annotation(structure: Any, name: str, default: str = "") -> np.ndarray:
    if name in structure.get_annotation_categories():
        return np.asarray(getattr(structure, name)).astype(str)
    return np.full(len(structure), default, dtype=str)


def validate_backbone_preserved(
    staged_structure: Any,
    sampled_structure: Any,
    *,
    protein_chain_id: str,
) -> dict[str, Any]:
    staged_mask = (staged_structure.chain_id.astype(str) == protein_chain_id) & np.isin(
        staged_structure.atom_name.astype(str), list(BACKBONE_ATOMS)
    )
    sampled_mask = sampled_structure.chain_id.astype(str) == protein_chain_id
    staged = staged_structure[staged_mask]
    sampled = sampled_structure[sampled_mask]
    errors: list[str] = []
    if len(staged) != len(sampled):
        errors.append(f"atom_count:{len(staged)}!={len(sampled)}")
    n_compare = min(len(staged), len(sampled))
    mismatch_counts: dict[str, int] = {}
    for annotation in ("chain_id", "res_id", "ins_code", "atom_name", "element", "hetero"):
        left = _annotation(staged, annotation)[:n_compare]
        right = _annotation(sampled, annotation)[:n_compare]
        mismatches = int(np.sum(left != right))
        mismatch_counts[annotation] = mismatches
        if mismatches:
            errors.append(f"{annotation}_mismatches:{mismatches}")
    coordinate_max_abs_error = (
        float(np.max(np.abs(staged.coord[:n_compare] - sampled.coord[:n_compare])))
        if n_compare
        else 0.0
    )
    if coordinate_max_abs_error > 0.00051:
        errors.append(f"coordinate_max_abs_error:{coordinate_max_abs_error:.8g}")
    return {
        "ok": not errors,
        "errors": errors,
        "coordinate_max_abs_error": coordinate_max_abs_error,
        "categorical_mismatch_counts": mismatch_counts,
        "n_staged_atoms": len(staged),
        "n_sampled_atoms": len(sampled),
    }


def validate_inputs(
    staging: pd.DataFrame,
    sampling: pd.DataFrame,
    *,
    expected_count: int,
    expected_designs_per_sample: int,
) -> None:
    staging_required = {
        "source_index",
        "sample_id",
        "kind",
        "protein_length",
        "ccd_code",
        "pdb_ccd_code",
        "protein_chain_id",
        "ligand_chain_id",
        "source_cif_path",
        "source_cif_sha256",
        "staged_pdb_path",
        "staged_pdb_sha256",
        "status",
    }
    sampling_required = {
        "source_index",
        "source_sample_id",
        "sample_index",
        "designed_sample_id",
        "model_name",
        "design_index",
        "sampled_pdb_path",
        "sampled_pdb_sha256",
        "protein_sequence",
        "seed",
        "status",
    }
    missing_staging = sorted(staging_required - set(staging.columns))
    missing_sampling = sorted(sampling_required - set(sampling.columns))
    if missing_staging or missing_sampling:
        raise ValueError(
            f"Manifest columns missing: staging={missing_staging}, sampling={missing_sampling}"
        )
    expected_sampling_count = expected_count * expected_designs_per_sample
    if len(staging) != expected_count or len(sampling) != expected_sampling_count:
        raise ValueError(
            f"Manifest row count mismatch: staging={len(staging)}, "
            f"sampling={len(sampling)}, expected_staging={expected_count}, "
            f"expected_sampling={expected_sampling_count}"
        )
    if not staging["status"].isin(["staged", "reused"]).all():
        raise ValueError("Staging manifest contains failed rows")
    if not (sampling["status"] == "sampled").all():
        raise ValueError("Sampling manifests contain failed rows")
    if staging["sample_id"].duplicated().any():
        raise ValueError("Staging manifest sample IDs are not one-to-one")
    if set(staging["sample_id"].astype(str)) != set(sampling["source_sample_id"].astype(str)):
        raise ValueError("Staging and sampling sample membership differs")
    expected_design_indices = set(range(expected_designs_per_sample))
    if set(sampling["sample_index"].astype(int)) != expected_design_indices:
        raise ValueError(
            "ADFLIP sample_index set differs from configured designs: "
            f"{sorted(set(sampling['sample_index'].astype(int)))} != "
            f"{sorted(expected_design_indices)}"
        )
    if set(sampling["design_index"].astype(int)) != expected_design_indices:
        raise ValueError(
            "ADFLIP design_index set differs from configured designs: "
            f"{sorted(set(sampling['design_index'].astype(int)))} != "
            f"{sorted(expected_design_indices)}"
        )
    source_design_indices = sampling.groupby("source_sample_id")["design_index"].agg(
        lambda values: set(map(int, values))
    )
    if not source_design_indices.map(lambda value: value == expected_design_indices).all():
        raise ValueError("Sampling manifest source/design matrix is incomplete")
    if not sampling["sample_index"].astype(int).eq(
        sampling["design_index"].astype(int)
    ).all():
        raise ValueError("ADFLIP sample_index and design_index differ")
    if set(sampling["model_name"].astype(str)) != {"adflip"}:
        raise ValueError("Sampling manifests must contain only model_name=adflip")
    source_indices = staging.set_index("sample_id")["source_index"].astype(int)
    observed_indices = (
        sampling[["source_sample_id", "source_index"]]
        .drop_duplicates()
        .set_index("source_sample_id")["source_index"]
        .astype(int)
    )
    if len(observed_indices) != expected_count:
        raise ValueError("Sampling source_index is inconsistent within a source")
    observed_indices = observed_indices.reindex(source_indices.index)
    if not np.array_equal(source_indices.to_numpy(), observed_indices.to_numpy()):
        raise ValueError("Staging and sampling source_index values differ")
    if sampling["designed_sample_id"].duplicated().any():
        raise ValueError("Sampling manifests contain duplicate designed_sample_id values")


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    expected_count = int(
        cfg.dataset.expected_count
        if args.mode == "full"
        else cfg.dataset.expected_unique_ccd_count
    )
    expected_designs_per_sample = int(cfg.sequence_design.expected_designs_per_sample)
    design_indices = list(map(int, cfg.sequence_design.design_indices))
    output_slot_by_index = {
        design_index: output_slot
        for output_slot, design_index in enumerate(design_indices)
    }
    condition = str(OmegaConf.select(cfg, "dataset.condition", default=""))
    staging_manifest = cfg_path(cfg, f"staging.{args.mode}.manifest_csv")
    staging = pd.read_csv(staging_manifest, keep_default_na=False)
    sampling, sampling_paths = load_sampling_shards(
        cfg_path(cfg, f"adflip.{args.mode}.manifest_dir"),
        int(OmegaConf.select(cfg, f"adflip.{args.mode}.num_shards")),
    )
    validate_inputs(
        staging,
        sampling,
        expected_count=expected_count,
        expected_designs_per_sample=expected_designs_per_sample,
    )

    output_root = cfg_path(cfg, f"backmapping.{args.mode}.root")
    cif_dir = cfg_path(cfg, f"backmapping.{args.mode}.cif_dir")
    design_manifest = cfg_path(cfg, f"backmapping.{args.mode}.manifest_csv")
    validation_json = cfg_path(cfg, f"backmapping.{args.mode}.validation_json")
    output_root.mkdir(parents=True, exist_ok=True)
    cif_dir.mkdir(parents=True, exist_ok=True)

    staging_by_id = staging.set_index("sample_id", drop=False)
    rows: list[dict[str, Any]] = []
    for sample_record in sampling.sort_values("source_index").to_dict(orient="records"):
        sample_id = str(sample_record["source_sample_id"])
        source = staging_by_id.loc[sample_id]
        sampled_pdb = Path(str(sample_record["sampled_pdb_path"]))
        backmapped_cif = cif_dir / f"{sample_record['designed_sample_id']}.cif"
        row: dict[str, Any] = {
            "mode": args.mode,
            "condition": condition,
            "model_name": "adflip",
            "source_sample_id": sample_id,
            "partial_diffusion_source_index": int(
                source.get("partial_diffusion_source_index", source["source_index"])
            ),
            "design_index": int(sample_record["design_index"]),
            "output_slot": output_slot_by_index[
                int(sample_record["design_index"])
            ],
            "designed_sample_id": str(sample_record["designed_sample_id"]),
            "kind": str(source["kind"]),
            "protein_length": int(source["protein_length"]),
            "ccd_code": str(source["ccd_code"]),
            "pdb_ccd_code": str(source["pdb_ccd_code"]),
            "protein_chain_id": str(source["protein_chain_id"]),
            "ligand_chain_id": str(source["ligand_chain_id"]),
            "source_cif_path": str(source["source_cif_path"]),
            "staged_pdb_path": str(source["staged_pdb_path"]),
            "sampled_pdb_path": str(sampled_pdb),
            "backmapped_cif_path": str(backmapped_cif),
            "sampling_seed": int(sample_record["seed"]),
            "status": "failed",
            "error": "",
        }
        try:
            source_cif = Path(str(source["source_cif_path"]))
            staged_pdb = Path(str(source["staged_pdb_path"]))
            if sha256_file(source_cif) != str(source["source_cif_sha256"]):
                raise ValueError(f"Staged source CIF digest mismatch: {source_cif}")
            if sha256_file(staged_pdb) != str(source["staged_pdb_sha256"]):
                raise ValueError(f"Staged PDB digest mismatch: {staged_pdb}")
            if not sampled_pdb.is_file():
                raise FileNotFoundError(sampled_pdb)
            if sha256_file(sampled_pdb) != str(sample_record["sampled_pdb_sha256"]):
                raise ValueError(f"Sampled PDB digest mismatch: {sampled_pdb}")

            staged_structure = read_structure(staged_pdb)
            sampled_structure = read_structure(sampled_pdb)
            sampled_sequence = protein_sequence(
                sampled_structure, str(source["protein_chain_id"])
            )
            if sampled_sequence != str(sample_record["protein_sequence"]):
                raise ValueError("Sampling manifest/PDB protein sequence mismatch")
            if len(sampled_sequence) != int(source["protein_length"]):
                raise ValueError("Sampled protein length differs from staging")

            backbone_validation = validate_backbone_preserved(
                staged_structure,
                sampled_structure,
                protein_chain_id=str(source["protein_chain_id"]),
            )
            if not backbone_validation["ok"]:
                raise ValueError(
                    "Sampled PDB changed the protein backbone: "
                    + "; ".join(backbone_validation["errors"])
                )
            ligand_validation = validate_ligand_context_preserved(
                staged_structure,
                sampled_structure,
                ligand_chain_id=str(source["ligand_chain_id"]),
            )
            if not ligand_validation["ok"]:
                raise ValueError(
                    "Sampled PDB changed ligand context: "
                    + "; ".join(map(str, ligand_validation["errors"]))
                )

            restored = restore_ligand_ccd_and_roles(
                sampled_structure,
                protein_chain_id=str(source["protein_chain_id"]),
                ligand_chain_id=str(source["ligand_chain_id"]),
                pdb_ccd_code=str(source["pdb_ccd_code"]),
                original_ccd_code=str(source["ccd_code"]),
            )
            status = "reused"
            if args.overwrite or not backmapped_cif.exists():
                write_semantic_cif(backmapped_cif, restored)
                status = "backmapped"
            entity_validation = validate_semantic_cif_entities(
                backmapped_cif,
                protein_chain_id=str(source["protein_chain_id"]),
                ligand_chain_id=str(source["ligand_chain_id"]),
                original_ccd_code=str(source["ccd_code"]),
                pdb_ccd_code=str(source["pdb_ccd_code"]),
            )
            observed_cif = read_structure(backmapped_cif)
            roundtrip = validate_pdb_roundtrip(restored, observed_cif)
            if not roundtrip["ok"]:
                raise ValueError("; ".join(map(str, roundtrip["errors"])))
            observed_ccd = unique_ligand_ccd(
                observed_cif, str(source["ligand_chain_id"])
            )
            if observed_ccd != str(source["ccd_code"]):
                raise ValueError(
                    f"Backmapped ligand CCD {observed_ccd!r} != {source['ccd_code']!r}"
                )
            row.update(
                {
                    "status": status,
                    "protein_sequence": protein_sequence(
                        observed_cif, str(source["protein_chain_id"])
                    ),
                    "sampled_pdb_sha256": sha256_file(sampled_pdb),
                    "backmapped_cif_sha256": sha256_file(backmapped_cif),
                    "n_atoms": len(observed_cif),
                    "backbone_coordinate_max_abs_error": backbone_validation[
                        "coordinate_max_abs_error"
                    ],
                    "ligand_coordinate_max_abs_error": ligand_validation[
                        "coordinate_max_abs_error"
                    ],
                    "cif_coordinate_max_abs_error": roundtrip[
                        "coordinate_max_abs_error"
                    ],
                    "categorical_mismatch_count": int(
                        sum(roundtrip["categorical_mismatch_counts"].values())
                    ),
                    "protein_entity_id": entity_validation["protein_entity_id"],
                    "ligand_entity_id": entity_validation["ligand_entity_id"],
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    frame = (
        pd.DataFrame(rows)
        .sort_values(["source_sample_id", "design_index"])
        .reset_index(drop=True)
    )
    atomic_write_csv(design_manifest, frame)
    failed = frame[frame["status"] == "failed"]
    sampled_pdb_dir = cfg_path(cfg, f"adflip.{args.mode}.pdb_dir")
    expected_pdbs = {Path(path) for path in frame["sampled_pdb_path"]}
    observed_pdbs = set(sampled_pdb_dir.glob("*.pdb"))
    missing_pdbs = sorted(map(str, expected_pdbs - observed_pdbs))
    extra_pdbs = sorted(map(str, observed_pdbs - expected_pdbs))
    expected_cifs = {Path(path) for path in frame["backmapped_cif_path"]}
    observed_cifs = set(cif_dir.glob("*.cif"))
    missing_cifs = sorted(map(str, expected_cifs - observed_cifs))
    extra_cifs = sorted(map(str, observed_cifs - expected_cifs))
    duplicate_design_ids = int(frame["designed_sample_id"].duplicated().sum())
    report: dict[str, Any] = {
        "status": (
            "complete"
            if failed.empty
            and not missing_pdbs
            and not extra_pdbs
            and not missing_cifs
            and not extra_cifs
            and duplicate_design_ids == 0
            and len(frame) == expected_count * expected_designs_per_sample
            else "failed"
        ),
        "mode": args.mode,
        "staging_manifest": str(staging_manifest),
        "staging_manifest_sha256": sha256_file(staging_manifest),
        "sampling_manifests": [str(path) for path in sampling_paths],
        "design_manifest": str(design_manifest),
        "design_manifest_sha256": sha256_file(design_manifest),
        "expected_rows": expected_count * expected_designs_per_sample,
        "observed_rows": len(frame),
        "status_counts": dict(Counter(frame["status"])),
        "diagnostics": {
            "failed": len(failed),
            "missing_pdb": len(missing_pdbs),
            "extra_pdb": len(extra_pdbs),
            "missing_cif": len(missing_cifs),
            "extra_cif": len(extra_cifs),
            "unmapped": 0,
            "skipped": 0,
            "duplicate_design_id": duplicate_design_ids,
            "categorical_mismatches": int(
                frame.get("categorical_mismatch_count", pd.Series(dtype=int))
                .fillna(0)
                .sum()
            ),
        },
        "failed_examples": failed[["designed_sample_id", "error"]]
        .head(20)
        .to_dict(orient="records"),
        "contract": {
            "loader": "atomworks.read_any -> get_structure",
            "sampled_protein": "N/CA/C/O backbone with ADFLIP residue labels",
            "ligand": "all staged ligand atoms preserved",
            "ccd_backmapping": "PDB surrogate -> canonical source CCD",
            "protein_entity": "polypeptide(L)",
            "ligand_entity": "non-polymer",
        },
    }
    atomic_write_json(validation_json, report)

    if report["status"] == "complete" and args.mode == "per_ccd_smoke":
        smoke_ids = set(map(str, cfg.dataset.smoke_sample_ids))
        smoke = frame[frame["source_sample_id"].isin(smoke_ids)].copy()
        if set(smoke["source_sample_id"].astype(str)) != smoke_ids:
            raise ValueError("AF3 smoke IDs are missing from the per-CCD design manifest")
        atomic_write_csv(cfg_path(cfg, "backmapping.smoke_manifest_csv"), smoke)

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "complete":
        raise RuntimeError(f"Backmapping validation failed; see {validation_json}")


if __name__ == "__main__":
    main()
