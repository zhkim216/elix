"""Validate LASErMPNN outputs and restore PDB-safe ligand aliases in CIF."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from allatom_design.eval.benchmarks.denovoval.lasermpnn.atom_mapping import (
    read_atom_mapping,
    restore_transport_ligand_metadata,
)
from allatom_design.eval.benchmarks.denovoval.sequence_design_inputs import (
    atomic_write_csv,
    atomic_write_json,
    cfg_path,
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
    "denovoval/lasermpnn/default.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backmap denovoval LASErMPNN designs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifests(pattern: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No sampling manifests match {pattern}")
    frames: list[pd.DataFrame] = []
    required = {
        "source_sample_id", "design_index", "designed_sample_id", "sampled_pdb_path",
        "sampled_pdb_sha256", "input_pdb_path", "input_pdb_sha256", "protein_sequence",
        "model_weights_sha256", "sequence_temp", "first_shell_sequence_temp", "chi_temp",
        "disabled_residues", "transport_ligand_chain_id",
        "transport_ligand_res_name", "atom_mapping_path", "atom_mapping_sha256",
        "status", "error",
    }
    for path in paths:
        frame = pd.read_csv(path, keep_default_na=False)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frame["sampling_manifest_path"] = str(path)
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    duplicates = output["designed_sample_id"].duplicated(keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate designed sample IDs: "
            f"{output.loc[duplicates, 'designed_sample_id'].head(20).tolist()}"
        )
    return output, paths


def compare_ligands(reference: Any, observed: Any, ligand_chain: str) -> dict[str, Any]:
    left = reference[reference.chain_id.astype(str) == ligand_chain]
    right = observed[observed.chain_id.astype(str) == ligand_chain]
    if not len(left) or not len(right):
        raise ValueError("Reference or sampled PDB has no ligand chain")
    left_keys = list(zip(map(str, left.atom_name), map(str, left.element), strict=True))
    right_keys = list(zip(map(str, right.atom_name), map(str, right.element), strict=True))
    if left_keys != right_keys:
        raise ValueError("Sampled PDB ligand atom names/elements/order differ from input")
    coordinate_error = float(np.max(np.abs(left.coord - right.coord)))
    if coordinate_error > 0.00051:
        raise ValueError(f"Sampled PDB ligand coordinate error {coordinate_error}")
    if "CAP" in set(map(str, right.res_name)):
        raise ValueError("Synthetic CAP leaked into sampled PDB")
    return {
        "ligand_atom_count": len(right),
        "ligand_hydrogen_count": int(np.sum(right.element.astype(str) == "H")),
        "ligand_coordinate_max_abs_error": coordinate_error,
    }


def expected_keys(staging: pd.DataFrame, designs: int) -> set[tuple[str, int]]:
    return {
        (str(sample_id), design_index)
        for sample_id in staging["sample_id"]
        for design_index in range(1, designs + 1)
    }


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    staging_path = cfg_path(cfg, "staging.full_manifest_csv")
    staging = pd.read_csv(staging_path, keep_default_na=False)
    required_staging = {
        "sample_id", "kind", "protein_length", "ccd_code", "pdb_ccd_code",
        "protein_chain_id", "ligand_chain_id", "source_cif_path", "staged_pdb_path",
        "protonated_pdb_path", "protonated_pdb_sha256",
        "transport_ligand_chain_id", "transport_ligand_res_name",
        "atom_mapping_path", "atom_mapping_sha256", "status",
    }
    missing_staging = sorted(required_staging - set(staging.columns))
    if missing_staging:
        raise ValueError(f"{staging_path} is missing columns: {missing_staging}")
    if not staging["status"].eq("prepared").all():
        raise ValueError(f"{staging_path} contains unprepared rows")
    if args.smoke:
        smoke_ids = list(map(str, cfg.dataset.smoke_sample_ids))
        staging = staging.set_index("sample_id").loc[smoke_ids].reset_index()
    sampling, sampling_paths = load_manifests(
        str(cfg.sampling.smoke_manifest_glob if args.smoke else cfg.sampling.manifest_glob)
    )
    if args.smoke:
        sampling = sampling[sampling["source_sample_id"].isin(staging["sample_id"])].copy()
    designs = int(cfg.sequence_design.expected_designs_per_sample)
    design_indices = list(
        map(
            int,
            OmegaConf.select(
                cfg,
                "sequence_design.design_indices",
                default=list(range(1, designs + 1)),
            ),
        )
    )
    output_slot_by_index = {
        design_index: output_slot
        for output_slot, design_index in enumerate(design_indices)
    }
    condition = str(OmegaConf.select(cfg, "dataset.condition", default=""))
    wanted = expected_keys(staging, designs)
    observed = {
        (str(row.source_sample_id), int(row.design_index))
        for row in sampling.itertuples(index=False)
    }
    missing_keys = sorted(wanted - observed)
    extra_keys = sorted(observed - wanted)
    cif_root = cfg_path(cfg, "backmapping.smoke_cif_root" if args.smoke else "backmapping.full_cif_root")
    manifest_path = cfg_path(
        cfg, "backmapping.smoke_manifest_csv" if args.smoke else "backmapping.full_manifest_csv"
    )
    validation_path = cfg_path(
        cfg, "backmapping.smoke_validation_json" if args.smoke else "backmapping.full_validation_json"
    )
    protein_chain = str(cfg.structure.protein_chain_id)
    ligand_chain = str(cfg.structure.ligand_chain_id)
    source_by_id = staging.set_index("sample_id", drop=False)
    rows: list[dict[str, Any]] = []
    for sample in sampling.itertuples(index=False):
        sample_id = str(sample.source_sample_id)
        design_index = int(sample.design_index)
        output_cif = cif_root / f"{sample_id}_{design_index}.cif"
        row: dict[str, Any] = {
            "condition": condition,
            "model_name": "lasermpnn",
            "source_sample_id": sample_id,
            "design_index": design_index,
            "output_slot": output_slot_by_index[design_index],
            "designed_sample_id": str(sample.designed_sample_id),
            "kind": "", "protein_length": "", "ccd_code": "", "pdb_ccd_code": "",
            "protein_chain_id": protein_chain, "ligand_chain_id": ligand_chain,
            "source_cif_path": "", "staged_pdb_path": "", "protonated_pdb_path": "",
            "transport_ligand_chain_id": "", "transport_ligand_res_name": "",
            "atom_mapping_path": "", "atom_mapping_sha256": "",
            "sampling_manifest_path": str(sample.sampling_manifest_path),
            "sampled_pdb_path": str(sample.sampled_pdb_path),
            "backmapped_cif_path": str(output_cif), "protein_sequence": "",
            "status": "failed", "error": "",
        }
        try:
            if sample_id not in source_by_id.index:
                raise KeyError(f"Unknown source sample {sample_id}")
            source = source_by_id.loc[sample_id]
            row.update(
                {
                    "kind": str(source["kind"]),
                    "partial_diffusion_source_index": int(
                        source.get(
                            "partial_diffusion_source_index",
                            source["source_index"],
                        )
                    ),
                    "protein_length": int(source["protein_length"]),
                    "ccd_code": str(source["ccd_code"]),
                    "pdb_ccd_code": str(source["pdb_ccd_code"]),
                    "source_cif_path": str(source["source_cif_path"]),
                    "staged_pdb_path": str(source["staged_pdb_path"]),
                    "protonated_pdb_path": str(source["protonated_pdb_path"]),
                    "transport_ligand_chain_id": str(source["transport_ligand_chain_id"]),
                    "transport_ligand_res_name": str(source["transport_ligand_res_name"]),
                    "atom_mapping_path": str(source["atom_mapping_path"]),
                    "atom_mapping_sha256": str(source["atom_mapping_sha256"]),
                }
            )
            if str(sample.status) not in {"success", "reused"}:
                raise ValueError(f"Sampling failed: {sample.error}")
            sampled_path = Path(str(sample.sampled_pdb_path))
            protonated_path = Path(str(source["protonated_pdb_path"]))
            mapping_path = Path(str(source["atom_mapping_path"]))
            transport_chain = str(source["transport_ligand_chain_id"])
            for column in (
                "transport_ligand_chain_id",
                "transport_ligand_res_name",
                "atom_mapping_path",
                "atom_mapping_sha256",
            ):
                if str(getattr(sample, column)) != str(source[column]):
                    raise ValueError(
                        f"Sampling/staging provenance mismatch for {column}: "
                        f"{getattr(sample, column)} != {source[column]}"
                    )
            if sha256_file(sampled_path) != str(sample.sampled_pdb_sha256):
                raise ValueError("Sampled PDB digest differs from sampling manifest")
            if sha256_file(protonated_path) != str(sample.input_pdb_sha256):
                raise ValueError("Protonated input digest differs from sampling manifest")
            if sha256_file(mapping_path) != str(source["atom_mapping_sha256"]):
                raise ValueError("Atom-mapping sidecar digest differs from staging manifest")
            mapping_rows = read_atom_mapping(mapping_path)
            sampled_structure = read_structure(sampled_path)
            protonated_structure = read_structure(protonated_path)
            ligand_stats = compare_ligands(
                protonated_structure, sampled_structure, transport_chain
            )
            canonicalized = restore_transport_ligand_metadata(
                sampled_structure,
                mapping_rows,
                transport_chain_id=transport_chain,
                canonical_chain_id=ligand_chain,
                pdb_ccd_code=str(source["pdb_ccd_code"]),
            )
            restored = restore_ligand_ccd_and_roles(
                canonicalized,
                protein_chain_id=protein_chain,
                ligand_chain_id=ligand_chain,
                pdb_ccd_code=str(source["pdb_ccd_code"]),
                original_ccd_code=str(source["ccd_code"]),
            )
            sequence = protein_sequence(restored, protein_chain)
            if sequence != str(sample.protein_sequence):
                raise ValueError("Sample manifest/PDB protein sequence mismatch")
            status = "reused"
            if args.overwrite or not output_cif.is_file():
                write_semantic_cif(output_cif, restored)
                status = "backmapped"
            reparsed = read_structure(output_cif)
            roundtrip = validate_pdb_roundtrip(restored, reparsed)
            if not roundtrip["ok"]:
                raise ValueError("; ".join(map(str, roundtrip["errors"])))
            if unique_ligand_ccd(reparsed, ligand_chain) != str(source["ccd_code"]):
                raise ValueError("Backmapped CIF did not restore the original CCD code")
            if protein_sequence(reparsed, protein_chain) != sequence:
                raise ValueError("Protein sequence changed during CIF backmapping")
            validate_semantic_cif_entities(
                output_cif,
                protein_chain_id=protein_chain,
                ligand_chain_id=ligand_chain,
                original_ccd_code=str(source["ccd_code"]),
                pdb_ccd_code=str(source["pdb_ccd_code"]),
            )
            row.update(ligand_stats)
            row.update(
                {
                    "protein_sequence": sequence,
                    "sampled_pdb_sha256": sha256_file(sampled_path),
                    "atom_mapping_sha256": sha256_file(mapping_path),
                    "backmapped_cif_sha256": sha256_file(output_cif),
                    "n_atoms": len(reparsed),
                    "coordinate_max_abs_error": roundtrip["coordinate_max_abs_error"],
                    "categorical_mismatch_count": int(
                        sum(roundtrip["categorical_mismatch_counts"].values())
                    ),
                    "status": status,
                }
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    frame = pd.DataFrame(rows)
    atomic_write_csv(manifest_path, frame)
    failed = frame[frame["status"] == "failed"]
    report = {
        "status": "complete"
        if failed.empty and not missing_keys and not extra_keys and len(frame) == len(wanted)
        else "failed",
        "mode": "smoke" if args.smoke else "full",
        "staging_manifest": str(staging_path),
        "staging_manifest_sha256": sha256_file(staging_path),
        "sampling_manifests": [str(path) for path in sampling_paths],
        "design_manifest": str(manifest_path),
        "design_manifest_sha256": sha256_file(manifest_path),
        "source_count": len(staging),
        "expected_design_rows": len(wanted),
        "observed_design_rows": len(frame),
        "kind_counts": dict(Counter(frame["kind"])),
        "status_counts": dict(Counter(frame["status"])),
        "diagnostics": {
            "failed": len(failed), "missing_design_keys": len(missing_keys),
            "extra_design_keys": len(extra_keys),
            "unmapped": int(
                failed["error"].str.contains("mapping|Transport", case=False).sum()
            ),
            "skipped": 0,
            "categorical_mismatches": int(
                frame.get("categorical_mismatch_count", pd.Series(dtype=int)).fillna(0).sum()
            ),
        },
        "examples": {
            "failed": failed[["designed_sample_id", "error"]].head(20).to_dict(orient="records"),
            "missing_design_keys": missing_keys[:20], "extra_design_keys": extra_keys[:20],
        },
        "contract": {
            "raw_sample_gate": "immutable protonated ligand must be present before reconstruction",
            "ccd_backmapping": "stock-NISE transport identity -> atom-mapping sidecar -> canonical chain/names -> original CCD",
            "hydrogen_policy": "retain protonated ligand H in CIF; metric loader removes H",
            "synthetic_cap": "forbidden",
        },
    }
    atomic_write_json(validation_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "complete":
        raise RuntimeError(f"Backmapping incomplete; see {validation_path}")


if __name__ == "__main__":
    main()
