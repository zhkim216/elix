from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from biotite.structure import get_residue_starts
from omegaconf import DictConfig, OmegaConf

from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    allocate_ccd_aliases,
    prepare_pdb_structure,
    protein_sequence,
    read_structure,
    sha256_file,
    unique_ligand_ccd,
    validate_pdb_roundtrip,
    write_ccd_aliases,
    write_pdb,
)


SAMPLE_ID_PATTERN = re.compile(
    r"^(?P<ccd_code>.+)_len(?P<protein_length>[0-9]+)_(?P<replicate_index>[0-9]+)$"
)
REQUIRED_SOURCE_COLUMNS = {
    "pdb_key",
    "binder_pn_unit_iids",
    "context_pn_unit_iids",
    "frame_pn_unit_iids",
    "template_pn_unit_iids",
    "fixed_pos_seq",
    "fixed_pos_scn",
    "fixed_pos_override_seq",
    "pos_restrict_aatype",
}
ROLE_COLUMN_VALUES = {
    "binder_pn_unit_iids": ["A_1"],
    "context_pn_unit_iids": ["L_1"],
    "frame_pn_unit_iids": ["A_1"],
    "template_pn_unit_iids": [],
}
EMPTY_CONSTRAINT_COLUMNS = (
    "fixed_pos_seq",
    "fixed_pos_scn",
    "fixed_pos_override_seq",
    "pos_restrict_aatype",
)
SMALL_MOLECULE_REPLICATES_PER_LENGTH = 10
METAL_REPLICATES_PER_LENGTH = 20


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


def cif_tree_sha256(paths: list[Path]) -> str:
    """Hash sorted CIF names and content digests without depending on absolute paths."""
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _assert_sha256(path: Path, expected: str, *, label: str) -> str:
    observed = sha256_file(path)
    if expected and observed != expected:
        raise ValueError(f"{label} SHA256 mismatch: {observed} != {expected}: {path}")
    return observed


def _parse_sample_ids(sample_ids: pd.Series) -> pd.DataFrame:
    parsed_rows: list[dict[str, Any]] = []
    malformed: list[str] = []
    for source_index, sample_id in enumerate(map(str, sample_ids)):
        match = SAMPLE_ID_PATTERN.fullmatch(sample_id)
        if match is None:
            malformed.append(sample_id)
            continue
        parsed_rows.append(
            {
                "source_index": source_index,
                "sample_id": sample_id,
                "ccd_code": match.group("ccd_code"),
                "protein_length": int(match.group("protein_length")),
                "replicate_index": int(match.group("replicate_index")),
            }
        )
    if malformed:
        raise ValueError(f"Malformed denovoval sample IDs: {malformed[:20]}")
    return pd.DataFrame(parsed_rows)


def _assign_kinds(
    source: pd.DataFrame,
    *,
    metal_ccd_codes: list[str] | None = None,
) -> pd.Series:
    if metal_ccd_codes is not None:
        metals = {str(value).strip() for value in metal_ccd_codes}
        if not metals or "" in metals:
            raise ValueError("dataset.metal_ccd_codes must contain non-empty CCD codes")
        observed_ccds = set(map(str, source["ccd_code"]))
        missing = sorted(metals - observed_ccds)
        if missing:
            raise ValueError(
                "dataset.metal_ccd_codes contains CCDs absent from the source: "
                f"{missing}"
            )
        return source["ccd_code"].map(
            lambda ccd_code: "metal" if str(ccd_code) in metals else "small_molecule"
        )

    kinds_by_ccd: dict[str, str] = {}
    for ccd_code, ccd_rows in source.groupby("ccd_code", sort=False):
        lengths = sorted(set(map(int, ccd_rows["protein_length"])))
        if lengths != [150, 300]:
            raise ValueError(f"CCD {ccd_code!r} has protein lengths {lengths}, expected [150, 300]")
        group_sizes: set[int] = set()
        for length, group in ccd_rows.groupby("protein_length", sort=False):
            indices = sorted(map(int, group["replicate_index"]))
            expected_indices = list(range(len(indices)))
            if indices != expected_indices:
                raise ValueError(
                    f"CCD {ccd_code!r} length {length} indices {indices} != "
                    f"{expected_indices}"
                )
            group_sizes.add(len(indices))
        if group_sizes == {SMALL_MOLECULE_REPLICATES_PER_LENGTH}:
            kinds_by_ccd[str(ccd_code)] = "small_molecule"
        elif group_sizes == {METAL_REPLICATES_PER_LENGTH}:
            kinds_by_ccd[str(ccd_code)] = "metal"
        else:
            raise ValueError(
                f"CCD {ccd_code!r} has unsupported replicate group sizes "
                f"{sorted(group_sizes)}"
            )
    return source["ccd_code"].map(kinds_by_ccd)


def load_canonical_source(cfg: DictConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    sampling_inputs_csv = cfg_path(cfg, "dataset.sampling_inputs_csv")
    sample_id_list = cfg_path(cfg, "dataset.sample_id_list")
    source_cif_dir = cfg_path(cfg, "dataset.source_cif_dir")
    expected_count = int(cfg.dataset.expected_count)

    sampling_inputs_sha256 = _assert_sha256(
        sampling_inputs_csv,
        str(cfg.dataset.sampling_inputs_sha256),
        label="sampling_inputs.csv",
    )
    sample_id_list_sha256 = _assert_sha256(
        sample_id_list,
        str(cfg.dataset.sample_id_list_sha256),
        label="denovoval.txt",
    )

    runtime_source = pd.read_csv(sampling_inputs_csv, keep_default_na=False)
    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS - set(runtime_source.columns))
    if missing_columns:
        raise ValueError(f"{sampling_inputs_csv} is missing columns: {missing_columns}")
    if len(runtime_source) != expected_count:
        raise ValueError(
            f"Source CSV has {len(runtime_source)} rows, expected {expected_count}"
        )
    if runtime_source["pdb_key"].duplicated().any():
        duplicates = runtime_source.loc[
            runtime_source["pdb_key"].duplicated(keep=False), "pdb_key"
        ].tolist()
        raise ValueError(f"Duplicate pdb_key values: {duplicates[:20]}")

    for column, expected_value in ROLE_COLUMN_VALUES.items():
        invalid: list[str] = []
        for sample_id, raw_value in zip(
            runtime_source["pdb_key"], runtime_source[column], strict=True
        ):
            try:
                observed = json.loads(str(raw_value))
            except json.JSONDecodeError:
                invalid.append(str(sample_id))
                continue
            if observed != expected_value:
                invalid.append(str(sample_id))
        if invalid:
            raise ValueError(
                f"Canonical role column {column!r} differs for {invalid[:20]}"
            )
    for column in EMPTY_CONSTRAINT_COLUMNS:
        invalid = runtime_source.loc[runtime_source[column].astype(str) != "", "pdb_key"]
        if not invalid.empty:
            raise ValueError(
                f"Canonical constraint column {column!r} is non-empty for "
                f"{invalid.head(20).tolist()}"
            )

    listed_ids = [
        line.strip()
        for line in sample_id_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    csv_ids = list(map(str, runtime_source["pdb_key"]))
    if len(listed_ids) != expected_count or len(set(listed_ids)) != expected_count:
        raise ValueError(
            f"denovoval.txt count/uniqueness differs from {expected_count}: "
            f"rows={len(listed_ids)}, unique={len(set(listed_ids))}"
        )
    if listed_ids != csv_ids:
        first_mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(listed_ids, csv_ids))
                if left != right
            ),
            min(len(listed_ids), len(csv_ids)),
        )
        raise ValueError(
            "denovoval.txt order differs from sampling_inputs.csv at index "
            f"{first_mismatch}"
        )

    cif_paths = sorted(source_cif_dir.glob("*.cif"), key=lambda path: path.name)
    cif_stems = [path.stem for path in cif_paths]
    if len(cif_paths) != expected_count or len(set(cif_stems)) != expected_count:
        raise ValueError(
            f"CIF inventory count/uniqueness differs from {expected_count}: "
            f"files={len(cif_paths)}, unique={len(set(cif_stems))}"
        )
    missing_cifs = sorted(set(csv_ids) - set(cif_stems))
    extra_cifs = sorted(set(cif_stems) - set(csv_ids))
    if missing_cifs or extra_cifs:
        raise ValueError(
            f"Canonical CIF membership differs: missing={missing_cifs[:20]}, "
            f"extra={extra_cifs[:20]}"
        )
    observed_tree_sha256 = cif_tree_sha256(cif_paths)
    expected_tree_sha256 = str(cfg.dataset.cif_tree_sha256)
    if expected_tree_sha256 and observed_tree_sha256 != expected_tree_sha256:
        raise ValueError(
            f"CIF tree SHA256 mismatch: {observed_tree_sha256} != "
            f"{expected_tree_sha256}"
        )

    source = _parse_sample_ids(runtime_source["pdb_key"])
    partial_index_column = "partial_diffusion_source_index"
    if partial_index_column in runtime_source.columns:
        partial_indices = pd.to_numeric(
            runtime_source[partial_index_column], errors="raise"
        )
        if not np.all(partial_indices == partial_indices.astype(int)):
            raise ValueError(f"{partial_index_column} contains non-integral values")
        if partial_indices.duplicated().any():
            duplicates = partial_indices[partial_indices.duplicated(keep=False)].tolist()
            raise ValueError(
                f"Duplicate {partial_index_column} values: {duplicates[:20]}"
            )
        source[partial_index_column] = partial_indices.astype(int).to_numpy()
    else:
        if bool(
            OmegaConf.select(
                cfg,
                "dataset.partial_diffusion_source_index_required",
                default=False,
            )
        ):
            raise ValueError(
                f"{sampling_inputs_csv} is missing required column "
                f"{partial_index_column!r}"
            )
        source[partial_index_column] = source["source_index"]
    condition = OmegaConf.select(cfg, "dataset.condition", default="")
    source["condition"] = "" if condition is None else str(condition)
    configured_metals = OmegaConf.select(cfg, "dataset.metal_ccd_codes")
    source["kind"] = _assign_kinds(
        source,
        metal_ccd_codes=(
            None if configured_metals is None else list(configured_metals)
        ),
    )
    source["source_cif_path"] = source["sample_id"].map(
        lambda sample_id: str(source_cif_dir / f"{sample_id}.cif")
    )
    diagnostics = {
        "sampling_inputs_csv": str(sampling_inputs_csv),
        "sampling_inputs_sha256": sampling_inputs_sha256,
        "sample_id_list": str(sample_id_list),
        "sample_id_list_sha256": sample_id_list_sha256,
        "source_cif_dir": str(source_cif_dir),
        "cif_tree_sha256": observed_tree_sha256,
        "source_row_count": len(source),
        "condition": source["condition"].iloc[0] if len(source) else "",
        "partial_diffusion_source_index_min": int(
            source[partial_index_column].min()
        ),
        "partial_diffusion_source_index_max": int(
            source[partial_index_column].max()
        ),
        "partial_diffusion_source_index_unique": bool(
            source[partial_index_column].is_unique
        ),
        "kind_counts": source["kind"].value_counts().to_dict(),
        "length_counts": source["protein_length"].value_counts().to_dict(),
        "ccd_family_count": int(source["ccd_code"].nunique()),
    }
    return source, diagnostics


def select_source_rows(
    source: pd.DataFrame,
    *,
    cfg: DictConfig,
    smoke: bool,
    sample_ids: list[str] | None,
    limit: int | None,
) -> pd.DataFrame:
    requested_ids: list[str] | None = None
    if sample_ids:
        requested_ids = list(dict.fromkeys(map(str, sample_ids)))
    elif smoke:
        requested_ids = list(map(str, cfg.dataset.smoke_sample_ids))
    selected = source
    if requested_ids is not None:
        missing = sorted(set(requested_ids) - set(source["sample_id"]))
        if missing:
            raise ValueError(f"Requested sample IDs are absent: {missing}")
        selected = source.set_index("sample_id").loc[requested_ids].reset_index()
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected.head(limit)
    return selected.copy()


def _validate_source_structure(
    row: pd.Series,
    *,
    protein_chain_id: str,
    ligand_chain_id: str,
    expected_ligand_residue_id: int | None = None,
) -> tuple[Any, dict[str, int]]:
    source_path = Path(str(row["source_cif_path"]))
    structure = read_structure(source_path)
    observed_chains = sorted(set(map(str, structure.chain_id)))
    expected_chains = sorted([protein_chain_id, ligand_chain_id])
    if observed_chains != expected_chains:
        raise ValueError(f"Observed chains {observed_chains} != {expected_chains}")
    auth_chain_mismatches = int(
        np.sum(structure.auth_asym_id.astype(str) != structure.chain_id.astype(str))
    )
    auth_residue_mismatches = int(
        np.sum(structure.auth_seq_id.astype(str) != structure.res_id.astype(str))
    )
    if auth_chain_mismatches or auth_residue_mismatches:
        raise ValueError(
            "Source auth/label identifiers differ: "
            f"chain={auth_chain_mismatches}, residue={auth_residue_mismatches}"
        )
    if "ins_code" in structure.get_annotation_categories() and any(
        str(value).strip() for value in structure.ins_code
    ):
        raise ValueError("Source contains insertion codes")

    protein_mask = structure.chain_id.astype(str) == protein_chain_id
    ligand_mask = structure.chain_id.astype(str) == ligand_chain_id
    if np.any(structure.hetero[protein_mask]) or not np.all(structure.hetero[ligand_mask]):
        raise ValueError("Source ATOM/HETATM roles do not match protein/ligand chains")
    observed_length = len(protein_sequence(structure, protein_chain_id))
    if observed_length != int(row["protein_length"]):
        raise ValueError(
            f"Protein length {observed_length} != {int(row['protein_length'])}"
        )
    protein_residue_ids = list(
        map(int, structure.res_id[protein_mask][get_residue_starts(structure[protein_mask])])
    )
    if protein_residue_ids != list(range(1, observed_length + 1)):
        raise ValueError("Protein residue IDs are not sequential 1..protein_length")
    ligand_starts = get_residue_starts(structure[ligand_mask])
    if len(ligand_starts) != 1:
        raise ValueError(f"Expected one ligand residue, found {len(ligand_starts)}")
    ligand_residue_id = int(structure.res_id[ligand_mask][ligand_starts[0]])
    expected_residue_id = (
        observed_length + 1
        if expected_ligand_residue_id is None
        else expected_ligand_residue_id
    )
    if ligand_residue_id != expected_residue_id:
        raise ValueError(
            f"Ligand residue ID {ligand_residue_id} != {expected_residue_id}"
        )
    observed_ccd = unique_ligand_ccd(structure, ligand_chain_id)
    if observed_ccd != str(row["ccd_code"]):
        raise ValueError(f"Ligand CCD {observed_ccd!r} != {row['ccd_code']!r}")
    return structure, {
        "auth_chain_mismatch_count": auth_chain_mismatches,
        "auth_residue_mismatch_count": auth_residue_mismatches,
        "ligand_residue_id": ligand_residue_id,
        "n_protein_atoms": int(np.sum(protein_mask)),
        "n_ligand_atoms": int(np.sum(ligand_mask)),
    }


def stage_row(
    row: pd.Series,
    *,
    staged_pdb_dir: Path,
    alias_by_original: dict[str, str],
    protein_chain_id: str,
    ligand_chain_id: str,
    expected_ligand_residue_id: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    ccd_code = str(row["ccd_code"])
    pdb_ccd_code = alias_by_original.get(ccd_code, ccd_code)
    source_cif_path = Path(str(row["source_cif_path"]))
    staged_pdb_path = staged_pdb_dir / f"{sample_id}.pdb"
    result: dict[str, Any] = {
        "source_index": int(row["source_index"]),
        "partial_diffusion_source_index": int(
            row["partial_diffusion_source_index"]
        ),
        "condition": str(row["condition"]),
        "sample_id": sample_id,
        "kind": str(row["kind"]),
        "protein_length": int(row["protein_length"]),
        "ccd_code": ccd_code,
        "pdb_ccd_code": pdb_ccd_code,
        "protein_chain_id": protein_chain_id,
        "ligand_chain_id": ligand_chain_id,
        "source_cif_path": str(source_cif_path),
        "staged_pdb_path": str(staged_pdb_path),
        "status": "failed",
        "error": "",
    }
    try:
        if not source_cif_path.exists():
            raise FileNotFoundError(source_cif_path)
        source_structure, source_stats = _validate_source_structure(
            row,
            protein_chain_id=protein_chain_id,
            ligand_chain_id=ligand_chain_id,
            expected_ligand_residue_id=expected_ligand_residue_id,
        )
        staged_structure = prepare_pdb_structure(
            source_structure,
            ligand_chain_id=ligand_chain_id,
            original_ccd_code=ccd_code,
            pdb_ccd_code=pdb_ccd_code,
        )
        status = "reused"
        if overwrite or not staged_pdb_path.exists():
            write_pdb(staged_pdb_path, staged_structure)
            status = "staged"
        observed_pdb = read_structure(staged_pdb_path)
        validation = validate_pdb_roundtrip(staged_structure, observed_pdb)
        if not validation["ok"]:
            raise ValueError("; ".join(map(str, validation["errors"])))
        if unique_ligand_ccd(observed_pdb, ligand_chain_id) != pdb_ccd_code:
            raise ValueError("Staged PDB ligand CCD alias is incorrect")
        result.update(
            {
                "status": status,
                "source_cif_sha256": sha256_file(source_cif_path),
                "staged_pdb_sha256": sha256_file(staged_pdb_path),
                "n_atoms": len(source_structure),
                **source_stats,
                "coordinate_max_abs_error": validation["coordinate_max_abs_error"],
                "occupancy_max_abs_error": validation["numeric_max_abs_error"]["occupancy"],
                "b_factor_max_abs_error": validation["numeric_max_abs_error"]["b_factor"],
                "categorical_mismatch_count": int(
                    sum(validation["categorical_mismatch_counts"].values())
                ),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def prepare_inputs(
    cfg: DictConfig,
    *,
    smoke: bool,
    sample_ids: list[str] | None,
    limit: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    source, source_diagnostics = load_canonical_source(cfg)
    selected = select_source_rows(
        source,
        cfg=cfg,
        smoke=smoke,
        sample_ids=sample_ids,
        limit=limit,
    )
    staged_pdb_dir = cfg_path(cfg, "staging.pdb_dir")
    ccd_mapping_csv = cfg_path(cfg, "staging.ccd_mapping_csv")
    manifest_csv = cfg_path(
        cfg,
        "staging.smoke_manifest_csv" if smoke else "staging.full_manifest_csv",
    )
    validation_json = cfg_path(
        cfg,
        "staging.smoke_validation_json" if smoke else "staging.full_validation_json",
    )
    protein_chain_id = str(cfg.structure.protein_chain_id)
    ligand_chain_id = str(cfg.structure.ligand_chain_id)
    configured_ligand_residue_id = OmegaConf.select(
        cfg, "structure.ligand_residue_id", default=None
    )
    expected_ligand_residue_id = (
        None
        if configured_ligand_residue_id is None
        else int(configured_ligand_residue_id)
    )

    staged_pdb_dir.mkdir(parents=True, exist_ok=True)
    canonical_ids = set(map(str, source["sample_id"]))
    stale_pdbs = sorted(
        str(path) for path in staged_pdb_dir.glob("*.pdb") if path.stem not in canonical_ids
    )
    if stale_pdbs:
        raise ValueError(f"Stale non-canonical staged PDBs: {stale_pdbs[:20]}")

    aliases = allocate_ccd_aliases(source["ccd_code"])
    write_ccd_aliases(ccd_mapping_csv, aliases)
    alias_by_original = {
        alias.original_ccd_code: alias.pdb_ccd_code for alias in aliases
    }
    rows = [
        stage_row(
            row,
            staged_pdb_dir=staged_pdb_dir,
            alias_by_original=alias_by_original,
            protein_chain_id=protein_chain_id,
            ligand_chain_id=ligand_chain_id,
            expected_ligand_residue_id=expected_ligand_residue_id,
            overwrite=overwrite,
        )
        for _, row in selected.iterrows()
    ]
    frame = pd.DataFrame(rows)
    atomic_write_csv(manifest_csv, frame)
    failed = frame[frame["status"] == "failed"]
    report = {
        "status": "complete" if failed.empty else "failed",
        **source_diagnostics,
        "selected_row_count": len(selected),
        "selected_sample_ids": list(map(str, selected["sample_id"])),
        "staging_manifest": str(manifest_csv),
        "staging_manifest_sha256": sha256_file(manifest_csv),
        "ccd_mapping_csv": str(ccd_mapping_csv),
        "ccd_mapping_sha256": sha256_file(ccd_mapping_csv),
        "alias_count": len(aliases),
        "status_counts": dict(Counter(frame["status"])),
        "diagnostics": {
            "missing": int(frame["error"].str.contains("FileNotFoundError").sum()),
            "failed": len(failed),
            "skipped": 0,
            "unmapped": 0,
            "stale_pdb": len(stale_pdbs),
            "categorical_mismatches": int(
                frame.get("categorical_mismatch_count", pd.Series(dtype=int)).fillna(0).sum()
            ),
            "auth_chain_mismatches": int(
                frame.get("auth_chain_mismatch_count", pd.Series(dtype=int)).fillna(0).sum()
            ),
            "auth_residue_mismatches": int(
                frame.get("auth_residue_mismatch_count", pd.Series(dtype=int)).fillna(0).sum()
            ),
        },
        "failed_examples": failed[["sample_id", "error"]]
        .head(20)
        .to_dict(orient="records"),
        "contract": {
            "loader": "atomworks.read_any -> get_structure",
            "include_bonds": False,
            "model": 1,
            "altloc": "first",
            "protein_chain_id": protein_chain_id,
            "ligand_chain_id": ligand_chain_id,
            "ligand_residue_id": (
                "protein_length_plus_one"
                if expected_ligand_residue_id is None
                else expected_ligand_residue_id
            ),
            "kind_derivation": (
                "dataset.metal_ccd_codes"
                if OmegaConf.select(cfg, "dataset.metal_ccd_codes") is not None
                else "10 replicates/length=small_molecule; 20=metal"
            ),
        },
    }
    atomic_write_json(validation_json, report)
    if not failed.empty:
        raise RuntimeError(f"Staging failed for {len(failed)} rows; see {validation_json}")
    return report
