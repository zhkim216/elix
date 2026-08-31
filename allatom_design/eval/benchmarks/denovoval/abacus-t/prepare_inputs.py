"""Stage canonical denovoval complexes and CCD-faithful SDFs for ABACUS-T.

The PDB is a coordinate/identity transport artifact.  The SDF bond graph is
rebuilt from the pinned wwPDB chemical-component mirror, while its coordinates
are copied from the ligand in the staged PDB that ABACUS-T will consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import string
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from atomworks.constants import STANDARD_AA
from atomworks.io.tools.rdkit import ccd_code_to_rdkit
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem

from allatom_design.eval.benchmarks.denovoval.lasermpnn.component_library import (
    ComponentDefinition,
    load_component,
)
from allatom_design.eval.benchmarks.denovoval.sequence_design_inputs import (
    _validate_source_structure,
    atomic_write_csv,
    atomic_write_json,
    cfg_path,
    load_canonical_source,
)
from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    CcdAlias,
    allocate_ccd_aliases,
    prepare_pdb_structure,
    read_structure,
    sha256_file,
    validate_pdb_roundtrip,
    write_ccd_aliases,
    write_pdb,
)


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/abacus-t/default.yaml"
)
DEFAULT_STAGING_ROOT = Path(
    "/scratch/users/zhkim216/out_dir/benchmarks/denovoval/abacus-t/staging"
)
SMOKE_SAMPLE_IDS = ("A1L3W_len150_0", "NA_len150_0")
ABACUST_FILTERED_CCD_CODES = frozenset({"NA", "K"})
MONATOMIC_ZERO_BOND_CCDS = frozenset(
    {"CA", "CO", "CU", "FE", "FE2", "K", "MG", "MN", "NA", "NI", "ZN"}
)
BASE36 = string.digits + string.ascii_uppercase
MANIFEST_COLUMNS = (
    "sample_id",
    "source_cif_path",
    "source_cif_sha256",
    "staged_pdb_path",
    "staged_pdb_sha256",
    "ligand_sdf_path",
    "ligand_sdf_sha256",
    "original_ccd_code",
    "pdb_ccd_code",
    "protein_chain_id",
    "ligand_chain_id",
    "ligand_res_id",
    "abacust_ligand_basename",
    "protein_length",
    "replicate_index",
    "kind",
    "ccd_component_path",
    "ccd_component_sha256",
    "ligand_atom_count",
    "ligand_bond_count",
    "normalized_graph_sha256",
    "status",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare denovoval PDB and CCD-explicit SDF inputs for ABACUS-T."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--per-ccd", action="store_true")
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--source-cif-dir", type=Path)
    parser.add_argument("--sampling-inputs-csv", type=Path)
    parser.add_argument("--sample-id-list", type=Path)
    parser.add_argument("--ccd-mirror", type=Path)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--validation-path", type=Path)
    return parser.parse_args()


def _alias_candidates() -> Iterable[str]:
    for prefix in "XYZQ":
        for first in BASE36:
            for second in BASE36:
                yield f"{prefix}{first}{second}"


def allocate_abacust_aliases(ccd_codes: Iterable[str]) -> list[CcdAlias]:
    """Extend the shared PDB alias policy with ABACUS-T's name filters."""
    normalized = sorted({str(code).strip().upper() for code in ccd_codes})
    if any(not code for code in normalized):
        raise ValueError("CCD codes must be non-empty")
    aliases = list(allocate_ccd_aliases(normalized))
    already_aliased = {row.original_ccd_code for row in aliases}
    reserved = set(normalized) | set(map(str, STANDARD_AA)) | {
        row.pdb_ccd_code for row in aliases
    }
    candidates = (
        candidate for candidate in _alias_candidates() if candidate not in reserved
    )
    for ccd_code in sorted(ABACUST_FILTERED_CCD_CODES & set(normalized)):
        if ccd_code in already_aliased:
            continue
        alias = next(candidates)
        reserved.add(alias)
        aliases.append(CcdAlias(ccd_code, alias, "abacust_residue_filter"))
    return aliases


def select_rows(
    source: pd.DataFrame,
    *,
    smoke: bool,
    per_ccd: bool,
    sample_ids: list[str] | None,
    limit: int | None,
) -> tuple[pd.DataFrame, str]:
    if smoke:
        selected = source.set_index("sample_id").loc[list(SMOKE_SAMPLE_IDS)].reset_index()
        mode = "smoke"
    elif per_ccd:
        selected = (
            source.sort_values(["ccd_code", "protein_length", "replicate_index"])
            .groupby("ccd_code", sort=True, as_index=False)
            .first()
        )
        if not (selected["protein_length"].astype(int) == 150).all():
            raise ValueError("Per-CCD representatives are not all length 150")
        mode = "per_ccd"
    else:
        selected = source.copy()
        mode = "full"

    if sample_ids:
        requested = list(dict.fromkeys(map(str, sample_ids)))
        missing = sorted(set(requested) - set(map(str, selected["sample_id"])))
        if missing:
            raise ValueError(f"Requested sample IDs are absent from {mode}: {missing}")
        selected = selected.set_index("sample_id").loc[requested].reset_index()
        if mode == "full":
            mode = "selected"
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected.head(limit)
        if mode == "full":
            mode = "selected"
    return selected.copy(), mode


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _install_text(path: Path, text: str, *, overwrite: bool) -> str:
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Artifact path exists but is not a file: {path}")
        if path.read_text(encoding="utf-8") == text:
            return "reused"
        if not overwrite:
            raise ValueError(f"Stale artifact differs; rerun with --overwrite: {path}")
    _atomic_write_text(path, text)
    return "written"


def _write_expected_pdb(
    output_path: Path, structure: Any, *, overwrite: bool
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path = output_path.with_name(f".{output_path.name}.expected.{os.getpid()}")
    expected_path.unlink(missing_ok=True)
    try:
        write_pdb(expected_path, structure)
        expected_digest = sha256_file(expected_path)
        if output_path.exists():
            if not output_path.is_file():
                raise ValueError(f"Artifact path exists but is not a file: {output_path}")
            if sha256_file(output_path) == expected_digest:
                return "reused"
            if not overwrite:
                raise ValueError(
                    f"Stale artifact differs; rerun with --overwrite: {output_path}"
                )
        os.replace(expected_path, output_path)
        return "written"
    finally:
        expected_path.unlink(missing_ok=True)


def _heavy_ligand_records(
    staged_structure: Any, *, ligand_chain_id: str
) -> list[dict[str, Any]]:
    mask = (staged_structure.chain_id.astype(str) == ligand_chain_id) & (
        np.char.upper(staged_structure.element.astype(str)) != "H"
    )
    ligand = staged_structure[mask]
    if len(ligand) == 0:
        raise ValueError(f"No heavy ligand atoms in chain {ligand_chain_id!r}")
    names = list(map(str, ligand.atom_name))
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate staged ligand atom names: {names}")
    charges = (
        np.asarray(ligand.charge, dtype=int)
        if "charge" in ligand.get_annotation_categories()
        else np.zeros(len(ligand), dtype=int)
    )
    return [
        {
            "atom_id": name,
            "element": str(element).upper(),
            "pdb_charge": int(charge),
            "coord": np.asarray(coord, dtype=float),
        }
        for name, element, charge, coord in zip(
            names, ligand.element, charges, ligand.coord, strict=True
        )
    ]


def _direct_monatomic_molecule(
    component: ComponentDefinition,
    heavy_records: list[dict[str, Any]],
) -> tuple[Chem.Mol, list[str], str]:
    if len(component.heavy_atom_ids) != 1 or component.heavy_bonds:
        raise ValueError(
            f"Direct monatomic path requires one heavy atom and zero bonds: "
            f"{component.ccd_code}"
        )
    atom_id = component.heavy_atom_ids[0]
    record_by_name = {row["atom_id"]: row for row in heavy_records}
    if set(record_by_name) != {atom_id}:
        raise ValueError(
            f"Staged/CCD monatomic names differ for {component.ccd_code}: "
            f"{sorted(record_by_name)} != {[atom_id]}"
        )
    component_index = component.atom_ids.index(atom_id)
    element = component.elements[component_index]
    rdkit_symbol = element[0].upper() + element[1:].lower()
    atom = Chem.Atom(rdkit_symbol)
    atom.SetFormalCharge(int(component.charges[component_index]))
    editable = Chem.RWMol()
    editable.AddAtom(atom)
    mol = editable.GetMol()
    Chem.SanitizeMol(mol)
    conformer = Chem.Conformer(1)
    conformer.SetAtomPosition(0, record_by_name[atom_id]["coord"].tolist())
    mol.AddConformer(conformer, assignId=True)
    return mol, [atom_id], "direct_monatomic_ccd"


def build_explicit_sdf_molecule(
    component: ComponentDefinition,
    heavy_records: list[dict[str, Any]],
    *,
    ccd_mirror: Path,
) -> tuple[Chem.Mol, list[str], str]:
    """Build a CCD molecule and replace every coordinate with staged PDB XYZ."""
    record_by_name = {row["atom_id"]: row for row in heavy_records}
    if set(record_by_name) != set(component.heavy_atom_ids):
        raise ValueError(
            f"Staged/CCD heavy atom names differ for {component.ccd_code}: "
            f"{sorted(record_by_name)} != {sorted(component.heavy_atom_ids)}"
        )
    for atom_id in component.heavy_atom_ids:
        component_index = component.atom_ids.index(atom_id)
        expected_element = component.elements[component_index].upper()
        expected_charge = int(component.charges[component_index])
        record = record_by_name[atom_id]
        if (
            str(record["element"]).upper() != expected_element
            or int(record["pdb_charge"]) != expected_charge
        ):
            raise ValueError(
                f"Staged PDB chemistry differs for {component.ccd_code}/{atom_id}: "
                f"element={record['element']}/{expected_element}, "
                f"charge={record['pdb_charge']}/{expected_charge}"
            )
    if len(component.heavy_atom_ids) == 1 and not component.heavy_bonds:
        return _direct_monatomic_molecule(component, heavy_records)

    try:
        component_array, mol = ccd_code_to_rdkit(
            component.ccd_code,
            ccd_mirror_path=ccd_mirror,
            hydrogen_policy="remove",
            return_atom_array=True,
        )
    except Exception as exc:
        raise ValueError(
            f"AtomWorks CCD conversion failed for multi-atom {component.ccd_code}"
        ) from exc
    heavy_names = [
        str(atom_name)
        for atom_name, element in zip(
            component_array.atom_name, component_array.element, strict=True
        )
        if str(element).upper() != "H"
    ]
    if mol.GetNumAtoms() != len(heavy_names):
        raise ValueError(
            f"AtomWorks atom count differs for {component.ccd_code}: "
            f"{mol.GetNumAtoms()} != {len(heavy_names)}"
        )
    record_by_name = {row["atom_id"]: row for row in heavy_records}
    if set(heavy_names) != set(record_by_name):
        raise ValueError(
            f"Staged/CCD heavy atom names differ for {component.ccd_code}: "
            f"{sorted(record_by_name)} != {sorted(heavy_names)}"
        )
    if set(heavy_names) != set(component.heavy_atom_ids):
        raise ValueError(f"AtomWorks/component atom names differ for {component.ccd_code}")

    component_index = {name: index for index, name in enumerate(component.atom_ids)}
    for sdf_index, atom_name in enumerate(heavy_names):
        atom = mol.GetAtomWithIdx(sdf_index)
        index = component_index[atom_name]
        expected_element = component.elements[index].upper()
        expected_charge = int(component.charges[index])
        if (
            atom.GetSymbol().upper() != expected_element
            or atom.GetFormalCharge() != expected_charge
        ):
            raise ValueError(
                f"AtomWorks chemistry differs for {component.ccd_code}/{atom_name}"
            )

    expected_edges = {
        tuple(sorted((bond.atom_id_1, bond.atom_id_2)))
        for bond in component.heavy_bonds
    }
    observed_edges = {
        tuple(sorted((heavy_names[bond.GetBeginAtomIdx()], heavy_names[bond.GetEndAtomIdx()])))
        for bond in mol.GetBonds()
    }
    if observed_edges != expected_edges:
        raise ValueError(
            f"AtomWorks/CCD bond endpoints differ for {component.ccd_code}: "
            f"extra={sorted(observed_edges - expected_edges)[:10]}, "
            f"missing={sorted(expected_edges - observed_edges)[:10]}"
        )

    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for sdf_index, atom_name in enumerate(heavy_names):
        conformer.SetAtomPosition(
            sdf_index, record_by_name[atom_name]["coord"].tolist()
        )
    mol.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(mol)
    return mol, heavy_names, "atomworks_ccd_explicit"


def _normalized_graph(mol: Chem.Mol) -> tuple[dict[str, Any], str]:
    """Return the exact graph fingerprint contract used by the scratch runner."""
    atoms = [
        [atom.GetAtomicNum(), atom.GetFormalCharge()] for atom in mol.GetAtoms()
    ]
    bonds = sorted(
        [
            min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            str(bond.GetBondType()),
        ]
        for bond in mol.GetBonds()
    )
    graph = {"atoms": atoms, "bonds": bonds}
    encoded = json.dumps(graph, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return graph, hashlib.sha256(encoded).hexdigest()


def validate_sdf_readback(
    path: Path,
    *,
    expected_mol: Chem.Mol,
    atom_names: list[str],
    coordinate_tolerance: float = 0.00011,
) -> dict[str, Any]:
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
    observed = supplier[0] if len(supplier) else None
    if observed is None:
        raise ValueError(f"Sanitized RDKit SDF readback failed: {path}")
    if observed.GetNumAtoms() != expected_mol.GetNumAtoms():
        raise ValueError(f"SDF atom count changed on readback: {path}")
    for index, atom_name in enumerate(atom_names):
        expected_atom = expected_mol.GetAtomWithIdx(index)
        observed_atom = observed.GetAtomWithIdx(index)
        if (
            observed_atom.GetSymbol().upper() != expected_atom.GetSymbol().upper()
            or observed_atom.GetFormalCharge() != expected_atom.GetFormalCharge()
        ):
            raise ValueError(f"SDF atom chemistry changed for {atom_name}: {path}")
    expected_graph, expected_digest = _normalized_graph(expected_mol)
    observed_graph, observed_digest = _normalized_graph(observed)
    if observed_graph != expected_graph or observed_digest != expected_digest:
        raise ValueError(f"SDF graph changed on sanitized readback: {path}")
    coordinate_error = float(
        np.max(
            np.abs(
                np.asarray(expected_mol.GetConformer().GetPositions(), dtype=float)
                - np.asarray(observed.GetConformer().GetPositions(), dtype=float)
            )
        )
    )
    if coordinate_error > coordinate_tolerance:
        raise ValueError(
            f"SDF coordinate error {coordinate_error:.8g} exceeds "
            f"{coordinate_tolerance}: {path}"
        )
    return {
        "normalized_graph_sha256": observed_digest,
        "sdf_coordinate_max_abs_error": coordinate_error,
    }


def sdf_metadata_rows(
    *,
    sample_id: str,
    original_ccd_code: str,
    pdb_ccd_code: str,
    mol: Chem.Mol,
    atom_names: list[str],
    component: ComponentDefinition,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    component_orders = {
        tuple(sorted((bond.atom_id_1, bond.atom_id_2))): bond.value_order
        for bond in component.heavy_bonds
    }
    if len(component_orders) != len(component.heavy_bonds):
        raise ValueError(f"Duplicate component heavy bonds for {component.ccd_code}")
    coordinates = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    atom_rows = []
    for index, atom_name in enumerate(atom_names):
        atom = mol.GetAtomWithIdx(index)
        atom_rows.append(
            {
                "sample_id": sample_id,
                "original_ccd_code": original_ccd_code,
                "pdb_ccd_code": pdb_ccd_code,
                "atom_id": atom_name,
                "sdf_atom_index": index,
                "element": atom.GetSymbol().upper(),
                "formal_charge": atom.GetFormalCharge(),
                "x": coordinates[index, 0],
                "y": coordinates[index, 1],
                "z": coordinates[index, 2],
            }
        )
    bond_rows = []
    for bond in mol.GetBonds():
        left_index, right_index = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left_name, right_name = atom_names[left_index], atom_names[right_index]
        pair = tuple(sorted((left_name, right_name)))
        if pair not in component_orders:
            raise ValueError(f"SDF bond is absent from CCD: {component.ccd_code}/{pair}")
        bond_rows.append(
            {
                "sample_id": sample_id,
                "original_ccd_code": original_ccd_code,
                "pdb_ccd_code": pdb_ccd_code,
                "atom_id_1": pair[0],
                "atom_id_2": pair[1],
                "sdf_atom_index_1": atom_names.index(pair[0]),
                "sdf_atom_index_2": atom_names.index(pair[1]),
                "ccd_value_order": component_orders[pair],
                "sdf_bond_type": str(bond.GetBondType()),
            }
        )
    bond_rows.sort(key=lambda row: (row["atom_id_1"], row["atom_id_2"]))
    return atom_rows, bond_rows


def prepare_row(
    row: pd.Series,
    *,
    pdb_dir: Path,
    ligand_sdf_dir: Path,
    ccd_mirror: Path,
    alias_by_original: dict[str, str],
    protein_chain_id: str,
    ligand_chain_id: str,
    expected_ligand_residue_id: int | None,
    overwrite: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sample_id = str(row["sample_id"])
    original_ccd = str(row["ccd_code"])
    pdb_ccd = alias_by_original.get(original_ccd, original_ccd)
    source_cif_path = Path(str(row["source_cif_path"])).resolve()
    source_structure, source_stats = _validate_source_structure(
        row,
        protein_chain_id=protein_chain_id,
        ligand_chain_id=ligand_chain_id,
        expected_ligand_residue_id=expected_ligand_residue_id,
    )
    staged_expected = prepare_pdb_structure(
        source_structure,
        ligand_chain_id=ligand_chain_id,
        original_ccd_code=original_ccd,
        pdb_ccd_code=pdb_ccd,
    )
    staged_pdb_path = (pdb_dir / f"{sample_id}.pdb").resolve()
    _write_expected_pdb(staged_pdb_path, staged_expected, overwrite=overwrite)
    staged_observed = read_structure(staged_pdb_path)
    pdb_validation = validate_pdb_roundtrip(staged_expected, staged_observed)
    if not pdb_validation["ok"]:
        raise ValueError(
            f"Staged PDB roundtrip failed for {sample_id}: "
            + "; ".join(map(str, pdb_validation["errors"]))
        )

    component = load_component(ccd_mirror, original_ccd)
    heavy_records = _heavy_ligand_records(
        staged_observed, ligand_chain_id=ligand_chain_id
    )
    mol, atom_names, sdf_method = build_explicit_sdf_molecule(
        component, heavy_records, ccd_mirror=ccd_mirror
    )
    mol.SetProp("_Name", sample_id)
    sdf_text = Chem.MolToMolBlock(mol) + "\n$$$$\n"
    ligand_sdf_path = (ligand_sdf_dir / f"{sample_id}.sdf").resolve()
    _install_text(ligand_sdf_path, sdf_text, overwrite=overwrite)
    sdf_validation = validate_sdf_readback(
        ligand_sdf_path, expected_mol=mol, atom_names=atom_names
    )
    atom_rows, bond_rows = sdf_metadata_rows(
        sample_id=sample_id,
        original_ccd_code=original_ccd,
        pdb_ccd_code=pdb_ccd,
        mol=mol,
        atom_names=atom_names,
        component=component,
    )
    ligand_res_id = int(source_stats["ligand_residue_id"])
    manifest_row = {
        "source_index": int(row["source_index"]),
        "sample_id": sample_id,
        "source_cif_path": str(source_cif_path),
        "source_cif_sha256": sha256_file(source_cif_path),
        "staged_pdb_path": str(staged_pdb_path),
        "staged_pdb_sha256": sha256_file(staged_pdb_path),
        "ligand_sdf_path": str(ligand_sdf_path),
        "ligand_sdf_sha256": sha256_file(ligand_sdf_path),
        "original_ccd_code": original_ccd,
        "ccd_code": original_ccd,
        "pdb_ccd_code": pdb_ccd,
        "protein_chain_id": protein_chain_id,
        "ligand_chain_id": ligand_chain_id,
        "ligand_res_id": ligand_res_id,
        "abacust_ligand_basename": (
            f"lig_{ligand_chain_id}_{pdb_ccd}_{ligand_res_id}_1"
        ),
        "protein_length": int(row["protein_length"]),
        "replicate_index": int(row["replicate_index"]),
        "kind": str(row["kind"]),
        "ccd_component_path": str(component.path.resolve()),
        "ccd_component_sha256": sha256_file(component.path),
        "ligand_atom_count": mol.GetNumAtoms(),
        "ligand_bond_count": mol.GetNumBonds(),
        "normalized_graph_sha256": sdf_validation["normalized_graph_sha256"],
        "status": "complete",
        "sdf_generation_method": sdf_method,
        "pdb_coordinate_max_abs_error": pdb_validation[
            "coordinate_max_abs_error"
        ],
        "sdf_coordinate_max_abs_error": sdf_validation[
            "sdf_coordinate_max_abs_error"
        ],
        "auth_chain_mismatch_count": source_stats["auth_chain_mismatch_count"],
        "auth_residue_mismatch_count": source_stats["auth_residue_mismatch_count"],
    }
    return manifest_row, atom_rows, bond_rows


def _path_from_cfg_or_default(
    cfg: DictConfig, key: str, default: Path
) -> Path:
    value = OmegaConf.select(cfg, key)
    return Path(str(value)).expanduser() if value is not None else default


def artifact_paths(
    cfg: DictConfig,
    *,
    mode: str,
    staging_root_override: Path | None,
    manifest_override: Path | None,
    validation_override: Path | None,
) -> dict[str, Path]:
    root = (
        staging_root_override.expanduser()
        if staging_root_override is not None
        else _path_from_cfg_or_default(cfg, "staging.root", DEFAULT_STAGING_ROOT)
    )
    suffix = "" if mode == "full" else f"_{mode}"
    return {
        "root": root,
        "pdb_dir": _path_from_cfg_or_default(cfg, "staging.pdb_dir", root / "pdbs"),
        "ligand_sdf_dir": _path_from_cfg_or_default(
            cfg, "staging.ligand_sdf_dir", root / "ligand_sdfs"
        ),
        "alias_csv": _path_from_cfg_or_default(
            cfg, "staging.ccd_alias_csv", root / "ccd_aliases.csv"
        ),
        "manifest_csv": manifest_override
        or _path_from_cfg_or_default(
            cfg,
            f"staging.{mode}_manifest_csv",
            root / f"staging_manifest{suffix}.csv",
        ),
        "validation_json": validation_override
        or _path_from_cfg_or_default(
            cfg,
            f"staging.{mode}_validation_json",
            root / f"staging_validation{suffix}.json",
        ),
        "atom_csv": _path_from_cfg_or_default(
            cfg,
            f"staging.{mode}_ligand_sdf_atoms_csv",
            root / f"ligand_sdf_atoms{suffix}.csv",
        ),
        "bond_csv": _path_from_cfg_or_default(
            cfg,
            f"staging.{mode}_ligand_sdf_bonds_csv",
            root / f"ligand_sdf_bonds{suffix}.csv",
        ),
    }


def _assert_no_stale_artifacts(
    *, pdb_dir: Path, ligand_sdf_dir: Path, canonical_ids: set[str]
) -> None:
    stale = [
        str(path)
        for directory, pattern in ((pdb_dir, "*.pdb"), (ligand_sdf_dir, "*.sdf"))
        for path in directory.glob(pattern)
        if path.stem not in canonical_ids
    ]
    if stale:
        raise ValueError(f"Stale non-canonical ABACUS staging artifacts: {stale[:20]}")


def audit_manifest(
    manifest: pd.DataFrame, selected: pd.DataFrame
) -> list[str]:
    errors: list[str] = []
    missing_columns = sorted(set(MANIFEST_COLUMNS) - set(manifest.columns))
    if missing_columns:
        return [f"manifest missing columns: {missing_columns}"]
    expected_ids = list(map(str, selected["sample_id"]))
    observed_ids = list(map(str, manifest["sample_id"]))
    if observed_ids != expected_ids:
        errors.append("manifest sample order/membership differs from selection")
    if manifest["sample_id"].duplicated().any():
        errors.append("manifest has duplicate sample_id values")
    if set(map(str, manifest["status"])) != {"complete"}:
        errors.append("manifest contains non-complete status")
    for row in manifest.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        for path_key, digest_key in (
            ("source_cif_path", "source_cif_sha256"),
            ("staged_pdb_path", "staged_pdb_sha256"),
            ("ligand_sdf_path", "ligand_sdf_sha256"),
            ("ccd_component_path", "ccd_component_sha256"),
        ):
            path = Path(str(row[path_key]))
            if not path.is_file():
                errors.append(f"{sample_id}: missing {path_key}={path}")
            elif sha256_file(path) != str(row[digest_key]):
                errors.append(f"{sample_id}: digest mismatch for {path_key}")
        sdf_path = Path(str(row["ligand_sdf_path"]))
        if sdf_path.is_file():
            supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
            mol = supplier[0] if len(supplier) else None
            if mol is None:
                errors.append(f"{sample_id}: sanitized SDF readback failed")
            else:
                if mol.GetNumAtoms() != int(row["ligand_atom_count"]):
                    errors.append(f"{sample_id}: SDF atom count differs from manifest")
                if mol.GetNumBonds() != int(row["ligand_bond_count"]):
                    errors.append(f"{sample_id}: SDF bond count differs from manifest")
                if _normalized_graph(mol)[1] != str(row["normalized_graph_sha256"]):
                    errors.append(f"{sample_id}: SDF graph fingerprint differs")
    return errors


def prepare_inputs(
    cfg: DictConfig,
    *,
    smoke: bool,
    per_ccd: bool,
    sample_ids: list[str] | None,
    limit: int | None,
    overwrite: bool,
    validate_only: bool,
    staging_root_override: Path | None = None,
    manifest_override: Path | None = None,
    validation_override: Path | None = None,
) -> dict[str, Any]:
    source, source_diagnostics = load_canonical_source(cfg)
    selected, mode = select_rows(
        source,
        smoke=smoke,
        per_ccd=per_ccd,
        sample_ids=sample_ids,
        limit=limit,
    )
    paths = artifact_paths(
        cfg,
        mode=mode,
        staging_root_override=staging_root_override,
        manifest_override=manifest_override,
        validation_override=validation_override,
    )
    protein_chain_id = str(OmegaConf.select(cfg, "structure.protein_chain_id") or "A")
    ligand_chain_id = str(OmegaConf.select(cfg, "structure.ligand_chain_id") or "L")
    ccd_mirror = cfg_path(cfg, "chemistry.ccd_mirror")
    _assert_no_stale_artifacts(
        pdb_dir=paths["pdb_dir"],
        ligand_sdf_dir=paths["ligand_sdf_dir"],
        canonical_ids=set(map(str, source["sample_id"])),
    )

    if validate_only:
        if not paths["manifest_csv"].is_file():
            raise FileNotFoundError(paths["manifest_csv"])
        frame = pd.read_csv(paths["manifest_csv"], keep_default_na=False)
        errors = audit_manifest(frame, selected)
        if errors:
            raise ValueError(f"Existing staging audit failed: {errors[:20]}")
        report = {
            "status": "complete",
            "mode": mode,
            "selected_row_count": len(selected),
            "manifest_csv": str(paths["manifest_csv"]),
            "validation_only": True,
            "diagnostics": {"failed": 0, "missing": 0, "stale": 0},
        }
        atomic_write_json(paths["validation_json"], report)
        return report

    aliases = allocate_abacust_aliases(source["ccd_code"])
    alias_by_original = {
        row.original_ccd_code: row.pdb_ccd_code for row in aliases
    }
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["pdb_dir"].mkdir(parents=True, exist_ok=True)
    paths["ligand_sdf_dir"].mkdir(parents=True, exist_ok=True)
    alias_tmp = paths["alias_csv"].with_name(
        f".{paths['alias_csv'].name}.expected.{os.getpid()}"
    )
    write_ccd_aliases(alias_tmp, aliases)
    try:
        _install_text(
            paths["alias_csv"],
            alias_tmp.read_text(encoding="utf-8"),
            overwrite=overwrite,
        )
    finally:
        alias_tmp.unlink(missing_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    bond_rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        manifest_row, sample_atoms, sample_bonds = prepare_row(
            row,
            pdb_dir=paths["pdb_dir"],
            ligand_sdf_dir=paths["ligand_sdf_dir"],
            ccd_mirror=ccd_mirror,
            alias_by_original=alias_by_original,
            protein_chain_id=protein_chain_id,
            ligand_chain_id=ligand_chain_id,
            expected_ligand_residue_id=(
                int(cfg.structure.ligand_residue_id)
                if OmegaConf.select(cfg, "structure.ligand_residue_id") is not None
                else None
            ),
            overwrite=overwrite,
        )
        manifest_rows.append(manifest_row)
        atom_rows.extend(sample_atoms)
        bond_rows.extend(sample_bonds)

    frame = pd.DataFrame(manifest_rows)
    missing_manifest_columns = sorted(set(MANIFEST_COLUMNS) - set(frame.columns))
    if missing_manifest_columns:
        raise AssertionError(f"Internal manifest columns missing: {missing_manifest_columns}")
    atoms = pd.DataFrame(atom_rows)
    bonds = pd.DataFrame(
        bond_rows,
        columns=[
            "sample_id",
            "original_ccd_code",
            "pdb_ccd_code",
            "atom_id_1",
            "atom_id_2",
            "sdf_atom_index_1",
            "sdf_atom_index_2",
            "ccd_value_order",
            "sdf_bond_type",
        ],
    )
    if len(atoms) != int(frame["ligand_atom_count"].sum()):
        raise AssertionError("SDF atom table count does not reconcile with manifest")
    if len(bonds) != int(frame["ligand_bond_count"].sum()):
        raise AssertionError("SDF bond table count does not reconcile with manifest")

    atomic_write_csv(paths["manifest_csv"], frame)
    atomic_write_csv(paths["atom_csv"], atoms)
    atomic_write_csv(paths["bond_csv"], bonds)
    errors = audit_manifest(frame, selected)
    if errors:
        raise AssertionError(f"Published staging manifest failed audit: {errors[:20]}")
    expected_ccd_count = int(
        OmegaConf.select(cfg, "dataset.expected_unique_ccd_count")
        or source["ccd_code"].nunique()
    )
    observed_monatomic = sorted(
        set(
            frame.loc[
                frame["sdf_generation_method"] == "direct_monatomic_ccd",
                "original_ccd_code",
            ].astype(str)
        )
    )
    report = {
        "status": "complete",
        "mode": mode,
        **source_diagnostics,
        "selected_row_count": len(selected),
        "selected_unique_ccd_count": int(selected["ccd_code"].nunique()),
        "expected_unique_ccd_count": expected_ccd_count,
        "selected_sample_ids": list(map(str, selected["sample_id"])),
        "manifest_csv": str(paths["manifest_csv"]),
        "manifest_sha256": sha256_file(paths["manifest_csv"]),
        "ligand_sdf_atoms_csv": str(paths["atom_csv"]),
        "ligand_sdf_atoms_sha256": sha256_file(paths["atom_csv"]),
        "ligand_sdf_bonds_csv": str(paths["bond_csv"]),
        "ligand_sdf_bonds_sha256": sha256_file(paths["bond_csv"]),
        "ccd_alias_csv": str(paths["alias_csv"]),
        "ccd_alias_sha256": sha256_file(paths["alias_csv"]),
        "alias_count": len(aliases),
        "monatomic_ccd_codes": observed_monatomic,
        "diagnostics": {
            "failed": 0,
            "missing": 0,
            "skipped": 0,
            "unmapped": 0,
            "stale": 0,
            "categorical_mismatches": 0,
            "auth_chain_mismatches": int(frame["auth_chain_mismatch_count"].sum()),
            "auth_residue_mismatches": int(frame["auth_residue_mismatch_count"].sum()),
        },
        "contract": {
            "structure_loader": "atomworks.read_any -> get_structure",
            "protein_chain_id": protein_chain_id,
            "ligand_chain_id": ligand_chain_id,
            "sdf_chemistry": "pinned CCD mirror via AtomWorks; direct monatomic exception",
            "sdf_coordinates": "staged PDB ligand",
            "smoke_sample_ids": list(SMOKE_SAMPLE_IDS),
        },
    }
    atomic_write_json(paths["validation_json"], report)
    return report


def _load_cfg(args: argparse.Namespace) -> DictConfig:
    if not args.config.is_file():
        raise FileNotFoundError(
            f"Config not found: {args.config}. Pass --config plus explicit path overrides."
        )
    cfg = OmegaConf.load(args.config)
    overrides = {
        "dataset.source_cif_dir": args.source_cif_dir,
        "dataset.sampling_inputs_csv": args.sampling_inputs_csv,
        "dataset.sample_id_list": args.sample_id_list,
        "chemistry.ccd_mirror": args.ccd_mirror,
        "staging.root": args.staging_root,
    }
    for key, value in overrides.items():
        if value is not None:
            OmegaConf.update(cfg, key, str(value.expanduser()), merge=False)
    if args.staging_root is not None:
        # Config interpolation values may otherwise continue pointing at the old root.
        for key in tuple(cfg.staging.keys()):
            if key != "root":
                del cfg.staging[key]
    return cfg


def main() -> None:
    args = parse_args()
    if args.validate_only and args.overwrite:
        raise ValueError("--validate-only and --overwrite are mutually exclusive")
    cfg = _load_cfg(args)
    report = prepare_inputs(
        cfg,
        smoke=args.smoke,
        per_ccd=args.per_ccd,
        sample_ids=args.sample_id,
        limit=args.limit,
        overwrite=args.overwrite,
        validate_only=args.validate_only,
        staging_root_override=args.staging_root,
        manifest_override=args.manifest_path,
        validation_override=args.validation_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
