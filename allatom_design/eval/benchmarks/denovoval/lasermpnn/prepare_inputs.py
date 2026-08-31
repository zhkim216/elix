"""Build CCD-faithful PDB and protonated-PDB inputs for denovoval LASErMPNN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import prody as pr
from atomworks.constants import STANDARD_AA
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem

from allatom_design.eval.benchmarks.denovoval.lasermpnn.atom_mapping import (
    write_atom_mapping,
)

from allatom_design.eval.benchmarks.denovoval.lasermpnn.component_library import (
    allocate_aliases,
    atomworks_fallback_molecule,
    component_edges,
    conect_lines,
    load_component,
    mirror_ccd_codes,
    molecule_heavy_name_order,
    parse_conect_edges,
)
from allatom_design.eval.benchmarks.denovoval.sequence_design_inputs import (
    _validate_source_structure,
    atomic_write_csv,
    atomic_write_json,
    cfg_path,
    load_canonical_source,
    select_source_rows,
)
from allatom_design.eval.benchmarks.sequence_design.structure_bridge import (
    prepare_pdb_structure,
    read_structure,
    sha256_file,
    validate_pdb_roundtrip,
    write_pdb,
)


DEFAULT_CONFIG = Path(
    "/home/users/zhkim216/code/elix/allatom_design/configs/benchmarks/"
    "denovoval/lasermpnn/default.yaml"
)
STOCK_NISE_TRANSPORT_CHAIN = "B"
STOCK_NISE_TRANSPORT_RES_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare denovoval PDB/protonated-PDB inputs for LASErMPNN."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--per-ccd", action="store_true")
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Revalidate the existing manifest, artifact paths, and SHA-256 digests "
            "and rewrite only the validation JSON."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def append_conect(path: Path, edges: list[tuple[int, int]]) -> None:
    body = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("CONECT") and line.strip() != "END"
    ]
    atomic_write_text(path, "\n".join(body + conect_lines(edges) + ["END", ""]))


def write_aliases(path: Path, aliases: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["original_ccd_code", "pdb_ccd_code", "reason"]
        )
        writer.writeheader()
        for alias in aliases:
            writer.writerow(asdict(alias))
    os.replace(temporary, path)


def serials_are_contiguous(path: Path) -> bool:
    structure = pr.parsePDB(str(path))
    if not isinstance(structure, pr.AtomGroup):
        return False
    return list(map(int, structure.getSerials())) == list(range(1, len(structure) + 1))


def target_component_set_sha256(mirror: Path, ccd_codes: Any) -> str:
    digest = hashlib.sha256()
    for ccd_code in sorted(set(map(str, ccd_codes))):
        path = mirror / ccd_code[0] / ccd_code / f"{ccd_code}.cif"
        digest.update(ccd_code.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def write_staged_pdb(
    source_structure: Any,
    *,
    output_path: Path,
    original_ccd: str,
    pdb_ccd: str,
    ligand_chain_id: str,
    component: Any,
) -> dict[str, Any]:
    staged = prepare_pdb_structure(
        source_structure,
        ligand_chain_id=ligand_chain_id,
        original_ccd_code=original_ccd,
        pdb_ccd_code=pdb_ccd,
    )
    if "atom_id" in staged.get_annotation_categories():
        staged.del_annotation("atom_id")
    staged.set_annotation("atom_id", np.arange(1, len(staged) + 1, dtype=int))
    write_pdb(output_path, staged)
    ligand_mask = staged.chain_id.astype(str) == ligand_chain_id
    edges = component_edges(
        component,
        atom_names=staged.atom_name[ligand_mask],
        serials=staged.atom_id[ligand_mask],
        include_hydrogen=False,
    )
    append_conect(output_path, edges)
    observed = read_structure(output_path)
    validation = validate_pdb_roundtrip(staged, observed)
    if not validation["ok"]:
        raise ValueError("; ".join(map(str, validation["errors"])))
    if not serials_are_contiguous(output_path):
        raise ValueError("Staged PDB atom serials are not exactly 1..N")
    if parse_conect_edges(output_path) != {tuple(sorted(edge)) for edge in edges}:
        raise ValueError("Staged PDB CONECT graph differs from the CCD heavy graph")
    return {
        "staging_bond_count": len(edges),
        "staging_coordinate_max_abs_error": validation["coordinate_max_abs_error"],
    }


@lru_cache(maxsize=None)
def _load_stock_nise_main(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "denovoval_stock_nise_protonation", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load stock NISE script {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "main", None)):
        raise AttributeError(f"Stock NISE script has no callable main(): {script_path}")
    return module.main


def _run_stock_nise(
    staged_path: Path, output_path: Path, smiles: str, script_path: Path
) -> None:
    if not script_path.is_file():
        raise FileNotFoundError(script_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        _load_stock_nise_main(script_path)(
            str(staged_path), str(temporary), smiles
        )
        if not temporary.is_file():
            raise ValueError("Stock NISE did not create its output PDB")
        os.replace(temporary, output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_stock_transport_from_molecule(
    *, staged_path: Path, output_path: Path, mol: Chem.Mol, pdb_ccd: str
) -> None:
    """Write an AtomWorks-generated molecule with stock-NISE PDB metadata."""
    staged = pr.parsePDB(str(staged_path))
    if not isinstance(staged, pr.AtomGroup):
        raise ValueError(f"ProDy failed to parse {staged_path}")
    protein = staged.select("protein")
    if protein is None:
        raise ValueError(f"No protein atoms in {staged_path}")
    molecule_stream = io.StringIO(Chem.MolToPDBBlock(mol, flavor=(4 | 8)))
    rdkit_conect = [
        line for line in molecule_stream.getvalue().splitlines()
        if line.startswith("CONECT")
    ]
    ligand = pr.parsePDBStream(molecule_stream)
    if not isinstance(ligand, pr.AtomGroup):
        raise ValueError("ProDy failed to parse AtomWorks RDKit PDB block")
    ligand.setResnames(pdb_ccd)
    ligand.setResnums(STOCK_NISE_TRANSPORT_RES_ID)
    ligand.setChids(STOCK_NISE_TRANSPORT_CHAIN)
    ligand.setOccupancies(1.0)
    ligand.setBetas(0.0)
    renamed: list[str] = []
    element_counts: defaultdict[str, int] = defaultdict(int)
    for atom_name in map(str, ligand.getNames()):
        element = "".join(character for character in atom_name if not character.isdigit())
        element_counts[element] += 1
        renamed.append(f"{element}{element_counts[element]}")
    ligand.setNames(renamed)

    protein = protein.copy()
    protein_length = len(protein)
    final_stream = io.StringIO()
    pr.writePDBStream(final_stream, protein + ligand)
    if "TER " in final_stream.getvalue():
        protein_length += 1
    offset_conect: list[str] = []
    for line in rdkit_conect:
        record, *serials = line.split()
        offset_conect.append(
            record + "".join(
                str(int(serial) + protein_length).rjust(5, " ")
                for serial in serials
            )
        )
    body = final_stream.getvalue().rsplit("END", 1)[0]
    atomic_write_text(output_path, body + "\n".join(offset_conect))


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def _build_atom_mapping(
    *,
    sample_id: str,
    original_ccd: str,
    source_ligand: pr.Atomic,
    output_path: Path,
    transport_chain_id: str,
    heavy_source_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed = pr.parsePDB(str(output_path))
    if not isinstance(observed, pr.AtomGroup):
        raise ValueError(f"ProDy failed to parse {output_path}")
    transport = observed.select(f"chain {transport_chain_id}")
    if transport is None:
        raise ValueError(f"No transport ligand chain {transport_chain_id} in {output_path}")
    source_names = list(map(str, source_ligand.getNames()))
    if set(heavy_source_names) != set(source_names):
        raise ValueError("Transport heavy-source mapping does not cover the source ligand")
    source_index_by_name = {name: index for index, name in enumerate(source_names)}
    source_coord_by_name = dict(zip(source_names, source_ligand.getCoords(), strict=True))
    source_element_by_name = dict(
        zip(source_names, map(str, source_ligand.getElements()), strict=True)
    )
    source_serial_by_name = dict(
        zip(source_names, map(int, source_ligand.getSerials()), strict=True)
    )
    source_chain = str(source_ligand.getChids()[0])
    source_res_id = int(source_ligand.getResnums()[0])

    transport_elements = [str(value).upper() for value in transport.getElements()]
    transport_heavy_indices = [
        index for index, element in enumerate(transport_elements) if element != "H"
    ]
    if len(transport_heavy_indices) != len(heavy_source_names):
        raise ValueError(
            "Transport/source heavy atom count mismatch: "
            f"{len(transport_heavy_indices)} != {len(heavy_source_names)}"
        )
    heavy_name_by_transport_index = dict(
        zip(transport_heavy_indices, heavy_source_names, strict=True)
    )
    coordinate_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for index, (serial, atom_name, element, chain_id, res_id, res_name) in enumerate(
        zip(
            map(int, transport.getSerials()),
            map(str, transport.getNames()),
            transport_elements,
            map(str, transport.getChids()),
            map(int, transport.getResnums()),
            map(str, transport.getResnames()),
            strict=True,
        )
    ):
        source_name = heavy_name_by_transport_index.get(index, "")
        if source_name:
            source_element = source_element_by_name[source_name].upper()
            if element != source_element:
                raise ValueError(
                    f"Transport/source element mismatch for {source_name}: "
                    f"{element} != {source_element}"
                )
            coordinate_errors.append(
                float(
                    np.max(
                        np.abs(
                            transport.getCoords()[index] - source_coord_by_name[source_name]
                        )
                    )
                )
            )
            atom_role = "source_heavy"
            source_atom_index: int | str = source_index_by_name[source_name] + 1
            source_atom_serial: int | str = source_serial_by_name[source_name]
        else:
            source_element = ""
            atom_role = "generated_hydrogen"
            source_atom_index = ""
            source_atom_serial = ""
        rows.append(
            {
                "sample_id": sample_id,
                "transport_atom_index": index + 1,
                "transport_atom_serial": serial,
                "transport_atom_name": atom_name,
                "transport_element": element,
                "transport_chain_id": chain_id,
                "transport_res_id": res_id,
                "transport_res_name": res_name,
                "atom_role": atom_role,
                "source_atom_index": source_atom_index,
                "source_atom_serial": source_atom_serial,
                "source_atom_name": source_name,
                "source_element": source_element,
                "source_chain_id": source_chain,
                "source_res_id": source_res_id,
                "source_res_name": original_ccd,
                "parent_transport_atom_index": "",
                "parent_source_atom_name": "",
            }
        )

    edges = parse_conect_edges(output_path)
    serial_to_index = {
        int(row["transport_atom_serial"]): int(row["transport_atom_index"])
        for row in rows
    }
    element_by_serial = {
        int(row["transport_atom_serial"]): str(row["transport_element"])
        for row in rows
    }
    source_name_by_serial = {
        int(row["transport_atom_serial"]): str(row["source_atom_name"])
        for row in rows
        if row["atom_role"] == "source_heavy"
    }
    neighbours: defaultdict[int, list[int]] = defaultdict(list)
    for left, right in edges:
        if left in serial_to_index and right in serial_to_index:
            neighbours[left].append(right)
            neighbours[right].append(left)
    coordinates_by_serial = dict(
        zip(map(int, transport.getSerials()), transport.getCoords(), strict=True)
    )
    h_parent_distances: list[float] = []
    for row in rows:
        if row["atom_role"] != "generated_hydrogen":
            continue
        serial = int(row["transport_atom_serial"])
        heavy_parents = [
            neighbour
            for neighbour in neighbours[serial]
            if element_by_serial.get(neighbour, "").upper() != "H"
        ]
        if len(heavy_parents) != 1:
            raise ValueError(
                f"Transport H {row['transport_atom_name']} has {len(heavy_parents)} heavy parents"
            )
        parent = heavy_parents[0]
        if parent not in source_name_by_serial:
            raise ValueError(
                f"Transport H {row['transport_atom_name']} is bonded to an unmapped atom"
            )
        row["parent_transport_atom_index"] = serial_to_index[parent]
        row["parent_source_atom_name"] = source_name_by_serial[parent]
        h_parent_distances.append(
            float(np.linalg.norm(coordinates_by_serial[serial] - coordinates_by_serial[parent]))
        )
    heavy_coordinate_error = max(coordinate_errors, default=0.0)
    if heavy_coordinate_error > 0.00051:
        raise ValueError(
            f"Protonation changed ligand heavy coordinates: {heavy_coordinate_error}"
        )
    if h_parent_distances and (
        min(h_parent_distances) < 0.7 or max(h_parent_distances) > 1.6
    ):
        raise ValueError(
            "Generated H-parent bond length is outside [0.7, 1.6] A: "
            f"min={min(h_parent_distances):.3f}, max={max(h_parent_distances):.3f}"
        )
    unique_res_ids = sorted(set(map(int, transport.getResnums())))
    unique_res_names = sorted(set(map(str, transport.getResnames())))
    if len(unique_res_ids) != 1 or len(unique_res_names) != 1:
        raise ValueError("Transport ligand is not one residue")
    return rows, {
        "transport_ligand_chain_id": transport_chain_id,
        "transport_ligand_res_id": unique_res_ids[0],
        "transport_ligand_res_name": unique_res_names[0],
        "n_ligand_heavy_atoms": len(transport_heavy_indices),
        "n_ligand_hydrogens": len(rows) - len(transport_heavy_indices),
        "ligand_heavy_coordinate_max_abs_error": heavy_coordinate_error,
        "hydrogen_parent_distance_min": min(h_parent_distances) if h_parent_distances else "",
        "hydrogen_parent_distance_max": max(h_parent_distances) if h_parent_distances else "",
    }


def write_protonated_pdb(
    *,
    staged_path: Path,
    output_path: Path,
    protein_chain_id: str,
    ligand_chain_id: str,
    pdb_ccd: str,
    component: Any,
    ccd_mirror: Path,
    kind: str,
    sample_id: str = "test",
    original_ccd: str | None = None,
    stock_nise_script: Path | None = None,
    atomworks_fallback_ccds: frozenset[str] = frozenset(),
    atomworks_fallback_sample_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    staged = pr.parsePDB(str(staged_path))
    if not isinstance(staged, pr.AtomGroup):
        raise ValueError(f"ProDy failed to parse {staged_path}")
    protein = staged.select(f"chain {protein_chain_id}")
    source_ligand = staged.select(f"chain {ligand_chain_id} and not element H")
    if protein is None or source_ligand is None:
        raise ValueError("Staged PDB lost the protein or ligand chain")
    protein = protein.copy()
    source_ligand = source_ligand.copy()
    source_names = list(map(str, source_ligand.getNames()))
    original_ccd = original_ccd or component.ccd_code
    if kind == "metal":
        _atomic_copy(staged_path, output_path)
        method = "not_applicable_metal"
        fallback_reason = ""
        transport_chain_id = ligand_chain_id
        heavy_source_names = source_names
    elif (
        component.ccd_code in atomworks_fallback_ccds
        or sample_id in atomworks_fallback_sample_ids
    ):
        mol = atomworks_fallback_molecule(component, source_ligand, ccd_mirror)
        heavy_source_names = molecule_heavy_name_order(mol, source_names)
        _write_stock_transport_from_molecule(
            staged_path=staged_path,
            output_path=output_path,
            mol=mol,
            pdb_ccd=pdb_ccd,
        )
        if component.ccd_code in atomworks_fallback_ccds:
            method = "atomworks_ccd_explicit_exception"
            fallback_reason = "stock_nise_openeye_ccd_known_failure"
        else:
            method = "atomworks_sample_explicit_exception"
            fallback_reason = "stock_nise_proximity_graph_sample_failure"
        transport_chain_id = STOCK_NISE_TRANSPORT_CHAIN
    else:
        if component.smiles is None:
            raise ValueError(
                f"No parseable OpenEye canonical SMILES for {component.ccd_code}"
            )
        if stock_nise_script is None:
            raise ValueError("stock_nise_script is required for small molecules")
        _run_stock_nise(
            staged_path, output_path, component.smiles, stock_nise_script
        )
        method = "stock_nise_openeye_canonical"
        fallback_reason = ""
        transport_chain_id = STOCK_NISE_TRANSPORT_CHAIN
        heavy_source_names = source_names

    mapping_rows, mapping_stats = _build_atom_mapping(
        sample_id=sample_id,
        original_ccd=original_ccd,
        source_ligand=source_ligand,
        output_path=output_path,
        transport_chain_id=transport_chain_id,
        heavy_source_names=heavy_source_names,
    )
    observed = pr.parsePDB(str(output_path))
    observed_protein = observed.select(f"chain {protein_chain_id}")
    if observed_protein is None:
        raise ValueError("Protonated PDB lost protein atoms")
    protein_error = float(
        np.max(np.abs(protein.getCoords() - observed_protein.getCoords()))
    )
    if protein_error > 0.00051:
        raise ValueError(f"Protonation changed protein coordinates: {protein_error}")
    if not serials_are_contiguous(output_path):
        raise ValueError("Protonated PDB atom serials are not exactly 1..N")
    observed_serials = set(map(int, observed.getSerials()))
    observed_edges = parse_conect_edges(output_path)
    missing_endpoints = sorted(
        {serial for edge in observed_edges for serial in edge} - observed_serials
    )
    if missing_endpoints:
        raise ValueError(
            f"Protonated PDB CONECT references absent atom serials: {missing_endpoints}"
        )
    return {
        "protonation_method": method,
        "fallback_reason": fallback_reason,
        **mapping_stats,
        "protonated_bond_count": len(observed_edges),
        "protein_coordinate_max_abs_error": protein_error,
        "atom_mapping_row_count": len(mapping_rows),
    }, mapping_rows


def prepare_row(
    row: pd.Series,
    *,
    cfg: DictConfig,
    alias_by_original: dict[str, str],
    overwrite: bool,
) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    ccd_code = str(row["ccd_code"])
    pdb_ccd = alias_by_original.get(ccd_code, ccd_code)
    staged_path = cfg_path(cfg, "staging.pdb_dir") / f"{sample_id}.pdb"
    protonated_path = cfg_path(cfg, "staging.protonated_pdb_dir") / f"{sample_id}.pdb"
    atom_mapping_path = cfg_path(cfg, "staging.atom_mapping_dir") / f"{sample_id}.csv"
    result: dict[str, Any] = {
        "source_index": int(row["source_index"]),
        "sample_id": sample_id,
        "kind": str(row["kind"]),
        "protein_length": int(row["protein_length"]),
        "ccd_code": ccd_code,
        "pdb_ccd_code": pdb_ccd,
        "protein_chain_id": str(cfg.structure.protein_chain_id),
        "ligand_chain_id": str(cfg.structure.ligand_chain_id),
        "source_cif_path": str(row["source_cif_path"]),
        "staged_pdb_path": str(staged_path),
        "protonated_pdb_path": str(protonated_path),
        "atom_mapping_path": str(atom_mapping_path),
        "status": "failed",
        "error": "",
    }
    try:
        atom_mapping_path.unlink(missing_ok=True)
        component = load_component(cfg_path(cfg, "chemistry.ccd_mirror"), ccd_code)
        source_structure, source_stats = _validate_source_structure(
            row,
            protein_chain_id=str(cfg.structure.protein_chain_id),
            ligand_chain_id=str(cfg.structure.ligand_chain_id),
            expected_ligand_residue_id=(
                int(cfg.structure.ligand_residue_id)
                if OmegaConf.select(cfg, "structure.ligand_residue_id") is not None
                else None
            ),
        )
        source_ligand = source_structure[
            source_structure.chain_id.astype(str) == str(cfg.structure.ligand_chain_id)
        ]
        if tuple(map(str, source_ligand.atom_name)) != component.heavy_atom_ids:
            raise ValueError("Canonical CIF ligand atom order differs from CCD heavy atom order")
        if tuple(map(str.upper, source_ligand.element)) != component.heavy_elements:
            raise ValueError("Canonical CIF ligand elements differ from CCD heavy elements")
        if overwrite or not staged_path.is_file():
            staging_stats = write_staged_pdb(
                source_structure,
                output_path=staged_path,
                original_ccd=ccd_code,
                pdb_ccd=pdb_ccd,
                ligand_chain_id=str(cfg.structure.ligand_chain_id),
                component=component,
            )
            stage_status = "staged"
        else:
            staging_stats = write_staged_pdb(
                source_structure,
                output_path=staged_path,
                original_ccd=ccd_code,
                pdb_ccd=pdb_ccd,
                ligand_chain_id=str(cfg.structure.ligand_chain_id),
                component=component,
            )
            stage_status = "revalidated"
        protonation_stats, mapping_rows = write_protonated_pdb(
            staged_path=staged_path,
            output_path=protonated_path,
            protein_chain_id=str(cfg.structure.protein_chain_id),
            ligand_chain_id=str(cfg.structure.ligand_chain_id),
            pdb_ccd=pdb_ccd,
            component=component,
            ccd_mirror=cfg_path(cfg, "chemistry.ccd_mirror"),
            kind=str(row["kind"]),
            sample_id=sample_id,
            original_ccd=ccd_code,
            stock_nise_script=cfg_path(cfg, "chemistry.stock_nise_script"),
            atomworks_fallback_ccds=frozenset(
                map(str, cfg.chemistry.atomworks_fallback_ccds)
            ),
            atomworks_fallback_sample_ids=frozenset(
                map(str, cfg.chemistry.atomworks_fallback_sample_ids)
            ),
        )
        write_atom_mapping(atom_mapping_path, mapping_rows)
        result.update(source_stats)
        result.update(staging_stats)
        result.update(protonation_stats)
        result.update(
            {
                "component_cif_path": str(component.path),
                "component_cif_sha256": sha256_file(component.path),
                "smiles": component.smiles or "",
                "smiles_type": component.smiles_type or "",
                "smiles_program": component.smiles_program or "",
                "smiles_program_version": component.smiles_program_version or "",
                "source_cif_sha256": sha256_file(row["source_cif_path"]),
                "staged_pdb_sha256": sha256_file(staged_path),
                "protonated_pdb_sha256": sha256_file(protonated_path),
                "atom_mapping_sha256": sha256_file(atom_mapping_path),
                "staging_status": stage_status,
                "status": "prepared",
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def audit_existing_manifest(
    frame: pd.DataFrame,
    selected: pd.DataFrame,
) -> list[str]:
    """Return integrity errors without regenerating chemistry artifacts."""

    required_columns = {
        "sample_id",
        "status",
        "source_cif_path",
        "source_cif_sha256",
        "staged_pdb_path",
        "staged_pdb_sha256",
        "protonated_pdb_path",
        "protonated_pdb_sha256",
        "atom_mapping_path",
        "atom_mapping_sha256",
        "component_cif_path",
        "component_cif_sha256",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        return [f"manifest missing columns: {missing_columns}"]

    errors: list[str] = []
    expected_sample_ids = list(map(str, selected["sample_id"]))
    observed_sample_ids = list(map(str, frame["sample_id"]))
    if observed_sample_ids != expected_sample_ids:
        missing = sorted(set(expected_sample_ids) - set(observed_sample_ids))
        extra = sorted(set(observed_sample_ids) - set(expected_sample_ids))
        errors.append(
            "manifest sample order/membership differs from selected source: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )
    duplicate_ids = sorted(
        frame.loc[frame["sample_id"].duplicated(keep=False), "sample_id"]
        .astype(str)
        .unique()
    )
    if duplicate_ids:
        errors.append(f"manifest has duplicate sample IDs: {duplicate_ids[:20]}")

    artifact_fields = (
        ("source_cif_path", "source_cif_sha256"),
        ("staged_pdb_path", "staged_pdb_sha256"),
        ("protonated_pdb_path", "protonated_pdb_sha256"),
        ("atom_mapping_path", "atom_mapping_sha256"),
        ("component_cif_path", "component_cif_sha256"),
    )
    for row in frame.to_dict(orient="records"):
        sample_id = str(row["sample_id"])
        for path_field, digest_field in artifact_fields:
            path = Path(str(row[path_field]))
            if not path.is_file():
                errors.append(f"{sample_id}: missing {path_field}={path}")
                continue
            expected_digest = str(row[digest_field])
            observed_digest = sha256_file(path)
            if observed_digest != expected_digest:
                errors.append(
                    f"{sample_id}: stale {path_field} digest "
                    f"{observed_digest} != {expected_digest}"
                )
    return errors


def main() -> None:
    args = parse_args()
    if args.report_only and args.overwrite:
        raise ValueError("--report-only and --overwrite are mutually exclusive")
    cfg = OmegaConf.load(args.config)
    source, source_diagnostics = load_canonical_source(cfg)
    if args.per_ccd:
        selected = (
            source.sort_values(["ccd_code", "protein_length", "replicate_index"])
            .groupby("ccd_code", sort=True, as_index=False)
            .first()
        )
        if args.sample_id:
            requested = list(dict.fromkeys(map(str, args.sample_id)))
            missing = sorted(set(requested) - set(selected["sample_id"]))
            if missing:
                raise ValueError(
                    "--per-ccd sample IDs must be the deterministic representative: "
                    f"{missing}"
                )
            selected = selected.set_index("sample_id").loc[requested].reset_index()
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be positive")
            selected = selected.head(args.limit)
    else:
        selected = select_source_rows(
            source,
            cfg=cfg,
            smoke=False,
            sample_ids=args.sample_id,
            limit=args.limit,
        )
    aliases = allocate_aliases(
        source["ccd_code"],
        reserved_codes=mirror_ccd_codes(cfg_path(cfg, "chemistry.ccd_mirror")),
        amino_acids=STANDARD_AA,
    )
    alias_by_original = {
        alias.original_ccd_code: alias.pdb_ccd_code for alias in aliases
    }
    observed_component_set_sha256 = target_component_set_sha256(
        cfg_path(cfg, "chemistry.ccd_mirror"), source["ccd_code"]
    )
    expected_component_set_sha256 = str(cfg.chemistry.target_component_set_sha256)
    if observed_component_set_sha256 != expected_component_set_sha256:
        raise ValueError(
            "Target components.cif set digest differs: "
            f"{observed_component_set_sha256} != {expected_component_set_sha256}"
        )
    alias_path = cfg_path(cfg, "staging.ccd_alias_csv")
    manifest = cfg_path(
        cfg, "staging.per_ccd_manifest_csv" if args.per_ccd else "staging.full_manifest_csv"
    )
    validation = cfg_path(
        cfg,
        "staging.per_ccd_validation_json" if args.per_ccd else "staging.full_validation_json",
    )
    print(f"canonical rows: {len(source)}; selected rows: {len(selected)}")
    print(f"staging PDBs: {cfg_path(cfg, 'staging.pdb_dir')}")
    print(f"protonated PDBs: {cfg_path(cfg, 'staging.protonated_pdb_dir')}")
    if args.dry_run:
        return
    artifact_integrity_errors: list[str] = []
    if args.report_only:
        if not manifest.is_file():
            raise FileNotFoundError(f"Existing staging manifest not found: {manifest}")
        if not alias_path.is_file():
            raise FileNotFoundError(f"Existing CCD alias CSV not found: {alias_path}")
        frame = pd.read_csv(manifest, keep_default_na=False)
        artifact_integrity_errors = audit_existing_manifest(frame, selected)
    else:
        cfg_path(cfg, "staging.pdb_dir").mkdir(parents=True, exist_ok=True)
        cfg_path(cfg, "staging.protonated_pdb_dir").mkdir(parents=True, exist_ok=True)
        cfg_path(cfg, "staging.atom_mapping_dir").mkdir(parents=True, exist_ok=True)
        write_aliases(alias_path, aliases)
        rows = [
            prepare_row(
                row,
                cfg=cfg,
                alias_by_original=alias_by_original,
                overwrite=args.overwrite,
            )
            for _, row in selected.iterrows()
        ]
        frame = pd.DataFrame(rows)
        atomic_write_csv(manifest, frame)
    failed = frame[frame["status"] != "prepared"]
    fallback_ccds = sorted(map(str, cfg.chemistry.atomworks_fallback_ccds))
    fallback_sample_ids = sorted(
        map(str, cfg.chemistry.atomworks_fallback_sample_ids)
    )
    unexpected_fallbacks = frame[
        (
            (frame.get("protonation_method", "") == "atomworks_ccd_explicit_exception")
            & (~frame["ccd_code"].isin(fallback_ccds))
        )
        | (
            (frame.get("protonation_method", "") == "atomworks_sample_explicit_exception")
            & (~frame["sample_id"].isin(fallback_sample_ids))
        )
    ]
    missing_mapping = frame[
        (frame["status"] == "prepared")
        & (~frame["atom_mapping_path"].map(lambda value: Path(str(value)).is_file()))
    ]
    artifact_missing_count = sum(
        ": missing " in error for error in artifact_integrity_errors
    )
    artifact_stale_count = sum(
        ": stale " in error for error in artifact_integrity_errors
    )
    artifact_unresolved_count = (
        len(artifact_integrity_errors)
        - artifact_missing_count
        - artifact_stale_count
    )
    components_cif = cfg_path(cfg, "chemistry.components_cif")
    report = {
        "status": "complete"
        if (
            failed.empty
            and missing_mapping.empty
            and unexpected_fallbacks.empty
            and not artifact_integrity_errors
        )
        else "failed",
        **source_diagnostics,
        "validation_mode": (
            "existing_manifest_digest_audit"
            if args.report_only
            else "prepare_and_validate"
        ),
        "selected_row_count": len(frame),
        "selected_unique_ccd_count": int(frame["ccd_code"].nunique()),
        "selected_length_counts": {
            str(length): int(count)
            for length, count in sorted(Counter(frame["protein_length"]).items())
        },
        "manifest_csv": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "ccd_alias_csv": str(alias_path),
        "ccd_alias_sha256": sha256_file(alias_path),
        "target_component_set_sha256": observed_component_set_sha256,
        "components_cif": str(components_cif),
        "components_cif_sha256": sha256_file(components_cif),
        "kind_counts": dict(Counter(frame["kind"])),
        "status_counts": dict(Counter(frame["status"])),
        "protonation_method_counts": dict(
            Counter(
                str(value)
                for value in frame.get("protonation_method", pd.Series(dtype=str))
                if str(value) not in {"", "nan"}
            )
        ),
        "diagnostics": {
            "failed": len(failed),
            "missing": int(failed["error"].str.contains("FileNotFoundError").sum())
            + artifact_missing_count,
            "unmapped": len(missing_mapping),
            "skipped": 0,
            "stale": artifact_stale_count,
            "unresolved": artifact_unresolved_count,
            "unexpected_fallback": len(unexpected_fallbacks),
        },
        "failed_examples": failed[["sample_id", "ccd_code", "error"]]
        .head(20)
        .to_dict(orient="records"),
        "unexpected_fallback_examples": unexpected_fallbacks[
            ["sample_id", "ccd_code", "protonation_method", "fallback_reason"]
        ].head(20).to_dict(orient="records"),
        "artifact_integrity_error_examples": artifact_integrity_errors[:20],
        "contract": {
            "membership_and_coordinates": "canonical denovoval sampling_inputs.csv + cifs",
            "chemistry": "OpenEye SMILES_CANONICAL from the pinned wwPDB components.cif mirror",
            "serialized_conect": "staging and protonated PDB CONECT records use explicit CCD/generated adjacency only",
            "stock_nise_internal_graph": "stock NISE may reconstruct a proximity graph after its ProDy rewrite; only pinned diagnosed failures use AtomWorks",
            "small_molecule_protonation": "stock NISE writer with OpenEye canonical SMILES",
            "metal_protonation": "not applicable; zero added hydrogen",
            "atomworks_explicit_exception_ccds": fallback_ccds,
            "atomworks_explicit_exception_sample_ids": fallback_sample_ids,
            "fallback_policy": "only configured CCD-wide and conformer-specific known stock-NISE failures use AtomWorks; every other stock failure is fatal",
            "metadata_policy": "preserve stock-NISE transport metadata and reconstruct canonical heavy identity through the per-sample atom-mapping sidecar",
            "formal_charge_policy": "do not restore source or CCD formal charges after protonation",
            "hydrogen_parent_policy": "record generated H parents for reconstruction; do not compare them with explicit CCD hydrogen topology",
            "serials": "contiguous ATOM/HETATM serials 1..N",
        },
    }
    atomic_write_json(validation, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "complete":
        raise RuntimeError(
            "Preparation failed validation: "
            f"failed={len(failed)}, unmapped={len(missing_mapping)}, "
            f"unexpected_fallback={len(unexpected_fallbacks)}, "
            f"stale={len(artifact_integrity_errors)}; see {validation}"
        )


if __name__ == "__main__":
    main()
