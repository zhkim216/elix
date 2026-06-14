"""Metal RASA computation for denovoval single-metal CIFs."""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure import filter_amino_acids, sasa
from biotite.structure.info import vdw_radius_single
from biotite.structure.io.pdbx import CIFFile, get_structure

from allatom_design.data.preprocessing.atomworks.denovoval.records import SourceRecord


METAL_ELEMENT_OVERRIDES = {"FE2": "FE"}
DEFAULT_PROBE_RADIUS_ANGSTROM = 1.4


def expected_element(ccd_code: str) -> str:
    return METAL_ELEMENT_OVERRIDES.get(ccd_code.upper(), ccd_code.upper())


def vdw_radius(element: str) -> float:
    radius = vdw_radius_single(element.strip().upper())
    if radius is None:
        return 1.7
    return float(radius)


def read_cif_atom_array(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        cif = CIFFile.read(handle)
    return get_structure(
        cif,
        model=1,
        extra_fields=["atom_id", "b_factor", "occupancy", "charge"],
    )


def finite_coord_mask(atom_array: Any) -> np.ndarray:
    return np.isfinite(atom_array.coord).all(axis=1)


def protein_atom_mask(atom_array: Any) -> np.ndarray:
    amino_acid_mask = filter_amino_acids(atom_array)
    if bool(np.any(amino_acid_mask)):
        return amino_acid_mask
    return ~atom_array.hetero


def heavy_atom_mask(atom_array: Any) -> np.ndarray:
    element = np.char.upper(atom_array.element.astype(str))
    return ~np.isin(element, ["H", "D", "T", ""])


def make_vdw_radii(
    atom_array: Any,
    target_indices: set[int],
    metal_element: str,
) -> tuple[np.ndarray, dict[str, int]]:
    radii = np.empty(len(atom_array), dtype=float)
    fallback_counts: Counter[str] = Counter()
    metal_radius = vdw_radius(metal_element)
    for index, element in enumerate(atom_array.element.astype(str)):
        if index in target_indices:
            radii[index] = metal_radius
            continue
        normalized = element.strip().upper()
        radius = vdw_radius_single(normalized)
        if radius is None:
            fallback_counts[normalized or "<blank>"] += 1
            radius = 1.7
        radii[index] = float(radius)
    return radii, dict(fallback_counts)


def compute_target_sasa(
    *,
    atom_array: Any,
    target_index: int,
    occluder_mask: np.ndarray,
    all_radii: np.ndarray,
    free_sasa: float,
    probe_radius: float,
    point_number: int,
) -> tuple[str, str, int, float | None, float | None]:
    occluder_mask = occluder_mask.copy()
    occluder_mask[target_index] = False
    occluder_indices = np.where(occluder_mask)[0]
    keep_indices = np.concatenate([np.array([target_index], dtype=int), occluder_indices])
    local = atom_array[keep_indices]
    local_radii = all_radii[keep_indices]
    atom_filter = np.zeros(len(local), dtype=bool)
    atom_filter[0] = True
    try:
        local_sasa = sasa(
            local,
            probe_radius=probe_radius,
            atom_filter=atom_filter,
            ignore_ions=False,
            point_number=point_number,
            vdw_radii=local_radii,
        )
    except Exception as exc:  # noqa: BLE001 - keep batch RASA auditing going.
        return "sasa_failed", f"{type(exc).__name__}: {exc}", int(len(occluder_indices)), None, None

    complex_sasa = float(local_sasa[0])
    if math.isnan(complex_sasa):
        return "sasa_nan", "Biotite returned NaN for target metal", int(len(occluder_indices)), None, None
    if complex_sasa < 0:
        return "sasa_negative", f"Biotite returned negative SASA {complex_sasa}", int(len(occluder_indices)), None, None
    return "ok", "", int(len(occluder_indices)), complex_sasa, complex_sasa / free_sasa


def process_metal_record(
    record: SourceRecord,
    *,
    point_number: int,
    probe_radius: float = DEFAULT_PROBE_RADIUS_ANGSTROM,
) -> dict[str, Any]:
    metal_element = expected_element(record.ccd_code)
    free_sasa = 4.0 * math.pi * (vdw_radius(metal_element) + probe_radius) ** 2
    base_record: dict[str, Any] = {
        "sample_id": record.sample_id,
        "kind": record.kind,
        "ccd_code": record.ccd_code,
        "metal_element": metal_element,
        "length": record.length,
        "source_path": str(record.source_path),
        "source_condition": record.source_condition,
        "metal_rasa_status": "",
        "metal_rasa_detail": "",
        "n_atoms": 0,
        "n_protein_heavy_atoms": 0,
        "n_target_metal_sites": 0,
        "site_index_in_file": "",
        "atom_index": "",
        "chain_id": "",
        "res_id": "",
        "res_name": "",
        "atom_name": "",
        "element": "",
        "free_sasa": free_sasa,
        "protein_all_heavy_n_occluder_atoms": "",
        "protein_all_heavy_complex_sasa": "",
        "rasa_value": "",
        "radius_fallback_counts_json": "{}",
    }

    try:
        atom_array = read_cif_atom_array(record.source_path)
    except Exception as exc:  # noqa: BLE001 - report parse failures per sample.
        base_record["metal_rasa_status"] = "parse_failed"
        base_record["metal_rasa_detail"] = f"{type(exc).__name__}: {exc}"
        return base_record

    finite_mask = finite_coord_mask(atom_array)
    protein_heavy_finite = protein_atom_mask(atom_array) & heavy_atom_mask(atom_array) & finite_mask
    if "hetero" in atom_array.get_annotation_categories():
        hetero_mask = atom_array.hetero
    else:
        hetero_mask = ~protein_atom_mask(atom_array)
    element = np.char.upper(atom_array.element.astype(str))
    res_name = np.char.upper(atom_array.res_name.astype(str))
    target_mask = finite_mask & hetero_mask & ((element == metal_element) | (res_name == record.ccd_code))
    target_indices = np.where(target_mask)[0]

    base_record.update(
        {
            "metal_rasa_status": "parsed",
            "n_atoms": int(len(atom_array)),
            "n_protein_heavy_atoms": int(protein_heavy_finite.sum()),
            "n_target_metal_sites": int(len(target_indices)),
        }
    )
    if len(target_indices) == 0:
        base_record["metal_rasa_detail"] = (
            f"No target metal site found for ccd={record.ccd_code}, element={metal_element}"
        )
        return base_record

    all_radii, fallback_counts = make_vdw_radii(
        atom_array,
        set(int(index) for index in target_indices),
        metal_element,
    )
    base_record["radius_fallback_counts_json"] = json.dumps(fallback_counts, sort_keys=True)

    site_records = []
    for site_index, target_index in enumerate(target_indices, start=1):
        status, detail, n_occluders, complex_sasa, ratio = compute_target_sasa(
            atom_array=atom_array,
            target_index=int(target_index),
            occluder_mask=protein_heavy_finite,
            all_radii=all_radii,
            free_sasa=free_sasa,
            probe_radius=probe_radius,
            point_number=point_number,
        )
        site_records.append((status, detail, site_index, int(target_index), n_occluders, complex_sasa, ratio))

    ok_records = [site for site in site_records if site[0] == "ok" and site[6] is not None]
    if not ok_records:
        status, detail, *_ = site_records[0]
        base_record["metal_rasa_status"] = status
        base_record["metal_rasa_detail"] = detail
        return base_record

    status, detail, site_index, target_index, n_occluders, complex_sasa, ratio = min(
        ok_records,
        key=lambda site: (float(site[6]), int(site[2])),
    )
    base_record.update(
        {
            "metal_rasa_status": status,
            "metal_rasa_detail": detail,
            "site_index_in_file": site_index,
            "atom_index": target_index,
            "chain_id": atom_array.chain_id[target_index],
            "res_id": atom_array.res_id[target_index],
            "res_name": atom_array.res_name[target_index],
            "atom_name": atom_array.atom_name[target_index],
            "element": atom_array.element[target_index],
            "protein_all_heavy_n_occluder_atoms": n_occluders,
            "protein_all_heavy_complex_sasa": complex_sasa,
            "rasa_value": ratio,
        }
    )
    return base_record
