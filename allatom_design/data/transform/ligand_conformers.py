from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import atomworks.constants as aw_const
import atomworks.enums as aw_enums
import numpy as np
from atomworks.io.tools.rdkit import atom_array_to_rdkit
from biotite.structure import AtomArray
from biotite.structure.info import vdw_radius_single
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from allatom_design.data.const import METAL_ELEMENTS

WATER_RES_NAMES = {"HOH", "WAT", "DOD"}
CLASH_TARGET_ATOMS = {"sidechain", "backbone", "all_protein"}


@dataclass(frozen=True)
class TargetLigand:
    pn_unit_iid: str
    res_name: str
    mask: np.ndarray
    heavy_mask: np.ndarray


@dataclass(frozen=True)
class LigandConformerDecoy:
    atom_array: AtomArray
    rank: int
    cluster_id: int | None
    rdkit_conformer_id: int
    candidate_seed: int
    rmsd_to_native: float
    atom_order_rmsd_to_native: float
    clash_metrics: dict[str, Any]


@dataclass(frozen=True)
class LigandConformerGenerationResult:
    decoys: list[LigandConformerDecoy]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _NativeThresholdSelection:
    position: int
    get_best_rms_to_native: float
    atom_order_rmsd_to_native: float
    aligned_coords: np.ndarray


def select_target_ligand(
    atom_array: AtomArray,
    *,
    query_pn_unit_iids: list[str] | None = None,
) -> TargetLigand:
    """Select one non-metal small-molecule ligand for conformer generation."""
    candidates = find_query_small_molecule_ligands(
        atom_array,
        query_pn_unit_iids=query_pn_unit_iids,
    )
    if query_pn_unit_iids:
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one query small-molecule ligand, "
                f"found {len(candidates)}"
            )
        return candidates[0]

    if len(candidates) != 1:
        raise ValueError(
            "Automatic ligand conformer selection requires exactly one "
            f"non-metal small-molecule ligand, found {len(candidates)}"
        )
    return candidates[0]


def find_query_small_molecule_ligands(
    atom_array: AtomArray,
    *,
    query_pn_unit_iids: list[str] | None = None,
) -> list[TargetLigand]:
    """Return non-metal small-molecule ligands matching query PN-unit IDs."""
    candidates = _small_molecule_ligand_groups(atom_array)
    if not query_pn_unit_iids:
        return candidates
    query_ids = {str(pn_unit_iid) for pn_unit_iid in query_pn_unit_iids}
    return [
        candidate for candidate in candidates
        if candidate.pn_unit_iid in query_ids
    ]


def generate_ligand_conformer_decoys(
    atom_array: AtomArray,
    *,
    target_ligand: TargetLigand,
    num_candidates: int,
    num_representatives: int,
    rmsd_cluster_cutoff: float,
    seed: int,
    num_threads: int = 1,
    uff_optimize: bool = True,
    clash_target_atoms: str = "all_protein",
    vdw_overlap_cutoff: float = 0.5,
) -> LigandConformerGenerationResult:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if num_representatives <= 0:
        return LigandConformerGenerationResult(
            decoys=[],
            metadata={
                "conformer_selection_metric": "rdkit_get_best_rms_to_native_heavy",
                "conformer_selection_cutoff": float(rmsd_cluster_cutoff),
                "rdkit_requested_candidates": int(num_candidates),
                "rdkit_generated_candidates": 0,
                "rdkit_failed_candidates": 0,
                "rdkit_fallback_random_coords_count": 0,
                "rdkit_native_threshold_hit_count": 0,
            },
        )
    native_rmsd_cutoff = float(rmsd_cluster_cutoff)
    if native_rmsd_cutoff <= 0:
        raise ValueError("rmsd_cluster_cutoff/native RMSD cutoff must be positive")

    ligand_heavy = atom_array[target_ligand.heavy_mask]
    native_coords = np.asarray(ligand_heavy.coord, dtype=float)
    native_elements = _normalized_elements(ligand_heavy)
    mol = _atom_array_to_mol(ligand_heavy)

    work_mol, rdkit_metadata = _generate_candidate_mol(
        mol,
        num_candidates=num_candidates,
        seed=seed,
        num_threads=num_threads,
        uff_optimize=uff_optimize,
    )
    if work_mol.GetNumConformers() == 0:
        raise ValueError("RDKit generated no valid conformers")

    heavy_atom_indices = [
        atom.GetIdx() for atom in work_mol.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    rdkit_elements = [
        work_mol.GetAtomWithIdx(idx).GetSymbol().upper()
        for idx in heavy_atom_indices
    ]
    if rdkit_elements != native_elements:
        raise ValueError(
            "RDKit heavy-atom order does not match target ligand atom order: "
            f"native={native_elements} rdkit={rdkit_elements}"
        )

    selections, hit_count = _select_native_threshold_conformers(
        work_mol,
        atom_ids=heavy_atom_indices,
        native_coords=native_coords,
        rmsd_cutoff=native_rmsd_cutoff,
        target_count=min(num_representatives, work_mol.GetNumConformers()),
        num_threads=num_threads,
    )

    decoys: list[LigandConformerDecoy] = []
    conformers = list(work_mol.GetConformers())
    conformer_metadata = rdkit_metadata["conformers"]
    for rank, selection in enumerate(selections, start=1):
        position = selection.position
        conformer = conformers[position]
        aligned_coords = selection.aligned_coords
        decoy_atom_array = atom_array.copy()
        decoy_atom_array.coord[target_ligand.heavy_mask] = aligned_coords
        clash_metrics = compute_ligand_protein_clash_metrics(
            decoy_atom_array,
            ligand_mask=target_ligand.heavy_mask,
            clash_target_atoms=clash_target_atoms,
            vdw_overlap_cutoff=vdw_overlap_cutoff,
        )
        decoys.append(
            LigandConformerDecoy(
                atom_array=decoy_atom_array,
                rank=rank,
                cluster_id=None,
                rdkit_conformer_id=int(conformer.GetId()),
                candidate_seed=int(conformer_metadata[position]["candidate_seed"]),
                rmsd_to_native=float(selection.get_best_rms_to_native),
                atom_order_rmsd_to_native=float(selection.atom_order_rmsd_to_native),
                clash_metrics=clash_metrics,
            )
        )
    return LigandConformerGenerationResult(
        decoys=decoys,
        metadata={
            "conformer_selection_metric": "rdkit_get_best_rms_to_native_heavy",
            "conformer_selection_cutoff": native_rmsd_cutoff,
            "rdkit_requested_candidates": int(rdkit_metadata["requested_candidates"]),
            "rdkit_generated_candidates": int(rdkit_metadata["generated_candidates"]),
            "rdkit_failed_candidates": int(rdkit_metadata["failed_candidates"]),
            "rdkit_failed_candidate_seeds": ";".join(
                str(candidate_seed)
                for candidate_seed in rdkit_metadata["failed_candidate_seeds"]
            ),
            "rdkit_fallback_random_coords_used": bool(
                rdkit_metadata["fallback_random_coords_used"]
            ),
            "rdkit_fallback_random_coords_count": int(
                rdkit_metadata["fallback_random_coords_count"]
            ),
            "rdkit_candidate_seed_start": rdkit_metadata["candidate_seed_start"],
            "rdkit_candidate_seed_end": rdkit_metadata["candidate_seed_end"],
            "rdkit_native_threshold_hit_count": int(hit_count),
        },
    )


def compute_ligand_protein_clash_metrics(
    atom_array: AtomArray,
    *,
    ligand_mask: np.ndarray,
    clash_target_atoms: str = "all_protein",
    vdw_overlap_cutoff: float = 0.5,
) -> dict[str, Any]:
    if clash_target_atoms not in CLASH_TARGET_ATOMS:
        raise ValueError(
            f"clash_target_atoms must be one of {sorted(CLASH_TARGET_ATOMS)}, "
            f"got {clash_target_atoms!r}"
        )

    ligand_heavy = ligand_mask & _heavy_atom_mask(atom_array) & _finite_coord_mask(atom_array)
    protein_heavy = (
        _protein_atom_mask(atom_array)
        & _heavy_atom_mask(atom_array)
        & _finite_coord_mask(atom_array)
    )
    backbone = np.isin(np.asarray(atom_array.atom_name).astype(str), aw_const.PROTEIN_BACKBONE_ATOM_NAMES)
    if clash_target_atoms == "sidechain":
        target_mask = protein_heavy & ~backbone
    elif clash_target_atoms == "backbone":
        target_mask = protein_heavy & backbone
    else:
        target_mask = protein_heavy

    ligand_indices = np.flatnonzero(ligand_heavy)
    target_indices = np.flatnonzero(target_mask)
    if len(ligand_indices) == 0 or len(target_indices) == 0:
        return {
            "clash_target_atoms": clash_target_atoms,
            "vdw_overlap_cutoff": float(vdw_overlap_cutoff),
            "has_clash": False,
            "num_clashing_pairs": 0,
            "min_heavy_atom_distance": None,
            "max_vdw_overlap": None,
        }

    ligand_coords = atom_array.coord[ligand_indices].astype(float)
    target_coords = atom_array.coord[target_indices].astype(float)
    deltas = ligand_coords[:, None, :] - target_coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)

    ligand_radii = np.asarray([
        _vdw_radius(atom_array.element[idx]) for idx in ligand_indices
    ])
    target_radii = np.asarray([
        _vdw_radius(atom_array.element[idx]) for idx in target_indices
    ])
    overlaps = ligand_radii[:, None] + target_radii[None, :] - distances
    clashing = overlaps >= float(vdw_overlap_cutoff)
    return {
        "clash_target_atoms": clash_target_atoms,
        "vdw_overlap_cutoff": float(vdw_overlap_cutoff),
        "has_clash": bool(np.any(clashing)),
        "num_clashing_pairs": int(np.count_nonzero(clashing)),
        "min_heavy_atom_distance": float(np.min(distances)),
        "max_vdw_overlap": float(np.max(overlaps)),
    }


def _small_molecule_ligand_groups(atom_array: AtomArray) -> list[TargetLigand]:
    if "pn_unit_iid" not in atom_array.get_annotation_categories():
        raise ValueError("pn_unit_iid annotation is required for ligand selection")

    nonpolymer_mask = _nonpolymer_atom_mask(atom_array)
    groups: list[TargetLigand] = []
    for pn_unit_iid in sorted(np.unique(atom_array.pn_unit_iid[nonpolymer_mask]).tolist()):
        mask = nonpolymer_mask & (atom_array.pn_unit_iid == pn_unit_iid)
        if not np.any(mask):
            continue
        res_names = sorted(set(np.asarray(atom_array.res_name[mask]).astype(str).tolist()))
        res_name = res_names[0] if res_names else ""
        elements = set(_normalized_elements(atom_array[mask]))
        if res_name.upper() in WATER_RES_NAMES:
            continue
        if elements and elements.issubset(METAL_ELEMENTS):
            continue
        heavy_mask = mask & _heavy_atom_mask(atom_array) & _finite_coord_mask(atom_array)
        if not np.any(heavy_mask):
            continue
        groups.append(
            TargetLigand(
                pn_unit_iid=str(pn_unit_iid),
                res_name=str(res_name),
                mask=mask,
                heavy_mask=heavy_mask,
            )
        )
    return groups


def _nonpolymer_atom_mask(atom_array: AtomArray) -> np.ndarray:
    annotations = atom_array.get_annotation_categories()
    if "chain_type" in annotations:
        chain_types = np.asarray(atom_array.chain_type).astype(int)
        nonpolymer_values = [chain_type.value for chain_type in aw_enums.ChainTypeInfo.NON_POLYMERS]
        return np.isin(chain_types, nonpolymer_values)
    if "is_polymer" in annotations:
        return ~np.asarray(atom_array.is_polymer).astype(bool)
    return np.asarray(atom_array.hetero).astype(bool)


def _protein_atom_mask(atom_array: AtomArray) -> np.ndarray:
    annotations = atom_array.get_annotation_categories()
    if "chain_type" in annotations:
        chain_types = np.asarray(atom_array.chain_type).astype(int)
        protein_values = [chain_type.value for chain_type in aw_enums.ChainTypeInfo.PROTEINS]
        return np.isin(chain_types, protein_values)
    return np.zeros(len(atom_array), dtype=bool)


def _heavy_atom_mask(atom_array: AtomArray) -> np.ndarray:
    return np.asarray([str(element).upper() != "H" for element in atom_array.element])


def _finite_coord_mask(atom_array: AtomArray) -> np.ndarray:
    return np.isfinite(np.asarray(atom_array.coord, dtype=float)).all(axis=-1)


def _normalized_elements(atom_array: AtomArray) -> list[str]:
    return [str(element).strip().upper() for element in atom_array.element]


def _atom_array_to_mol(ligand_heavy: AtomArray) -> Chem.Mol:
    try:
        return atom_array_to_rdkit(
            ligand_heavy,
            set_coord=True,
            hydrogen_policy="remove",
            sanitize=True,
        )
    except Exception:
        return atom_array_to_rdkit(
            ligand_heavy,
            set_coord=True,
            hydrogen_policy="remove",
            sanitize=False,
        )


def _generate_candidate_mol(
    mol: Chem.Mol,
    *,
    num_candidates: int,
    seed: int,
    num_threads: int,
    uff_optimize: bool,
) -> tuple[Chem.Mol, dict[str, Any]]:
    work_mol = Chem.AddHs(copy.deepcopy(mol), addCoords=True)
    work_mol.RemoveAllConformers()

    conformer_metadata: list[dict[str, Any]] = []
    failed_candidate_seeds: list[int] = []
    candidate_seed_start = _first_candidate_seed(seed)
    for candidate_index in range(int(num_candidates)):
        candidate_seed = candidate_seed_start + candidate_index
        candidate_mol = Chem.AddHs(copy.deepcopy(mol), addCoords=True)
        candidate_mol.RemoveAllConformers()
        conf_id, fallback_used = _embed_single_conformer(
            candidate_mol,
            seed=candidate_seed,
            num_threads=num_threads,
        )
        if conf_id < 0:
            failed_candidate_seeds.append(candidate_seed)
            continue
        if uff_optimize:
            try:
                AllChem.UFFOptimizeMolecule(
                    candidate_mol,
                    confId=int(conf_id),
                    maxIters=200,
                )
            except Exception:
                pass
        new_conf_id = work_mol.AddConformer(
            Chem.Conformer(candidate_mol.GetConformer(int(conf_id))),
            assignId=True,
        )
        conformer_metadata.append(
            {
                "candidate_index": int(candidate_index),
                "candidate_seed": int(candidate_seed),
                "fallback_random_coords_used": bool(fallback_used),
                "rdkit_conformer_id": int(new_conf_id),
            }
        )

    return work_mol, {
        "requested_candidates": int(num_candidates),
        "generated_candidates": int(len(conformer_metadata)),
        "failed_candidates": int(len(failed_candidate_seeds)),
        "failed_candidate_seeds": failed_candidate_seeds,
        "fallback_random_coords_used": any(
            bool(entry["fallback_random_coords_used"])
            for entry in conformer_metadata
        ),
        "fallback_random_coords_count": sum(
            1 for entry in conformer_metadata
            if bool(entry["fallback_random_coords_used"])
        ),
        "candidate_seed_start": int(candidate_seed_start),
        "candidate_seed_end": int(candidate_seed_start + int(num_candidates) - 1),
        "conformers": conformer_metadata,
    }


def _first_candidate_seed(seed: int) -> int:
    # RDKit treats seed 0 pathologically for some ligands; keep candidate seeds
    # positive and deterministic.
    return max(int(seed) + 1, 1)


def _embed_single_conformer(
    mol: Chem.Mol,
    *,
    seed: int,
    num_threads: int,
) -> tuple[int, bool]:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = int(num_threads)
    params.pruneRmsThresh = -1.0
    params.onlyHeavyAtomsForRMS = True
    conf_id = int(AllChem.EmbedMolecule(mol, params))
    if conf_id >= 0:
        return conf_id, False

    fallback = AllChem.ETKDGv3()
    fallback.randomSeed = int(seed) + 104729
    fallback.numThreads = int(num_threads)
    fallback.pruneRmsThresh = -1.0
    fallback.onlyHeavyAtomsForRMS = True
    fallback.useRandomCoords = True
    return int(AllChem.EmbedMolecule(mol, fallback)), True


def _select_native_threshold_conformers(
    mol: Chem.Mol,
    *,
    atom_ids: list[int],
    native_coords: np.ndarray,
    rmsd_cutoff: float,
    target_count: int,
    num_threads: int,
) -> tuple[list[_NativeThresholdSelection], int]:
    heavy_mol = _heavy_only_conformer_mol(
        mol,
        atom_ids=atom_ids,
        native_coords=native_coords,
    )
    raw_coords_by_position = [
        _conformer_coords(conformer, atom_ids)
        for conformer in mol.GetConformers()
    ]
    for coords in raw_coords_by_position:
        _add_coords_as_conformer(heavy_mol, coords)

    hits: list[_NativeThresholdSelection] = []
    for position, conformer_coords in enumerate(raw_coords_by_position):
        get_best_rms, aligned_coords = _get_best_rms_aligned_coords(
            heavy_mol,
            prb_conf_id=position + 1,
            ref_conf_id=0,
            num_threads=num_threads,
        )
        if get_best_rms <= float(rmsd_cutoff):
            hits.append(
                _NativeThresholdSelection(
                    position=position,
                    get_best_rms_to_native=float(get_best_rms),
                    atom_order_rmsd_to_native=float(
                        _rmsd(_kabsch_align(conformer_coords, native_coords), native_coords)
                    ),
                    aligned_coords=aligned_coords,
                )
            )

    hits.sort(key=lambda selection: (selection.get_best_rms_to_native, selection.position))
    return hits[:target_count], len(hits)


def _heavy_only_conformer_mol(
    mol: Chem.Mol,
    *,
    atom_ids: list[int],
    native_coords: np.ndarray,
) -> Chem.Mol:
    heavy_mol = Chem.RemoveHs(copy.deepcopy(mol))
    heavy_mol.RemoveAllConformers()
    if heavy_mol.GetNumAtoms() != len(atom_ids):
        raise ValueError(
            "Heavy-only RDKit molecule does not match selected atom count: "
            f"mol={heavy_mol.GetNumAtoms()} atom_ids={len(atom_ids)}"
        )
    _add_coords_as_conformer(heavy_mol, native_coords)
    return heavy_mol


def _add_coords_as_conformer(mol: Chem.Mol, coords: np.ndarray) -> int:
    if mol.GetNumAtoms() != len(coords):
        raise ValueError(
            "Coordinate count does not match RDKit atom count: "
            f"coords={len(coords)} atoms={mol.GetNumAtoms()}"
        )
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx, coord in enumerate(np.asarray(coords, dtype=float)):
        conformer.SetAtomPosition(
            int(atom_idx),
            tuple(float(value) for value in coord),
        )
    return int(mol.AddConformer(conformer, assignId=True))


def _conformer_coords(conformer: Chem.Conformer, atom_ids: list[int]) -> np.ndarray:
    return np.asarray(
        [
            list(conformer.GetAtomPosition(atom_idx))
            for atom_idx in atom_ids
        ],
        dtype=float,
    )


def _get_best_rms_aligned_coords(
    mol: Chem.Mol,
    *,
    prb_conf_id: int,
    ref_conf_id: int,
    num_threads: int,
) -> tuple[float, np.ndarray]:
    rmsd, transform, atom_map = rdMolAlign.GetBestAlignmentTransform(
        mol,
        mol,
        prbCid=int(prb_conf_id),
        refCid=int(ref_conf_id),
        numThreads=int(num_threads),
    )
    transformed = Chem.Conformer(mol.GetConformer(int(prb_conf_id)))
    AllChem.TransformConformer(transformed, transform)
    transformed_coords = np.asarray(
        [
            list(transformed.GetAtomPosition(atom_idx))
            for atom_idx in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )
    aligned_coords = np.full_like(transformed_coords, np.nan)
    for probe_atom_idx, ref_atom_idx in atom_map:
        aligned_coords[int(ref_atom_idx)] = transformed_coords[int(probe_atom_idx)]
    if not np.isfinite(aligned_coords).all():
        raise ValueError("RDKit best alignment atom map did not cover all ligand atoms")
    return float(rmsd), aligned_coords


def _kabsch_align(mobile_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    mobile_center = mobile_coords.mean(axis=0)
    target_center = target_coords.mean(axis=0)
    mobile_centered = mobile_coords - mobile_center
    target_centered = target_coords - target_center
    covariance = mobile_centered.T @ target_centered
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_t[-1, :] *= -1
        rotation = right_t.T @ left.T
    return mobile_centered @ rotation + target_center


def _com_align(mobile_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    return mobile_coords - mobile_coords.mean(axis=0) + target_coords.mean(axis=0)


def _rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    delta = coords_a - coords_b
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=-1))))


def _vdw_radius(element: Any) -> float:
    normalized = str(element).strip().upper()
    try:
        radius = vdw_radius_single(normalized)
    except Exception:
        radius = None
    if radius is None or not np.isfinite(radius):
        return 1.7
    return float(radius)
