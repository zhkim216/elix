#!/usr/bin/env python3
"""Evaluate and select SwitchCraft designs using the paper's stated criteria.

This command is deliberately manifest-driven.  It reads structures in place and
writes membership views; it never copies or links a generated structure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from utils.geometry import compute_rmsd
from utils.protein import from_pdb_string
from utils.residue_constants import atom_order


FAMILIES = (
    "positive_allostery",
    "negative_allostery",
    "motif_switching",
    "ligand_modification",
    "ligand_discrimination",
)
FROZEN_FAMILY_CASES = {
    "positive_allostery": (120, 100, 2),
    "negative_allostery": (120, 100, 2),
    "motif_switching": (1, 1000, 2),
    "ligand_modification": (1, 558, 2),
    "ligand_discrimination": (1, 465, 3),
}
STATE_LABELS = ("A", "B", "C")
SAMPLES_PER_STATE = 5
CA_INDEX = atom_order["CA"]
SCHEMA_VERSION = 1
_MOTIF_REFERENCE_CA_CACHE: dict[str, np.ndarray] = {}


class SelectionError(RuntimeError):
    """A contract or input error that must fail the run closed."""


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


@dataclass(frozen=True)
class DesignTask:
    case: Mapping[str, Any]
    config: Mapping[str, Any]
    design_index: int
    source_root: str
    motif_reference_paths: Mapping[str, str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _bool(value: bool) -> bool:
    return bool(value)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: Sequence[float], ddof: int) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=ddof))


def _is_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.lower() == "false")


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_state_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        value = CPUUnpickler(handle).load()
    if not isinstance(value, Mapping):
        raise SelectionError(f"state PKL is not a mapping: {path}")
    if "exception" not in value or not _is_false(value["exception"]):
        raise SelectionError(f"state PKL exception is not false: {path}")
    if "coords" not in value or len(_as_numpy(value["coords"])) != SAMPLES_PER_STATE:
        raise SelectionError(f"state PKL does not contain exactly five entries: {path}")
    for key in ("plddt", "ptm", "ligand_iptm"):
        if key in value and len(_as_numpy(value[key])) != SAMPLES_PER_STATE:
            raise SelectionError(f"state PKL {key} does not contain exactly five entries: {path}")
    return value


def _diagnostic(value: Mapping[str, Any], key: str, sample_index: int) -> float | None:
    if key not in value:
        return None
    item = _as_numpy(value[key])[sample_index]
    # pLDDT intentionally averages the whole saved token axis.
    return float(np.mean(item))


def _parse_protein(path: Path) -> tuple[Any, np.ndarray, np.ndarray]:
    parsed = from_pdb_string(path.read_text())
    ca = np.asarray(parsed.atom_positions[:, CA_INDEX], dtype=np.float32)
    valid = np.asarray(parsed.atom_mask[:, CA_INDEX] > 0.5, dtype=bool)
    if ca.ndim != 2 or ca.shape[1] != 3 or len(ca) != len(valid):
        raise SelectionError(f"invalid parsed CA coordinate shape: {path}")
    if not np.isfinite(ca[valid]).all():
        raise SelectionError(f"non-finite valid CA coordinate: {path}")
    return parsed, ca, valid


def _motif_reference_ca(path: Path) -> np.ndarray:
    cache_key = str(path.resolve())
    if cache_key not in _MOTIF_REFERENCE_CA_CACHE:
        _, ca, valid = _parse_protein(path)
        if not valid.all():
            raise SelectionError(f"reference motif has invalid CA coordinates: {path}")
        _MOTIF_REFERENCE_CA_CACHE[cache_key] = ca
    return _MOTIF_REFERENCE_CA_CACHE[cache_key]


def _validate_ligand_ca(ca: np.ndarray, valid: np.ndarray, path: Path) -> np.ndarray:
    if len(ca) != 50 or len(valid) != 50 or not valid.all():
        raise SelectionError(
            f"expected exactly 50 protein residues with valid CA coordinates: {path}"
        )
    return ca


def _aligned_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise SelectionError(f"RMSD coordinate shape mismatch: {first.shape} vs {second.shape}")
    value = compute_rmsd(torch.from_numpy(first), torch.from_numpy(second))
    result = float(value.item())
    if not np.isfinite(result):
        raise SelectionError("RMSD is not finite")
    return result


def _load_motif_mask(path: Path, residue_count: int) -> np.ndarray:
    with path.open("rb") as handle:
        spec = CPUUnpickler(handle).load()
    if not isinstance(spec, Mapping) or "motif_mask" not in spec:
        raise SelectionError(f"motif_mask missing from spec: {path}")
    mask = _as_numpy(spec["motif_mask"])
    if mask.ndim != 1 or len(mask) != residue_count:
        raise SelectionError(
            f"motif mask length {len(mask)} does not match {residue_count} residues: {path}"
        )
    if mask.dtype != np.bool_ and not np.isin(mask, (0, 1)).all():
        raise SelectionError(f"motif mask is not boolean: {path}")
    return mask.astype(bool)


def _identity_columns(task: DesignTask, design_dir: Path) -> dict[str, Any]:
    case = task.case
    source_root = Path(task.source_root)
    identity = {
        "family": case["family"],
        "case_id": case["case_id"],
        "case_index": case["case_index"],
        "motif_key": case["motif_key"],
        "ligand_key": case["ligand_key"],
        "motifs": _canonical_json(case["motifs"]),
        "num_states": case["num_states"],
        "case_design_count": case["num_designs"],
        "design_index": task.design_index,
        "design_name": f"design{task.design_index}",
        "config_path": str(Path(case["config_path"]).resolve()),
        "config_sha256": case["config_sha256"],
        "case_design_root": str(Path(case["design_root"]).resolve()),
        "design_path": str(design_dir.resolve()),
        "design_path_relative_to_source_root": str(
            design_dir.resolve().relative_to(source_root.resolve())
        ),
    }
    for state_index, state_label in enumerate(STATE_LABELS[: case["num_states"]]):
        identity[f"state_{state_label}_config"] = _canonical_json(task.config["states"][state_index])
    return identity


def _state_diagnostics(
    output: Mapping[str, Any], state_label: str
) -> tuple[list[dict[str, float | None]], dict[str, float | None]]:
    samples = []
    for sample_index in range(SAMPLES_PER_STATE):
        samples.append(
            {
                "plddt_mean": _diagnostic(output, "plddt", sample_index),
                "ptm": _diagnostic(output, "ptm", sample_index),
                "ligand_iptm": _diagnostic(output, "ligand_iptm", sample_index),
            }
        )
    aggregate: dict[str, float | None] = {}
    for name in ("plddt_mean", "ptm", "ligand_iptm"):
        values = [row[name] for row in samples if row[name] is not None]
        aggregate[f"state_{state_label}_{name}"] = _mean(values) if values else None
    return samples, aggregate


def allostery_gates(
    family: str, state_a_mean: float, state_a_std: float, state_b_mean: float, state_b_std: float
) -> dict[str, Any]:
    if family == "positive_allostery":
        a_margin, b_margin = state_a_mean - 1.0, 1.0 - state_b_mean
        a_pass, b_pass = state_a_mean > 1.0, state_b_mean <= 1.0
    elif family == "negative_allostery":
        a_margin, b_margin = 1.0 - state_a_mean, state_b_mean - 1.0
        a_pass, b_pass = state_a_mean <= 1.0, state_b_mean > 1.0
    else:
        raise ValueError(f"not an allostery family: {family}")
    std_a_margin, std_b_margin = 0.5 - state_a_std, 0.5 - state_b_std
    std_a_pass, std_b_pass = state_a_std <= 0.5, state_b_std <= 0.5
    delta = abs(state_a_mean - state_b_mean)
    appendix_delta_margin = delta - 0.5
    main_delta_margin = delta - 1.0
    component_pass = a_pass and b_pass and std_a_pass and std_b_pass
    appendix_pass = component_pass and delta >= 0.5
    main_pass = component_pass and delta > 1.0
    appendix_score = min(a_margin, b_margin, std_a_margin, std_b_margin, appendix_delta_margin)
    main_score = min(a_margin, b_margin, std_a_margin, std_b_margin, main_delta_margin)
    return {
        "state_A_direction_margin": a_margin,
        "state_A_direction_pass": _bool(a_pass),
        "state_B_direction_margin": b_margin,
        "state_B_direction_pass": _bool(b_pass),
        "state_A_std_margin": std_a_margin,
        "state_A_std_pass": _bool(std_a_pass),
        "state_B_std_margin": std_b_margin,
        "state_B_std_pass": _bool(std_b_pass),
        "direction_and_std_pass": _bool(component_pass),
        "absolute_mean_delta": delta,
        "appendix_b1_delta_margin": appendix_delta_margin,
        "appendix_b1_delta_pass": _bool(delta >= 0.5),
        "main_text_4_1_delta_margin": main_delta_margin,
        "main_text_4_1_delta_pass": _bool(delta > 1.0),
        "appendix_b1_pass": _bool(appendix_pass),
        "appendix_b1_score": appendix_score,
        "main_text_4_1_pass": _bool(main_pass),
        "main_text_4_1_score": main_score,
    }


def motif_switching_gates(metrics: Mapping[str, float]) -> dict[str, Any]:
    expected = {
        ("A", "3ixt"): ("le", 1.0),
        ("A", "1ycr"): ("gt", 1.0),
        ("B", "3ixt"): ("gt", 1.0),
        ("B", "1ycr"): ("le", 1.0),
    }
    result: dict[str, Any] = {}
    for (state, motif), (operator, threshold) in expected.items():
        mean = metrics[f"motif_{motif}_state_{state}_mean"]
        if operator == "le":
            margin, passed = threshold - mean, mean <= threshold
        else:
            margin, passed = mean - threshold, mean > threshold
        std = metrics[f"motif_{motif}_state_{state}_std"]
        std_margin, std_passed = 0.5 - std, std <= 0.5
        prefix = f"motif_{motif}_state_{state}"
        result[f"{prefix}_direction_margin"] = margin
        result[f"{prefix}_direction_pass"] = _bool(passed)
        result[f"{prefix}_std_margin"] = std_margin
        result[f"{prefix}_std_pass"] = _bool(std_passed)
        result[f"{prefix}_component_pass"] = _bool(passed and std_passed)
    motif_switch_passes = []
    motif_switch_scores = []
    for motif in ("3ixt", "1ycr"):
        motif_pass = all(
            result[f"motif_{motif}_state_{state}_component_pass"] for state in ("A", "B")
        )
        motif_score = min(
            result[f"motif_{motif}_state_{state}_{component}_margin"]
            for state in ("A", "B")
            for component in ("direction", "std")
        )
        result[f"motif_{motif}_switch_pass"] = _bool(motif_pass)
        result[f"motif_{motif}_switch_score"] = motif_score
        motif_switch_passes.append(motif_pass)
        motif_switch_scores.append(motif_score)
    result["appendix_b2_pass"] = _bool(all(motif_switch_passes))
    result["appendix_b2_score"] = min(motif_switch_scores)
    return result


def ligand_modification_gates(cross: float, intra_a: float, intra_b: float) -> dict[str, Any]:
    cross_margin, a_margin, b_margin = cross - 2.0, 1.0 - intra_a, 1.0 - intra_b
    cross_pass, a_pass, b_pass = cross > 2.0, intra_a < 1.0, intra_b < 1.0
    literal_pass = cross_pass and (a_pass or b_pass)
    both_pass = cross_pass and a_pass and b_pass
    return {
        "cross_AB_margin": cross_margin,
        "cross_AB_pass": _bool(cross_pass),
        "intra_A_margin": a_margin,
        "intra_A_pass": _bool(a_pass),
        "intra_B_margin": b_margin,
        "intra_B_pass": _bool(b_pass),
        "appendix_b3_literal_or_pass": _bool(literal_pass),
        "appendix_b3_literal_or_score": min(cross_margin, max(a_margin, b_margin)),
        "appendix_b3_both_states_pass": _bool(both_pass),
        "appendix_b3_both_states_score": min(cross_margin, a_margin, b_margin),
    }


def ligand_discrimination_gates(metrics: Mapping[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    margins: list[float] = []
    passes: list[bool] = []
    for state in STATE_LABELS:
        value = metrics[f"intra_{state}_mean"]
        margin, passed = 1.0 - value, value < 1.0
        result[f"intra_{state}_margin"] = margin
        result[f"intra_{state}_pass"] = _bool(passed)
        margins.append(margin)
        passes.append(passed)
    for pair in ("AB", "AC", "BC"):
        value = metrics[f"cross_{pair}_mean"]
        margin, passed = value - 1.0, value > 1.0
        result[f"cross_{pair}_margin"] = margin
        result[f"cross_{pair}_pass"] = _bool(passed)
        margins.append(margin)
        passes.append(passed)
    result["appendix_b5_pass"] = _bool(all(passes))
    result["appendix_b5_score"] = min(margins)
    return result


def _evaluate_motif_design(task: DesignTask, design_dir: Path) -> dict[str, Any]:
    motifs = list(task.case["motifs"])
    references = {
        motif: _motif_reference_ca(Path(task.motif_reference_paths[motif])) for motif in motifs
    }

    state_outputs = []
    state_sample_diags = []
    for state_index in range(task.case["num_states"]):
        output = _load_state_pickle(design_dir / f"state{state_index}.pkl")
        diagnostics, _ = _state_diagnostics(output, STATE_LABELS[state_index])
        state_outputs.append(output)
        state_sample_diags.append(diagnostics)

    parsed_samples: list[list[tuple[Any, np.ndarray, np.ndarray]]] = []
    for state_index in range(task.case["num_states"]):
        parsed_samples.append(
            [
                _parse_protein(design_dir / f"state{state_index}_sample{sample_index}.pdb")
                for sample_index in range(SAMPLES_PER_STATE)
            ]
        )
    residue_count = len(parsed_samples[0][0][1])
    masks = {
        motif: _load_motif_mask(design_dir / f"{motif}_spec.pkl", residue_count)
        for motif in motifs
    }

    identity = _identity_columns(task, design_dir)
    samples: list[dict[str, Any]] = []
    motif_values: dict[tuple[str, str], list[float]] = {
        (state, motif): []
        for state in STATE_LABELS[: task.case["num_states"]]
        for motif in motifs
    }
    for state_index, state_label in enumerate(STATE_LABELS[: task.case["num_states"]]):
        for sample_index, (_, ca, valid) in enumerate(parsed_samples[state_index]):
            if len(ca) != residue_count:
                raise SelectionError("sample residue counts differ within a motif design")
            row = {
                **identity,
                "state_index": state_index,
                "state_label": state_label,
                "state_config": _canonical_json(task.config["states"][state_index]),
                "sample_index": sample_index,
                "pdb_path": str((design_dir / f"state{state_index}_sample{sample_index}.pdb").resolve()),
                "cif_path": str((design_dir / f"state{state_index}_sample{sample_index}.cif").resolve()),
                "state_pkl_path": str((design_dir / f"state{state_index}.pkl").resolve()),
                "protein_residue_count": len(ca),
                "valid_ca_count": int(valid.sum()),
                **state_sample_diags[state_index][sample_index],
            }
            for motif in motifs:
                mask = masks[motif]
                selected = ca[mask]
                if not valid[mask].all():
                    raise SelectionError(f"invalid selected motif CA: state {state_label}, motif {motif}")
                if len(selected) != len(references[motif]):
                    raise SelectionError(
                        f"motif/reference length mismatch for {motif}: {len(selected)} vs {len(references[motif])}"
                    )
                rmsd = _aligned_rmsd(selected, references[motif])
                row[f"motif_{motif}_rmsd_angstrom"] = rmsd
                motif_values[(state_label, motif)].append(rmsd)
            samples.append(row)

    metrics: dict[str, Any] = dict(identity)
    for state_index, state_label in enumerate(STATE_LABELS[: task.case["num_states"]]):
        _, aggregate = _state_diagnostics(state_outputs[state_index], state_label)
        metrics.update(aggregate)
        metrics[f"state_{state_label}_sample_count"] = len(parsed_samples[state_index])
        for motif in motifs:
            values = motif_values[(state_label, motif)]
            metrics[f"motif_{motif}_state_{state_label}_count"] = len(values)
            metrics[f"motif_{motif}_state_{state_label}_mean"] = _mean(values)
            metrics[f"motif_{motif}_state_{state_label}_std"] = _std(values, ddof=1)

    if task.case["family"] in ("positive_allostery", "negative_allostery"):
        motif = motifs[0]
        metrics.update(
            allostery_gates(
                task.case["family"],
                metrics[f"motif_{motif}_state_A_mean"],
                metrics[f"motif_{motif}_state_A_std"],
                metrics[f"motif_{motif}_state_B_mean"],
                metrics[f"motif_{motif}_state_B_std"],
            )
        )
    else:
        metrics.update(motif_switching_gates(metrics))
    return {"metrics": metrics, "samples": samples, "pairs": []}


def _evaluate_ligand_design(task: DesignTask, design_dir: Path) -> dict[str, Any]:
    state_count = task.case["num_states"]
    identity = _identity_columns(task, design_dir)
    states: list[list[np.ndarray]] = []
    samples: list[dict[str, Any]] = []
    metrics: dict[str, Any] = dict(identity)
    for state_index, state_label in enumerate(STATE_LABELS[:state_count]):
        output = _load_state_pickle(design_dir / f"state{state_index}.pkl")
        diagnostics, aggregate = _state_diagnostics(output, state_label)
        metrics.update(aggregate)
        state_coords = []
        for sample_index in range(SAMPLES_PER_STATE):
            pdb_path = design_dir / f"state{state_index}_sample{sample_index}.pdb"
            _, ca, valid = _parse_protein(pdb_path)
            selected = _validate_ligand_ca(ca, valid, pdb_path)
            state_coords.append(selected)
            samples.append(
                {
                    **identity,
                    "state_index": state_index,
                    "state_label": state_label,
                    "state_config": _canonical_json(task.config["states"][state_index]),
                    "sample_index": sample_index,
                    "pdb_path": str(pdb_path.resolve()),
                    "cif_path": str((design_dir / f"state{state_index}_sample{sample_index}.cif").resolve()),
                    "state_pkl_path": str((design_dir / f"state{state_index}.pkl").resolve()),
                    "protein_residue_count": len(ca),
                    "valid_ca_count": int(valid.sum()),
                    "selected_valid_ca_count": 50,
                    **diagnostics[sample_index],
                }
            )
        states.append(state_coords)
        metrics[f"state_{state_label}_sample_count"] = len(state_coords)

    pairs: list[dict[str, Any]] = []
    intra_values: dict[str, list[float]] = {}
    for state_index, state_label in enumerate(STATE_LABELS[:state_count]):
        values = []
        for first in range(SAMPLES_PER_STATE):
            for second in range(first + 1, SAMPLES_PER_STATE):
                rmsd = _aligned_rmsd(states[state_index][first], states[state_index][second])
                values.append(rmsd)
                pairs.append(
                    {
                        **identity,
                        "pair_kind": "intra",
                        "state_1_index": state_index,
                        "state_1_label": state_label,
                        "sample_1_index": first,
                        "state_2_index": state_index,
                        "state_2_label": state_label,
                        "sample_2_index": second,
                        "pdb_1_path": str((design_dir / f"state{state_index}_sample{first}.pdb").resolve()),
                        "pdb_2_path": str((design_dir / f"state{state_index}_sample{second}.pdb").resolve()),
                        "aligned_ca_count": 50,
                        "ca_rmsd_angstrom": rmsd,
                    }
                )
        intra_values[state_label] = values
        metrics[f"intra_{state_label}_count"] = len(values)
        metrics[f"intra_{state_label}_mean"] = _mean(values)
        metrics[f"intra_{state_label}_std"] = _std(values, ddof=0)

    for first_state in range(state_count):
        for second_state in range(first_state + 1, state_count):
            first_label, second_label = STATE_LABELS[first_state], STATE_LABELS[second_state]
            pair_label = first_label + second_label
            values = []
            for first_sample in range(SAMPLES_PER_STATE):
                for second_sample in range(SAMPLES_PER_STATE):
                    rmsd = _aligned_rmsd(
                        states[first_state][first_sample], states[second_state][second_sample]
                    )
                    values.append(rmsd)
                    pairs.append(
                        {
                            **identity,
                            "pair_kind": "cross",
                            "state_1_index": first_state,
                            "state_1_label": first_label,
                            "sample_1_index": first_sample,
                            "state_2_index": second_state,
                            "state_2_label": second_label,
                            "sample_2_index": second_sample,
                            "pdb_1_path": str((design_dir / f"state{first_state}_sample{first_sample}.pdb").resolve()),
                            "pdb_2_path": str((design_dir / f"state{second_state}_sample{second_sample}.pdb").resolve()),
                            "aligned_ca_count": 50,
                            "ca_rmsd_angstrom": rmsd,
                        }
                    )
            metrics[f"cross_{pair_label}_count"] = len(values)
            metrics[f"cross_{pair_label}_mean"] = _mean(values)
            metrics[f"cross_{pair_label}_std"] = _std(values, ddof=0)

    if task.case["family"] == "ligand_modification":
        metrics.update(
            ligand_modification_gates(
                metrics["cross_AB_mean"], metrics["intra_A_mean"], metrics["intra_B_mean"]
            )
        )
    else:
        metrics.update(ligand_discrimination_gates(metrics))
    return {"metrics": metrics, "samples": samples, "pairs": pairs}


def _worker_init() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def evaluate_design(task: DesignTask) -> dict[str, Any]:
    design_dir = Path(task.case["design_root"]) / f"design{task.design_index}"
    try:
        if task.case["family"] in (
            "positive_allostery",
            "negative_allostery",
            "motif_switching",
        ):
            return _evaluate_motif_design(task, design_dir)
        return _evaluate_ligand_design(task, design_dir)
    except Exception as error:
        raise SelectionError(
            f"case={task.case['case_id']} design={task.design_index}: {error}"
        ) from error


def selection_memberships(metrics: Mapping[str, Any]) -> list[tuple[str, float]]:
    family = metrics["family"]
    definitions = {
        "positive_allostery": (
            ("appendix_b1", "appendix_b1_pass", "appendix_b1_score"),
            ("main_text_4_1", "main_text_4_1_pass", "main_text_4_1_score"),
        ),
        "negative_allostery": (
            ("appendix_b1", "appendix_b1_pass", "appendix_b1_score"),
            ("main_text_4_1", "main_text_4_1_pass", "main_text_4_1_score"),
        ),
        "motif_switching": (("appendix_b2", "appendix_b2_pass", "appendix_b2_score"),),
        "ligand_modification": (
            (
                "appendix_b3_literal_or",
                "appendix_b3_literal_or_pass",
                "appendix_b3_literal_or_score",
            ),
            (
                "appendix_b3_both_states",
                "appendix_b3_both_states_pass",
                "appendix_b3_both_states_score",
            ),
        ),
        "ligand_discrimination": (("appendix_b5", "appendix_b5_pass", "appendix_b5_score"),),
    }
    return [
        (selection_set, float(metrics[score_key]))
        for selection_set, pass_key, score_key in definitions[family]
        if metrics[pass_key]
    ]


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        manifest = json.loads(path.read_text())
    except Exception as error:
        raise SelectionError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise SelectionError("manifest root must be an object")
    return manifest, sha256_file(path)


def _load_and_validate_config(case: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(case["config_path"])
    if not path.is_file():
        raise SelectionError(f"missing config: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != case["config_sha256"]:
        raise SelectionError(
            f"config SHA256 mismatch for {case['case_id']}: {actual_hash} != {case['config_sha256']}"
        )
    try:
        config = json.loads(path.read_text())
    except Exception as error:
        raise SelectionError(f"config is not valid JSON/YAML subset: {path}: {error}") from error
    if config.get("num_states") != case["num_states"]:
        raise SelectionError(f"num_states mismatch for {case['case_id']}")
    if config.get("motifs") != case["motifs"]:
        raise SelectionError(f"motif identity mismatch for {case['case_id']}")
    if not isinstance(config.get("states"), list) or len(config["states"]) != case["num_states"]:
        raise SelectionError(f"state identity mismatch for {case['case_id']}")
    if case["family"] in ("ligand_modification", "ligand_discrimination") and config.get("length") != 50:
        raise SelectionError(f"ligand family length is not 50 for {case['case_id']}")
    losses = config.get("losses")
    if not isinstance(losses, list):
        raise SelectionError(f"loss list missing for {case['case_id']}")
    motif_state_losses = {
        (loss.get("motif"), loss.get("state"), loss.get("type"))
        for loss in losses
        if loss.get("type") in ("MotifLoss", "AntiMotifLoss")
    }
    if case["family"] == "positive_allostery":
        expected = {(0, 0, "AntiMotifLoss"), (0, 1, "MotifLoss")}
        if motif_state_losses != expected:
            raise SelectionError(f"positive allostery direction mismatch for {case['case_id']}")
    elif case["family"] == "negative_allostery":
        expected = {(0, 0, "MotifLoss"), (0, 1, "AntiMotifLoss")}
        if motif_state_losses != expected:
            raise SelectionError(f"negative allostery direction mismatch for {case['case_id']}")
    elif case["family"] == "motif_switching":
        expected = {
            (0, 0, "MotifLoss"),
            (0, 1, "AntiMotifLoss"),
            (1, 0, "AntiMotifLoss"),
            (1, 1, "MotifLoss"),
        }
        if motif_state_losses != expected:
            raise SelectionError(f"motif switching direction mismatch for {case['case_id']}")
    return config


def _validate_manifest_contract(manifest: Mapping[str, Any], source_root: Path) -> None:
    required = {"schema_version", "output_root", "cases", "scientific_sources", "total_designs"}
    missing = required - set(manifest)
    if missing:
        raise SelectionError(f"manifest missing keys: {sorted(missing)}")
    if manifest["schema_version"] != 2:
        raise SelectionError(f"unsupported manifest schema_version: {manifest['schema_version']}")
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != 243 or manifest.get("case_count") != 243:
        raise SelectionError("manifest must contain exactly 243 cases")
    if manifest["total_designs"] != 26023:
        raise SelectionError("manifest total_designs must be 26023")
    case_ids = [case.get("case_id") for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise SelectionError("manifest case_id values are not unique")
    case_indexes = [case.get("case_index") for case in cases]
    if sorted(case_indexes) != list(range(243)):
        raise SelectionError("manifest case_index values must be exactly 0..242")
    for family, (case_count, designs, states) in FROZEN_FAMILY_CASES.items():
        family_cases = [case for case in cases if case.get("family") == family]
        if len(family_cases) != case_count:
            raise SelectionError(f"{family} must contain exactly {case_count} cases")
        if any(case.get("num_designs") != designs for case in family_cases):
            raise SelectionError(f"{family} num_designs must be {designs}")
        if any(case.get("num_states") != states for case in family_cases):
            raise SelectionError(f"{family} num_states must be {states}")
    if {case.get("family") for case in cases} != set(FAMILIES):
        raise SelectionError("manifest contains an unexpected family")
    motif_cases = [case for case in cases if case.get("family") == "motif_switching"]
    if (
        motif_cases[0].get("case_id") != "motif_switching__3ixt__1ycr__oqo"
        or motif_cases[0].get("motifs") != ["3ixt", "1ycr"]
    ):
        raise SelectionError("motif_switching motifs must be exactly 3ixt,1ycr")
    ligand_modification = [case for case in cases if case.get("family") == "ligand_modification"]
    if ligand_modification[0].get("case_id") != "ligand_modification__hem__oxy":
        raise SelectionError("unexpected ligand_modification case identity")
    ligand_discrimination = [
        case for case in cases if case.get("family") == "ligand_discrimination"
    ]
    if ligand_discrimination[0].get("case_id") != "ligand_discrimination__apo__oqo__ca":
        raise SelectionError("unexpected ligand_discrimination case identity")

    motif_keys = manifest.get("motifs")
    ligand_entries = manifest.get("ligands")
    if not isinstance(motif_keys, list) or len(motif_keys) != 24:
        raise SelectionError("manifest motifs must contain exactly 24 identities")
    if not isinstance(ligand_entries, list) or len(ligand_entries) != 5:
        raise SelectionError("manifest ligands must contain exactly five identities")
    ligand_keys = [entry.get("key") for entry in ligand_entries]
    ligand_state_entries = [entry.get("state_entry") for entry in ligand_entries]
    if len(set(ligand_keys)) != 5 or any(not value for value in ligand_state_entries):
        raise SelectionError("manifest ligand identities/state entries are invalid")
    expected_pairs = {(motif, ligand) for motif in motif_keys for ligand in ligand_keys}
    for family in ("positive_allostery", "negative_allostery"):
        family_cases = [case for case in cases if case.get("family") == family]
        observed_pairs = set()
        for case in family_cases:
            if case.get("motifs") != [case.get("motif_key")]:
                raise SelectionError(f"allostery motif identity mismatch for {case.get('case_id')}")
            observed_pairs.add((case.get("motif_key"), case.get("ligand_key")))
        if observed_pairs != expected_pairs:
            raise SelectionError(f"{family} does not contain the exact motif-ligand Cartesian set")
    for case in cases:
        family_root = source_root / case["family"]
        try:
            relative = Path(case["design_root"]).resolve().relative_to(family_root.resolve())
        except ValueError as error:
            raise SelectionError(f"design_root escapes family root for {case['case_id']}") from error
        if case["family"] in ("positive_allostery", "negative_allostery"):
            expected_relative = Path(case["ligand_key"]) / case["motif_key"]
        elif case["family"] == "motif_switching":
            expected_relative = Path("3ixt_1ycr")
        else:
            expected_relative = Path(".")
        if relative != expected_relative:
            raise SelectionError(f"unexpected design_root mapping for {case['case_id']}")


def _required_design_files(case: Mapping[str, Any], design_dir: Path) -> set[str]:
    required = set()
    for state_index in range(case["num_states"]):
        required.add(f"state{state_index}.pkl")
        for sample_index in range(SAMPLES_PER_STATE):
            required.add(f"state{state_index}_sample{sample_index}.pdb")
            required.add(f"state{state_index}_sample{sample_index}.cif")
    required.update(f"{motif}_spec.pkl" for motif in case["motifs"])
    return required


def _preflight_case_files(case: Mapping[str, Any]) -> None:
    root = Path(case["design_root"])
    if not root.is_dir():
        raise SelectionError(f"missing design_root for {case['case_id']}: {root}")
    pattern = re.compile(r"design(\d+)")
    observed = {
        int(match.group(1))
        for entry in root.iterdir()
        if entry.is_dir() and (match := pattern.fullmatch(entry.name))
    }
    expected = set(range(case["num_designs"]))
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise SelectionError(
            f"case={case['case_id']} design directories are not exact; missing={missing} extra={extra}"
        )
    artifact_pattern = re.compile(
        r"(?:state\d+\.pkl|state\d+_sample\d+\.(?:pdb|cif)|.+_spec\.pkl)"
    )
    for design_index in range(case["num_designs"]):
        design_dir = root / f"design{design_index}"
        required = _required_design_files(case, design_dir)
        observed_files = {
            entry.name for entry in design_dir.iterdir() if entry.is_file() and artifact_pattern.fullmatch(entry.name)
        }
        missing = sorted(required - observed_files)
        extra = sorted(observed_files - required)
        if missing or extra:
            raise SelectionError(
                f"case={case['case_id']} design={design_index} required files are not exact; "
                f"missing={missing} extra={extra}"
            )
        empty = sorted(name for name in required if (design_dir / name).stat().st_size == 0)
        if empty:
            raise SelectionError(f"case={case['case_id']} design={design_index} empty files={empty}")


def preflight(
    manifest: Mapping[str, Any], manifest_path: Path, families: Sequence[str]
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, str], Path, str]:
    source_root = Path(manifest["output_root"]).resolve()
    if not source_root.is_dir():
        raise SelectionError(f"manifest output_root is not a directory: {source_root}")
    _validate_manifest_contract(manifest, source_root)
    configs: dict[str, dict[str, Any]] = {}
    for case in sorted(manifest["cases"], key=lambda row: row["case_index"]):
        configs[case["case_id"]] = _load_and_validate_config(case)
    ligand_states = {entry["key"]: entry["state_entry"] for entry in manifest["ligands"]}
    for case in manifest["cases"]:
        config_states = configs[case["case_id"]]["states"]
        if case["family"] in ("positive_allostery", "negative_allostery"):
            expected_states = [[], [ligand_states[case["ligand_key"]]]]
        elif case["family"] == "motif_switching":
            expected_states = [[], ["ccd:OQO"]]
        elif case["family"] == "ligand_modification":
            expected_states = [["ccd:HEM"], ["ccd:HEM", "ccd:OXY"]]
        else:
            expected_states = [[], ["ccd:OQO"], ["ccd:CA"]]
        if config_states != expected_states:
            raise SelectionError(f"state identity mismatch for {case['case_id']}")

    scientific = manifest["scientific_sources"]
    paper_path = Path(scientific["paper_path"]).resolve()
    if not paper_path.is_file():
        raise SelectionError(f"missing paper: {paper_path}")
    paper_hash = sha256_file(paper_path)
    if paper_hash != scientific["paper_sha256"]:
        raise SelectionError("paper SHA256 does not match manifest")
    motif_root = Path(scientific["motif_specs_root"]).resolve()
    if not motif_root.is_dir():
        raise SelectionError(f"missing motif reference root: {motif_root}")
    motifs = sorted(
        {motif for case in manifest["cases"] if case["family"] in families for motif in case["motifs"]}
    )
    motif_paths = {}
    for motif in motifs:
        path = motif_root / f"{motif}.pdb"
        if not path.is_file() or path.stat().st_size == 0:
            raise SelectionError(f"missing/empty motif reference: {path}")
        motif_paths[motif] = str(path)

    for case in sorted(manifest["cases"], key=lambda row: row["case_index"]):
        if case["family"] in families:
            _preflight_case_files(case)
    return source_root, configs, motif_paths, paper_path, paper_hash


def _fieldnames(rows: Sequence[Mapping[str, Any]], fallback: Sequence[str] = ()) -> list[str]:
    fields: list[str] = []
    for key in fallback:
        if key not in fields:
            fields.append(key)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _selection_definitions(family: str) -> dict[str, str]:
    return {
        "positive_allostery": {
            "appendix_b1": "A mean > 1; B mean <= 1; both sample std <= 0.5; abs(A-B) >= 0.5",
            "main_text_4_1": "A mean > 1; B mean <= 1; both sample std <= 0.5; abs(A-B) > 1",
        },
        "negative_allostery": {
            "appendix_b1": "A mean <= 1; B mean > 1; both sample std <= 0.5; abs(A-B) >= 0.5",
            "main_text_4_1": "A mean <= 1; B mean > 1; both sample std <= 0.5; abs(A-B) > 1",
        },
        "motif_switching": {
            "appendix_b2": "A: 3ixt mean <= 1 and 1ycr mean > 1; B inverse; all sample std <= 0.5"
        },
        "ligand_modification": {
            "appendix_b3_literal_or": "cross AB mean > 2 and (intra A mean < 1 or intra B mean < 1)",
            "appendix_b3_both_states": "cross AB mean > 2 and intra A mean < 1 and intra B mean < 1",
        },
        "ligand_discrimination": {
            "appendix_b5": "all A/B/C intra means < 1 and all AB/AC/BC cross means > 1"
        },
    }[family]


def _case_output_relative(case: Mapping[str, Any], source_root: Path) -> Path:
    family_root = source_root / case["family"]
    relative = Path(case["design_root"]).resolve().relative_to(family_root.resolve())
    return Path(f"{case['family']}_selected") / relative


def _runtime_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_case_outputs(
    stage_root: Path,
    output_root: Path,
    source_root: Path,
    case: Mapping[str, Any],
    config: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    manifest_hash: str,
    paper_path: Path,
    paper_hash: str,
    evaluator_path: Path,
    evaluator_hash: str,
    runtime_commit: str,
    limited: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    relative = _case_output_relative(case, source_root)
    case_stage = stage_root / relative
    case_stage.mkdir(parents=True, exist_ok=True)
    metrics = [dict(result["metrics"]) for result in results]
    samples = [dict(row) for result in results for row in result["samples"]]
    pairs = [dict(row) for result in results for row in result["pairs"]]

    selected: list[dict[str, Any]] = []
    for metric in metrics:
        for selection_set, score in selection_memberships(metric):
            selected.append({"selection_set": selection_set, "selection_score": score, **metric})
    selected.sort(key=lambda row: (row["selection_set"], -row["selection_score"], row["design_index"]))
    rank_counts: dict[str, int] = {}
    for row in selected:
        selection_set = row["selection_set"]
        rank_counts[selection_set] = rank_counts.get(selection_set, 0) + 1
        row["selection_rank"] = rank_counts[selection_set]

    _write_csv(case_stage / "metrics.csv", metrics, _fieldnames(metrics))
    selected_fallback = ["selection_set", "selection_score", "selection_rank"] + _fieldnames(metrics)
    _write_csv(case_stage / "selected.csv", selected, _fieldnames(selected, selected_fallback))
    _write_csv(case_stage / "samples.csv", samples, _fieldnames(samples))
    output_files = ["metrics.csv", "selected.csv", "samples.csv", "summary.json"]
    if case["family"] in ("ligand_modification", "ligand_discrimination"):
        _write_csv(case_stage / "pairs.csv", pairs, _fieldnames(pairs))
        output_files.insert(3, "pairs.csv")

    selection_counts = {
        selection_set: sum(row["selection_set"] == selection_set for row in selected)
        for selection_set in _selection_definitions(case["family"])
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "family": case["family"],
        "case_id": case["case_id"],
        "case_index": case["case_index"],
        "config_path": str(Path(case["config_path"]).resolve()),
        "config_sha256": case["config_sha256"],
        "state_identities": config["states"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_hash,
        "paper_path": str(paper_path),
        "paper_sha256": paper_hash,
        "source_root": str(source_root),
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": evaluator_hash,
        "runtime_git_commit": runtime_commit,
        "samples_per_state": SAMPLES_PER_STATE,
        "alignment": "centered equal-weight reflection-corrected Kabsch RMSD via utils.geometry.compute_rmsd",
        "atom_selection": (
            "motif-mask-selected protein-residue CA atoms"
            if case["motifs"]
            else "first 50 parsed protein-residue CA atoms"
        ),
        "standard_deviation_ddof": 1 if case["motifs"] else 0,
        "selection_set_definitions": _selection_definitions(case["family"]),
        "selection_score_definition": (
            "weakest signed component margin; ligand modification literal OR uses "
            "min(cross margin, max(intra A margin, intra B margin))"
        ),
        "selection_counts": selection_counts,
        "expected_design_count": case["num_designs"],
        "evaluated_design_count": len(metrics),
        "status": "incomplete_limit" if limited else "complete",
        "output_path": str((output_root / relative).resolve()),
        "output_filenames": output_files,
    }
    (case_stage / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    summary_rows = [
        {
            "family": case["family"],
            "case_id": case["case_id"],
            "case_index": case["case_index"],
            "selection_set": selection_set,
            "expected_design_count": case["num_designs"],
            "evaluated_design_count": len(metrics),
            "selected_design_count": selection_counts[selection_set],
            "status": summary["status"],
            "case_output_path": summary["output_path"],
        }
        for selection_set in _selection_definitions(case["family"])
    ]
    core = [
        {
            key: row[key]
            for key in (
                "selection_set",
                "family",
                "case_id",
                "case_index",
                "design_index",
                "design_name",
                "config_path",
                "config_sha256",
                "design_path",
                "design_path_relative_to_source_root",
                "selection_rank",
                "selection_score",
            )
        }
        for row in selected
    ]
    return summary_rows, core, selected


def _promote(stage_root: Path, output_root: Path, families: Sequence[str], overwrite: bool) -> None:
    names = [f"{family}_selected" for family in families] + [
        "paper_selection_summary.csv",
        "paper_selected.csv",
    ]
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    backup_root = output_root / f".paper_selection_backup-{uuid.uuid4().hex}"
    try:
        if overwrite:
            backup_root.mkdir()
        for name in names:
            source, target = stage_root / name, output_root / name
            if target.exists():
                if not overwrite:
                    raise SelectionError(f"target exists; use --overwrite: {target}")
                backup = backup_root / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, backup)
                backups.append((backup, target))
            os.replace(source, target)
            promoted.append(target)
        if backup_root.exists():
            shutil.rmtree(backup_root)
    except Exception:
        for target in reversed(promoted):
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        for backup, target in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        if backup_root.exists():
            shutil.rmtree(backup_root)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-per-case", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.workers < 1:
        raise SelectionError("--workers must be at least 1")
    if args.limit_per_case is not None and args.limit_per_case < 1:
        raise SelectionError("--limit-per-case must be at least 1")
    families = [family for family in FAMILIES if family in set(args.families)]
    manifest_path = args.manifest.resolve()
    manifest, manifest_hash = _read_manifest(manifest_path)
    source_root = Path(manifest["output_root"]).resolve()
    output_root = (args.output_root or source_root).resolve()
    if args.limit_per_case is not None and output_root == source_root:
        raise SelectionError("--limit-per-case requires --output-root different from source root")
    output_root.mkdir(parents=True, exist_ok=True)

    targets = [output_root / f"{family}_selected" for family in families]
    targets.extend((output_root / "paper_selection_summary.csv", output_root / "paper_selected.csv"))
    existing = [target for target in targets if target.exists()]
    if existing and not args.overwrite:
        raise SelectionError(f"target exists; use --overwrite: {existing[0]}")

    source_root, configs, motif_paths, paper_path, paper_hash = preflight(
        manifest, manifest_path, families
    )
    cases = [
        case
        for case in sorted(manifest["cases"], key=lambda row: row["case_index"])
        if case["family"] in families
    ]
    tasks = []
    for case in cases:
        evaluated_count = (
            min(case["num_designs"], args.limit_per_case)
            if args.limit_per_case is not None
            else case["num_designs"]
        )
        tasks.extend(
            DesignTask(case, configs[case["case_id"]], design_index, str(source_root), motif_paths)
            for design_index in range(evaluated_count)
        )

    if args.workers == 1:
        _worker_init()
        evaluated = list(map(evaluate_design, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
            evaluated = list(pool.map(evaluate_design, tasks, chunksize=8))

    repo_root = Path(__file__).resolve().parents[1]
    evaluator_path = Path(__file__).resolve()
    evaluator_hash = sha256_file(evaluator_path)
    runtime_commit = _runtime_commit(repo_root)
    stage_root = output_root / f".paper_selection_stage-{uuid.uuid4().hex}"
    stage_root.mkdir()
    try:
        summary_rows: list[dict[str, Any]] = []
        global_selected: list[dict[str, Any]] = []
        cursor = 0
        for case in cases:
            count = (
                min(case["num_designs"], args.limit_per_case)
                if args.limit_per_case is not None
                else case["num_designs"]
            )
            case_results = evaluated[cursor : cursor + count]
            cursor += count
            case_summary_rows, case_selected, _ = _write_case_outputs(
                stage_root,
                output_root,
                source_root,
                case,
                configs[case["case_id"]],
                case_results,
                manifest_path,
                manifest_hash,
                paper_path,
                paper_hash,
                evaluator_path,
                evaluator_hash,
                runtime_commit,
                args.limit_per_case is not None,
            )
            summary_rows.extend(case_summary_rows)
            global_selected.extend(case_selected)
        summary_rows.sort(key=lambda row: (row["case_index"], row["selection_set"]))
        global_selected.sort(
            key=lambda row: (row["case_index"], row["selection_set"], row["selection_rank"])
        )
        _write_csv(
            stage_root / "paper_selection_summary.csv",
            summary_rows,
            _fieldnames(summary_rows),
        )
        global_fallback = [
            "selection_set",
            "family",
            "case_id",
            "case_index",
            "design_index",
            "design_name",
            "config_path",
            "config_sha256",
            "design_path",
            "design_path_relative_to_source_root",
            "selection_rank",
            "selection_score",
        ]
        _write_csv(
            stage_root / "paper_selected.csv",
            global_selected,
            _fieldnames(global_selected, global_fallback),
        )
        _promote(stage_root, output_root, families, args.overwrite)
        stage_root.rmdir()
    except Exception:
        if stage_root.exists():
            shutil.rmtree(stage_root)
        raise
    print(
        f"evaluated {len(evaluated)} designs across {len(cases)} cases; "
        f"wrote {output_root}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SelectionError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
