"""
Calculate sequence recovery from native and designed CIF files.

Usage:
    python -m allatom_design.eval.metrics.sequence_recovery \
        --native_cif_dir /path/to/native_cifs \
        --designed_sample_dir /path/to/samples \
        --sampling_inputs_csv /path/to/sampling_inputs.csv \
        --output_csv /path/to/output.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from omegaconf import OmegaConf
from tqdm import tqdm

from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise

from allatom_design.eval.utils.pocket_constraints import (
    annotate_ligand_pocket,
    resolve_pocket_annotation_method,
)
from allatom_design.utils.atom_array_utils import get_valid_standard_aa_residue_mask
from allatom_design.utils.sample_io_utils import load_example_with_parse

# CIF parse configs from seq_des_multi.yaml
NATIVE_CIF_PARSE_CFG = {
    "add_missing_atoms": True,
    "remove_waters": True,
    "remove_ccds": [],
    "fix_ligands_at_symmetry_centers": True,
    "fix_arginines": True,
    "convert_mse_to_met": True,
    "hydrogen_policy": "remove",
    "extra_fields": "all",
}

DESIGNED_CIF_PARSE_CFG = {
    "add_missing_atoms": False,
    "remove_waters": False,
    "remove_ccds": [],
    "fix_ligands_at_symmetry_centers": False,
    "fix_arginines": False,
    "convert_mse_to_met": True,
    "hydrogen_policy": "remove",
    "extra_fields": None,
}


@dataclass(frozen=True)
class SequenceRecoveryMetricConfig:
    enabled: bool = True
    pocket_distances_for_seq_recovery: Sequence[float] | None = None
    pocket_distance_bins: Sequence[tuple[float, float]] | None = None
    n_min_ligand_atoms: int = 1
    pocket_annotation_method: str = "all_atom"


def resolve_sequence_recovery_pocket_annotation_method(
    *,
    pocket_cfg: dict | None = None,
    input_sample_is_designed: bool = False,
) -> str:
    raw_method = None
    if pocket_cfg is not None:
        raw_method = pocket_cfg.get("seq_recovery_pocket_annotation_method", None)
    if raw_method is None or str(raw_method).lower() == "auto":
        return "calpha" if input_sample_is_designed else "all_atom"
    return resolve_pocket_annotation_method(str(raw_method))


def build_sequence_recovery_metric_config(
    *,
    pocket_cfg: dict | None,
    input_sample_is_designed: bool,
    pocket_distance_bins: Sequence[tuple[float, float]] | None = None,
    enabled: bool = True,
) -> SequenceRecoveryMetricConfig:
    pocket_distances = None
    n_min_ligand_atoms = 1
    if pocket_cfg is not None:
        pocket_distances = pocket_cfg.get("pocket_distances_for_seq_recovery", None)
        n_min_ligand_atoms = pocket_cfg.get("n_min_ligand_atoms_for_seq_recovery", 1)
    return SequenceRecoveryMetricConfig(
        enabled=enabled,
        pocket_distances_for_seq_recovery=pocket_distances,
        pocket_distance_bins=pocket_distance_bins,
        n_min_ligand_atoms=int(n_min_ligand_atoms),
        pocket_annotation_method=resolve_sequence_recovery_pocket_annotation_method(
            pocket_cfg=pocket_cfg,
            input_sample_is_designed=input_sample_is_designed,
        ),
    )


def make_sequence_recovery_metric_config(
    *,
    input_sample_is_designed: bool,
    pocket_distances_for_seq_recovery: Sequence[float] | None = None,
    pocket_distance_bins: Sequence[tuple[float, float]] | None = None,
    n_min_ligand_atoms: int = 1,
    pocket_annotation_method: str | None = None,
    enabled: bool = True,
) -> SequenceRecoveryMetricConfig:
    if pocket_annotation_method is None:
        pocket_annotation_method = "calpha" if input_sample_is_designed else "all_atom"
    return SequenceRecoveryMetricConfig(
        enabled=enabled,
        pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
        pocket_distance_bins=pocket_distance_bins,
        n_min_ligand_atoms=n_min_ligand_atoms,
        pocket_annotation_method=resolve_pocket_annotation_method(pocket_annotation_method),
    )


def _annotate_sequence_recovery_pocket(
    atom_array: AtomArray,
    *,
    pocket_distance: float,
    n_min_ligand_atoms: int,
    annotation_name: str,
    pocket_annotation_method: str,
    receptor_pn_unit_iids: list[str] | None = None,
    ligand_pn_unit_iids: list[str] | None = None,
) -> AtomArray:
    return annotate_ligand_pocket(
        atom_array=atom_array,
        pocket_distance=pocket_distance,
        n_min_ligand_atoms=n_min_ligand_atoms,
        annotation_name=annotation_name,
        pocket_annotation_method=pocket_annotation_method,
        receptor_pn_unit_iids=receptor_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
    )


def _shared_residue_key_annotation(
    input_atom_array: AtomArray,
    designed_atom_array: AtomArray,
) -> str:
    input_annotations = set(input_atom_array.get_annotation_categories())
    designed_annotations = set(designed_atom_array.get_annotation_categories())
    if "pn_unit_iid" in input_annotations and "pn_unit_iid" in designed_annotations:
        input_values = np.asarray(input_atom_array.get_annotation("pn_unit_iid")).astype(str)
        designed_values = np.asarray(designed_atom_array.get_annotation("pn_unit_iid")).astype(str)
        if np.any(input_values != "") and np.any(designed_values != ""):
            return "pn_unit_iid"
    return "chain_id"


def _ins_codes(atom_array: AtomArray) -> np.ndarray:
    if "ins_code" in atom_array.get_annotation_categories():
        return np.asarray(atom_array.get_annotation("ins_code")).astype(str)
    return np.full(len(atom_array), "", dtype=object)


def _ca_residue_lookup(
    atom_array: AtomArray,
    ca_mask: np.ndarray,
    *,
    key_annotation: str,
) -> dict[tuple[str, int, str], str]:
    lookup: dict[tuple[str, int, str], str] = {}
    chain_values = np.asarray(atom_array.get_annotation(key_annotation)).astype(str)
    ins_codes = _ins_codes(atom_array)
    ca_indices = np.where(ca_mask)[0]
    for idx in ca_indices:
        key = (
            str(chain_values[idx]),
            int(atom_array.res_id[idx]),
            str(ins_codes[idx]),
        )
        if key in lookup:
            raise ValueError(
                "Duplicate CA residue key in sequence recovery atom array: "
                f"{key_annotation}={key[0]}, res_id={key[1]}, ins_code={key[2]!r}"
            )
        lookup[key] = str(atom_array.res_name[idx])
    return lookup


def _ca_residue_keys(
    atom_array: AtomArray,
    ca_mask: np.ndarray,
    *,
    key_annotation: str,
) -> list[tuple[str, int, str]]:
    chain_values = np.asarray(atom_array.get_annotation(key_annotation)).astype(str)
    ins_codes = _ins_codes(atom_array)
    return [
        (
            str(chain_values[idx]),
            int(atom_array.res_id[idx]),
            str(ins_codes[idx]),
        )
        for idx in np.where(ca_mask)[0]
    ]


def _recovery_ratio_for_keys(
    native_lookup: dict[tuple[str, int, str], str],
    designed_lookup: dict[tuple[str, int, str], str],
    native_keys: Sequence[tuple[str, int, str]],
) -> tuple[float, int]:
    if len(native_keys) == 0:
        return float("nan"), 0
    matched = [
        designed_lookup.get(key) == native_lookup[key]
        for key in native_keys
    ]
    n_missing = sum(key not in designed_lookup for key in native_keys)
    return float(np.mean(matched)), int(n_missing)


def calculate_sequence_recovery(input_atom_array: AtomArray, designed_atom_array: AtomArray,
                                pocket_distances_for_seq_recovery: Sequence[float] | None = None,
                                pocket_distance_bins: Sequence[tuple[float, float]] | None = None,
                                n_min_ligand_atoms: int = 1,
                                pocket_annotation_method: str = "all_atom") -> dict[str, float]:
    """
    Calculate sequence recovery and pocket sequence recovery between input and designed atom arrays.
    """
    if pocket_distances_for_seq_recovery is None:
        pocket_distances_for_seq_recovery = (4.0, 5.0, 6.0)
    pocket_annotation_method = resolve_pocket_annotation_method(pocket_annotation_method)
    seq_recovery_metrics = {}

    input_valid_residue_mask = get_valid_standard_aa_residue_mask(input_atom_array)

    input_seq_mask = input_valid_residue_mask & (input_atom_array.atom_name == "CA")
    designed_valid_residue_mask = get_valid_standard_aa_residue_mask(designed_atom_array)
    designed_seq_mask = designed_valid_residue_mask & (designed_atom_array.atom_name == "CA")

    key_annotation = _shared_residue_key_annotation(input_atom_array, designed_atom_array)
    input_lookup = _ca_residue_lookup(
        input_atom_array,
        input_seq_mask,
        key_annotation=key_annotation,
    )
    designed_lookup = _ca_residue_lookup(
        designed_atom_array,
        designed_seq_mask,
        key_annotation=key_annotation,
    )
    input_keys = list(input_lookup.keys())
    seq_recovery_ratio, n_missing_designed = _recovery_ratio_for_keys(
        input_lookup,
        designed_lookup,
        input_keys,
    )
    seq_recovery_metrics["seq_recovery_ratio"] = seq_recovery_ratio
    seq_recovery_metrics["seq_recovery_n_residues"] = int(len(input_keys))
    seq_recovery_metrics["seq_recovery_n_missing_designed"] = n_missing_designed
    seq_recovery_metrics["seq_recovery_n_extra_designed"] = int(
        len(set(designed_lookup) - set(input_lookup))
    )

    edge_to_residue_mask: dict[float, np.ndarray] = {}

    for pocket_distance in pocket_distances_for_seq_recovery:
        input_atom_array = _annotate_sequence_recovery_pocket(
            input_atom_array,
            pocket_distance=pocket_distance,
            n_min_ligand_atoms=n_min_ligand_atoms,
            annotation_name=f"is_ligand_pocket_{pocket_distance}",
            pocket_annotation_method=pocket_annotation_method,
        )
        input_pocket_residue_mask = apply_and_spread_residue_wise(
            input_atom_array,
            input_atom_array.get_annotation(f"is_ligand_pocket_{pocket_distance}"),
            function=np.any,
        )
        edge_to_residue_mask[float(pocket_distance)] = input_pocket_residue_mask
        input_pocket_seq_mask = input_seq_mask & input_pocket_residue_mask
        input_pocket_keys = _ca_residue_keys(
            input_atom_array,
            input_pocket_seq_mask,
            key_annotation=key_annotation,
        )
        pocket_ratio, pocket_missing = _recovery_ratio_for_keys(
            input_lookup,
            designed_lookup,
            input_pocket_keys,
        )

        seq_recovery_metrics[f"pocket_recovery_ratio_{pocket_distance}"] = pocket_ratio
        seq_recovery_metrics[f"pocket_n_residues_{pocket_distance}"] = int(len(input_pocket_keys))
        seq_recovery_metrics[f"pocket_n_missing_designed_{pocket_distance}"] = pocket_missing

    if pocket_distance_bins:
        for lo, hi in pocket_distance_bins:
            lo_f, hi_f = float(lo), float(hi)
            for d in (lo_f, hi_f):
                if d == 0.0:
                    continue
                if d not in edge_to_residue_mask:
                    ann = f"is_ligand_pocket_{d}"
                    input_atom_array = _annotate_sequence_recovery_pocket(
                        input_atom_array,
                        pocket_distance=d,
                        n_min_ligand_atoms=n_min_ligand_atoms,
                        annotation_name=ann,
                        pocket_annotation_method=pocket_annotation_method,
                    )
                    edge_to_residue_mask[d] = apply_and_spread_residue_wise(
                        input_atom_array,
                        input_atom_array.get_annotation(ann),
                        function=np.any,
                    )

            if lo_f == 0.0:
                bin_residue_mask = edge_to_residue_mask[hi_f]
            else:
                bin_residue_mask = edge_to_residue_mask[hi_f] & ~edge_to_residue_mask[lo_f]

            input_bin_seq_mask = input_seq_mask & bin_residue_mask
            input_bin_keys = _ca_residue_keys(
                input_atom_array,
                input_bin_seq_mask,
                key_annotation=key_annotation,
            )

            key = f"pocket_recovery_bin_{lo_f}_to_{hi_f}"
            n_key = f"pocket_n_residues_bin_{lo_f}_to_{hi_f}"
            missing_key = f"pocket_n_missing_designed_bin_{lo_f}_to_{hi_f}"
            bin_ratio, bin_missing = _recovery_ratio_for_keys(
                input_lookup,
                designed_lookup,
                input_bin_keys,
            )
            seq_recovery_metrics[key] = bin_ratio
            seq_recovery_metrics[n_key] = int(len(input_bin_keys))
            seq_recovery_metrics[missing_key] = bin_missing

    return seq_recovery_metrics


def _compute_recovery(native_aa, designed_aa, pocket_distances):
    """Compute sequence recovery with the runtime chain-aware metric contract."""
    return calculate_sequence_recovery(
        native_aa,
        designed_aa,
        pocket_distances_for_seq_recovery=pocket_distances,
        n_min_ligand_atoms=5,
        pocket_annotation_method="all_atom",
    )


def calculate_sequence_recovery_from_folders(
    native_cif_dir: str | Path,
    designed_sample_dir: str | Path,
    sampling_inputs_csv: str | Path,
    output_csv: str | Path | None = None,
    pocket_distances: Sequence[float] | None = None,
    native_cif_parse_cfg: dict | None = None,
    designed_cif_parse_cfg: dict | None = None,
) -> pd.DataFrame:
    """Calculate sequence recovery for designed samples against native reference structures."""
    if pocket_distances is None:
        pocket_distances = (4.0, 5.0, 6.0)
    if native_cif_parse_cfg is None:
        native_cif_parse_cfg = NATIVE_CIF_PARSE_CFG
    if designed_cif_parse_cfg is None:
        designed_cif_parse_cfg = DESIGNED_CIF_PARSE_CFG

    native_cif_dir = Path(native_cif_dir)
    designed_sample_dir = Path(designed_sample_dir)
    pd.read_csv(sampling_inputs_csv)  # validate CSV exists

    native_cfg = OmegaConf.create(native_cif_parse_cfg)
    designed_cfg = OmegaConf.create(designed_cif_parse_cfg)

    # Group designed samples by pdb_id
    pdb_id_to_samples: dict[str, list[Path]] = defaultdict(list)
    for cif_path in sorted(designed_sample_dir.glob("*.cif")):
        pdb_id_to_samples[cif_path.stem.split("_")[0]].append(cif_path)

    results = []
    native_cache = {}

    for pdb_id, sample_paths in tqdm(pdb_id_to_samples.items(), desc="Calculating sequence recovery"):
        native_cif_path = native_cif_dir / f"{pdb_id}.cif"
        if not native_cif_path.exists():
            print(f"Warning: native CIF not found for {pdb_id}, skipping")
            continue

        if pdb_id not in native_cache:
            try:
                native_cache[pdb_id] = load_example_with_parse(str(native_cif_path), native_cfg)["atom_array"]
            except Exception as e:
                print(f"Warning: failed to parse native CIF for {pdb_id}: {e}")
                continue

        native_aa = native_cache[pdb_id]

        for sample_path in sample_paths:
            try:
                designed_aa = load_example_with_parse(str(sample_path), designed_cfg)["atom_array"]
            except Exception as e:
                print(f"Warning: failed to parse {sample_path.name}: {e}")
                continue

            metrics = _compute_recovery(native_aa, designed_aa, pocket_distances)
            results.append({"pdb_id": pdb_id, "designed_sample_id": sample_path.stem, **metrics})

    results_df = pd.DataFrame(results)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"Saved results to {output_csv}")

    if len(results_df) > 0:
        print(f"\n--- Summary ({len(results_df)} samples, {results_df['pdb_id'].nunique()} pdb_ids) ---")
        for col in results_df.select_dtypes(include="number").columns:
            print(f"  {col}: mean={results_df[col].mean():.4f}, std={results_df[col].std():.4f}")

    return results_df


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calculate sequence recovery from CIF folders")
    parser.add_argument("--native_cif_dir", type=str, required=True)
    parser.add_argument("--designed_sample_dir", type=str, required=True)
    parser.add_argument("--sampling_inputs_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--pocket_distances", nargs="+", type=float, default=[4.0, 5.0, 6.0])
    args = parser.parse_args(argv)

    calculate_sequence_recovery_from_folders(
        native_cif_dir=args.native_cif_dir,
        designed_sample_dir=args.designed_sample_dir,
        sampling_inputs_csv=args.sampling_inputs_csv,
        output_csv=args.output_csv,
        pocket_distances=args.pocket_distances,
    )


if __name__ == "__main__":
    main()
