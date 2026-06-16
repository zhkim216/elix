from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
from omegaconf import DictConfig

from allatom_design.data.transform.ligand_conformers import (
    compute_ligand_protein_clash_metrics,
    generate_ligand_conformer_decoys,
    select_target_ligand,
)
from allatom_design.eval.utils.data_utils import (
    _matched_sampling_input_row,
    parse_query_pn_unit_iids,
)
from allatom_design.eval.utils.ensemble_conditioning import (
    normalize_ensemble_conditioning_cfg,
)
from allatom_design.utils.sample_io_utils import load_example_with_parse, save_cif_file


@dataclass(frozen=True)
class LigandConformerStagingResult:
    root_dir: Path
    pdb_paths: list[str]
    member_groups: list[list[str]]
    sampling_inputs_df: pd.DataFrame | None
    member_to_group_id: dict[str, int]
    member_to_coefficient: dict[str, float]
    member_to_target_id: dict[str, str]
    aggregation_scheme: str
    manifest_path: Path

    def expand_pos_constraints(
        self,
        pos_constraint_df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        return expand_pos_constraint_df_for_ligand_conformer_members(
            pos_constraint_df,
            self,
        )

    def iter_member_batches(self, *, max_members: int) -> Iterator[list[str]]:
        return iter_ligand_conformer_member_batches(
            self.member_groups,
            max_members=max_members,
        )

    def target_count(self, batch_pdb_paths: list[str] | None = None) -> int:
        if batch_pdb_paths is None:
            return len(self.member_groups)
        return len({
            self.member_to_group_id[Path(path).stem]
            for path in batch_pdb_paths
        })

    def annotate_batch(
        self,
        batch: dict[str, Any],
        *,
        batch_pdb_paths: list[str],
        device: str | torch.device | None,
    ) -> dict[str, Any]:
        group_ids = []
        coefficients = []
        missing = []
        for path in batch_pdb_paths:
            member_sample_id = Path(path).stem
            if member_sample_id not in self.member_to_group_id:
                missing.append(member_sample_id)
                continue
            group_ids.append(self.member_to_group_id[member_sample_id])
            coefficients.append(self.member_to_coefficient[member_sample_id])
        if missing:
            raise KeyError(f"Missing ligand conformer staging metadata for: {missing[:5]}")

        annotated = dict(batch)
        annotated["tied_sampling_ids"] = torch.as_tensor(
            group_ids,
            dtype=torch.long,
            device=device,
        )
        annotated["tied_sampling_aggregation_scheme"] = self.aggregation_scheme
        if self.aggregation_scheme == "weighted_mean":
            annotated["tied_sampling_weights"] = torch.as_tensor(
                coefficients,
                dtype=torch.float32,
                device=device,
            )
        return annotated


def stage_ligand_conformer_ensembles(
    *,
    pdb_paths: list[str],
    out_dir: str | Path,
    ensemble_cfg: dict[str, Any] | DictConfig,
    sampling_inputs_df: pd.DataFrame | None,
    cif_parse_cfg: dict[str, Any] | DictConfig | None,
    cif_save_cfg: dict[str, Any] | DictConfig | None,
    csv_suffix: str = "",
) -> LigandConformerStagingResult:
    cfg = normalize_ensemble_conditioning_cfg(ensemble_cfg)
    small_molecule_cfg = cfg["small_molecule"]
    if small_molecule_cfg["mode"] != "ligand_conformer":
        raise ValueError(
            "stage_ligand_conformer_ensembles requires "
            "small_molecule.mode='ligand_conformer'"
        )

    root_dir = Path(out_dir) / "samples_with_ligand_conformers"
    root_dir.mkdir(parents=True, exist_ok=True)

    max_decoys = max(int(cfg["total_members"]) - 1, 0)
    seed = small_molecule_cfg["seed"]
    if seed is None:
        seed = cfg["noise_seed"]
    seed = 0 if seed is None else int(seed)

    manifest_rows: list[dict[str, Any]] = []
    member_groups: list[list[str]] = []
    member_to_group_id: dict[str, int] = {}
    member_to_coefficient: dict[str, float] = {}
    member_to_target_id: dict[str, str] = {}
    sampling_extra_rows: list[dict[str, Any]] = []
    runtime_sampling_inputs_df = (
        sampling_inputs_df.copy()
        if sampling_inputs_df is not None
        else None
    )

    for group_id, pdb_path in enumerate(pdb_paths):
        input_path = Path(pdb_path)
        target_sample_id = input_path.stem
        pdb_id = target_sample_id.split("_")[0]
        sampling_row = _matched_sampling_input_row(
            sampling_inputs_df,
            pdb_id=pdb_id,
            pdb_key=target_sample_id,
        )
        query_pn_unit_iids = _query_pn_unit_iids_from_sampling_row(sampling_row)

        example = load_example_with_parse(str(input_path), cif_parse_cfg)
        atom_array = example["atom_array"]
        target_ligand = select_target_ligand(
            atom_array,
            query_pn_unit_iids=query_pn_unit_iids or None,
        )
        original_clash_metrics = compute_ligand_protein_clash_metrics(
            atom_array,
            ligand_mask=target_ligand.heavy_mask,
            clash_target_atoms=small_molecule_cfg["clash_target_atoms"],
            vdw_overlap_cutoff=small_molecule_cfg["vdw_overlap_cutoff"],
        )

        warnings: list[str] = []
        decoys = []
        generation_metadata: dict[str, Any] = {}
        if max_decoys > 0:
            try:
                generation_result = generate_ligand_conformer_decoys(
                    atom_array,
                    target_ligand=target_ligand,
                    num_candidates=small_molecule_cfg["num_conformer_candidates"],
                    num_representatives=small_molecule_cfg["num_conformer_candidates"],
                    rmsd_cluster_cutoff=small_molecule_cfg["rmsd_cluster_cutoff"],
                    seed=seed + group_id * int(small_molecule_cfg["num_conformer_candidates"]),
                    num_threads=small_molecule_cfg["num_threads"],
                    uff_optimize=small_molecule_cfg["uff_optimize"],
                    clash_target_atoms=small_molecule_cfg["clash_target_atoms"],
                    vdw_overlap_cutoff=small_molecule_cfg["vdw_overlap_cutoff"],
                )
                decoys = generation_result.decoys
                generation_metadata = generation_result.metadata
                if int(generation_metadata.get("rdkit_native_threshold_hit_count", 0)) == 0:
                    warnings.append(
                        "no_ligand_conformer_decoys_within_native_cutoff: "
                        f"cutoff={small_molecule_cfg['rmsd_cluster_cutoff']}"
                    )
            except Exception as exc:
                warnings.append(f"conformer_generation_failed: {type(exc).__name__}: {exc}")

        candidate_decoys = decoys
        if small_molecule_cfg["exclude_clashing_decoys"]:
            candidate_decoys = [
                decoy for decoy in decoys if not bool(decoy.clash_metrics["has_clash"])
            ]
            if len(candidate_decoys) < min(max_decoys, len(decoys)):
                warnings.append(
                    "excluded_clashing_decoys: "
                    f"{len(decoys) - len(candidate_decoys)}"
                )

        selected_decoys = candidate_decoys[:max_decoys]
        if max_decoys > 0 and len(selected_decoys) < max_decoys:
            warnings.append(
                "fewer_ligand_conformer_decoys_than_requested: "
                f"requested={max_decoys}, selected={len(selected_decoys)}"
            )

        coefficients = compute_ligand_conformer_member_coefficients(
            num_decoys=len(selected_decoys),
            scheme=cfg["weights"]["scheme"],
            ref_weight=cfg["weights"]["ref_weight"],
            decoy_total_weight=cfg["weights"]["decoy_total_weight"],
        )

        group_paths: list[str] = []
        original_out_path = root_dir / f"{target_sample_id}.cif"
        save_cif_file(atom_array, original_out_path, cif_save_cfg=cif_save_cfg)
        group_paths.append(str(original_out_path))
        manifest_rows.append(
            _manifest_row(
                target_sample_id=target_sample_id,
                member_sample_id=original_out_path.stem,
                member_path=original_out_path,
                member_role="original",
                member_coefficient=coefficients[0],
                target_ligand=target_ligand,
                clash_metrics=original_clash_metrics,
                warning="; ".join(warnings),
                generation_metadata=generation_metadata,
            )
        )

        for decoy_idx, decoy in enumerate(selected_decoys, start=1):
            decoy_out_path = root_dir / f"{target_sample_id}_ligconf_{decoy_idx}.cif"
            save_cif_file(decoy.atom_array, decoy_out_path, cif_save_cfg=cif_save_cfg)
            group_paths.append(str(decoy_out_path))
            manifest_rows.append(
                _manifest_row(
                    target_sample_id=target_sample_id,
                    member_sample_id=decoy_out_path.stem,
                    member_path=decoy_out_path,
                    member_role="ligand_conformer_decoy",
                    member_coefficient=coefficients[decoy_idx],
                    target_ligand=target_ligand,
                    clash_metrics=decoy.clash_metrics,
                    warning="; ".join(warnings),
                    rank=decoy.rank,
                    cluster_id=decoy.cluster_id,
                    rdkit_conformer_id=decoy.rdkit_conformer_id,
                    candidate_seed=decoy.candidate_seed,
                    rmsd_to_native=decoy.rmsd_to_native,
                    atom_order_rmsd_to_native=decoy.atom_order_rmsd_to_native,
                    generation_metadata=generation_metadata,
                )
            )

        for member_idx, member_path in enumerate(group_paths):
            member_sample_id = Path(member_path).stem
            member_to_group_id[member_sample_id] = group_id
            member_to_coefficient[member_sample_id] = coefficients[member_idx]
            member_to_target_id[member_sample_id] = target_sample_id
            if runtime_sampling_inputs_df is not None and not _sampling_df_has_pdb_key(
                runtime_sampling_inputs_df,
                member_sample_id,
            ):
                sampling_extra_rows.append(
                    _sampling_row_for_member(
                        sampling_inputs_df=runtime_sampling_inputs_df,
                        source_row=sampling_row,
                        member_sample_id=member_sample_id,
                        target_sample_id=target_sample_id,
                    )
                )

        member_groups.append(group_paths)

    if runtime_sampling_inputs_df is not None and sampling_extra_rows:
        runtime_sampling_inputs_df = pd.concat(
            [
                runtime_sampling_inputs_df,
                pd.DataFrame(sampling_extra_rows, columns=runtime_sampling_inputs_df.columns),
            ],
            ignore_index=True,
        )

    manifest_path = root_dir / f"ligand_conformer_manifest{csv_suffix}.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    return LigandConformerStagingResult(
        root_dir=root_dir,
        pdb_paths=[path for group in member_groups for path in group],
        member_groups=member_groups,
        sampling_inputs_df=runtime_sampling_inputs_df,
        member_to_group_id=member_to_group_id,
        member_to_coefficient=member_to_coefficient,
        member_to_target_id=member_to_target_id,
        aggregation_scheme=cfg["weights"]["scheme"],
        manifest_path=manifest_path,
    )


def compute_ligand_conformer_member_coefficients(
    *,
    num_decoys: int,
    scheme: str,
    ref_weight: float,
    decoy_total_weight: float,
) -> list[float]:
    if num_decoys < 0:
        raise ValueError("num_decoys must be non-negative")
    num_members = num_decoys + 1
    if scheme == "mean":
        return [1.0 / num_members] * num_members
    if scheme == "sqrt":
        return [1.0 / (float(num_members) ** 0.5)] * num_members
    if scheme != "weighted_mean":
        raise ValueError(
            "ligand conformer member coefficient scheme must be "
            f"'mean', 'sqrt', or 'weighted_mean'; got {scheme!r}"
        )
    if ref_weight < 0 or decoy_total_weight < 0:
        raise ValueError("member coefficients must be non-negative")
    if num_decoys == 0:
        return [1.0]
    total = float(ref_weight) + float(decoy_total_weight)
    if total <= 0:
        raise ValueError("member coefficients must have positive total")
    return [
        float(ref_weight) / total,
        *([float(decoy_total_weight) / total / num_decoys] * num_decoys),
    ]


def iter_ligand_conformer_member_batches(
    member_groups: list[list[str]],
    *,
    max_members: int,
) -> Iterator[list[str]]:
    if max_members < 1:
        raise ValueError("max_members must be >= 1")

    current: list[str] = []
    for group in member_groups:
        if current and len(current) + len(group) > max_members:
            yield current
            current = []
        if len(group) > max_members:
            yield list(group)
            continue
        current.extend(group)
    if current:
        yield current


def annotate_ligand_conformer_batch(
    batch: dict[str, Any],
    *,
    batch_pdb_paths: list[str],
    staging_result: LigandConformerStagingResult,
    device: str | torch.device | None,
) -> dict[str, Any]:
    return staging_result.annotate_batch(
        batch,
        batch_pdb_paths=batch_pdb_paths,
        device=device,
    )


def expand_pos_constraint_df_for_ligand_conformer_members(
    pos_constraint_df: pd.DataFrame | None,
    staging_result: LigandConformerStagingResult,
) -> pd.DataFrame | None:
    if pos_constraint_df is None:
        return None
    if "pdb_key" not in pos_constraint_df.columns:
        return pos_constraint_df

    existing_keys = set(pos_constraint_df["pdb_key"].astype(str))
    extra_rows = []
    for member_sample_id, target_sample_id in staging_result.member_to_target_id.items():
        if member_sample_id in existing_keys:
            continue
        source_rows = pos_constraint_df[
            pos_constraint_df["pdb_key"].astype(str) == target_sample_id
        ]
        if source_rows.empty:
            continue
        row = source_rows.iloc[0].copy()
        row["pdb_key"] = member_sample_id
        extra_rows.append(row.to_dict())

    if not extra_rows:
        return pos_constraint_df
    return pd.concat(
        [pos_constraint_df, pd.DataFrame(extra_rows, columns=pos_constraint_df.columns)],
        ignore_index=True,
    )


def ligand_conformer_target_count(
    batch_pdb_paths: list[str],
    staging_result: LigandConformerStagingResult,
) -> int:
    return staging_result.target_count(batch_pdb_paths)


def _query_pn_unit_iids_from_sampling_row(row: pd.Series | None) -> list[str]:
    if row is None:
        return []
    for column in ("query_pn_unit_iids", "query_pn_unit_iids_json"):
        if column not in row.index:
            continue
        parsed = parse_query_pn_unit_iids(row[column])
        if parsed:
            return parsed
    return []


def _sampling_df_has_pdb_key(sampling_inputs_df: pd.DataFrame, pdb_key: str) -> bool:
    if "pdb_key" not in sampling_inputs_df.columns:
        return False
    return bool(
        (
            sampling_inputs_df["pdb_key"].astype(str).str.lower()
            == str(pdb_key).lower()
        ).any()
    )


def _sampling_row_for_member(
    *,
    sampling_inputs_df: pd.DataFrame,
    source_row: pd.Series | None,
    member_sample_id: str,
    target_sample_id: str,
) -> dict[str, Any]:
    if source_row is None:
        row = {column: "" for column in sampling_inputs_df.columns}
    else:
        row = source_row.to_dict()
    if "pdb_key" in row:
        row["pdb_key"] = member_sample_id
    if "pdb_id" in row and (row["pdb_id"] is None or str(row["pdb_id"]).strip() == ""):
        row["pdb_id"] = target_sample_id.split("_")[0]
    return row


def _manifest_row(
    *,
    target_sample_id: str,
    member_sample_id: str,
    member_path: Path,
    member_role: str,
    member_coefficient: float,
    target_ligand: Any,
    clash_metrics: dict[str, Any],
    warning: str,
    rank: int | None = None,
    cluster_id: int | None = None,
    rdkit_conformer_id: int | None = None,
    candidate_seed: int | None = None,
    rmsd_to_native: float | None = None,
    atom_order_rmsd_to_native: float | None = None,
    generation_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "target_sample_id": target_sample_id,
        "member_sample_id": member_sample_id,
        "member_path": str(member_path),
        "member_role": member_role,
        "member_coefficient": float(member_coefficient),
        "target_ligand_pn_unit_iid": target_ligand.pn_unit_iid,
        "target_ligand_res_name": target_ligand.res_name,
        "rank": rank,
        "cluster_id": cluster_id,
        "rdkit_conformer_id": rdkit_conformer_id,
        "candidate_seed": candidate_seed,
        "rmsd_to_native": rmsd_to_native,
        "get_best_rms_to_native": rmsd_to_native,
        "atom_order_rmsd_to_native": atom_order_rmsd_to_native,
        "has_clash": bool(clash_metrics["has_clash"]),
        "num_clashing_pairs": int(clash_metrics["num_clashing_pairs"]),
        "min_heavy_atom_distance": clash_metrics["min_heavy_atom_distance"],
        "max_vdw_overlap": clash_metrics["max_vdw_overlap"],
        "clash_target_atoms": clash_metrics["clash_target_atoms"],
        "vdw_overlap_cutoff": clash_metrics["vdw_overlap_cutoff"],
        "warning": warning,
    }
    if generation_metadata:
        row.update(generation_metadata)
    return row
