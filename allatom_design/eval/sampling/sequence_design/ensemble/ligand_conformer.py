from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
from omegaconf import DictConfig

from allatom_design.data.transform.ligand_conformers import (
    compute_ligand_protein_clash_metrics,
    find_query_small_molecule_ligands,
    generate_ligand_conformer_decoys,
)
from allatom_design.eval.utils.sampling_inputs import (
    matched_sampling_input_row,
    resolve_query_pn_unit_iids_from_sampling_row,
)
from allatom_design.eval.sampling.sequence_design.ensemble.conditioning import (
    normalize_ensemble_conditioning_cfg,
)
from allatom_design.eval.sampling.sequence_design.ensemble.staging import (
    EnsembleStagingResult as LigandConformerStagingResult,
    expand_pos_constraint_df_for_members,
    iter_member_batches,
    sampling_df_has_pdb_key,
    sampling_row_for_member,
)
from allatom_design.utils.sample_io_utils import load_example_with_parse, save_cif_file


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

    total_members = int(cfg["total_members"])
    max_decoys = max(total_members - 1, 0)
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
        sampling_row = matched_sampling_input_row(
            sampling_inputs_df,
            pdb_id=pdb_id,
            pdb_key=target_sample_id,
        )
        query_pn_unit_iids = resolve_query_pn_unit_iids_from_sampling_row(sampling_row)

        example = load_example_with_parse(str(input_path), cif_parse_cfg)
        atom_array = example["atom_array"]
        target_ligand_candidates = find_query_small_molecule_ligands(
            atom_array,
            query_pn_unit_iids=query_pn_unit_iids or None,
        )
        warnings: list[str] = []
        if len(target_ligand_candidates) > 1:
            raise ValueError(
                "Expected at most one query small-molecule ligand for "
                "ligand_conformer ensemble conditioning, found "
                f"{len(target_ligand_candidates)} for target_sample_id="
                f"{target_sample_id}"
            )
        if target_ligand_candidates:
            target_ligand = target_ligand_candidates[0]
            original_clash_metrics = compute_ligand_protein_clash_metrics(
                atom_array,
                ligand_mask=target_ligand.heavy_mask,
                clash_target_atoms=small_molecule_cfg["clash_target_atoms"],
                vdw_overlap_cutoff=small_molecule_cfg["vdw_overlap_cutoff"],
            )
        else:
            target_ligand = None
            original_clash_metrics = None
            warnings.append("no_query_small_molecule_ligand")

        decoys = []
        generation_metadata: dict[str, Any] = {}
        if target_ligand is not None and max_decoys > 0:
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
        num_fallback_copies = max_decoys - len(selected_decoys)
        if num_fallback_copies > 0:
            warnings.append(
                "fallback_original_copy: "
                f"count={num_fallback_copies}"
            )

        coefficients = compute_ligand_conformer_member_coefficients(
            num_decoys=len(selected_decoys),
            num_fallback_copies=num_fallback_copies,
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

        member_idx = 1
        for decoy in selected_decoys:
            decoy_idx = member_idx
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
            member_idx += 1

        for fallback_idx in range(member_idx, total_members):
            fallback_out_path = root_dir / f"{target_sample_id}_ligconf_{fallback_idx}.cif"
            save_cif_file(atom_array, fallback_out_path, cif_save_cfg=cif_save_cfg)
            group_paths.append(str(fallback_out_path))
            manifest_rows.append(
                _manifest_row(
                    target_sample_id=target_sample_id,
                    member_sample_id=fallback_out_path.stem,
                    member_path=fallback_out_path,
                    member_role="fallback_original_copy",
                    member_coefficient=coefficients[fallback_idx],
                    target_ligand=target_ligand,
                    clash_metrics=original_clash_metrics,
                    warning="; ".join(warnings),
                    generation_metadata=generation_metadata,
                )
            )

        for member_idx, member_path in enumerate(group_paths):
            member_sample_id = Path(member_path).stem
            member_to_group_id[member_sample_id] = group_id
            member_to_coefficient[member_sample_id] = coefficients[member_idx]
            member_to_target_id[member_sample_id] = target_sample_id
            if runtime_sampling_inputs_df is not None and not sampling_df_has_pdb_key(
                runtime_sampling_inputs_df,
                member_sample_id,
            ):
                sampling_extra_rows.append(
                    sampling_row_for_member(
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
        label="ligand conformer",
    )


def compute_ligand_conformer_member_coefficients(
    *,
    num_decoys: int,
    num_fallback_copies: int,
    scheme: str,
    ref_weight: float,
    decoy_total_weight: float,
) -> list[float]:
    if num_decoys < 0:
        raise ValueError("num_decoys must be non-negative")
    if num_fallback_copies < 0:
        raise ValueError("num_fallback_copies must be non-negative")
    num_non_original_members = num_decoys + num_fallback_copies
    num_members = num_non_original_members + 1
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
    if num_non_original_members == 0:
        return [1.0]
    total = float(ref_weight) + float(decoy_total_weight)
    if total <= 0:
        raise ValueError("member coefficients must have positive total")
    return [
        float(ref_weight) / total,
        *(
            [float(decoy_total_weight) / total / num_non_original_members]
            * num_non_original_members
        ),
    ]


def iter_ligand_conformer_member_batches(
    member_groups: list[list[str]],
    *,
    max_members: int,
) -> Iterator[list[str]]:
    return iter_member_batches(member_groups, max_members=max_members)


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
    return expand_pos_constraint_df_for_members(pos_constraint_df, staging_result)


def ligand_conformer_target_count(
    batch_pdb_paths: list[str],
    staging_result: LigandConformerStagingResult,
) -> int:
    return staging_result.target_count(batch_pdb_paths)


def _manifest_row(
    *,
    target_sample_id: str,
    member_sample_id: str,
    member_path: Path,
    member_role: str,
    member_coefficient: float,
    target_ligand: Any | None,
    clash_metrics: dict[str, Any] | None,
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
        "target_ligand_pn_unit_iid": (
            None if target_ligand is None else target_ligand.pn_unit_iid
        ),
        "target_ligand_res_name": (
            None if target_ligand is None else target_ligand.res_name
        ),
        "rank": rank,
        "cluster_id": cluster_id,
        "rdkit_conformer_id": rdkit_conformer_id,
        "candidate_seed": candidate_seed,
        "rmsd_to_native": rmsd_to_native,
        "get_best_rms_to_native": rmsd_to_native,
        "atom_order_rmsd_to_native": atom_order_rmsd_to_native,
        "has_clash": None if clash_metrics is None else bool(clash_metrics["has_clash"]),
        "num_clashing_pairs": (
            None if clash_metrics is None else int(clash_metrics["num_clashing_pairs"])
        ),
        "min_heavy_atom_distance": (
            None if clash_metrics is None else clash_metrics["min_heavy_atom_distance"]
        ),
        "max_vdw_overlap": (
            None if clash_metrics is None else clash_metrics["max_vdw_overlap"]
        ),
        "clash_target_atoms": (
            None if clash_metrics is None else clash_metrics["clash_target_atoms"]
        ),
        "vdw_overlap_cutoff": (
            None if clash_metrics is None else clash_metrics["vdw_overlap_cutoff"]
        ),
        "warning": warning,
    }
    if generation_metadata:
        row.update(generation_metadata)
    return row
