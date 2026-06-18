from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from allatom_design.eval.utils.data_utils import (
    _matched_sampling_input_row,
    parse_query_pn_unit_iids,
)
from allatom_design.eval.utils.ensemble_conditioning import (
    normalize_ensemble_conditioning_cfg,
)
from allatom_design.eval.utils.ensemble_staging import (
    EnsembleStagingResult,
    compute_member_coefficients,
    sampling_df_has_pdb_key,
)


RANK_RE = re.compile(r"(?:^|_)rank(?P<rank>\d+)(?:_|$)")
DEFAULT_RUNTIME_SAMPLING_COLUMNS = [
    "pdb_key",
    "pdb_id",
    "query_pn_unit_iids",
    "ccd_code",
]


def stage_pharm_retrieval_ensembles(
    *,
    pdb_paths: list[str],
    out_dir: str | Path,
    ensemble_cfg: dict[str, Any] | DictConfig,
    sampling_inputs_df: pd.DataFrame | None,
    csv_suffix: str = "",
) -> EnsembleStagingResult:
    cfg = normalize_ensemble_conditioning_cfg(ensemble_cfg)
    small_molecule_cfg = cfg["small_molecule"]
    if small_molecule_cfg["mode"] != "pharm_retrieval":
        raise ValueError(
            "stage_pharm_retrieval_ensembles requires "
            "small_molecule.mode='pharm_retrieval'"
        )

    pharm_cfg = dict(small_molecule_cfg["pharm_retrieval"])
    rank_indices = _normalize_rank_indices(pharm_cfg.get("rank_indices"))
    if not rank_indices:
        raise ValueError(
            "pharm_retrieval rank_indices must contain at least one rank index; "
            "disable ensemble_conditioning for the baseline query-only run"
        )

    cif_root = _optional_path(pharm_cfg.get("cif_root"))
    selected_queries_tsv = _optional_path(pharm_cfg.get("selected_queries_tsv"))
    query_metadata = _load_query_metadata(selected_queries_tsv)
    default_query_pn_unit_iids = _normalize_query_pn_unit_iids(
        pharm_cfg.get("query_pn_unit_iids")
    )

    root_dir = Path(out_dir) / "samples_with_pharm_retrieval"
    root_dir.mkdir(parents=True, exist_ok=True)

    total_members = int(cfg["total_members"])
    coefficients = compute_member_coefficients(
        num_members=total_members,
        scheme=cfg["weights"]["scheme"],
        label="pharm retrieval",
    )

    manifest_rows: list[dict[str, Any]] = []
    member_groups: list[list[str]] = []
    member_to_group_id: dict[str, int] = {}
    member_to_coefficient: dict[str, float] = {}
    member_to_target_id: dict[str, str] = {}
    sampling_extra_rows: list[dict[str, Any]] = []
    runtime_sampling_inputs_df = _initial_sampling_inputs_df(sampling_inputs_df)
    seen_member_sample_ids: set[str] = set()

    for group_id, pdb_path in enumerate(pdb_paths):
        input_path = Path(pdb_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing pharm retrieval query CIF: {input_path}")
        _validate_input_under_cif_root(input_path, cif_root)

        query_dir = input_path.parent
        query_id = query_dir.name
        target_sample_id = input_path.stem
        sampling_row = _matched_sampling_input_row(
            sampling_inputs_df,
            pdb_id=query_id,
            pdb_key=target_sample_id,
        )
        query_pn_unit_iids = _query_pn_unit_iids_for_target(
            query_id=query_id,
            sampling_row=sampling_row,
            query_metadata=query_metadata,
            default_query_pn_unit_iids=default_query_pn_unit_iids,
        )

        query_members = _discover_query_members(query_dir)
        rank_members = _discover_rank_members(query_dir)
        missing_ranks = [rank for rank in rank_indices if rank not in rank_members]
        if missing_ranks:
            raise FileNotFoundError(
                f"Missing pharm retrieval rank CIFs for query_id={query_id}: "
                f"rank_indices={missing_ranks}"
            )

        selected_members: list[tuple[Path, str, int | None]] = [
            (path, "query_original", None)
            for path in query_members
        ]
        selected_members.extend(
            (rank_members[rank], "pharm_rank", rank)
            for rank in rank_indices
        )
        if len(selected_members) != total_members:
            raise ValueError(
                f"pharm_retrieval target {target_sample_id} produced "
                f"{len(selected_members)} members, but ensemble_conditioning."
                f"total_members={total_members}. query_original_members="
                f"{len(query_members)}, rank_indices={rank_indices}"
            )

        group_paths: list[str] = []
        for member_order, (source_path, member_role, rank_index) in enumerate(selected_members):
            member_sample_id = source_path.stem
            if member_sample_id in seen_member_sample_ids:
                raise ValueError(
                    "Duplicate pharm retrieval staged member sample id: "
                    f"{member_sample_id}"
                )
            seen_member_sample_ids.add(member_sample_id)

            out_path = root_dir / source_path.name
            shutil.copy2(source_path, out_path)
            group_paths.append(str(out_path))

            member_to_group_id[member_sample_id] = group_id
            member_to_coefficient[member_sample_id] = coefficients[member_order]
            member_to_target_id[member_sample_id] = target_sample_id

            if not sampling_df_has_pdb_key(runtime_sampling_inputs_df, member_sample_id):
                sampling_extra_rows.append(
                    _sampling_row_for_pharm_member(
                        sampling_inputs_df=runtime_sampling_inputs_df,
                        source_row=sampling_row,
                        member_sample_id=member_sample_id,
                        query_id=query_id,
                        query_pn_unit_iids=query_pn_unit_iids,
                    )
                )

            manifest_rows.append(
                {
                    "target_sample_id": target_sample_id,
                    "query_id": query_id,
                    "member_sample_id": member_sample_id,
                    "member_path": str(out_path),
                    "source_path": str(source_path),
                    "member_role": member_role,
                    "rank_index": "" if rank_index is None else str(rank_index),
                    "member_order": member_order,
                    "member_coefficient": coefficients[member_order],
                    "n_query_members": len(query_members),
                    "n_rank_members": len(rank_indices),
                    "n_members": len(selected_members),
                    "requested_rank_indices": repr(rank_indices),
                    "query_pn_unit_iids": repr(query_pn_unit_iids),
                    "selected_queries_tsv": (
                        None if selected_queries_tsv is None else str(selected_queries_tsv)
                    ),
                }
            )

        member_groups.append(group_paths)

    if sampling_extra_rows:
        runtime_sampling_inputs_df = pd.concat(
            [
                runtime_sampling_inputs_df,
                pd.DataFrame(sampling_extra_rows, columns=runtime_sampling_inputs_df.columns),
            ],
            ignore_index=True,
        )

    manifest_path = root_dir / f"pharm_retrieval_manifest{csv_suffix}.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    return EnsembleStagingResult(
        root_dir=root_dir,
        pdb_paths=[path for group in member_groups for path in group],
        member_groups=member_groups,
        sampling_inputs_df=runtime_sampling_inputs_df,
        member_to_group_id=member_to_group_id,
        member_to_coefficient=member_to_coefficient,
        member_to_target_id=member_to_target_id,
        aggregation_scheme=cfg["weights"]["scheme"],
        manifest_path=manifest_path,
        label="pharm retrieval",
    )


def _normalize_rank_indices(raw_rank_indices: Any) -> list[int]:
    if raw_rank_indices is None:
        return []
    parsed = raw_rank_indices
    if isinstance(raw_rank_indices, str):
        stripped = raw_rank_indices.strip()
        if stripped == "":
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = [item.strip() for item in stripped.split(",") if item.strip()]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            "pharm_retrieval rank_indices must be a list of non-negative integers"
        )

    ranks: list[int] = []
    for value in parsed:
        if isinstance(value, bool):
            raise ValueError("pharm_retrieval rank_indices cannot contain booleans")
        try:
            rank = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pharm_retrieval rank_indices must be non-negative integers"
            ) from exc
        if rank < 0:
            raise ValueError(
                f"pharm_retrieval rank_indices must be non-negative, got {rank}"
            )
        ranks.append(rank)

    duplicates = sorted({rank for rank in ranks if ranks.count(rank) > 1})
    if duplicates:
        raise ValueError(
            f"pharm_retrieval rank_indices contains duplicate ranks: {duplicates}"
        )
    return ranks


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    value_str = str(value).strip()
    if value_str == "" or value_str.lower() == "null":
        return None
    return Path(value_str)


def _normalize_query_pn_unit_iids(raw_value: Any) -> list[str]:
    return parse_query_pn_unit_iids(raw_value)


def _load_query_metadata(path: Path | None) -> dict[str, pd.Series]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected_queries_tsv={path}")
    metadata_df = pd.read_csv(path, sep="\t", keep_default_na=False)
    required_columns = {"query_ccd", "query_pn_unit_iid"}
    missing = required_columns - set(metadata_df.columns)
    if missing:
        raise ValueError(
            f"selected_queries_tsv={path} missing required columns: {sorted(missing)}"
        )
    duplicated = metadata_df[metadata_df["query_ccd"].astype(str).duplicated(keep=False)]
    if not duplicated.empty:
        duplicates = sorted(duplicated["query_ccd"].astype(str).unique().tolist())
        raise ValueError(
            f"selected_queries_tsv={path} has duplicate query_ccd values: {duplicates}"
        )
    return {
        str(row["query_ccd"]): row
        for _, row in metadata_df.iterrows()
    }


def _initial_sampling_inputs_df(
    sampling_inputs_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if sampling_inputs_df is not None:
        runtime_df = sampling_inputs_df.copy()
        for column in DEFAULT_RUNTIME_SAMPLING_COLUMNS:
            if column not in runtime_df.columns:
                runtime_df[column] = ""
        return runtime_df
    return pd.DataFrame(columns=DEFAULT_RUNTIME_SAMPLING_COLUMNS)


def _validate_input_under_cif_root(input_path: Path, cif_root: Path | None) -> None:
    if cif_root is None:
        return
    try:
        input_path.resolve().relative_to(cif_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"pharm retrieval input path {input_path} is not under cif_root={cif_root}"
        ) from exc


def _query_pn_unit_iids_for_target(
    *,
    query_id: str,
    sampling_row: pd.Series | None,
    query_metadata: dict[str, pd.Series],
    default_query_pn_unit_iids: list[str],
) -> list[str]:
    row_iids = _query_pn_unit_iids_from_sampling_row(sampling_row)
    if row_iids:
        return row_iids

    metadata_row = query_metadata.get(query_id)
    if metadata_row is not None:
        metadata_iids = parse_query_pn_unit_iids(metadata_row["query_pn_unit_iid"])
        if metadata_iids:
            return metadata_iids

    if default_query_pn_unit_iids:
        return list(default_query_pn_unit_iids)

    raise ValueError(
        "pharm retrieval requires query_pn_unit_iids from sampling_inputs_df, "
        f"selected_queries_tsv, or small_molecule.pharm_retrieval.query_pn_unit_iids; "
        f"missing for query_id={query_id}"
    )


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


def _discover_query_members(query_dir: Path) -> list[Path]:
    query_members = sorted(
        [
            path
            for path in query_dir.glob("*.cif")
            if RANK_RE.search(path.stem) is None
        ]
    )
    if not query_members:
        raise FileNotFoundError(
            f"No rankless query/original CIF found in pharm retrieval directory {query_dir}"
        )
    return query_members


def _discover_rank_members(query_dir: Path) -> dict[int, Path]:
    rank_members: dict[int, Path] = {}
    duplicates: dict[int, list[Path]] = {}
    for path in sorted(query_dir.glob("*.cif")):
        rank_match = RANK_RE.search(path.stem)
        if rank_match is None:
            continue
        rank = int(rank_match.group("rank"))
        if rank in rank_members:
            duplicates.setdefault(rank, [rank_members[rank]]).append(path)
            continue
        rank_members[rank] = path
    if duplicates:
        details = {
            rank: [str(path) for path in paths]
            for rank, paths in sorted(duplicates.items())
        }
        raise ValueError(f"Duplicate pharm retrieval rank CIFs in {query_dir}: {details}")
    return rank_members


def _sampling_row_for_pharm_member(
    *,
    sampling_inputs_df: pd.DataFrame,
    source_row: pd.Series | None,
    member_sample_id: str,
    query_id: str,
    query_pn_unit_iids: list[str],
) -> dict[str, Any]:
    if source_row is None:
        row = {column: "" for column in sampling_inputs_df.columns}
    else:
        row = source_row.to_dict()
        for column in sampling_inputs_df.columns:
            row.setdefault(column, "")
    if "pdb_key" in row:
        row["pdb_key"] = member_sample_id
    if "pdb_id" in row:
        row["pdb_id"] = query_id
    if "query_pn_unit_iids" in row:
        row["query_pn_unit_iids"] = repr(query_pn_unit_iids)
    if "query_pn_unit_iids_json" in row:
        row["query_pn_unit_iids_json"] = repr(query_pn_unit_iids)
    if "ccd_code" in row:
        row["ccd_code"] = query_id
    return row
