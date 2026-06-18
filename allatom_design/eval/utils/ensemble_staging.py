from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch


@dataclass(frozen=True)
class EnsembleStagingResult:
    root_dir: Path
    pdb_paths: list[str]
    member_groups: list[list[str]]
    sampling_inputs_df: pd.DataFrame | None
    member_to_group_id: dict[str, int]
    member_to_coefficient: dict[str, float]
    member_to_target_id: dict[str, str]
    aggregation_scheme: str
    manifest_path: Path
    label: str = "ensemble"

    def expand_pos_constraints(
        self,
        pos_constraint_df: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        return expand_pos_constraint_df_for_members(pos_constraint_df, self)

    def iter_member_batches(self, *, max_members: int) -> Iterator[list[str]]:
        return iter_member_batches(self.member_groups, max_members=max_members)

    def target_count(self, batch_pdb_paths: list[str] | None = None) -> int:
        if batch_pdb_paths is None:
            return len(self.member_groups)
        return len(
            {
                self.member_to_group_id[Path(path).stem]
                for path in batch_pdb_paths
            }
        )

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
            raise KeyError(f"Missing {self.label} staging metadata for: {missing[:5]}")

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


def compute_member_coefficients(
    *,
    num_members: int,
    scheme: str,
    label: str = "ensemble",
) -> list[float]:
    if num_members < 1:
        raise ValueError("num_members must be >= 1")
    if scheme in {"mean", "weighted_mean"}:
        return [1.0 / num_members] * num_members
    if scheme == "sqrt":
        return [1.0 / (float(num_members) ** 0.5)] * num_members
    raise ValueError(
        f"{label} member coefficient scheme must be "
        f"'mean', 'sqrt', or 'weighted_mean'; got {scheme!r}"
    )


def iter_member_batches(
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


def expand_pos_constraint_df_for_members(
    pos_constraint_df: pd.DataFrame | None,
    staging_result: EnsembleStagingResult,
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


def sampling_df_has_pdb_key(sampling_inputs_df: pd.DataFrame, pdb_key: str) -> bool:
    if "pdb_key" not in sampling_inputs_df.columns:
        return False
    return bool(
        (
            sampling_inputs_df["pdb_key"].astype(str).str.lower()
            == str(pdb_key).lower()
        ).any()
    )


def sampling_row_for_member(
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
