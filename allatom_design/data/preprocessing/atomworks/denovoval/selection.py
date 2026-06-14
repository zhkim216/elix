"""Selection policies for the denovoval small-molecule/metal validation set."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from allatom_design.data.preprocessing.atomworks.denovoval.records import METAL, SMALL_MOLECULE


def select_small_molecule_rows(
    df: pd.DataFrame,
    *,
    target_per_length_ccd: int,
    allow_underfilled: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Select up to target rows per ``(length, ccd)`` with unique conf/foldseek pairs."""
    candidates = df[
        (df["kind"] == SMALL_MOLECULE)
        & df["foldseek_cluster"].notna()
        & (df["foldseek_cluster"].astype(str) != "")
    ].copy()
    messages: list[str] = []
    selected_frames: list[pd.DataFrame] = []
    for (length, ccd_code), group in candidates.groupby(["length", "ccd_code"], sort=True):
        group = group.sort_values(
            ["conf_cluster", "foldseek_cluster", "foldseek_cluster_size", "sample_id"],
            kind="mergesort",
        )
        seen_pairs: set[tuple[str, str]] = set()
        selected_indices: list[Any] = []
        for index, row in group.iterrows():
            pair = (str(row["conf_cluster"]), str(row["foldseek_cluster"]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            selected_indices.append(index)
            if len(selected_indices) >= target_per_length_ccd:
                break

        if len(selected_indices) < target_per_length_ccd:
            message = (
                f"small_molecule {ccd_code} len{int(length)} underfilled: "
                f"selected {len(selected_indices)}/{target_per_length_ccd} unique "
                "(conf_cluster, foldseek_cluster) pairs"
            )
            messages.append(message)
            if not allow_underfilled:
                raise ValueError(message)

        selected = group.loc[selected_indices].copy()
        selected["selection_policy"] = "unique_conf_foldseek_pair"
        selected["selection_index_within_group"] = np.arange(len(selected), dtype=int)
        selected_frames.append(selected)

    return _concat_or_empty(selected_frames, df), messages


def select_metal_rows(
    df: pd.DataFrame,
    *,
    target_per_length_ccd: int,
    max_rasa: float,
    allow_underfilled: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Select metals after RASA filtering, prioritizing foldseek cluster diversity."""
    metal_df = df[df["kind"] == METAL].copy()
    rasa = pd.to_numeric(metal_df["rasa_value"], errors="coerce")
    candidates = metal_df[
        (metal_df["foldseek_cluster"].notna())
        & (metal_df["foldseek_cluster"].astype(str) != "")
        & (metal_df["metal_rasa_status"] == "ok")
        & (rasa <= max_rasa)
    ].copy()
    if not candidates.empty:
        candidates["rasa_value"] = pd.to_numeric(candidates["rasa_value"], errors="coerce").astype(float)

    messages: list[str] = []
    selected_frames: list[pd.DataFrame] = []
    for (length, ccd_code), group in candidates.groupby(["length", "ccd_code"], sort=True):
        group = group.sort_values(
            ["rasa_value", "foldseek_cluster", "foldseek_cluster_size", "sample_id"],
            kind="mergesort",
        )
        first_pass = group.drop_duplicates("foldseek_cluster", keep="first")
        selected = first_pass.iloc[:target_per_length_ccd].copy()

        if len(selected) < target_per_length_ccd:
            selected_ids = set(selected["sample_id"].astype(str))
            remaining = group[~group["sample_id"].astype(str).isin(selected_ids)].copy()
            fill_n = target_per_length_ccd - len(selected)
            if fill_n > 0 and not remaining.empty:
                selected = pd.concat([selected, remaining.iloc[:fill_n]], ignore_index=False)

        if len(selected) < target_per_length_ccd:
            message = (
                f"metal {ccd_code} len{int(length)} underfilled after RASA<={max_rasa}: "
                f"selected {len(selected)}/{target_per_length_ccd}"
            )
            messages.append(message)
            if not allow_underfilled:
                raise ValueError(message)

        selected = selected.copy()
        selected["selection_policy"] = "rasa_filtered_max_foldseek_diversity"
        selected["selection_index_within_group"] = np.arange(len(selected), dtype=int)
        selected_frames.append(selected)

    return _concat_or_empty(selected_frames, df), messages


def assign_final_names(selected_df: pd.DataFrame) -> pd.DataFrame:
    """Assign compact final names ``{ccd}_len{length}_{idx}.cif``."""
    if selected_df.empty:
        return selected_df.copy()

    df = selected_df.copy().sort_values(
        ["ccd_code", "length", "kind", "selection_index_within_group", "sample_id"],
        kind="mergesort",
    )
    counters: Counter[tuple[str, int]] = Counter()
    final_ids: list[str] = []
    final_filenames: list[str] = []
    for row in df.itertuples(index=False):
        key = (str(row.ccd_code), int(row.length))
        index = counters[key]
        counters[key] += 1
        final_id = f"{key[0]}_len{key[1]}_{index}"
        final_ids.append(final_id)
        final_filenames.append(f"{final_id}.cif")

    df.insert(0, "final_id", final_ids)
    df.insert(1, "final_filename", final_filenames)
    df.insert(2, "selection_rank", np.arange(1, len(df) + 1, dtype=int))
    return df.reset_index(drop=True)


def build_selection_summary(
    *,
    all_records_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    warning_messages: list[str],
) -> dict[str, Any]:
    selected_group_counts = (
        selected_df.groupby(["kind", "length", "ccd_code"], dropna=False)
        .size()
        .reset_index(name="n_selected")
        .to_dict(orient="records")
        if not selected_df.empty
        else []
    )
    discovered_group_counts = (
        all_records_df.groupby(["kind", "length", "ccd_code"], dropna=False)
        .size()
        .reset_index(name="n_discovered")
        .to_dict(orient="records")
        if not all_records_df.empty
        else []
    )
    return {
        "n_discovered": int(len(all_records_df)),
        "n_selected": int(len(selected_df)),
        "selected_by_kind": selected_df["kind"].value_counts().sort_index().to_dict()
        if not selected_df.empty
        else {},
        "discovered_group_counts": discovered_group_counts,
        "selected_group_counts": selected_group_counts,
        "warnings": warning_messages,
    }


def _concat_or_empty(frames: list[pd.DataFrame], like_df: pd.DataFrame) -> pd.DataFrame:
    if not frames:
        return like_df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)
