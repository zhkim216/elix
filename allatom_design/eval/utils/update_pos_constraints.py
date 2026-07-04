from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from allatom_design.eval.utils.input_files import get_pdb_files
from allatom_design.eval.utils.input_preprocessing import preprocess_input
from allatom_design.eval.utils.pocket_constraints import create_pos_constraint_dict_from_pocket
from allatom_design.eval.utils.sampling_inputs import (
    ROLE_LIST_COLUMNS,
    ROLE_POS_CONSTRAINT_COLUMNS,
    ROLE_SAMPLING_COLUMNS,
    normalize_role_sampling_inputs_df,
    role_binder_pn_unit_iids_from_sampling_row,
    role_context_pn_unit_iids_from_sampling_row,
)
from allatom_design.utils.sample_io_utils import load_example_with_parse


DEFAULT_CIF_PARSE_CFG = {
    "add_missing_atoms": True,
    "remove_waters": True,
    "remove_ccds": [],
    "fix_ligands_at_symmetry_centers": True,
    "fix_arginines": True,
    "convert_mse_to_met": True,
    "hydrogen_policy": "remove",
    "extra_fields": "all",
}
DEFAULT_PREPROCESS_CFG = {
    "undesired_res_names": [],
    "b_factor_min": None,
    "b_factor_max": None,
    "min_residues_for_polymers": 0,
    "remove_terminal_oxygen_protein": True,
    "remove_terminal_oxygen_nucleic_acid": True,
}


def update_pos_constraints(
    *,
    sampling_inputs_csv: str | Path,
    pdb_dir: str | Path,
    output_csv: str | Path,
    constraint_type: str,
    pocket_distance: float = 6.0,
    pdb_name_ext: str = ".cif",
    pocket_annotation_method: str | None = "calpha",
    fix_sidechains: bool = False,
    override: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    source_df = pd.read_csv(sampling_inputs_csv, keep_default_na=False)
    role_df = normalize_role_sampling_inputs_df(source_df, label=str(sampling_inputs_csv))
    if limit is not None:
        role_df = role_df.head(limit).copy()

    pdb_paths = get_pdb_files(
        pdb_dir=str(pdb_dir),
        pdb_name_list=None,
        pdb_name_ext=pdb_name_ext,
    )
    path_by_pdb_key = _path_by_pdb_key(pdb_paths)
    example_cache: dict[str, dict[str, Any]] = {}

    out_df = role_df.loc[:, list(ROLE_SAMPLING_COLUMNS)].copy()
    updated_rows = 0
    skipped_empty_context = 0

    for row_index, row in role_df.iterrows():
        context_iids = role_context_pn_unit_iids_from_sampling_row(row)
        if not context_iids:
            skipped_empty_context += 1
            continue

        existing_constraints = {
            column: _text_cell(row[column])
            for column in ("fixed_pos_seq", "fixed_pos_scn")
        }
        if any(existing_constraints.values()) and not override:
            raise ValueError(
                f"row {row_index} ({row['pdb_key']}): fixed_pos_seq/fixed_pos_scn "
                "already populated; pass --override to replace auto-generated constraints"
            )

        pdb_key = str(row["pdb_key"])
        if pdb_key not in path_by_pdb_key:
            raise FileNotFoundError(f"pdb_key {pdb_key!r} not found under {pdb_dir}")

        example = example_cache.get(pdb_key)
        if example is None:
            example = load_example_with_parse(
                path_by_pdb_key[pdb_key],
                OmegaConf.create(DEFAULT_CIF_PARSE_CFG),
            )
            example = preprocess_input(
                example=example,
                preprocess_cfg=OmegaConf.create(DEFAULT_PREPROCESS_CFG),
                sample_is_designed=False,
            )
            example_cache[pdb_key] = example

        constraint_row, _ = create_pos_constraint_dict_from_pocket(
            pdb_key=pdb_key,
            atom_array=example["atom_array"],
            pocket_distance=pocket_distance,
            constraint_type=constraint_type,
            receptor_pn_unit_iids=role_binder_pn_unit_iids_from_sampling_row(row),
            ligand_pn_unit_iids=context_iids,
            pocket_annotation_method=pocket_annotation_method,
            return_ligand_mpnn_format=False,
        )
        fixed_pos_seq = _text_cell(constraint_row.get("fixed_pos_seq", ""))
        out_df.at[row_index, "fixed_pos_seq"] = fixed_pos_seq
        out_df.at[row_index, "fixed_pos_scn"] = fixed_pos_seq if fix_sidechains and fixed_pos_seq else ""
        updated_rows += 1

    out_df = _serialize_role_sampling_df(out_df)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(
        f"Wrote {len(out_df)} role sampling rows to {output_csv}; "
        f"updated={updated_rows}, skipped_empty_context={skipped_empty_context}, "
        f"unique_source_cifs={len(example_cache)}"
    )
    return out_df


def _path_by_pdb_key(pdb_paths: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    duplicates: list[str] = []
    for pdb_path in pdb_paths:
        key = Path(pdb_path).stem
        if key in out:
            duplicates.append(key)
            continue
        out[key] = pdb_path
    if duplicates:
        raise ValueError(f"Duplicate pdb_key stems under pdb_dir: {duplicates[:10]}")
    return out


def _serialize_role_sampling_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[:, list(ROLE_SAMPLING_COLUMNS)].copy()
    for column in ROLE_LIST_COLUMNS:
        out[column] = out[column].apply(lambda values: json.dumps(list(values)))
    for column in ROLE_POS_CONSTRAINT_COLUMNS:
        out[column] = out[column].apply(_text_cell)
    return out


def _text_cell(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, (float, np.floating)) and np.isnan(raw_value):
        return ""
    return str(raw_value).strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update role-schema sampling CSV fixed_pos_* columns in-place by output path.",
    )
    parser.add_argument("--sampling-inputs-csv", required=True)
    parser.add_argument("--pdb-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--constraint-type", required=True, choices=("pocket", "scaffold"))
    parser.add_argument("--pocket-distance", type=float, default=6.0)
    parser.add_argument("--pdb-name-ext", default=".cif")
    parser.add_argument(
        "--pocket-annotation-method",
        default="calpha",
        choices=("all_atom", "calpha", "pseudocb"),
    )
    parser.add_argument("--fix-sidechains", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    update_pos_constraints(
        sampling_inputs_csv=args.sampling_inputs_csv,
        pdb_dir=args.pdb_dir,
        output_csv=args.output_csv,
        constraint_type=args.constraint_type,
        pocket_distance=args.pocket_distance,
        pdb_name_ext=args.pdb_name_ext,
        pocket_annotation_method=args.pocket_annotation_method,
        fix_sidechains=args.fix_sidechains,
        override=args.override,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
