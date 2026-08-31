from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import atomworks.enums as aw_enums
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from allatom_design.data.transform.sd_featurizer import featurizer_designed_samples
from allatom_design.eval.utils.input_preprocessing import preprocess_input
from allatom_design.eval.utils.sampling_inputs import (
    is_role_sampling_inputs,
    resolve_query_pn_unit_iids,
)
from allatom_design.utils.sample_io_utils import load_example_with_parse, save_cif_file


def load_sampling_inputs_csv(sampling_inputs_csv: str | None) -> pd.DataFrame | None:
    if sampling_inputs_csv is None:
        return None

    sampling_inputs_df = pd.read_csv(sampling_inputs_csv, keep_default_na=False)
    if is_role_sampling_inputs(sampling_inputs_df):
        raise ValueError(
            "standalone AF3 structure-prediction entrypoints do not consume "
            "role-schema sampling_inputs_csv directly; run sequence sampling through "
            "allatom_design/eval/sampling/run_elix.py so derived runtime sample IDs "
            "are carried into AF3 evaluation"
        )
    if "pdb_id" not in sampling_inputs_df.columns and "pdb_key" not in sampling_inputs_df.columns:
        raise ValueError(
            "sampling_inputs_csv must contain at least one of 'pdb_id' or 'pdb_key' "
            f"columns: {sampling_inputs_csv}"
        )
    return sampling_inputs_df


def filter_sample_paths_by_sampling_inputs(
    sample_paths: list[str],
    sampling_inputs_df: pd.DataFrame | None,
) -> list[str]:
    if sampling_inputs_df is None:
        return sample_paths

    pdb_ids = set()
    if "pdb_id" in sampling_inputs_df.columns:
        pdb_ids = set(sampling_inputs_df["pdb_id"].astype(str).str.lower())

    pdb_keys = set()
    if "pdb_key" in sampling_inputs_df.columns:
        pdb_keys = set(sampling_inputs_df["pdb_key"].astype(str).str.lower())

    filtered_paths = []
    for sample_path in sample_paths:
        sample_key = Path(sample_path).stem.lower()
        sample_pdb_id = sample_key.split("_")[0]
        if sample_key in pdb_keys or sample_pdb_id in pdb_ids:
            filtered_paths.append(sample_path)
    return filtered_paths


def prepare_tc_template_cif(
    atom_array,
    out_path: str,
    cif_save_args: dict,
) -> str:
    """
    Prepare a template CIF file for AF3 template-conditioned prediction.

    1. Separate protein and ligand atom arrays.
    2. Keep only observed protein residues and existing ligands.
    3. Add dummy b_factor if missing.
    4. Save to CIF and fix formal charges through the repo CIF writer.
    """
    prot_mask = atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
    ligand_mask = np.isin(atom_array.chain_type, list(aw_enums.ChainTypeInfo.NON_POLYMERS))

    prot_atom_array = atom_array[prot_mask]
    ligand_atom_array = atom_array[ligand_mask]
    tc_atom_array = prot_atom_array + ligand_atom_array
    tc_atom_array.set_annotation(
        "atom_id",
        np.arange(1, len(tc_atom_array) + 1),
    )

    if "b_factor" not in tc_atom_array.get_annotation_categories():
        tc_atom_array.set_annotation("b_factor", np.zeros(len(tc_atom_array)))

    cif_save_args = dict(cif_save_args)
    cif_save_args.setdefault("file_type", "cif")
    return str(save_cif_file(tc_atom_array, out_path, cif_save_cfg=OmegaConf.create(cif_save_args)))


def extract_pdb_chain_info(atom_array) -> dict[str, list[str]]:
    pdb_chain_info = defaultdict(list)

    prot_atom_array = atom_array[atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L]
    ligand_atom_array = atom_array[
        np.isin(atom_array.chain_type, list(aw_enums.ChainTypeInfo.NON_POLYMERS))
    ]

    protein_pn_unit_iids = [
        str(pn_unit_iid) for pn_unit_iid in np.unique(prot_atom_array.pn_unit_iid)
    ]
    ligand_pn_unit_iids = [
        str(pn_unit_iid) for pn_unit_iid in np.unique(ligand_atom_array.pn_unit_iid)
    ]
    ligand_ccd_codes = [
        str(ligand_atom_array[_pn_unit_iid_mask(ligand_atom_array, pn_unit_iid)].res_name[0])
        for pn_unit_iid in ligand_pn_unit_iids
    ]

    for pn_unit_iid in protein_pn_unit_iids:
        pdb_chain_info["protein_pn_unit_iids"].append(str(pn_unit_iid))

    for pn_unit_iid, ccd_code in zip(ligand_pn_unit_iids, ligand_ccd_codes):
        pdb_chain_info["ligand_pn_unit_iids"].append(str(pn_unit_iid))
        pdb_chain_info["ligand_ccd_codes"].append(str(ccd_code))

    return pdb_chain_info


def _pn_unit_iid_mask(atom_array, pn_unit_iid: str) -> np.ndarray:
    return np.array([str(x) == str(pn_unit_iid) for x in atom_array.pn_unit_iid], dtype=bool)


def _filter_chain_info_to_query(
    pdb_chain_info: dict[str, list[str]],
    query_pn_unit_iids: list[str],
) -> dict[str, list[str]]:
    query_set = set(map(str, query_pn_unit_iids))
    filtered_info = defaultdict(list)

    for pn_unit_iid in pdb_chain_info["protein_pn_unit_iids"]:
        if str(pn_unit_iid) in query_set:
            filtered_info["protein_pn_unit_iids"].append(str(pn_unit_iid))

    for pn_unit_iid, ccd_code in zip(
        pdb_chain_info["ligand_pn_unit_iids"],
        pdb_chain_info["ligand_ccd_codes"],
    ):
        if str(pn_unit_iid) in query_set:
            filtered_info["ligand_pn_unit_iids"].append(str(pn_unit_iid))
            filtered_info["ligand_ccd_codes"].append(str(ccd_code))

    return filtered_info


def _filter_atom_array_to_query(atom_array, query_pn_unit_iids: list[str]):
    query_mask = np.zeros(len(atom_array), dtype=bool)
    for pn_unit_iid in query_pn_unit_iids:
        query_mask |= _pn_unit_iid_mask(atom_array, pn_unit_iid)
    return atom_array[query_mask]


def load_af3_eval_sample(
    *,
    sample_path: str,
    cif_parse_cfg: DictConfig | dict[str, Any],
    preprocess_cfg: DictConfig | dict[str, Any],
    featurizer_cfg: DictConfig | dict[str, Any],
    sample_is_designed: bool,
    sampling_inputs_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    example = load_example_with_parse(sample_path, cif_parse_cfg)
    example = preprocess_input(
        example=example,
        preprocess_cfg=preprocess_cfg,
        sample_is_designed=sample_is_designed,
    )

    featurizer_cfg_dict = OmegaConf.to_container(featurizer_cfg, resolve=True)
    featurizer = featurizer_designed_samples(**featurizer_cfg_dict)
    example = featurizer(example)

    sample_key = Path(sample_path).stem
    sample_pdb_id = sample_key.split("_")[0]
    atom_array = example["atom_array"]
    pdb_chain_info = extract_pdb_chain_info(atom_array)

    if sampling_inputs_df is not None:
        query_pn_unit_iids = resolve_query_pn_unit_iids(
            atom_array=atom_array,
            sampling_inputs_df=sampling_inputs_df,
            pdb_id=sample_pdb_id,
            pdb_key=sample_key,
        )
        pdb_chain_info = _filter_chain_info_to_query(pdb_chain_info, query_pn_unit_iids)
        atom_array = _filter_atom_array_to_query(atom_array, query_pn_unit_iids)

    return {
        "atom_array": atom_array,
        "pdb_chain_info": pdb_chain_info,
    }
