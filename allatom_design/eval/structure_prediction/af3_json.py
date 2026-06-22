from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from atomworks.constants import STANDARD_AA
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array_plus import AtomArray
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.sequence import get_1_from_3_letter_code

from allatom_design.data.const import METAL_ELEMENTS
from allatom_design.eval.config import get_json_config_value
from allatom_design.eval.chemical_components import normalize_ccd_code


def make_af3_protein_sequence_entry(
    *,
    chain_id: str,
    sequence: str,
    modifications: list[dict] | None = None,
    templates: list[dict] | None = None,
) -> dict:
    """Build a protein sequence entry accepted by AF3's custom input path."""
    return {
        "protein": {
            "id": chain_id,
            "sequence": sequence,
            "modifications": modifications or [],
            "unpairedMsa": "",
            "pairedMsa": "",
            "templates": templates or [],
        }
    }


def _resolve_af3_ligand_ccd_code(
    *,
    designed_sample_atom_array: AtomArray,
    ligand_pn_unit_iid: str,
    ligand_ccd_code: str,
) -> str:
    """Use the element symbol for single-atom metal ligands with synthetic names."""
    raw_ccd_code = str(ligand_ccd_code).strip()
    normalized_ccd_code = normalize_ccd_code(raw_ccd_code)
    if normalized_ccd_code in METAL_ELEMENTS:
        return normalized_ccd_code

    ligand_mask = designed_sample_atom_array.pn_unit_iid == ligand_pn_unit_iid
    ligand_atom_array = designed_sample_atom_array[ligand_mask]
    if len(ligand_atom_array) != 1:
        return raw_ccd_code

    element = normalize_ccd_code(ligand_atom_array.element[0])
    if element not in METAL_ELEMENTS:
        return raw_ccd_code

    print(
        "Using single-atom metal element as AF3 ligand CCD: "
        f"pn_unit_iid={ligand_pn_unit_iid}, res_name={raw_ccd_code}, element={element}"
    )
    return element


def _get_user_ccd_path(json_config: dict | DictConfig, pdb_chain_info: dict) -> str | None:
    user_ccd_path = pdb_chain_info.get("af3_user_ccd_path")
    if user_ccd_path is None:
        user_ccd_path = get_json_config_value(
            json_config,
            "user_ccd_path",
            "userCCDPath",
        )
    return str(user_ccd_path) if user_ccd_path else None


def _af3_ligand_ccd_codes_for_json(
    *,
    pdb_chain_info: dict,
    ligand_pn_unit_iids: list[str],
    ligand_ccd_codes: list[str],
) -> tuple[list[str], bool]:
    af3_ligand_ccd_codes = pdb_chain_info.get("af3_ligand_ccd_codes")
    if af3_ligand_ccd_codes is None:
        return list(ligand_ccd_codes), True

    codes = [str(code).strip() for code in af3_ligand_ccd_codes]
    if len(codes) != len(ligand_pn_unit_iids):
        raise ValueError(
            "af3_ligand_ccd_codes must have the same length as "
            "ligand_pn_unit_iids: "
            f"{len(codes)} != {len(ligand_pn_unit_iids)}"
        )
    if any(not code for code in codes):
        raise ValueError("af3_ligand_ccd_codes contains an empty component ID")
    return codes, False


def make_af3_json(
    af3_ss_input_dir: str = None,
    af3_tc_input_dir: str = None,
    sample_dict: dict = None,
    metadata: pd.DataFrame = None,
    json_config: dict = None,
    make_tc_input: bool = False,
) -> dict:
    """
    Create AF3 JSON input files for single-sequence and template-conditioned inference.

    The input ``sample_dict`` is mutated by adding ``af3_ss_json_paths`` and,
    when requested, ``af3_tc_json_paths`` entries for each input sample.
    """
    json_config = json_config or {}
    model_seeds = list(json_config.get("model_seeds", [42]))
    version = int(json_config.get("version", 2))

    if metadata is not None:
        protein_columns = ["q_pn_unit_is_protein"]
        nonpolymer_ligand_columns = [
            "q_pn_unit_is_small_molecule",
            "q_pn_unit_is_metal",
            "q_pn_unit_non_polymer_res_names",
        ]

        pdb_chain_info = {}

        expanded_protein_columns = []
        expanded_nonpolymer_ligand_columns = []
        for column in protein_columns:
            expanded_protein_columns.extend([f"{column}_{i}" for i in [1, 2]])
        for column in nonpolymer_ligand_columns:
            expanded_nonpolymer_ligand_columns.extend([f"{column}_{i}" for i in [1, 2]])

        for _, row in metadata.iterrows():
            pdb_key = row["pdb_id"]
            pdb_chain_info[pdb_key] = {}
            pdb_chain_info[pdb_key]["protein_pn_unit_iids"] = []
            pdb_chain_info[pdb_key]["ligand_pn_unit_iids"] = []
            pdb_chain_info[pdb_key]["ligand_ccd_codes"] = []

            for column in expanded_protein_columns:
                if row[column]:
                    suffix = column.split("_")[-1]
                    protein_pn_unit_iid = row[f"q_pn_unit_iid_{suffix}"]
                    pdb_chain_info[pdb_key]["protein_pn_unit_iids"].append(protein_pn_unit_iid)

            for column in expanded_nonpolymer_ligand_columns:
                if row[column]:
                    suffix = column.split("_")[-1]
                    ligand_pn_unit_iid = row[f"q_pn_unit_iid_{suffix}"]
                    ligand_ccd_code = row[f"q_pn_unit_non_polymer_res_names_{suffix}"]
                    pdb_chain_info[pdb_key]["ligand_pn_unit_iids"].append(ligand_pn_unit_iid)
                    pdb_chain_info[pdb_key]["ligand_ccd_codes"].append(ligand_ccd_code)

    for input_sample_id in tqdm(sample_dict.keys(), desc="Creating AF3 JSONs"):
        sample_dict[input_sample_id]["af3_ss_json_paths"] = []
        if make_tc_input:
            sample_dict[input_sample_id]["af3_tc_json_paths"] = []
        subsample_dict = sample_dict[input_sample_id]
        for dsidx, designed_sample_id in enumerate(subsample_dict["designed_sample_id"]):
            designed_sample_atom_array = subsample_dict["designed_sample_atom_array"][dsidx]
            pdb_chain_info = subsample_dict["pdb_chain_info"]
            native_res_name_by_chain_res_id = subsample_dict.get("native_res_name_by_chain_res_id") or {}

            if make_tc_input:
                template_sample_path = subsample_dict["designed_sample_path_for_af3_tc"][dsidx]
            else:
                template_sample_path = None

            job_name = designed_sample_id

            protein_pn_unit_iids = pdb_chain_info["protein_pn_unit_iids"]
            ligand_pn_unit_iids = pdb_chain_info["ligand_pn_unit_iids"]
            ligand_ccd_codes = pdb_chain_info["ligand_ccd_codes"]
            af3_ligand_ccd_codes, should_resolve_ligand_ccd = _af3_ligand_ccd_codes_for_json(
                pdb_chain_info=pdb_chain_info,
                ligand_pn_unit_iids=ligand_pn_unit_iids,
                ligand_ccd_codes=ligand_ccd_codes,
            )
            user_ccd_path = _get_user_ccd_path(json_config, pdb_chain_info)
            json_version = max(version, 3) if user_ccd_path else version

            ss_sequences = []
            tc_sequences = []
            for protein_pn_unit_iid in protein_pn_unit_iids:
                chain_mask = designed_sample_atom_array.pn_unit_iid == protein_pn_unit_iid
                _res_starts = get_residue_starts(designed_sample_atom_array[chain_mask])
                _res_ids = designed_sample_atom_array[chain_mask].res_id[_res_starts]
                _res_ids_0based = _res_ids - np.min(_res_ids)

                full_length = np.max(_res_ids) - np.min(_res_ids) + 1
                chain_seq_with_gaps = np.full(full_length, "UNK")
                chain_id = protein_pn_unit_iid.split("_")[0]
                min_res_id = int(np.min(_res_ids))
                for offset, res_id in enumerate(range(min_res_id, min_res_id + full_length)):
                    native_res_name = native_res_name_by_chain_res_id.get((chain_id, res_id))
                    if native_res_name is not None:
                        chain_seq_with_gaps[offset] = native_res_name

                chain_seq = designed_sample_atom_array[chain_mask].res_name[_res_starts]
                chain_seq_with_gaps[_res_ids_0based] = chain_seq

                chain_hetero = designed_sample_atom_array[chain_mask].hetero[_res_starts]
                hetero_flags_with_gaps = np.full(full_length, False)
                hetero_flags_with_gaps[_res_ids_0based] = chain_hetero

                modifications = []
                sequence_letters = []
                for idx, (res_name, is_hetero) in enumerate(zip(chain_seq_with_gaps, hetero_flags_with_gaps)):
                    one_letter = get_1_from_3_letter_code(
                        res_name,
                        chain_type=ChainType.POLYPEPTIDE_L,
                        use_closest_canonical=True,
                    )
                    sequence_letters.append(one_letter)

                    if is_hetero and res_name not in STANDARD_AA and res_name != "UNK":
                        modifications.append({
                            "ptmType": res_name,
                            "ptmPosition": idx + 1,
                        })

                sequence_with_gaps = "".join(sequence_letters)

                if make_tc_input:
                    query_indices = template_indices = [int(x) for x in list(_res_ids_0based)]

                chain_id = protein_pn_unit_iid.split("_")[0]
                ss_sequences.append(
                    make_af3_protein_sequence_entry(
                        chain_id=chain_id,
                        sequence=sequence_with_gaps,
                        modifications=modifications,
                        templates=[],
                    )
                )

                if make_tc_input:
                    tc_sequences.append(
                        make_af3_protein_sequence_entry(
                            chain_id=chain_id,
                            sequence=sequence_with_gaps,
                            modifications=modifications,
                            templates=[
                                {
                                    "mmcifPath": template_sample_path,
                                    "queryIndices": query_indices,
                                    "templateIndices": template_indices,
                                    "templateChainId": chain_id,
                                }
                            ],
                        )
                    )

            for ligand_pn_unit_iid, ligand_ccd_code in zip(ligand_pn_unit_iids, af3_ligand_ccd_codes):
                if should_resolve_ligand_ccd:
                    ligand_ccd_code = _resolve_af3_ligand_ccd_code(
                        designed_sample_atom_array=designed_sample_atom_array,
                        ligand_pn_unit_iid=ligand_pn_unit_iid,
                        ligand_ccd_code=ligand_ccd_code,
                    )
                ss_sequences.append({
                    "ligand": {
                        "id": ligand_pn_unit_iid.split("_")[0],
                        "ccdCodes": [ligand_ccd_code],
                    }
                })

                if make_tc_input:
                    tc_sequences.append({
                        "ligand": {
                            "id": ligand_pn_unit_iid.split("_")[0],
                            "ccdCodes": [ligand_ccd_code],
                        }
                    })

            sample_af3_ss_json = {
                "name": job_name,
                "sequences": ss_sequences,
                "modelSeeds": model_seeds,
                "dialect": "alphafold3",
                "version": json_version,
            }
            if user_ccd_path:
                sample_af3_ss_json["userCCDPath"] = user_ccd_path

            if make_tc_input:
                sample_af3_tc_json = {
                    "name": job_name,
                    "sequences": tc_sequences,
                    "modelSeeds": model_seeds,
                    "dialect": "alphafold3",
                    "version": json_version,
                }
                if user_ccd_path:
                    sample_af3_tc_json["userCCDPath"] = user_ccd_path

            json_path_ss = Path(af3_ss_input_dir, f"{job_name}.json")
            with open(json_path_ss, "w") as f:
                json.dump(sample_af3_ss_json, f)

            sample_dict[input_sample_id]["af3_ss_json_paths"].append(json_path_ss)

            if make_tc_input:
                json_path_tc = Path(af3_tc_input_dir, f"{job_name}.json")
                with open(json_path_tc, "w") as f:
                    json.dump(sample_af3_tc_json, f)
                sample_dict[input_sample_id]["af3_tc_json_paths"].append(json_path_tc)

    return sample_dict
