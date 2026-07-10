from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from atomworks.constants import STANDARD_AA
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array_plus import AtomArray
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.sequence import get_1_from_3_letter_code

from allatom_design.eval.config import get_json_config_value
from allatom_design.eval.utils.sampling_inputs import normalize_pn_unit_roles


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
) -> list[str]:
    af3_ligand_ccd_codes = pdb_chain_info.get("af3_ligand_ccd_codes")
    if af3_ligand_ccd_codes is None:
        return list(ligand_ccd_codes)

    codes = [str(code).strip() for code in af3_ligand_ccd_codes]
    if len(codes) != len(ligand_pn_unit_iids):
        raise ValueError(
            "af3_ligand_ccd_codes must have the same length as "
            "ligand_pn_unit_iids: "
            f"{len(codes)} != {len(ligand_pn_unit_iids)}"
        )
    if any(not code for code in codes):
        raise ValueError("af3_ligand_ccd_codes contains an empty component ID")
    return codes


def build_af3_chain_id_to_pn_unit_iid(
    *,
    protein_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
) -> dict[str, str]:
    """Map each serialized AF3 chain ID back to its source PN-unit IID."""
    chain_id_to_pn_unit_iid: dict[str, str] = {}
    for unit_kind, pn_unit_iids in (
        ("protein", protein_pn_unit_iids),
        ("ligand", ligand_pn_unit_iids),
    ):
        for raw_pn_unit_iid in pn_unit_iids:
            if not isinstance(raw_pn_unit_iid, str):
                raise ValueError(
                    f"{unit_kind} pn_unit_iid must be a string; "
                    f"got {raw_pn_unit_iid!r}"
                )
            pn_unit_iid = raw_pn_unit_iid
            component_parts: list[tuple[str, str]] = []
            for component in pn_unit_iid.split(","):
                pn_unit_id, separator, transformation_id = component.partition("_")
                if not separator or not pn_unit_id or not transformation_id:
                    raise ValueError(
                        "Malformed pn_unit_iid component; expected "
                        "'<PN-unit ID>_<transformation ID>', "
                        f"got {component!r} in {raw_pn_unit_iid!r}"
                    )
                component_parts.append((pn_unit_id, transformation_id))

            transformation_ids = {
                transformation_id
                for _, transformation_id in component_parts
            }
            if len(transformation_ids) != 1:
                raise ValueError(
                    "All components of a compound pn_unit_iid must use the same "
                    f"transformation ID; got {raw_pn_unit_iid!r}"
                )

            af3_chain_id = component_parts[0][0]
            if not af3_chain_id.isalpha() or af3_chain_id.islower():
                raise ValueError(
                    "AF3 serialization requires an uppercase alphabetic chain ID in "
                    f"pn_unit_iid; got {raw_pn_unit_iid!r}"
                )

            if af3_chain_id in chain_id_to_pn_unit_iid:
                existing_pn_unit_iid = chain_id_to_pn_unit_iid[af3_chain_id]
                if existing_pn_unit_iid == pn_unit_iid:
                    raise ValueError(
                        "Duplicate AF3 chain ID: "
                        f"pn_unit_iid {pn_unit_iid!r} appears more than once "
                        f"as chain ID {af3_chain_id!r}"
                    )
                raise ValueError(
                    "AF3 chain ID collision: "
                    f"{existing_pn_unit_iid!r} and {pn_unit_iid!r} both serialize "
                    f"as chain ID {af3_chain_id!r}"
                )
            chain_id_to_pn_unit_iid[af3_chain_id] = pn_unit_iid
    return chain_id_to_pn_unit_iid


def make_af3_json(
    af3_ss_input_dir: str = None,
    af3_tc_input_dir: str = None,
    sample_dict: dict = None,
    json_config: dict = None,
    make_ss_input: bool = True,
    make_tc_input: bool = False,
) -> dict:
    """
    Create AF3 JSON input files for single-sequence and template-conditioned inference.

    The input ``sample_dict`` is mutated by adding JSON path lists for the
    requested modes. TC inputs always template exactly one protein PN unit. Role
    metadata selects it explicitly; legacy single-protein inputs remain supported.
    """
    json_config = json_config or {}
    model_seeds = list(json_config.get("model_seeds", [42]))
    version = int(json_config.get("version", 2))

    if not make_ss_input and not make_tc_input:
        raise ValueError("At least one of make_ss_input or make_tc_input must be true")
    if make_ss_input and af3_ss_input_dir is None:
        raise ValueError("af3_ss_input_dir is required when make_ss_input=True")
    if make_tc_input and af3_tc_input_dir is None:
        raise ValueError("af3_tc_input_dir is required when make_tc_input=True")
    if make_ss_input:
        Path(af3_ss_input_dir).mkdir(parents=True, exist_ok=True)
    if make_tc_input:
        Path(af3_tc_input_dir).mkdir(parents=True, exist_ok=True)

    for input_sample_id in tqdm(sample_dict.keys(), desc="Creating AF3 JSONs"):
        if make_ss_input:
            sample_dict[input_sample_id]["af3_ss_json_paths"] = []
        if make_tc_input:
            sample_dict[input_sample_id]["af3_tc_json_paths"] = []
        subsample_dict = sample_dict[input_sample_id]
        pn_unit_roles = subsample_dict.get("pn_unit_roles")
        normalized_roles = (
            normalize_pn_unit_roles(pn_unit_roles)
            if pn_unit_roles is not None
            else None
        )
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
            af3_chain_id_to_pn_unit_iid = build_af3_chain_id_to_pn_unit_iid(
                protein_pn_unit_iids=protein_pn_unit_iids,
                ligand_pn_unit_iids=ligand_pn_unit_iids,
            )
            pn_unit_iid_to_af3_chain_id = {
                pn_unit_iid: chain_id
                for chain_id, pn_unit_iid in af3_chain_id_to_pn_unit_iid.items()
            }
            af3_ligand_ccd_codes = _af3_ligand_ccd_codes_for_json(
                pdb_chain_info=pdb_chain_info,
                ligand_pn_unit_iids=ligand_pn_unit_iids,
                ligand_ccd_codes=ligand_ccd_codes,
            )
            user_ccd_path = _get_user_ccd_path(json_config, pdb_chain_info)
            json_version = max(version, 3) if user_ccd_path else version

            selected_template_pn_unit_iid = None
            if make_tc_input:
                if normalized_roles is None:
                    if len(protein_pn_unit_iids) != 1:
                        raise ValueError(
                            "TC input without pn_unit_roles requires exactly one protein "
                            "PN unit; add roles to select one template from "
                            f"{protein_pn_unit_iids} for {input_sample_id}"
                        )
                    selected_template_pn_unit_iid = protein_pn_unit_iids[0]
                else:
                    template_pn_unit_iids = normalized_roles["template_pn_unit_iids"]
                    if len(template_pn_unit_iids) != 1:
                        raise ValueError(
                            "Role-driven TC input requires exactly one template protein PN unit; "
                            f"got {template_pn_unit_iids} for {input_sample_id}"
                        )
                    selected_template_pn_unit_iid = template_pn_unit_iids[0]
                    if selected_template_pn_unit_iid not in protein_pn_unit_iids:
                        raise ValueError(
                            "template_pn_unit_iids must select a protein PN unit present in "
                            f"pdb_chain_info; got {selected_template_pn_unit_iid!r} for "
                            f"proteins {protein_pn_unit_iids}"
                        )

            ss_sequences = []
            tc_sequences = []
            for protein_pn_unit_iid in protein_pn_unit_iids:
                chain_mask = designed_sample_atom_array.pn_unit_iid == protein_pn_unit_iid
                _res_starts = get_residue_starts(designed_sample_atom_array[chain_mask])
                _res_ids = designed_sample_atom_array[chain_mask].res_id[_res_starts]
                _res_ids_0based = _res_ids - np.min(_res_ids)

                full_length = np.max(_res_ids) - np.min(_res_ids) + 1
                chain_seq_with_gaps = np.full(full_length, "UNK")
                chain_id = pn_unit_iid_to_af3_chain_id[protein_pn_unit_iid]
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

                if make_ss_input:
                    ss_sequences.append(
                        make_af3_protein_sequence_entry(
                            chain_id=chain_id,
                            sequence=sequence_with_gaps,
                            modifications=modifications,
                            templates=[],
                        )
                    )

                if make_tc_input:
                    template_selected = (
                        protein_pn_unit_iid == selected_template_pn_unit_iid
                    )
                    tc_sequences.append(
                        make_af3_protein_sequence_entry(
                            chain_id=chain_id,
                            sequence=sequence_with_gaps,
                            modifications=modifications,
                            templates=(
                                [
                                    {
                                        "mmcifPath": template_sample_path,
                                        "queryIndices": query_indices,
                                        "templateIndices": template_indices,
                                        "templateChainId": chain_id,
                                    }
                                ]
                                if template_selected
                                else []
                            ),
                        )
                    )

            for ligand_pn_unit_iid, ligand_ccd_code in zip(ligand_pn_unit_iids, af3_ligand_ccd_codes):
                chain_id = pn_unit_iid_to_af3_chain_id[ligand_pn_unit_iid]
                if make_ss_input:
                    ss_sequences.append({
                        "ligand": {
                            "id": chain_id,
                            "ccdCodes": [ligand_ccd_code],
                        }
                    })

                if make_tc_input:
                    tc_sequences.append({
                        "ligand": {
                            "id": chain_id,
                            "ccdCodes": [ligand_ccd_code],
                        }
                    })

            if make_ss_input:
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

            if make_ss_input:
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
