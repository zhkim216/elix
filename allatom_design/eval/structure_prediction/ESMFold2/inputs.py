"""Input serialization and prediction parsing for ESMFold2."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.constants import STANDARD_AA
from atomworks.enums import ChainType
from atomworks.io.utils.selection import get_residue_starts
from atomworks.io.utils.sequence import get_1_from_3_letter_code
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.data.transform.sd_featurizer import featurizer_af3_prediction
from allatom_design.eval.utils.input_preprocessing import preprocess_input
from allatom_design.eval.utils.sampling_inputs import normalize_pn_unit_roles
from allatom_design.utils.sample_io_utils import load_example_with_parse


INPUT_SCHEMA_VERSION = 1
INPUT_MODE = "single_sequence_no_msa"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _chain_id_to_pn_unit_iid(
    *,
    protein_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for unit_kind, pn_unit_iids in (
        ("protein", protein_pn_unit_iids),
        ("ligand", ligand_pn_unit_iids),
    ):
        for raw_pn_unit_iid in pn_unit_iids:
            if not isinstance(raw_pn_unit_iid, str):
                raise ValueError(
                    f"{unit_kind} pn_unit_iid must be a string, got {raw_pn_unit_iid!r}"
                )
            components = []
            for component in raw_pn_unit_iid.split(","):
                pn_unit_id, separator, transformation_id = component.partition("_")
                if not separator or not pn_unit_id or not transformation_id:
                    raise ValueError(
                        "Malformed pn_unit_iid component; expected "
                        f"'<PN-unit ID>_<transformation ID>', got {component!r}"
                    )
                components.append((pn_unit_id, transformation_id))
            if len({transform_id for _, transform_id in components}) != 1:
                raise ValueError(
                    "All components of a compound pn_unit_iid must have the same "
                    f"transformation ID: {raw_pn_unit_iid!r}"
                )
            chain_id = components[0][0]
            if not chain_id.isalpha() or chain_id.islower():
                raise ValueError(
                    "ESMFold2 serialization requires an uppercase alphabetic "
                    f"chain ID, got {raw_pn_unit_iid!r}"
                )
            if chain_id in mapping:
                raise ValueError(
                    f"ESMFold2 chain ID collision for {chain_id!r}: "
                    f"{mapping[chain_id]!r} and {raw_pn_unit_iid!r}"
                )
            mapping[chain_id] = raw_pn_unit_iid
    return mapping


def _residue_metadata(atom_array: Any, pn_unit_iid: str) -> dict[str, Any]:
    chain = atom_array[atom_array.pn_unit_iid == pn_unit_iid]
    residue_starts = get_residue_starts(chain)
    if len(residue_starts) == 0:
        raise ValueError(f"PN unit {pn_unit_iid!r} has no residues")
    res_ids = np.asarray(chain.res_id[residue_starts], dtype=int)
    if np.any(np.diff(res_ids) <= 0):
        raise ValueError(
            f"Residue IDs must be strictly increasing for {pn_unit_iid!r}: "
            f"{res_ids.tolist()}"
        )
    return {
        "source_res_ids": res_ids.tolist(),
        "source_res_names": [str(value) for value in chain.res_name[residue_starts]],
        "source_hetero": [bool(value) for value in chain.hetero[residue_starts]],
    }


def _protein_sequence(
    *,
    source_res_names: list[str],
    source_hetero: list[bool],
) -> tuple[str, list[dict[str, Any]]]:
    sequence: list[str] = []
    modifications: list[dict[str, Any]] = []
    for position, (res_name, is_hetero) in enumerate(
        zip(source_res_names, source_hetero, strict=True)
    ):
        sequence.append(
            get_1_from_3_letter_code(
                res_name,
                chain_type=ChainType.POLYPEPTIDE_L,
                use_closest_canonical=True,
            )
        )
        if is_hetero and res_name not in STANDARD_AA and res_name != "UNK":
            modifications.append({"position": position, "ccd": res_name})
    return "".join(sequence), modifications


def _reject_unsupported_conditioning(subsample_dict: dict[str, Any]) -> None:
    roles = subsample_dict.get("pn_unit_roles")
    if roles is not None:
        template_iids = normalize_pn_unit_roles(roles)["template_pn_unit_iids"]
        if template_iids:
            raise NotImplementedError(
                "ESMFold2 integration is single-sequence only and does not accept "
                f"template PN units: {template_iids}"
            )
    chain_info = subsample_dict["pdb_chain_info"]
    for key in ("esmfold2_user_ccd_path", "af3_user_ccd_path"):
        if chain_info.get(key):
            raise NotImplementedError(
                f"ESMFold2 custom user CCD input is not supported ({key})"
            )
    for key in ("covalent_bonds", "bonded_atom_pairs", "af3_bonded_atom_pairs"):
        if chain_info.get(key):
            raise NotImplementedError(
                f"ESMFold2 covalent-bond input is not supported ({key})"
            )


def build_esmfold2_input_record(
    *,
    input_sample_id: str,
    designed_sample_id: str,
    designed_sample_atom_array: Any,
    subsample_dict: dict[str, Any],
) -> dict[str, Any]:
    """Build the versioned, no-MSA record consumed by the ESMFold2 runner."""
    if Path(designed_sample_id).name != designed_sample_id:
        raise ValueError(
            f"designed_sample_id must be a filename-safe basename: "
            f"{designed_sample_id!r}"
        )
    _reject_unsupported_conditioning(subsample_dict)
    chain_info = subsample_dict["pdb_chain_info"]
    protein_iids = [str(value) for value in chain_info["protein_pn_unit_iids"]]
    ligand_iids = [str(value) for value in chain_info["ligand_pn_unit_iids"]]
    if not protein_iids:
        raise ValueError("ESMFold2 integration requires at least one protein PN unit")
    chain_mapping = _chain_id_to_pn_unit_iid(
        protein_pn_unit_iids=protein_iids,
        ligand_pn_unit_iids=ligand_iids,
    )
    chain_id_by_iid = {
        pn_unit_iid: chain_id for chain_id, pn_unit_iid in chain_mapping.items()
    }

    ligand_codes_raw = chain_info.get(
        "esmfold2_ligand_ccd_codes",
        chain_info.get(
            "af3_ligand_ccd_codes",
            chain_info.get("ligand_ccd_codes", []),
        ),
    )
    ligand_codes = [str(value).strip() for value in ligand_codes_raw]
    if len(ligand_codes) != len(ligand_iids) or any(not code for code in ligand_codes):
        raise ValueError(
            "ESMFold2 ligand CCD codes must align one-to-one with ligand PN units: "
            f"codes={ligand_codes}, ligand_pn_unit_iids={ligand_iids}"
        )

    sequences: list[dict[str, Any]] = []
    for pn_unit_iid in protein_iids:
        metadata = _residue_metadata(designed_sample_atom_array, pn_unit_iid)
        sequence, modifications = _protein_sequence(
            source_res_names=metadata["source_res_names"],
            source_hetero=metadata["source_hetero"],
        )
        sequences.append(
            {
                "type": "protein",
                "id": chain_id_by_iid[pn_unit_iid],
                "source_pn_unit_iid": pn_unit_iid,
                "sequence": sequence,
                "modifications": modifications,
                "msa": None,
                **metadata,
            }
        )
    for pn_unit_iid, ligand_code in zip(ligand_iids, ligand_codes, strict=True):
        metadata = _residue_metadata(designed_sample_atom_array, pn_unit_iid)
        sequences.append(
            {
                "type": "ligand",
                "id": chain_id_by_iid[pn_unit_iid],
                "source_pn_unit_iid": pn_unit_iid,
                "ccd": [ligand_code],
                **metadata,
            }
        )

    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "mode": INPUT_MODE,
        "name": designed_sample_id,
        "input_sample_id": input_sample_id,
        "chain_id_to_pn_unit_iid": chain_mapping,
        "sequences": sequences,
    }


def write_esmfold2_inputs(
    *,
    sample_dict: dict[str, dict[str, Any]],
    input_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Persist one exact ESMFold2 input record per designed sequence."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    for input_sample_id in tqdm(sample_dict, desc="Creating ESMFold2 inputs"):
        subsample_dict = sample_dict[input_sample_id]
        paths: list[Path] = []
        for index, designed_sample_id in enumerate(
            subsample_dict["designed_sample_id"]
        ):
            record = build_esmfold2_input_record(
                input_sample_id=str(input_sample_id),
                designed_sample_id=str(designed_sample_id),
                designed_sample_atom_array=(
                    subsample_dict["designed_sample_atom_array"][index]
                ),
                subsample_dict=subsample_dict,
            )
            path = input_dir / f"{designed_sample_id}.json"
            _atomic_write_json(path, record)
            paths.append(path)
        subsample_dict["esmfold2_input_paths"] = paths
    return sample_dict


def load_esmfold2_input_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as handle:
        record = json.load(handle)
    if record.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ESMFold2 input schema in {path}: "
            f"{record.get('schema_version')!r}"
        )
    if record.get("mode") != INPUT_MODE:
        raise ValueError(
            f"ESMFold2 input must use {INPUT_MODE!r}, got "
            f"{record.get('mode')!r} in {path}"
        )
    return record


def structure_prediction_input_from_record(
    record: dict[str, Any],
    *,
    use_source_residue_indices: bool = True,
) -> Any:
    """Materialize the Biohub input lazily so base Elix need not import ESM."""
    from esm.models.esmfold2 import (
        LigandInput,
        Modification,
        ProteinInput,
        StructurePredictionInput,
    )

    sequences = []
    for entry in record["sequences"]:
        if entry["type"] == "protein":
            if entry.get("msa") is not None:
                raise ValueError("ESMFold2 integration forbids MSA input")
            sequences.append(
                ProteinInput(
                    id=str(entry["id"]),
                    sequence=str(entry["sequence"]),
                    modifications=[
                        Modification(
                            position=int(modification["position"]),
                            ccd=str(modification["ccd"]),
                        )
                        for modification in entry.get("modifications", [])
                    ],
                    msa=None,
                    source_residue_indices=(
                        [
                            int(residue_id)
                            for residue_id in entry["source_res_ids"]
                        ]
                        if use_source_residue_indices
                        else None
                    ),
                )
            )
        elif entry["type"] == "ligand":
            sequences.append(
                LigandInput(
                    id=str(entry["id"]),
                    ccd=[str(value) for value in entry["ccd"]],
                )
            )
        else:
            raise ValueError(f"Unsupported ESMFold2 sequence type: {entry['type']!r}")
    return StructurePredictionInput(sequences=sequences)


def prepare_esmfold2_prediction(
    *,
    pdb_path: str,
    cif_parse_cfg: DictConfig | dict[str, Any],
    preprocess_cfg: DictConfig | dict[str, Any],
    featurizer_cfg: DictConfig | dict[str, Any],
) -> dict[str, Any]:
    """Parse an ESMFold2 CIF with the production prediction reader."""
    example = load_example_with_parse(pdb_path, cif_parse_cfg)
    example = preprocess_input(
        example=example,
        preprocess_cfg=preprocess_cfg,
        sample_is_designed=True,
    )
    resolved_featurizer_cfg = OmegaConf.to_container(featurizer_cfg, resolve=True)
    return featurizer_af3_prediction(**resolved_featurizer_cfg)(example)
