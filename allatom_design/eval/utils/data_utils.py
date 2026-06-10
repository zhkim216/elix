from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from omegaconf import DictConfig, OmegaConf
from joblib import Parallel, delayed
import copy

from allatom_design.data.datasets.atomworks_sd import sd_collator
from allatom_design.data.transform.preprocess import preprocess_transform, preprocess_transform_designed_samples
from allatom_design.data.transform.sd_featurizer import (
    featurizer_af3_prediction,
    sd_featurizer_for_design,
)
from allatom_design.eval.utils.eval_setup_utils import get_pdb_files
from allatom_design.eval.utils.cfg_utils import guidance_is_enabled
from allatom_design.utils.atom_array_utils import get_res_name_by_chain_res_id
from allatom_design.utils.sample_io_utils import load_example_with_parse
from allatom_design.utils.tensor_utils import to

import atomworks.enums as aw_enums

########################################################
# Sequence design model input preparation utilities
########################################################

def create_sample_dict(
    *,
    sample_paths: list[str] | None = None,
    sample_ids: list[str] | None = None,
    prefix: str = "input",
) -> dict[str, dict[str, str]]:
    """
    Build a sample dictionary keyed by sample ID.

    Each entry contains:
    - `{prefix}_sample_path`: source structure path
    - `{prefix}_sample_id`: sample identifier
    """
    if sample_paths is None:
        sample_paths = []

    if sample_ids is None:
        sample_ids = [Path(sample_path).stem for sample_path in sample_paths]

    sample_dict: dict[str, dict[str, str]] = defaultdict(dict)
    for i, sample_id in enumerate(sample_ids):
        sample_dict[sample_id][f"{prefix}_sample_path"] = sample_paths[i]
        sample_dict[sample_id][f"{prefix}_sample_id"] = sample_ids[i]
    return sample_dict


def prepare_sample_dict(
    cfg: DictConfig | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,
    prefix: str = "input",
) -> dict[str, dict[str, str]]:
    """
    Resolve input structure files from `cfg.pdb_cfg` and package them into `sample_dict`.

    Notes:
    - Uses `get_pdb_files(**cfg.pdb_cfg)` as the canonical source of sample paths.
    - In debug mode, keeps a tiny subset and applies the existing hardcoded debug CIF behavior.
    """
    if cfg is None:
        raise ValueError("cfg must be provided")
    sample_paths = get_pdb_files(**cfg.pdb_cfg)

    if cfg.debug:
        sample_paths = sample_paths[:cfg.num_debug_samples]

    return create_sample_dict(sample_paths=sample_paths, prefix=prefix)

def get_sd_batch(
    pdb_paths: list[str] | None = None,
    *,
    sample_is_designed: bool = False,
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,
    device: str | None = None,
    parallel_pool: Parallel | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Given a list of pdb file paths, return a batch of sequence design model features.
    """
    if pdb_paths is None:
        return {}

    if parallel_pool is None:
        batch_examples = [
            get_sd_example(
                pdb_path=pdb_path,
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                featurizer_cfg=featurizer_cfg,
                sampling_inputs_df=sampling_inputs_df,
                sample_is_designed=sample_is_designed,
            )
            for pdb_path in pdb_paths
        ]
    else:
        batch_examples = parallel_pool(
            delayed(get_sd_example)(
                pdb_path=pdb_path,
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg,
                featurizer_cfg=featurizer_cfg,
                sampling_inputs_df=sampling_inputs_df,
                sample_is_designed=sample_is_designed,
            )
            for pdb_path in pdb_paths
        )

    native_res_name_by_chain_res_id = [
        example.pop("native_res_name_by_chain_res_id")
        for example in batch_examples
    ]

    batch = sd_collator(batch_examples)
    batch["native_res_name_by_chain_res_id"] = native_res_name_by_chain_res_id
    batch = to(batch, device)
    return batch


def get_sd_example(
    pdb_path: str | None = None,
    *,
    sample_is_designed: bool = False,
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Given a pdb file path, return a dictionary of sequence design model features.
    """
    if pdb_path is None:
        raise ValueError("pdb_path must be provided")

    example = load_example_with_parse(pdb_path, cif_parse_cfg)

    example = preprocess_input(
        example=example,
        preprocess_cfg=preprocess_cfg,
        sample_is_designed=sample_is_designed,
    )
    native_res_name_by_chain_res_id = get_res_name_by_chain_res_id(example["atom_array"])

    pdb_id = Path(pdb_path).stem.split("_")[0]
    example["query_pn_unit_iids"] = resolve_query_pn_unit_iids(
        atom_array=example["atom_array"],
        sampling_inputs_df=sampling_inputs_df,
        pdb_id=pdb_id,
    )

    featurizer_cfg = OmegaConf.to_container(featurizer_cfg, resolve=True)
    featurizer = sd_featurizer_for_design(**featurizer_cfg, sample_is_designed=sample_is_designed)

    featurized = featurizer(example)
    featurized["native_res_name_by_chain_res_id"] = native_res_name_by_chain_res_id
    return featurized

def preprocess_input(
    example: dict[str, Any],
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    sample_is_designed: bool = False,
) -> dict[str, Any]:
    """
    Preprocess an already-loaded example using SD preprocess transforms.
    """
    preprocess_cfg = OmegaConf.to_container(preprocess_cfg, resolve=True)

    if sample_is_designed:
        pipeline = preprocess_transform_designed_samples(**preprocess_cfg)
    else:
        pipeline = preprocess_transform(**preprocess_cfg)

    return pipeline(example)

def _parse_query_pn_unit_iids(raw_value: Any) -> list[str]:
    """
    Parse query_pn_unit_iids from a CSV/metadata cell into a normalized list[str].
    """
    if raw_value is None:
        return []

    if isinstance(raw_value, (float, np.floating)) and np.isnan(raw_value):
        return []

    parsed = raw_value
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = stripped

    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()

    if isinstance(parsed, (list, tuple, set)):
        return [str(x) for x in parsed if str(x) != ""]

    return [str(parsed)] if str(parsed) != "" else []


def _matched_sampling_input_row(
    sampling_inputs_df: pd.DataFrame | None,
    pdb_id: str | None,
) -> pd.Series | None:
    if sampling_inputs_df is None or pdb_id is None or "pdb_id" not in sampling_inputs_df.columns:
        return None

    pdb_id_normalized = str(pdb_id).lower()
    matched = sampling_inputs_df[sampling_inputs_df["pdb_id"].astype(str).str.lower() == pdb_id_normalized]
    if matched.empty:
        return None
    return matched.iloc[0]


def _resolve_query_pn_unit_iids_from_sampling_row(row: pd.Series | None) -> list[str]:
    if row is None:
        return []

    for column in ("query_pn_unit_iids", "query_pn_unit_iids_json"):
        if column not in row.index:
            continue
        parsed = _parse_query_pn_unit_iids(row[column])
        if len(parsed) > 0:
            return parsed
    return []


def _metadata_ccd_code(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (float, np.floating)) and np.isnan(raw_value):
        return None

    ccd_code = str(raw_value).strip()
    if ccd_code == "" or ccd_code.lower() == "nan":
        return None
    return ccd_code


def _sampling_ligand_ccd_by_iid(
    sampling_row: pd.Series | None,
    ligand_pn_unit_iids: list[str],
) -> dict[str, str]:
    if sampling_row is None or len(ligand_pn_unit_iids) != 1 or "ccd_code" not in sampling_row.index:
        return {}

    ccd_code = _metadata_ccd_code(sampling_row["ccd_code"])
    if ccd_code is None:
        return {}
    return {ligand_pn_unit_iids[0]: ccd_code}


def resolve_query_pn_unit_iids(
    *,
    atom_array: AtomArray,
    sampling_inputs_df: pd.DataFrame | None = None,
    pdb_id: str | None = None,
) -> list[str]:
    """
    Resolve query pn_unit_iids from sampling_inputs_df if available; otherwise fallback to all unique pn_unit_iid.
    """
    parsed = _resolve_query_pn_unit_iids_from_sampling_row(
        _matched_sampling_input_row(sampling_inputs_df, pdb_id)
    )
    if len(parsed) > 0:
        return parsed

    if "pn_unit_iid" in atom_array.get_annotation_categories():
        return [str(x) for x in np.unique(atom_array.pn_unit_iid).tolist()]

    raise ValueError("pn_unit_iid annotation is required")



########################################################
# Sequence design model output preparation utilities
########################################################
def _build_guidance_row(output: dict, i: int, n: int) -> dict[str, Any]:
    """Build per-sample guidance metrics (U_cond/uncond and their gamma-mixtures)."""
    def _get_field(key):
        return output.get(key, [None] * n)[i]

    gamma_val = output["gamma"][i]
    u_cond_per_res = _get_field("U_cond_per_res")
    u_uncond_per_res = _get_field("U_uncond_per_res")
    u_cond_pocket = _get_field("U_cond_pocket")
    u_uncond_pocket = _get_field("U_uncond_pocket")
    u_cond_pocket_per_res = _get_field("U_cond_pocket_per_res")
    u_uncond_pocket_per_res = _get_field("U_uncond_pocket_per_res")

    def _mix(uc, uu):
        if uc is None or uu is None or gamma_val is None:
            return None
        return gamma_val * uc + (1.0 - gamma_val) * uu

    return {
        "gamma": gamma_val,
        "schedule_label": _get_field("schedule_label"),
        "U_cond": output["U_cond"][i],
        "U_uncond": output["U_uncond"][i],
        "U_mixed": output["U"][i],
        "U_cond_per_res": u_cond_per_res,
        "U_uncond_per_res": u_uncond_per_res,
        "U_mixed_per_res": _mix(u_cond_per_res, u_uncond_per_res),
        "U_cond_pocket": u_cond_pocket,
        "U_uncond_pocket": u_uncond_pocket,
        "U_mixed_pocket": _mix(u_cond_pocket, u_uncond_pocket),
        "U_cond_pocket_per_res": u_cond_pocket_per_res,
        "U_uncond_pocket_per_res": u_uncond_pocket_per_res,
        "U_mixed_pocket_per_res": _mix(u_cond_pocket_per_res, u_uncond_pocket_per_res),
        "N_pocket": _get_field("N_pocket"),
    }


def collect_design_outputs(
    *,
    sample_dict: dict,
    outputs: dict,
    log_dir_per_ckpt: Path,
    csv_suffix: str,
    guidance_cfg: DictConfig | dict | None,
    sampling_inputs_df: pd.DataFrame | None = None,
) -> dict:
    sample_dict_per_ckpt = copy.deepcopy(sample_dict)
    guidance_enabled = guidance_is_enabled(guidance_cfg)

    seq_recovery_metrics_list = []
    guidance_metrics_list = []
    for example_id, output in outputs.items():
        sample_ids = output["designed_sample_id"]
        seq_recovery_metrics = output["seq_recovery_metrics"]

        for i, sample_id in enumerate(sample_ids):
            row = {
                "example_id": example_id,
                "designed_sample_id": sample_id,
                **seq_recovery_metrics[i],
            }
            if guidance_enabled:
                guidance_row = _build_guidance_row(output, i, len(sample_ids))
                row.update(guidance_row)
                guidance_metrics_list.append({
                    "example_id": example_id,
                    "designed_sample_id": sample_id,
                    **guidance_row,
                })
            seq_recovery_metrics_list.append(row)

            metrics_to_print = ", ".join(
                f"{key}: {value:.3f}" for key, value in seq_recovery_metrics[i].items()
            )
            print(f"sample {i} of {example_id}: {metrics_to_print}")

    pd.DataFrame(seq_recovery_metrics_list).to_csv(
        Path(log_dir_per_ckpt, f"seq_recovery_metrics{csv_suffix}.csv"),
        index=False,
    )
    if guidance_enabled and len(guidance_metrics_list) > 0:
        pd.DataFrame(guidance_metrics_list).to_csv(
            Path(log_dir_per_ckpt, f"guidance_metrics{csv_suffix}.csv"),
            index=False,
        )

    for example_id, output in outputs.items():
        sample_dict_per_ckpt[example_id]["designed_sample_id"] = output["designed_sample_id"]
        sample_dict_per_ckpt[example_id]["designed_sample_atom_array"] = output["designed_sample_atom_array"]
        sample_dict_per_ckpt[example_id]["designed_sample_seq"] = output["designed_sample_seq"]
        sample_dict_per_ckpt[example_id]["designed_sample_path"] = output["designed_sample_path"]
        sample_dict_per_ckpt[example_id]["designed_sample_path_for_af3_tc"] = output["designed_sample_path_for_af3_tc"]
        if "native_res_name_by_chain_res_id" in output:
            sample_dict_per_ckpt[example_id]["native_res_name_by_chain_res_id"] = output[
                "native_res_name_by_chain_res_id"
            ]

    for example_id in sample_dict_per_ckpt.keys():
        pdb_chain_info = defaultdict(list)
        previous_chain_info = sample_dict_per_ckpt[example_id].get("pdb_chain_info", {})
        designed_sample_atom_array = sample_dict_per_ckpt[example_id]["designed_sample_atom_array"][0]
        designed_sample_prot_atom_array = designed_sample_atom_array[
            designed_sample_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L
        ]
        designed_sample_ligand_atom_array = designed_sample_atom_array[
            np.isin(designed_sample_atom_array.chain_type, list(aw_enums.ChainTypeInfo.NON_POLYMERS))
        ]
        protein_pn_unit_iids = [
            str(pn_unit_iid) for pn_unit_iid in np.unique(designed_sample_prot_atom_array.pn_unit_iid)
        ]
        ligand_pn_unit_iids = [
            str(pn_unit_iid) for pn_unit_iid in np.unique(designed_sample_ligand_atom_array.pn_unit_iid)
        ]
        ligand_ccd_codes = [
            str(designed_sample_ligand_atom_array[
                designed_sample_ligand_atom_array.pn_unit_iid == pn_unit_iid
            ].res_name[0])
            for pn_unit_iid in ligand_pn_unit_iids
        ]
        ligand_ccd_code_by_iid = dict(zip(ligand_pn_unit_iids, ligand_ccd_codes))

        input_sample_id = sample_dict_per_ckpt[example_id].get("input_sample_id", example_id)
        pdb_id = str(input_sample_id).split("_")[0]
        sampling_row = _matched_sampling_input_row(sampling_inputs_df, pdb_id)
        query_pn_unit_iids = _resolve_query_pn_unit_iids_from_sampling_row(sampling_row)
        if len(query_pn_unit_iids) == 0 and previous_chain_info:
            query_pn_unit_iids = [
                *previous_chain_info.get("protein_pn_unit_iids", []),
                *previous_chain_info.get("ligand_pn_unit_iids", []),
            ]

        if len(query_pn_unit_iids) > 0:
            query_pn_unit_iid_set = set(map(str, query_pn_unit_iids))
            protein_pn_unit_iids = [
                pn_unit_iid for pn_unit_iid in protein_pn_unit_iids
                if pn_unit_iid in query_pn_unit_iid_set
            ]
            ligand_pn_unit_iids = [
                pn_unit_iid for pn_unit_iid in ligand_pn_unit_iids
                if pn_unit_iid in query_pn_unit_iid_set
            ]

        previous_ligand_ccd_by_iid = {
            str(pn_unit_iid): str(ccd_code)
            for pn_unit_iid, ccd_code in zip(
                previous_chain_info.get("ligand_pn_unit_iids", []),
                previous_chain_info.get("ligand_ccd_codes", []),
            )
        }
        sampling_ligand_ccd_by_iid = _sampling_ligand_ccd_by_iid(sampling_row, ligand_pn_unit_iids)

        for pn_unit_iid in protein_pn_unit_iids:
            pdb_chain_info["protein_pn_unit_iids"].append(str(pn_unit_iid))
        for pn_unit_iid in ligand_pn_unit_iids:
            ccd_code = (
                sampling_ligand_ccd_by_iid.get(pn_unit_iid)
                or previous_ligand_ccd_by_iid.get(pn_unit_iid)
                or ligand_ccd_code_by_iid[pn_unit_iid]
            )
            pdb_chain_info["ligand_pn_unit_iids"].append(str(pn_unit_iid))
            pdb_chain_info["ligand_ccd_codes"].append(str(ccd_code))
        if "af3_ligand_ccd_codes" in previous_chain_info:
            previous_af3_ligand_ccd_by_iid = {
                str(pn_unit_iid): str(ccd_code)
                for pn_unit_iid, ccd_code in zip(
                    previous_chain_info.get("ligand_pn_unit_iids", []),
                    previous_chain_info.get("af3_ligand_ccd_codes", []),
                )
            }
            af3_ligand_ccd_codes = [
                previous_af3_ligand_ccd_by_iid[pn_unit_iid]
                for pn_unit_iid in ligand_pn_unit_iids
                if pn_unit_iid in previous_af3_ligand_ccd_by_iid
            ]
            if len(af3_ligand_ccd_codes) == len(ligand_pn_unit_iids):
                pdb_chain_info["af3_ligand_ccd_codes"].extend(af3_ligand_ccd_codes)
        if "af3_user_ccd_path" in previous_chain_info:
            pdb_chain_info["af3_user_ccd_path"] = previous_chain_info["af3_user_ccd_path"]

        sample_dict_per_ckpt[example_id]["pdb_chain_info"] = pdb_chain_info

    return sample_dict_per_ckpt


########################################################
# AF3 prediction input preparation utilities
########################################################
def prepare_af3_prediction(
    pdb_path: str | None = None,
    cif_parse_cfg: DictConfig | dict[str, Any] | None = None,
    preprocess_cfg: DictConfig | dict[str, Any] | None = None,
    featurizer_cfg: DictConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Given a pdb file path from AF3 prediction, return sequence design model features.
    """

    example = load_example_with_parse(pdb_path, cif_parse_cfg)

    example = preprocess_input(
        example=example,
        preprocess_cfg=preprocess_cfg,
        sample_is_designed=True,
    )

    featurizer_cfg = OmegaConf.to_container(featurizer_cfg, resolve=True)
    featurizer = featurizer_af3_prediction(**featurizer_cfg)
    return featurizer(example)

########################################################
# Selectivity experiment input preparation utilities
########################################################
def resolve_selectivity_row(
    *,
    sampling_inputs_df: pd.DataFrame,
    pdb_id: str,
    guidance_direction: int,
) -> dict[str, Any]:
    """Resolve one backbone's selectivity-assay context from the paired CSV.

    The paired CSV schema has columns `pdb_id_{1,2}`, `query_pn_unit_iids_{1,2}`,
    `ccd_code_{1,2}`. A single `pdb_id` may appear at either `_1` (the H-bond-rich
    position) or `_2` (the H-bond-poor position) across rows; this function
    locates it and returns the self/partner pair plus the guidance target.

    Args:
        sampling_inputs_df: DataFrame loaded from the paired selectivity CSV.
        pdb_id: Backbone identifier (case-insensitive).
        guidance_direction: 1 or 2. Selects `ccd_code_{guidance_direction}` as
            the guidance target — independent of which slot the backbone
            occupies. One pass with `guidance_direction=1` designs every
            backbone with the potential pulling toward the H-bond-rich CCD;
            `guidance_direction=2` pulls toward the H-bond-poor CCD.

    Returns:
        dict with keys:
            pdb_id_self, query_pn_unit_iids_self, ccd_self,
            pdb_id_partner, query_pn_unit_iids_partner, ccd_partner,
            guidance_target_ccd, pocket_subcluster_id,
            self_position (1 or 2).

    Raises:
        ValueError: if `guidance_direction` is not in {1, 2}, required columns
            are missing, or `pdb_id` is absent from both slots.
    """
    if guidance_direction not in (1, 2):
        raise ValueError(f"guidance_direction must be 1 or 2, got {guidance_direction}")

    required_cols = {
        "pdb_id_1", "pdb_id_2",
        "query_pn_unit_iids_1", "query_pn_unit_iids_2",
        "ccd_code_1", "ccd_code_2",
    }
    missing = required_cols - set(sampling_inputs_df.columns)
    if missing:
        raise ValueError(f"sampling_inputs_df missing columns: {sorted(missing)}")

    pdb_lc = str(pdb_id).lower()
    for self_pos in (1, 2):
        other_pos = 3 - self_pos
        hit = sampling_inputs_df[
            sampling_inputs_df[f"pdb_id_{self_pos}"].astype(str).str.lower() == pdb_lc
        ]
        if not hit.empty:
            row = hit.iloc[0]
            out = {
                "pdb_id_self": str(row[f"pdb_id_{self_pos}"]),
                "query_pn_unit_iids_self":
                    parse_query_pn_unit_iids(row[f"query_pn_unit_iids_{self_pos}"]),
                "ccd_self": str(row[f"ccd_code_{self_pos}"]),
                "pdb_id_partner": str(row[f"pdb_id_{other_pos}"]),
                "query_pn_unit_iids_partner":
                    parse_query_pn_unit_iids(row[f"query_pn_unit_iids_{other_pos}"]),
                "ccd_partner": str(row[f"ccd_code_{other_pos}"]),
                "guidance_target_ccd": str(row[f"ccd_code_{guidance_direction}"]),
                "self_position": self_pos,
            }
            if "pocket_subcluster_id" in sampling_inputs_df.columns:
                out["pocket_subcluster_id"] = int(row["pocket_subcluster_id"])
            return out

    raise ValueError(
        f"pdb_id={pdb_id} not found in either pdb_id_1 or pdb_id_2 column of "
        f"sampling_inputs_df (rows={len(sampling_inputs_df)})"
    )
