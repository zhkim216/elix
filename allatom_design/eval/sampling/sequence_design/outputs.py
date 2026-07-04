from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

import atomworks.enums as aw_enums
import numpy as np
import pandas as pd
import torch
from atomworks.io.utils import non_rcsb
from omegaconf import DictConfig

from allatom_design.utils.sample_io_utils import save_cif_file
from allatom_design.utils.atom_array_utils import (
    clean_up_and_renumber_atom_array,
    insert_unk_residues_for_gaps_in_atom_array,
)
from allatom_design.eval.sampling.sequence_design.guidance import (
    POTTS_GUIDANCE_AUX_KEYS,
    POTTS_GUIDANCE_OUTPUT_KEYS,
    guidance_is_enabled,
)
from allatom_design.eval.selectivity import SELECTIVITY_GUIDANCE_METADATA_KEYS
from allatom_design.eval.metrics.sequence_recovery import (
    SequenceRecoveryMetricConfig,
    calculate_sequence_recovery,
)
from allatom_design.eval.utils.sampling_inputs import (
    matched_sampling_input_row,
    resolve_query_pn_unit_iids_from_sampling_row,
    sampling_ligand_ccd_by_iid as ligand_ccd_by_iid_from_sampling_row,
)


def _format_designed_sample_id(
    *,
    example_id: str,
    sample_idx: int,
    tag: str | None = None,
    sample_token_prefix: str | None = None,
) -> str:
    sample_token = f"{sample_token_prefix}sample{sample_idx}" if sample_token_prefix else f"sample{sample_idx}"
    if tag is None:
        return f"{example_id}_{sample_token}"
    return f"{example_id}_{tag}_{sample_token}"


def append_sampling_aux_output(
    *,
    output: dict[str, list],
    example_id: str,
    sample_idx: int,
    aux_entry: dict[str, Any],
    tag_counter: dict[str, int],
    sample_token_prefix: str | None,
    selectivity_metadata: dict[str, Any],
) -> str:
    """Append sampling-energy metadata and return the designed sample ID."""
    gamma_val = aux_entry.get("gamma") if isinstance(aux_entry, dict) else None
    guidance_scale_val = aux_entry.get("guidance_scale") if isinstance(aux_entry, dict) else gamma_val
    guidance_mode_val = aux_entry.get("guidance_mode") if isinstance(aux_entry, dict) else None
    schedule_label_val = aux_entry.get("schedule_label") if isinstance(aux_entry, dict) else None

    if schedule_label_val is not None and not str(schedule_label_val).startswith("gamma_"):
        tag = str(schedule_label_val)
        sub_sample_idx = tag_counter[tag]
        tag_counter[tag] += 1
        designed_sample_id = _format_designed_sample_id(
            example_id=example_id,
            sample_idx=sub_sample_idx,
            tag=tag,
            sample_token_prefix=sample_token_prefix,
        )
    elif guidance_scale_val is not None:
        tag_name = "guidance_scale" if guidance_mode_val == "selectivity" else "gamma"
        tag = f"{tag_name}{guidance_scale_val:.2f}"
        sub_sample_idx = tag_counter[tag]
        tag_counter[tag] += 1
        designed_sample_id = _format_designed_sample_id(
            example_id=example_id,
            sample_idx=sub_sample_idx,
            tag=tag,
            sample_token_prefix=sample_token_prefix,
        )
    else:
        designed_sample_id = _format_designed_sample_id(
            example_id=example_id,
            sample_idx=sample_idx,
            sample_token_prefix=sample_token_prefix,
        )

    output["designed_sample_id"].append(designed_sample_id)
    output["U"].append(aux_entry["U"])
    output["gamma"].append(gamma_val)
    output["schedule_label"].append(schedule_label_val)
    for guidance_key in POTTS_GUIDANCE_AUX_KEYS:
        if guidance_mode_val == "selectivity" and guidance_key in selectivity_metadata:
            output[guidance_key].append(selectivity_metadata.get(guidance_key))
            continue
        output[guidance_key].append(aux_entry.get(guidance_key) if isinstance(aux_entry, dict) else None)
    for guidance_key in SELECTIVITY_GUIDANCE_METADATA_KEYS:
        if guidance_key in POTTS_GUIDANCE_AUX_KEYS:
            continue
        output[guidance_key].append(selectivity_metadata.get(guidance_key))

    return designed_sample_id


def _build_guidance_row(output: dict, i: int, n: int) -> dict[str, Any]:
    """Build per-sample guidance metrics for branch-mixed Potts sampling."""
    def _get_field(key):
        return output.get(key, [None] * n)[i]

    gamma_val = output["gamma"][i]
    guidance_scale = _get_field("guidance_scale")
    scale_val = guidance_scale if guidance_scale is not None else gamma_val
    u_cond_per_res = _get_field("U_cond_per_res")
    u_uncond_per_res = _get_field("U_uncond_per_res")
    u_cond_pocket = _get_field("U_cond_pocket")
    u_uncond_pocket = _get_field("U_uncond_pocket")
    u_cond_pocket_per_res = _get_field("U_cond_pocket_per_res")
    u_uncond_pocket_per_res = _get_field("U_uncond_pocket_per_res")

    def _mix(uc, uu):
        if uc is None or uu is None or scale_val is None:
            return None
        return scale_val * uc + (1.0 - scale_val) * uu

    row = {
        "guidance_mode": _get_field("guidance_mode"),
        "guidance_scale": guidance_scale,
        "positive_branch_label": _get_field("positive_branch_label"),
        "negative_branch_label": _get_field("negative_branch_label"),
        "gamma": gamma_val,
        "schedule_label": _get_field("schedule_label"),
        "U_positive": _get_field("U_positive"),
        "U_negative": _get_field("U_negative"),
        "U_positive_per_res": _get_field("U_positive_per_res"),
        "U_negative_per_res": _get_field("U_negative_per_res"),
        "U_positive_pocket": _get_field("U_positive_pocket"),
        "U_negative_pocket": _get_field("U_negative_pocket"),
        "U_positive_pocket_per_res": _get_field("U_positive_pocket_per_res"),
        "U_negative_pocket_per_res": _get_field("U_negative_pocket_per_res"),
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
    for key in SELECTIVITY_GUIDANCE_METADATA_KEYS:
        row[key] = _get_field(key)
    return row


def record_designed_sample_output(
    *,
    output: dict[str, list],
    designed_sample_id: str,
    designed_atom_array,
    input_atom_array,
    native_res_name_by_chain_res_id: dict,
    sample_out_dir: Path,
    sample_out_dir_for_af3_tc: Path,
    cif_save_cfg: DictConfig | dict | None,
    sequence_recovery_metric_config: SequenceRecoveryMetricConfig,
) -> None:
    """Save one designed sample and append its output metadata."""
    chain_info = non_rcsb.initialize_chain_info_from_atom_array(designed_atom_array)
    output["designed_sample_seq"].append(
        ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
    )

    designed_atom_array = clean_up_and_renumber_atom_array(designed_atom_array)
    output["designed_sample_atom_array"].append(designed_atom_array)

    designed_atom_array_with_gaps = designed_atom_array.copy()
    designed_atom_array_with_gaps = insert_unk_residues_for_gaps_in_atom_array(
        designed_atom_array_with_gaps,
        missing_res_name_by_chain_res_id=native_res_name_by_chain_res_id,
    )

    out_file = sample_out_dir / f"{designed_sample_id}.cif"
    save_cif_file(designed_atom_array, out_file, cif_save_cfg=cif_save_cfg)
    output["designed_sample_path"].append(str(out_file))

    out_file_for_af3_tc = sample_out_dir_for_af3_tc / f"{designed_sample_id}.cif"
    save_cif_file(designed_atom_array_with_gaps, out_file_for_af3_tc, cif_save_cfg=cif_save_cfg)
    output["designed_sample_path_for_af3_tc"].append(str(out_file_for_af3_tc))

    if sequence_recovery_metric_config.enabled:
        seq_recovery_metrics = calculate_sequence_recovery(
            input_atom_array,
            designed_atom_array,
            pocket_distances_for_seq_recovery=(
                sequence_recovery_metric_config.pocket_distances_for_seq_recovery
            ),
            pocket_distance_bins=sequence_recovery_metric_config.pocket_distance_bins,
            n_min_ligand_atoms=sequence_recovery_metric_config.n_min_ligand_atoms,
            pocket_annotation_method=sequence_recovery_metric_config.pocket_annotation_method,
        )
        output["seq_recovery_metrics"].append(seq_recovery_metrics)


def record_sampled_batch_outputs(
    *,
    outputs: dict[str, dict[str, Any]],
    batch: dict[str, Any],
    id_to_atom_arrays: dict[str, list],
    id_to_aux: dict[str, list[dict[str, Any]]],
    guidance_metadata_by_example_id: dict[str, dict[str, Any]],
    sample_token_prefix: str | None,
    sample_out_dir: Path,
    sample_out_dir_for_af3_tc: Path,
    cif_save_cfg: DictConfig | dict | None,
    sequence_recovery_metric_config: SequenceRecoveryMetricConfig,
) -> None:
    """Record designed atom arrays and metadata produced for one runtime batch."""
    example_id_to_batch_idx = {eid: idx for idx, eid in enumerate(batch["example_id"])}

    for sample_idx, (example_id, atom_arrays) in enumerate(id_to_atom_arrays.items()):
        if example_id not in outputs:
            outputs[example_id] = defaultdict(list)
        output = outputs[example_id]
        aux = id_to_aux[example_id]
        batch_idx = example_id_to_batch_idx[example_id]
        input_atom_array = batch["atom_array"][batch_idx]
        native_res_name_by_chain_res_id = batch["native_res_name_by_chain_res_id"][batch_idx]
        if "reference_sample_atom_array" not in output:
            output["reference_sample_atom_array"] = input_atom_array.copy()
        if "sequence_recovery_metric_config" not in output:
            output["sequence_recovery_metric_config"] = sequence_recovery_metric_config
        output["native_res_name_by_chain_res_id"] = native_res_name_by_chain_res_id

        # Per-(schedule, gamma) counter so that tagged sample ids reset
        # within each schedule run (e.g. ramp_up_t0.5_sample0,
        # ramp_up_t0.5_sample1, ... and gamma0.50_sample0, ...).
        tag_counter: dict[str, int] = defaultdict(int)

        for ai, designed_atom_array in enumerate(atom_arrays):
            selectivity_metadata = guidance_metadata_by_example_id.get(example_id, {})
            designed_sample_id = append_sampling_aux_output(
                output=output,
                example_id=example_id,
                sample_idx=ai,
                aux_entry=aux[ai],
                tag_counter=tag_counter,
                sample_token_prefix=sample_token_prefix,
                selectivity_metadata=selectivity_metadata,
            )

            record_designed_sample_output(
                output=output,
                designed_sample_id=designed_sample_id,
                designed_atom_array=designed_atom_array,
                input_atom_array=input_atom_array,
                native_res_name_by_chain_res_id=native_res_name_by_chain_res_id,
                sample_out_dir=sample_out_dir,
                sample_out_dir_for_af3_tc=sample_out_dir_for_af3_tc,
                cif_save_cfg=cif_save_cfg,
                sequence_recovery_metric_config=sequence_recovery_metric_config,
            )


def convert_output_tensors_to_python(outputs: dict[str, dict[str, Any]]) -> None:
    for example_outputs in outputs.values():
        for key, value in example_outputs.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                example_outputs[key] = [tensor.detach().cpu().item() for tensor in value]
            elif isinstance(value, torch.Tensor):
                example_outputs[key] = value.detach().cpu().item()


def write_sample_metadata(
    *,
    outputs: dict[str, dict[str, Any]],
    sample_out_dir: Path,
    csv_suffix: str,
) -> Path:
    sample_metadata = {}
    for example_id, example_outputs in outputs.items():
        for idx, designed_sample_id in enumerate(example_outputs["designed_sample_id"]):
            meta_entry = {
                "example_id": example_id,
                "designed_sample_id": designed_sample_id,
                "designed_sample_path": example_outputs["designed_sample_path"][idx],
                "designed_sample_seq": example_outputs["designed_sample_seq"][idx],
                "U": example_outputs["U"][idx],
            }
            if "gamma" in example_outputs:
                meta_entry["gamma"] = example_outputs["gamma"][idx]
            for guidance_key in POTTS_GUIDANCE_OUTPUT_KEYS:
                if guidance_key in example_outputs:
                    meta_entry[guidance_key] = example_outputs[guidance_key][idx]
            sample_metadata[designed_sample_id] = meta_entry

    sample_metadata_path = sample_out_dir / f"sample_metadata{csv_suffix}.pt"
    torch.save(sample_metadata, sample_metadata_path)
    print(f"Saved sample_metadata{csv_suffix}.pt with {len(sample_metadata)} samples to {sample_metadata_path}")
    return sample_metadata_path


def _designed_sample_chain_components(designed_sample_atom_array) -> tuple[list[str], list[str], dict[str, str]]:
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
    return protein_pn_unit_iids, ligand_pn_unit_iids, ligand_ccd_code_by_iid


def _previous_ligand_ccd_by_iid(previous_chain_info: dict) -> dict[str, str]:
    return {
        str(pn_unit_iid): str(ccd_code)
        for pn_unit_iid, ccd_code in zip(
            previous_chain_info.get("ligand_pn_unit_iids", []),
            previous_chain_info.get("ligand_ccd_codes", []),
        )
    }


def _filter_query_pn_unit_iids(
    *,
    protein_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
    query_pn_unit_iids: list[str],
) -> tuple[list[str], list[str]]:
    if len(query_pn_unit_iids) == 0:
        return protein_pn_unit_iids, ligand_pn_unit_iids

    query_pn_unit_iid_set = set(map(str, query_pn_unit_iids))
    query_protein_pn_unit_iids = [
        pn_unit_iid
        for pn_unit_iid in protein_pn_unit_iids
        if pn_unit_iid in query_pn_unit_iid_set
    ]
    query_ligand_pn_unit_iids = [
        pn_unit_iid
        for pn_unit_iid in ligand_pn_unit_iids
        if pn_unit_iid in query_pn_unit_iid_set
    ]
    if query_protein_pn_unit_iids:
        protein_pn_unit_iids = query_protein_pn_unit_iids
    if query_ligand_pn_unit_iids:
        ligand_pn_unit_iids = query_ligand_pn_unit_iids
    return protein_pn_unit_iids, ligand_pn_unit_iids


def _build_designed_sample_pdb_chain_info(
    *,
    designed_sample_atom_array,
    previous_chain_info: dict,
    input_sample_id: str,
    sampling_inputs_df: pd.DataFrame | None,
) -> dict:
    pdb_chain_info = defaultdict(list)
    protein_pn_unit_iids, ligand_pn_unit_iids, ligand_ccd_code_by_iid = (
        _designed_sample_chain_components(designed_sample_atom_array)
    )

    pdb_id = str(input_sample_id).split("_")[0]
    sampling_row = matched_sampling_input_row(
        sampling_inputs_df,
        pdb_id,
        pdb_key=input_sample_id,
        sample_id=input_sample_id,
    )
    query_pn_unit_iids = resolve_query_pn_unit_iids_from_sampling_row(sampling_row)
    if len(query_pn_unit_iids) == 0 and previous_chain_info:
        query_pn_unit_iids = [
            *previous_chain_info.get("protein_pn_unit_iids", []),
            *previous_chain_info.get("ligand_pn_unit_iids", []),
        ]
    protein_pn_unit_iids, ligand_pn_unit_iids = _filter_query_pn_unit_iids(
        protein_pn_unit_iids=protein_pn_unit_iids,
        ligand_pn_unit_iids=ligand_pn_unit_iids,
        query_pn_unit_iids=query_pn_unit_iids,
    )

    previous_ligand_ccd_by_iid = _previous_ligand_ccd_by_iid(previous_chain_info)
    sampling_ligand_ccd_by_iid = ligand_ccd_by_iid_from_sampling_row(sampling_row, ligand_pn_unit_iids)

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

    return pdb_chain_info


def _build_design_metric_rows(
    *,
    outputs: dict,
    guidance_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seq_recovery_metrics_list = []
    guidance_metrics_list = []
    for example_id, output in outputs.items():
        sample_ids = output["designed_sample_id"]
        seq_recovery_metrics = output.get("seq_recovery_metrics")

        for i, sample_id in enumerate(sample_ids):
            row = {"example_id": example_id, "designed_sample_id": sample_id}
            has_seq_metrics = seq_recovery_metrics is not None and i < len(seq_recovery_metrics)
            if has_seq_metrics:
                row.update(seq_recovery_metrics[i])
            if guidance_enabled:
                guidance_row = _build_guidance_row(output, i, len(sample_ids))
                guidance_metrics_list.append({
                    "example_id": example_id,
                    "designed_sample_id": sample_id,
                    **guidance_row,
                })
                if has_seq_metrics:
                    row.update(guidance_row)
            if has_seq_metrics:
                seq_recovery_metrics_list.append(row)

                metrics_to_print = ", ".join(
                    f"{key}: {value:.3f}" for key, value in seq_recovery_metrics[i].items()
                )
                print(f"sample {i} of {example_id}: {metrics_to_print}")

    return seq_recovery_metrics_list, guidance_metrics_list


def _write_design_metric_tables(
    *,
    outputs: dict,
    log_dir_per_ckpt: Path,
    csv_suffix: str,
    guidance_enabled: bool,
) -> None:
    seq_recovery_metrics_list, guidance_metrics_list = _build_design_metric_rows(
        outputs=outputs,
        guidance_enabled=guidance_enabled,
    )

    if len(seq_recovery_metrics_list) > 0:
        pd.DataFrame(seq_recovery_metrics_list).to_csv(
            Path(log_dir_per_ckpt, f"seq_recovery_metrics{csv_suffix}.csv"),
            index=False,
        )
    if guidance_enabled and len(guidance_metrics_list) > 0:
        pd.DataFrame(guidance_metrics_list).to_csv(
            Path(log_dir_per_ckpt, f"guidance_metrics{csv_suffix}.csv"),
            index=False,
        )


def _attach_designed_sample_outputs(
    *,
    sample_dict_per_ckpt: dict,
    outputs: dict,
) -> None:
    for example_id, output in outputs.items():
        sample_entry = sample_dict_per_ckpt[example_id]
        sample_entry["designed_sample_id"] = output["designed_sample_id"]
        sample_entry["designed_sample_atom_array"] = output["designed_sample_atom_array"]
        sample_entry["designed_sample_seq"] = output["designed_sample_seq"]
        sample_entry["designed_sample_path"] = output["designed_sample_path"]
        sample_entry["designed_sample_path_for_af3_tc"] = output["designed_sample_path_for_af3_tc"]
        if "reference_sample_atom_array" in output:
            sample_entry["reference_sample_atom_array"] = output["reference_sample_atom_array"]
        if "sequence_recovery_metric_config" in output:
            sample_entry["sequence_recovery_metric_config"] = output["sequence_recovery_metric_config"]
        if (
            "binding_site_plddt_reference_atom_array" not in sample_entry
            and "reference_sample_atom_array" in output
        ):
            sample_entry["binding_site_plddt_reference_atom_array"] = output["reference_sample_atom_array"]
        if (
            "binding_site_plddt_metric_config" not in sample_entry
            and "sequence_recovery_metric_config" in output
            and output["sequence_recovery_metric_config"].enabled
        ):
            sample_entry["binding_site_plddt_metric_config"] = output["sequence_recovery_metric_config"]
        if "native_res_name_by_chain_res_id" in output:
            sample_entry["native_res_name_by_chain_res_id"] = output[
                "native_res_name_by_chain_res_id"
            ]


def _attach_designed_sample_chain_info(
    *,
    sample_dict_per_ckpt: dict,
    sampling_inputs_df: pd.DataFrame | None,
) -> None:
    for example_id in sample_dict_per_ckpt.keys():
        sample_entry = sample_dict_per_ckpt[example_id]
        previous_chain_info = sample_entry.get("pdb_chain_info", {})
        designed_sample_atom_array = sample_entry["designed_sample_atom_array"][0]
        input_sample_id = sample_entry.get("input_sample_id", example_id)
        sample_entry["pdb_chain_info"] = _build_designed_sample_pdb_chain_info(
            designed_sample_atom_array=designed_sample_atom_array,
            previous_chain_info=previous_chain_info,
            input_sample_id=input_sample_id,
            sampling_inputs_df=sampling_inputs_df,
        )


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

    _write_design_metric_tables(
        outputs=outputs,
        log_dir_per_ckpt=log_dir_per_ckpt,
        csv_suffix=csv_suffix,
        guidance_enabled=guidance_enabled,
    )
    _attach_designed_sample_outputs(
        sample_dict_per_ckpt=sample_dict_per_ckpt,
        outputs=outputs,
    )
    _attach_designed_sample_chain_info(
        sample_dict_per_ckpt=sample_dict_per_ckpt,
        sampling_inputs_df=sampling_inputs_df,
    )

    return sample_dict_per_ckpt
