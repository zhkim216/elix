from __future__ import annotations

import gc
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from allatom_design.eval.structure_prediction.af3_input_utils import (
    extract_pdb_chain_info,
    load_af3_eval_sample,
    load_sampling_inputs_csv,
)
from allatom_design.eval.structure_prediction.run_tc_af3 import prepare_tc_template_cif
from allatom_design.eval.utils.data_utils import (
    parse_query_pn_unit_iids,
    prepare_af3_prediction,
)
from allatom_design.eval.utils.eval_setup_utils import get_pdb_files, wandb_setup
from allatom_design.eval.utils.folding_utils import (
    find_pred_sample_path_af3,
    make_af3_json,
    run_af3_template_conditioned,
)
from allatom_design.eval.utils.metrics import compute_docking_metrics_atomarray
from allatom_design.eval.utils.selectivity import (
    SELECTIVITY_GUIDANCE_METADATA_KEYS,
    normalize_target_ligand_side,
    resolve_selectivity_guidance_branches,
)


def _sampling_key_rows(sampling_inputs_df: pd.DataFrame) -> dict[str, pd.Series]:
    if "pdb_key" not in sampling_inputs_df.columns:
        raise ValueError("dual-ligand selectivity sampling_inputs_csv requires a pdb_key column")
    return {
        str(row["pdb_key"]): row
        for _, row in sampling_inputs_df.iterrows()
    }


def resolve_input_sample_id(sample_key: str, sampling_inputs_df: pd.DataFrame) -> str | None:
    rows_by_key = _sampling_key_rows(sampling_inputs_df)
    if sample_key in rows_by_key:
        return sample_key

    candidates = [pdb_key for pdb_key in rows_by_key if sample_key.startswith(f"{pdb_key}_")]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        return max(candidates, key=len)
    return None


def filter_selectivity_sample_paths(
    sample_paths: list[str],
    sampling_inputs_df: pd.DataFrame,
) -> list[str]:
    filtered = []
    for sample_path in sample_paths:
        sample_key = Path(sample_path).stem
        if resolve_input_sample_id(sample_key, sampling_inputs_df) is not None:
            filtered.append(sample_path)
    return filtered


def _pn_unit_mask(atom_array, pn_unit_iid: str) -> np.ndarray:
    return np.asarray(atom_array.pn_unit_iid).astype(str) == str(pn_unit_iid)


def _filter_atom_array_and_chain_info(atom_array, query_pn_unit_iids: list[str]):
    query_set = set(map(str, query_pn_unit_iids))
    keep_mask = np.zeros(len(atom_array), dtype=bool)
    for pn_unit_iid in query_set:
        keep_mask |= _pn_unit_mask(atom_array, pn_unit_iid)
    filtered_atom_array = atom_array[keep_mask]

    all_chain_info = extract_pdb_chain_info(filtered_atom_array)
    pdb_chain_info = defaultdict(list)
    for pn_unit_iid in all_chain_info["protein_pn_unit_iids"]:
        if pn_unit_iid in query_set:
            pdb_chain_info["protein_pn_unit_iids"].append(pn_unit_iid)
    for pn_unit_iid, ccd_code in zip(
        all_chain_info["ligand_pn_unit_iids"],
        all_chain_info["ligand_ccd_codes"],
    ):
        if pn_unit_iid in query_set:
            pdb_chain_info["ligand_pn_unit_iids"].append(pn_unit_iid)
            pdb_chain_info["ligand_ccd_codes"].append(ccd_code)
    return filtered_atom_array, pdb_chain_info


def _sampling_row_text(sampling_row: pd.Series, columns: tuple[str, ...]) -> str:
    for column in columns:
        if column not in sampling_row.index:
            continue
        value = sampling_row.get(column, "")
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def ligand_metadata_by_iid(sampling_row: pd.Series) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for ligand_side in (1, 2):
        iid_col = f"ligand_{ligand_side}_pn_unit_iid"
        ccd_col = f"ligand_{ligand_side}_ccd_code"
        original_col = f"ligand_{ligand_side}_original_pn_unit_iid"
        if iid_col not in sampling_row.index:
            continue
        ligand_iid = _sampling_row_text(sampling_row, (iid_col,))
        if not ligand_iid:
            continue
        metadata[ligand_iid] = {
            "ligand_side": ligand_side,
            "ligand_pn_unit_iid": ligand_iid,
            "ligand_ccd_code": _sampling_row_text(sampling_row, (ccd_col,)),
            "original_ligand_pn_unit_iid": _sampling_row_text(sampling_row, (original_col,)),
            "ligand_smiles": _sampling_row_text(
                sampling_row,
                (
                    f"ligand_{ligand_side}_smiles",
                    f"ligand_{ligand_side}_ligand_smiles",
                    f"ligand_smiles_{ligand_side}",
                ),
            ),
        }
    return metadata


def _reference_ligand_iid_for_metrics(
    *,
    ligand_iid: str,
    ligand_meta: dict[str, Any],
    input_sample_is_designed: bool,
) -> str:
    """Return the ligand iid present in the atom array used as metric reference."""
    if input_sample_is_designed:
        return ligand_iid
    return ligand_meta.get("original_ligand_pn_unit_iid") or ligand_iid


def _base_metric_row(
    *,
    input_sample_id: str,
    designed_sample_id: str,
    sampling_row: pd.Series,
    ligand_iid: str,
    ligand_ccd_code: str,
    diffusion_idx: int | None,
    guidance_mode: str | None = None,
    guidance_scale: float | str | None = None,
    guidance_target_ligand_side: int | None = None,
) -> dict[str, Any]:
    ligand_meta = ligand_metadata_by_iid(sampling_row).get(ligand_iid, {})
    branch_metadata = resolve_selectivity_guidance_branches(
        sampling_row,
        target_ligand_side=guidance_target_ligand_side,
        example_id=input_sample_id,
    )
    row = {
        "input_sample_id": input_sample_id,
        "designed_sample_id": designed_sample_id,
        "guidance_mode": guidance_mode or "",
        "guidance_scale": guidance_scale if guidance_scale is not None else "",
        "ligand_side": ligand_meta.get("ligand_side", ""),
        "ligand_pn_unit_iid": ligand_iid,
        "ligand_ccd_code": ligand_meta.get("ligand_ccd_code") or ligand_ccd_code,
        "original_ligand_pn_unit_iid": ligand_meta.get("original_ligand_pn_unit_iid", ""),
        "diffusion_idx": diffusion_idx,
    }
    for key in SELECTIVITY_GUIDANCE_METADATA_KEYS:
        row[key] = branch_metadata.get(key, "")
    return row


def planned_per_ligand_metric_rows(
    *,
    input_sample_id: str,
    designed_sample_id: str,
    sampling_row: pd.Series,
    ligand_pn_unit_iids: list[str],
    ligand_ccd_codes: list[str],
    guidance_mode: str | None = None,
    guidance_scale: float | str | None = None,
    guidance_target_ligand_side: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for ligand_iid, ligand_ccd_code in zip(ligand_pn_unit_iids, ligand_ccd_codes):
        row = _base_metric_row(
            input_sample_id=input_sample_id,
            designed_sample_id=designed_sample_id,
            sampling_row=sampling_row,
            ligand_iid=ligand_iid,
            ligand_ccd_code=ligand_ccd_code,
            diffusion_idx=None,
            guidance_mode=guidance_mode,
            guidance_scale=guidance_scale,
            guidance_target_ligand_side=guidance_target_ligand_side,
        )
        row.update(
            {
                "ligand_rmsd": np.nan,
                "ligand_plddt": np.nan,
                "binding_site_rmsd": np.nan,
                "binding_site_plddt": np.nan,
                "iptm": np.nan,
                "interface_min_pae": np.nan,
                "metric_status": "planned_generate_inputs_only",
            }
        )
        rows.append(row)
    return rows


def compute_per_ligand_docking_metric_rows(
    *,
    pred_atom_array,
    sample_atom_array,
    pred_sample_path: str | Path,
    input_sample_id: str,
    designed_sample_id: str,
    sampling_row: pd.Series,
    diffusion_idx: int,
    receptor_pn_unit_iids: list[str],
    ligand_pn_unit_iids: list[str],
    ligand_ccd_codes: list[str],
    pocket_distance_for_docking_metrics: float,
    input_sample_is_designed: bool,
    guidance_mode: str | None = None,
    guidance_scale: float | str | None = None,
    guidance_target_ligand_side: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for ligand_iid, ligand_ccd_code in zip(ligand_pn_unit_iids, ligand_ccd_codes):
        row = _base_metric_row(
            input_sample_id=input_sample_id,
            designed_sample_id=designed_sample_id,
            sampling_row=sampling_row,
            ligand_iid=ligand_iid,
            ligand_ccd_code=ligand_ccd_code,
            diffusion_idx=diffusion_idx,
            guidance_mode=guidance_mode,
            guidance_scale=guidance_scale,
            guidance_target_ligand_side=guidance_target_ligand_side,
        )
        ligand_meta = ligand_metadata_by_iid(sampling_row).get(ligand_iid, {})
        reference_ligand_iid = _reference_ligand_iid_for_metrics(
            ligand_iid=ligand_iid,
            ligand_meta=ligand_meta,
            input_sample_is_designed=input_sample_is_designed,
        )
        ligand_smiles = ligand_meta.get("ligand_smiles") or ""
        try:
            metrics = compute_docking_metrics_atomarray(
                pred_atom_array=pred_atom_array,
                sample_atom_array=sample_atom_array,
                pred_sample_path=str(pred_sample_path),
                save_aligned=False,
                pocket_distance_for_docking_metrics=pocket_distance_for_docking_metrics,
                receptor_pn_unit_iids=receptor_pn_unit_iids,
                ligand_pn_unit_iids=[ligand_iid],
                ligand_ccd_codes=[ligand_ccd_code],
                ligand_smiles=[ligand_smiles] if ligand_smiles else None,
                reference_ligand_pn_unit_iids=[reference_ligand_iid],
                ref_sample_is_designed=input_sample_is_designed,
            )
        except Exception as exc:
            metrics = {
                "ligand_rmsd": np.nan,
                "ligand_plddt": np.nan,
                "binding_site_rmsd": np.nan,
                "binding_site_plddt": np.nan,
                "error": f"{type(exc).__name__}: {exc}",
            }
        row.update(metrics)
        row["metric_status"] = "ok" if not row.get("error") else "error"
        rows.append(row)
    return rows


def build_sample_dict(
    *,
    sample_paths: list[str],
    sampling_inputs_df: pd.DataFrame,
    cfg: DictConfig,
    log_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.Series]]:
    rows_by_key = _sampling_key_rows(sampling_inputs_df)
    sample_dict: dict[str, dict[str, Any]] = {}
    tc_template_dir = log_dir / "samples_for_af3_tc"
    tc_template_dir.mkdir(parents=True, exist_ok=True)
    cif_save_args = OmegaConf.to_container(cfg.cif_cfg.save, resolve=True) if cfg.cif_cfg.get("save") else {}

    if cfg.input_sample_is_designed:
        cif_parse_cfg = cfg.cif_cfg.parse.designed_samples
        preprocess_cfg_input = cfg.preprocess_cfg.designed_samples
    else:
        cif_parse_cfg = cfg.cif_cfg.parse.native_samples
        preprocess_cfg_input = cfg.preprocess_cfg.native_samples

    for sample_path in tqdm(sample_paths, desc="Loading selectivity samples"):
        sample_id = Path(sample_path).stem
        input_sample_id = resolve_input_sample_id(sample_id, sampling_inputs_df)
        if input_sample_id is None:
            print(f"Skipping {sample_id}: no matching pdb_key in sampling_inputs_csv")
            continue
        sampling_row = rows_by_key[input_sample_id]
        query_iids = parse_query_pn_unit_iids(sampling_row["query_pn_unit_iids"])

        try:
            example = load_af3_eval_sample(
                sample_path=sample_path,
                cif_parse_cfg=cif_parse_cfg,
                preprocess_cfg=preprocess_cfg_input,
                featurizer_cfg=cfg.featurizer_cfg.prepare_designed_samples,
                sample_is_designed=cfg.input_sample_is_designed,
                sampling_inputs_df=None,
            )
            atom_array, pdb_chain_info = _filter_atom_array_and_chain_info(
                example["atom_array"],
                query_iids,
            )
            if len(pdb_chain_info["ligand_pn_unit_iids"]) != 2:
                raise ValueError(
                    f"Expected two ligands after query filtering, got "
                    f"{pdb_chain_info['ligand_pn_unit_iids']}"
                )
            tc_template_path = prepare_tc_template_cif(
                atom_array=atom_array,
                out_path=str(tc_template_dir / f"{sample_id}.cif"),
                cif_save_args=cif_save_args,
            )
        except Exception as exc:
            print(f"Failed to load/prepare {sample_id}: {type(exc).__name__}: {exc}")
            continue

        entry = sample_dict.setdefault(
            input_sample_id,
            {
                "input_sample_path": sample_path,
                "input_sample_id": input_sample_id,
                "designed_sample_id": [],
                "designed_sample_atom_array": [],
                "designed_sample_path_for_af3_tc": [],
                "pdb_chain_info": pdb_chain_info,
            },
        )
        entry["designed_sample_id"].append(sample_id)
        entry["designed_sample_atom_array"].append(atom_array)
        entry["designed_sample_path_for_af3_tc"].append(tc_template_path)

    return sample_dict, rows_by_key


def run_selectivity_dual_ligand_tc_af3(cfg: DictConfig) -> Path:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if cfg.debug:
        cfg.wandb.project = f"debug_{cfg.wandb.project}"
        cfg.exp_name = f"debug_{cfg.exp_name}"

    log_dir = Path(
        wandb_setup(
            base_out_dir=cfg.base_out_dir,
            exp_name=cfg.exp_name,
            cfg_dict=cfg_dict,
            **cfg.wandb,
        )
    )
    with open(log_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f)

    sampling_inputs_df = load_sampling_inputs_csv(cfg.sampling_inputs_csv)
    sample_paths = get_pdb_files(**cfg.pdb_cfg)
    before_filter = len(sample_paths)
    sample_paths = filter_selectivity_sample_paths(sample_paths, sampling_inputs_df)
    print(f"Filtered selectivity sample paths: {before_filter} -> {len(sample_paths)}")
    if cfg.debug:
        sample_paths = sample_paths[: cfg.num_debug_samples]

    sample_dict, rows_by_key = build_sample_dict(
        sample_paths=sample_paths,
        sampling_inputs_df=sampling_inputs_df,
        cfg=cfg,
        log_dir=log_dir,
    )
    if len(sample_dict) == 0:
        raise RuntimeError("No samples were loaded for dual-ligand TC AF3")

    af3_ss_input_dir = log_dir / "af3_ss_inputs"
    af3_tc_input_dir = log_dir / "af3_tc_inputs"
    af3_tc_pred_dir = log_dir / "af3_tc_preds"
    af3_ss_input_dir.mkdir(parents=True, exist_ok=True)
    af3_tc_input_dir.mkdir(parents=True, exist_ok=True)
    af3_tc_pred_dir.mkdir(parents=True, exist_ok=True)

    sample_dict = make_af3_json(
        af3_ss_input_dir=af3_ss_input_dir,
        af3_tc_input_dir=af3_tc_input_dir,
        sample_dict=sample_dict,
        metadata=None,
        json_config=cfg.struct_pred_cfg.af3.json_config,
        make_tc_input=True,
    )

    metric_rows: list[dict[str, Any]] = []
    generate_inputs_only = bool(cfg.get("generate_inputs_only", False))
    guidance_mode = cfg.get("guidance_mode", "")
    guidance_scale = cfg.get("guidance_scale", "")
    guidance_target_ligand_side = normalize_target_ligand_side(
        cfg.get("guidance_target_ligand_side", None)
    )
    for input_sample_id, subsample_dict in tqdm(sample_dict.items(), desc="Dual-ligand TC AF3"):
        sampling_row = rows_by_key[input_sample_id]
        pdb_chain_info = subsample_dict["pdb_chain_info"]
        protein_pn_unit_iids = pdb_chain_info["protein_pn_unit_iids"]
        ligand_pn_unit_iids = pdb_chain_info["ligand_pn_unit_iids"]
        ligand_ccd_codes = pdb_chain_info["ligand_ccd_codes"]

        for dsidx, designed_sample_id in enumerate(subsample_dict["designed_sample_id"]):
            designed_sample_atom_array = subsample_dict["designed_sample_atom_array"][dsidx]
            tc_json_path = subsample_dict["af3_tc_json_paths"][dsidx]

            if generate_inputs_only:
                metric_rows.extend(
                    planned_per_ligand_metric_rows(
                        input_sample_id=input_sample_id,
                        designed_sample_id=designed_sample_id,
                        sampling_row=sampling_row,
                        ligand_pn_unit_iids=ligand_pn_unit_iids,
                        ligand_ccd_codes=ligand_ccd_codes,
                        guidance_mode=guidance_mode,
                        guidance_scale=guidance_scale,
                        guidance_target_ligand_side=guidance_target_ligand_side,
                    )
                )
                continue

            if not cfg.struct_pred_cfg.calculate_metrics_only:
                run_af3_template_conditioned(
                    str(tc_json_path),
                    str(af3_tc_pred_dir),
                    runner_path=cfg.struct_pred_cfg.af3.runner_path,
                    inference_config=cfg.struct_pred_cfg.af3.inference_config,
                )
                gc.collect()

            _, pred_tc_sample_paths = find_pred_sample_path_af3(
                out_dir=str(af3_tc_pred_dir),
                job_name=designed_sample_id,
            )
            if len(pred_tc_sample_paths) == 0:
                print(f"No AF3 TC predicted structure found for {designed_sample_id}")
                continue

            for diffusion_idx, pred_tc_sample_path in enumerate(pred_tc_sample_paths):
                pred_example = prepare_af3_prediction(
                    pdb_path=pred_tc_sample_path,
                    preprocess_cfg=cfg.preprocess_cfg.af3_predictions,
                    featurizer_cfg=cfg.featurizer_cfg.prepare_af3_predictions,
                )
                metric_rows.extend(
                    compute_per_ligand_docking_metric_rows(
                        pred_atom_array=pred_example["atom_array"],
                        sample_atom_array=designed_sample_atom_array,
                        pred_sample_path=pred_tc_sample_path,
                        input_sample_id=input_sample_id,
                        designed_sample_id=designed_sample_id,
                        sampling_row=sampling_row,
                        diffusion_idx=diffusion_idx,
                        receptor_pn_unit_iids=protein_pn_unit_iids,
                        ligand_pn_unit_iids=ligand_pn_unit_iids,
                        ligand_ccd_codes=ligand_ccd_codes,
                        pocket_distance_for_docking_metrics=cfg.pocket_cfg.pocket_distance_for_docking_metrics,
                        input_sample_is_designed=cfg.input_sample_is_designed,
                        guidance_mode=guidance_mode,
                        guidance_scale=guidance_scale,
                        guidance_target_ligand_side=guidance_target_ligand_side,
                    )
                )

    metrics_path = log_dir / "all_dual_ligand_tc_metrics_long.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    print(f"Wrote {len(metric_rows)} per-ligand metric rows -> {metrics_path}")
    print(f"Results saved to {log_dir}")
    return log_dir


@hydra.main(
    config_path="../../configs/eval/structure_prediction",
    config_name="run_tc_eval_af3",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    run_selectivity_dual_ligand_tc_af3(cfg)


if __name__ == "__main__":
    main()
