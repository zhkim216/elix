import lightning as L
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import torch
from typing import Any, Iterator
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import gc

from atomworks.io.utils import non_rcsb
from allatom_design.utils.sample_io_utils import save_cif_file
from allatom_design.utils.atom_array_utils import insert_unk_residues_for_gaps_in_atom_array, clean_up_and_renumber_atom_array

from allatom_design.eval.utils.cfg_utils import require_cfg_value, resolve_sampling_cfg
from allatom_design.eval.utils.eval_setup_utils import get_checkpoints, load_seq_des_model, _parallel_context
from allatom_design.eval.utils.mask_utils import initialize_sampling_masks
from allatom_design.eval.utils.constraint_utils import parse_fixed_pos_info, parse_pos_restrict_aatype_info
from allatom_design.eval.utils.data_utils import get_sd_batch, collect_design_outputs
from allatom_design.eval.utils.sequence_recovery import calculate_sequence_recovery
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


def design_sequence(
    *,
    model: SeqDenoiser = None,
    input_sample_is_designed: bool = False,
    cif_parse_cfg: DictConfig = None,
    preprocess_cfg: DictConfig = None,
    featurizer_cfg: DictConfig = None,
    cif_save_cfg: DictConfig = None,
    sampling_cfg: DictConfig = None,
    sampling_inputs_df: pd.DataFrame = None,
    pdb_paths: list[str] = None,
    pdb_cfg: DictConfig | None = None,
    device: str = None,
    out_dir: str = None,
    pos_constraint_df: pd.DataFrame | None = None,
    protein_only: bool = False,
    pocket_distances_for_seq_recovery: list[float] = None,
    pocket_distance_bins: list[tuple[float, float]] | None = None,
    pocket_n_min_ligand_atoms_for_seq_recovery: int = 5,
    guidance_cfg: DictConfig | None = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """
    Given a list of processed structure files, run sequence design on them.

    If out_dir is not None, PDBs with sampled sequences will be saved to the provided directory. In this case, run_aux
    will be a dictionary with the following keys:
        - "out_pdb": list of output PDB paths
        - "pred_seqs": list of predicted sequences as a string for each sample

        Args:
        ...
        protein_only: If True, condition only on protein atoms (exclude ligands from atom_cond_mask).
    """
    # Set up outputs.
    outputs = {}

    # directory for output PDBs
    sample_out_dir = f"{out_dir}/samples"
    Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
    sample_out_dir_for_af3_tc = f"{out_dir}/samples_for_af3_tc"
    Path(sample_out_dir_for_af3_tc).mkdir(parents=True, exist_ok=True)

    # Validate pos_constraint_df.
    if pos_constraint_df is not None:
        valid_columns = ["pdb_key", "fixed_pos_seq", "fixed_pos_scn", "fixed_pos_override_seq", "pos_restrict_aatype"]
        if not set(pos_constraint_df.columns).issubset(valid_columns):
            # Columns in input df must be a subset of valid columns.
            print(f"Invalid columns in pos_constraint_df. Expected subset of {valid_columns}. Found: {pos_constraint_df.columns}")
            cols_to_keep = [c for c in valid_columns if c in pos_constraint_df.columns]
            print(f"Keeping columns: {cols_to_keep}")
            pos_constraint_df = pos_constraint_df[cols_to_keep]

        # Set index to pdb name.
        pos_constraint_df = pos_constraint_df.set_index("pdb_key")

        # Set empty string to NaN for easier parsing.
        pos_constraint_df = pos_constraint_df.replace("", np.nan)

    # Print omitted amino acids.
    if sampling_cfg.verbose and sampling_cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {sampling_cfg.omit_aas}")

    # Process PDBs in parallel.
    parallel_context = _parallel_context(sampling_cfg.num_workers)

    # Begin sampling.
    pbar = tqdm(
        total=len(pdb_paths),
        desc=f"Sampling {len(pdb_paths)} PDBs, {sampling_cfg.num_seqs_per_pdb} sequences per PDB...",
    )

    with parallel_context as parallel_pool:
        for bi in range(0, len(pdb_paths), sampling_cfg.batch_size):
            batch_pdb_paths = pdb_paths[bi : bi + sampling_cfg.batch_size]
            B = len(batch_pdb_paths)

            batch = get_sd_batch(pdb_paths = batch_pdb_paths,
                                 sample_is_designed = input_sample_is_designed,
                                 cif_parse_cfg = cif_parse_cfg,
                                 preprocess_cfg = preprocess_cfg,
                                 featurizer_cfg = featurizer_cfg,
                                 device=device,
                                 parallel_pool=parallel_pool,
                                 sampling_inputs_df=sampling_inputs_df)

            # Initialize seq_cond and atom_cond masks.
            batch = initialize_sampling_masks(batch, protein_only=protein_only)

            # Parse fixed positions.
            batch = parse_fixed_pos_info(batch, pos_constraint_df, verbose=sampling_cfg.verbose)

            # Restrict aatype sampling at certain positions.
            sampling_inputs = OmegaConf.to_container(sampling_cfg, resolve=True)
            sampling_inputs["pos_restrict_aatype"] = parse_pos_restrict_aatype_info(
                batch, pos_constraint_df, verbose=sampling_cfg.verbose
            )

            # Inject guidance_cfg into potts_sampling_cfg so it reaches
            # ElixMPNNDenoiser.potts_sample. A None / disabled guidance_cfg
            # is a no-op (denoiser falls back to the cond-only path).
            if guidance_cfg is not None:
                guidance_cfg_dict = (
                    OmegaConf.to_container(guidance_cfg, resolve=True)
                    if isinstance(guidance_cfg, DictConfig)
                    else dict(guidance_cfg)
                )
                sampling_inputs.setdefault("potts_sampling_cfg", {})
                sampling_inputs["potts_sampling_cfg"]["guidance_cfg"] = guidance_cfg_dict

            # Run sampling.
            id_to_atom_arrays, id_to_aux = model.sample(batch, sampling_inputs=sampling_inputs)

            # Save outputs.
            example_id_to_batch_idx = {eid: idx for idx, eid in enumerate(batch["example_id"])}

            for sample_idx, (example_id, atom_arrays) in enumerate(id_to_atom_arrays.items()):
                if example_id not in outputs:
                    outputs[example_id] = defaultdict(list)
                aux = id_to_aux[example_id]

                # Per-(schedule, gamma) counter so that tagged sample ids reset
                # within each schedule run (e.g. ramp_up_t0.5_sample0,
                # ramp_up_t0.5_sample1, ... and gamma0.50_sample0, ...).
                tag_counter: dict[str, int] = defaultdict(int)

                for ai, designed_atom_array in enumerate(atom_arrays):
                    gamma_val = aux[ai].get("gamma") if isinstance(aux[ai], dict) else None
                    schedule_label_val = aux[ai].get("schedule_label") if isinstance(aux[ai], dict) else None
                    if schedule_label_val is not None and not str(schedule_label_val).startswith("gamma_"):
                        tag = str(schedule_label_val)
                        sub_ai = tag_counter[tag]
                        tag_counter[tag] += 1
                        designed_sample_id = f"{example_id}_{tag}_sample{sub_ai}"
                    elif gamma_val is not None:
                        tag = f"gamma{gamma_val:.2f}"
                        sub_ai = tag_counter[tag]
                        tag_counter[tag] += 1
                        designed_sample_id = f"{example_id}_{tag}_sample{sub_ai}"
                    else:
                        designed_sample_id = f"{example_id}_sample{ai}"
                    outputs[example_id]["designed_sample_id"].append(designed_sample_id)
                    outputs[example_id]["U"].append(aux[ai]["U"])
                    outputs[example_id]["gamma"].append(gamma_val)
                    outputs[example_id]["schedule_label"].append(schedule_label_val)
                    for guidance_key in (
                        "U_cond",
                        "U_uncond",
                        "U_cond_per_res",
                        "U_uncond_per_res",
                        "U_cond_pocket",
                        "U_uncond_pocket",
                        "U_cond_pocket_per_res",
                        "U_uncond_pocket_per_res",
                        "N_pocket",
                    ):
                        outputs[example_id][guidance_key].append(
                            aux[ai].get(guidance_key) if isinstance(aux[ai], dict) else None
                        )

                    # Save atom_array and sequence
                    chain_info = non_rcsb.initialize_chain_info_from_atom_array(designed_atom_array)
                    outputs[example_id]["designed_sample_seq"].append(
                        ":".join(info["processed_entity_canonical_sequence"] for info in chain_info.values())
                    )

                    # Clean up designed atom array for saving
                    designed_atom_array = clean_up_and_renumber_atom_array(designed_atom_array)

                    # Save samp_atom_array_no_sidechain to outputs
                    outputs[example_id]["designed_sample_atom_array"].append(designed_atom_array)

                    # atom_array with gaps for af3 template conditioning
                    designed_atom_array_with_gaps = designed_atom_array.copy()

                    # Insert UNK atoms for gaps in protein backbone atom array
                    designed_atom_array_with_gaps = insert_unk_residues_for_gaps_in_atom_array(designed_atom_array_with_gaps)

                    # Save designed atom array to cif file
                    out_file = f"{sample_out_dir}/{designed_sample_id}.cif"
                    save_cif_file(designed_atom_array, out_file, cif_save_cfg=cif_save_cfg)
                    outputs[example_id]["designed_sample_path"].append(out_file)

                    # Save designed atom array with gaps for af3 template conditioning
                    out_file_for_af3_tc = f"{sample_out_dir_for_af3_tc}/{designed_sample_id}.cif"
                    save_cif_file(designed_atom_array_with_gaps, out_file_for_af3_tc, cif_save_cfg=cif_save_cfg)
                    outputs[example_id]["designed_sample_path_for_af3_tc"].append(out_file_for_af3_tc)

                    input_atom_array = batch["atom_array"][example_id_to_batch_idx[example_id]]

                    # Calculate sequence recovery metrics
                    seq_recovery_metrics = calculate_sequence_recovery(input_atom_array, designed_atom_array,
                                                                       pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
                                                                       pocket_distance_bins=pocket_distance_bins,
                                                                       n_min_ligand_atoms=pocket_n_min_ligand_atoms_for_seq_recovery)
                    outputs[example_id]["seq_recovery_metrics"].append(seq_recovery_metrics)
            pbar.update(B)
    pbar.close()

    # Convert tensors to CPU values
    for example_id, example_outputs in outputs.items():
        for k, v in example_outputs.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                example_outputs[k] = [t.detach().cpu().item() for t in v]
            elif isinstance(v, torch.Tensor):
                example_outputs[k] = v.detach().cpu().item()


    # Save sample_metadata.pt for later use
    sample_metadata = {}
    for example_id, example_outputs in outputs.items():
        for idx in range(len(example_outputs["designed_sample_id"])):
            designed_sample_id = example_outputs["designed_sample_id"][idx]
            meta_entry = {
                "example_id": example_id,
                "designed_sample_id": designed_sample_id,
                "designed_sample_path": example_outputs["designed_sample_path"][idx],
                "designed_sample_seq": example_outputs["designed_sample_seq"][idx],
                "U": example_outputs["U"][idx],
            }
            if "gamma" in example_outputs:
                meta_entry["gamma"] = example_outputs["gamma"][idx]
            for guidance_key in (
                "U_cond",
                "U_uncond",
                "U_cond_per_res",
                "U_uncond_per_res",
                "U_cond_pocket",
                "U_uncond_pocket",
                "U_cond_pocket_per_res",
                "U_uncond_pocket_per_res",
                "N_pocket",
            ):
                if guidance_key in example_outputs:
                    meta_entry[guidance_key] = example_outputs[guidance_key][idx]
            sample_metadata[designed_sample_id] = meta_entry
    torch.save(sample_metadata, f"{sample_out_dir}/sample_metadata.pt")
    print(f"Saved sample_metadata.pt with {len(sample_metadata)} samples to {sample_out_dir}")

    return outputs

def iter_design_sequence_per_checkpoint(
    *,
    seed: int = 0,
    input_sample_is_designed: bool = False,
    sample_dict: dict | None = None,
    design_cfg: DictConfig | None = None,
    cif_parse_cfg: DictConfig | None = None,
    preprocess_cfg: DictConfig | None = None,
    featurizer_cfg: DictConfig | None = None,
    cif_save_cfg: DictConfig | None = None,
    sampling_inputs_df: pd.DataFrame | None = None,
    log_dir: Path | None = None,
    pos_constraint_df: pd.DataFrame | None = None,
    protein_only: bool = False,
    pocket_distances_for_seq_recovery: list[float] | None = None,
    pocket_distance_bins: list[tuple[float, float]] | None = None,
    pocket_n_min_ligand_atoms_for_seq_recovery: int = 5,
    csv_suffix: str = "",
    guidance_cfg: DictConfig | None = None,
) -> Iterator[tuple[dict, Path, dict]]:
    if sample_dict is None:
        raise ValueError("sample_dict must be provided")
    if design_cfg is None:
        raise ValueError("design_cfg must be provided")
    if log_dir is None:
        raise ValueError("log_dir must be provided")

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_infos = get_checkpoints(design_cfg)
    input_sample_paths = [sample_dict[sid]["input_sample_path"] for sid in sample_dict.keys()]

    for ckpt_info in tqdm(ckpt_infos, desc="Designing sequence"):
        sd_ckpt = ckpt_info["ckpt_path"]
        log_dir_per_ckpt = log_dir / f"step_{ckpt_info['global_step']}_epoch_{ckpt_info['epoch']}"
        log_dir_per_ckpt.mkdir(parents=True, exist_ok=True)

        L.seed_everything(seed)
        seq_des_model = load_seq_des_model(cfg=design_cfg, device=device, ckpt_path=sd_ckpt)

        outputs = design_sequence(
            model=seq_des_model["model"],
            input_sample_is_designed=input_sample_is_designed,
            cif_parse_cfg=cif_parse_cfg,
            preprocess_cfg=preprocess_cfg,
            featurizer_cfg=featurizer_cfg,
            cif_save_cfg=cif_save_cfg,
            sampling_cfg=seq_des_model["sampling_cfg"],
            sampling_inputs_df=sampling_inputs_df,
            pdb_paths=input_sample_paths,
            device=device,
            out_dir=str(log_dir_per_ckpt),
            pos_constraint_df=pos_constraint_df,
            protein_only=protein_only,
            pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
            pocket_distance_bins=pocket_distance_bins,
            pocket_n_min_ligand_atoms_for_seq_recovery=pocket_n_min_ligand_atoms_for_seq_recovery,
            guidance_cfg=guidance_cfg,
        )

        sample_dict_per_ckpt = collect_design_outputs(
            sample_dict=sample_dict,
            outputs=outputs,
            log_dir_per_ckpt=log_dir_per_ckpt,
            csv_suffix=csv_suffix,
            guidance_cfg=guidance_cfg,
        )

        del seq_des_model
        del outputs
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        yield sample_dict_per_ckpt, log_dir_per_ckpt, ckpt_info
