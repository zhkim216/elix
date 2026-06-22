from dataclasses import dataclass
import gc
from pathlib import Path
from typing import Any, Callable, Iterator

import lightning as L
from omegaconf import DictConfig
import pandas as pd
import torch
from tqdm import tqdm

from allatom_design.eval.sampling.sequence_design.checkpoints import (
    ckpt_label,
    get_checkpoints,
    load_seq_des_model,
    parallel_context,
)
from allatom_design.eval.sampling.sequence_design.ensemble.conditioning import (
    ensemble_conditioning_enabled,
    make_ensemble_potts_aux_provider,
)
from allatom_design.eval.sampling.sequence_design.inputs import get_sd_batch
from allatom_design.eval.sampling.sequence_design.outputs import (
    collect_design_outputs,
    convert_output_tensors_to_python,
    record_sampled_batch_outputs,
    write_sample_metadata,
)
from allatom_design.eval.sampling.sequence_design.runtime_plan import (
    SamplingRuntimePlan,
    build_sampling_runtime_plan,
)
from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser


@dataclass(frozen=True)
class SequenceDesignRunSpec:
    seed: int
    input_sample_is_designed: bool
    sample_dict: dict
    design_cfg: DictConfig
    cif_parse_cfg: DictConfig | None
    preprocess_cfg: DictConfig | None
    featurizer_cfg: DictConfig | None
    cif_save_cfg: DictConfig | None
    sampling_inputs_df: pd.DataFrame | None
    selectivity_residue_alignment_df: pd.DataFrame | None
    log_dir: Path
    pos_constraint_df: pd.DataFrame | None
    protein_only: bool
    pocket_distances_for_seq_recovery: list[float] | None
    pocket_distance_bins: list[tuple[float, float]] | None
    pocket_n_min_ligand_atoms_for_seq_recovery: int
    csv_suffix: str
    guidance_cfg: DictConfig | None

    def validate(self) -> None:
        if self.sample_dict is None:
            raise ValueError("sample_dict must be provided")
        if self.design_cfg is None:
            raise ValueError("design_cfg must be provided")
        if self.log_dir is None:
            raise ValueError("log_dir must be provided")

    @property
    def input_sample_paths(self) -> list[str]:
        self.validate()
        return [
            self.sample_dict[sample_id]["input_sample_path"]
            for sample_id in self.sample_dict.keys()
        ]


def build_sequence_design_run_spec_from_cfg(
    *,
    cfg: DictConfig,
    sample_dict: dict,
    cif_parse_cfg: DictConfig | None,
    preprocess_cfg: DictConfig | None,
    sampling_inputs_df: pd.DataFrame | None,
    selectivity_residue_alignment_df: pd.DataFrame | None = None,
    log_dir: Path,
    pos_constraint_df: pd.DataFrame | None,
    csv_suffix: str = "",
    guidance_cfg: DictConfig | None = None,
    pocket_distance_bins: list[tuple[float, float]] | None = None,
) -> SequenceDesignRunSpec:
    sampling_cfg = cfg.get("sampling_cfg", None)
    if guidance_cfg is None and sampling_cfg is not None:
        guidance_cfg = sampling_cfg.get("guidance", None)

    return SequenceDesignRunSpec(
        seed=cfg.seed,
        input_sample_is_designed=cfg.input_sample_is_designed,
        sample_dict=sample_dict,
        design_cfg=cfg,
        cif_parse_cfg=cif_parse_cfg,
        preprocess_cfg=preprocess_cfg,
        featurizer_cfg=cfg.featurizer_cfg.design,
        cif_save_cfg=cfg.cif_cfg.save,
        sampling_inputs_df=sampling_inputs_df,
        selectivity_residue_alignment_df=selectivity_residue_alignment_df,
        log_dir=log_dir,
        pos_constraint_df=pos_constraint_df,
        protein_only=cfg.get("protein_only", False),
        pocket_distances_for_seq_recovery=cfg.pocket_cfg.pocket_distances_for_seq_recovery,
        pocket_distance_bins=pocket_distance_bins,
        pocket_n_min_ligand_atoms_for_seq_recovery=cfg.pocket_cfg.get(
            "n_min_ligand_atoms_for_seq_recovery",
            1,
        ),
        csv_suffix=csv_suffix,
        guidance_cfg=guidance_cfg,
    )


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
    selectivity_residue_alignment_df: pd.DataFrame | None = None,
    pdb_paths: list[str] = None,
    pdb_cfg: DictConfig | None = None,
    device: str = None,
    out_dir: str = None,
    pos_constraint_df: pd.DataFrame | None = None,
    protein_only: bool = False,
    pocket_distances_for_seq_recovery: list[float] = None,
    pocket_distance_bins: list[tuple[float, float]] | None = None,
    pocket_n_min_ligand_atoms_for_seq_recovery: int = 5,
    csv_suffix: str = "",
    guidance_cfg: DictConfig | None = None,
    sampling_plan: SamplingRuntimePlan | None = None,
) -> dict[str, dict[str, Any]]:
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
    sample_out_dir = Path(out_dir, "samples")
    sample_out_dir.mkdir(parents=True, exist_ok=True)
    sample_out_dir_for_af3_tc = Path(out_dir, "samples_for_af3_tc")
    sample_out_dir_for_af3_tc.mkdir(parents=True, exist_ok=True)

    if sampling_plan is None:
        sampling_plan = build_sampling_runtime_plan(
            pdb_paths=pdb_paths,
            sampling_cfg=sampling_cfg,
            sampling_inputs_df=sampling_inputs_df,
            selectivity_residue_alignment_df=selectivity_residue_alignment_df,
            pos_constraint_df=pos_constraint_df,
            guidance_cfg=guidance_cfg,
            out_dir=out_dir,
            cif_parse_cfg=cif_parse_cfg,
            cif_save_cfg=cif_save_cfg,
            csv_suffix=csv_suffix,
        )
    if sampling_plan.staging_message is not None:
        print(sampling_plan.staging_message)

    # Print omitted amino acids.
    if sampling_cfg.verbose and sampling_cfg.omit_aas is not None:
        print(f"Omitting aatype sampling for: {sampling_cfg.omit_aas}")
    sample_token_prefix = sampling_cfg.get("sample_token_prefix", None)

    # Process PDBs in parallel.
    parallel_context_manager = parallel_context(sampling_cfg.num_workers)

    # Begin sampling.
    n_sampling_targets = sampling_plan.target_count()
    pbar = tqdm(
        total=n_sampling_targets,
        desc=f"Sampling {n_sampling_targets} PDBs, {sampling_cfg.num_seqs_per_pdb} sequences per PDB...",
    )
    batch_iter = sampling_plan.iter_batches(batch_size=sampling_cfg.batch_size)

    with parallel_context_manager as parallel_pool:
        for batch_pdb_paths in batch_iter:
            B = sampling_plan.target_count(batch_pdb_paths)

            batch = get_sd_batch(pdb_paths = batch_pdb_paths,
                                 sample_is_designed = input_sample_is_designed,
                                 cif_parse_cfg = cif_parse_cfg,
                                 preprocess_cfg = preprocess_cfg,
                                 featurizer_cfg = featurizer_cfg,
                                 device=device,
                                 parallel_pool=parallel_pool,
                                 sampling_inputs_df=sampling_plan.sampling_inputs_df)
            runtime_batch = sampling_plan.prepare_batch(
                batch=batch,
                batch_pdb_paths=batch_pdb_paths,
                sampling_cfg=sampling_cfg,
                protein_only=protein_only,
                device=device,
            )
            batch = runtime_batch.batch
            sampling_inputs = runtime_batch.sampling_inputs
            guidance_cfg_dict = runtime_batch.guidance_cfg_dict
            guidance_metadata_by_example_id = runtime_batch.guidance_metadata_by_example_id

            potts_aux_provider = None
            if ensemble_conditioning_enabled(sampling_inputs):
                if guidance_cfg_dict is not None and guidance_cfg_dict.get("enabled", False):
                    raise NotImplementedError(
                        "ensemble_conditioning is not supported together with Potts guidance"
                    )
                ensemble_cfg = sampling_inputs["potts_sampling_cfg"]["ensemble_conditioning"]
                potts_aux_provider = make_ensemble_potts_aux_provider(
                    denoiser=model.denoiser,
                    ensemble_cfg=ensemble_cfg,
                    cif_save_cfg=cif_save_cfg,
                )

            # Run sampling.
            id_to_atom_arrays, id_to_aux = model.sample(
                batch,
                sampling_inputs=sampling_inputs,
                potts_aux_provider=potts_aux_provider,
            )
            record_sampled_batch_outputs(
                outputs=outputs,
                batch=batch,
                id_to_atom_arrays=id_to_atom_arrays,
                id_to_aux=id_to_aux,
                guidance_metadata_by_example_id=guidance_metadata_by_example_id,
                sample_token_prefix=sample_token_prefix,
                sample_out_dir=sample_out_dir,
                sample_out_dir_for_af3_tc=sample_out_dir_for_af3_tc,
                cif_save_cfg=cif_save_cfg,
                pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
                pocket_distance_bins=pocket_distance_bins,
                pocket_n_min_ligand_atoms_for_seq_recovery=pocket_n_min_ligand_atoms_for_seq_recovery,
            )
            pbar.update(B)
    pbar.close()

    convert_output_tensors_to_python(outputs)
    write_sample_metadata(
        outputs=outputs,
        sample_out_dir=sample_out_dir,
        csv_suffix=csv_suffix,
    )

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
    selectivity_residue_alignment_df: pd.DataFrame | None = None,
    log_dir: Path | None = None,
    pos_constraint_df: pd.DataFrame | None = None,
    protein_only: bool = False,
    pocket_distances_for_seq_recovery: list[float] | None = None,
    pocket_distance_bins: list[tuple[float, float]] | None = None,
    pocket_n_min_ligand_atoms_for_seq_recovery: int = 5,
    csv_suffix: str = "",
    guidance_cfg: DictConfig | None = None,
    runtime_plan_builder: Callable[..., SamplingRuntimePlan] | None = None,
) -> Iterator[tuple[dict, Path, dict]]:
    run_spec = SequenceDesignRunSpec(
        seed=seed,
        input_sample_is_designed=input_sample_is_designed,
        sample_dict=sample_dict,
        design_cfg=design_cfg,
        cif_parse_cfg=cif_parse_cfg,
        preprocess_cfg=preprocess_cfg,
        featurizer_cfg=featurizer_cfg,
        cif_save_cfg=cif_save_cfg,
        sampling_inputs_df=sampling_inputs_df,
        selectivity_residue_alignment_df=selectivity_residue_alignment_df,
        log_dir=log_dir,
        pos_constraint_df=pos_constraint_df,
        protein_only=protein_only,
        pocket_distances_for_seq_recovery=pocket_distances_for_seq_recovery,
        pocket_distance_bins=pocket_distance_bins,
        pocket_n_min_ligand_atoms_for_seq_recovery=pocket_n_min_ligand_atoms_for_seq_recovery,
        csv_suffix=csv_suffix,
        guidance_cfg=guidance_cfg,
    )
    yield from iter_design_sequence_for_run_spec(
        run_spec,
        runtime_plan_builder=runtime_plan_builder,
    )


def iter_design_sequence_for_run_spec(
    run_spec: SequenceDesignRunSpec,
    *,
    runtime_plan_builder: Callable[..., SamplingRuntimePlan] | None = None,
) -> Iterator[tuple[dict, Path, dict]]:
    run_spec.validate()
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_infos = get_checkpoints(run_spec.design_cfg)
    input_sample_paths = run_spec.input_sample_paths

    for ckpt_info in tqdm(ckpt_infos, desc="Designing sequence"):
        sd_ckpt = ckpt_info["ckpt_path"]
        log_dir_per_ckpt = run_spec.log_dir / ckpt_label(ckpt_info)
        log_dir_per_ckpt.mkdir(parents=True, exist_ok=True)

        L.seed_everything(run_spec.seed)
        seq_des_model = load_seq_des_model(cfg=run_spec.design_cfg, device=device, ckpt_path=sd_ckpt)
        sampling_plan = None
        if runtime_plan_builder is not None:
            sampling_plan = runtime_plan_builder(
                pdb_paths=input_sample_paths,
                sampling_cfg=seq_des_model["sampling_cfg"],
                out_dir=str(log_dir_per_ckpt),
                cif_parse_cfg=run_spec.cif_parse_cfg,
                cif_save_cfg=run_spec.cif_save_cfg,
                csv_suffix=run_spec.csv_suffix,
            )

        outputs = design_sequence(
            model=seq_des_model["model"],
            input_sample_is_designed=run_spec.input_sample_is_designed,
            cif_parse_cfg=run_spec.cif_parse_cfg,
            preprocess_cfg=run_spec.preprocess_cfg,
            featurizer_cfg=run_spec.featurizer_cfg,
            cif_save_cfg=run_spec.cif_save_cfg,
            sampling_cfg=seq_des_model["sampling_cfg"],
            sampling_inputs_df=run_spec.sampling_inputs_df,
            selectivity_residue_alignment_df=run_spec.selectivity_residue_alignment_df,
            pdb_paths=input_sample_paths,
            device=device,
            out_dir=str(log_dir_per_ckpt),
            pos_constraint_df=run_spec.pos_constraint_df,
            protein_only=run_spec.protein_only,
            pocket_distances_for_seq_recovery=run_spec.pocket_distances_for_seq_recovery,
            pocket_distance_bins=run_spec.pocket_distance_bins,
            pocket_n_min_ligand_atoms_for_seq_recovery=(
                run_spec.pocket_n_min_ligand_atoms_for_seq_recovery
            ),
            csv_suffix=run_spec.csv_suffix,
            guidance_cfg=run_spec.guidance_cfg,
            sampling_plan=sampling_plan,
        )

        sample_dict_per_ckpt = collect_design_outputs(
            sample_dict=run_spec.sample_dict,
            outputs=outputs,
            log_dir_per_ckpt=log_dir_per_ckpt,
            csv_suffix=run_spec.csv_suffix,
            guidance_cfg=run_spec.guidance_cfg,
            sampling_inputs_df=run_spec.sampling_inputs_df,
        )

        del seq_des_model
        del outputs
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        yield sample_dict_per_ckpt, log_dir_per_ckpt, ckpt_info
