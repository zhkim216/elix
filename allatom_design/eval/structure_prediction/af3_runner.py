from __future__ import annotations

import gc
import importlib.util
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import torch
from omegaconf import DictConfig

from allatom_design.eval.config import (
    config_value_as_bool,
    get_config_value,
)
from allatom_design.eval.chemical_components import normalize_ccd_code


# Global caches for AF3 runner module, ModelRunner, and DataPipelineConfig.
_AF3_RUNNER_MOD = None
_AF3_MODEL_RUNNER = None
_AF3_DATA_PIPELINE_CONFIG = None
_AF3_MODEL_RUNNER_CACHE_KEY = None
_AF3_DATA_PIPELINE_CONFIG_CACHE_KEY = None
_AF3_MAX_TEMPLATE_DATE = None
_AF3_BUCKETS = None

DEFAULT_AF3_MAX_TEMPLATE_DATE = "2021-09-30"


@lru_cache(maxsize=1)
def _af3_glycan_ccd_codes() -> frozenset[str]:
    from alphafold3.constants import chemical_component_sets

    return frozenset(
        normalize_ccd_code(code)
        for code in (
            chemical_component_sets.GLYCAN_LINKING_LIGANDS
            | chemical_component_sets.GLYCAN_OTHER_LIGANDS
        )
    )


def _fold_input_has_glycan_ligand(fold_input) -> bool:
    glycan_ccd_codes = _af3_glycan_ccd_codes()
    ligands = getattr(fold_input, "ligands", [])
    if callable(ligands):
        ligands = ligands()
    for ligand in ligands:
        ccd_ids = getattr(ligand, "ccd_ids", None)
        if ccd_ids is None:
            continue
        if any(normalize_ccd_code(code) in glycan_ccd_codes for code in ccd_ids):
            return True
    return False


def _resolve_fix_standalone_glycans(mode_config: dict | DictConfig, fold_input) -> bool:
    override = get_config_value(mode_config, "fix_standalone_glycans", None)
    if override is not None:
        return config_value_as_bool(override)
    return _fold_input_has_glycan_ligand(fold_input)


def _json_needs_fix_standalone_glycans(json_path: str | Path, mode_config: dict | DictConfig) -> bool:
    override = get_config_value(mode_config, "fix_standalone_glycans", None)
    if override is not None:
        return config_value_as_bool(override)

    from alphafold3.common import folding_input

    return any(
        _fold_input_has_glycan_ligand(fold_input)
        for fold_input in folding_input.load_fold_inputs_from_path(Path(json_path))
    )


def _resolve_max_template_date(mode_config: dict | DictConfig | None) -> str:
    value = get_config_value(mode_config, "max_template_date", None)
    if value is None:
        return DEFAULT_AF3_MAX_TEMPLATE_DATE
    return str(value)


def _af3_overwrite_enabled(mode_config: dict | DictConfig | None) -> bool:
    return config_value_as_bool(get_config_value(mode_config, "overwrite", False))


def _prepare_af3_sample_dir(
    *,
    json_path: str,
    out_dir: str,
    mode_config: dict | DictConfig | None,
) -> bool:
    """Return True when an existing prediction should be reused."""
    sample_name = Path(json_path).stem
    sample_dir = Path(out_dir) / sample_name
    overwrite = _af3_overwrite_enabled(mode_config)

    sample_cif_files = list(sample_dir.rglob("*.cif")) if sample_dir.exists() else []
    if sample_cif_files and not overwrite:
        print(f"AF3 prediction already exists for {sample_name}")
        return True

    if overwrite and sample_dir.exists():
        print(f"Overwriting AF3 prediction for {sample_name}: removing {sample_dir}")
        shutil.rmtree(sample_dir)

    return False


def _load_af3_runner(runner_path: str):
    """Load run_alphafold.py as a module dynamically. Cached after first load."""
    global _AF3_RUNNER_MOD
    if _AF3_RUNNER_MOD is not None:
        return _AF3_RUNNER_MOD

    spec = importlib.util.spec_from_file_location("af3_runner", runner_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _AF3_RUNNER_MOD = mod
    return mod


def _get_af3_model_runner_and_config(
    runner_path: str,
    inference_config: dict,
    mode: str = "ss",
):
    """
    Get or create cached AF3 ModelRunner and DataPipelineConfig.
    Caches are keyed by the runtime values that affect model/data-pipeline setup.
    """
    import datetime as dt
    import pathlib
    import jax

    global _AF3_MODEL_RUNNER, _AF3_DATA_PIPELINE_CONFIG
    global _AF3_MODEL_RUNNER_CACHE_KEY, _AF3_DATA_PIPELINE_CONFIG_CACHE_KEY
    global _AF3_MAX_TEMPLATE_DATE, _AF3_BUCKETS

    runner = _load_af3_runner(runner_path)
    try:
        from absl import flags as absl_flags

        if not absl_flags.FLAGS.is_parsed():
            absl_flags.FLAGS([str(runner_path)])
    except Exception as exc:
        print(f"[AF3 init] Warning: failed to pre-parse AF3 absl flags: {exc}")

    base_config = inference_config.get("base", {})
    mode_config = inference_config.get(mode, {})
    flash_attn = base_config.get("flash_attention_implementation", "triton")
    model_dir = base_config.get("model_dir", "")
    template_pair_scale = mode_config.get("template_pair_scale", 1.0)
    model_runner_cache_key = (
        str(runner_path),
        str(model_dir),
        flash_attn,
        mode_config.get("num_diffusion_samples", 5),
        mode_config.get("num_recycles", 3),
        mode_config.get("ligand_protein_template_conditioning_mode", 0),
        mode_config.get("mask_template_sidechains", True),
        mode_config.get("mask_template_sequence", True),
        template_pair_scale,
    )

    if (
        _AF3_MODEL_RUNNER is None
        or _AF3_MODEL_RUNNER_CACHE_KEY != model_runner_cache_key
    ):
        torch.cuda.empty_cache()

        devices = jax.local_devices(backend="gpu")
        print(f"[AF3 init] Found devices: {devices}, using device 0: {devices[0]}")

        model_config = runner.make_model_config(
            flash_attention_implementation=flash_attn,
            num_diffusion_samples=mode_config.get("num_diffusion_samples", 5),
            num_recycles=mode_config.get("num_recycles", 3),
            return_embeddings=False,
            return_distogram=False,
            ligand_protein_template_conditioning_mode=mode_config.get(
                "ligand_protein_template_conditioning_mode",
                0,
            ),
            mask_template_sidechains=mode_config.get("mask_template_sidechains", True),
            mask_template_sequence=mode_config.get("mask_template_sequence", True),
            template_pair_scale=template_pair_scale,
        )

        _AF3_MODEL_RUNNER = runner.ModelRunner(
            config=model_config,
            device=devices[0],
            model_dir=pathlib.Path(model_dir),
        )
        print("[AF3 init] Loading model parameters...")
        _ = _AF3_MODEL_RUNNER.model_params
        print("[AF3 init] Model parameters loaded and cached.")
        _AF3_MODEL_RUNNER_CACHE_KEY = model_runner_cache_key

    max_template_date_str = _resolve_max_template_date(mode_config)
    db_dir = base_config.get("db_dir", "")
    data_pipeline_cache_key = (str(db_dir), max_template_date_str)

    if (
        _AF3_DATA_PIPELINE_CONFIG is None
        or _AF3_DATA_PIPELINE_CONFIG_CACHE_KEY != data_pipeline_cache_key
    ):
        import shutil
        from alphafold3.data import pipeline

        _AF3_MAX_TEMPLATE_DATE = dt.date.fromisoformat(max_template_date_str)
        buckets_list = [
            256,
            512,
            768,
            1024,
            1280,
            1536,
            2048,
            2560,
            3072,
            3584,
            4096,
            4608,
            5120,
        ]
        _AF3_BUCKETS = tuple(buckets_list)
        expand_path = lambda x: runner.replace_db_dir(x, [db_dir])
        _AF3_DATA_PIPELINE_CONFIG = pipeline.DataPipelineConfig(
            jackhmmer_binary_path=shutil.which("jackhmmer"),
            nhmmer_binary_path=shutil.which("nhmmer"),
            hmmalign_binary_path=shutil.which("hmmalign"),
            hmmsearch_binary_path=shutil.which("hmmsearch"),
            hmmbuild_binary_path=shutil.which("hmmbuild"),
            small_bfd_database_path=expand_path("${DB_DIR}/bfd-first_non_consensus_sequences.fasta"),
            mgnify_database_path=expand_path("${DB_DIR}/mgy_clusters_2022_05.fa"),
            uniprot_cluster_annot_database_path=expand_path("${DB_DIR}/uniprot_all_2021_04.fa"),
            uniref90_database_path=expand_path("${DB_DIR}/uniref90_2022_05.fa"),
            ntrna_database_path=expand_path("${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"),
            rfam_database_path=expand_path("${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"),
            rna_central_database_path=expand_path("${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta"),
            pdb_database_path=expand_path("${DB_DIR}/mmcif_files"),
            seqres_database_path=expand_path("${DB_DIR}/pdb_seqres_2022_09_28.fasta"),
            max_template_date=_AF3_MAX_TEMPLATE_DATE,
        )
        print("[AF3 init] DataPipelineConfig created and cached.")
        _AF3_DATA_PIPELINE_CONFIG_CACHE_KEY = data_pipeline_cache_key

    return runner, _AF3_MODEL_RUNNER, _AF3_DATA_PIPELINE_CONFIG


def _run_af3_inprocess(
    json_path: str,
    out_dir: str,
    runner_path: str,
    inference_config: dict,
    mode: str = "ss",
) -> None:
    """
    Run AF3 in-process without subprocess, reusing a cached ModelRunner.
    This avoids GPU exclusive mode issues and prevents GPU memory accumulation
    from repeated model loading.
    """
    import pathlib
    from alphafold3.common import folding_input

    mode_config = inference_config.get(mode, {})
    if _prepare_af3_sample_dir(
        json_path=json_path,
        out_dir=out_dir,
        mode_config=mode_config,
    ):
        return

    runner, model_runner, data_pipeline_config = _get_af3_model_runner_and_config(
        runner_path=runner_path,
        inference_config=inference_config,
        mode=mode,
    )

    fold_inputs = folding_input.load_fold_inputs_from_path(pathlib.Path(json_path))

    for fold_input_item in fold_inputs:
        output_dir = os.path.join(out_dir, fold_input_item.sanitised_name())
        fix_standalone_glycans = _resolve_fix_standalone_glycans(
            mode_config=mode_config,
            fold_input=fold_input_item,
        )
        try:
            runner.process_fold_input(
                fold_input=fold_input_item,
                data_pipeline_config=data_pipeline_config,
                model_runner=model_runner,
                output_dir=output_dir,
                buckets=_AF3_BUCKETS,
                ref_max_modified_date=_AF3_MAX_TEMPLATE_DATE,
                conformer_max_iterations=None,
                resolve_msa_overlaps=True,
                max_templates=mode_config.get("max_templates", 0),
                ligand_protein_template_conditioning_mode=mode_config.get(
                    "ligand_protein_template_conditioning_mode",
                    0,
                ),
                fix_standalone_glycans=fix_standalone_glycans,
                force_output_dir=True,
            )
        except SystemExit as exc:
            if exc.code != 0 and exc.code is not None:
                raise RuntimeError(f"AF3 process_fold_input exited with code {exc.code}")
        except Exception as exc:
            print(f"AF3 prediction failed for {Path(json_path).stem}: {exc}")
            raise
        gc.collect()


def run_af3_single_sequence(
    json_path: str,
    out_dir: str,
    runner_path: str,
    inference_config: dict = None,
    use_subprocess: bool = False,
) -> None:
    """Run AF3 single-sequence inference."""
    ss_config = inference_config.ss
    if use_subprocess:
        if _prepare_af3_sample_dir(
            json_path=json_path,
            out_dir=out_dir,
            mode_config=ss_config,
        ):
            return

        cmd = [
            sys.executable,
            runner_path,
            f"--json_path={json_path}",
            f"--output_dir={out_dir}",
            f"--model_dir={inference_config.base.get('model_dir', None)}",
            "--run_data_pipeline=True",
            "--run_inference=True",
            f"--db_dir={inference_config.base.get('db_dir', None)}",
            f"--flash_attention_implementation={inference_config.base.get('flash_attention_implementation', 'triton')}",
            f"--num_recycles={ss_config.get('num_recycles', 3)}",
            f"--num_diffusion_samples={ss_config.get('num_diffusion_samples', 5)}",
            f"--max_templates={ss_config.get('max_templates', 0)}",
            f"--max_template_date={_resolve_max_template_date(ss_config)}",
            f"--ligand_protein_template_conditioning_mode={ss_config.get('ligand_protein_template_conditioning_mode', 0)}",
            f"--template_pair_scale={ss_config.get('template_pair_scale', 1.0)}",
        ]
        if _af3_overwrite_enabled(ss_config):
            cmd.append("--force_output_dir=True")
        if _json_needs_fix_standalone_glycans(json_path, ss_config):
            cmd.append("--fix_standalone_glycans=True")
        env = os.environ.copy()
        subprocess.run(cmd, check=True, env=env)
        return

    _run_af3_inprocess(
        json_path=json_path,
        out_dir=out_dir,
        runner_path=runner_path,
        inference_config=inference_config,
        mode="ss",
    )


def run_af3_template_conditioned(
    json_path: str,
    out_dir: str,
    runner_path: str,
    inference_config: dict = None,
    use_subprocess: bool = False,
) -> None:
    """Run AF3 template-conditioned inference."""
    tc_config = inference_config.tc
    if use_subprocess:
        if _prepare_af3_sample_dir(
            json_path=json_path,
            out_dir=out_dir,
            mode_config=tc_config,
        ):
            return

        cmd = [
            sys.executable,
            runner_path,
            f"--json_path={json_path}",
            f"--output_dir={out_dir}",
            f"--model_dir={inference_config.base.get('model_dir', None)}",
            "--run_data_pipeline=True",
            "--run_inference=True",
            f"--db_dir={inference_config.base.get('db_dir', None)}",
            f"--flash_attention_implementation={inference_config.base.get('flash_attention_implementation', 'triton')}",
            f"--num_recycles={tc_config.get('num_recycles', 3)}",
            f"--num_diffusion_samples={tc_config.get('num_diffusion_samples', 5)}",
            f"--max_templates={tc_config.get('max_templates', 1)}",
            f"--ligand_protein_template_conditioning_mode={tc_config.get('ligand_protein_template_conditioning_mode', 1)}",
            f"--mask_template_sidechains={tc_config.get('mask_template_sidechains', True)}",
            f"--mask_template_sequence={tc_config.get('mask_template_sequence', True)}",
            f"--template_pair_scale={tc_config.get('template_pair_scale', 1.0)}",
            f"--max_template_date={_resolve_max_template_date(tc_config)}",
        ]
        if _af3_overwrite_enabled(tc_config):
            cmd.append("--force_output_dir=True")
        if _json_needs_fix_standalone_glycans(json_path, tc_config):
            cmd.append("--fix_standalone_glycans=True")
        env = os.environ.copy()
        subprocess.run(cmd, check=True, env=env)
        return

    _run_af3_inprocess(
        json_path=json_path,
        out_dir=out_dir,
        runner_path=runner_path,
        inference_config=inference_config,
        mode="tc",
    )


def find_pred_sample_path_af3(
    out_dir: str = None,
    job_name: str = None,
) -> tuple[list[Path], list[Path]]:
    """Find AF3 prediction sample paths for a given job name."""
    prediction_dir = Path(out_dir, job_name)
    sample_dirs = []
    sample_cif_paths = []
    if not prediction_dir.exists():
        return sample_dirs, sample_cif_paths
    for path in prediction_dir.iterdir():
        if path.is_dir():
            sample_dirs.append(path)
            cif_path = [p for p in path.glob("*.cif") if p.stem.endswith("model")][0]
            sample_cif_paths.append(cif_path)

    return sample_dirs, sample_cif_paths
