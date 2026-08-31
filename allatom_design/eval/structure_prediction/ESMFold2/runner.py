"""In-process, resumable ESMFold2 inference."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .inputs import (
    load_esmfold2_input_record,
    structure_prediction_input_from_record,
)


MANIFEST_SCHEMA_VERSION = 1
PREDICTION_CONTRACT_VERSION = 2


@dataclass(frozen=True)
class ESMFold2RuntimeConfig:
    model_id: str
    model_revision: str | None
    local_files_only: bool
    ccd_cache: str | None
    use_source_residue_indices: bool
    seeds: tuple[int, ...]
    num_loops: int
    num_sampling_steps: int
    num_diffusion_samples: int
    lm_dropout: float
    overwrite: bool
    strict_input_fingerprint: bool

    @classmethod
    def from_config(
        cls, config: DictConfig | dict[str, Any]
    ) -> "ESMFold2RuntimeConfig":
        values = (
            OmegaConf.to_container(config, resolve=True)
            if isinstance(config, DictConfig)
            else dict(config)
        )
        seeds = tuple(int(value) for value in values["seeds"])
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(
                f"ESMFold2 seeds must be non-empty and unique, got {seeds}"
            )
        runtime = cls(
            model_id=str(values["model_id"]),
            model_revision=(
                None
                if values.get("model_revision") is None
                else str(values["model_revision"])
            ),
            local_files_only=bool(values.get("local_files_only", True)),
            ccd_cache=(
                None if values.get("ccd_cache") is None else str(values["ccd_cache"])
            ),
            use_source_residue_indices=bool(
                values.get("use_source_residue_indices", True)
            ),
            seeds=seeds,
            num_loops=int(values["num_loops"]),
            num_sampling_steps=int(values["num_sampling_steps"]),
            num_diffusion_samples=int(values["num_diffusion_samples"]),
            lm_dropout=float(values["lm_dropout"]),
            overwrite=bool(values.get("overwrite", False)),
            strict_input_fingerprint=bool(values.get("strict_input_fingerprint", True)),
        )
        if runtime.num_loops < 0:
            raise ValueError("num_loops must be non-negative")
        if runtime.num_sampling_steps <= 0:
            raise ValueError("num_sampling_steps must be positive")
        if runtime.num_diffusion_samples <= 0:
            raise ValueError("num_diffusion_samples must be positive")
        if not 0.0 <= runtime.lm_dropout < 1.0:
            raise ValueError("lm_dropout must be in [0, 1)")
        return runtime

    @property
    def expected_prediction_count(self) -> int:
        return len(self.seeds) * self.num_diffusion_samples

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "local_files_only": self.local_files_only,
            "ccd_cache": self.ccd_cache,
            "use_source_residue_indices": self.use_source_residue_indices,
            "seeds": list(self.seeds),
            "num_loops": self.num_loops,
            "num_sampling_steps": self.num_sampling_steps,
            "num_diffusion_samples": self.num_diffusion_samples,
            "lm_dropout": self.lm_dropout,
            "mode": "single_sequence_no_msa",
        }


@dataclass(frozen=True)
class ESMFold2PredictionArtifact:
    seed: int
    sample_index: int
    model_cif_path: Path
    confidence_json_path: Path
    summary_json_path: Path

    @property
    def prediction_id(self) -> str:
        return f"seed_{self.seed}_sample_{self.sample_index}"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _artifact(
    *,
    job_dir: Path,
    job_name: str,
    seed: int,
    sample_index: int,
) -> ESMFold2PredictionArtifact:
    sample_dir = job_dir / f"seed_{seed}" / f"sample_{sample_index}"
    stem = f"{job_name}_model"
    return ESMFold2PredictionArtifact(
        seed=seed,
        sample_index=sample_index,
        model_cif_path=sample_dir / f"{stem}.cif",
        confidence_json_path=sample_dir / f"{job_name}_confidences.json",
        summary_json_path=sample_dir / f"{job_name}_summary.json",
    )


def _expected_artifacts(
    *,
    job_dir: Path,
    job_name: str,
    runtime: ESMFold2RuntimeConfig,
) -> list[ESMFold2PredictionArtifact]:
    return [
        _artifact(
            job_dir=job_dir,
            job_name=job_name,
            seed=seed,
            sample_index=sample_index,
        )
        for seed in runtime.seeds
        for sample_index in range(runtime.num_diffusion_samples)
    ]


def _input_fingerprint(
    *,
    record: dict[str, Any],
    runtime: ESMFold2RuntimeConfig,
) -> str:
    return _sha256_payload(
        {
            "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
            "input": record,
            "inference": runtime.fingerprint_payload(),
        }
    )


def _complete_artifact(artifact: ESMFold2PredictionArtifact) -> bool:
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            artifact.model_cif_path,
            artifact.confidence_json_path,
            artifact.summary_json_path,
        )
    )


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _direct_url(package: str) -> dict[str, Any] | None:
    try:
        text = importlib.metadata.distribution(package).read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return json.loads(text) if text else None


def _editable_git_source(package: str) -> dict[str, Any] | None:
    direct_url = _direct_url(package)
    if not direct_url or not direct_url.get("dir_info", {}).get("editable"):
        return None
    source = Path(unquote(urlparse(str(direct_url["url"])).path)).resolve()

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source), *args],
            text=True,
        ).strip()

    return {
        "source": str(source),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _atom_plddts(result: Any) -> list[float]:
    complex_result = result.complex
    token_plddt = np.asarray(complex_result.plddt, dtype=float)
    atom_plddt = np.empty(len(complex_result.atom_positions), dtype=float)
    for value, (start, end) in zip(
        token_plddt, complex_result.token_to_atoms, strict=True
    ):
        atom_plddt[int(start) : int(end)] = float(value) * 100.0
    elements = np.char.upper(np.asarray(complex_result.atom_elements, dtype=str))
    # Elix's production prediction reader removes hydrogens before applying the
    # adjacent confidence sidecar.
    keep = (elements != "H") & (elements != "D")
    return atom_plddt[keep].tolist()


def _result_summary(
    *,
    result: Any,
    artifact: ESMFold2PredictionArtifact,
) -> dict[str, Any]:
    complex_result = result.complex
    plddt = np.asarray(complex_result.plddt, dtype=float) * 100.0
    chain_lookup = complex_result.metadata.chain_lookup
    token_chains = np.asarray(
        [chain_lookup[int(chain_id)] for chain_id in complex_result.chain_id],
        dtype=object,
    )
    return {
        "prediction_id": artifact.prediction_id,
        "seed": artifact.seed,
        "sample_index": artifact.sample_index,
        "model_cif_path": str(artifact.model_cif_path),
        "confidence_json_path": str(artifact.confidence_json_path),
        "mean_plddt": float(np.mean(plddt)),
        "chain_mean_plddt": {
            str(chain): float(np.mean(plddt[token_chains == chain]))
            for chain in dict.fromkeys(token_chains.tolist())
        },
        "ptm": None if result.ptm is None else float(result.ptm),
        "iptm": None if result.iptm is None else float(result.iptm),
        "num_tokens": int(len(plddt)),
        "num_atoms_without_hydrogen": len(_atom_plddts(result)),
    }


class ESMFold2Session:
    """One lazily loaded ESMFold2 model reused across many designed sequences."""

    def __init__(self, runtime: ESMFold2RuntimeConfig):
        self.runtime = runtime
        self._model = None
        self._model_commit = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers.models.esmfold2.modeling_esmfold2 import (
            ESMFold2Model,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("ESMFold2 inference requires CUDA")
        kwargs: dict[str, Any] = {
            "local_files_only": self.runtime.local_files_only,
        }
        if self.runtime.model_revision is not None:
            kwargs["revision"] = self.runtime.model_revision
        self._model = (
            ESMFold2Model.from_pretrained(
                self.runtime.model_id,
                **kwargs,
            )
            .cuda()
            .eval()
        )
        self._model_commit = getattr(self._model.config, "_commit_hash", None)
        torch.cuda.reset_peak_memory_stats()
        print(
            f"Loaded {self.runtime.model_id} revision={self._model_commit} "
            f"on {torch.cuda.get_device_name(0)}",
            flush=True,
        )

    def close(self) -> None:
        if self._model is None:
            return
        import torch

        self._model = None
        torch.cuda.empty_cache()

    def _runtime_metadata(self) -> dict[str, Any]:
        import torch

        return {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": _package_version("transformers"),
            "esm": _package_version("esm"),
            "esm_direct_url": _direct_url("esm"),
            "esm_editable_git": _editable_git_source("esm"),
            "transformers_direct_url": _direct_url("transformers"),
            "model_commit": self._model_commit,
            "model_parameter_dtype": str(next(self._model.parameters()).dtype),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }

    def predict(
        self,
        *,
        input_path: str | Path,
        prediction_dir: str | Path,
    ) -> list[ESMFold2PredictionArtifact]:
        input_path = Path(input_path)
        record = load_esmfold2_input_record(input_path)
        job_name = str(record["name"])
        job_dir = Path(prediction_dir) / job_name
        manifest_path = job_dir / "manifest.json"
        fingerprint = _input_fingerprint(record=record, runtime=self.runtime)
        expected = _expected_artifacts(
            job_dir=job_dir,
            job_name=job_name,
            runtime=self.runtime,
        )

        if self.runtime.overwrite and job_dir.exists():
            shutil.rmtree(job_dir)
        if job_dir.exists() and not manifest_path.is_file():
            if any(job_dir.iterdir()):
                raise RuntimeError(f"Existing ESMFold2 job has no manifest: {job_dir}")
        existing_manifest = None
        if manifest_path.is_file():
            with manifest_path.open() as handle:
                existing_manifest = json.load(handle)
            existing_fingerprint = existing_manifest.get("input_fingerprint")
            if (
                existing_fingerprint != fingerprint
                and self.runtime.strict_input_fingerprint
            ):
                raise RuntimeError(
                    "Existing ESMFold2 predictions have a stale input/config "
                    f"fingerprint: {job_dir}"
                )
            if existing_fingerprint != fingerprint:
                raise RuntimeError(
                    "Refusing to reuse mismatched ESMFold2 predictions without "
                    "overwrite=true"
                )
            if existing_manifest.get("status") == "complete" and all(
                _complete_artifact(artifact) for artifact in expected
            ):
                return expected

        job_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "prediction_contract_version": PREDICTION_CONTRACT_VERSION,
            "status": "running",
            "job_name": job_name,
            "input_path": str(input_path.resolve()),
            "input_fingerprint": fingerprint,
            "inference": self.runtime.fingerprint_payload(),
            "predictions": [],
        }
        _atomic_write_json(manifest_path, manifest)

        self._ensure_model()
        from esm.models.esmfold2 import ESMFold2InputBuilder

        structure_input = structure_prediction_input_from_record(
            record,
            use_source_residue_indices=(
                self.runtime.use_source_residue_indices
            ),
        )
        input_builder = ESMFold2InputBuilder(
            ccd_cache=(
                None if self.runtime.ccd_cache is None else Path(self.runtime.ccd_cache)
            ),
        )
        started = time.time()
        try:
            for seed in self.runtime.seeds:
                seed_artifacts = [
                    artifact for artifact in expected if artifact.seed == seed
                ]
                missing = [
                    artifact
                    for artifact in seed_artifacts
                    if not _complete_artifact(artifact)
                ]
                if not missing:
                    continue
                fold_started = time.time()
                result_or_results = input_builder.fold(
                    self._model,
                    structure_input,
                    num_loops=self.runtime.num_loops,
                    num_sampling_steps=self.runtime.num_sampling_steps,
                    num_diffusion_samples=(self.runtime.num_diffusion_samples),
                    seed=seed,
                    lm_dropout=self.runtime.lm_dropout,
                    msa_max_depth=None,
                    complex_id=job_name,
                )
                results = (
                    result_or_results
                    if isinstance(result_or_results, list)
                    else [result_or_results]
                )
                if len(results) != self.runtime.num_diffusion_samples:
                    raise RuntimeError(
                        f"{job_name} seed={seed}: expected "
                        f"{self.runtime.num_diffusion_samples} diffusion samples, "
                        f"found {len(results)}"
                    )
                for result, artifact in zip(results, seed_artifacts, strict=True):
                    if _complete_artifact(artifact):
                        continue
                    artifact.model_cif_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_write_text(
                        artifact.model_cif_path,
                        result.complex.to_mmcif(),
                    )
                    confidence = {
                        "atom_plddts": _atom_plddts(result),
                        "ptm": (None if result.ptm is None else float(result.ptm)),
                        "iptm": (None if result.iptm is None else float(result.iptm)),
                    }
                    _atomic_write_json(artifact.confidence_json_path, confidence)
                    summary = _result_summary(result=result, artifact=artifact)
                    _atomic_write_json(artifact.summary_json_path, summary)
                print(
                    f"ESMFold2 {job_name} seed={seed} complete in "
                    f"{time.time() - fold_started:.2f}s",
                    flush=True,
                )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = str(exc)
            manifest["runtime"] = self._runtime_metadata()
            manifest["elapsed_seconds"] = time.time() - started
            _atomic_write_json(manifest_path, manifest)
            raise

        if not all(_complete_artifact(artifact) for artifact in expected):
            raise RuntimeError(
                f"ESMFold2 job did not produce every expected artifact: {job_dir}"
            )
        manifest["status"] = "complete"
        manifest["runtime"] = self._runtime_metadata()
        manifest["elapsed_seconds"] = time.time() - started
        manifest["predictions"] = [
            {
                "prediction_id": artifact.prediction_id,
                "seed": artifact.seed,
                "sample_index": artifact.sample_index,
                "model_cif_path": str(artifact.model_cif_path.relative_to(job_dir)),
                "confidence_json_path": str(
                    artifact.confidence_json_path.relative_to(job_dir)
                ),
                "summary_json_path": str(
                    artifact.summary_json_path.relative_to(job_dir)
                ),
            }
            for artifact in expected
        ]
        _atomic_write_json(manifest_path, manifest)
        return expected


def summarize_esmfold2_prediction_outputs(
    *,
    input_path: str | Path,
    prediction_dir: str | Path,
    runtime: ESMFold2RuntimeConfig,
) -> dict[str, Any]:
    record = load_esmfold2_input_record(input_path)
    job_name = str(record["name"])
    job_dir = Path(prediction_dir) / job_name
    manifest_path = job_dir / "manifest.json"
    expected = _expected_artifacts(
        job_dir=job_dir,
        job_name=job_name,
        runtime=runtime,
    )
    expected_cifs = {artifact.model_cif_path for artifact in expected}
    found_cifs = (
        set(job_dir.glob("seed_*/sample_*/*_model.cif")) if job_dir.is_dir() else set()
    )
    surplus = sorted(found_cifs - expected_cifs)
    complete = [artifact for artifact in expected if _complete_artifact(artifact)]
    malformed = [
        artifact.model_cif_path.parent
        for artifact in expected
        if any(
            path.exists()
            for path in (
                artifact.model_cif_path,
                artifact.confidence_json_path,
                artifact.summary_json_path,
            )
        )
        and not _complete_artifact(artifact)
    ]
    fingerprint_ok: bool | None = None
    fingerprint_error = ""
    manifest_status = "missing"
    if manifest_path.is_file():
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        manifest_status = str(manifest.get("status", "unknown"))
        expected_fingerprint = _input_fingerprint(record=record, runtime=runtime)
        fingerprint_ok = manifest.get("input_fingerprint") == expected_fingerprint
        if not fingerprint_ok:
            fingerprint_error = (
                "manifest input/config fingerprint does not match current input"
            )
    return {
        "artifacts": complete,
        "n_expected": runtime.expected_prediction_count,
        "n_found": len(complete),
        "n_malformed": len(malformed),
        "n_surplus": len(surplus),
        "malformed_prediction_dirs": malformed,
        "surplus_prediction_paths": surplus,
        "input_fingerprint_ok": fingerprint_ok,
        "input_fingerprint_error": fingerprint_error,
        "manifest_status": manifest_status,
    }
