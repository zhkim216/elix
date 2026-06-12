from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from allatom_design.model.seq_denoiser.denoisers.elix_mpnn_denoiser import (
    _aggregate_potts_params,
)
from allatom_design.utils.feature_utils import slice_feats
from allatom_design.utils.sample_io_utils import save_cif_file


_ENSEMBLE_NOISE_KEYS = (
    "noised_coords",
    "noised_ca_coords",
    "noised_n_coords",
    "noised_c_coords",
    "noised_o_coords",
    "noised_pseudo_cb_coords",
)


def ensemble_conditioning_enabled(sampling_inputs: dict[str, Any]) -> bool:
    cfg = _raw_ensemble_cfg(sampling_inputs)
    return bool(cfg) and bool(cfg.get("enabled", False))


def make_ensemble_potts_aux_provider(
    *,
    denoiser: Any,
    ensemble_cfg: dict[str, Any] | DictConfig,
    cif_save_cfg: dict[str, Any] | DictConfig | None = None,
) -> Callable:
    cfg = normalize_ensemble_conditioning_cfg(ensemble_cfg)

    def provider(
        *,
        batch: dict[str, Any],
        sampling_inputs: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        return compute_ensemble_potts_params(
            denoiser=denoiser,
            batch=batch,
            sampling_inputs=sampling_inputs,
            ensemble_cfg=cfg,
            cif_save_cfg=cif_save_cfg,
        )

    return provider


def normalize_ensemble_conditioning_cfg(
    cfg: dict[str, Any] | DictConfig | None,
) -> dict[str, Any]:
    if cfg is None:
        cfg_dict: dict[str, Any] = {}
    elif isinstance(cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        cfg_dict = dict(cfg)

    noise_std_raw = cfg_dict.get("noise_std", {}) or {}
    if isinstance(noise_std_raw, (int, float)):
        noise_std = {
            "protein": float(noise_std_raw),
            "metal": float(noise_std_raw),
            "nonpolymer": float(noise_std_raw),
        }
    else:
        noise_std = dict(noise_std_raw)

    normalized = {
        "enabled": bool(cfg_dict.get("enabled", False)),
        "num_ensembles": int(cfg_dict.get("num_ensembles", 1)),
        "reduce": str(cfg_dict.get("reduce", "mean")),
        "noise_seed": cfg_dict.get("noise_seed", None),
        "save_noisy_inputs_dir": cfg_dict.get("save_noisy_inputs_dir", None),
        "noise_std": {
            "protein": float(noise_std.get("protein", 0.0)),
            "metal": float(noise_std.get("metal", 0.0)),
            "nonpolymer": float(noise_std.get("nonpolymer", 0.0)),
        },
    }

    if normalized["num_ensembles"] < 1:
        raise ValueError(
            "ensemble_conditioning.num_ensembles must be >= 1, "
            f"got {normalized['num_ensembles']}"
        )
    if normalized["reduce"] not in {"mean", "sqrt"}:
        raise ValueError(
            "ensemble_conditioning.reduce must be 'mean' or 'sqrt', "
            f"got {normalized['reduce']!r}"
        )
    for key, value in normalized["noise_std"].items():
        if value < 0:
            raise ValueError(
                f"ensemble_conditioning.noise_std.{key} must be non-negative, got {value}"
            )

    return normalized


def compute_ensemble_potts_params(
    *,
    denoiser: Any,
    batch: dict[str, Any],
    sampling_inputs: dict[str, Any],
    ensemble_cfg: dict[str, Any] | DictConfig,
    cif_save_cfg: dict[str, Any] | DictConfig | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    cfg = normalize_ensemble_conditioning_cfg(ensemble_cfg)
    if "tied_sampling_ids" in batch:
        raise NotImplementedError(
            "ensemble_conditioning is not supported for batches that already "
            "contain tied_sampling_ids"
        )

    num_ensembles = cfg["num_ensembles"]
    repeated_batch = repeat_batch_for_ensembles(batch, num_ensembles)
    repeated_batch = apply_ensemble_noise(repeated_batch, cfg)

    _save_noisy_inputs_if_requested(
        repeated_batch,
        cfg,
        cif_save_cfg=cif_save_cfg,
    )

    potts_decoder_aux, repeated_batch, sampling_inputs = denoiser.compute_potts_params(
        repeated_batch,
        sampling_inputs,
    )
    original_batch_size = batch["restype"].shape[0]
    inverse = torch.arange(
        original_batch_size,
        device=potts_decoder_aux["h"].device,
    ).repeat_interleave(num_ensembles)
    tied_sampling_inputs = {
        "inverse": inverse,
        "unique_ids": torch.arange(original_batch_size, device=inverse.device),
    }
    potts_decoder_aux = _aggregate_potts_params(
        potts_decoder_aux,
        tied_sampling_inputs,
        reduce=cfg["reduce"],
    )

    representative_indices = list(range(0, original_batch_size * num_ensembles, num_ensembles))
    representative_batch = slice_feats(repeated_batch, representative_indices)
    output_batch = dict(batch)
    output_batch["protein_residue_node_mask"] = representative_batch["protein_residue_node_mask"]
    output_batch["token_exists_mask"] = representative_batch["token_exists_mask"]
    return potts_decoder_aux, output_batch, sampling_inputs


def repeat_batch_for_ensembles(
    batch: dict[str, Any],
    num_ensembles: int,
) -> dict[str, Any]:
    repeated = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            repeated[key] = value.repeat_interleave(num_ensembles, dim=0)
        elif isinstance(value, list):
            repeated[key] = [
                item
                for item in value
                for _ in range(num_ensembles)
            ]
        else:
            repeated[key] = value
    return repeated


def apply_ensemble_noise(
    batch: dict[str, Any],
    ensemble_cfg: dict[str, Any] | DictConfig,
) -> dict[str, Any]:
    cfg = normalize_ensemble_conditioning_cfg(ensemble_cfg)
    _require_batch_keys(
        batch,
        [
            "coords",
            "atom_pad_mask",
            "atom_resolved_mask",
            "atom_is_protein_chain",
            "atom_is_metal_chain",
            "atom_is_small_molecule_chain",
            "atom_is_prot_std_aa",
            "atom_array",
            "token_pad_mask",
        ],
    )

    coords = batch["coords"]
    noise_scale = _atom_noise_scale(batch, cfg).to(device=coords.device, dtype=coords.dtype)
    valid_atom_mask = (
        batch["atom_pad_mask"].to(dtype=coords.dtype)
        * batch["atom_resolved_mask"].to(dtype=coords.dtype)
    )
    noise_scale = noise_scale * valid_atom_mask

    if torch.max(noise_scale).item() == 0.0:
        noised_coords = coords.clone()
    else:
        generator = torch.Generator(device=coords.device)
        seed = cfg["noise_seed"]
        if seed is None:
            seed = torch.initial_seed()
        generator.manual_seed(int(seed))
        noise = torch.randn(
            coords.shape,
            device=coords.device,
            dtype=coords.dtype,
            generator=generator,
        )
        noised_coords = coords + noise * noise_scale.unsqueeze(-1)

    noised_batch = dict(batch)
    noised_batch["noised_coords"] = noised_coords
    _recompute_noised_backbone_fields(noised_batch)
    return noised_batch


def _raw_ensemble_cfg(sampling_inputs: dict[str, Any]) -> dict[str, Any] | None:
    potts_sampling_cfg = sampling_inputs.get("potts_sampling_cfg", {}) or {}
    return potts_sampling_cfg.get("ensemble_conditioning", None)


def _atom_noise_scale(
    batch: dict[str, Any],
    cfg: dict[str, Any],
) -> torch.Tensor:
    noise_std = cfg["noise_std"]
    protein_mask = batch["atom_is_protein_chain"].float()
    metal_mask = batch["atom_is_metal_chain"].float()
    nonpolymer_mask = batch["atom_is_small_molecule_chain"].float()
    return (
        protein_mask * noise_std["protein"]
        + metal_mask * noise_std["metal"]
        + nonpolymer_mask * noise_std["nonpolymer"]
    )


def _recompute_noised_backbone_fields(batch: dict[str, Any]) -> None:
    coords = batch["noised_coords"]
    batch_size, num_tokens, _ = batch["noised_ca_coords"].shape
    device = coords.device
    dtype = coords.dtype

    ca_coords = torch.zeros((batch_size, num_tokens, 3), device=device, dtype=dtype)
    n_coords = torch.zeros_like(ca_coords)
    c_coords = torch.zeros_like(ca_coords)
    o_coords = torch.zeros_like(ca_coords)
    pseudo_cb_coords = torch.zeros_like(ca_coords)

    for batch_idx, atom_array in enumerate(batch["atom_array"]):
        n_atoms = len(atom_array)
        atom_names = np.asarray(atom_array.atom_name).astype(str)
        occupancies = np.asarray(atom_array.occupancy) > 0
        token_ids = np.asarray(atom_array.token_id).astype(int)
        protein_std_atom_mask = (
            batch["atom_is_prot_std_aa"][batch_idx, :n_atoms]
            .bool()
            .detach()
            .cpu()
            .numpy()
        )

        candidate_tokens = sorted(set(token_ids[protein_std_atom_mask & occupancies].tolist()))
        for token_idx in candidate_tokens:
            if token_idx < 0 or token_idx >= num_tokens:
                continue
            token_mask = protein_std_atom_mask & occupancies & (token_ids == token_idx)
            atom_indices_by_name = {
                atom_name: np.flatnonzero(token_mask & (atom_names == atom_name))
                for atom_name in ("N", "CA", "C", "O")
            }
            if any(len(indices) != 1 for indices in atom_indices_by_name.values()):
                continue

            n_coord = coords[batch_idx, int(atom_indices_by_name["N"][0])]
            ca_coord = coords[batch_idx, int(atom_indices_by_name["CA"][0])]
            c_coord = coords[batch_idx, int(atom_indices_by_name["C"][0])]
            o_coord = coords[batch_idx, int(atom_indices_by_name["O"][0])]
            b_vec = ca_coord - n_coord
            c_vec = c_coord - ca_coord
            a_vec = torch.cross(b_vec, c_vec, dim=-1)
            pseudo_cb = -0.58273431 * a_vec + 0.56802827 * b_vec - 0.54067466 * c_vec + ca_coord

            n_coords[batch_idx, token_idx] = n_coord
            ca_coords[batch_idx, token_idx] = ca_coord
            c_coords[batch_idx, token_idx] = c_coord
            o_coords[batch_idx, token_idx] = o_coord
            pseudo_cb_coords[batch_idx, token_idx] = pseudo_cb

    batch["noised_ca_coords"] = ca_coords
    batch["noised_n_coords"] = n_coords
    batch["noised_c_coords"] = c_coords
    batch["noised_o_coords"] = o_coords
    batch["noised_pseudo_cb_coords"] = pseudo_cb_coords


def _save_noisy_inputs_if_requested(
    batch: dict[str, Any],
    cfg: dict[str, Any],
    *,
    cif_save_cfg: dict[str, Any] | DictConfig | None,
) -> None:
    save_root_raw = cfg.get("save_noisy_inputs_dir", None)
    if save_root_raw is None:
        return

    save_root = Path(str(save_root_raw))
    num_ensembles = cfg["num_ensembles"]
    if len(batch["atom_array"]) % num_ensembles != 0:
        raise ValueError("Repeated batch size is not divisible by num_ensembles")

    label = _ensemble_label(cfg)
    num_examples = len(batch["atom_array"]) // num_ensembles
    for example_idx in range(num_examples):
        first_member_idx = example_idx * num_ensembles
        example_id = str(batch["example_id"][first_member_idx])
        out_dir = save_root / example_id / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for member_idx in range(num_ensembles):
            batch_idx = first_member_idx + member_idx
            atom_array = batch["atom_array"][batch_idx].copy()
            n_atoms = len(atom_array)
            atom_array.coord = (
                batch["noised_coords"][batch_idx, :n_atoms]
                .detach()
                .cpu()
                .numpy()
            )
            out_file = out_dir / f"ensemble_{member_idx:03d}.cif"
            save_cif_file(atom_array, out_file, cif_save_cfg=cif_save_cfg)


def _ensemble_label(cfg: dict[str, Any]) -> str:
    noise_std = cfg["noise_std"]
    values = [
        noise_std["protein"],
        noise_std["metal"],
        noise_std["nonpolymer"],
    ]
    if values[0] == values[1] == values[2]:
        return f"M{cfg['num_ensembles']}_std{_format_float_label(values[0])}"
    return (
        f"M{cfg['num_ensembles']}"
        f"_p{_format_float_label(noise_std['protein'])}"
        f"_m{_format_float_label(noise_std['metal'])}"
        f"_n{_format_float_label(noise_std['nonpolymer'])}"
    )


def _format_float_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _require_batch_keys(batch: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in batch]
    missing.extend([key for key in _ENSEMBLE_NOISE_KEYS if key not in batch])
    if missing:
        raise KeyError(
            "Batch is missing required keys for ensemble conditioning: "
            + ", ".join(sorted(set(missing)))
        )
