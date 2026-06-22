from __future__ import annotations

import copy
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

DEFAULT_ENSEMBLE_CONDITIONING_CFG: dict[str, Any] = {
    "enabled": False,
    "total_members": 1,
    "weights": {
        "scheme": "mean",
        "ref_weight": 0.7,
        "decoy_total_weight": 0.3,
    },
    "noise_seed": None,
    "save_noisy_inputs_dir": None,
    "protein": {
        "mode": "gaussian_noise",
        "noise_std": 0.0,
    },
    "metal": {
        "mode": "gaussian_noise",
        "noise_std": 0.0,
    },
    "small_molecule": {
        "mode": "gaussian_noise",
        "noise_std": 0.0,
        "num_conformer_candidates": 50,
        "rmsd_cluster_cutoff": 2.0,
        "seed": None,
        "num_threads": 1,
        "uff_optimize": True,
        "clash_target_atoms": "all_protein",
        "vdw_overlap_cutoff": 0.5,
        "exclude_clashing_decoys": False,
        "pharm_retrieval": {
            "cif_root": None,
            "selected_queries_tsv": None,
            "rank_indices": [],
            "query_pn_unit_iids": None,
        },
    },
    "noise_std": {
        "protein": 0.0,
        "metal": 0.0,
        "nonpolymer": 0.0,
    },
}


def ensemble_conditioning_enabled(sampling_inputs: dict[str, Any]) -> bool:
    cfg = _raw_ensemble_cfg(sampling_inputs)
    return bool(cfg) and _as_bool(cfg.get("enabled", False))


def ligand_conformer_conditioning_enabled(sampling_inputs: dict[str, Any]) -> bool:
    cfg = _raw_ensemble_cfg(sampling_inputs)
    if not cfg:
        return False
    normalized = normalize_ensemble_conditioning_cfg(cfg)
    return (
        bool(normalized["enabled"])
        and normalized["small_molecule"]["mode"] == "ligand_conformer"
    )


def pharm_retrieval_conditioning_enabled(sampling_inputs: dict[str, Any]) -> bool:
    cfg = _raw_ensemble_cfg(sampling_inputs)
    if not cfg:
        return False
    normalized = normalize_ensemble_conditioning_cfg(cfg)
    return (
        bool(normalized["enabled"])
        and normalized["small_molecule"]["mode"] == "pharm_retrieval"
    )


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

    normalized = copy.deepcopy(DEFAULT_ENSEMBLE_CONDITIONING_CFG)
    _reject_legacy_ensemble_keys(cfg_dict)

    noise_std_raw = cfg_dict.get("noise_std", {}) or {}
    if isinstance(noise_std_raw, (int, float)):
        noise_std = {
            "protein": float(noise_std_raw),
            "metal": float(noise_std_raw),
            "nonpolymer": float(noise_std_raw),
        }
    else:
        noise_std = dict(noise_std_raw)
    weights_raw = cfg_dict.get("weights", {}) or {}
    if not isinstance(weights_raw, dict):
        raise ValueError("ensemble_conditioning.weights must be a mapping")
    weights = dict(weights_raw)

    normalized["enabled"] = _as_bool(cfg_dict.get("enabled", normalized["enabled"]))
    total_members_raw = cfg_dict.get("total_members", normalized["total_members"])
    normalized["total_members"] = int(total_members_raw)
    normalized["weights"]["scheme"] = str(
        weights.get("scheme", normalized["weights"]["scheme"])
    )
    normalized["weights"]["ref_weight"] = float(
        weights.get("ref_weight", normalized["weights"]["ref_weight"])
    )
    normalized["weights"]["decoy_total_weight"] = float(
        weights.get(
            "decoy_total_weight",
            normalized["weights"]["decoy_total_weight"],
        )
    )
    normalized["noise_seed"] = cfg_dict.get("noise_seed", normalized["noise_seed"])
    normalized["save_noisy_inputs_dir"] = cfg_dict.get(
        "save_noisy_inputs_dir",
        normalized["save_noisy_inputs_dir"],
    )

    _merge_entity_cfg(
        normalized,
        cfg_dict,
        entity_key="protein",
        flat_noise_key="protein",
        noise_std=noise_std,
    )
    _merge_entity_cfg(
        normalized,
        cfg_dict,
        entity_key="metal",
        flat_noise_key="metal",
        noise_std=noise_std,
    )
    _merge_entity_cfg(
        normalized,
        cfg_dict,
        entity_key="small_molecule",
        flat_noise_key="nonpolymer",
        noise_std=noise_std,
    )
    _refresh_legacy_noise_std(normalized)

    if normalized["total_members"] < 1:
        raise ValueError(
            "ensemble_conditioning.total_members must be >= 1, "
            f"got {normalized['total_members']}"
        )
    weights_cfg = normalized["weights"]
    if weights_cfg["scheme"] not in {"mean", "sqrt", "weighted_mean"}:
        raise ValueError(
            "ensemble_conditioning.weights.scheme must be 'mean', 'sqrt', "
            f"or 'weighted_mean', got {weights_cfg['scheme']!r}"
        )
    if weights_cfg["ref_weight"] < 0:
        raise ValueError(
            "ensemble_conditioning.weights.ref_weight must be non-negative, "
            f"got {weights_cfg['ref_weight']}"
        )
    if weights_cfg["decoy_total_weight"] < 0:
        raise ValueError(
            "ensemble_conditioning.weights.decoy_total_weight must be non-negative, "
            f"got {weights_cfg['decoy_total_weight']}"
        )
    if weights_cfg["ref_weight"] + weights_cfg["decoy_total_weight"] <= 0:
        raise ValueError("ensemble_conditioning weights must have positive total")
    _validate_entity_cfg(
        normalized,
        "protein",
        allowed_modes={"gaussian_noise", "none"},
    )
    _validate_entity_cfg(normalized, "metal", allowed_modes={"gaussian_noise", "none"})
    _validate_entity_cfg(
        normalized,
        "small_molecule",
        allowed_modes={"gaussian_noise", "ligand_conformer", "pharm_retrieval", "none"},
    )
    if (
        weights_cfg["scheme"] == "weighted_mean"
        and normalized["small_molecule"]["mode"] != "ligand_conformer"
    ):
        raise ValueError(
            "ensemble_conditioning.weights.scheme='weighted_mean' requires "
            "small_molecule.mode='ligand_conformer'"
        )
    small_molecule_cfg = normalized["small_molecule"]
    pharm_cfg = small_molecule_cfg["pharm_retrieval"]
    if small_molecule_cfg["mode"] == "pharm_retrieval":
        _validate_pharm_retrieval_cfg(pharm_cfg)
    if int(small_molecule_cfg["num_conformer_candidates"]) < 1:
        raise ValueError(
            "ensemble_conditioning.small_molecule.num_conformer_candidates "
            "must be >= 1"
        )
    if float(small_molecule_cfg["rmsd_cluster_cutoff"]) <= 0:
        raise ValueError(
            "ensemble_conditioning.small_molecule.rmsd_cluster_cutoff "
            "must be positive"
        )
    if int(small_molecule_cfg["num_threads"]) < 1:
        raise ValueError(
            "ensemble_conditioning.small_molecule.num_threads must be >= 1"
        )
    if float(small_molecule_cfg["vdw_overlap_cutoff"]) < 0:
        raise ValueError(
            "ensemble_conditioning.small_molecule.vdw_overlap_cutoff "
            "must be non-negative"
        )
    if small_molecule_cfg["clash_target_atoms"] not in {
        "sidechain",
        "backbone",
        "all_protein",
    }:
        raise ValueError(
            "ensemble_conditioning.small_molecule.clash_target_atoms must be "
            "'sidechain', 'backbone', or 'all_protein'"
        )
    for key, value in normalized["noise_std"].items():
        if float(value) < 0:
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
        noised_batch = apply_ensemble_noise(batch, cfg)
        return denoiser.compute_potts_params(
            noised_batch,
            sampling_inputs,
        )

    total_members = cfg["total_members"]
    repeated_batch = repeat_batch_for_ensembles(batch, total_members)
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
    ).repeat_interleave(total_members)
    tied_sampling_inputs = {
        "inverse": inverse,
        "unique_ids": torch.arange(original_batch_size, device=inverse.device),
    }
    potts_decoder_aux = _aggregate_potts_params(
        potts_decoder_aux,
        tied_sampling_inputs,
        reduce=cfg["weights"]["scheme"],
    )

    representative_indices = list(
        range(0, original_batch_size * total_members, total_members)
    )
    representative_batch = slice_feats(repeated_batch, representative_indices)
    output_batch = dict(batch)
    output_batch["protein_residue_node_mask"] = representative_batch["protein_residue_node_mask"]
    output_batch["token_exists_mask"] = representative_batch["token_exists_mask"]
    return potts_decoder_aux, output_batch, sampling_inputs


def repeat_batch_for_ensembles(
    batch: dict[str, Any],
    total_members: int,
) -> dict[str, Any]:
    repeated = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            repeated[key] = value.repeat_interleave(total_members, dim=0)
        elif isinstance(value, list):
            repeated[key] = [
                item
                for item in value
                for _ in range(total_members)
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _reject_legacy_ensemble_keys(cfg_dict: dict[str, Any]) -> None:
    legacy_keys = {
        "num_ensembles": "total_members",
        "reduce": "weights.scheme",
        "original_weight": "weights.ref_weight",
        "decoy_total_weight": "weights.decoy_total_weight",
    }
    present = [
        f"{key} -> {replacement}"
        for key, replacement in legacy_keys.items()
        if key in cfg_dict
    ]
    if present:
        raise ValueError(
            "Unsupported legacy ensemble_conditioning keys: "
            + ", ".join(present)
        )


def _merge_entity_cfg(
    normalized: dict[str, Any],
    cfg_dict: dict[str, Any],
    *,
    entity_key: str,
    flat_noise_key: str,
    noise_std: dict[str, Any],
) -> None:
    entity_cfg = dict(cfg_dict.get(entity_key, {}) or {})
    default_entity = normalized[entity_key]
    default_entity["mode"] = str(entity_cfg.get("mode", default_entity["mode"]))
    if "noise_std" in entity_cfg:
        default_entity["noise_std"] = float(entity_cfg["noise_std"])
    else:
        default_entity["noise_std"] = float(
            noise_std.get(flat_noise_key, default_entity["noise_std"])
        )
    if entity_key != "small_molecule":
        return

    for key in (
        "num_conformer_candidates",
        "num_threads",
    ):
        default_entity[key] = int(entity_cfg.get(key, default_entity[key]))
    for key in (
        "rmsd_cluster_cutoff",
        "vdw_overlap_cutoff",
    ):
        default_entity[key] = float(entity_cfg.get(key, default_entity[key]))
    default_entity["seed"] = entity_cfg.get("seed", default_entity["seed"])
    default_entity["uff_optimize"] = _as_bool(
        entity_cfg.get("uff_optimize", default_entity["uff_optimize"])
    )
    default_entity["exclude_clashing_decoys"] = _as_bool(
        entity_cfg.get(
            "exclude_clashing_decoys",
            default_entity["exclude_clashing_decoys"],
        )
    )
    default_entity["clash_target_atoms"] = str(
        entity_cfg.get("clash_target_atoms", default_entity["clash_target_atoms"])
    )
    pharm_cfg = dict(entity_cfg.get("pharm_retrieval", {}) or {})
    default_entity["pharm_retrieval"] = dict(default_entity["pharm_retrieval"])
    default_entity["pharm_retrieval"]["cif_root"] = pharm_cfg.get(
        "cif_root",
        default_entity["pharm_retrieval"]["cif_root"],
    )
    default_entity["pharm_retrieval"]["selected_queries_tsv"] = pharm_cfg.get(
        "selected_queries_tsv",
        default_entity["pharm_retrieval"]["selected_queries_tsv"],
    )
    default_entity["pharm_retrieval"]["rank_indices"] = pharm_cfg.get(
        "rank_indices",
        default_entity["pharm_retrieval"]["rank_indices"],
    )
    default_entity["pharm_retrieval"]["query_pn_unit_iids"] = pharm_cfg.get(
        "query_pn_unit_iids",
        default_entity["pharm_retrieval"]["query_pn_unit_iids"],
    )


def _refresh_legacy_noise_std(normalized: dict[str, Any]) -> None:
    normalized["noise_std"]["protein"] = (
        normalized["protein"]["noise_std"]
        if normalized["protein"]["mode"] == "gaussian_noise"
        else 0.0
    )
    normalized["noise_std"]["metal"] = (
        normalized["metal"]["noise_std"]
        if normalized["metal"]["mode"] == "gaussian_noise"
        else 0.0
    )
    normalized["noise_std"]["nonpolymer"] = (
        normalized["small_molecule"]["noise_std"]
        if normalized["small_molecule"]["mode"] == "gaussian_noise"
        else 0.0
    )


def _validate_entity_cfg(
    normalized: dict[str, Any],
    entity_key: str,
    *,
    allowed_modes: set[str],
) -> None:
    entity_cfg = normalized[entity_key]
    if entity_cfg["mode"] not in allowed_modes:
        allowed = ", ".join(sorted(allowed_modes))
        raise ValueError(
            f"ensemble_conditioning.{entity_key}.mode must be one of "
            f"{allowed}; got {entity_cfg['mode']!r}"
        )
    if float(entity_cfg["noise_std"]) < 0:
        raise ValueError(
            f"ensemble_conditioning.{entity_key}.noise_std must be "
            f"non-negative, got {entity_cfg['noise_std']}"
        )


def _validate_pharm_retrieval_cfg(pharm_cfg: dict[str, Any]) -> None:
    rank_indices = pharm_cfg.get("rank_indices", [])
    if not isinstance(rank_indices, (list, tuple)):
        raise ValueError(
            "ensemble_conditioning.small_molecule.pharm_retrieval.rank_indices "
            "must be a list of non-negative integers"
        )
    normalized_ranks = []
    for value in rank_indices:
        if isinstance(value, bool):
            raise ValueError(
                "ensemble_conditioning.small_molecule.pharm_retrieval.rank_indices "
                "cannot contain booleans"
            )
        try:
            rank = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "ensemble_conditioning.small_molecule.pharm_retrieval.rank_indices "
                "must be non-negative integers"
            ) from exc
        if rank < 0:
            raise ValueError(
                "ensemble_conditioning.small_molecule.pharm_retrieval.rank_indices "
                f"must be non-negative, got {rank}"
            )
        normalized_ranks.append(rank)
    duplicates = sorted({rank for rank in normalized_ranks if normalized_ranks.count(rank) > 1})
    if duplicates:
        raise ValueError(
            "ensemble_conditioning.small_molecule.pharm_retrieval.rank_indices "
            f"contains duplicate ranks: {duplicates}"
        )
    pharm_cfg["rank_indices"] = normalized_ranks


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
    total_members = cfg["total_members"]
    if len(batch["atom_array"]) % total_members != 0:
        raise ValueError("Repeated batch size is not divisible by total_members")

    label = _ensemble_label(cfg)
    num_examples = len(batch["atom_array"]) // total_members
    for example_idx in range(num_examples):
        first_member_idx = example_idx * total_members
        example_id = str(batch["example_id"][first_member_idx])
        out_dir = save_root / example_id / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for member_idx in range(total_members):
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
        return f"M{cfg['total_members']}_std{_format_float_label(values[0])}"
    return (
        f"M{cfg['total_members']}"
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
