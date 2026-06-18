from pathlib import Path
from types import SimpleNamespace

import atomworks.enums as aw_enums
import numpy as np
import torch
import pandas as pd
import pytest
from atomworks.constants import METAL_ELEMENTS as ATOMWORKS_METAL_ELEMENTS
from biotite.structure import AtomArray
from omegaconf import OmegaConf
from rdkit import Chem

from allatom_design.data.transform.ligand_conformers import (
    compute_ligand_protein_clash_metrics,
    find_query_small_molecule_ligands,
    select_target_ligand,
    _conformer_coords,
    _generate_candidate_mol,
    _select_native_threshold_conformers,
)
from allatom_design.data.const import METAL_ELEMENTS
from allatom_design.eval.utils.cfg_utils import (
    get_stage2_potts_only_cond,
    guidance_is_enabled,
    resolve_sampling_cfg,
)
from allatom_design.eval.utils import ensemble_conditioning as ensemble_conditioning_module
from allatom_design.eval.utils import ligand_conformer_retrieval as ligand_conformer_retrieval_module
from allatom_design.eval.utils.ensemble_conditioning import (
    DEFAULT_ENSEMBLE_CONDITIONING_CFG,
    apply_ensemble_noise,
    ligand_conformer_conditioning_enabled,
    normalize_ensemble_conditioning_cfg,
    pharm_retrieval_conditioning_enabled,
    repeat_batch_for_ensembles,
)
from allatom_design.eval.utils.ligand_conformer_retrieval import (
    LigandConformerStagingResult,
    compute_ligand_conformer_member_coefficients,
    stage_ligand_conformer_ensembles,
    _manifest_row,
)
from allatom_design.eval.utils.pharm_retrieval import (
    stage_pharm_retrieval_ensembles,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_atom_array(n_atoms: int = 4) -> AtomArray:
    atom_array = AtomArray(n_atoms)
    atom_array.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ][:n_atoms],
        dtype=np.float32,
    )
    atom_array.atom_name = np.array(["N", "CA", "C", "O"][:n_atoms])
    atom_array.occupancy = np.ones(n_atoms, dtype=np.float32)
    atom_array.set_annotation("token_id", np.zeros(n_atoms, dtype=np.int64))
    return atom_array


def _make_batch() -> dict:
    atom_array = _make_atom_array()
    coords = torch.as_tensor(atom_array.coord, dtype=torch.float32).unsqueeze(0)
    batch = {
        "coords": coords,
        "noised_coords": coords.clone(),
        "noised_ca_coords": torch.zeros(1, 1, 3),
        "noised_n_coords": torch.zeros(1, 1, 3),
        "noised_c_coords": torch.zeros(1, 1, 3),
        "noised_o_coords": torch.zeros(1, 1, 3),
        "noised_pseudo_cb_coords": torch.zeros(1, 1, 3),
        "atom_pad_mask": torch.ones(1, 4),
        "atom_resolved_mask": torch.ones(1, 4),
        "atom_is_protein_chain": torch.ones(1, 4),
        "atom_is_metal_chain": torch.zeros(1, 4),
        "atom_is_small_molecule_chain": torch.zeros(1, 4),
        "atom_is_prot_std_aa": torch.ones(1, 4),
        "token_pad_mask": torch.ones(1, 1),
        "atom_array": [atom_array],
        "example_id": ["example"],
    }
    return batch


def test_normalize_ensemble_conditioning_cfg_uses_canonical_defaults():
    assert normalize_ensemble_conditioning_cfg(None) == DEFAULT_ENSEMBLE_CONDITIONING_CFG


def test_normalize_ensemble_conditioning_cfg_merges_partial_config():
    normalized = normalize_ensemble_conditioning_cfg(
        {
            "enabled": True,
            "total_members": 4,
            "weights": {"scheme": "sqrt"},
            "noise_std": {
                "protein": 0.2,
            },
        }
    )

    assert normalized == {
        **DEFAULT_ENSEMBLE_CONDITIONING_CFG,
        "enabled": True,
        "total_members": 4,
        "weights": {
            **DEFAULT_ENSEMBLE_CONDITIONING_CFG["weights"],
            "scheme": "sqrt",
        },
        "protein": {
            **DEFAULT_ENSEMBLE_CONDITIONING_CFG["protein"],
            "noise_std": 0.2,
        },
        "noise_std": {
            "protein": 0.2,
            "metal": 0.0,
            "nonpolymer": 0.0,
        },
    }


@pytest.mark.parametrize(
    "legacy_key",
    ["num_ensembles", "reduce", "original_weight", "decoy_total_weight"],
)
def test_normalize_ensemble_conditioning_cfg_rejects_legacy_public_keys(legacy_key):
    with pytest.raises(
        ValueError,
        match="Unsupported legacy ensemble_conditioning keys",
    ):
        normalize_ensemble_conditioning_cfg({legacy_key: 1})


def test_normalize_ensemble_conditioning_cfg_expands_scalar_noise_std():
    normalized = normalize_ensemble_conditioning_cfg({"noise_std": 0.3})

    assert normalized["noise_std"] == {
        "protein": 0.3,
        "metal": 0.3,
        "nonpolymer": 0.3,
    }
    assert normalized["protein"]["noise_std"] == 0.3
    assert normalized["metal"]["noise_std"] == 0.3
    assert normalized["small_molecule"]["noise_std"] == 0.3


def test_normalize_ensemble_conditioning_cfg_supports_ligand_conformer_mode():
    normalized = normalize_ensemble_conditioning_cfg(
        {
            "enabled": True,
            "total_members": 8,
            "weights": {
                "scheme": "weighted_mean",
                "ref_weight": 0.8,
                "decoy_total_weight": 0.2,
            },
            "protein": {"mode": "gaussian_noise", "noise_std": 0.1},
            "small_molecule": {
                "mode": "ligand_conformer",
                "num_conformer_candidates": 50,
                "rmsd_cluster_cutoff": 2.0,
                "clash_target_atoms": "all_protein",
            },
        }
    )

    assert normalized["total_members"] == 8
    assert normalized["weights"]["scheme"] == "weighted_mean"
    assert normalized["weights"]["ref_weight"] == 0.8
    assert normalized["weights"]["decoy_total_weight"] == 0.2
    assert normalized["small_molecule"]["mode"] == "ligand_conformer"
    assert normalized["noise_std"] == {
        "protein": 0.1,
        "metal": 0.0,
        "nonpolymer": 0.0,
    }


def test_normalize_ensemble_conditioning_cfg_supports_pharm_retrieval_mode():
    normalized = normalize_ensemble_conditioning_cfg(
        {
            "enabled": True,
            "total_members": 3,
            "small_molecule": {
                "mode": "pharm_retrieval",
                "pharm_retrieval": {
                    "cif_root": "/tmp/cifs",
                    "selected_queries_tsv": "/tmp/selected_queries.tsv",
                    "rank_indices": [0, 2],
                    "query_pn_unit_iids": ["L_1"],
                },
            },
        }
    )

    assert normalized["small_molecule"]["mode"] == "pharm_retrieval"
    assert normalized["small_molecule"]["pharm_retrieval"] == {
        "cif_root": "/tmp/cifs",
        "selected_queries_tsv": "/tmp/selected_queries.tsv",
        "rank_indices": [0, 2],
        "query_pn_unit_iids": ["L_1"],
    }
    assert normalized["noise_std"] == {
        "protein": 0.0,
        "metal": 0.0,
        "nonpolymer": 0.0,
    }


def test_weighted_mean_is_ligand_conformer_only():
    with pytest.raises(
        ValueError,
        match="requires small_molecule.mode='ligand_conformer'",
    ):
        normalize_ensemble_conditioning_cfg(
            {
                "weights": {"scheme": "weighted_mean"},
                "small_molecule": {"mode": "gaussian_noise"},
            }
        )


@pytest.mark.parametrize(
    "weights",
    [
        {"ref_weight": -0.1, "decoy_total_weight": 0.3},
        {"ref_weight": 0.7, "decoy_total_weight": -0.1},
        {"ref_weight": 0.0, "decoy_total_weight": 0.0},
    ],
)
def test_normalize_ensemble_conditioning_cfg_validates_weight_values(weights):
    with pytest.raises(ValueError, match="weights"):
        normalize_ensemble_conditioning_cfg(
            {
                "total_members": 4,
                "weights": {
                    "scheme": "weighted_mean",
                    **weights,
                },
                "small_molecule": {"mode": "ligand_conformer"},
            }
        )


def test_ligand_conformer_conditioning_enabled_reads_sampling_inputs():
    sampling_inputs = {
        "potts_sampling_cfg": {
            "ensemble_conditioning": {
                "enabled": True,
                "small_molecule": {"mode": "ligand_conformer"},
            }
        }
    }

    assert ligand_conformer_conditioning_enabled(sampling_inputs) is True


def test_pharm_retrieval_conditioning_enabled_reads_sampling_inputs():
    sampling_inputs = {
        "potts_sampling_cfg": {
            "ensemble_conditioning": {
                "enabled": True,
                "total_members": 3,
                "small_molecule": {
                    "mode": "pharm_retrieval",
                    "pharm_retrieval": {"rank_indices": [0, 2]},
                },
            }
        }
    }

    assert pharm_retrieval_conditioning_enabled(sampling_inputs) is True


@pytest.mark.parametrize(
    "config_path",
    [
        "allatom_design/configs/seq_des/elix_mpnn_inference.yaml",
        "allatom_design/configs_local/seq_des/elix_mpnn_inference.yaml",
    ],
)
def test_seq_des_yaml_ensemble_conditioning_defaults_match_runtime_defaults(config_path):
    cfg = OmegaConf.load(REPO_ROOT / config_path)

    assert (
        OmegaConf.to_container(
            cfg.potts_sampling_cfg.ensemble_conditioning,
            resolve=True,
        )
        == DEFAULT_ENSEMBLE_CONDITIONING_CFG
    )


def test_run_elix_resolved_sampling_ensemble_conditioning_defaults_match_runtime_defaults():
    cfg = OmegaConf.load(REPO_ROOT / "allatom_design/configs/eval/sampling/run_elix.yaml")
    cfg.sampling_cfg.base_cfg_path = str(
        REPO_ROOT / "allatom_design/configs/seq_des/elix_mpnn_inference.yaml"
    )

    resolved_sampling_cfg = resolve_sampling_cfg(cfg)

    assert (
        OmegaConf.to_container(
            resolved_sampling_cfg.potts_sampling_cfg.ensemble_conditioning,
            resolve=True,
        )
        == DEFAULT_ENSEMBLE_CONDITIONING_CFG
    )


def test_cfg_utils_reads_nested_stage_and_guidance_flags():
    assert get_stage2_potts_only_cond({}) is None
    assert (
        get_stage2_potts_only_cond(
            {
                "sampling_cfg": {
                    "overrides": {
                        "potts_sampling_cfg": {
                            "potts_only_cond": True,
                        },
                    },
                },
            }
        )
        is True
    )

    assert guidance_is_enabled(None) is False
    assert guidance_is_enabled({"enabled": True}) is True
    assert guidance_is_enabled({"enabled": "false"}) is False
    assert guidance_is_enabled({"sampling_cfg": {"guidance": {"enabled": True}}}) is True


def test_repeat_batch_for_ensembles_repeats_tensors_and_lists():
    batch = _make_batch()

    repeated = repeat_batch_for_ensembles(batch, 3)

    assert repeated["coords"].shape[0] == 3
    assert repeated["example_id"] == ["example", "example", "example"]
    torch.testing.assert_close(repeated["coords"][0], batch["coords"][0])
    torch.testing.assert_close(repeated["coords"][2], batch["coords"][0])


def test_apply_ensemble_noise_uses_category_specific_std():
    batch = _make_batch()
    batch["atom_is_protein_chain"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    batch["atom_is_metal_chain"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    batch["atom_is_small_molecule_chain"] = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    batch["atom_is_prot_std_aa"] = torch.zeros(1, 4)

    noised = apply_ensemble_noise(
        batch,
        {
            "enabled": True,
            "total_members": 1,
            "weights": {"scheme": "mean"},
            "noise_seed": 11,
            "noise_std": {
                "protein": 0.0,
                "metal": 0.2,
                "nonpolymer": 0.0,
            },
        },
    )

    delta = noised["noised_coords"] - batch["coords"]
    torch.testing.assert_close(delta[0, 0], torch.zeros(3))
    assert not torch.allclose(delta[0, 1], torch.zeros(3))
    torch.testing.assert_close(delta[0, 2], torch.zeros(3))
    torch.testing.assert_close(delta[0, 3], torch.zeros(3))


def test_apply_ensemble_noise_recomputes_token_backbone_fields_from_noised_atoms():
    batch = _make_batch()

    noised = apply_ensemble_noise(
        batch,
        {
            "enabled": True,
            "total_members": 1,
            "weights": {"scheme": "mean"},
            "noise_seed": 7,
            "noise_std": {
                "protein": 0.1,
                "metal": 0.0,
                "nonpolymer": 0.0,
            },
        },
    )

    noised_coords = noised["noised_coords"][0]
    torch.testing.assert_close(noised["noised_n_coords"][0, 0], noised_coords[0])
    torch.testing.assert_close(noised["noised_ca_coords"][0, 0], noised_coords[1])
    torch.testing.assert_close(noised["noised_c_coords"][0, 0], noised_coords[2])
    torch.testing.assert_close(noised["noised_o_coords"][0, 0], noised_coords[3])

    b_vec = noised_coords[1] - noised_coords[0]
    c_vec = noised_coords[2] - noised_coords[1]
    a_vec = torch.cross(b_vec, c_vec, dim=-1)
    expected_pseudo_cb = (
        -0.58273431 * a_vec
        + 0.56802827 * b_vec
        - 0.54067466 * c_vec
        + noised_coords[1]
    )
    torch.testing.assert_close(noised["noised_pseudo_cb_coords"][0, 0], expected_pseudo_cb)


def test_save_noisy_inputs_uses_total_members_in_label(monkeypatch, tmp_path):
    batch = repeat_batch_for_ensembles(_make_batch(), 2)
    saved_paths = []

    def fake_save_cif_file(atom_array, out_file, cif_save_cfg=None):
        saved_paths.append(Path(out_file))

    monkeypatch.setattr(
        ensemble_conditioning_module,
        "save_cif_file",
        fake_save_cif_file,
    )
    cfg = normalize_ensemble_conditioning_cfg(
        {
            "total_members": 2,
            "save_noisy_inputs_dir": str(tmp_path),
            "noise_std": 0.0,
        }
    )

    ensemble_conditioning_module._save_noisy_inputs_if_requested(
        batch,
        cfg,
        cif_save_cfg=None,
    )

    assert saved_paths == [
        tmp_path / "example" / "M2_std0" / "ensemble_000.cif",
        tmp_path / "example" / "M2_std0" / "ensemble_001.cif",
    ]


def _make_atom_array_for_clash() -> AtomArray:
    atom_array = AtomArray(4)
    atom_array.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    atom_array.atom_name = np.array(["C1", "N", "CA", "CB"])
    atom_array.element = np.array(["C", "N", "C", "C"])
    atom_array.res_name = np.array(["LIG", "ALA", "ALA", "ALA"])
    atom_array.hetero = np.array([True, False, False, False])
    atom_array.occupancy = np.ones(4, dtype=np.float32)
    atom_array.set_annotation(
        "chain_type",
        np.array(
            [
                aw_enums.ChainType.NON_POLYMER.value,
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                aw_enums.ChainType.POLYPEPTIDE_L.value,
            ]
        ),
    )
    atom_array.set_annotation("pn_unit_iid", np.array(["L_1", "A_1", "A_1", "A_1"]))
    return atom_array


def test_clash_target_atoms_flag_controls_protein_atom_set():
    atom_array = _make_atom_array_for_clash()
    ligand_mask = atom_array.pn_unit_iid == "L_1"

    all_metrics = compute_ligand_protein_clash_metrics(
        atom_array,
        ligand_mask=ligand_mask,
        clash_target_atoms="all_protein",
        vdw_overlap_cutoff=0.5,
    )
    sidechain_metrics = compute_ligand_protein_clash_metrics(
        atom_array,
        ligand_mask=ligand_mask,
        clash_target_atoms="sidechain",
        vdw_overlap_cutoff=0.5,
    )
    backbone_metrics = compute_ligand_protein_clash_metrics(
        atom_array,
        ligand_mask=ligand_mask,
        clash_target_atoms="backbone",
        vdw_overlap_cutoff=0.5,
    )

    assert all_metrics["has_clash"] is True
    assert sidechain_metrics["has_clash"] is True
    assert backbone_metrics["has_clash"] is False


def _make_atom_array_for_metal_ligand_selection() -> AtomArray:
    atom_array = AtomArray(2)
    atom_array.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    atom_array.atom_name = np.array(["U", "C1"])
    atom_array.element = np.array(["U", "C"])
    atom_array.res_name = np.array(["U", "LIG"])
    atom_array.hetero = np.ones(2, dtype=bool)
    atom_array.occupancy = np.ones(2, dtype=np.float32)
    atom_array.set_annotation(
        "chain_type",
        np.array(
            [
                aw_enums.ChainType.NON_POLYMER.value,
                aw_enums.ChainType.NON_POLYMER.value,
            ]
        ),
    )
    atom_array.set_annotation("pn_unit_iid", np.array(["U_1", "L_1"]))
    return atom_array


def _make_atom_array_for_metal_only_query() -> AtomArray:
    atom_array = AtomArray(5)
    atom_array.coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    atom_array.atom_name = np.array(["N", "CA", "C", "O", "CA"])
    atom_array.element = np.array(["N", "C", "C", "O", "CA"])
    atom_array.res_name = np.array(["ALA", "ALA", "ALA", "ALA", "CA"])
    atom_array.hetero = np.array([False, False, False, False, True])
    atom_array.occupancy = np.ones(5, dtype=np.float32)
    atom_array.set_annotation(
        "chain_type",
        np.array(
            [
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                aw_enums.ChainType.POLYPEPTIDE_L.value,
                aw_enums.ChainType.NON_POLYMER.value,
            ]
        ),
    )
    atom_array.set_annotation(
        "pn_unit_iid",
        np.array(["A_1", "A_1", "A_1", "A_1", "M_1"]),
    )
    return atom_array


def test_atomworks_only_metal_elements_are_excluded_from_ligand_selection():
    assert "U" in METAL_ELEMENTS
    atom_array = _make_atom_array_for_metal_ligand_selection()

    target = select_target_ligand(atom_array)

    assert target.pn_unit_iid == "L_1"
    with pytest.raises(ValueError, match="found 0"):
        select_target_ligand(atom_array, query_pn_unit_iids=["U_1"])


def test_query_small_molecule_ligand_helper_can_return_zero_for_metal_query():
    atom_array = _make_atom_array_for_metal_only_query()

    assert (
        find_query_small_molecule_ligands(
            atom_array,
            query_pn_unit_iids=["A_1", "M_1"],
        )
        == []
    )
    with pytest.raises(ValueError, match="found 0"):
        select_target_ligand(atom_array, query_pn_unit_iids=["A_1", "M_1"])


def test_metal_elements_constant_aliases_atomworks_source_of_truth():
    assert METAL_ELEMENTS is ATOMWORKS_METAL_ELEMENTS


def _heavy_atom_indices(mol: Chem.Mol) -> list[int]:
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]


def test_generate_candidate_mol_uses_distinct_nonzero_seed_per_candidate():
    mol = Chem.MolFromSmiles("CCCOCCCO")

    work_mol, metadata = _generate_candidate_mol(
        mol,
        num_candidates=5,
        seed=0,
        num_threads=1,
        uff_optimize=False,
    )

    assert metadata["generated_candidates"] == 5
    assert metadata["candidate_seed_start"] == 1
    assert metadata["candidate_seed_end"] == 5
    assert [entry["candidate_seed"] for entry in metadata["conformers"]] == [
        1,
        2,
        3,
        4,
        5,
    ]

    heavy_atom_indices = _heavy_atom_indices(work_mol)
    coords = np.stack(
        [
            _conformer_coords(conformer, heavy_atom_indices)
            for conformer in work_mol.GetConformers()
        ]
    )
    raw_rms_to_first = np.sqrt(((coords - coords[0]) ** 2).sum(axis=-1).mean(axis=-1))
    assert np.max(raw_rms_to_first) > 0.1


def test_select_native_threshold_conformers_uses_get_best_rms_cutoff():
    mol = Chem.MolFromSmiles("CCCOCCCO")
    work_mol, _ = _generate_candidate_mol(
        mol,
        num_candidates=4,
        seed=0,
        num_threads=1,
        uff_optimize=False,
    )
    heavy_atom_indices = _heavy_atom_indices(work_mol)
    native_coords = _conformer_coords(work_mol.GetConformer(0), heavy_atom_indices)

    selected, hit_count = _select_native_threshold_conformers(
        work_mol,
        atom_ids=heavy_atom_indices,
        native_coords=native_coords,
        rmsd_cutoff=1e-6,
        target_count=1,
        num_threads=1,
    )

    assert hit_count >= 1
    assert len(selected) == 1
    assert selected[0].position == 0
    assert selected[0].get_best_rms_to_native == pytest.approx(0.0, abs=1e-6)
    assert selected[0].aligned_coords.shape == native_coords.shape


def test_select_native_threshold_conformers_returns_empty_when_no_hit():
    mol = Chem.MolFromSmiles("CCCOCCCO")
    work_mol, _ = _generate_candidate_mol(
        mol,
        num_candidates=4,
        seed=0,
        num_threads=1,
        uff_optimize=False,
    )
    heavy_atom_indices = _heavy_atom_indices(work_mol)
    native_coords = _conformer_coords(work_mol.GetConformer(0), heavy_atom_indices)
    distorted_native_coords = native_coords * 5.0

    selected, hit_count = _select_native_threshold_conformers(
        work_mol,
        atom_ids=heavy_atom_indices,
        native_coords=distorted_native_coords,
        rmsd_cutoff=0.01,
        target_count=4,
        num_threads=1,
    )

    assert selected == []
    assert hit_count == 0


def test_ligand_conformer_manifest_records_native_threshold_selection_metadata():
    row = _manifest_row(
        target_sample_id="input",
        member_sample_id="input_ligconf_1",
        member_path=Path("/tmp/staged/input_ligconf_1.cif"),
        member_role="ligand_conformer_decoy",
        member_coefficient=0.3,
        target_ligand=SimpleNamespace(pn_unit_iid="L_1", res_name="LIG"),
        clash_metrics={
            "has_clash": False,
            "num_clashing_pairs": 0,
            "min_heavy_atom_distance": None,
            "max_vdw_overlap": None,
            "clash_target_atoms": "all_protein",
            "vdw_overlap_cutoff": 0.5,
        },
        warning="",
        cluster_id=None,
        generation_metadata={
            "conformer_selection_metric": "rdkit_get_best_rms_to_native_heavy",
            "conformer_selection_cutoff": 2.0,
        },
    )

    assert row["conformer_selection_metric"] == "rdkit_get_best_rms_to_native_heavy"
    assert row["conformer_selection_cutoff"] == 2.0
    assert row["cluster_id"] is None


def test_ligand_conformer_staging_weighted_mean_uses_weight_split(
    monkeypatch,
    tmp_path,
):
    atom_array = _make_atom_array()
    saved_paths = []
    clash_metrics = {
        "has_clash": False,
        "num_clashing_pairs": 0,
        "min_heavy_atom_distance": None,
        "max_vdw_overlap": None,
        "clash_target_atoms": "all_protein",
        "vdw_overlap_cutoff": 0.5,
    }

    def fake_load_example_with_parse(path, cif_parse_cfg):
        return {"atom_array": atom_array.copy()}

    def fake_save_cif_file(atom_array, out_file, cif_save_cfg=None):
        out_file = Path(out_file)
        saved_paths.append(out_file)
        out_file.write_text(out_file.stem)

    def fake_find_query_small_molecule_ligands(atom_array, query_pn_unit_iids=None):
        return [
            SimpleNamespace(
                heavy_mask=np.ones(len(atom_array), dtype=bool),
                pn_unit_iid="L_1",
                res_name="LIG",
            )
        ]

    def fake_generate_ligand_conformer_decoys(*args, **kwargs):
        decoys = [
            SimpleNamespace(
                atom_array=atom_array.copy(),
                clash_metrics=clash_metrics,
                rank=rank,
                cluster_id=rank,
                rdkit_conformer_id=rank,
                candidate_seed=100 + rank,
                rmsd_to_native=0.5 + rank * 0.1,
                atom_order_rmsd_to_native=0.6 + rank * 0.1,
            )
            for rank in range(2)
        ]
        return SimpleNamespace(
            decoys=decoys,
            metadata={"rdkit_native_threshold_hit_count": 2},
        )

    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "load_example_with_parse",
        fake_load_example_with_parse,
    )
    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "save_cif_file",
        fake_save_cif_file,
    )
    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "find_query_small_molecule_ligands",
        fake_find_query_small_molecule_ligands,
    )
    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "compute_ligand_protein_clash_metrics",
        lambda *args, **kwargs: clash_metrics,
    )
    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "generate_ligand_conformer_decoys",
        fake_generate_ligand_conformer_decoys,
    )

    staging = stage_ligand_conformer_ensembles(
        pdb_paths=["/inputs/input.cif"],
        out_dir=tmp_path,
        ensemble_cfg={
            "enabled": True,
            "total_members": 4,
            "weights": {
                "scheme": "weighted_mean",
                "ref_weight": 0.7,
                "decoy_total_weight": 0.3,
            },
            "small_molecule": {"mode": "ligand_conformer"},
        },
        sampling_inputs_df=pd.DataFrame([{"pdb_id": "input"}]),
        cif_parse_cfg=None,
        cif_save_cfg=None,
    )
    manifest = pd.read_csv(staging.manifest_path)

    assert [Path(path).name for path in staging.member_groups[0]] == [
        "input.cif",
        "input_ligconf_1.cif",
        "input_ligconf_2.cif",
        "input_ligconf_3.cif",
    ]
    assert [path.name for path in saved_paths] == [
        "input.cif",
        "input_ligconf_1.cif",
        "input_ligconf_2.cif",
        "input_ligconf_3.cif",
    ]
    assert manifest["member_role"].tolist() == [
        "original",
        "ligand_conformer_decoy",
        "ligand_conformer_decoy",
        "fallback_original_copy",
    ]
    assert manifest["member_coefficient"].tolist() == pytest.approx(
        [0.7, 0.1, 0.1, 0.1]
    )


def _patch_fake_ligand_conformer_staging_deps(
    monkeypatch,
    *,
    num_decoys: int,
    fail_if_generate_called: bool = False,
) -> tuple[list[Path], dict[str, object]]:
    atom_array = _make_atom_array()
    saved_paths = []
    clash_metrics = {
        "has_clash": False,
        "num_clashing_pairs": 0,
        "min_heavy_atom_distance": None,
        "max_vdw_overlap": None,
        "clash_target_atoms": "all_protein",
        "vdw_overlap_cutoff": 0.5,
    }

    def fake_save_cif_file(atom_array, out_file, cif_save_cfg=None):
        out_file = Path(out_file)
        saved_paths.append(out_file)
        out_file.write_text(out_file.stem)

    def fake_generate_ligand_conformer_decoys(*args, **kwargs):
        if fail_if_generate_called:
            raise AssertionError("conformer generation should not be called")
        decoys = [
            SimpleNamespace(
                atom_array=atom_array.copy(),
                clash_metrics=clash_metrics,
                rank=rank,
                cluster_id=rank,
                rdkit_conformer_id=rank,
                candidate_seed=100 + rank,
                rmsd_to_native=0.5 + rank * 0.1,
                atom_order_rmsd_to_native=0.6 + rank * 0.1,
            )
            for rank in range(num_decoys)
        ]
        return SimpleNamespace(
            decoys=decoys,
            metadata={"rdkit_native_threshold_hit_count": num_decoys},
        )

    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "load_example_with_parse",
        lambda path, cif_parse_cfg: {"atom_array": atom_array.copy()},
    )
    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "save_cif_file",
        fake_save_cif_file,
    )
    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "find_query_small_molecule_ligands",
        lambda atom_array, query_pn_unit_iids=None: [
            SimpleNamespace(
                heavy_mask=np.ones(len(atom_array), dtype=bool),
                pn_unit_iid="L_1",
                res_name="LIG",
            )
        ],
    )
    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "compute_ligand_protein_clash_metrics",
        lambda *args, **kwargs: clash_metrics,
    )
    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "generate_ligand_conformer_decoys",
        fake_generate_ligand_conformer_decoys,
    )
    return saved_paths, clash_metrics


def test_ligand_conformer_staging_weighted_mean_falls_back_for_missing_decoys(
    monkeypatch,
    tmp_path,
):
    saved_paths, _ = _patch_fake_ligand_conformer_staging_deps(
        monkeypatch,
        num_decoys=1,
    )

    staging = stage_ligand_conformer_ensembles(
        pdb_paths=["/inputs/input.cif"],
        out_dir=tmp_path,
        ensemble_cfg={
            "enabled": True,
            "total_members": 4,
            "weights": {
                "scheme": "weighted_mean",
                "ref_weight": 0.7,
                "decoy_total_weight": 0.3,
            },
            "small_molecule": {"mode": "ligand_conformer"},
        },
        sampling_inputs_df=pd.DataFrame([{"pdb_id": "input"}]),
        cif_parse_cfg=None,
        cif_save_cfg=None,
    )
    manifest = pd.read_csv(staging.manifest_path)

    assert [Path(path).name for path in staging.member_groups[0]] == [
        "input.cif",
        "input_ligconf_1.cif",
        "input_ligconf_2.cif",
        "input_ligconf_3.cif",
    ]
    assert [path.name for path in saved_paths] == [
        "input.cif",
        "input_ligconf_1.cif",
        "input_ligconf_2.cif",
        "input_ligconf_3.cif",
    ]
    assert manifest["member_role"].tolist() == [
        "original",
        "ligand_conformer_decoy",
        "fallback_original_copy",
        "fallback_original_copy",
    ]
    assert manifest["member_coefficient"].tolist() == pytest.approx(
        [0.7, 0.1, 0.1, 0.1]
    )
    assert manifest["warning"].str.contains(
        "fewer_ligand_conformer_decoys_than_requested: requested=3, selected=1"
    ).all()
    assert manifest["warning"].str.contains("fallback_original_copy: count=2").all()


def test_ligand_conformer_staging_original_only_skips_decoy_generation(
    monkeypatch,
    tmp_path,
):
    saved_paths, _ = _patch_fake_ligand_conformer_staging_deps(
        monkeypatch,
        num_decoys=0,
        fail_if_generate_called=True,
    )

    staging = stage_ligand_conformer_ensembles(
        pdb_paths=["/inputs/input.cif"],
        out_dir=tmp_path,
        ensemble_cfg={
            "enabled": True,
            "total_members": 1,
            "weights": {
                "scheme": "weighted_mean",
                "ref_weight": 0.7,
                "decoy_total_weight": 0.3,
            },
            "small_molecule": {"mode": "ligand_conformer"},
        },
        sampling_inputs_df=pd.DataFrame([{"pdb_id": "input"}]),
        cif_parse_cfg=None,
        cif_save_cfg=None,
    )
    manifest = pd.read_csv(staging.manifest_path)

    assert [Path(path).name for path in staging.member_groups[0]] == [
        "input.cif",
    ]
    assert [path.name for path in saved_paths] == [
        "input.cif",
    ]
    assert manifest["member_role"].tolist() == [
        "original",
    ]
    assert manifest["member_coefficient"].tolist() == pytest.approx([1.0])


def test_ligand_conformer_staging_mean_keeps_single_reference_with_decoys(
    monkeypatch,
    tmp_path,
):
    saved_paths, _ = _patch_fake_ligand_conformer_staging_deps(
        monkeypatch,
        num_decoys=2,
    )

    staging = stage_ligand_conformer_ensembles(
        pdb_paths=["/inputs/input.cif"],
        out_dir=tmp_path,
        ensemble_cfg={
            "enabled": True,
            "total_members": 3,
            "weights": {"scheme": "mean"},
            "small_molecule": {"mode": "ligand_conformer"},
        },
        sampling_inputs_df=pd.DataFrame([{"pdb_id": "input"}]),
        cif_parse_cfg=None,
        cif_save_cfg=None,
    )
    manifest = pd.read_csv(staging.manifest_path)

    assert [Path(path).name for path in staging.member_groups[0]] == [
        "input.cif",
        "input_ligconf_1.cif",
        "input_ligconf_2.cif",
    ]
    assert [path.name for path in saved_paths] == [
        "input.cif",
        "input_ligconf_1.cif",
        "input_ligconf_2.cif",
    ]
    assert manifest["member_role"].tolist() == [
        "original",
        "ligand_conformer_decoy",
        "ligand_conformer_decoy",
    ]
    assert manifest["member_coefficient"].tolist() == pytest.approx([1 / 3] * 3)


def test_compute_ligand_conformer_member_coefficients_uses_weighted_mean_split():
    coefficients = compute_ligand_conformer_member_coefficients(
        num_decoys=7,
        num_fallback_copies=0,
        scheme="weighted_mean",
        ref_weight=0.7,
        decoy_total_weight=0.3,
    )

    assert coefficients == pytest.approx([0.7, *([0.3 / 7] * 7)])
    assert sum(coefficients) == pytest.approx(1.0)


def test_compute_ligand_conformer_member_coefficients_splits_fallback_copies():
    coefficients = compute_ligand_conformer_member_coefficients(
        num_decoys=1,
        num_fallback_copies=2,
        scheme="weighted_mean",
        ref_weight=0.7,
        decoy_total_weight=0.3,
    )

    assert coefficients == pytest.approx([0.7, 0.1, 0.1, 0.1])
    assert sum(coefficients) == pytest.approx(1.0)


def test_compute_ligand_conformer_member_coefficients_mean_and_sqrt():
    assert compute_ligand_conformer_member_coefficients(
        num_decoys=2,
        num_fallback_copies=1,
        scheme="mean",
        ref_weight=0.7,
        decoy_total_weight=0.3,
    ) == pytest.approx([0.25] * 4)
    assert compute_ligand_conformer_member_coefficients(
        num_decoys=2,
        num_fallback_copies=1,
        scheme="sqrt",
        ref_weight=0.7,
        decoy_total_weight=0.3,
    ) == pytest.approx([0.5] * 4)


def test_compute_ligand_conformer_member_coefficients_original_only_fallback():
    assert compute_ligand_conformer_member_coefficients(
        num_decoys=0,
        num_fallback_copies=0,
        scheme="weighted_mean",
        ref_weight=0.7,
        decoy_total_weight=0.3,
    ) == [1.0]


def test_ligand_conformer_staging_uses_fallback_original_copies_for_metal_query(
    monkeypatch,
    tmp_path,
):
    atom_array = _make_atom_array_for_metal_only_query()
    saved_paths = []

    def fake_load_example_with_parse(path, cif_parse_cfg):
        return {"atom_array": atom_array.copy()}

    def fake_save_cif_file(atom_array, out_file, cif_save_cfg=None):
        out_file = Path(out_file)
        saved_paths.append(out_file)
        out_file.write_text(out_file.stem)

    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "load_example_with_parse",
        fake_load_example_with_parse,
    )
    monkeypatch.setattr(
        ligand_conformer_retrieval_module,
        "save_cif_file",
        fake_save_cif_file,
    )
    sampling_inputs_df = pd.DataFrame(
        [{"pdb_id": "metal", "query_pn_unit_iids": "['A_1', 'M_1']"}]
    )

    staging = stage_ligand_conformer_ensembles(
        pdb_paths=["/inputs/metal.cif"],
        out_dir=tmp_path,
        ensemble_cfg={
            "enabled": True,
            "total_members": 3,
            "weights": {"scheme": "mean"},
            "small_molecule": {"mode": "ligand_conformer"},
        },
        sampling_inputs_df=sampling_inputs_df,
        cif_parse_cfg=None,
        cif_save_cfg=None,
    )
    manifest = pd.read_csv(staging.manifest_path)

    assert [Path(path).name for path in staging.member_groups[0]] == [
        "metal.cif",
        "metal_ligconf_1.cif",
        "metal_ligconf_2.cif",
    ]
    assert [path.name for path in saved_paths] == [
        "metal.cif",
        "metal_ligconf_1.cif",
        "metal_ligconf_2.cif",
    ]
    assert manifest["member_role"].tolist() == [
        "original",
        "fallback_original_copy",
        "fallback_original_copy",
    ]
    assert manifest["member_coefficient"].tolist() == pytest.approx([1 / 3] * 3)
    assert manifest["target_ligand_pn_unit_iid"].isna().all()
    assert manifest["warning"].str.contains("fallback_original_copy: count=2").all()


def _write_dummy_cif(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"data_{path.stem}\n#\n")


def test_pharm_retrieval_staging_selects_query_and_rank_indices(tmp_path):
    cif_root = tmp_path / "cifs"
    query_dir = cif_root / "5UF"
    query_path = query_dir / "len150_5UF_0.cif"
    rank0_path = query_dir / "len150_5UF_rank0_ABC_0100.cif"
    rank2_path = query_dir / "len150_5UF_rank2_DEF_0090.cif"
    _write_dummy_cif(query_path)
    _write_dummy_cif(rank0_path)
    _write_dummy_cif(rank2_path)
    selected_queries_tsv = tmp_path / "selected_queries.tsv"
    selected_queries_tsv.write_text(
        "query_ccd\tquery_pn_unit_iid\n"
        "5UF\tL_1\n"
    )

    staging = stage_pharm_retrieval_ensembles(
        pdb_paths=[str(query_path)],
        out_dir=tmp_path / "out",
        ensemble_cfg={
            "enabled": True,
            "total_members": 3,
            "weights": {"scheme": "mean"},
            "small_molecule": {
                "mode": "pharm_retrieval",
                "pharm_retrieval": {
                    "cif_root": str(cif_root),
                    "selected_queries_tsv": str(selected_queries_tsv),
                    "rank_indices": [0, 2],
                },
            },
        },
        sampling_inputs_df=None,
    )

    assert [Path(path).name for path in staging.pdb_paths] == [
        query_path.name,
        rank0_path.name,
        rank2_path.name,
    ]
    assert staging.target_count() == 1
    assert list(staging.iter_member_batches(max_members=8)) == [staging.pdb_paths]

    manifest = pd.read_csv(staging.manifest_path, keep_default_na=False)
    assert manifest["member_role"].tolist() == [
        "query_original",
        "pharm_rank",
        "pharm_rank",
    ]
    assert manifest["rank_index"].tolist() == ["", "0", "2"]
    assert manifest["requested_rank_indices"].unique().tolist() == ["[0, 2]"]

    runtime_rows = staging.sampling_inputs_df.set_index("pdb_key")
    assert set(runtime_rows.index) == {
        "len150_5UF_0",
        "len150_5UF_rank0_ABC_0100",
        "len150_5UF_rank2_DEF_0090",
    }
    assert runtime_rows.loc["len150_5UF_rank0_ABC_0100", "pdb_id"] == "5UF"
    assert runtime_rows.loc["len150_5UF_rank0_ABC_0100", "query_pn_unit_iids"] == "['L_1']"
    assert runtime_rows.loc["len150_5UF_rank0_ABC_0100", "ccd_code"] == "5UF"

    batch = staging.annotate_batch(
        {},
        batch_pdb_paths=staging.pdb_paths,
        device="cpu",
    )
    torch.testing.assert_close(batch["tied_sampling_ids"], torch.tensor([0, 0, 0]))
    assert batch["tied_sampling_aggregation_scheme"] == "mean"


def test_pharm_retrieval_staging_rejects_missing_rank(tmp_path):
    cif_root = tmp_path / "cifs"
    query_dir = cif_root / "5UF"
    query_path = query_dir / "len150_5UF_0.cif"
    _write_dummy_cif(query_path)

    with pytest.raises(FileNotFoundError, match="rank_indices=\\[7\\]"):
        stage_pharm_retrieval_ensembles(
            pdb_paths=[str(query_path)],
            out_dir=tmp_path / "out",
            ensemble_cfg={
                "enabled": True,
                "total_members": 2,
                "small_molecule": {
                    "mode": "pharm_retrieval",
                    "pharm_retrieval": {
                        "cif_root": str(cif_root),
                        "rank_indices": [7],
                        "query_pn_unit_iids": ["L_1"],
                    },
                },
            },
            sampling_inputs_df=None,
        )


def test_pharm_retrieval_staging_adds_exact_keys_to_sampling_df_without_pdb_key(tmp_path):
    cif_root = tmp_path / "cifs"
    query_dir = cif_root / "5UF"
    query_path = query_dir / "len150_5UF_0.cif"
    rank0_path = query_dir / "len150_5UF_rank0_ABC_0100.cif"
    _write_dummy_cif(query_path)
    _write_dummy_cif(rank0_path)

    staging = stage_pharm_retrieval_ensembles(
        pdb_paths=[str(query_path)],
        out_dir=tmp_path / "out",
        ensemble_cfg={
            "enabled": True,
            "total_members": 2,
            "small_molecule": {
                "mode": "pharm_retrieval",
                "pharm_retrieval": {
                    "cif_root": str(cif_root),
                    "rank_indices": [0],
                },
            },
        },
        sampling_inputs_df=pd.DataFrame(
            [{"pdb_id": "5UF", "query_pn_unit_iids": "['L_2']"}]
        ),
    )

    runtime_rows = staging.sampling_inputs_df.set_index("pdb_key")
    assert "len150_5UF_0" in runtime_rows.index
    assert "len150_5UF_rank0_ABC_0100" in runtime_rows.index
    assert runtime_rows.loc["len150_5UF_rank0_ABC_0100", "query_pn_unit_iids"] == "['L_2']"


def test_pharm_retrieval_staging_rejects_duplicate_rank_indices(tmp_path):
    cif_root = tmp_path / "cifs"
    query_dir = cif_root / "5UF"
    query_path = query_dir / "len150_5UF_0.cif"
    rank0_path = query_dir / "len150_5UF_rank0_ABC_0100.cif"
    _write_dummy_cif(query_path)
    _write_dummy_cif(rank0_path)

    with pytest.raises(ValueError, match="duplicate ranks"):
        stage_pharm_retrieval_ensembles(
            pdb_paths=[str(query_path)],
            out_dir=tmp_path / "out",
            ensemble_cfg={
                "enabled": True,
                "total_members": 3,
                "small_molecule": {
                    "mode": "pharm_retrieval",
                    "pharm_retrieval": {
                        "cif_root": str(cif_root),
                        "rank_indices": [0, 0],
                        "query_pn_unit_iids": ["L_1"],
                    },
                },
            },
            sampling_inputs_df=None,
        )


def test_ligand_conformer_staging_result_owns_runtime_sampling_contract():
    staging_result = LigandConformerStagingResult(
        root_dir=Path("/tmp/staged"),
        pdb_paths=[
            "/tmp/staged/input.cif",
            "/tmp/staged/input_ligconf_1.cif",
            "/tmp/staged/other.cif",
            "/tmp/staged/other_ligconf_1.cif",
            "/tmp/staged/single.cif",
        ],
        member_groups=[
            ["/tmp/staged/input.cif", "/tmp/staged/input_ligconf_1.cif"],
            ["/tmp/staged/other.cif", "/tmp/staged/other_ligconf_1.cif"],
            ["/tmp/staged/single.cif"],
        ],
        sampling_inputs_df=None,
        member_to_group_id={
            "input": 0,
            "input_ligconf_1": 0,
            "other": 1,
            "other_ligconf_1": 1,
            "single": 2,
        },
        member_to_coefficient={
            "input": 0.5,
            "input_ligconf_1": 0.5,
            "other": 0.5,
            "other_ligconf_1": 0.5,
            "single": 1.0,
        },
        member_to_target_id={
            "input": "input",
            "input_ligconf_1": "input",
            "other": "other",
            "other_ligconf_1": "other",
            "single": "single",
        },
        aggregation_scheme="weighted_mean",
        manifest_path=Path("/tmp/staged/ligand_conformer_manifest.csv"),
    )
    pos_constraint_df = pd.DataFrame(
        [{"pdb_key": "input", "fixed_pos_seq": "A:1", "fixed_pos_scn": ""}]
    )

    assert staging_result.target_count() == 3
    assert staging_result.target_count(
        ["/tmp/staged/input.cif", "/tmp/staged/input_ligconf_1.cif"]
    ) == 1
    assert list(staging_result.iter_member_batches(max_members=3)) == [
        ["/tmp/staged/input.cif", "/tmp/staged/input_ligconf_1.cif"],
        [
            "/tmp/staged/other.cif",
            "/tmp/staged/other_ligconf_1.cif",
            "/tmp/staged/single.cif",
        ],
    ]

    expanded = staging_result.expand_pos_constraints(pos_constraint_df)

    assert set(expanded["pdb_key"]) == {"input", "input_ligconf_1"}
    decoy_row = expanded.set_index("pdb_key").loc["input_ligconf_1"]
    assert decoy_row["fixed_pos_seq"] == "A:1"

    batch = staging_result.annotate_batch(
        {},
        batch_pdb_paths=["/tmp/staged/input.cif", "/tmp/staged/input_ligconf_1.cif"],
        device="cpu",
    )

    torch.testing.assert_close(batch["tied_sampling_ids"], torch.tensor([0, 0]))
    assert batch["tied_sampling_aggregation_scheme"] == "weighted_mean"
    torch.testing.assert_close(
        batch["tied_sampling_weights"],
        torch.tensor([0.5, 0.5]),
    )


def test_ligand_conformer_mean_annotation_omits_tied_weights():
    staging_result = LigandConformerStagingResult(
        root_dir=Path("/tmp/staged"),
        pdb_paths=["/tmp/staged/input.cif", "/tmp/staged/input_ligconf_1.cif"],
        member_groups=[["/tmp/staged/input.cif", "/tmp/staged/input_ligconf_1.cif"]],
        sampling_inputs_df=None,
        member_to_group_id={"input": 0, "input_ligconf_1": 0},
        member_to_coefficient={"input": 0.5, "input_ligconf_1": 0.5},
        member_to_target_id={"input": "input", "input_ligconf_1": "input"},
        aggregation_scheme="mean",
        manifest_path=Path("/tmp/staged/ligand_conformer_manifest.csv"),
    )

    batch = staging_result.annotate_batch(
        {},
        batch_pdb_paths=["/tmp/staged/input.cif", "/tmp/staged/input_ligconf_1.cif"],
        device="cpu",
    )

    assert batch["tied_sampling_aggregation_scheme"] == "mean"
    assert "tied_sampling_weights" not in batch
