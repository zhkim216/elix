import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, BondList
from rdkit import Chem

from allatom_design.data.transform import custom_transforms as custom_transforms_module
from allatom_design.data.transform import sd_featurizer as sd_featurizer_module
from allatom_design.data.transform.bonds import (
    AddLigandBondOrderFeatures,
    get_af3_token_bond_features,
    get_ligand_bond_order_features,
)
from allatom_design.data.transform.custom_transforms import (
    AddCachedRDKitFeatures,
    AddCachedRDKitChiralityFeatures,
    AnnotateNonPolymerCovalentBondEndpoints,
    FeaturizeCoordsAndMasks,
    PadSDFeats,
)


def _atom_array(
    atom_names: list[str],
    *,
    res_name: str = "LIG",
    eligible: bool = True,
) -> AtomArray:
    atom_array = AtomArray(len(atom_names))
    atom_array.atom_name = np.asarray(atom_names)
    atom_array.res_name = np.full(len(atom_names), res_name)
    atom_array.res_id = np.ones(len(atom_names), dtype=int)
    atom_array.chain_id = np.full(len(atom_names), "A")
    atom_array.coord = np.arange(len(atom_names) * 3, dtype=float).reshape(-1, 3)
    atom_array.occupancy = np.ones(len(atom_names), dtype=float)
    atom_array.element = np.full(len(atom_names), "C")
    atom_array.set_annotation("is_polymer", np.zeros(len(atom_names), dtype=bool))
    atom_array.set_annotation(
        "is_covalent_modification", np.zeros(len(atom_names), dtype=bool)
    )
    atom_array.set_annotation(
        "is_nonpolymer_covalent_attachment", np.zeros(len(atom_names), dtype=bool)
    )
    atom_array.set_annotation("atom_is_small_molecule_chain", np.full(len(atom_names), eligible))
    atom_array.set_annotation("atom_is_metal_chain", np.zeros(len(atom_names), dtype=bool))
    atom_array.set_annotation("atom_is_nucleic_acid_chain", np.zeros(len(atom_names), dtype=bool))
    return atom_array


def _rdkit_mol(*, with_conformer: bool = True) -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCO")
    if with_conformer:
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for atom_idx in range(mol.GetNumAtoms()):
            conformer.SetAtomPosition(atom_idx, (float(atom_idx), 0.0, 0.0))
        mol.AddConformer(conformer)
    return mol


def _write_cache(tmp_path, entry, res_name: str = "LIG"):
    cache_path = tmp_path / res_name / f"{res_name}.pt"
    cache_path.parent.mkdir(parents=True)
    torch.save(entry, cache_path)


def _hydrogenbond_features(tmp_path, smiles: str, atom_names: list[str]) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    assert mol.GetNumAtoms() == len(atom_names)
    _write_cache(tmp_path, {"mol": mol, "atom_names": np.asarray(atom_names)})
    atom_array = _atom_array(atom_names)
    return AddCachedRDKitFeatures(
        tmp_path, add_hydrogenbond_feature=True
    ).forward({"atom_array": atom_array, "feats": {}})["feats"]


def test_af3_token_bonds_use_true_polymer_status_and_apply_cutoff():
    atom_array = AtomArray(3)
    atom_array.res_name = np.asarray(["POL", "POL", "LIG"])
    atom_array.res_id = np.asarray([1, 2, 3])
    atom_array.chain_id = np.asarray(["A", "A", "B"])
    atom_array.atom_name = np.asarray(["A1", "A2", "A3"])
    atom_array.coord = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    atom_array.set_annotation("atomize", np.asarray([True, False, False]))
    atom_array.set_annotation("is_polymer", np.asarray([True, True, False]))
    atom_array.bonds = BondList(
        3,
        np.asarray(
            [
                [0, 1, 1],  # atomized polymer--polymer: must be removed
                [1, 2, 1],  # non-atomized polymer--non-polymer: must be kept
                [0, 2, 1],  # eligible chemistry, but beyond the cutoff
            ]
        ),
    )

    token_bonds = get_af3_token_bond_features(atom_array, distance_cutoff=2.4)

    expected = np.zeros((3, 3), dtype=bool)
    expected[1, 2] = expected[2, 1] = True
    np.testing.assert_array_equal(token_bonds, expected)
    assert not np.diag(token_bonds).any()


def test_ligand_bond_order_maps_all_biotite_types_and_is_symmetric():
    biotite_types = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    atom_array = AtomArray(len(biotite_types) + 1)
    atom_array.bonds = BondList(
        len(atom_array),
        np.asarray([[0, atom_idx + 1, bond_type] for atom_idx, bond_type in enumerate(biotite_types)]),
    )

    bond_order = get_ligand_bond_order_features(atom_array)

    expected_types = [5, 1, 2, 3, 5, 4, 4, 4, 5, 4]
    np.testing.assert_array_equal(bond_order[0, 1:], expected_types)
    np.testing.assert_array_equal(bond_order, bond_order.T)
    assert bond_order.dtype == np.int8
    assert bond_order[1, 2] == 0
    assert np.all(np.diag(bond_order) == 0)


def test_ligand_bond_order_transform_writes_compact_pair_feature():
    atom_array = AtomArray(2)
    atom_array.bonds = BondList(2, np.asarray([[0, 1, 2]]))

    result = AddLigandBondOrderFeatures().forward({"atom_array": atom_array, "feats": {}})

    assert result["feats"]["atom_ligand_bond_order"].dtype == np.int8
    np.testing.assert_array_equal(result["feats"]["atom_ligand_bond_order"], [[0, 2], [2, 0]])


@pytest.mark.parametrize(
    ("smiles", "atom_names", "expected_hba", "expected_hbd"),
    [
        ("CCO", ["C1", "C2", "O1"], [0, 0, 1], [0, 0, 1]),
        ("CC=O", ["C1", "C2", "O1"], [0, 0, 1], [0, 0, 0]),
        ("CC(=O)N", ["C1", "C2", "O1", "N1"], [0, 0, 1, 0], [0, 0, 0, 1]),
        ("C[N+](C)(C)C", ["C1", "N1", "C2", "C3", "C4"], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]),
    ],
)
def test_shepherd_hydrogenbond_chemical_fixtures(
    tmp_path,
    smiles,
    atom_names,
    expected_hba,
    expected_hbd,
):
    feats = _hydrogenbond_features(tmp_path, smiles, atom_names)

    np.testing.assert_array_equal(feats["atom_is_HBA"], expected_hba)
    np.testing.assert_array_equal(feats["atom_is_HBD"], expected_hbd)
    np.testing.assert_array_equal(
        feats["atom_hydrogenbond_feature_mask"], np.ones(len(atom_names), dtype=bool)
    )


def test_vendored_shepherd_definition_marks_cached_atp_n6_donor_only():
    cache_root = Path("/home/yjhk/model-dev/datasets/atomworks_cached_residue_data")
    cache_path = cache_root / "ATP" / "ATP.pt"
    if not cache_path.exists():
        pytest.skip(f"Local AtomWorks ATP cache is unavailable: {cache_path}")

    entry = torch.load(cache_path, map_location="cpu", weights_only=False)
    atom_names = [str(name).strip() for name in entry["atom_names"]]
    atom_array = _atom_array(atom_names, res_name="ATP")
    feats = AddCachedRDKitFeatures(
        cache_root, add_hydrogenbond_feature=True
    ).forward({"atom_array": atom_array, "feats": {}})["feats"]

    n6_idx = atom_names.index("N6")
    assert not feats["atom_is_HBA"][n6_idx]
    assert feats["atom_is_HBD"][n6_idx]
    assert feats["atom_hydrogenbond_feature_mask"][n6_idx]


def test_hydrogenbond_maps_reordered_subset_including_unresolved_atom(tmp_path):
    mol = Chem.MolFromSmiles("CC(=O)N")
    _write_cache(
        tmp_path,
        {"mol": mol, "atom_names": np.asarray(["C1", "C2", "O1", "N1"])},
    )
    atom_array = _atom_array(["N1", "O1", "C1"])
    atom_array.occupancy[0] = 0.0

    feats = AddCachedRDKitFeatures(
        tmp_path, add_hydrogenbond_feature=True
    ).forward({"atom_array": atom_array, "feats": {}})["feats"]

    np.testing.assert_array_equal(feats["atom_is_HBA"], [0, 1, 0])
    np.testing.assert_array_equal(feats["atom_is_HBD"], [1, 0, 0])
    np.testing.assert_array_equal(feats["atom_hydrogenbond_feature_mask"], [1, 1, 1])


def test_hydrogenbond_only_masks_unmapped_runtime_atom_and_warns_once(
    tmp_path,
    caplog,
):
    mol = Chem.MolFromSmiles("CC(=O)N")
    _write_cache(
        tmp_path,
        {"mol": mol, "atom_names": np.asarray(["C1", "C2", "O1", "N1"])},
    )
    atom_array = _atom_array(["N1", "X1", "O1"])
    transform = AddCachedRDKitFeatures(tmp_path, add_hydrogenbond_feature=True)

    with caplog.at_level(logging.WARNING, logger=custom_transforms_module.__name__):
        first = transform.forward({"atom_array": atom_array, "feats": {}})["feats"]
        second = transform.forward({"atom_array": atom_array, "feats": {}})["feats"]

    np.testing.assert_array_equal(first["atom_is_HBA"], [0, 0, 1])
    np.testing.assert_array_equal(first["atom_is_HBD"], [1, 0, 0])
    np.testing.assert_array_equal(first["atom_hydrogenbond_feature_mask"], [1, 0, 1])
    np.testing.assert_array_equal(second["atom_hydrogenbond_feature_mask"], [1, 0, 1])
    warnings = [
        record for record in caplog.records
        if "incomplete_atom_name_coverage" in record.message
    ]
    assert len(warnings) == 1
    assert "CCD LIG" in warnings[0].message


@pytest.mark.parametrize(
    ("cache_case", "expected_reason"),
    [
        ("missing_file", "missing_cache_file"),
        ("count_mismatch", "cached_atom_names_mol_count_mismatch"),
    ],
)
def test_hydrogenbond_cache_failure_masks_ccd_and_deduplicates_warning(
    tmp_path,
    caplog,
    cache_case,
    expected_reason,
):
    if cache_case == "count_mismatch":
        _write_cache(
            tmp_path,
            {"mol": Chem.MolFromSmiles("CCO"), "atom_names": np.asarray(["C1", "O1"])},
        )
    atom_array = _atom_array(["C1", "O1"])
    transform = AddCachedRDKitFeatures(tmp_path, add_hydrogenbond_feature=True)

    with caplog.at_level(logging.WARNING, logger=custom_transforms_module.__name__):
        first = transform.forward({"atom_array": atom_array, "feats": {}})["feats"]
        second = transform.forward({"atom_array": atom_array, "feats": {}})["feats"]

    np.testing.assert_array_equal(first["atom_is_HBA"], [0, 0])
    np.testing.assert_array_equal(first["atom_is_HBD"], [0, 0])
    np.testing.assert_array_equal(first["atom_hydrogenbond_feature_mask"], [0, 0])
    np.testing.assert_array_equal(second["atom_hydrogenbond_feature_mask"], [0, 0])
    warnings = [record for record in caplog.records if expected_reason in record.message]
    assert len(warnings) == 1


def test_hydrogenbond_factory_failure_masks_ccd_and_warns(tmp_path, monkeypatch, caplog):
    _write_cache(
        tmp_path,
        {"mol": Chem.MolFromSmiles("CO"), "atom_names": np.asarray(["C1", "O1"])},
    )

    def fail_factory(_path):
        raise RuntimeError("broken fdef")

    monkeypatch.setattr(
        custom_transforms_module.ChemicalFeatures,
        "BuildFeatureFactory",
        fail_factory,
    )
    transform = AddCachedRDKitFeatures(tmp_path, add_hydrogenbond_feature=True)
    with caplog.at_level(logging.WARNING, logger=custom_transforms_module.__name__):
        feats = transform.forward(
            {"atom_array": _atom_array(["C1", "O1"]), "feats": {}}
        )["feats"]

    np.testing.assert_array_equal(feats["atom_hydrogenbond_feature_mask"], [0, 0])
    assert any("feature_factory_unavailable" in record.message for record in caplog.records)


def test_hydrogenbond_excludes_polymer_without_cache_access(tmp_path):
    atom_array = _atom_array(["N", "CA", "C"], res_name="ALA")
    atom_array.is_polymer[:] = True
    transform = AddCachedRDKitFeatures(tmp_path, add_hydrogenbond_feature=True)
    transform._load_residue_entry = lambda _res_name: pytest.fail("polymer accessed cache")

    feats = transform.forward({"atom_array": atom_array, "feats": {}})["feats"]

    np.testing.assert_array_equal(feats["atom_is_HBA"], [0, 0, 0])
    np.testing.assert_array_equal(feats["atom_is_HBD"], [0, 0, 0])
    np.testing.assert_array_equal(feats["atom_hydrogenbond_feature_mask"], [0, 0, 0])


def test_covalent_endpoint_annotation_survives_slice_and_only_masks_endpoint(tmp_path):
    atom_array = _atom_array(["P", "C1", "O1"])
    atom_array.res_name = np.asarray(["ALA", "LIG", "LIG"])
    atom_array.res_id = np.asarray([1, 2, 2])
    atom_array.is_polymer[:] = [True, False, False]
    atom_array.is_covalent_modification[:] = True
    atom_array.bonds = BondList(3, np.asarray([[0, 2, 1]]))
    AnnotateNonPolymerCovalentBondEndpoints().forward({"atom_array": atom_array})
    np.testing.assert_array_equal(
        atom_array.is_nonpolymer_covalent_attachment, [0, 0, 1]
    )

    cropped_atom_array = atom_array[[1, 2]]
    np.testing.assert_array_equal(
        cropped_atom_array.is_nonpolymer_covalent_attachment, [0, 1]
    )
    _write_cache(
        tmp_path,
        {"mol": Chem.MolFromSmiles("CO"), "atom_names": np.asarray(["C1", "O1"])},
    )
    feats = AddCachedRDKitFeatures(
        tmp_path, add_hydrogenbond_feature=True
    ).forward({"atom_array": cropped_atom_array, "feats": {}})["feats"]

    np.testing.assert_array_equal(feats["atom_is_HBA"], [0, 0])
    np.testing.assert_array_equal(feats["atom_is_HBD"], [0, 0])
    np.testing.assert_array_equal(feats["atom_hydrogenbond_feature_mask"], [1, 0])


def test_combined_cached_rdkit_features_load_residue_once(tmp_path, monkeypatch):
    cached_mol = _rdkit_mol()
    entry = {"mol": cached_mol, "atom_names": np.asarray(["C1", "C2", "O1"])}
    _write_cache(tmp_path, entry)
    original_load = custom_transforms_module.torch.load
    load_count = 0

    def counting_load(*args, **kwargs):
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(custom_transforms_module.torch, "load", counting_load)
    transform = AddCachedRDKitFeatures(
        tmp_path,
        add_chirality=True,
        add_hydrogenbond_feature=True,
    )
    transform.forward({"atom_array": _atom_array(["C1", "C2", "O1"]), "feats": {}})

    assert load_count == 1


def test_cached_chirality_maps_reordered_cropped_atoms_and_clones_mol(tmp_path, monkeypatch):
    cached_mol = _rdkit_mol()
    _write_cache(
        tmp_path,
        {"mol": cached_mol, "atom_names": np.asarray(["C1", "C2", "O1"])},
    )
    atom_array = _atom_array(["O1", "C1"])

    def fake_get_chiral_centers(mol):
        assert mol is not cached_mol
        return [
            {"chiral_center_idx": 0, "chirality": "R"},
            {"chiral_center_idx": 2, "chirality": "S"},
        ]

    monkeypatch.setattr(custom_transforms_module, "get_chiral_centers", fake_get_chiral_centers)
    transform = AddCachedRDKitChiralityFeatures(tmp_path)

    result = transform.forward({"atom_array": atom_array, "feats": {}})

    encoded = result["feats"]["atom_cached_rdkit_chirality_tag"]
    np.testing.assert_array_equal(encoded, [2, 1])
    np.testing.assert_array_equal(result["feats"]["atom_cached_rdkit_chirality_mask"], [1, 1])
    assert encoded.dtype == np.int64


@pytest.mark.parametrize(
    ("cache_case", "expected_reason"),
    [
        ("missing_file", "missing_cache_file"),
        ("empty_entry", "empty_cache_entry"),
        ("missing_mol", "missing_cached_mol"),
        ("no_conformer", "missing_cached_conformer"),
        ("incomplete_coverage", "incomplete_atom_name_coverage"),
    ],
)
def test_cached_chirality_failure_omits_conditioning_and_deduplicates_warning(
    tmp_path,
    caplog,
    cache_case,
    expected_reason,
):
    runtime_names = ["C1", "C2", "O1"]
    if cache_case == "empty_entry":
        _write_cache(tmp_path, None)
    elif cache_case == "missing_mol":
        _write_cache(tmp_path, {"mol": None, "atom_names": np.asarray(runtime_names)})
    elif cache_case == "no_conformer":
        _write_cache(
            tmp_path,
            {"mol": _rdkit_mol(with_conformer=False), "atom_names": np.asarray(runtime_names)},
        )
    elif cache_case == "incomplete_coverage":
        _write_cache(
            tmp_path,
            {"mol": _rdkit_mol(), "atom_names": np.asarray(["C1", "C2", "X1"])},
        )

    atom_array = _atom_array(runtime_names)
    atom_array.set_annotation("stereo", np.full(len(atom_array), "R"))
    transform = AddCachedRDKitChiralityFeatures(tmp_path)

    with caplog.at_level(logging.WARNING, logger=custom_transforms_module.__name__):
        first = transform.forward({"atom_array": atom_array, "feats": {}})
        second = transform.forward({"atom_array": atom_array, "feats": {}})

    np.testing.assert_array_equal(first["feats"]["atom_cached_rdkit_chirality_tag"], [0, 0, 0])
    np.testing.assert_array_equal(second["feats"]["atom_cached_rdkit_chirality_tag"], [0, 0, 0])
    np.testing.assert_array_equal(first["feats"]["atom_cached_rdkit_chirality_mask"], [0, 0, 0])
    warnings = [record for record in caplog.records if expected_reason in record.message]
    assert len(warnings) == 1
    assert "CCD LIG" in warnings[0].message
    assert "CACHE_UNKNOWN" not in warnings[0].message


@pytest.mark.parametrize("res_name", ["ALA", "A", "DA"])
def test_cached_chirality_standard_residue_is_n_without_cache_access_or_warning(
    tmp_path,
    caplog,
    res_name,
):
    atom_array = _atom_array(["X1", "X2", "X3"], res_name=res_name)
    transform = AddCachedRDKitChiralityFeatures(tmp_path)
    transform._load_residue_entry = lambda _res_name: pytest.fail("standard residue accessed cache")

    with caplog.at_level(logging.WARNING, logger=custom_transforms_module.__name__):
        result = transform.forward({"atom_array": atom_array, "feats": {}})

    np.testing.assert_array_equal(result["feats"]["atom_cached_rdkit_chirality_tag"], [0, 0, 0])
    np.testing.assert_array_equal(result["feats"]["atom_cached_rdkit_chirality_mask"], [0, 0, 0])
    assert not caplog.records


def test_cached_chirality_noneligible_residue_is_n_without_cache_access(tmp_path):
    atom_array = _atom_array(["C1", "C2"], eligible=False)
    transform = AddCachedRDKitChiralityFeatures(tmp_path)
    transform._load_residue_entry = lambda _res_name: pytest.fail("noneligible residue accessed cache")

    result = transform.forward({"atom_array": atom_array, "feats": {}})

    np.testing.assert_array_equal(result["feats"]["atom_cached_rdkit_chirality_tag"], [0, 0])
    np.testing.assert_array_equal(result["feats"]["atom_cached_rdkit_chirality_mask"], [0, 0])


@pytest.mark.parametrize(
    ("cached_names", "runtime_names", "centers", "expected_reason"),
    [
        (["C1", "C2"], ["C1", "C2"], [], "cached_atom_names_mol_count_mismatch"),
        (["C1", "C1", "O1"], ["C1", "C2", "O1"], [], "duplicate_cached_atom_names"),
        (["C1", "C2", "O1"], ["C1", "C1", "O1"], [], "duplicate_runtime_atom_names"),
        (
            ["C1", "C2", "O1"],
            ["C1", "C2", "O1"],
            [{"chiral_center_idx": 3, "chirality": "R"}],
            "invalid_cached_chiral_center_index",
        ),
        (
            ["C1", "C2", "O1"],
            ["C1", "C2", "O1"],
            [{"chiral_center_idx": 1, "chirality": "?"}],
            "invalid_cached_chirality_label",
        ),
    ],
)
def test_cached_chirality_broken_invariants_omit_conditioning(
    tmp_path,
    monkeypatch,
    cached_names,
    runtime_names,
    centers,
    expected_reason,
    caplog,
):
    _write_cache(tmp_path, {"mol": _rdkit_mol(), "atom_names": np.asarray(cached_names)})
    atom_array = _atom_array(runtime_names)
    monkeypatch.setattr(custom_transforms_module, "get_chiral_centers", lambda _mol: centers)
    transform = AddCachedRDKitChiralityFeatures(tmp_path)

    with caplog.at_level(logging.WARNING, logger=custom_transforms_module.__name__):
        result = transform.forward({"atom_array": atom_array, "feats": {}})

    np.testing.assert_array_equal(
        result["feats"]["atom_cached_rdkit_chirality_tag"],
        np.zeros(len(runtime_names), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        result["feats"]["atom_cached_rdkit_chirality_mask"],
        np.zeros(len(runtime_names), dtype=bool),
    )
    warnings = [record for record in caplog.records if expected_reason in record.message]
    assert len(warnings) == 1


def _minimal_featurize_atom_array() -> AtomArray:
    atom_array = _atom_array(["C1"])
    atom_array.set_annotation("atomize", np.asarray([True]))
    atom_array.set_annotation("chain_type", np.asarray([8]))
    atom_array.set_annotation("is_polymer", np.asarray([False]))
    atom_array.set_annotation("hetero", np.asarray([True]))
    atom_array.set_annotation("atomic_number", np.asarray([6]))
    atom_array.set_annotation("charge", np.asarray([0.0]))
    atom_array.set_annotation("is_aromatic", np.asarray([False]))
    atom_array.set_annotation("stereo", np.asarray(["N"]))
    atom_array.set_annotation("is_covalent_modification", np.asarray([False]))
    atom_array.set_annotation("is_ligand_pocket", np.asarray([False]))
    atom_array.set_annotation("atom_is_protein_chain", np.asarray([False]))
    atom_array.set_annotation("atom_is_peptide_chain", np.asarray([False]))
    return atom_array


def _minimal_feats(*, with_optional_features: bool) -> dict:
    feats = {
        "atom_to_token_map": torch.tensor([0]),
        "is_atomized": torch.tensor([True]),
        "token_bonds": np.zeros((1, 1), dtype=bool),
    }
    if with_optional_features:
        feats["atom_cached_rdkit_chirality_tag"] = np.asarray([2], dtype=np.int64)
        feats["atom_cached_rdkit_chirality_mask"] = np.asarray([True])
        feats["atom_is_HBA"] = np.asarray([True])
        feats["atom_is_HBD"] = np.asarray([False])
        feats["atom_hydrogenbond_feature_mask"] = np.asarray([True])
        feats["atom_ligand_bond_order"] = np.zeros((1, 1), dtype=np.int8)
    return feats


def test_featurize_converts_optional_features_and_does_not_create_absent_keys():
    transform = FeaturizeCoordsAndMasks()
    with_optional = transform.forward(
        {
            "atom_array": _minimal_featurize_atom_array(),
            "feats": _minimal_feats(with_optional_features=True),
            "is_inference": True,
        }
    )["feats"]
    without_optional = transform.forward(
        {
            "atom_array": _minimal_featurize_atom_array(),
            "feats": _minimal_feats(with_optional_features=False),
            "is_inference": True,
        }
    )["feats"]

    assert with_optional["atom_cached_rdkit_chirality_tag"].dtype == torch.long
    assert with_optional["atom_cached_rdkit_chirality_mask"].dtype == torch.float32
    assert with_optional["atom_is_HBA"].dtype == torch.float32
    assert with_optional["atom_is_HBD"].dtype == torch.float32
    assert with_optional["atom_hydrogenbond_feature_mask"].dtype == torch.float32
    assert with_optional["atom_ligand_bond_order"].dtype == torch.int8
    assert "atom_cached_rdkit_chirality_tag" not in without_optional
    assert "atom_cached_rdkit_chirality_mask" not in without_optional
    assert "atom_is_HBA" not in without_optional
    assert "atom_is_HBD" not in without_optional
    assert "atom_hydrogenbond_feature_mask" not in without_optional
    assert "atom_ligand_bond_order" not in without_optional


def test_pair_features_are_padded_on_both_axes():
    feats = {
        "token_index": torch.arange(2),
        "atom_resolved_mask": torch.ones(3),
        "token_bonds": torch.ones((2, 2)),
        "atom_cached_rdkit_chirality_tag": torch.tensor([1, 2, 0]),
        "atom_cached_rdkit_chirality_mask": torch.tensor([1.0, 1.0, 0.0]),
        "atom_is_HBA": torch.tensor([1.0, 0.0, 0.0]),
        "atom_is_HBD": torch.tensor([0.0, 1.0, 0.0]),
        "atom_hydrogenbond_feature_mask": torch.tensor([1.0, 1.0, 0.0]),
        "atom_ligand_bond_order": torch.ones((3, 3), dtype=torch.int8),
    }

    result = PadSDFeats(max_tokens=4, max_atoms=5).forward({"feats": feats})["feats"]

    assert result["token_bonds"].shape == (4, 4)
    assert result["atom_ligand_bond_order"].shape == (5, 5)
    assert result["atom_cached_rdkit_chirality_tag"].shape == (5,)
    assert result["atom_cached_rdkit_chirality_mask"].shape == (5,)
    assert result["atom_is_HBA"].shape == (5,)
    assert result["atom_is_HBD"].shape == (5,)
    assert result["atom_hydrogenbond_feature_mask"].shape == (5,)
    assert torch.all(result["token_bonds"][2:, :] == 0)
    assert torch.all(result["token_bonds"][:, 2:] == 0)
    assert torch.all(result["atom_ligand_bond_order"][3:, :] == 0)
    assert torch.all(result["atom_ligand_bond_order"][:, 3:] == 0)


@pytest.mark.parametrize(
    "builder",
    [sd_featurizer_module.sd_featurizer, sd_featurizer_module.sd_featurizer_for_design],
)
def test_feature_producer_gates_control_pipeline_and_require_cache_root(builder, tmp_path):
    off_pipeline = builder(
        max_tokens=8,
        max_atoms=16,
        use_ligand_cached_rdkit_chirality=False,
        add_hydrogenbond_feature=False,
        use_ligand_bond_order=False,
    )
    assert not any(isinstance(t, AddCachedRDKitFeatures) for t in off_pipeline.transforms)
    assert not any(
        isinstance(t, AnnotateNonPolymerCovalentBondEndpoints)
        for t in off_pipeline.transforms
    )
    assert not any(isinstance(t, AddLigandBondOrderFeatures) for t in off_pipeline.transforms)

    on_pipeline = builder(
        max_tokens=8,
        max_atoms=16,
        residue_cache_dir=str(tmp_path),
        use_ligand_cached_rdkit_chirality=True,
        add_hydrogenbond_feature=True,
        use_ligand_bond_order=True,
    )
    cached_transform = next(
        t for t in on_pipeline.transforms if isinstance(t, AddCachedRDKitFeatures)
    )
    assert cached_transform.add_chirality
    assert cached_transform.add_hydrogenbond_feature
    assert any(
        isinstance(t, AnnotateNonPolymerCovalentBondEndpoints)
        for t in on_pipeline.transforms
    )
    assert any(isinstance(t, AddLigandBondOrderFeatures) for t in on_pipeline.transforms)

    with pytest.raises(ValueError, match="residue_cache_dir must be set"):
        builder(
            residue_cache_dir=None,
            use_ligand_cached_rdkit_chirality=True,
        )

    with pytest.raises(ValueError, match="residue_cache_dir must be set"):
        builder(
            residue_cache_dir=None,
            add_hydrogenbond_feature=True,
        )


@pytest.mark.parametrize(
    "builder",
    [sd_featurizer_module.sd_featurizer, sd_featurizer_module.sd_featurizer_for_design],
)
def test_hydrogenbond_false_gate_does_not_load_feature_definition(builder, monkeypatch):
    monkeypatch.setattr(
        custom_transforms_module.ChemicalFeatures,
        "BuildFeatureFactory",
        lambda _path: pytest.fail("false gate loaded Shepherd FDEF"),
    )

    pipeline = builder(
        max_tokens=8,
        max_atoms=16,
        use_ligand_cached_rdkit_chirality=False,
        add_hydrogenbond_feature=False,
    )

    assert not any(isinstance(t, AddCachedRDKitFeatures) for t in pipeline.transforms)
