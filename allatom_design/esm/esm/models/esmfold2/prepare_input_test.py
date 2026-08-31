"""Tests for ESMFold2 input preparation (prepare_input)."""

import pytest
from rdkit import Chem

from esm.models.esmfold2.prepare_input import (
    build_chains_from_input,
    build_feature_tensors,
    compute_token_bonds,
)
from esm.models.esmfold2.processor import clean_esmfold2_input
from esm.models.esmfold2.types import (
    LigandInput,
    ProteinInput,
    StructurePredictionInput,
)


@pytest.mark.parametrize(
    "smiles",
    [
        "c1ccccc1",  # benzene: 6 atoms, 6 bonds
        # The drug-like ligand from the SMILES-vs-CCD issue.
        "COC1=CC=C(N2C3=C(C(C(N)=O)=N2)CCN(C4=CC=C(N5CCCCC5=O)C=C4)C3=O)C=C1",
    ],
)
def test_smiles_ligand_bonds_match_molecular_graph(smiles: str):
    """SMILES ligand bonds must match the molecular graph, not a clique (#313)."""
    spi = StructurePredictionInput(sequences=[LigandInput(id="B", smiles=smiles)])
    chains, tokens, atoms = build_chains_from_input(spi, seed=0)
    token_bonds = compute_token_bonds(tokens, atoms, spi, chains)

    mol = Chem.MolFromSmiles(smiles)
    assert len(tokens) == mol.GetNumAtoms()
    n_edges = int(token_bonds.sum().item()) // 2  # symmetric matrix
    assert n_edges == mol.GetNumBonds()
    assert n_edges < len(tokens) * (len(tokens) - 1) // 2  # not a clique


def test_source_residue_indices_reach_model_features() -> None:
    spi = StructurePredictionInput(
        sequences=[ProteinInput(id="A", sequence="MK", source_residue_indices=[7, 9])]
    )
    chains, tokens, atoms = build_chains_from_input(spi)
    features = build_feature_tensors(chains, tokens, atoms, spi)

    assert [token.residue_index for token in tokens] == [0, 1]
    assert features["residue_index"].tolist() == [7, 9]


def test_default_residue_indices_remain_dense() -> None:
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence="MK")])
    chains, tokens, atoms = build_chains_from_input(spi)
    features = build_feature_tensors(chains, tokens, atoms, spi)

    assert features["residue_index"].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("source_residue_indices", "message"),
    [([7], "one value per residue"), ([7, 7], "strictly increasing")],
)
def test_source_residue_indices_are_validated(
    source_residue_indices: list[int], message: str
) -> None:
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(
                id="A", sequence="MK", source_residue_indices=source_residue_indices
            )
        ]
    )
    with pytest.raises(ValueError, match=message):
        build_chains_from_input(spi)


def test_source_residue_indices_reject_chainbreak_syntax() -> None:
    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence="MK:MK", source_residue_indices=[7, 9, 7, 9])
        ]
    )
    with pytest.raises(ValueError, match="one ProteinInput per chain"):
        clean_esmfold2_input(spi)
