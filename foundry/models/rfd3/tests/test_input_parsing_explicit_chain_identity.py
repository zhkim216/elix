"""Regression tests for explicit chain identity during direct AtomArray parsing."""

# (JH) added: Cover the RFD3 explicit-chain-identity compatibility adapter.

import unittest

import numpy as np
from atomworks.enums import ChainType
from atomworks.io.tools.inference import components_to_atom_array
from biotite import structure as struc

from rfd3.inference.input_parsing import (
    DesignInputSpecification,
    prepare_pipeline_input_from_atom_array,
    preserve_explicit_nonpolymer_identity,
)
from rfd3.transforms.conditioning_base import set_default_conditioning_annotations
from rfd3.utils.inference import set_common_annotations


def _inference_ready_atom_array(components):
    atom_array = components_to_atom_array(components)
    atom_array = set_default_conditioning_annotations(
        atom_array, motif=True, dtype=int
    )
    return set_common_annotations(atom_array)


def _bond_exists(atom_array, index_a, index_b, bond_type):
    bonds = atom_array.bonds.as_array()
    mask = (
        ((bonds[:, 0] == index_a) & (bonds[:, 1] == index_b))
        | ((bonds[:, 0] == index_b) & (bonds[:, 1] == index_a))
    ) & (bonds[:, 2] == bond_type)
    return bool(np.any(mask))


class TestExplicitChainIdentityParsing(unittest.TestCase):
    def test_separate_chain_ligand_default_and_legacy_residue_ids(self):
        ligand_input = _inference_ready_atom_array(
            [
                {
                    "ccd_code": "GLU",
                    "chain_type": "non-polymer",
                    "is_polymer": False,
                    "chain_id": "L",
                }
            ]
        )

        cases = [("default", None, 1), ("explicit_false", False, 1), ("legacy_true", True, 3)]
        for case, allow_legacy, expected_ligand_id in cases:
            with self.subTest(allow_ligand_on_existing_chain=allow_legacy):
                kwargs = {
                    "allow_ligand_on_existing_chain": allow_legacy
                } if allow_legacy is not None else {}
                built = DesignInputSpecification(
                    input=None,
                    atom_array_input=ligand_input,
                    length="2-2",
                    ligand="GLU",
                    **kwargs,
                ).build()
                protein_ids = set(built.res_id[built.chain_id == "A"].tolist())
                ligand_ids = set(built.res_id[built.chain_id == "L"].tolist())

                self.assertEqual(protein_ids, {1, 2})
                self.assertEqual(ligand_ids, {expected_ligand_id})

    def test_standard_amino_acid_ligand_keeps_explicit_non_polymer_identity(self):
        for ccd_code, expected_atoms in [("GLU", 10), ("MET", 9)]:
            with self.subTest(ccd_code=ccd_code):
                atom_array = _inference_ready_atom_array(
                    [
                        {
                            "ccd_code": ccd_code,
                            "chain_type": "non-polymer",
                            "is_polymer": False,
                            "chain_id": "L",
                        }
                    ]
                )
                # Reproduce an AtomWorks CCD-template round trip: standard
                # amino-acid templates carry hetero=False even when the chain
                # identity is explicitly non-polymer.
                atom_array.hetero[:] = False
                original_bonds = atom_array.bonds.as_array().copy()

                data = prepare_pipeline_input_from_atom_array(atom_array)
                parsed = data["atom_array"]

                self.assertEqual(len(parsed), expected_atoms)
                self.assertTrue(
                    np.all(parsed.chain_type == ChainType.NON_POLYMER)
                )
                self.assertFalse(np.any(parsed.is_polymer))
                self.assertTrue(np.all(parsed.hetero))
                self.assertEqual(
                    data["chain_info"]["L"]["chain_type"],
                    ChainType.NON_POLYMER,
                )
                self.assertIs(data["chain_info"]["L"]["is_polymer"], False)

                # Parsing must operate on a copy rather than rewriting the
                # caller's input.
                self.assertFalse(np.any(atom_array.hetero))
                np.testing.assert_array_equal(
                    atom_array.bonds.as_array(), original_bonds
                )

    def test_polymer_modified_residue_keeps_hetero_annotation(self):
        atom_array = _inference_ready_atom_array(
            [
                {
                    "seq": "A(MSE)A",
                    "chain_type": "polypeptide(l)",
                    "is_polymer": True,
                    "chain_id": "A",
                }
            ]
        )
        original_hetero = atom_array.hetero.copy()

        parsed = prepare_pipeline_input_from_atom_array(atom_array)["atom_array"]
        mse_mask = parsed.res_name == "MSE"
        ala_mask = parsed.res_name == "ALA"

        self.assertTrue(np.all(parsed.chain_type == ChainType.POLYPEPTIDE_L))
        self.assertTrue(np.all(parsed.is_polymer))
        self.assertTrue(np.all(parsed.hetero[mse_mask]))
        self.assertFalse(np.any(parsed.hetero[ala_mask]))
        np.testing.assert_array_equal(atom_array.hetero, original_hetero)

    def test_other_chain_type_is_polymer_mismatches_are_not_rejected(self):
        atom_array = _inference_ready_atom_array(
            [
                {
                    "ccd_code": "GLU",
                    "chain_type": "non-polymer",
                    "is_polymer": False,
                    "chain_id": "L",
                }
            ]
        )
        atom_array.is_polymer[:] = True
        atom_array.hetero[:] = False

        adapted = preserve_explicit_nonpolymer_identity(atom_array)
        prepare_pipeline_input_from_atom_array(atom_array)

        # This malformed combination is outside the narrow compatibility
        # adapter.  Preserve AtomWorks' existing inference behavior instead of
        # imposing a new repository-wide validation contract.
        self.assertTrue(np.all(adapted.chain_type == ChainType.NON_POLYMER))
        self.assertTrue(np.all(adapted.is_polymer))
        self.assertFalse(np.any(adapted.hetero))
        self.assertFalse(np.any(atom_array.hetero))

    def test_non_polymer_normalization_preserves_covalent_bond_and_input(self):
        atom_array = _inference_ready_atom_array(
            [
                {
                    "seq": "C",
                    "chain_type": "polypeptide(l)",
                    "is_polymer": True,
                    "chain_id": "A",
                },
                {
                    "ccd_code": "GLU",
                    "chain_type": "non-polymer",
                    "is_polymer": False,
                    "chain_id": "L",
                },
            ]
        )
        ligand_mask = atom_array.chain_id == "L"
        atom_array.hetero[ligand_mask] = False
        protein_index = int(
            np.where(
                (atom_array.chain_id == "A") & (atom_array.atom_name == "SG")
            )[0][0]
        )
        ligand_index = int(
            np.where(
                (atom_array.chain_id == "L") & (atom_array.atom_name == "CD")
            )[0][0]
        )
        atom_array.bonds.add_bond(
            protein_index, ligand_index, struc.BondType.SINGLE
        )
        original_bonds = atom_array.bonds.as_array().copy()

        parsed = prepare_pipeline_input_from_atom_array(atom_array)["atom_array"]
        parsed_protein_index = int(
            np.where((parsed.chain_id == "A") & (parsed.atom_name == "SG"))[0][0]
        )
        parsed_ligand_index = int(
            np.where((parsed.chain_id == "L") & (parsed.atom_name == "CD"))[0][0]
        )

        self.assertTrue(
            _bond_exists(
                parsed,
                parsed_protein_index,
                parsed_ligand_index,
                struc.BondType.SINGLE,
            )
        )
        self.assertTrue(
            np.all(
                parsed.chain_type[parsed.chain_id == "L"]
                == ChainType.NON_POLYMER
            )
        )
        self.assertFalse(np.any(atom_array.hetero[ligand_mask]))
        np.testing.assert_array_equal(atom_array.bonds.as_array(), original_bonds)


if __name__ == "__main__":
    unittest.main()
