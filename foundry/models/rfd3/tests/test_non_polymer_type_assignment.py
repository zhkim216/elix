"""Regression coverage for AtomWorks NON_POLYMER molecule typing."""

# (JH) added: Cover NON_POLYMER-aware RFD3 type and sequence features.

import numpy as np
import unittest
from atomworks.enums import ChainType
from atomworks.ml.encoding_definitions import AF3SequenceEncoding
from biotite.structure import AtomArray

from rfd3.transforms.design_transforms import AddGroundTruthSequence
from rfd3.transforms.util_transforms import (
    EncodeAF3TokenLevelFeatures,
    assign_types_,
)


# (JH) added: Cover CCD names that look polymeric but AtomWorks identifies as
# non-polymers, so the input parser and model features cannot silently diverge again.
def _single_token(res_name: str, chain_type: ChainType | None) -> AtomArray:
    atom_array = AtomArray(1)
    atom_array.chain_id = np.array(["L"])
    atom_array.res_id = np.array([1])
    atom_array.res_name = np.array([res_name])
    atom_array.atom_name = np.array(["CA"])
    atom_array.element = np.array(["C"])
    if chain_type is not None:
        atom_array.set_annotation(
            "chain_type", np.array([chain_type.value], dtype=np.int8)
        )
    return atom_array


class TestNonPolymerTypeAssignment(unittest.TestCase):
    def test_assign_types_honors_non_polymer_chain_type(self):
        for res_name in ["MET", "GLU", "GDP", "UDP", "7MG"]:
            with self.subTest(res_name=res_name):
                atom_array = assign_types_(
                    _single_token(res_name, ChainType.NON_POLYMER)
                )

                self.assertEqual(atom_array.is_ligand.tolist(), [True])
                self.assertEqual(atom_array.is_protein.tolist(), [False])
                self.assertEqual(atom_array.is_rna.tolist(), [False])
                self.assertEqual(atom_array.is_dna.tolist(), [False])
                self.assertEqual(atom_array.is_residue.tolist(), [False])

    def test_assign_types_preserves_polymer_and_missing_chain_type_fallbacks(self):
        polymer = assign_types_(_single_token("MET", ChainType.POLYPEPTIDE_L))
        no_chain_type = assign_types_(_single_token("ALA", None))

        self.assertEqual(polymer.is_protein.tolist(), [True])
        self.assertEqual(polymer.is_residue.tolist(), [True])
        self.assertEqual(polymer.is_ligand.tolist(), [False])
        self.assertEqual(no_chain_type.is_protein.tolist(), [True])
        self.assertEqual(no_chain_type.is_residue.tolist(), [True])
        self.assertEqual(no_chain_type.is_ligand.tolist(), [False])

    def test_token_features_encode_non_polymer_met_as_ligand_unknown(self):
        atom_array = _single_token("MET", ChainType.NON_POLYMER)
        atom_array.set_annotation("atomize", np.array([True]))
        atom_array.set_annotation("pn_unit_iid", np.array(["L_1"]))
        atom_array.set_annotation("pn_unit_entity", np.array(["MET"]))
        atom_array.set_annotation("chain_entity", np.array(["MET"]))
        atom_array.set_annotation("within_chain_res_idx", np.array([0]))
        atom_array.set_annotation(
            "is_motif_atom_with_fixed_seq", np.array([True])
        )
        atom_array.set_annotation("is_C_terminus", np.array([False]))
        atom_array.set_annotation("is_N_terminus", np.array([False]))

        transform = EncodeAF3TokenLevelFeatures(AF3SequenceEncoding())
        output = transform.forward({"atom_array": atom_array})
        unknown_index = int(AF3SequenceEncoding().encode(["UNK"])[0])

        self.assertEqual(output["feats"]["is_ligand"].tolist(), [True])
        self.assertEqual(output["feats"]["is_protein"].tolist(), [False])
        self.assertEqual(
            np.argmax(output["feats"]["restype"], axis=-1).tolist(),
            [unknown_index],
        )

    def test_ground_truth_sequence_masks_ligand_with_amino_acid_ccd_name(self):
        atom_array = _single_token("MET", ChainType.NON_POLYMER)
        atom_array.set_annotation("is_ligand", np.array([True]))

        output = AddGroundTruthSequence(AF3SequenceEncoding()).forward(
            {"atom_array": atom_array}
        )
        unknown_index = int(AF3SequenceEncoding().encode(["UNK"])[0])

        self.assertEqual(
            output["ground_truth"]["sequence_gt_I"].tolist(), [unknown_index]
        )
        self.assertEqual(
            output["ground_truth"]["sequence_valid_mask"].tolist(), [False]
        )


if __name__ == "__main__":
    unittest.main()
