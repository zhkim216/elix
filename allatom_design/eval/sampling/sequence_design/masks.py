import torch
from torchtyping import TensorType


def initialize_sampling_masks(batch: dict[str, TensorType["b ..."]], protein_only: bool = False) -> dict[str, torch.Tensor]:
    """
    Initialize the sampling masks for the batch. Modifies batch in place and returns it.
    """
    # Initialize sequence mask: always condition on non-protein or non-standard residues.
    seq_cond_mask = torch.zeros_like(batch["token_pad_mask"])
    standard_aa_prot_token_mask = batch["token_is_protein_chain"] * (~batch["is_atomized"]) * batch["token_resolved_mask"] * batch["token_pad_mask"]

    seq_cond_mask = torch.where(standard_aa_prot_token_mask.bool(),
                                    seq_cond_mask,
                                    batch["token_resolved_mask"])

    batch["seq_cond_mask"] = seq_cond_mask * batch["token_pad_mask"] * batch["token_resolved_mask"]

    # Initialize atom mask: condition on backbone atoms of standard amino acids in protein chains or all atoms in non-standard residues and non-protein chains
    standard_aa_prot_atom_mask = batch["atom_is_protein_chain"] * (1 - batch["atom_is_atomized"]) * batch["atom_resolved_mask"] * batch["atom_pad_mask"]
    standard_aa_prot_bb_atom_mask = standard_aa_prot_atom_mask * batch["prot_bb_atom_mask"]

    batch["atom_cond_mask"] = torch.where(standard_aa_prot_atom_mask.bool(),
                                          standard_aa_prot_bb_atom_mask,
                                          batch["atom_resolved_mask"])

    if protein_only:
        batch["seq_cond_mask"] = batch["seq_cond_mask"] * batch["token_is_protein_chain"]
        batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_is_protein_chain"]

    # Ensure that all atoms in atom_cond_mask are resolved and atom_cond_mask is masked out the padding atoms
    batch["atom_cond_mask"] = batch["atom_cond_mask"] * batch["atom_pad_mask"] * batch["atom_resolved_mask"]

    return batch
