import numpy as np
import torch
from torchtyping import TensorType

from allatom_design.eval.utils.sampling_inputs import (
    is_role_sampling_inputs,
    matched_role_sampling_input_row,
    role_context_pn_unit_iids_from_sampling_row,
)


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


def apply_role_context_conditioning_masks(
    batch: dict[str, TensorType["b ..."]],
    sampling_inputs_df=None,
    *,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """Condition sequence and atoms for context role units in role-schema sampling."""
    if sampling_inputs_df is None or not is_role_sampling_inputs(sampling_inputs_df):
        return batch

    seq_cond_mask = batch["seq_cond_mask"].clone()
    atom_cond_mask = batch["atom_cond_mask"].clone()

    for i, example_id in enumerate(batch["example_id"]):
        sampling_row = matched_role_sampling_input_row(
            sampling_inputs_df,
            sample_id=str(example_id),
        )
        if sampling_row is None:
            continue

        context_iids = role_context_pn_unit_iids_from_sampling_row(sampling_row)
        if not context_iids:
            if verbose:
                print(f"{example_id}: role-schema context is empty; no role context conditioning applied.")
            continue

        atom_array = batch["atom_array"][i]
        context_atom_mask_np = _atom_pn_unit_mask(atom_array, context_iids)
        if context_atom_mask_np.sum() == 0:
            raise ValueError(
                f"{example_id}: context_pn_unit_iids not found after preprocessing: {context_iids}"
            )

        device = atom_cond_mask.device
        context_atom_mask = torch.zeros_like(atom_cond_mask[i], dtype=torch.bool)
        n_atoms = min(len(context_atom_mask_np), context_atom_mask.shape[0])
        context_atom_mask[:n_atoms] = torch.as_tensor(
            context_atom_mask_np[:n_atoms],
            dtype=torch.bool,
            device=device,
        )

        atom_cond_mask[i] = torch.where(
            context_atom_mask,
            batch["atom_resolved_mask"][i],
            atom_cond_mask[i],
        )

        token_context_mask = torch.zeros_like(seq_cond_mask[i], dtype=torch.bool)
        valid_context_atoms = (
            context_atom_mask
            & batch["atom_pad_mask"][i].bool()
            & batch["atom_resolved_mask"][i].bool()
        )
        if valid_context_atoms.any():
            token_indices = batch["atom_to_token_map"][i][valid_context_atoms].long()
            token_indices = token_indices[
                (token_indices >= 0) & (token_indices < token_context_mask.shape[0])
            ]
            token_context_mask[token_indices] = True

        seq_cond_mask[i] = torch.where(
            token_context_mask,
            batch["token_resolved_mask"][i],
            seq_cond_mask[i],
        )

        if verbose:
            print(
                f"{example_id}: role context conditioned "
                f"{int(token_context_mask.sum().item())} tokens from {context_iids}"
            )

    batch["seq_cond_mask"] = seq_cond_mask
    batch["atom_cond_mask"] = atom_cond_mask
    return batch


def _atom_pn_unit_mask(atom_array, pn_unit_iids: list[str]):
    if "pn_unit_iid" not in atom_array.get_annotation_categories():
        raise ValueError("pn_unit_iid annotation is required for role context conditioning")
    query = {str(pn_unit_iid) for pn_unit_iid in pn_unit_iids}
    return np.asarray([str(pn_unit_iid) in query for pn_unit_iid in atom_array.pn_unit_iid], dtype=bool)
