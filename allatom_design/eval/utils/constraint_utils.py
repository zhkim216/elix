import ast
import re

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from torch import Tensor
from atomworks.io.utils import non_rcsb
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise
from atomworks.ml.utils.token import apply_token_wise, get_token_starts, spread_token_wise
from biotite.structure import AtomArray, get_residue_starts
from omegaconf import DictConfig
from torchtyping import TensorType

import atomworks.enums as aw_enums
from atomworks.ml.transforms.filters import remove_unresolved_tokens

import allatom_design.data.const as const
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets, annotate_ligand_pockets_pseudocb
from allatom_design.utils.sample_io_utils import load_example_with_parse
from allatom_design.eval.utils.data_utils import preprocess_input

def parse_fixed_pos_info(
    batch: dict[str, TensorType["b ..."]], pos_constraint_df: pd.DataFrame | None, verbose: bool = False
) -> dict[str, torch.Tensor]:
    """
    Given a pos_constraint_df containing fixed positions for each PDB, return a batch updated with:
    - a mask for seq-level and atom-level conditioning
    - possibly overridden "res_type"

    The pos_constraint_df should have the following format:
    index: PDB name (not including extension)
    columns: ["fixed_pos_seq", "fixed_pos_scn"]
    where each entry is a comma-separated string of positions in the format "A1-100,B1-100", "A1-10,A15-20", or np.nan.
    """

    seq_cond_mask, atom_cond_mask = batch["seq_cond_mask"].clone(), batch["atom_cond_mask"].clone()

    if pos_constraint_df is None:
        if verbose:
            print("No fixed positions specified, redesigning all positions.")
        return batch

    for i, example_id in enumerate(batch["example_id"]):
        if verbose:
            print(f"\n======================== {example_id} ========================")

        if example_id not in pos_constraint_df.index:
            if verbose:
                print(f"No fixed positions found for {example_id}")
            continue

        ### Get fixed positions from df ###
        row = pos_constraint_df.loc[example_id]
        fixed_pos_seq, fixed_pos_scn = (
            row.get("fixed_pos_seq", np.nan),
            row.get("fixed_pos_scn", np.nan),
        )  # get fixed positions for this PDB

        # Set up example
        example = {k: v[i] for k, v in batch.items()}

        ### Override sequence at specified positions and condition on them ###
        fixed_pos_override_seq = row.get("fixed_pos_override_seq", np.nan)
        if not pd.isna(fixed_pos_override_seq):
            if verbose:
                print(f"{example_id}: Overriding sequence at positions {fixed_pos_override_seq}")

            # parse the override string into a list of positions and aatypes
            pdb_pos, override_abs_pos, override_aatypes = parse_fixed_pos_override_seq_str(
                fixed_pos_override_seq, example["atom_array"]
            )
            for abs_pos_i, aa in zip(override_abs_pos, override_aatypes):
                # Update restype in batch.
                batch["restype"][i, abs_pos_i] = F.one_hot(
                    torch.tensor(const.AF3_ENCODING.encode_aa_seq(aa), device=batch["restype"].device),
                    num_classes=const.AF3_ENCODING.n_tokens,
                )

            # Update atom array annotations.
            token_pad_mask = batch["token_pad_mask"][
                i
            ].bool()  # we need to get rid of padding since atom_arrays are not padded
            resnames = const.AF3_ENCODING.idx_to_token[batch["restype"][i][token_pad_mask].argmax(dim=-1).cpu().numpy()]
            atomwise_resnames = spread_token_wise(batch["atom_array"][i], resnames)
            batch["atom_array"][i].set_annotation("res_name", atomwise_resnames)

            # add to fixed_pos_seq
            fixed_pos_seq = f"{fixed_pos_seq}," if not pd.isna(fixed_pos_seq) else ""
            fixed_pos_seq += ",".join(pdb_pos)  # add the positions to the fixed_pos_seq to condition on them

        ### Create override masks based on fixed sequence and sidechain positions ###
        if not pd.isna(fixed_pos_seq):
            # sequence override
            if verbose:
                print(f"{example_id}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_pos_str(fixed_pos_seq, example["atom_array"])
            seq_cond_mask[i, abs_fixed_pos_seq] = 1

            # print fixed sequence
            if verbose:
                print("Fixed sequence:")
                visualize_conditioning_sequences(
                    example["atom_array"],
                    seq_cond_mask[i][example["token_pad_mask"].bool()],
                    example["asym_id"][example["token_pad_mask"].bool()],
                    example["feat_metadata"]["asym_name"],
                )
        else:
            if verbose:
                print(f"{example_id}: No fixed sequence positions specified.")

        if not pd.isna(fixed_pos_scn):
            # sidechain override
            if verbose:
                print(f"{example_id}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_pos_str(fixed_pos_scn, example["atom_array"])
            scn_atom_mask = torch.isin(
                example["atom_to_token_map"],
                torch.tensor(abs_fixed_pos_scn, device=example["atom_to_token_map"].device),
            )
            atom_cond_mask[i] = torch.where(scn_atom_mask, example["atom_resolved_mask"], atom_cond_mask[i])

            # ensure that we're not fixing sidechains when we override the PDB sequence
            scn_cond_num_atoms = apply_token_wise(example["atom_array"], scn_atom_mask.cpu().numpy(), np.sum)
            if not pd.isna(fixed_pos_override_seq):
                assert (
                    scn_cond_num_atoms[override_abs_pos] == 0
                ).all(), "Cannot fix sidechains at positions where the sequence from the PDB is overridden."

            # print fixed sidechains
            if verbose:
                print("Fixed sidechains:")
                visualize_conditioning_sequences(
                    example["atom_array"],
                    torch.tensor(scn_cond_num_atoms > 0),
                    example["asym_id"][example["token_pad_mask"].bool()].cpu(),
                    example["feat_metadata"]["asym_name"],
                )
        else:
            if verbose:
                print(f"{example_id}: No fixed sidechain positions specified.")

    # Update batch
    batch["seq_cond_mask"] = seq_cond_mask
    batch["atom_cond_mask"] = atom_cond_mask
    return batch


def parse_pos_restrict_aatype_info(
    batch: dict[str, TensorType["b ..."]], pos_constraint_df: pd.DataFrame | None, verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Given a pos_constraint_df containing position restrictions for each PDB, return:
    - a mask indicating which positions have restricted amino acid sampling
    - a mask indicating which amino acids are allowed at each position

    The pos_constraint_df should have the following format:
    index: PDB name (not including extension)
    columns: ["pos_restrict_aatype"]
    where each entry is a comma-separated string of positions in the format "A1:AVG,B10:ILMV", or None.
    """
    B, N = batch["token_pad_mask"].shape
    K = const.AF3_ENCODING.n_tokens

    if pos_constraint_df is None:
        if verbose:
            print("No amino acid restrictions specified, allowing all amino acids at all positions.")
        return None

    # Initialize masks for the entire batch
    restrict_pos_mask = torch.zeros((B, N), dtype=torch.float32, device=batch["token_pad_mask"].device)
    allowed_aatype_mask = torch.ones((B, N, K), dtype=torch.float32, device=batch["token_pad_mask"].device)

    if verbose:
        print("\n************** Position-wise amino acid restrictions **************")

    for i, pdb_key in enumerate(batch["example_id"]):
        if pdb_key not in pos_constraint_df.index:
            if verbose:
                print(f"{pdb_key}: No amino acid restrictions specified.")
            continue

        # Get position restrictions from df
        row = pos_constraint_df.loc[pdb_key]
        pos_restrict_aatype = row.get("pos_restrict_aatype", np.nan)

        if pd.isna(pos_restrict_aatype):
            if verbose:
                print(f"{pdb_key}: No position-wise amino acid restrictions specified.")
            continue

        # Set up example
        example = {k: v[i] for k, v in batch.items()}

        if verbose:
            print(f"{pdb_key}: Restricting amino acid sampling at positions {pos_restrict_aatype}")

        # Parse the restriction string into lists of positions and allowed amino acids
        pdb_pos, abs_pos, allowed_aatypes = parse_pos_restrict_aatype_str(pos_restrict_aatype, example["atom_array"])

        # Mark positions with restrictions
        restrict_pos_mask[i, abs_pos] = 1.0

        # Apply restrictions for each position
        for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
            # First, disallow all amino acids at this position
            allowed_aatype_mask[i, pos_idx, :] = 0.0

            # Then allow only the specified amino acids
            for aa in allowed_aa:
                if aa in const.PROT_LETTER_TO_TOKEN:
                    allowed_aatype_mask[i, pos_idx, const.AF3_ENCODING.encode_aa(aa)] = 1.0
                else:
                    print(
                        f"Warning: Unknown amino acid '{aa}' in restriction for {pdb_key} "
                        f"at position {pdb_pos[abs_pos.index(pos_idx)]}"
                    )

        if verbose:
            # Print a summary of the restrictions
            for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
                pos_str = pdb_pos[abs_pos.index(pos_idx)]
                print(f" * Position {pos_str}: Restricted to {allowed_aa}")

    if verbose:
        print("\n********************************************************\n")

    return restrict_pos_mask, allowed_aatype_mask


def parse_fixed_pos_str(fixed_pos_str: str, atom_array: AtomArray) -> TensorType["k", int]:
    """
    Parse a list of fixed positions in the format ["A", "B1", "C10-25", ...] and
    return the corresponding list of absolute indices.

    Args:
        fixed_pos_list (str): Comma-separated string representing fixed positions (e.g., "A,B1,C10-25").
        atom_array (AtomArray): AtomArray object containing the atom array.

    Returns:
        TensorType["k", int]: List of absolute indices to set to 1 in the masks.
    """
    chain_annotation = "chain_id"  # we use chain_id for fixing positions
    residue_index = atom_array.res_id[get_token_starts(atom_array)]
    fixed_indices = []

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return fixed_indices  # no positions specified

    fixed_pos_list = [item.strip() for item in fixed_pos_str.split(",") if item.strip()]

    for pos in fixed_pos_list:
        # Match pattern like "A10" or "A10-25"
        match_with_residues = re.match(r"([A-Za-z])(\d+)(?:-(\d+))?$", pos)
        # Match pattern for just a chain ID, e.g., "A"
        match_chain_only = re.match(r"([A-Za-z])$", pos)

        if match_with_residues:
            chain_letter = match_with_residues.group(1)
            start_residue = int(match_with_residues.group(2))
            end_residue = int(match_with_residues.group(3)) if match_with_residues.group(3) else start_residue

            if chain_letter not in atom_array.get_annotation(chain_annotation):
                raise ValueError(
                    f"Chain ID {chain_letter} not found in chain annotation: {np.unique(atom_array.get_annotation(chain_annotation))}."
                )

            # For the given chain, create a mask for all residues in the desired range
            atomwise_range_mask = (
                (atom_array.get_annotation(chain_annotation) == chain_letter)
                & (atom_array.res_id >= start_residue)
                & (atom_array.res_id <= end_residue)
            )
            range_mask = apply_token_wise(atom_array, atomwise_range_mask, np.any)  # get per-token mask
            matching_indices = np.where(range_mask)[0]

            # Check that each residue in the requested range; warn if not found
            found_residues = set(residue_index[matching_indices].tolist())

            for r in range(start_residue, end_residue + 1):
                if r not in found_residues:
                    print(f"Warning: Requested position {chain_letter}{r} not found in structure.")

            # Extend our fixed indices with whatever we did find
            fixed_indices.extend(matching_indices.tolist())
        elif match_chain_only:
            chain_letter = match_chain_only.group(1)

            if chain_letter not in atom_array.get_annotation(chain_annotation):
                raise ValueError(
                    f"Chain ID {chain_letter} not found in chain annotation: {np.unique(atom_array.get_annotation(chain_annotation))}."
                )

            # For the given chain, create a mask for all residues
            atomwise_chain_mask = atom_array.get_annotation(chain_annotation) == chain_letter
            chain_mask = apply_token_wise(atom_array, atomwise_chain_mask, np.any)
            matching_indices = np.where(chain_mask)[0]
            fixed_indices.extend(matching_indices.tolist())
        else:
            raise ValueError(f"Invalid position format: {pos}")

    return fixed_indices


def parse_fixed_pos_override_seq_str(
    override_str: str, atom_array: AtomArray
) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a fixed position sequence override string in the format "A26:A,A27:L" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and override amino acids (e.g., ["A", "L"]).

    Args:
        override_str (str): Comma-separated string of position overrides
                           in the format "<chain+residue>:<desired aatype>"
        atom_array (AtomArray): AtomArray object containing the atom array.

    Returns:
        tuple: (pdb_pos, abs_pos, override_aatypes) - lists with corresponding entries
    """
    if not override_str or override_str.strip() == "":
        return [], [], []

    pdb_pos = []
    override_aatypes = []

    # Split by comma and process each override
    overrides = [o.strip() for o in override_str.split(",") if o.strip()]

    for override in overrides:
        # Split by colon to get position and override aatype
        parts = override.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid override format: {override}. Expected format: 'A26:A'")

        pos, aatype = parts[0].strip(), parts[1].strip()

        if len(aatype) != 1 or aatype not in const.PROT_LETTER_TO_TOKEN:
            raise ValueError(f"Invalid aatype: {aatype} in {override}. Expected single letter amino acid code.")

        pdb_pos.append(pos)
        override_aatypes.append(aatype)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), atom_array)

    return pdb_pos, abs_pos, override_aatypes


def parse_pos_restrict_aatype_str(
    pos_restrict_str: str, atom_array: AtomArray
) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a position restriction string in the format "A26:AVG,A27:VG" into three lists:
    PDB positions (e.g., ["A26", "A27"]), absolute positions in the tensor, and allowed aatypes (e.g., ["AVG", "VG"]).

    Args:
        pos_restrict_str (str): Comma-separated string of position restrictions
                               in the format "<chain+residue>:<allowed aatypes>"
        atom_array (AtomArray): AtomArray object containing the atom array.

    Returns:
        tuple: (pdb_pos, abs_pos, allowed_aatypes) - lists with corresponding entries
    """
    if not pos_restrict_str or pos_restrict_str.strip() == "":
        return [], [], []

    pdb_pos = []
    allowed_aatypes = []

    # Split by comma and process each restriction.
    restrictions = [r.strip() for r in pos_restrict_str.split(",") if r.strip()]

    for restriction in restrictions:
        # Split by colon to get position and allowed aatypes.
        parts = restriction.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid restriction format: {restriction}. Expected format: 'A26:AVG'")

        pos, aatypes = parts[0].strip(), parts[1].strip()
        pdb_pos.append(pos)
        allowed_aatypes.append(aatypes)

    # Get absolute positions for the given chain+residue
    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), atom_array)

    return pdb_pos, abs_pos, allowed_aatypes


def visualize_conditioning_sequences(
    atom_array: AtomArray,
    cond_mask: TensorType["n", int],
    asym_id: TensorType["n", int],
    asym_names: list[str],
) -> str:
    """
    Visualize the conditioning sequence for a given atom array.
    """
    chain_info = non_rcsb.initialize_chain_info_from_atom_array(atom_array)
    sequences = {}

    # Map from chain_name to asym_id.
    chain_names = [x.split("_")[0] for x in asym_names]  # for now, ignore transforms
    chain_name_to_asym_id = {chain_name: i for i, chain_name in enumerate(chain_names)}

    for chain_name, info in chain_info.items():
        sequence = info["processed_entity_canonical_sequence"]
        # Replace with "-" where cond_mask is 0
        chain_cond_mask = cond_mask[asym_id == chain_name_to_asym_id[chain_name]]
        sequence = "".join([x if chain_cond_mask[i] else "-" for i, x in enumerate(sequence)])
        sequences[chain_name] = sequence

    for chain_name, sequence in sequences.items():
        print(f"Chain {chain_name}: {sequence}")

def _indices_to_pos_string(chain_ids: np.ndarray, res_ids: np.ndarray) -> str:
    """Convert (chain_id, res_id) array to "A1-10,B5-8" format"""
    chain_to_res = {}
    for chain_id, res_id in zip(chain_ids, res_ids):
        chain_to_res.setdefault(chain_id, []).append(res_id)

    pos_parts = []
    for chain_id in sorted(chain_to_res.keys()):
        res_list = sorted(set(chain_to_res[chain_id]))
        if not res_list:
            continue

        # Group consecutive res_id into ranges
        ranges = []
        start = end = res_list[0]
        for r in res_list[1:]:
            if r == end + 1:
                end = r
            else:
                ranges.append((start, end))
                start = end = r
        ranges.append((start, end))

        for s, e in ranges:
            pos_parts.append(f"{chain_id}{s}" if s == e else f"{chain_id}{s}-{e}")

    return ",".join(pos_parts)

def create_pos_constraint_dict_from_pocket(
    pdb_key: str,
    atom_array: AtomArray,
    pocket_distance: float = 5.0,
    constraint_type: str = "pocket",  # "pocket" or "scaffold"
    receptor_pn_unit_iids: list[str] = None,
    ligand_pn_unit_iids: list[str] = None,
    use_pseudocb_for_pocket_annotation: bool = False,
    sample_path: str = None,
    return_ligand_mpnn_format: bool = False,
) -> dict:
    """
    Create a pos_constraint_dict from an atom array based on ligand pocket annotation.

    Args:
        pdb_key: Identifier for the PDB entry
        atom_array: AtomArray containing protein and ligand atoms
        pocket_distance: Distance threshold for pocket identification (Angstroms)
        constraint_type: "pocket" to constrain pocket residues, "scaffold" to constrain non-pocket residues
        receptor_pn_unit_iids: List of receptor (protein) pn_unit_iids
        ligand_pn_unit_iids: List of ligand pn_unit_iids
        sample_path: Path to the CIF file (required if return_ligand_mpnn_format=True)
        return_ligand_mpnn_format: If True, also include LigandMPNN CSV fields (pdb_path, chains, fixed_residues)

    Returns:
        Dictionary with pdb_key, fixed_pos_seq, fixed_pos_scn, and metadata.
        If return_ligand_mpnn_format=True, also includes pdb_path, chains, fixed_residues for LigandMPNN.
    """
    # Annotate ligand pockets
    if use_pseudocb_for_pocket_annotation:
        annotated_atom_array = annotate_ligand_pockets_pseudocb(
            atom_array=atom_array,
            pocket_distance=pocket_distance,
            n_min_ligand_atoms=1,
            annotation_name="is_ligand_pocket"
        )
    else:
        annotated_atom_array = annotate_ligand_pockets(
            atom_array=atom_array,
            pocket_distance=pocket_distance,
            receptor_pn_unit_iids=receptor_pn_unit_iids,
            ligand_pn_unit_iids=ligand_pn_unit_iids,
            annotation_name="is_ligand_pocket"
        )

    # Spread residue-wise: if any atom in a residue is in pocket, mark all atoms in that residue as pocket
    residue_wise_pocket_mask = apply_and_spread_residue_wise(annotated_atom_array, annotated_atom_array.get_annotation("is_ligand_pocket"), function=np.any)
    protein_mask = annotated_atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L

    if constraint_type == "pocket":
        constrained_mask = protein_mask & residue_wise_pocket_mask
    elif constraint_type == "scaffold":
        constrained_mask = protein_mask & ~residue_wise_pocket_mask
    else:
        raise ValueError(f"Invalid constraint type: {constraint_type}")

    # Get constrained atom array
    constrained_atom_array = annotated_atom_array[constrained_mask]

    # Early return if no constrained residues
    if len(constrained_atom_array) == 0:
        return {
            'pdb_key': pdb_key,
            'fixed_pos_seq': "",
            'fixed_pos_scn': np.nan,
            'pocket_distance': pocket_distance,
            'constraint_type': constraint_type,
            'num_constrained_residues': 0,
        }, {}

    # Get residue starts
    res_starts = get_residue_starts(constrained_atom_array)
    chain_ids = constrained_atom_array.chain_id[res_starts]
    res_ids = constrained_atom_array.res_id[res_starts]

    result = {
        'pdb_key': pdb_key,
        'fixed_pos_seq': _indices_to_pos_string(chain_ids, res_ids),
        'fixed_pos_scn': np.nan,
        'pocket_distance': pocket_distance,
        'constraint_type': constraint_type,
        'num_constrained_residues': len(res_starts),
    }

    # Add LigandMPNN format fields if requested
    results_for_ligand_mpnn = {}
    if return_ligand_mpnn_format:
        results_for_ligand_mpnn['pdb_path'] = sample_path if sample_path else ""
        fixed_residues_list = [f"{cid}{rid}" for cid, rid in zip(chain_ids, res_ids)]
        results_for_ligand_mpnn['fixed_residues'] = " ".join(fixed_residues_list)
        # Get chains to parse (protein + ligand)
        protein_chain_ids = list({pn_unit_iid.split("_")[0] for pn_unit_iid in receptor_pn_unit_iids})
        ligand_chain_ids = list({pn_unit_iid.split("_")[0] for pn_unit_iid in ligand_pn_unit_iids})
        results_for_ligand_mpnn['chains'] = ",".join(protein_chain_ids + ligand_chain_ids)

    return result, results_for_ligand_mpnn