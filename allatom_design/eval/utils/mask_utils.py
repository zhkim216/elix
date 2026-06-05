"""
Utils for sampling from sequence design models.
"""

import re
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from atomworks.io.utils import non_rcsb
from atomworks.io.utils.io_utils import to_cif_string
from atomworks.ml.utils.token import apply_token_wise, get_token_starts, spread_token_wise
from atomworks.ml.transforms.atom_array import apply_and_spread_residue_wise

import atomworks.enums as aw_enums
from biotite.structure import AtomArray, get_residue_starts
from omegaconf import DictConfig, OmegaConf
from torchtyping import TensorType
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None  # type: ignore[assignment]

    def delayed(func):
        raise ImportError("joblib is required when using parallel sequence-design utilities") from None

import allatom_design.data.const as const

from allatom_design.eval.utils.data_utils import (
    get_sd_batch,
    preprocess_input,
)
from allatom_design.data.transform.custom_transforms import annotate_ligand_pockets, annotate_ligand_pockets_pseudocb
from atomworks.ml.transforms.filters import remove_unresolved_tokens

from allatom_design.model.seq_denoiser.sd_model import SeqDenoiser
from allatom_design.data.transform.pad import pad_dim
from allatom_design.eval.utils.eval_setup_utils import get_pdb_files
from allatom_design.utils.sample_io_utils import save_cif_file, load_example_with_parse
from allatom_design.utils.atom_array_utils import clean_up_and_renumber_atom_array, insert_unk_residues_for_gaps_in_atom_array
from allatom_design.eval.utils.sequence_recovery import calculate_sequence_recovery


###########################################################
# Sampling Mask Initialization
###########################################################

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
