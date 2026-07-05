import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from atomworks.io.utils import non_rcsb
from atomworks.ml.utils.token import apply_token_wise, get_token_starts, spread_token_wise
from biotite.structure import AtomArray
from torchtyping import TensorType

import allatom_design.data.const as const

RUNTIME_POS_CONSTRAINT_COLUMNS = (
    "pdb_key",
    "fixed_pos_seq",
    "fixed_pos_scn",
)
OPTIONAL_POS_CONSTRAINT_COLUMNS = (
    "fixed_pos_override_seq",
    "pos_restrict_aatype",
)
POS_CONSTRAINT_COLUMNS = RUNTIME_POS_CONSTRAINT_COLUMNS + OPTIONAL_POS_CONSTRAINT_COLUMNS
POS_CONSTRAINT_METADATA_COLUMNS = (
    "pocket_distance",
    "constraint_type",
    "num_constrained_residues",
)


def _available_chain_ids(atom_array: AtomArray, chain_annotation: str) -> list[str]:
    chain_ids = np.asarray(atom_array.get_annotation(chain_annotation)).astype(str)
    return sorted(
        {str(chain_id) for chain_id in chain_ids if str(chain_id) != ""},
        key=lambda x: (-len(x), x),
    )


def _missing_chain_error(chain_id: str, available_chain_ids: list[str]) -> ValueError:
    return ValueError(
        f"Chain ID {chain_id} not found in chain annotation: {np.array(available_chain_ids)}."
    )


def _parse_fixed_pos_token(
    pos: str,
    available_chain_ids: list[str],
) -> tuple[str, int | None, int | None]:
    for chain_id in available_chain_ids:
        if not pos.startswith(chain_id):
            continue
        suffix = pos[len(chain_id):]
        if not suffix:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", suffix)
        if match is not None:
            start_residue = int(match.group(1))
            end_residue = int(match.group(2)) if match.group(2) else start_residue
            return chain_id, start_residue, end_residue

    if pos in available_chain_ids:
        return pos, None, None

    inferred = re.fullmatch(r"([A-Za-z]+)(\d+)(?:-(\d+))?", pos)
    if inferred is not None:
        raise _missing_chain_error(inferred.group(1), available_chain_ids)
    if re.fullmatch(r"[A-Za-z]+", pos) is not None:
        raise _missing_chain_error(pos, available_chain_ids)
    raise ValueError(f"Invalid position format: {pos}")


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

        row = pos_constraint_df.loc[example_id]
        fixed_pos_seq, fixed_pos_scn = (
            row.get("fixed_pos_seq", np.nan),
            row.get("fixed_pos_scn", np.nan),
        )

        example = {k: v[i] for k, v in batch.items()}

        fixed_pos_override_seq = row.get("fixed_pos_override_seq", np.nan)
        if not pd.isna(fixed_pos_override_seq):
            if verbose:
                print(f"{example_id}: Overriding sequence at positions {fixed_pos_override_seq}")

            pdb_pos, override_abs_pos, override_aatypes = parse_fixed_pos_override_seq_str(
                fixed_pos_override_seq, example["atom_array"]
            )
            for abs_pos_i, aa in zip(override_abs_pos, override_aatypes):
                batch["restype"][i, abs_pos_i] = F.one_hot(
                    torch.tensor(const.AF3_ENCODING.encode_aa_seq(aa), device=batch["restype"].device),
                    num_classes=const.AF3_ENCODING.n_tokens,
                )

            token_pad_mask = batch["token_pad_mask"][i].bool()
            resnames = const.AF3_ENCODING.idx_to_token[batch["restype"][i][token_pad_mask].argmax(dim=-1).cpu().numpy()]
            atomwise_resnames = spread_token_wise(batch["atom_array"][i], resnames)
            batch["atom_array"][i].set_annotation("res_name", atomwise_resnames)

            fixed_pos_seq = f"{fixed_pos_seq}," if not pd.isna(fixed_pos_seq) else ""
            fixed_pos_seq += ",".join(pdb_pos)

        if not pd.isna(fixed_pos_seq):
            if verbose:
                print(f"{example_id}: Fixing sequence at positions {fixed_pos_seq}")
            abs_fixed_pos_seq = parse_fixed_pos_str(fixed_pos_seq, example["atom_array"])
            seq_cond_mask[i, abs_fixed_pos_seq] = 1

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
            if verbose:
                print(f"{example_id}: Fixing sidechains at positions {fixed_pos_scn}")
            abs_fixed_pos_scn = parse_fixed_pos_str(fixed_pos_scn, example["atom_array"])
            scn_atom_mask = torch.isin(
                example["atom_to_token_map"],
                torch.tensor(abs_fixed_pos_scn, device=example["atom_to_token_map"].device),
            )
            atom_cond_mask[i] = torch.where(scn_atom_mask, example["atom_resolved_mask"], atom_cond_mask[i])

            scn_cond_num_atoms = apply_token_wise(example["atom_array"], scn_atom_mask.cpu().numpy(), np.sum)
            if not pd.isna(fixed_pos_override_seq):
                assert (
                    scn_cond_num_atoms[override_abs_pos] == 0
                ).all(), "Cannot fix sidechains at positions where the sequence from the PDB is overridden."

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

    restrict_pos_mask = torch.zeros((B, N), dtype=torch.float32, device=batch["token_pad_mask"].device)
    allowed_aatype_mask = torch.ones((B, N, K), dtype=torch.float32, device=batch["token_pad_mask"].device)

    if verbose:
        print("\n************** Position-wise amino acid restrictions **************")

    for i, pdb_key in enumerate(batch["example_id"]):
        if pdb_key not in pos_constraint_df.index:
            if verbose:
                print(f"{pdb_key}: No amino acid restrictions specified.")
            continue

        row = pos_constraint_df.loc[pdb_key]
        pos_restrict_aatype = row.get("pos_restrict_aatype", np.nan)

        if pd.isna(pos_restrict_aatype):
            if verbose:
                print(f"{pdb_key}: No position-wise amino acid restrictions specified.")
            continue

        example = {k: v[i] for k, v in batch.items()}

        if verbose:
            print(f"{pdb_key}: Restricting amino acid sampling at positions {pos_restrict_aatype}")

        pdb_pos, abs_pos, allowed_aatypes = parse_pos_restrict_aatype_str(pos_restrict_aatype, example["atom_array"])

        restrict_pos_mask[i, abs_pos] = 1.0

        for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
            allowed_aatype_mask[i, pos_idx, :] = 0.0

            for aa in allowed_aa:
                if aa in const.PROT_LETTER_TO_TOKEN:
                    allowed_aatype_mask[i, pos_idx, const.AF3_ENCODING.encode_aa(aa)] = 1.0
                else:
                    print(
                        f"Warning: Unknown amino acid '{aa}' in restriction for {pdb_key} "
                        f"at position {pdb_pos[abs_pos.index(pos_idx)]}"
                    )

        if verbose:
            for pos_idx, allowed_aa in zip(abs_pos, allowed_aatypes):
                pos_str = pdb_pos[abs_pos.index(pos_idx)]
                print(f" * Position {pos_str}: Restricted to {allowed_aa}")

    if verbose:
        print("\n********************************************************\n")

    return restrict_pos_mask, allowed_aatype_mask


def parse_fixed_pos_str(fixed_pos_str: str, atom_array: AtomArray) -> TensorType["k", int]:
    """
    Parse fixed positions like ["A", "B1", "C10-25", "GA", "GA2-4"] and return
    the corresponding list of absolute token indices.
    """
    chain_annotation = "chain_id"
    chain_ids = _available_chain_ids(atom_array, chain_annotation)
    atom_chain_ids = np.asarray(atom_array.get_annotation(chain_annotation)).astype(str)
    residue_index = atom_array.res_id[get_token_starts(atom_array)]
    fixed_indices = []

    fixed_pos_str = fixed_pos_str.strip()
    if not fixed_pos_str:
        return fixed_indices

    fixed_pos_list = [item.strip() for item in fixed_pos_str.split(",") if item.strip()]

    for pos in fixed_pos_list:
        chain_id, start_residue, end_residue = _parse_fixed_pos_token(pos, chain_ids)

        if start_residue is None:
            atomwise_chain_mask = atom_chain_ids == chain_id
            chain_mask = apply_token_wise(atom_array, atomwise_chain_mask, np.any)
            matching_indices = np.where(chain_mask)[0]
            fixed_indices.extend(matching_indices.tolist())
            continue

        atomwise_range_mask = (
            (atom_chain_ids == chain_id)
            & (atom_array.res_id >= start_residue)
            & (atom_array.res_id <= end_residue)
        )
        range_mask = apply_token_wise(atom_array, atomwise_range_mask, np.any)
        matching_indices = np.where(range_mask)[0]

        found_residues = set(residue_index[matching_indices].tolist())

        for r in range(start_residue, end_residue + 1):
            if r not in found_residues:
                print(f"Warning: Requested position {chain_id}{r} not found in structure.")

        fixed_indices.extend(matching_indices.tolist())

    return fixed_indices


def parse_fixed_pos_override_seq_str(
    override_str: str, atom_array: AtomArray
) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a fixed position sequence override string in the format "A26:A,A27:L" into three lists.
    """
    if not override_str or override_str.strip() == "":
        return [], [], []

    pdb_pos = []
    override_aatypes = []

    overrides = [o.strip() for o in override_str.split(",") if o.strip()]

    for override in overrides:
        parts = override.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid override format: {override}. Expected format: 'A26:A'")

        pos, aatype = parts[0].strip(), parts[1].strip()

        if len(aatype) != 1 or aatype not in const.PROT_LETTER_TO_TOKEN:
            raise ValueError(f"Invalid aatype: {aatype} in {override}. Expected single letter amino acid code.")

        pdb_pos.append(pos)
        override_aatypes.append(aatype)

    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), atom_array)

    return pdb_pos, abs_pos, override_aatypes


def parse_pos_restrict_aatype_str(
    pos_restrict_str: str, atom_array: AtomArray
) -> tuple[list[str], list[int], list[str]]:
    """
    Parse a position restriction string in the format "A26:AVG,A27:VG" into three lists.
    """
    if not pos_restrict_str or pos_restrict_str.strip() == "":
        return [], [], []

    pdb_pos = []
    allowed_aatypes = []

    restrictions = [r.strip() for r in pos_restrict_str.split(",") if r.strip()]

    for restriction in restrictions:
        parts = restriction.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid restriction format: {restriction}. Expected format: 'A26:AVG'")

        pos, aatypes = parts[0].strip(), parts[1].strip()
        pdb_pos.append(pos)
        allowed_aatypes.append(aatypes)

    abs_pos = parse_fixed_pos_str(",".join(pdb_pos), atom_array)

    return pdb_pos, abs_pos, allowed_aatypes


def _format_token_labels(atom_array: AtomArray, token_starts: np.ndarray) -> np.ndarray:
    chain_ids = np.asarray(atom_array.chain_id[token_starts]).astype(str)
    res_ids = np.asarray(atom_array.res_id[token_starts]).astype(str)
    if "ins_code" in atom_array.get_annotation_categories():
        ins_codes = np.asarray(atom_array.ins_code[token_starts]).astype(str)
    else:
        ins_codes = np.full(len(token_starts), "", dtype=object)
    if "atom_name" in atom_array.get_annotation_categories():
        atom_names = np.asarray(atom_array.atom_name[token_starts]).astype(str)
    else:
        atom_names = np.full(len(token_starts), "", dtype=object)

    labels = [
        f"{chain_id}{res_id}{ins_code.strip()}"
        for chain_id, res_id, ins_code in zip(chain_ids, res_ids, ins_codes)
    ]
    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    disambiguated_labels = []
    for idx, (label, atom_name) in enumerate(zip(labels, atom_names)):
        if label_counts[label] > 1:
            suffix = atom_name.strip() or f"token{idx}"
            disambiguated_labels.append(f"{label}:{suffix}")
        else:
            disambiguated_labels.append(label)
    return np.asarray(disambiguated_labels, dtype=object)


def _format_observed_token_axis_sequence(
    observed_labels: np.ndarray,
    observed_symbols: list[str],
    canonical_len: int,
    mask_len: int,
    truncated_note: str | None = None,
) -> str:
    widths = [
        max(len(str(label)), len(str(symbol)))
        for label, symbol in zip(observed_labels, observed_symbols)
    ]
    label_line = " ".join(str(label).ljust(width) for label, width in zip(observed_labels, widths)).rstrip()
    symbol_line = " ".join(str(symbol).ljust(width) for symbol, width in zip(observed_symbols, widths)).rstrip()
    lines = [
        (
            "observed-token-axis "
            f"(not canonical residue positions; canonical_len={canonical_len}, "
            f"token_len={mask_len}; unresolved canonical positions omitted)"
        ),
        f"  residue_labels: {label_line}",
        f"  fixed_tokens:    {symbol_line}",
    ]
    if truncated_note is not None:
        lines.append(f"  note: {truncated_note}")
    return "\n".join(lines)


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

    cond_mask_np = cond_mask.detach().cpu().numpy() if torch.is_tensor(cond_mask) else np.asarray(cond_mask)
    asym_id_np = asym_id.detach().cpu().numpy() if torch.is_tensor(asym_id) else np.asarray(asym_id)
    cond_mask_np = cond_mask_np.astype(bool)

    chain_names = [x.split("_")[0] for x in asym_names]
    chain_name_to_asym_id = {chain_name: i for i, chain_name in enumerate(chain_names)}
    token_starts = get_token_starts(atom_array)
    token_chain_ids = np.asarray(atom_array.chain_id[token_starts]).astype(str)
    token_res_names = np.asarray(atom_array.res_name[token_starts]).astype(str)
    token_letters = np.asarray(
        [const.PROT_TOKEN_TO_LETTER.get(res_name, "X") for res_name in token_res_names],
        dtype=object,
    )
    token_labels = _format_token_labels(atom_array, token_starts)

    for chain_name, info in chain_info.items():
        if chain_name not in chain_name_to_asym_id:
            sequences[chain_name] = "<chain not present in asym_name metadata>"
            continue

        canonical_sequence = info["processed_entity_canonical_sequence"]
        chain_cond_mask = cond_mask_np[asym_id_np == chain_name_to_asym_id[chain_name]]
        if len(canonical_sequence) == len(chain_cond_mask):
            sequences[chain_name] = "".join(
                aa if chain_cond_mask[i] else "-"
                for i, aa in enumerate(canonical_sequence)
            )
            continue

        chain_token_positions = np.where(asym_id_np == chain_name_to_asym_id[chain_name])[0]
        if len(token_letters) >= len(cond_mask_np):
            observed_letters = token_letters[chain_token_positions]
            observed_labels = token_labels[chain_token_positions]
        else:
            chain_token_mask = token_chain_ids == chain_name
            observed_letters = token_letters[chain_token_mask]
            observed_labels = token_labels[chain_token_mask]
        if len(observed_letters) == len(chain_cond_mask):
            observed_symbols = [
                aa if chain_cond_mask[i] else "-"
                for i, aa in enumerate(observed_letters)
            ]
            truncated_note = None
        else:
            n = min(len(observed_letters), len(chain_cond_mask))
            observed_symbols = [
                observed_letters[i] if chain_cond_mask[i] else "-"
                for i in range(n)
            ]
            observed_labels = observed_labels[:n]
            truncated_note = (
                f"truncated to {n} tokens; observed_tokens={len(observed_letters)}, "
                f"mask_tokens={len(chain_cond_mask)}"
            )
        sequences[chain_name] = _format_observed_token_axis_sequence(
            observed_labels=observed_labels,
            observed_symbols=observed_symbols,
            canonical_len=len(canonical_sequence),
            mask_len=len(chain_cond_mask),
            truncated_note=truncated_note,
        )

    for chain_name, sequence in sequences.items():
        print(f"Chain {chain_name}: {sequence}")
