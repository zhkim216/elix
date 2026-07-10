from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from biotite.structure import AtomArray

import atomworks.enums as aw_enums
from atomworks.io.utils.sequence import get_1_from_3_letter_code


ResidueKey = tuple[str, int]


@dataclass(frozen=True)
class CaMatch:
    """Resolved exact-residue CA correspondence between two structures."""

    sample_indices: np.ndarray
    pred_indices: np.ndarray
    sample_count: int
    pred_count: int

    @property
    def matched_count(self) -> int:
        return int(len(self.sample_indices))

    @property
    def sample_coverage(self) -> float:
        return float(self.matched_count / self.sample_count) if self.sample_count else 0.0

    @property
    def pred_coverage(self) -> float:
        return float(self.matched_count / self.pred_count) if self.pred_count else 0.0


@dataclass(frozen=True)
class ThreeWayCaMatch:
    """Resolved exact-residue CA correspondence across reference/design/prediction."""

    residue_keys: tuple[ResidueKey, ...]
    reference_indices: np.ndarray
    designed_indices: np.ndarray
    pred_indices: np.ndarray
    reference_count: int
    designed_count: int
    pred_count: int

    @property
    def matched_count(self) -> int:
        return len(self.residue_keys)

    @staticmethod
    def _coverage(matched_count: int, total_count: int) -> float:
        return float(matched_count / total_count) if total_count else 0.0

    @property
    def reference_coverage(self) -> float:
        return self._coverage(self.matched_count, self.reference_count)

    @property
    def designed_coverage(self) -> float:
        return self._coverage(self.matched_count, self.designed_count)

    @property
    def pred_coverage(self) -> float:
        return self._coverage(self.matched_count, self.pred_count)


def _candidate_ca_mask(atom_array: AtomArray, pn_unit_iids: Sequence[str]) -> np.ndarray:
    return (
        np.isin(atom_array.pn_unit_iid, list(pn_unit_iids))
        & (atom_array.atom_name == "CA")
        & (atom_array.chain_type == aw_enums.ChainType.POLYPEPTIDE_L)
    )


def _is_resolved_ca(atom_array: AtomArray, index: int) -> bool:
    return bool(
        atom_array.res_name[index] != "UNK"
        and not np.isnan(atom_array.coord[index]).any()
    )


def ca_by_pn_unit_res_id(
    atom_array: AtomArray,
    pn_unit_iids: Sequence[str],
    *,
    residue_keys: Sequence[ResidueKey] | None = None,
) -> tuple[dict[ResidueKey, int], int]:
    """Index protein CA records by exact ``(pn_unit_iid, label res_id)``."""
    normalized_iids = [str(pn_unit_iid) for pn_unit_iid in pn_unit_iids]
    candidate_indices = np.where(_candidate_ca_mask(atom_array, normalized_iids))[0]
    found_iids = {str(atom_array.pn_unit_iid[index]) for index in candidate_indices}
    missing_iids = [pn_unit_iid for pn_unit_iid in normalized_iids if pn_unit_iid not in found_iids]
    if missing_iids:
        raise ValueError(f"No protein CA records for PN units {missing_iids}")

    allowed_keys = None if residue_keys is None else set(residue_keys)
    by_key: dict[ResidueKey, int] = {}
    resolved_count = 0
    for index in candidate_indices:
        key = (
            str(atom_array.pn_unit_iid[index]),
            int(atom_array.res_id[index]),
        )
        if allowed_keys is not None and key not in allowed_keys:
            continue
        if key in by_key:
            raise ValueError(f"Ambiguous protein CA correspondence key {key}")
        by_key[key] = int(index)
        resolved_count += int(_is_resolved_ca(atom_array, int(index)))
    return by_key, resolved_count


def _assert_designed_prediction_residue_names_match(
    *,
    designed_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    designed_indices: np.ndarray,
    pred_indices: np.ndarray,
    residue_keys: Sequence[ResidueKey],
) -> None:
    designed_names = designed_atom_array.res_name[designed_indices]
    pred_names = pred_atom_array.res_name[pred_indices]
    if np.array_equal(designed_names, pred_names):
        return
    designed_canonical = np.asarray(
        [
            get_1_from_3_letter_code(
                str(res_name),
                chain_type=aw_enums.ChainType.POLYPEPTIDE_L,
                use_closest_canonical=True,
            )
            for res_name in designed_names
        ]
    )
    pred_canonical = np.asarray(
        [
            get_1_from_3_letter_code(
                str(res_name),
                chain_type=aw_enums.ChainType.POLYPEPTIDE_L,
                use_closest_canonical=True,
            )
            for res_name in pred_names
        ]
    )
    if np.array_equal(designed_canonical, pred_canonical):
        return
    mismatches = [
        f"{key}:{designed_name}({designed_one})!={pred_name}({pred_one})"
        for key, designed_name, pred_name, designed_one, pred_one in zip(
            residue_keys,
            designed_names,
            pred_names,
            designed_canonical,
            pred_canonical,
        )
        if designed_one != pred_one
    ]
    raise ValueError(
        "Designed and predicted protein CA residue identities do not match "
        "after canonical projection: "
        + ", ".join(mismatches[:10])
    )


def match_resolved_ca(
    *,
    sample_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    pn_unit_iids: Sequence[str],
) -> CaMatch:
    """Match resolved CAs by exact label residue ID and verify canonical identity."""
    sample_by_key, sample_count = ca_by_pn_unit_res_id(sample_atom_array, pn_unit_iids)
    pred_by_key, pred_count = ca_by_pn_unit_res_id(pred_atom_array, pn_unit_iids)
    resolved_keys = [
        key
        for key, sample_index in sample_by_key.items()
        if key in pred_by_key
        and _is_resolved_ca(sample_atom_array, sample_index)
        and _is_resolved_ca(pred_atom_array, pred_by_key[key])
    ]
    if not resolved_keys:
        raise ValueError(
            "No common resolved protein CA exact residue IDs for PN units "
            f"{list(pn_unit_iids)}"
        )

    sample_indices = np.asarray([sample_by_key[key] for key in resolved_keys], dtype=int)
    pred_indices = np.asarray([pred_by_key[key] for key in resolved_keys], dtype=int)
    _assert_designed_prediction_residue_names_match(
        designed_atom_array=sample_atom_array,
        pred_atom_array=pred_atom_array,
        designed_indices=sample_indices,
        pred_indices=pred_indices,
        residue_keys=resolved_keys,
    )
    return CaMatch(
        sample_indices=sample_indices,
        pred_indices=pred_indices,
        sample_count=sample_count,
        pred_count=pred_count,
    )


def match_reference_designed_predicted_ca(
    *,
    reference_atom_array: AtomArray,
    designed_atom_array: AtomArray,
    pred_atom_array: AtomArray,
    residue_keys: Sequence[ResidueKey],
) -> ThreeWayCaMatch:
    """Match requested CA residue IDs across three structures without offset fallback.

    Reference residue names may differ from the designed sequence.  Designed and
    AF3-predicted residues must have the same canonical identity for every CA used
    in the fit; the evaluator separately validates AF3 output names against the
    exact serialized JSON sequence, including explicit modifications.
    """
    ordered_keys = tuple(dict.fromkeys((str(iid), int(res_id)) for iid, res_id in residue_keys))
    if not ordered_keys:
        raise ValueError("No reference protein residue IDs were selected for CA alignment")
    pn_unit_iids = list(dict.fromkeys(key[0] for key in ordered_keys))
    reference_by_key, reference_count = ca_by_pn_unit_res_id(
        reference_atom_array,
        pn_unit_iids,
        residue_keys=ordered_keys,
    )
    designed_by_key, designed_count = ca_by_pn_unit_res_id(
        designed_atom_array,
        pn_unit_iids,
        residue_keys=ordered_keys,
    )
    pred_by_key, pred_count = ca_by_pn_unit_res_id(
        pred_atom_array,
        pn_unit_iids,
        residue_keys=ordered_keys,
    )
    resolved_keys = tuple(
        key
        for key in ordered_keys
        if key in reference_by_key
        and key in designed_by_key
        and key in pred_by_key
        and _is_resolved_ca(reference_atom_array, reference_by_key[key])
        and _is_resolved_ca(designed_atom_array, designed_by_key[key])
        and _is_resolved_ca(pred_atom_array, pred_by_key[key])
    )
    if not resolved_keys:
        raise ValueError("No common resolved protein CA exact residue IDs across reference/design/prediction")

    reference_indices = np.asarray([reference_by_key[key] for key in resolved_keys], dtype=int)
    designed_indices = np.asarray([designed_by_key[key] for key in resolved_keys], dtype=int)
    pred_indices = np.asarray([pred_by_key[key] for key in resolved_keys], dtype=int)
    _assert_designed_prediction_residue_names_match(
        designed_atom_array=designed_atom_array,
        pred_atom_array=pred_atom_array,
        designed_indices=designed_indices,
        pred_indices=pred_indices,
        residue_keys=resolved_keys,
    )
    return ThreeWayCaMatch(
        residue_keys=resolved_keys,
        reference_indices=reference_indices,
        designed_indices=designed_indices,
        pred_indices=pred_indices,
        reference_count=reference_count,
        designed_count=designed_count,
        pred_count=pred_count,
    )
