"""Stable serialization for LASErMPNN autoregressive decoding orders."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def serialize_decoding_order(local_indices: Iterable[int]) -> str:
    """Serialize zero-based, per-design residue indices without batch offsets."""
    return " ".join(str(int(index)) for index in local_indices)


def decoding_order_sha256(serialized_order: str) -> str:
    """Hash the canonical decoding-order serialization."""
    return hashlib.sha256(serialized_order.encode("ascii")).hexdigest()


def sample_seed(sample_id: str, base_seed: int) -> int:
    """Derive the reproducible per-input seed used by the sampler."""
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
