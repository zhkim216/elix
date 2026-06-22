"""Evaluation metrics for sequence design / docking / self-consistency.

This package was split out of the former single-file ``eval_metrics.py`` module.
"""

from allatom_design.eval.metrics.af3_confidence import extract_af3_confidence_metrics
from allatom_design.eval.metrics.docking import compute_docking_metrics_atomarray
from allatom_design.eval.metrics.ligand_rmsd import (
    calculate_ligand_rmsd_with_binding_site_superposition,
)
from allatom_design.eval.metrics.self_consistency import (
    compute_self_consistency_metrics_atomarray,
)
from allatom_design.eval.metrics.tmalign import _compute_tmalign_score

__all__ = [
    "calculate_sequence_recovery",
    "extract_af3_confidence_metrics",
    "compute_docking_metrics_atomarray",
    "calculate_ligand_rmsd_with_binding_site_superposition",
    "compute_self_consistency_metrics_atomarray",
    "_compute_tmalign_score",
]


def __getattr__(name):
    if name == "calculate_sequence_recovery":
        from allatom_design.eval.metrics.sequence_recovery import calculate_sequence_recovery

        return calculate_sequence_recovery
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
