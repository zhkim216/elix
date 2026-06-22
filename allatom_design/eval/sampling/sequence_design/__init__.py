"""ELIX sequence-design sampling workflow helpers."""

__all__ = [
    "SequenceDesignRunSpec",
    "build_sequence_design_run_spec_from_cfg",
    "build_two_stage_design_context",
    "design_sequence",
    "design_sequence_two_stage",
    "iter_design_sequence_for_run_spec",
    "iter_design_sequence_per_checkpoint",
]


def __getattr__(name):
    if name in {
        "SequenceDesignRunSpec",
        "build_sequence_design_run_spec_from_cfg",
        "design_sequence",
        "iter_design_sequence_for_run_spec",
        "iter_design_sequence_per_checkpoint",
    }:
        from allatom_design.eval.sampling.sequence_design import core

        return getattr(core, name)
    if name in {"build_two_stage_design_context", "design_sequence_two_stage"}:
        from allatom_design.eval.sampling.sequence_design import two_stage

        return getattr(two_stage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
