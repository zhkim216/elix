"""Sequence-design ensemble staging helpers."""

__all__ = [
    "DEFAULT_ENSEMBLE_CONDITIONING_CFG",
    "EnsembleStagingResult",
    "annotate_ligand_conformer_batch",
    "apply_ensemble_noise",
    "compute_ensemble_potts_params",
    "compute_member_coefficients",
    "compute_ligand_conformer_member_coefficients",
    "ensemble_conditioning_enabled",
    "expand_pos_constraint_df_for_members",
    "expand_pos_constraint_df_for_ligand_conformer_members",
    "iter_member_batches",
    "iter_ligand_conformer_member_batches",
    "ligand_conformer_conditioning_enabled",
    "ligand_conformer_target_count",
    "make_ensemble_potts_aux_provider",
    "normalize_ensemble_conditioning_cfg",
    "pharm_retrieval_conditioning_enabled",
    "repeat_batch_for_ensembles",
    "sampling_df_has_pdb_key",
    "sampling_row_for_member",
    "stage_ligand_conformer_ensembles",
    "stage_pharm_retrieval_ensembles",
]


def __getattr__(name):
    if name in {
        "DEFAULT_ENSEMBLE_CONDITIONING_CFG",
        "apply_ensemble_noise",
        "compute_ensemble_potts_params",
        "ensemble_conditioning_enabled",
        "ligand_conformer_conditioning_enabled",
        "make_ensemble_potts_aux_provider",
        "normalize_ensemble_conditioning_cfg",
        "pharm_retrieval_conditioning_enabled",
        "repeat_batch_for_ensembles",
    }:
        from allatom_design.eval.sampling.sequence_design.ensemble import conditioning

        return getattr(conditioning, name)
    if name in {
        "annotate_ligand_conformer_batch",
        "compute_ligand_conformer_member_coefficients",
        "expand_pos_constraint_df_for_ligand_conformer_members",
        "iter_ligand_conformer_member_batches",
        "ligand_conformer_target_count",
        "stage_ligand_conformer_ensembles",
    }:
        from allatom_design.eval.sampling.sequence_design.ensemble import ligand_conformer

        return getattr(ligand_conformer, name)
    if name == "stage_pharm_retrieval_ensembles":
        from allatom_design.eval.sampling.sequence_design.ensemble import pharm_retrieval

        return getattr(pharm_retrieval, name)
    if name in {
        "EnsembleStagingResult",
        "compute_member_coefficients",
        "expand_pos_constraint_df_for_members",
        "iter_member_batches",
        "sampling_df_has_pdb_key",
        "sampling_row_for_member",
    }:
        from allatom_design.eval.sampling.sequence_design.ensemble import staging

        return getattr(staging, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
