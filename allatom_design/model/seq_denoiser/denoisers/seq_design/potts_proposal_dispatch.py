"""Resolve the single proposal mode used by Potts sequence sampling."""

from enum import Enum
from typing import Literal


class PottsProposalMode(str, Enum):
    """Mutually exclusive proposal modes supported by ``sample_potts``."""

    DLMC = "dlmc"
    CHROMATIC = "chromatic"
    GUIDED_DLMC = "guided_dlmc"
    LOCAL_MIXTURE_DLMC = "local_mixture_dlmc"


def resolve_potts_proposal_mode(
    proposal: Literal["dlmc", "chromatic"] | str,
    *,
    has_guidance: bool,
    has_local_mixture: bool,
    rejection_step: bool,
) -> PottsProposalMode:
    """Validate proposal compatibility and return one authoritative mode."""
    if has_guidance and has_local_mixture:
        raise NotImplementedError(
            "Classifier-free guidance and local Potts mixing cannot be combined."
        )

    if has_local_mixture:
        if proposal != "dlmc":
            raise NotImplementedError(
                "Local Potts mixing is only supported with proposal='dlmc'."
            )
        if rejection_step:
            raise NotImplementedError(
                "Metropolis-Hastings rejection step is not supported for Potts "
                "mixing modes; set rejection_step=false."
            )
        return PottsProposalMode.LOCAL_MIXTURE_DLMC

    if has_guidance:
        if proposal != "dlmc":
            raise NotImplementedError(
                "Potts guidance is only supported with the DLMC proposal; got "
                f"proposal={proposal!r}."
            )
        return PottsProposalMode.GUIDED_DLMC

    if proposal == "chromatic":
        return PottsProposalMode.CHROMATIC
    if proposal == "dlmc":
        return PottsProposalMode.DLMC
    raise NotImplementedError(f"Unknown Potts proposal: {proposal!r}")
