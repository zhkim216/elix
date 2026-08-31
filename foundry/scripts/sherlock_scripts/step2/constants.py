"""Constants and path resolution for denovoval step2 (diffused-ligand + all-buried RFD3).

Step2 generalizes the metal buried-token workflow to every ligand CCD in the step1 list:
the ligand is co-diffused (``select_fixed_atoms`` empty) and marked fully buried
(``select_buried`` ALL), and RFD3 grows a pocket around it. RASA burial is guided with
classifier-free guidance at scale 2 (paper-faithful).

Paths resolve to Sherlock ($SCRATCH) when SCRATCH is set, else to the local curation tree.
"""

from __future__ import annotations

import os
from pathlib import Path


# --- Generation parameters -------------------------------------------------------------
DEFAULT_LENGTHS: tuple[int, ...] = (150, 300)

# Full run: 300 designs per (ccd, length) = diffusion_batch_size * n_batches.
FULL_DIFFUSION_BATCH_SIZE = 10
FULL_N_BATCHES = 30

# RASA-buried classifier-free guidance. RFD3 SI §3.4: diffused-ligand binders are
# generated "with the RASA condition set to buried and a CFG scale of 2". We guide the
# RASA feature only (H-bond donor/acceptor features are not conditioned in this workflow).
CFG_SCALE = 2.0
CFG_FEATURES: tuple[str, ...] = ("ref_atomwise_rasa",)

# Expected formal charges for the single-atom metal ions (charge sanity check only).
# Note FE is Fe(III), FE2 is Fe(II). Multi-atom ligands are not charge-checked.
METAL_FORMAL_CHARGES: dict[str, int] = {
    "CA": 2, "CO": 2, "CU": 2, "FE": 3, "FE2": 2, "K": 1,
    "MG": 2, "MN": 2, "NA": 1, "NI": 2, "ZN": 2,
}


# --- Path resolution -------------------------------------------------------------------
_LOCAL_REPO_DATASETS = Path("/home/yjhk/model-dev/datasets")


def _on_sherlock() -> bool:
    return bool(os.environ.get("SCRATCH"))


def default_step1_ccd_list() -> Path:
    """The step1 output TSV that seeds the CCD set (source of truth)."""
    base = (
        Path(os.environ["SCRATCH"]) / "datasets" / "evaluation_datasets"
        if _on_sherlock()
        else _LOCAL_REPO_DATASETS / "evaluation_datasets"
    )
    return (
        base
        / "curation"
        / "ver2"
        / "outputs"
        / "denovoval"
        / "step1"
        / "denovoval_ccd_list.tsv"
    )


def default_source_data_root() -> Path:
    """Root holding staged ligand CIFs (inputs/) and RFD3 JSONs (rfd3_inputs/)."""
    if _on_sherlock():
        return (
            Path(os.environ["SCRATCH"])
            / "datasets" / "evaluation_datasets" / "curation" / "ver2" / "outputs"
            / "denovoval" / "step2" / "source_data"
        )
    return (
        _LOCAL_REPO_DATASETS
        / "evaluation_datasets" / "curation" / "ver2" / "outputs" / "denovoval" / "step2" / "source_data"
    )


def default_ccd_mirror() -> Path:
    if os.environ.get("CCD_MIRROR_PATH"):
        return Path(os.environ["CCD_MIRROR_PATH"])
    if _on_sherlock():
        return Path(os.environ["SCRATCH"]) / "datasets" / "ccd_mirror"
    return _LOCAL_REPO_DATASETS / "ccd_mirror"
