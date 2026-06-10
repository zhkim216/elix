import pandas as pd
from omegaconf import DictConfig
from contextlib import nullcontext
from joblib import Parallel

def load_optional_csv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path)

def print_phase(label: str) -> None:
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80 + "\n")

def normalize_ccd_code(code) -> str:
    """Normalize a CCD code / element symbol for membership and key comparison.

    Uppercases and strips whitespace, matching the convention of
    ``atomworks.constants.METAL_ELEMENTS`` (which is stored uppercase).
    Do NOT use this for human-readable element symbols (e.g. "Fe"); see
    ``allatom_design.utils.sample_io_utils._normalize_element_symbol`` for that.
    """
    return str(code).strip().upper()
