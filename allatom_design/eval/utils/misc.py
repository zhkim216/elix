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
