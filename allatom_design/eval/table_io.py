import pandas as pd


def load_optional_csv(path: str | None) -> pd.DataFrame | None:
    if path is None:
        return None
    return pd.read_csv(path, keep_default_na=False)
