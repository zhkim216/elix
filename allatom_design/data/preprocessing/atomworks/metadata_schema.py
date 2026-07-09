"""Schema helpers for AtomWorks metadata parquet files."""

import pandas as pd

NUMERIC_NULLABLE_COLS = (
    "resolution",
    "q_pn_unit_n_coordination_partners_metal",
    "q_pn_unit_n_coordination_partners_halide",
    "q_pn_unit_n_neighboring_heavy_atoms_small_molecule",
    "q_pn_unit_avg_occupancy_nonpolymer",
)


def coerce_nullable_numeric_columns(df: pd.DataFrame) -> None:
    """Normalize nullable numeric columns before parquet schema inference."""
    for col in NUMERIC_NULLABLE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
