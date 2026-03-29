"""
data_loader.py — Read, normalise and validate the research CSV.

Responsibilities:
- Load any CSV that contains trade/campaign records.
- Normalise column names (strip whitespace, lower-case optional).
- Validate that required columns are present.
- Coerce numeric factor columns; rows that fail coercion are flagged, not dropped silently.
- Expose a single public function: load_data(path, ...) -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    NUMERIC_FACTOR_COLS,
    RETURN_COL,
    SETUP_ID_COL,
    WIN_COL,
)
from utils import die, info, warn


# Columns that must be present for the framework to function
REQUIRED_COLUMNS: list[str] = [SETUP_ID_COL, RETURN_COL, WIN_COL]


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from column names.
    Also handle a UTF-8 BOM that sometimes appears in the first column name.
    """
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df


def _validate_required(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        die(
            f"CSV '{path}' is missing required columns: {missing}\n"
            f"  Available columns: {list(df.columns)}"
        )


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce factor columns to float.  Values that cannot be converted become NaN.
    A summary warning is emitted for any column with more than zero failures.
    """
    for col in NUMERIC_FACTOR_COLS:
        if col not in df.columns:
            continue
        original = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        failed = df[col].isna() & original.notna()
        n_failed = int(failed.sum())
        if n_failed > 0:
            warn(f"Column '{col}': {n_failed} value(s) could not be coerced to float → set to NaN.")

    # Coerce the outcome columns too
    df[RETURN_COL] = pd.to_numeric(df[RETURN_COL], errors="coerce")
    df[WIN_COL] = df[WIN_COL].map(
        lambda v: (
            True if str(v).strip().lower() in {"true", "1", "yes"} else
            False if str(v).strip().lower() in {"false", "0", "no"} else
            np.nan
        )
    )
    return df


def _coerce_setup_id(df: pd.DataFrame) -> pd.DataFrame:
    """Convert SetupID to integer where possible; leave as-is otherwise."""
    df[SETUP_ID_COL] = pd.to_numeric(df[SETUP_ID_COL], errors="coerce")
    n_nan = df[SETUP_ID_COL].isna().sum()
    if n_nan:
        warn(f"SetupID column '{SETUP_ID_COL}': {n_nan} rows have unparseable values — they will be excluded.")
    df = df.dropna(subset=[SETUP_ID_COL]).copy()
    df[SETUP_ID_COL] = df[SETUP_ID_COL].astype(int)
    return df


def load_data(path: Path | str) -> pd.DataFrame:
    """
    Load and validate a research CSV.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for analysis.
        - SetupID is an int column.
        - return_pct and is_win are numeric / boolean.
        - Factor columns are float (NaN where missing or unparseable).
        - Rows missing return_pct or is_win are dropped with a warning.
    """
    path = Path(path)
    if not path.exists():
        die(f"Input file not found: {path}")

    info(f"Loading data from {path}")
    df = pd.read_csv(path, low_memory=False)
    info(f"  Raw rows: {len(df):,}  |  Raw columns: {len(df.columns)}")

    df = _normalise_columns(df)
    _validate_required(df, path)
    df = _coerce_numerics(df)
    df = _coerce_setup_id(df)

    # Drop rows where the outcome fields are missing
    n_before = len(df)
    df = df.dropna(subset=[RETURN_COL, WIN_COL]).copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        warn(f"Dropped {n_dropped} rows with missing return_pct / is_win.")

    info(f"  Clean rows: {len(df):,}  |  SetupIDs found: {sorted(df[SETUP_ID_COL].unique())}")
    return df.reset_index(drop=True)
