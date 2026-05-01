"""Load individual ETF CSV files into a normalised DataFrame."""

from pathlib import Path

import pandas as pd

_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%y", "%d-%m-%Y"]


def _parse_dates(series: pd.Series) -> pd.Series:
    """Try each known date format in turn; raise if none succeeds."""
    for fmt in _DATE_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt)
            return parsed
        except (ValueError, TypeError):
            continue
    # Last resort: let pandas infer (dayfirst=True for DD-MM-* ambiguity)
    return pd.to_datetime(series, dayfirst=True)


def load_etf_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load one ETF CSV and return a DataFrame with columns:
        date (datetime64), etf_id (str), adjusted_close (float)

    Dates are converted to ISO YYYY-MM-DD string in the returned frame so
    downstream callers work uniformly.
    """
    df = pd.read_csv(filepath, dtype=str)

    required = {"date", "etf_id", "adjusted_close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filepath}: missing columns {missing}")

    df["date"] = _parse_dates(df["date"])
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="raise")

    return df[["date", "etf_id", "adjusted_close"]].copy()
