"""Load individual ETF CSV files and FX rate series into normalised DataFrames."""

from pathlib import Path

import pandas as pd
import yfinance as yf

_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%y", "%d-%m-%Y"]

# Yahoo Finance tickers for the two required FX conversion rates (units: EUR per 1 foreign)
_FX_TICKERS: dict[str, str] = {
    "USDEUR": "USDEUR=X",
    "GBPEUR": "GBPEUR=X",
}


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


def load_fx_rates(
    start: str,
    end: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Return daily FX close rates aligned to trading days between start and end.

    Returns a DataFrame indexed by date (datetime64) with columns:
        USDEUR  – EUR per 1 USD
        GBPEUR  – EUR per 1 GBP

    If cache_dir is provided, raw downloads are cached as CSV there and
    re-used on subsequent calls to avoid network round-trips.
    """
    frames: dict[str, pd.Series] = {}

    for col, ticker in _FX_TICKERS.items():
        cache_path = (cache_dir / f"FX_{col}_raw.csv") if cache_dir else None

        if cache_path and cache_path.exists():
            raw = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            series = raw["close"].rename(col)
        else:
            raw_df = yf.download(ticker, start=start, end=end, progress=False)
            # yfinance 1.x returns a MultiIndex: ('Close', ticker)
            series = raw_df["Close"][ticker].rename(col)
            series.index = pd.to_datetime(series.index)

            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"close": series}).to_csv(cache_path)

        frames[col] = series

    fx = pd.DataFrame(frames)
    fx.index.name = "date"
    return fx
