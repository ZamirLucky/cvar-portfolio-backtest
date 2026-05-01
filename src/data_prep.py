"""
Build the cleaned merged price panel, EUR-aligned prices, and log returns.

Outputs
-------
data/processed/etf_prices_long.csv      – long format (date, etf_id, adjusted_close)
data/processed/etf_prices_panel.csv     – wide format, native currencies (date × ETF01..ETF06)
data/processed/etf_prices_panel_eur.csv – wide format, all prices converted to EUR
data/processed/etf_returns_log.csv      – daily log returns from the EUR price panel
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_etf_csv, load_fx_rates

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILES: dict[str, Path] = {
    "ETF01": PROJECT_ROOT / "data" / "raw" / "ETF01_raw.csv",
    "ETF02": PROJECT_ROOT / "data" / "raw" / "ETF02_clean.csv",
    "ETF03": PROJECT_ROOT / "data" / "raw" / "ETF03_raw.csv",
    "ETF04": PROJECT_ROOT / "data" / "raw" / "ETF04_raw.csv",
    "ETF05": PROJECT_ROOT / "data" / "raw" / "ETF05_raw.csv",
    "ETF06": PROJECT_ROOT / "data" / "raw" / "ETF06_raw.csv",
}

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

LONG_OUT = PROCESSED_DIR / "etf_prices_long.csv"
WIDE_OUT = PROCESSED_DIR / "etf_prices_panel.csv"
EUR_OUT = PROCESSED_DIR / "etf_prices_panel_eur.csv"
RETURNS_OUT = PROCESSED_DIR / "etf_returns_log.csv"

# Currency of the price series as downloaded (from raw CSV source data)
ETF_CURRENCY: dict[str, str] = {
    "ETF01": "USD",
    "ETF02": "GBP",
    "ETF03": "EUR",
    "ETF04": "EUR",
    "ETF05": "USD",
    "ETF06": "EUR",
}

# Which FX rate column to multiply by (None = already EUR)
_FX_COL: dict[str, str | None] = {
    "ETF01": "USDEUR",
    "ETF02": "GBPEUR",
    "ETF03": None,
    "ETF04": None,
    "ETF05": "USDEUR",
    "ETF06": None,
}


# ---------------------------------------------------------------------------
# Core build functions
# ---------------------------------------------------------------------------

def build_long_panel(raw_files: dict[str, Path] | None = None) -> pd.DataFrame:
    """Concatenate per-ETF files into a single long DataFrame."""
    if raw_files is None:
        raw_files = RAW_FILES

    frames: list[pd.DataFrame] = []
    for etf_id, path in raw_files.items():
        df = load_etf_csv(path)
        # Override etf_id from the filename mapping (authoritative)
        df["etf_id"] = etf_id
        frames.append(df)

    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["date", "etf_id"]).reset_index(drop=True)
    # Normalise date to ISO string for CSV output
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    return long


def build_wide_panel(long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot long panel to wide format, keeping only common trading dates
    (inner join across all ETFs).
    """
    wide = long.pivot(index="date", columns="etf_id", values="adjusted_close")
    wide.columns.name = None  # remove pivot artefact

    # Inner join: keep only dates present for every ETF
    wide = wide.dropna(how="any")

    # Reorder columns deterministically
    etf_cols = sorted(wide.columns.tolist())
    wide = wide[etf_cols]

    wide = wide.sort_index()
    wide.index.name = "date"
    return wide.reset_index()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_panel(wide: pd.DataFrame) -> None:
    """Assert panel integrity and print a summary. Raises AssertionError on failure."""
    assert wide["date"].duplicated().sum() == 0, "Duplicate dates found"
    assert wide["date"].isna().sum() == 0, "Blank dates found"
    assert wide.isna().sum().sum() == 0, "Missing values in panel"

    dates = pd.to_datetime(wide["date"])
    assert dates.is_monotonic_increasing, "Dates not sorted ascending"

    print("=" * 50)
    print("Panel validation PASSED")
    print(f"  First date : {wide['date'].iloc[0]}")
    print(f"  Last date  : {wide['date'].iloc[-1]}")
    print(f"  Rows       : {len(wide)}")
    print(f"  Columns    : {list(wide.columns)}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_price_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build, validate, and save both output files. Returns (long, wide)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw ETF files...")
    long = build_long_panel()

    print(f"Long panel: {len(long):,} rows across {long['etf_id'].nunique()} ETFs")
    long.to_csv(LONG_OUT, index=False)
    print(f"Saved -> {LONG_OUT.relative_to(PROJECT_ROOT)}")

    print("Building wide panel (inner join on common dates)...")
    wide = build_wide_panel(long)

    validate_panel(wide)
    wide.to_csv(WIDE_OUT, index=False)
    print(f"Saved -> {WIDE_OUT.relative_to(PROJECT_ROOT)}")

    return long, wide


# ---------------------------------------------------------------------------
# EUR currency alignment
# ---------------------------------------------------------------------------

def build_eur_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the native-currency wide panel to EUR-denominated prices.

    Parameters
    ----------
    panel : wide DataFrame with a 'date' column (ISO strings) and ETF01..ETF06

    Returns
    -------
    Wide DataFrame in the same shape, all columns in EUR.
    """
    panel = panel.copy()
    dates = pd.to_datetime(panel["date"])

    start = dates.min().strftime("%Y-%m-%d")
    # end is exclusive in yfinance, so add one day
    end = (dates.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Downloading FX rates ({start} to {end})...")
    fx = load_fx_rates(start, end, cache_dir=RAW_DIR)

    # Reindex FX to the exact panel dates; forward-fill any gaps (e.g. holidays)
    panel_dates = pd.DatetimeIndex(dates.values)
    fx = fx.reindex(panel_dates, method="ffill")

    n_missing_fx = fx.isna().any(axis=1).sum()
    if n_missing_fx > 0:
        dropped = panel_dates[fx.isna().any(axis=1)]
        print(f"WARNING: {n_missing_fx} panel date(s) have no FX data after ffill "
              f"and will be dropped:")
        for d in dropped:
            print(f"  {d.date()}")
        panel = panel[~panel_dates.isin(dropped)].reset_index(drop=True)
        fx = fx.dropna()
    else:
        print("FX alignment: no dates dropped.")

    fx = fx.reset_index(drop=True)

    # Multiply each non-EUR column by the appropriate FX rate
    etf_cols = [c for c in panel.columns if c != "date"]
    for etf in etf_cols:
        fx_col = _FX_COL.get(etf)
        if fx_col is not None:
            panel[etf] = panel[etf].values * fx[fx_col].values

    return panel


# ---------------------------------------------------------------------------
# Log returns
# ---------------------------------------------------------------------------

def build_log_returns(eur_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from the EUR price panel.

    log_return_t = ln(P_t / P_{t-1})

    First row is dropped (no prior price). Returned DataFrame has the same
    column layout with one fewer row.
    """
    price_cols = [c for c in eur_panel.columns if c != "date"]
    prices = eur_panel[price_cols].astype(float).values
    log_ret = np.log(prices[1:] / prices[:-1])

    returns = pd.DataFrame(log_ret, columns=price_cols)
    returns.insert(0, "date", eur_panel["date"].iloc[1:].values)
    returns = returns.reset_index(drop=True)
    return returns


# ---------------------------------------------------------------------------
# Validation (EUR panel + returns)
# ---------------------------------------------------------------------------

def validate_eur_pipeline(eur_panel: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Validate EUR price panel and log-return frame; print summary."""
    assert eur_panel["date"].duplicated().sum() == 0, "EUR panel: duplicate dates"
    assert eur_panel["date"].isna().sum() == 0, "EUR panel: blank dates"
    assert eur_panel.isna().sum().sum() == 0, "EUR panel: missing values"
    dates_eur = pd.to_datetime(eur_panel["date"])
    assert dates_eur.is_monotonic_increasing, "EUR panel: dates not sorted"

    assert len(returns) == len(eur_panel) - 1, "Returns row count mismatch"
    assert not np.isinf(returns.drop(columns="date").values).any(), \
        "Infinite values in log returns"
    assert returns.isna().sum().sum() == 0, "Missing values in log returns"

    print("=" * 55)
    print("EUR price panel validation PASSED")
    print(f"  First date : {eur_panel['date'].iloc[0]}")
    print(f"  Last date  : {eur_panel['date'].iloc[-1]}")
    print(f"  Rows       : {len(eur_panel)}")
    print(f"  Columns    : {list(eur_panel.columns)}")
    print("-" * 55)
    print("Log returns validation PASSED")
    print(f"  First date : {returns['date'].iloc[0]}")
    print(f"  Last date  : {returns['date'].iloc[-1]}")
    print(f"  Rows       : {len(returns)}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# EUR pipeline entry point
# ---------------------------------------------------------------------------

def build_eur_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert prices to EUR, compute log returns, validate, and save. Returns (eur_panel, returns)."""
    print("Loading merged price panel...")
    panel = pd.read_csv(WIDE_OUT)

    eur_panel = build_eur_panel(panel)
    eur_panel.to_csv(EUR_OUT, index=False)
    print(f"Saved -> {EUR_OUT.relative_to(PROJECT_ROOT)}")

    print("Computing daily log returns...")
    returns = build_log_returns(eur_panel)

    validate_eur_pipeline(eur_panel, returns)

    returns.to_csv(RETURNS_OUT, index=False)
    print(f"Saved -> {RETURNS_OUT.relative_to(PROJECT_ROOT)}")

    return eur_panel, returns


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eur":
        build_eur_pipeline()
    else:
        build_price_panel()
