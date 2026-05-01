"""
Build the cleaned merged price panel from per-ETF raw CSV files.

Outputs
-------
data/processed/etf_prices_long.csv   – long format (date, etf_id, adjusted_close)
data/processed/etf_prices_panel.csv  – wide format (date × ETF01..ETF06)
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_etf_csv

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
LONG_OUT = PROCESSED_DIR / "etf_prices_long.csv"
WIDE_OUT = PROCESSED_DIR / "etf_prices_panel.csv"


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


if __name__ == "__main__":
    build_price_panel()
