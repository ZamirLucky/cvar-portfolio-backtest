import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path

# ----------------------------
# File paths
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]  # repo root
YAHOO_FILE = ROOT / "data" / "raw" / "ETF02_raw.csv"
INVESTING_FILE = ROOT / "data" / "raw" / "ETF02_raw_investing.csv"
OUTPUT_FILE = ROOT / "data" / "raw" / "ETF02_clean.csv"

# ----------------------------
# Metadata
# ----------------------------
ETF_ID = "ETF02"
ISIN = "IE00B4K48X80"
TICKER = "SMEA"
CURRENCY = "GBP"

# Suspected Yahoo break date:
# Yahoo valid through 2023-11-24
# Investing used from 2023-11-27 onward
BREAK_DATE = pd.Timestamp("2023-11-27")

# ----------------------------
# Helper: parse Investing volume
# ----------------------------
def parse_volume(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper().replace(",", "")
    if s in {"", "-", "N/A", "NAN"}:
        return np.nan
    multiplier = 1
    if s.endswith("K"):
        multiplier = 1_000
        s = s[:-1]
    elif s.endswith("M"):
        multiplier = 1_000_000
        s = s[:-1]
    elif s.endswith("B"):
        multiplier = 1_000_000_000
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return np.nan

# ----------------------------
# Load Yahoo raw
# ----------------------------
y = pd.read_csv(YAHOO_FILE)

# Parse dates
y["date"] = pd.to_datetime(y["date"], dayfirst=True, errors="coerce")

# Force numeric columns
for col in ["open", "high", "low", "close", "adjusted_close", "volume"]:
    y[col] = pd.to_numeric(y[col], errors="coerce")

# Keep only Yahoo rows BEFORE the break
y = y[y["date"] < BREAK_DATE].copy()

# ----------------------------
# Load Investing raw
# ----------------------------
inv = pd.read_csv(INVESTING_FILE)

# Parse dates
inv["Date"] = pd.to_datetime(inv["Date"], errors="coerce")

# Clean numeric text columns
for col in ["Price", "Open", "High", "Low"]:
    inv[col] = (
        inv[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    inv[col] = pd.to_numeric(inv[col], errors="coerce")

# Parse volume
inv["Vol."] = inv["Vol."].apply(parse_volume)

# Keep only Investing rows FROM the break onward
inv = inv[inv["Date"] >= BREAK_DATE].copy()

# Convert pence/GBX-like scale to GBP
inv["open"] = inv["Open"] / 100.0
inv["high"] = inv["High"] / 100.0
inv["low"] = inv["Low"] / 100.0
inv["close"] = inv["Price"] / 100.0

# No separate adjusted close in Investing file
inv["adjusted_close"] = inv["close"]

inv["volume"] = inv["Vol."]

# Add metadata
inv["date"] = inv["Date"]
inv["etf_id"] = ETF_ID
inv["isin"] = ISIN
inv["ticker"] = TICKER
inv["currency"] = CURRENCY
inv["source"] = "Investing.com (SMEA London GBP) used from 2023-11-27 onward"
inv["download_date"] = date.today().isoformat()

inv = inv[
    [
        "date",
        "etf_id",
        "isin",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "currency",
        "source",
        "download_date",
    ]
].copy()

# ----------------------------
# Standardize Yahoo metadata
# ----------------------------
# Keep same schema/order
y = y[
    [
        "date",
        "etf_id",
        "isin",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "currency",
        "source",
        "download_date",
    ]
].copy()

# Optional: refresh source note for traceability
y["source"] = "Yahoo Finance via yfinance (SMEA.L) used through 2023-11-24"

# ----------------------------
# Combine
# ----------------------------
clean = pd.concat([y, inv], ignore_index=True)

# Drop duplicates by date, keeping last if any overlap
clean = clean.sort_values("date").drop_duplicates(subset=["date"], keep="last").copy()

# Format dates as ISO
clean["date"] = clean["date"].dt.strftime("%Y-%m-%d")

# Normalize download_date to ISO where possible
# If Yahoo file has dd-mm-yy, convert it; otherwise keep as-is
def normalize_download_date(x):
    try:
        return pd.to_datetime(x, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return str(x)

clean["download_date"] = clean["download_date"].apply(normalize_download_date)

# Final sort
clean = clean.sort_values("date").reset_index(drop=True)

# Save
clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# ----------------------------
# Validation prints
# ----------------------------
print(f"Saved {OUTPUT_FILE}")
print(clean.head(3))
print(clean.tail(3))
print("Rows:", len(clean))
print("First date:", clean['date'].iloc[0])
print("Last date:", clean['date'].iloc[-1])
print("Duplicate dates:", clean['date'].duplicated().sum())
print("Blank dates:", clean['date'].isna().sum())