from datetime import date
import pandas as pd
import yfinance as yf

# ETF01 metadata
ETF_ID = "ETF01"
ISIN = "IE00B5BMR087"
TICKER_SOURCE = "CSPX.L"   # Yahoo symbol for the LSE line
TICKER_DATASET = "CSPX"    # ticker you want stored in your dataset
CURRENCY = "USD"

# Download raw daily data
df = yf.download(
    TICKER_SOURCE,
    start="2016-01-01",
    end="2026-01-01",
    auto_adjust=False,
    progress=False
)

# Flatten columns if Yahoo returns a MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]

# Keep and rename required columns
df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
df.columns = ["date", "open", "high", "low", "close", "adjusted_close", "volume"]

# Add metadata columns
df["etf_id"] = ETF_ID
df["isin"] = ISIN
df["ticker"] = TICKER_DATASET
df["currency"] = CURRENCY
df["source"] = f"Yahoo Finance via yfinance ({TICKER_SOURCE})"
df["download_date"] = date.today().isoformat()

# Reorder to your raw schema
df = df[
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
]

# Save
df.to_csv("ETF01_raw.csv", index=False, encoding="utf-8")

print("Saved ETF01_raw.csv")
print(df.head())
print(df.tail())
print(df.shape)