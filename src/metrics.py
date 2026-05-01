"""
Reusable statistics and metrics for the CVaR portfolio backtest.

Functions here operate on DataFrames that have a 'date' column plus
one column per ETF. They return tidy DataFrames suitable for display
or CSV export.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

TRADING_DAYS = 252

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def price_summary_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Standard descriptive statistics for a wide EUR price panel.

    Parameters
    ----------
    panel : DataFrame with 'date' column + ETF price columns

    Returns
    -------
    DataFrame indexed by statistic, columns = ETF IDs
    """
    etf_cols = [c for c in panel.columns if c != "date"]
    prices = panel[etf_cols]

    desc = prices.describe()  # count, mean, std, min, 25%, 50%, 75%, max
    return desc


def return_summary_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Extended descriptive statistics for a wide log-return panel.

    Includes: count, mean, std, min, max, skewness, excess kurtosis,
    annualised mean return, annualised volatility.

    Parameters
    ----------
    returns : DataFrame with 'date' column + ETF return columns

    Returns
    -------
    DataFrame indexed by statistic, columns = ETF IDs
    """
    etf_cols = [c for c in returns.columns if c != "date"]
    rets = returns[etf_cols]

    rows: dict[str, pd.Series] = {}
    rows["count"] = rets.count().astype(float)
    rows["mean"] = rets.mean()
    rows["std"] = rets.std()
    rows["min"] = rets.min()
    rows["max"] = rets.max()
    rows["skewness"] = rets.apply(sp_stats.skew)
    rows["kurtosis"] = rets.apply(sp_stats.kurtosis)  # excess kurtosis
    rows["ann_mean"] = rets.mean() * TRADING_DAYS
    rows["ann_volatility"] = rets.std() * np.sqrt(TRADING_DAYS)

    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Correlation and covariance
# ---------------------------------------------------------------------------

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of log returns."""
    etf_cols = [c for c in returns.columns if c != "date"]
    return returns[etf_cols].corr()


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance matrix of log returns."""
    etf_cols = [c for c in returns.columns if c != "date"]
    return returns[etf_cols].cov()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_corr_matrix(corr: pd.DataFrame) -> None:
    """Assert the correlation matrix is symmetric with unit diagonal."""
    assert not corr.isna().any().any(), "NaNs in correlation matrix"
    assert np.allclose(corr.values, corr.values.T, atol=1e-10), \
        "Correlation matrix is not symmetric"
    assert np.allclose(np.diag(corr.values), 1.0, atol=1e-10), \
        "Diagonal of correlation matrix is not 1"


# ---------------------------------------------------------------------------
# Save all stats artefacts
# ---------------------------------------------------------------------------

def compute_and_save_stats(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute all statistics and save to CSV.

    Returns a dict with keys:
        price_stats, return_stats, corr, cov
    """
    out = output_dir or PROCESSED_DIR
    out.mkdir(parents=True, exist_ok=True)

    price_stats = price_summary_stats(prices)
    ret_stats = return_summary_stats(returns)
    corr = correlation_matrix(returns)
    cov = covariance_matrix(returns)

    # Validate before saving
    assert not price_stats.isna().any().any(), "NaNs in price stats"
    assert not ret_stats.isna().any().any(), "NaNs in return stats"
    validate_corr_matrix(corr)

    price_stats.to_csv(out / "summary_stats_prices.csv")
    ret_stats.to_csv(out / "summary_stats_returns.csv")
    corr.to_csv(out / "correlation_matrix.csv")
    cov.to_csv(out / "covariance_matrix.csv")

    print("Saved:")
    for name in ("summary_stats_prices.csv", "summary_stats_returns.csv",
                 "correlation_matrix.csv", "covariance_matrix.csv"):
        print(f"  data/processed/{name}")

    return {"price_stats": price_stats, "return_stats": ret_stats,
            "corr": corr, "cov": cov}
