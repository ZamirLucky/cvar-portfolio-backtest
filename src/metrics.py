"""
Reusable statistics and metrics for the CVaR portfolio backtest.

Two groups of functions live here:

Exploratory statistics (Phase 1)
---------------------------------
price_summary_stats, return_summary_stats, correlation_matrix,
covariance_matrix, validate_corr_matrix, compute_and_save_stats

Portfolio performance metrics (Phase 4)
-----------------------------------------
cagr, annualised_volatility, sharpe_ratio, sortino_ratio,
historical_var, historical_cvar, max_drawdown,
performance_metrics, performance_summary, build_and_save_performance_summary

Sign / annualisation conventions (Phase 4 functions)
------------------------------------------------------
- All inputs are assumed to be **daily log returns**.
- Annualisation uses TRADING_DAYS = 252.
- VaR and CVaR are returned as **positive loss magnitudes**
  (a value of 0.012 means "lose up to 1.2 % on the worst days").
- Max drawdown is returned as a **negative fraction**
  (−0.35 means a 35 % peak-to-trough decline).
- Sharpe and Sortino use the Markowitz definition (excess return over
  risk-free rate divided by annualised volatility / downside deviation).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

TRADING_DAYS = 252

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = _PROJECT_ROOT / "data" / "results"


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


# ===========================================================================
# Phase 4 — Portfolio performance metrics
# ===========================================================================

# ---------------------------------------------------------------------------
# Input validation helper
# ---------------------------------------------------------------------------

def _validate_series(r: pd.Series | np.ndarray, min_obs: int = 2) -> np.ndarray:
    """Convert to a 1-D float array and check it is usable.

    Raises
    ------
    ValueError
        If the input is empty, all-NaN, or has fewer than *min_obs* finite values.
    """
    arr = np.asarray(r, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Return series is empty.")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Return series contains no finite values (all NaN/Inf).")
    if finite.size < min_obs:
        raise ValueError(
            f"Need at least {min_obs} finite observations, got {finite.size}."
        )
    return finite   # only finite values are used in all metrics


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def cagr(
    returns: pd.Series | np.ndarray,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Compound Annual Growth Rate from daily log returns.

    Uses the log-return identity:

        CAGR = exp(mean(r) × trading_days) − 1

    which equals ``(total_growth)^(1/n_years) − 1`` where total growth is
    ``exp(sum(r))`` and n_years is ``n / trading_days``.

    Parameters
    ----------
    returns      : 1-D array-like of daily log returns
    trading_days : annualisation factor (default 252)

    Returns
    -------
    float  — annualised growth rate (e.g. 0.08 = 8 %)
    """
    arr = _validate_series(returns)
    return float(np.exp(arr.mean() * trading_days) - 1.0)


def annualised_volatility(
    returns: pd.Series | np.ndarray,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised volatility (standard deviation) of daily log returns.

        vol = std(r) × √trading_days

    Parameters
    ----------
    returns      : 1-D array-like of daily log returns
    trading_days : annualisation factor (default 252)

    Returns
    -------
    float  — annualised volatility (e.g. 0.15 = 15 %)
    """
    arr = _validate_series(returns)
    return float(arr.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio.

        Sharpe = (mean(r) × T − rf) / (std(r) × √T)

    where T = trading_days.

    Parameters
    ----------
    returns        : 1-D array-like of daily log returns
    risk_free_rate : annualised risk-free rate (default 0.0)
    trading_days   : annualisation factor (default 252)

    Returns
    -------
    float  — Sharpe ratio, or ``np.nan`` if volatility is zero.
    """
    arr = _validate_series(returns)
    vol = arr.std(ddof=1) * np.sqrt(trading_days)
    if vol < 1e-14:
        return float("nan")
    excess_return = arr.mean() * trading_days - risk_free_rate
    return float(excess_return / vol)


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio.

    Uses the downside deviation as denominator:

        downside_dev = √(mean(min(r − rf_daily, 0)²)) × √T

    where rf_daily = risk_free_rate / trading_days and the mean is taken
    over *all* observations (not only the negative ones).  This is the
    standard Sortino definition (Sortino & van der Meer, 1991).

        Sortino = (mean(r) × T − rf) / downside_dev

    Parameters
    ----------
    returns        : 1-D array-like of daily log returns
    risk_free_rate : annualised risk-free rate (default 0.0)
    trading_days   : annualisation factor (default 252)

    Returns
    -------
    float  — Sortino ratio.
             ``np.inf``  if excess return > 0 and no downside returns exist.
             ``np.nan``  if excess return = 0 and no downside returns exist.
             ``-np.inf`` if excess return < 0 and no downside returns exist.
    """
    arr = _validate_series(returns)
    rf_daily = risk_free_rate / trading_days
    excess_daily = arr - rf_daily
    downside     = np.minimum(excess_daily, 0.0)

    downside_var = np.mean(downside ** 2)
    if downside_var == 0.0:
        # No downside returns: Sortino sign matches excess-return sign
        excess_ann = arr.mean() * trading_days - risk_free_rate
        if excess_ann > 0:
            return float("inf")
        if excess_ann < 0:
            return float("-inf")
        return float("nan")

    downside_dev  = np.sqrt(downside_var) * np.sqrt(trading_days)
    excess_return = arr.mean() * trading_days - risk_free_rate
    return float(excess_return / downside_dev)


def historical_var(
    returns: pd.Series | np.ndarray,
    alpha: float = 0.95,
) -> float:
    """Historical (empirical) Value-at-Risk.

    Defined as:

        VaR_α = −percentile(r, (1 − α) × 100)

    **Sign convention: positive value = expected maximum loss.**
    A value of 0.025 means "on at most 5 % of days you lose more than 2.5 %"
    (for α = 0.95).  Negative values indicate that even the worst (1−α) tail
    contains only gains.

    Parameters
    ----------
    returns : 1-D array-like of daily log returns
    alpha   : confidence level in (0, 1), e.g. 0.95 or 0.99

    Returns
    -------
    float  — positive loss magnitude at confidence level α
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    arr = _validate_series(returns)
    return float(-np.percentile(arr, (1.0 - alpha) * 100.0))


def historical_cvar(
    returns: pd.Series | np.ndarray,
    alpha: float = 0.95,
) -> float:
    """Historical (empirical) Conditional Value-at-Risk (Expected Shortfall).

    Defined as:

        CVaR_α = −mean(r  where  r ≤ percentile(r, (1 − α) × 100))

    **Sign convention: positive value = expected average loss in the tail.**
    CVaR_α ≥ VaR_α always.

    Parameters
    ----------
    returns : 1-D array-like of daily log returns
    alpha   : confidence level in (0, 1)

    Returns
    -------
    float  — positive average loss in the worst (1 − α) fraction of days
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    arr  = _validate_series(returns)
    cutoff = np.percentile(arr, (1.0 - alpha) * 100.0)
    tail   = arr[arr <= cutoff]
    if tail.size == 0:
        return float(-cutoff)   # degenerate: all returns identical
    return float(-tail.mean())


def max_drawdown(
    returns: pd.Series | np.ndarray,
) -> float:
    """Maximum peak-to-trough drawdown from daily log returns.

    Computes the running wealth index W_t = exp(Σ r_s, s≤t),
    the running maximum M_t = max(W_s, s≤t), and:

        drawdown_t = W_t / M_t − 1

    Returns the minimum (most negative) drawdown.

    **Sign convention: negative fraction** (e.g. −0.35 = peak fell 35 %).
    Returns 0.0 for a monotonically non-decreasing wealth path.

    Parameters
    ----------
    returns : 1-D array-like of daily log returns

    Returns
    -------
    float  — maximum drawdown as a negative fraction in [−1, 0]
    """
    arr = _validate_series(returns, min_obs=1)
    # Prepend an implicit W_0 = 1.0 (before any return is applied) so that
    # a run of pure losses from the start registers correctly as a drawdown
    # from the starting wealth, not from the first-day wealth.
    cum_log = np.concatenate([[0.0], np.cumsum(arr)])
    wealth  = np.exp(cum_log)
    peak    = np.maximum.accumulate(wealth)
    dd      = wealth / peak - 1.0
    return float(dd.min())


# ---------------------------------------------------------------------------
# Aggregate: all metrics for one series
# ---------------------------------------------------------------------------

def performance_metrics(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.0,
    alpha: float = 0.95,
    trading_days: int = TRADING_DAYS,
) -> dict[str, float]:
    """Compute all performance metrics for a single return series.

    Parameters
    ----------
    returns        : 1-D array-like of daily log returns
    risk_free_rate : annualised risk-free rate (default 0.0)
    alpha          : VaR / CVaR confidence level (default 0.95)
    trading_days   : annualisation factor (default 252)

    Returns
    -------
    dict with keys:
        CAGR, Ann_Volatility, Sharpe, Sortino,
        VaR_{int(α*100)}, CVaR_{int(α*100)}, Max_Drawdown
    """
    tag = int(alpha * 100)
    return {
        "CAGR"              : cagr(returns, trading_days),
        "Ann_Volatility"    : annualised_volatility(returns, trading_days),
        "Sharpe"            : sharpe_ratio(returns, risk_free_rate, trading_days),
        "Sortino"           : sortino_ratio(returns, risk_free_rate, trading_days),
        f"VaR_{tag}"        : historical_var(returns, alpha),
        f"CVaR_{tag}"       : historical_cvar(returns, alpha),
        "Max_Drawdown"      : max_drawdown(returns),
    }


# ---------------------------------------------------------------------------
# Summary table across strategies
# ---------------------------------------------------------------------------

def performance_summary(
    strategy_returns: dict[str, pd.Series | np.ndarray],
    risk_free_rate: float = 0.0,
    alpha: float = 0.95,
    trading_days: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Build a performance summary table across multiple strategies.

    Parameters
    ----------
    strategy_returns : dict mapping strategy name → 1-D return series
    risk_free_rate   : annualised risk-free rate (default 0.0)
    alpha            : VaR / CVaR confidence level (default 0.95)
    trading_days     : annualisation factor (default 252)

    Returns
    -------
    pd.DataFrame  — rows = strategies, columns = metric names
    """
    rows = {}
    for name, r in strategy_returns.items():
        rows[name] = performance_metrics(r, risk_free_rate, alpha, trading_days)

    df = pd.DataFrame(rows).T
    df.index.name = "strategy"
    return df


# ---------------------------------------------------------------------------
# Load backtest results and save performance_summary.csv
# ---------------------------------------------------------------------------

def build_and_save_performance_summary(
    results_dir: Path | None = None,
    output_path: Path | None = None,
    risk_free_rate: float = 0.0,
    alpha: float = 0.95,
    strategy_names: list[str] | None = None,
) -> pd.DataFrame:
    """Load backtest return CSVs, compute metrics, save performance_summary.csv.

    Parameters
    ----------
    results_dir    : directory containing ``backtest_returns_*.csv`` files
                     (defaults to ``data/results/``)
    output_path    : path for the output CSV
                     (defaults to ``results_dir / performance_summary.csv``)
    risk_free_rate : annualised risk-free rate passed to all metric functions
    alpha          : VaR / CVaR confidence level
    strategy_names : explicit list of strategy slugs to load.
                     If None, all ``backtest_returns_*.csv`` files are used.

    Returns
    -------
    pd.DataFrame  — the summary table (also saved to *output_path*)
    """
    rdir = Path(results_dir) if results_dir else RESULTS_DIR

    if strategy_names is not None:
        files = {
            name: rdir / f"backtest_returns_{name}.csv"
            for name in strategy_names
        }
    else:
        files = {
            p.stem.replace("backtest_returns_", ""): p
            for p in sorted(rdir.glob("backtest_returns_*.csv"))
        }

    if not files:
        raise FileNotFoundError(
            f"No backtest_returns_*.csv files found in {rdir}."
        )

    strategy_returns: dict[str, pd.Series] = {}
    for name, path in files.items():
        df = pd.read_csv(path)
        if "portfolio_return" not in df.columns:
            raise ValueError(
                f"{path.name}: expected a 'portfolio_return' column."
            )
        strategy_returns[name] = df["portfolio_return"].dropna()

    summary = performance_summary(strategy_returns, risk_free_rate, alpha)

    out = Path(output_path) if output_path else rdir / "performance_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out)

    try:
        display_path = out.relative_to(_PROJECT_ROOT)
    except ValueError:
        display_path = out
    print(f"Performance summary saved -> {display_path}")
    print(summary.round(4).to_string())

    return summary
