"""
Rolling-window portfolio backtester.

Design
------
At each rebalance step the engine:
  1. Slices the in-sample window  [i-W : i)        (W rows, no future data)
  2. Calls the strategy function to produce weights
  3. Computes daily portfolio returns over the out-of-sample period  [i : i+F)
  4. Records the rebalance date, weights, and out-of-sample returns

The in-sample and out-of-sample slices are strictly disjoint by construction,
eliminating look-ahead bias.

Public API
----------
BacktestConfig       – configuration dataclass
BacktestResult       – result dataclass (weights + returns DataFrames)
run_backtest()       – run one strategy through the engine
run_all_strategies() – run EW, MV, and CVaR-95 in one call and save results
save_results()       – persist BacktestResult DataFrames to CSV
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.strategies import cvar_minimise, equal_weight, minimum_variance

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = _PROJECT_ROOT / "data" / "results"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Parameters that govern a rolling backtest run.

    Attributes
    ----------
    estimation_window : int
        Number of trading days used to estimate strategy weights (in-sample).
        Default: 252 (≈ 1 calendar year of daily data).
    rebalance_freq : int
        Number of trading days between consecutive rebalances (out-of-sample
        holding period length).  Default: 21 (≈ 1 calendar month).
    cvar_alpha : float
        Confidence level for CVaR-minimisation strategy.  Must be in (0, 1).
        Default: 0.95.
    results_dir : Path
        Directory where CSV outputs are written by ``save_results``.
        Default: ``data/results/`` relative to the project root.
    """

    estimation_window: int = 252
    rebalance_freq: int = 21
    cvar_alpha: float = 0.95
    results_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)

    def __post_init__(self) -> None:
        if self.estimation_window < 2:
            raise ValueError(
                f"estimation_window must be ≥ 2, got {self.estimation_window}."
            )
        if self.rebalance_freq < 1:
            raise ValueError(
                f"rebalance_freq must be ≥ 1, got {self.rebalance_freq}."
            )
        if not (0.0 < self.cvar_alpha < 1.0):
            raise ValueError(
                f"cvar_alpha must be in (0, 1), got {self.cvar_alpha}."
            )
        self.results_dir = Path(self.results_dir)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Output of a single strategy backtest run.

    Attributes
    ----------
    strategy_name : str
        Human-readable strategy identifier (used in file names).
    weights : pd.DataFrame
        One row per rebalance event.  Columns: ``date`` + one per asset.
        ``date`` is the first out-of-sample day the weights are active.
    portfolio_returns : pd.DataFrame
        One row per out-of-sample trading day.
        Columns: ``date``, ``portfolio_return``.
    config : BacktestConfig
        Configuration used to produce this result.
    """

    strategy_name: str
    weights: pd.DataFrame
    portfolio_returns: pd.DataFrame
    config: BacktestConfig


# ---------------------------------------------------------------------------
# Core rolling engine
# ---------------------------------------------------------------------------

def _iter_windows(
    n_rows: int,
    estimation_window: int,
    rebalance_freq: int,
) -> list[tuple[int, int, int, int]]:
    """Return a list of (in_start, in_end, oos_start, oos_end) index tuples.

    ``in_end == oos_start`` — the boundary is exclusive on the left and
    inclusive on the right, matching pandas ``iloc`` slice semantics.

    Parameters
    ----------
    n_rows           : total number of rows in the return series
    estimation_window: W — in-sample window size
    rebalance_freq   : F — out-of-sample holding period length

    Returns
    -------
    List of (in_start, in_end, oos_start, oos_end) tuples, one per rebalance.
    The last OOS window is truncated to ``n_rows`` if the data runs out.
    """
    windows = []
    i = estimation_window
    while i < n_rows:
        in_start  = i - estimation_window
        in_end    = i                           # exclusive → last in-sample row = i-1
        oos_start = i
        oos_end   = min(i + rebalance_freq, n_rows)
        windows.append((in_start, in_end, oos_start, oos_end))
        i += rebalance_freq
    return windows


def run_backtest(
    returns: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], pd.Series],
    config: BacktestConfig,
    strategy_name: str = "strategy",
) -> BacktestResult:
    """Run a single strategy through the rolling-window backtest engine.

    Parameters
    ----------
    returns : pd.DataFrame
        Full return history.  Must contain a ``date`` column (string or
        datetime) plus one numeric column per asset.  Rows must be sorted
        ascending by date.
    strategy_fn : Callable[[pd.DataFrame], pd.Series]
        Function that accepts an in-sample return DataFrame (numeric columns
        only, no ``date`` column) and returns a ``pd.Series`` of weights
        indexed by asset name.
    config : BacktestConfig
        Backtest configuration.
    strategy_name : str
        Label attached to the returned ``BacktestResult``.

    Returns
    -------
    BacktestResult
        Contains a weights DataFrame (one row per rebalance) and a
        portfolio_returns DataFrame (one row per out-of-sample trading day).

    Raises
    ------
    ValueError
        If ``returns`` has fewer rows than ``config.estimation_window + 1``,
        or if any strategy call returns weights that violate constraints.
    """
    # ------------------------------------------------------------------ #
    # 1. Separate dates from numeric return matrix                         #
    # ------------------------------------------------------------------ #
    if "date" not in returns.columns:
        raise ValueError("returns DataFrame must contain a 'date' column.")

    dates = pd.to_datetime(returns["date"]).reset_index(drop=True)
    R: pd.DataFrame = (
        returns.drop(columns="date")
        .reset_index(drop=True)
        .astype(float)
    )
    assets = list(R.columns)
    n = len(R)

    W = config.estimation_window
    F = config.rebalance_freq

    if n <= W:
        raise ValueError(
            f"returns has {n} rows but estimation_window={W}. "
            "Need at least estimation_window + 1 rows."
        )

    # ------------------------------------------------------------------ #
    # 2. Roll through rebalance windows                                    #
    # ------------------------------------------------------------------ #
    weight_rows: list[dict] = []
    return_rows: list[dict] = []

    for in_start, in_end, oos_start, oos_end in _iter_windows(n, W, F):

        # In-sample slice (strictly before oos_start)
        in_sample: pd.DataFrame = R.iloc[in_start:in_end]

        # Compute weights — strategy sees ONLY in-sample data
        weights: pd.Series = strategy_fn(in_sample)

        # Validate constraints (long-only, fully invested)
        _validate_weights(weights, strategy_name, in_end)

        # Out-of-sample returns matrix
        oos_R: np.ndarray = R.iloc[oos_start:oos_end].values  # (oos_len, n_assets)
        oos_dates          = dates.iloc[oos_start:oos_end]
        w_arr: np.ndarray  = weights.reindex(assets).values   # ensure asset order

        # Portfolio return each OOS day = dot(weights, asset_returns)
        port_ret: np.ndarray = oos_R @ w_arr                  # shape (oos_len,)

        # Record weights keyed to the first OOS date (= first day they're live)
        rebal_date = dates.iloc[oos_start]
        weight_rows.append({"date": rebal_date, **dict(zip(assets, w_arr))})

        # Record per-day OOS portfolio returns
        for date, ret in zip(oos_dates, port_ret):
            return_rows.append({"date": date, "portfolio_return": float(ret)})

    # ------------------------------------------------------------------ #
    # 3. Assemble result DataFrames                                        #
    # ------------------------------------------------------------------ #
    weights_df = pd.DataFrame(weight_rows)
    weights_df["date"] = weights_df["date"].dt.strftime("%Y-%m-%d")

    returns_df = pd.DataFrame(return_rows)
    returns_df["date"] = returns_df["date"].dt.strftime("%Y-%m-%d")

    return BacktestResult(
        strategy_name=strategy_name,
        weights=weights_df,
        portfolio_returns=returns_df,
        config=config,
    )


def _validate_weights(
    weights: pd.Series,
    strategy_name: str,
    rebal_index: int,
) -> None:
    """Raise ValueError if weights are not long-only and fully-invested."""
    tol = 1e-6
    if (weights < -tol).any():
        bad = weights[weights < -tol].to_dict()
        raise ValueError(
            f"[{strategy_name}] Negative weights at rebalance index {rebal_index}: {bad}"
        )
    if abs(weights.sum() - 1.0) > tol:
        raise ValueError(
            f"[{strategy_name}] Weights sum to {weights.sum():.8f} "
            f"at rebalance index {rebal_index} — expected 1.0."
        )


# ---------------------------------------------------------------------------
# Run all three strategies
# ---------------------------------------------------------------------------

def run_all_strategies(
    returns: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict[str, BacktestResult]:
    """Run Equal-Weight, Minimum-Variance, and CVaR-95 through the backtester.

    Parameters
    ----------
    returns : pd.DataFrame
        Full return history with a ``date`` column plus one column per asset.
    config : BacktestConfig, optional
        Defaults to ``BacktestConfig()`` if not supplied.

    Returns
    -------
    dict mapping strategy name → BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    alpha = config.cvar_alpha
    alpha_tag = f"cvar_{int(alpha * 100)}"

    strategies: dict[str, Callable] = {
        "equal_weight": lambda R: equal_weight(R.columns),
        "min_variance": minimum_variance,
        alpha_tag:      lambda R: cvar_minimise(R, alpha=alpha),
    }

    results: dict[str, BacktestResult] = {}
    for name, fn in strategies.items():
        print(f"Running {name} …", end=" ", flush=True)
        results[name] = run_backtest(returns, fn, config, strategy_name=name)
        n_rebal = len(results[name].weights)
        n_oos   = len(results[name].portfolio_returns)
        print(f"{n_rebal} rebalances, {n_oos} OOS days")

    return results


# ---------------------------------------------------------------------------
# Persist to CSV
# ---------------------------------------------------------------------------

def save_results(
    results: dict[str, BacktestResult],
    output_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """Save weights and portfolio-return CSVs for every strategy result.

    Parameters
    ----------
    results    : dict returned by ``run_all_strategies`` (or manual calls)
    output_dir : directory to write into; defaults to each result's
                 ``config.results_dir``

    Returns
    -------
    dict mapping strategy_name → {"returns": Path, "weights": Path}
    """
    saved: dict[str, dict[str, Path]] = {}

    for name, result in results.items():
        out_dir = Path(output_dir) if output_dir else result.config.results_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        slug = name.lower().replace(" ", "_").replace("-", "_")
        ret_path = out_dir / f"backtest_returns_{slug}.csv"
        wgt_path = out_dir / f"backtest_weights_{slug}.csv"

        result.portfolio_returns.to_csv(ret_path, index=False)
        result.weights.to_csv(wgt_path, index=False)

        saved[name] = {"returns": ret_path, "weights": wgt_path}

    return saved


# ---------------------------------------------------------------------------
# Validation summary (printed after a full run)
# ---------------------------------------------------------------------------

def print_summary(
    results: dict[str, BacktestResult],
    saved_paths: dict[str, dict[str, Path]],
) -> None:
    """Print a human-readable summary of a completed backtest run."""
    print()
    print("=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        cfg = result.config
        w   = result.weights
        r   = result.portfolio_returns

        first_rebal = w["date"].iloc[0]
        first_oos   = r["date"].iloc[0]
        n_rebal     = len(w)
        n_oos       = len(r)

        print(f"\nStrategy : {name}")
        print(f"  Config  : window={cfg.estimation_window}d, "
              f"freq={cfg.rebalance_freq}d, alpha={cfg.cvar_alpha}")
        print(f"  First rebalance date  : {first_rebal}")
        print(f"  First OOS return date : {first_oos}")
        print(f"  Rebalances            : {n_rebal}")
        print(f"  OOS return rows       : {n_oos}")
        if name in saved_paths:
            print(f"  Returns -> {saved_paths[name]['returns'].relative_to(_PROJECT_ROOT)}")
            print(f"  Weights -> {saved_paths[name]['weights'].relative_to(_PROJECT_ROOT)}")

    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _returns = pd.read_csv(
        _PROJECT_ROOT / "data" / "processed" / "etf_returns_log.csv"
    )
    _config  = BacktestConfig()
    _results = run_all_strategies(_returns, _config)
    _paths   = save_results(_results)
    print_summary(_results, _paths)
