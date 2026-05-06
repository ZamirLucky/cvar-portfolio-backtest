"""
Unit tests for src/backtester.py

Focuses on:
  - rolling-window split correctness
  - no look-ahead bias
  - output shape, types, and constraint satisfaction
  - all three strategies passing through the same engine
  - edge cases (partial last window, minimum viable data)

All tests use small synthetic DataFrames — no file I/O, no network.

Run from project root:
    pytest tests/test_backtester.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtester import (
    BacktestConfig,
    BacktestResult,
    _iter_windows,
    run_all_strategies,
    run_backtest,
    save_results,
)
from src.strategies import cvar_minimise, equal_weight, minimum_variance

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
TOL_SUM = 1e-6   # weights must sum to 1 within this margin
TOL_NEG = 0.0    # after clipping in to_weights_series, min weight is exactly ≥ 0

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
RNG_SEED = 7


def _make_returns(n_rows: int, n_assets: int, seed: int = RNG_SEED) -> pd.DataFrame:
    """Synthetic returns DataFrame with a 'date' column and n_assets ETF columns."""
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_rows, freq="B")   # business days
    data  = rng.normal(0.0, 0.01, (n_rows, n_assets))
    cols  = [f"A{i+1}" for i in range(n_assets)]
    df    = pd.DataFrame(data, columns=cols)
    df.insert(0, "date", dates.strftime("%Y-%m-%d"))
    return df


@pytest.fixture
def small_returns() -> pd.DataFrame:
    """50 rows × 3 assets — enough for multiple rebalances with W=20, F=5."""
    return _make_returns(50, 3)


@pytest.fixture
def medium_returns() -> pd.DataFrame:
    """120 rows × 4 assets — used for multi-strategy runs."""
    return _make_returns(120, 4)


@pytest.fixture
def cfg_small() -> BacktestConfig:
    """Tiny config: W=20, F=5 — produces 6 rebalances on a 50-row dataset."""
    return BacktestConfig(estimation_window=20, rebalance_freq=5, cvar_alpha=0.95)


@pytest.fixture
def cfg_medium() -> BacktestConfig:
    """Config: W=40, F=10 — 8 rebalances on 120-row dataset."""
    return BacktestConfig(estimation_window=40, rebalance_freq=10, cvar_alpha=0.95)


# ---------------------------------------------------------------------------
# TestBacktestConfig
# ---------------------------------------------------------------------------

class TestBacktestConfig:

    def test_default_values(self):
        cfg = BacktestConfig()
        assert cfg.estimation_window == 252
        assert cfg.rebalance_freq    == 21
        assert cfg.cvar_alpha        == 0.95

    def test_custom_values_accepted(self):
        cfg = BacktestConfig(estimation_window=60, rebalance_freq=10, cvar_alpha=0.99)
        assert cfg.estimation_window == 60
        assert cfg.rebalance_freq    == 10
        assert cfg.cvar_alpha        == 0.99

    def test_raises_on_window_too_small(self):
        with pytest.raises(ValueError, match="estimation_window"):
            BacktestConfig(estimation_window=1)

    def test_raises_on_zero_freq(self):
        with pytest.raises(ValueError, match="rebalance_freq"):
            BacktestConfig(rebalance_freq=0)

    def test_raises_on_invalid_alpha(self):
        with pytest.raises(ValueError, match="cvar_alpha"):
            BacktestConfig(cvar_alpha=1.0)


# ---------------------------------------------------------------------------
# TestIterWindows  (internal helper — tests rolling-split logic directly)
# ---------------------------------------------------------------------------

class TestIterWindows:

    def test_basic_counts(self):
        """(50 - 20) / 5 = 6 rebalance windows."""
        wins = _iter_windows(50, 20, 5)
        assert len(wins) == 6

    def test_first_window_starts_at_zero(self):
        wins = _iter_windows(50, 20, 5)
        in_start, in_end, oos_start, oos_end = wins[0]
        assert in_start  == 0
        assert in_end    == 20
        assert oos_start == 20
        assert oos_end   == 25

    def test_in_sample_size_is_always_W(self):
        wins = _iter_windows(50, 20, 5)
        for in_start, in_end, _, _ in wins:
            assert in_end - in_start == 20

    def test_no_overlap_between_in_sample_and_oos(self):
        """in_end == oos_start means slices [in_start:in_end) and [oos_start:oos_end)
        are strictly disjoint."""
        for in_start, in_end, oos_start, oos_end in _iter_windows(50, 20, 5):
            assert in_end <= oos_start   # no look-ahead

    def test_oos_windows_are_contiguous(self):
        """Each OOS window starts exactly where the previous one ended."""
        wins = _iter_windows(50, 20, 5)
        for (_, _, _, oos_end_prev), (_, _, oos_start_next, _) in zip(wins, wins[1:]):
            assert oos_end_prev == oos_start_next

    def test_partial_last_window_is_included(self):
        """When n is not a multiple of W+k*F, the tail is still returned."""
        # 53 rows, W=20, F=5 → 6 full + 1 partial (3-day) window = 7 total
        wins = _iter_windows(53, 20, 5)
        assert len(wins) == 7
        last = wins[-1]
        _, _, oos_start, oos_end = last
        assert oos_end - oos_start == 3   # partial: 53 - 50 = 3

    def test_no_windows_when_not_enough_data(self):
        """If n <= W, no windows are produced."""
        assert _iter_windows(20, 20, 5) == []
        assert _iter_windows(19, 20, 5) == []

    def test_minimum_viable_dataset(self):
        """n = W + 1 → exactly one window with a single OOS day."""
        wins = _iter_windows(21, 20, 5)
        assert len(wins) == 1
        in_start, in_end, oos_start, oos_end = wins[0]
        assert in_end    == 20
        assert oos_start == 20
        assert oos_end   == 21


# ---------------------------------------------------------------------------
# TestRunBacktest — output structure
# ---------------------------------------------------------------------------

class TestRunBacktest:

    def test_returns_backtest_result(self, small_returns, cfg_small):
        strategy = lambda R: equal_weight(R.columns)
        result = run_backtest(small_returns, strategy, cfg_small, "ew")
        assert isinstance(result, BacktestResult)

    def test_strategy_name_stored(self, small_returns, cfg_small):
        strategy = lambda R: equal_weight(R.columns)
        result = run_backtest(small_returns, strategy, cfg_small, "my_strategy")
        assert result.strategy_name == "my_strategy"

    def test_config_stored(self, small_returns, cfg_small):
        strategy = lambda R: equal_weight(R.columns)
        result = run_backtest(small_returns, strategy, cfg_small, "ew")
        assert result.config is cfg_small

    def test_weights_is_dataframe(self, small_returns, cfg_small):
        result = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        assert isinstance(result.weights, pd.DataFrame)

    def test_portfolio_returns_is_dataframe(self, small_returns, cfg_small):
        result = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        assert isinstance(result.portfolio_returns, pd.DataFrame)

    def test_weights_has_date_column(self, small_returns, cfg_small):
        result = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        assert "date" in result.weights.columns

    def test_portfolio_returns_has_date_and_return_columns(self, small_returns, cfg_small):
        result = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        assert "date"             in result.portfolio_returns.columns
        assert "portfolio_return" in result.portfolio_returns.columns

    def test_weights_columns_match_assets(self, small_returns, cfg_small):
        assets = [c for c in small_returns.columns if c != "date"]
        result = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        for a in assets:
            assert a in result.weights.columns

    # --- Row counts ---

    def test_correct_number_of_rebalances(self, small_returns, cfg_small):
        """50 rows, W=20, F=5 → 6 rebalances."""
        result = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        assert len(result.weights) == 6

    def test_oos_rows_cover_full_period(self, small_returns, cfg_small):
        """OOS rows = n - W = 50 - 20 = 30."""
        result = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        assert len(result.portfolio_returns) == 30

    def test_oos_non_empty(self, small_returns, cfg_small):
        result = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        assert len(result.portfolio_returns) > 0

    # --- Error cases ---

    def test_raises_when_data_too_short(self, cfg_small):
        """Exactly W rows is not enough — need at least W+1."""
        tiny = _make_returns(20, 3)   # W = 20 → too short
        with pytest.raises(ValueError, match="estimation_window"):
            run_backtest(tiny, lambda R: equal_weight(R.columns), cfg_small)

    def test_raises_when_no_date_column(self, cfg_small):
        no_date = pd.DataFrame({"A": [0.01, -0.01], "B": [0.02, -0.02]})
        with pytest.raises(ValueError, match="date"):
            run_backtest(no_date, lambda R: equal_weight(R.columns), cfg_small)


# ---------------------------------------------------------------------------
# TestNoLookAheadBias
# ---------------------------------------------------------------------------

class TestNoLookAheadBias:

    def test_first_oos_date_is_after_estimation_window(self, small_returns, cfg_small):
        """The first portfolio-return date must correspond to row W of the input."""
        result  = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        W       = cfg_small.estimation_window
        # Date at row W in the input
        expected_first_oos = pd.to_datetime(small_returns["date"].iloc[W]).strftime("%Y-%m-%d")
        actual_first_oos   = result.portfolio_returns["date"].iloc[0]
        assert actual_first_oos == expected_first_oos

    def test_first_rebalance_date_equals_first_oos_date(self, small_returns, cfg_small):
        """The first rebalance date (= first day weights are live) matches the
        first out-of-sample return date."""
        result = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        assert result.weights["date"].iloc[0] == result.portfolio_returns["date"].iloc[0]

    def test_weights_match_strategy_on_exact_in_sample_slice(self, small_returns, cfg_small):
        """Weights at the first rebalance must equal the strategy output when
        called manually on returns.iloc[:W] — the exact in-sample slice."""
        W      = cfg_small.estimation_window
        assets = [c for c in small_returns.columns if c != "date"]
        R_only = small_returns[assets].astype(float)

        # Direct strategy call on the exact in-sample window
        expected_w = equal_weight(assets)

        result   = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        actual_w = result.weights.iloc[0][assets].values.astype(float)

        np.testing.assert_allclose(actual_w, expected_w.values, atol=1e-10)

    def test_mv_weights_match_direct_call_on_in_sample(self, small_returns, cfg_small):
        """MV weights at the first rebalance must equal minimum_variance called
        on returns.iloc[:W] directly (the in-sample window only)."""
        W      = cfg_small.estimation_window
        assets = [c for c in small_returns.columns if c != "date"]
        in_sample = small_returns[assets].iloc[:W].astype(float)

        expected_w = minimum_variance(in_sample)
        result     = run_backtest(small_returns, minimum_variance, cfg_small)
        actual_w   = result.weights.iloc[0][assets].values.astype(float)

        np.testing.assert_allclose(actual_w, expected_w.values, atol=1e-8)

    def test_oos_dates_are_strictly_after_in_sample(self, small_returns, cfg_small):
        """Verify each OOS holding period starts at the correct row index,
        i.e., there is no gap or overlap with the preceding in-sample slice."""
        W   = cfg_small.estimation_window
        F   = cfg_small.rebalance_freq
        n   = len(small_returns)
        dates_all = pd.to_datetime(small_returns["date"]).tolist()

        result       = run_backtest(small_returns, lambda R: equal_weight(R.columns), cfg_small)
        oos_dates    = pd.to_datetime(result.portfolio_returns["date"]).tolist()
        rebal_dates  = pd.to_datetime(result.weights["date"]).tolist()

        # Each rebalance date must equal dates_all[W + k*F] for k=0,1,…
        for k, rebal_date in enumerate(rebal_dates):
            expected = dates_all[W + k * F]
            assert rebal_date == expected, (
                f"Rebalance {k}: expected {expected.date()}, got {rebal_date.date()}"
            )


# ---------------------------------------------------------------------------
# TestWeightConstraints
# ---------------------------------------------------------------------------

class TestWeightConstraints:

    @pytest.mark.parametrize("strategy_fn,name", [
        (lambda R: equal_weight(R.columns), "equal_weight"),
        (minimum_variance,                  "min_variance"),
        (lambda R: cvar_minimise(R, 0.95),  "cvar_95"),
    ])
    def test_weights_sum_to_one(self, small_returns, cfg_small, strategy_fn, name):
        result = run_backtest(small_returns, strategy_fn, cfg_small, name)
        asset_cols = [c for c in result.weights.columns if c != "date"]
        row_sums   = result.weights[asset_cols].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < TOL_SUM, (
            f"{name}: max weight-sum deviation = {(row_sums - 1.0).abs().max()}"
        )

    @pytest.mark.parametrize("strategy_fn,name", [
        (lambda R: equal_weight(R.columns), "equal_weight"),
        (minimum_variance,                  "min_variance"),
        (lambda R: cvar_minimise(R, 0.95),  "cvar_95"),
    ])
    def test_weights_non_negative(self, small_returns, cfg_small, strategy_fn, name):
        result     = run_backtest(small_returns, strategy_fn, cfg_small, name)
        asset_cols = [c for c in result.weights.columns if c != "date"]
        min_weight = result.weights[asset_cols].values.min()
        assert min_weight >= TOL_NEG, (
            f"{name}: found negative weight {min_weight}"
        )

    @pytest.mark.parametrize("strategy_fn,name", [
        (lambda R: equal_weight(R.columns), "equal_weight"),
        (minimum_variance,                  "min_variance"),
        (lambda R: cvar_minimise(R, 0.95),  "cvar_95"),
    ])
    def test_no_nan_in_weights(self, small_returns, cfg_small, strategy_fn, name):
        result     = run_backtest(small_returns, strategy_fn, cfg_small, name)
        asset_cols = [c for c in result.weights.columns if c != "date"]
        assert not result.weights[asset_cols].isna().any().any(), (
            f"{name}: NaN values found in weight DataFrame"
        )

    @pytest.mark.parametrize("strategy_fn,name", [
        (lambda R: equal_weight(R.columns), "equal_weight"),
        (minimum_variance,                  "min_variance"),
        (lambda R: cvar_minimise(R, 0.95),  "cvar_95"),
    ])
    def test_no_nan_in_portfolio_returns(self, small_returns, cfg_small, strategy_fn, name):
        result = run_backtest(small_returns, strategy_fn, cfg_small, name)
        assert not result.portfolio_returns["portfolio_return"].isna().any(), (
            f"{name}: NaN values in portfolio_return column"
        )


# ---------------------------------------------------------------------------
# TestAllThreeStrategies
# ---------------------------------------------------------------------------

class TestAllThreeStrategies:

    def test_run_all_strategies_returns_dict(self, medium_returns, cfg_medium):
        results = run_all_strategies(medium_returns, cfg_medium)
        assert isinstance(results, dict)

    def test_run_all_strategies_has_three_keys(self, medium_returns, cfg_medium):
        results = run_all_strategies(medium_returns, cfg_medium)
        assert len(results) == 3

    def test_expected_strategy_names_present(self, medium_returns, cfg_medium):
        results = run_all_strategies(medium_returns, cfg_medium)
        assert "equal_weight" in results
        assert "min_variance" in results
        assert "cvar_95"      in results

    def test_alpha_tag_matches_config(self, medium_returns):
        cfg = BacktestConfig(estimation_window=40, rebalance_freq=10, cvar_alpha=0.99)
        results = run_all_strategies(medium_returns, cfg)
        assert "cvar_99" in results

    def test_all_results_are_backtest_result(self, medium_returns, cfg_medium):
        for name, result in run_all_strategies(medium_returns, cfg_medium).items():
            assert isinstance(result, BacktestResult), f"{name} is not a BacktestResult"

    def test_all_strategies_produce_same_oos_dates(self, medium_returns, cfg_medium):
        """All three strategies must cover the exact same out-of-sample dates."""
        results   = run_all_strategies(medium_returns, cfg_medium)
        date_sets = [
            set(r.portfolio_returns["date"]) for r in results.values()
        ]
        assert date_sets[0] == date_sets[1] == date_sets[2]

    def test_all_strategies_produce_same_rebalance_dates(self, medium_returns, cfg_medium):
        results   = run_all_strategies(medium_returns, cfg_medium)
        date_sets = [set(r.weights["date"]) for r in results.values()]
        assert date_sets[0] == date_sets[1] == date_sets[2]

    def test_equal_weight_is_constant_across_rebalances(self, medium_returns, cfg_medium):
        """EW weights must be identical at every rebalance (1/N by definition)."""
        results    = run_all_strategies(medium_returns, cfg_medium)
        ew         = results["equal_weight"]
        asset_cols = [c for c in ew.weights.columns if c != "date"]
        n          = len(asset_cols)
        for _, row in ew.weights[asset_cols].iterrows():
            np.testing.assert_allclose(row.values, np.full(n, 1.0 / n), atol=1e-10)

    def test_portfolio_returns_are_finite(self, medium_returns, cfg_medium):
        results = run_all_strategies(medium_returns, cfg_medium)
        for name, r in results.items():
            vals = r.portfolio_returns["portfolio_return"].values
            assert np.isfinite(vals).all(), f"{name}: non-finite portfolio returns"


# ---------------------------------------------------------------------------
# TestSaveResults
# ---------------------------------------------------------------------------

class TestSaveResults:

    def test_save_creates_files(self, small_returns, cfg_small, tmp_path):
        results = run_all_strategies(small_returns, cfg_small)
        paths   = save_results(results, output_dir=tmp_path)

        for name, file_dict in paths.items():
            assert file_dict["returns"].exists(), f"Missing returns file for {name}"
            assert file_dict["weights"].exists(), f"Missing weights file for {name}"

    def test_saved_returns_file_is_readable(self, small_returns, cfg_small, tmp_path):
        results = run_all_strategies(small_returns, cfg_small)
        paths   = save_results(results, output_dir=tmp_path)

        for name, file_dict in paths.items():
            df = pd.read_csv(file_dict["returns"])
            assert "date"             in df.columns
            assert "portfolio_return" in df.columns
            assert len(df) > 0

    def test_saved_weights_file_is_readable(self, small_returns, cfg_small, tmp_path):
        results   = run_all_strategies(small_returns, cfg_small)
        paths     = save_results(results, output_dir=tmp_path)
        assets    = [c for c in small_returns.columns if c != "date"]

        for name, file_dict in paths.items():
            df = pd.read_csv(file_dict["weights"])
            assert "date" in df.columns
            for a in assets:
                assert a in df.columns, f"{name} weights missing column {a}"

    def test_file_names_contain_strategy_slug(self, small_returns, cfg_small, tmp_path):
        results = run_all_strategies(small_returns, cfg_small)
        paths   = save_results(results, output_dir=tmp_path)

        for name, file_dict in paths.items():
            slug = name.lower().replace(" ", "_")
            assert slug in file_dict["returns"].stem
            assert slug in file_dict["weights"].stem

    def test_returns_dict_maps_name_to_paths(self, small_returns, cfg_small, tmp_path):
        results = run_all_strategies(small_returns, cfg_small)
        paths   = save_results(results, output_dir=tmp_path)

        assert set(paths.keys()) == set(results.keys())
        for name, file_dict in paths.items():
            assert "returns" in file_dict
            assert "weights" in file_dict
