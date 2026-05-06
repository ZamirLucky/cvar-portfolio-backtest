"""
Unit tests for src/metrics.py — Phase 4 performance metric functions.

Every metric is tested against hand-checkable synthetic inputs so that
expected values can be verified independently without running the full
backtest pipeline.

Run from project root:
    pytest tests/test_metrics.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    _validate_series,
    annualised_volatility,
    build_and_save_performance_summary,
    cagr,
    historical_cvar,
    historical_var,
    max_drawdown,
    performance_metrics,
    performance_summary,
    sharpe_ratio,
    sortino_ratio,
)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
ATOL = 1e-9    # absolute tolerance for exact-formula results
RTOL = 1e-6    # relative tolerance where floating-point rounding is expected

TRADING_DAYS = 252


# ===========================================================================
# _validate_series
# ===========================================================================

class TestValidateSeries:

    def test_returns_finite_array(self):
        arr = _validate_series(pd.Series([0.01, -0.02, 0.03]))
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 3

    def test_strips_nan(self):
        arr = _validate_series(pd.Series([0.01, np.nan, 0.03]))
        assert len(arr) == 2   # NaN removed

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_series(pd.Series([], dtype=float))

    def test_raises_on_all_nan(self):
        with pytest.raises(ValueError, match="no finite values"):
            _validate_series(pd.Series([np.nan, np.nan]))

    def test_raises_below_min_obs(self):
        with pytest.raises(ValueError, match="at least 2 finite observations"):
            _validate_series(pd.Series([0.01]))

    def test_single_obs_ok_when_min_obs_1(self):
        arr = _validate_series(pd.Series([0.01]), min_obs=1)
        assert len(arr) == 1


# ===========================================================================
# CAGR
# ===========================================================================

class TestCAGR:

    def test_zero_returns(self):
        """All zero returns → CAGR = 0 %."""
        r = np.zeros(TRADING_DAYS)
        assert cagr(r) == pytest.approx(0.0, abs=ATOL)

    def test_flat_10_pct(self):
        """mean(r) = log(1.10)/252 over 252 days → CAGR = 10 %."""
        r = np.full(TRADING_DAYS, np.log(1.10) / TRADING_DAYS)
        assert cagr(r) == pytest.approx(0.10, rel=RTOL)

    def test_flat_negative(self):
        """mean(r) = log(0.90)/252 → CAGR ≈ −10 %."""
        r = np.full(TRADING_DAYS, np.log(0.90) / TRADING_DAYS)
        assert cagr(r) == pytest.approx(-0.10, rel=RTOL)

    def test_uses_mean_not_sum(self):
        """CAGR scales by observations: 504 obs of the same daily return
        still gives the same 1-year CAGR as 252 obs."""
        daily = np.log(1.10) / TRADING_DAYS
        assert cagr(np.full(TRADING_DAYS,       daily)) == pytest.approx(
               cagr(np.full(TRADING_DAYS * 2,   daily)), rel=RTOL)

    def test_returns_float(self):
        assert isinstance(cagr(np.zeros(50)), float)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            cagr(pd.Series([], dtype=float))


# ===========================================================================
# Annualised Volatility
# ===========================================================================

class TestAnnualisedVolatility:

    def test_constant_returns_zero_vol(self):
        """Constant return series → volatility = 0."""
        r = np.full(100, 0.001)
        assert annualised_volatility(r) == pytest.approx(0.0, abs=ATOL)

    def test_known_vol(self):
        """std(r) = σ → ann vol = σ × √252 (large n for precision)."""
        rng = np.random.default_rng(0)
        sigma = 0.01
        r = rng.normal(0.0, sigma, 50_000)
        expected = sigma * np.sqrt(TRADING_DAYS)
        assert annualised_volatility(r) == pytest.approx(expected, rel=0.02)

    def test_scales_with_sqrt_trading_days(self):
        """Halving trading_days divides vol by √2."""
        r = np.random.default_rng(1).normal(0, 0.01, 1000)
        v252 = annualised_volatility(r, trading_days=252)
        v126 = annualised_volatility(r, trading_days=126)
        assert v252 == pytest.approx(v126 * np.sqrt(2), rel=RTOL)

    def test_returns_float(self):
        assert isinstance(annualised_volatility(np.ones(50) * 0.001), float)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            annualised_volatility(pd.Series([], dtype=float))


# ===========================================================================
# Sharpe Ratio
# ===========================================================================

class TestSharpeRatio:

    def test_zero_volatility_returns_nan(self):
        """Constant returns → vol = 0 → Sharpe = NaN."""
        r = np.full(100, 0.001)
        assert np.isnan(sharpe_ratio(r))

    def test_known_sharpe(self):
        """With large n: Sharpe ≈ (μ×252) / (σ×√252) = μ×√252 / σ."""
        rng    = np.random.default_rng(2)
        mu, s  = 0.001, 0.01
        r      = rng.normal(mu, s, 100_000)
        expected = (mu * TRADING_DAYS) / (s * np.sqrt(TRADING_DAYS))
        assert sharpe_ratio(r) == pytest.approx(expected, rel=0.05)

    def test_positive_mean_positive_sharpe(self):
        rng = np.random.default_rng(3)
        r   = rng.normal(0.001, 0.01, 500)
        assert sharpe_ratio(r) > 0

    def test_negative_mean_negative_sharpe(self):
        rng = np.random.default_rng(4)
        r   = rng.normal(-0.001, 0.01, 500)
        assert sharpe_ratio(r) < 0

    def test_higher_rf_lowers_sharpe(self):
        rng = np.random.default_rng(5)
        r   = rng.normal(0.001, 0.01, 500)
        assert sharpe_ratio(r, risk_free_rate=0.0) > sharpe_ratio(r, risk_free_rate=0.05)

    def test_returns_float(self):
        r = np.random.default_rng(6).normal(0, 0.01, 100)
        assert isinstance(sharpe_ratio(r), float)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            sharpe_ratio(pd.Series([], dtype=float))


# ===========================================================================
# Sortino Ratio
# ===========================================================================

class TestSortinoRatio:

    def test_all_positive_returns_inf(self):
        """No downside returns → Sortino = +inf (positive excess return)."""
        r = np.full(100, 0.001)
        assert np.isinf(sortino_ratio(r)) and sortino_ratio(r) > 0

    def test_all_equal_zero_returns_nan(self):
        """Zero mean, no downside → Sortino = NaN."""
        r = np.zeros(100)
        assert np.isnan(sortino_ratio(r))

    def test_all_negative_returns_large_negative(self):
        """All negative returns → downside variance is non-zero → Sortino is
        a large but finite negative number, not −inf."""
        r      = np.full(100, -0.001)
        result = sortino_ratio(r)
        assert np.isfinite(result)
        assert result < 0

    def test_sortino_ge_sharpe_for_mixed_returns(self):
        """Sortino ≥ Sharpe when mean > 0 (downside_dev ≤ total_std)."""
        rng = np.random.default_rng(7)
        r   = rng.normal(0.001, 0.01, 1000)
        assert sortino_ratio(r) >= sharpe_ratio(r) - 1e-10

    def test_sortino_strictly_greater_than_sharpe_positive_mean(self):
        """For mixed positive/negative returns with positive mean,
        Sortino > Sharpe because downside_dev < total_std."""
        rng = np.random.default_rng(8)
        r   = rng.normal(0.001, 0.01, 10_000)
        assert sortino_ratio(r) > sharpe_ratio(r)

    def test_returns_float_on_valid_input(self):
        rng = np.random.default_rng(9)
        r   = rng.normal(0, 0.01, 200)
        result = sortino_ratio(r)
        assert isinstance(result, float) or np.isinf(result) or np.isnan(result)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            sortino_ratio(pd.Series([], dtype=float))


# ===========================================================================
# Historical VaR
# ===========================================================================

class TestHistoricalVaR:

    def test_positive_loss_for_series_with_losses(self):
        """Series with large losses → VaR_95 > 0."""
        r = np.array([-0.05] * 10 + [0.01] * 90)
        assert historical_var(r, alpha=0.95) > 0

    def test_var99_gte_var95(self):
        """Higher confidence → larger (or equal) VaR."""
        rng = np.random.default_rng(10)
        r   = rng.normal(0, 0.01, 1000)
        assert historical_var(r, alpha=0.99) >= historical_var(r, alpha=0.95)

    def test_var_monotone_in_alpha(self):
        """VaR_α increases with α."""
        rng   = np.random.default_rng(11)
        r     = rng.normal(-0.001, 0.015, 2000)
        alphas = [0.80, 0.90, 0.95, 0.99]
        vars_  = [historical_var(r, a) for a in alphas]
        assert all(vars_[i] <= vars_[i+1] for i in range(len(vars_)-1))

    def test_exact_value(self):
        """Exact VaR on a sorted array with a clean percentile."""
        # 100 evenly-spaced values; 5th percentile (1-0.95) is well-defined
        r      = np.linspace(-0.10, 0.10, 200)   # sorted, symmetric
        var_95 = historical_var(r, alpha=0.95)
        # 5th percentile of r: np.percentile(r, 5) ≈ -0.09
        expected = -np.percentile(r, 5.0)
        assert var_95 == pytest.approx(expected, rel=RTOL)

    def test_all_positive_returns_var_negative_or_zero(self):
        """No losses at any level → VaR ≤ 0 (tail is still a gain)."""
        r = np.full(100, 0.005)
        assert historical_var(r, alpha=0.95) <= 0.0

    def test_raises_on_bad_alpha(self):
        r = np.ones(50) * 0.01
        with pytest.raises(ValueError, match="alpha"):
            historical_var(r, alpha=1.0)
        with pytest.raises(ValueError, match="alpha"):
            historical_var(r, alpha=0.0)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            historical_var(pd.Series([], dtype=float))


# ===========================================================================
# Historical CVaR
# ===========================================================================

class TestHistoricalCVaR:

    def test_cvar_gte_var(self):
        """CVaR_α ≥ VaR_α always (average of the tail ≥ boundary of the tail)."""
        rng = np.random.default_rng(12)
        r   = rng.normal(-0.001, 0.015, 1000)
        for alpha in [0.90, 0.95, 0.99]:
            assert historical_cvar(r, alpha) >= historical_var(r, alpha) - 1e-10, (
                f"CVaR < VaR at alpha={alpha}"
            )

    def test_cvar99_gte_cvar95(self):
        rng = np.random.default_rng(13)
        r   = rng.normal(0, 0.01, 1000)
        assert historical_cvar(r, alpha=0.99) >= historical_cvar(r, alpha=0.95) - 1e-10

    def test_exact_value_small_array(self):
        """Hand-checkable: 5 values, alpha=0.80 → worst 20 % = worst 1 value."""
        # Use a large sorted array where the 20th percentile is exact
        # 10 values, np.percentile([...], 20) for alpha=0.80
        # sorted: -0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05
        # 20th percentile (index = 0.2*9=1.8): -0.04 + 0.8*(-0.03-(-0.04)) = -0.04+0.008 = -0.032
        # tail = values <= -0.032 = [-0.05, -0.04]
        # CVaR = -mean([-0.05, -0.04]) = 0.045
        r = np.array([-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05])
        cutoff = np.percentile(r, 20.0)  # use actual numpy value
        expected = -r[r <= cutoff].mean()
        assert historical_cvar(r, alpha=0.80) == pytest.approx(expected, rel=RTOL)

    def test_positive_loss_magnitude(self):
        """CVaR should be positive for a series that includes large losses."""
        r = np.array([-0.05] * 5 + [0.01] * 95)
        assert historical_cvar(r, alpha=0.95) > 0

    def test_raises_on_bad_alpha(self):
        r = np.ones(50) * 0.01
        with pytest.raises(ValueError, match="alpha"):
            historical_cvar(r, alpha=1.5)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            historical_cvar(pd.Series([], dtype=float))


# ===========================================================================
# Maximum Drawdown
# ===========================================================================

class TestMaxDrawdown:

    def test_flat_returns_zero_drawdown(self):
        """Constant zero returns → wealth flat → drawdown = 0."""
        r = np.zeros(50)
        assert max_drawdown(r) == pytest.approx(0.0, abs=ATOL)

    def test_monotone_rising_zero_drawdown(self):
        """Strictly increasing wealth → drawdown = 0."""
        r = np.full(50, 0.005)   # all positive
        assert max_drawdown(r) == pytest.approx(0.0, abs=ATOL)

    def test_known_50pct_drawdown(self):
        """Wealth doubles then halves: log(2) + (−log(2)) → drawdown = −50 %."""
        r  = pd.Series([np.log(2), -np.log(2)])
        dd = max_drawdown(r)
        # W = [2, 1], peak = [2, 2], drawdown = [0, 0.5/1-1] = [0, -0.5]
        assert dd == pytest.approx(-0.5, rel=RTOL)

    def test_all_losses_drawdown(self):
        """10 identical log-returns of -0.01.
        Wealth path (with implicit W_0=1): [1, e^-0.01, ..., e^-0.10].
        Peak stays at 1.0 throughout, so max drawdown = e^-0.10 - 1."""
        r        = np.full(10, -0.01)
        expected = np.exp(-0.10) - 1.0   # trough at final bar relative to W_0=1
        assert max_drawdown(r) == pytest.approx(expected, rel=RTOL)

    def test_drawdown_is_non_positive(self):
        """Max drawdown must always be ≤ 0."""
        rng = np.random.default_rng(14)
        r   = rng.normal(0, 0.01, 500)
        assert max_drawdown(r) <= 0.0

    def test_drawdown_gte_minus_one(self):
        """Max drawdown must always be ≥ −1 (wealth can't go below 0)."""
        rng = np.random.default_rng(15)
        r   = rng.normal(-0.005, 0.02, 500)
        assert max_drawdown(r) >= -1.0

    def test_larger_crash_gives_larger_drawdown(self):
        """A bigger single-day loss produces a worse (more negative) drawdown."""
        r_small = np.array([0.0, -0.10, 0.0])
        r_large = np.array([0.0, -0.30, 0.0])
        assert max_drawdown(r_large) < max_drawdown(r_small)

    def test_returns_float(self):
        assert isinstance(max_drawdown(np.zeros(10)), float)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            max_drawdown(pd.Series([], dtype=float))


# ===========================================================================
# performance_metrics (aggregate)
# ===========================================================================

class TestPerformanceMetrics:

    @pytest.fixture
    def sample_returns(self):
        rng = np.random.default_rng(16)
        return pd.Series(rng.normal(0.001, 0.01, 500))

    def test_returns_dict(self, sample_returns):
        result = performance_metrics(sample_returns)
        assert isinstance(result, dict)

    def test_expected_keys_alpha95(self, sample_returns):
        keys = performance_metrics(sample_returns, alpha=0.95).keys()
        for expected in ["CAGR", "Ann_Volatility", "Sharpe", "Sortino",
                         "VaR_95", "CVaR_95", "Max_Drawdown"]:
            assert expected in keys

    def test_expected_keys_alpha99(self, sample_returns):
        keys = performance_metrics(sample_returns, alpha=0.99).keys()
        assert "VaR_99" in keys
        assert "CVaR_99" in keys

    def test_no_nan_on_valid_returns(self, sample_returns):
        result = performance_metrics(sample_returns)
        for k, v in result.items():
            assert np.isfinite(v) or np.isinf(v), f"{k} = {v}"

    def test_max_drawdown_non_positive(self, sample_returns):
        assert performance_metrics(sample_returns)["Max_Drawdown"] <= 0.0

    def test_cvar_gte_var(self, sample_returns):
        m = performance_metrics(sample_returns, alpha=0.95)
        assert m["CVaR_95"] >= m["VaR_95"] - 1e-10

    def test_ann_vol_positive(self, sample_returns):
        assert performance_metrics(sample_returns)["Ann_Volatility"] > 0.0


# ===========================================================================
# performance_summary (multi-strategy table)
# ===========================================================================

class TestPerformanceSummary:

    @pytest.fixture
    def strategy_dict(self):
        rng = np.random.default_rng(17)
        return {
            "strategy_A": pd.Series(rng.normal(0.001, 0.01, 500)),
            "strategy_B": pd.Series(rng.normal(0.0005, 0.005, 500)),
        }

    def test_returns_dataframe(self, strategy_dict):
        df = performance_summary(strategy_dict)
        assert isinstance(df, pd.DataFrame)

    def test_rows_match_strategies(self, strategy_dict):
        df = performance_summary(strategy_dict)
        assert set(df.index) == {"strategy_A", "strategy_B"}

    def test_expected_columns_present(self, strategy_dict):
        df = performance_summary(strategy_dict)
        for col in ["CAGR", "Ann_Volatility", "Sharpe", "Sortino",
                    "VaR_95", "CVaR_95", "Max_Drawdown"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nans_in_table(self, strategy_dict):
        df = performance_summary(strategy_dict)
        # finite or inf is acceptable; NaN is not
        for col in df.columns:
            for strat in df.index:
                val = df.loc[strat, col]
                assert not (isinstance(val, float) and np.isnan(val)), (
                    f"NaN found at [{strat}, {col}]"
                )

    def test_higher_vol_strategy_has_larger_ann_vol(self, strategy_dict):
        df = performance_summary(strategy_dict)
        assert df.loc["strategy_A", "Ann_Volatility"] > df.loc["strategy_B", "Ann_Volatility"]

    def test_index_name_is_strategy(self, strategy_dict):
        df = performance_summary(strategy_dict)
        assert df.index.name == "strategy"


# ===========================================================================
# build_and_save_performance_summary (I/O)
# ===========================================================================

class TestBuildAndSavePerformanceSummary:

    def _write_fake_returns(self, tmp_path: object, strategies: dict) -> None:
        """Write synthetic backtest_returns_*.csv files to tmp_path."""
        for name, values in strategies.items():
            df = pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=len(values), freq="B")
                          .strftime("%Y-%m-%d"),
                "portfolio_return": values,
            })
            df.to_csv(tmp_path / f"backtest_returns_{name}.csv", index=False)

    def test_saves_csv(self, tmp_path):
        rng = np.random.default_rng(18)
        strategies = {
            "strat_a": rng.normal(0.001, 0.01, 300),
            "strat_b": rng.normal(0.0005, 0.005, 300),
        }
        self._write_fake_returns(tmp_path, strategies)
        out = tmp_path / "perf_summary.csv"
        build_and_save_performance_summary(
            results_dir=tmp_path, output_path=out
        )
        assert out.exists()

    def test_saved_csv_readable(self, tmp_path):
        rng = np.random.default_rng(19)
        strategies = {"alpha": rng.normal(0.001, 0.01, 300)}
        self._write_fake_returns(tmp_path, strategies)
        out = tmp_path / "perf_summary.csv"
        build_and_save_performance_summary(results_dir=tmp_path, output_path=out)
        df = pd.read_csv(out, index_col=0)
        assert "CAGR" in df.columns
        assert "alpha" in df.index

    def test_no_nans_in_saved_csv(self, tmp_path):
        rng = np.random.default_rng(20)
        strategies = {
            "s1": rng.normal(0.001, 0.01, 500),
            "s2": rng.normal(0.0, 0.02, 500),
        }
        self._write_fake_returns(tmp_path, strategies)
        out = tmp_path / "perf_summary.csv"
        build_and_save_performance_summary(results_dir=tmp_path, output_path=out)
        df = pd.read_csv(out, index_col=0)
        assert not df.isna().any().any()

    def test_raises_if_no_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="backtest_returns"):
            build_and_save_performance_summary(results_dir=empty_dir)

    def test_strategy_names_filter(self, tmp_path):
        rng = np.random.default_rng(21)
        strategies = {
            "s1": rng.normal(0.001, 0.01, 300),
            "s2": rng.normal(0.0, 0.02, 300),
        }
        self._write_fake_returns(tmp_path, strategies)
        out = tmp_path / "perf_summary.csv"
        df = build_and_save_performance_summary(
            results_dir=tmp_path,
            output_path=out,
            strategy_names=["s1"],   # only s1
        )
        assert list(df.index) == ["s1"]
        assert "s2" not in df.index
