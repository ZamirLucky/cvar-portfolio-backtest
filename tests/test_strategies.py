"""
Unit tests for src/strategies allocation functions.

Tests are grouped into four classes:
    TestEqualWeight      – equal_weight(assets)
    TestMinimumVariance  – minimum_variance(returns, ...)
    TestCVaRMinimise     – cvar_minimise(returns, alpha, ...)
    TestSharedBehaviour  – type consistency, NaN-free output, noise clipping

All fixtures are synthetic and seeded — no file I/O, no network, no notebooks.

Run from project root:
    pytest tests/test_strategies.py -v
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from src.strategies import cvar_minimise, equal_weight, minimum_variance
from src.strategies._common import to_weights_series

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
# to_weights_series clips negatives to 0 and renormalises — outputs are
# exactly non-negative after clipping, and sum to 1 to machine precision.
TOL_SUM = 1e-8
NOISE   = 1e-6  # realistic solver noise magnitude for sanity checks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
RNG_SEED = 42


@pytest.fixture
def assets_3() -> list[str]:
    return ["A", "B", "C"]


@pytest.fixture
def assets_6() -> list[str]:
    return ["ETF01", "ETF02", "ETF03", "ETF04", "ETF05", "ETF06"]


@pytest.fixture
def returns_3(assets_3) -> pd.DataFrame:
    """100-row × 3-col returns DataFrame with equal, modest volatility."""
    rng = np.random.default_rng(RNG_SEED)
    data = rng.normal(0.0, 0.01, (100, 3))
    return pd.DataFrame(data, columns=assets_3)


@pytest.fixture
def returns_6(assets_6) -> pd.DataFrame:
    """200-row × 6-col returns DataFrame mimicking the project universe."""
    rng = np.random.default_rng(RNG_SEED)
    # Slightly varied vol per asset — realistic but synthetic
    vols = [0.012, 0.011, 0.010, 0.013, 0.006, 0.004]
    data = np.column_stack([
        rng.normal(0.0, v, 200) for v in vols
    ])
    return pd.DataFrame(data, columns=assets_6)


@pytest.fixture
def returns_mixed_vol() -> pd.DataFrame:
    """3-asset DataFrame with clearly distinct volatilities.

    low  ~ N(0, 0.002)
    mid  ~ N(0, 0.010)
    high ~ N(0, 0.030)

    Used to verify that risk-minimising strategies favour low-vol assets.
    """
    rng = np.random.default_rng(RNG_SEED)
    data = np.column_stack([
        rng.normal(0.0, 0.002, 300),
        rng.normal(0.0, 0.010, 300),
        rng.normal(0.0, 0.030, 300),
    ])
    return pd.DataFrame(data, columns=["low", "mid", "high"])


@pytest.fixture
def returns_2row(assets_3) -> pd.DataFrame:
    """Minimal valid DataFrame — exactly 2 observations."""
    return pd.DataFrame(
        [[0.01, -0.02, 0.005], [-0.01, 0.02, -0.005]],
        columns=assets_3,
    )


# ---------------------------------------------------------------------------
# TestEqualWeight
# ---------------------------------------------------------------------------

class TestEqualWeight:

    def test_returns_series(self, assets_3):
        w = equal_weight(assets_3)
        assert isinstance(w, pd.Series)

    def test_series_name(self, assets_3):
        w = equal_weight(assets_3)
        assert w.name == "weight"

    def test_weights_sum_to_one(self, assets_3):
        w = equal_weight(assets_3)
        assert abs(w.sum() - 1.0) < TOL_SUM

    def test_weights_all_positive(self, assets_3):
        w = equal_weight(assets_3)
        assert (w >= 0).all()

    def test_weights_are_equal(self, assets_3):
        w = equal_weight(assets_3)
        expected = 1.0 / len(assets_3)
        assert np.allclose(w.values, expected)

    def test_index_matches_asset_order(self, assets_3):
        w = equal_weight(assets_3)
        assert list(w.index) == assets_3

    def test_index_matches_asset_order_6(self, assets_6):
        w = equal_weight(assets_6)
        assert list(w.index) == assets_6

    def test_accepts_pandas_index(self, assets_6):
        idx = pd.Index(assets_6)
        w = equal_weight(idx)
        assert list(w.index) == assets_6

    def test_single_asset(self):
        w = equal_weight(["only"])
        assert len(w) == 1
        assert abs(w.sum() - 1.0) < TOL_SUM
        assert w["only"] == pytest.approx(1.0)

    def test_two_assets(self):
        w = equal_weight(["X", "Y"])
        assert w["X"] == pytest.approx(0.5)
        assert w["Y"] == pytest.approx(0.5)

    def test_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            equal_weight([])


# ---------------------------------------------------------------------------
# TestMinimumVariance
# ---------------------------------------------------------------------------

class TestMinimumVariance:

    # --- Output type and shape ---

    def test_returns_series(self, returns_3):
        w = minimum_variance(returns_3)
        assert isinstance(w, pd.Series)

    def test_series_name(self, returns_3):
        w = minimum_variance(returns_3)
        assert w.name == "weight"

    def test_index_matches_dataframe_columns(self, returns_3, assets_3):
        w = minimum_variance(returns_3)
        assert list(w.index) == assets_3

    def test_length_matches_n_assets(self, returns_3):
        w = minimum_variance(returns_3)
        assert len(w) == returns_3.shape[1]

    # --- Constraints ---

    def test_weights_sum_to_one(self, returns_3):
        w = minimum_variance(returns_3)
        assert abs(w.sum() - 1.0) < TOL_SUM

    def test_weights_sum_to_one_6assets(self, returns_6):
        w = minimum_variance(returns_6)
        assert abs(w.sum() - 1.0) < TOL_SUM

    def test_weights_non_negative(self, returns_3):
        w = minimum_variance(returns_3)
        assert (w >= 0).all()

    def test_weights_non_negative_6assets(self, returns_6):
        w = minimum_variance(returns_6)
        assert (w >= 0).all()

    def test_no_nan_weights(self, returns_3):
        w = minimum_variance(returns_3)
        assert not w.isna().any()

    # --- Alternate input types ---

    def test_accepts_numpy_array(self, returns_3):
        """Bare ndarray: names fall back to 'asset_0', 'asset_1', …"""
        w = minimum_variance(returns_3.to_numpy())
        assert abs(w.sum() - 1.0) < TOL_SUM
        assert list(w.index) == ["asset_0", "asset_1", "asset_2"]

    def test_assets_kwarg_overrides_column_names(self, returns_3):
        override = ["X", "Y", "Z"]
        w = minimum_variance(returns_3, assets=override)
        assert list(w.index) == override

    def test_minimal_two_row_input(self, returns_2row):
        w = minimum_variance(returns_2row)
        assert abs(w.sum() - 1.0) < TOL_SUM

    def test_single_asset_returns_one(self):
        data = pd.DataFrame({"solo": [0.01, -0.01, 0.02, -0.02]})
        w = minimum_variance(data)
        assert w["solo"] == pytest.approx(1.0)

    # --- Economic sanity ---

    def test_favours_low_vol_asset(self, returns_mixed_vol):
        """Min-variance must allocate more weight to the low-vol asset."""
        w = minimum_variance(returns_mixed_vol)
        assert w["low"] > w["mid"] > w["high"]

    # --- Error cases ---

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            minimum_variance(pd.DataFrame())

    def test_raises_on_1d_array(self):
        with pytest.raises(ValueError, match="2-D"):
            minimum_variance(np.array([0.01, -0.02, 0.005]))

    def test_raises_on_single_row(self, assets_3):
        one_row = pd.DataFrame([[0.01, -0.02, 0.005]], columns=assets_3)
        with pytest.raises(ValueError, match="at least 2 observations"):
            minimum_variance(one_row)

    def test_raises_on_any_nan(self, assets_3):
        df = pd.DataFrame(
            [[0.01, np.nan, 0.005], [0.02, -0.01, 0.003]],
            columns=assets_3,
        )
        with pytest.raises(ValueError, match="NaN"):
            minimum_variance(df)

    def test_raises_on_all_nan(self, assets_3):
        df = pd.DataFrame(
            [[np.nan] * 3, [np.nan] * 3],
            columns=assets_3,
        )
        with pytest.raises(ValueError, match="NaN"):
            minimum_variance(df)


# ---------------------------------------------------------------------------
# TestCVaRMinimise
# ---------------------------------------------------------------------------

class TestCVaRMinimise:

    # --- Output type and shape ---

    def test_returns_series(self, returns_3):
        w = cvar_minimise(returns_3)
        assert isinstance(w, pd.Series)

    def test_series_name(self, returns_3):
        w = cvar_minimise(returns_3)
        assert w.name == "weight"

    def test_index_matches_dataframe_columns(self, returns_3, assets_3):
        w = cvar_minimise(returns_3)
        assert list(w.index) == assets_3

    def test_length_matches_n_assets(self, returns_3):
        w = cvar_minimise(returns_3)
        assert len(w) == returns_3.shape[1]

    # --- Constraints ---

    def test_weights_sum_to_one_alpha95(self, returns_3):
        w = cvar_minimise(returns_3, alpha=0.95)
        assert abs(w.sum() - 1.0) < TOL_SUM

    def test_weights_sum_to_one_alpha99(self, returns_3):
        w = cvar_minimise(returns_3, alpha=0.99)
        assert abs(w.sum() - 1.0) < TOL_SUM

    def test_weights_sum_to_one_6assets(self, returns_6):
        w = cvar_minimise(returns_6, alpha=0.95)
        assert abs(w.sum() - 1.0) < TOL_SUM

    def test_weights_non_negative_alpha95(self, returns_3):
        w = cvar_minimise(returns_3, alpha=0.95)
        assert (w >= 0).all()

    def test_weights_non_negative_alpha99(self, returns_3):
        w = cvar_minimise(returns_3, alpha=0.99)
        assert (w >= 0).all()

    def test_weights_non_negative_6assets(self, returns_6):
        w = cvar_minimise(returns_6, alpha=0.95)
        assert (w >= 0).all()

    def test_no_nan_weights(self, returns_3):
        w = cvar_minimise(returns_3)
        assert not w.isna().any()

    # --- Alpha parametrisation ---

    @pytest.mark.parametrize("alpha", [0.90, 0.95, 0.99])
    def test_valid_alpha_values(self, returns_6, alpha):
        w = cvar_minimise(returns_6, alpha=alpha)
        assert abs(w.sum() - 1.0) < TOL_SUM
        assert (w >= 0).all()

    def test_higher_alpha_more_conservative(self, returns_mixed_vol):
        """α=0.99 should place at least as much weight in the low-vol asset as α=0.95."""
        w95 = cvar_minimise(returns_mixed_vol, alpha=0.95)
        w99 = cvar_minimise(returns_mixed_vol, alpha=0.99)
        # The low-vol asset should be favoured more (or equally) at the tighter tail level
        assert w99["low"] >= w95["low"] - NOISE

    # --- Alternate input types ---

    def test_accepts_numpy_array(self, returns_3):
        w = cvar_minimise(returns_3.to_numpy(), alpha=0.95)
        assert abs(w.sum() - 1.0) < TOL_SUM
        assert list(w.index) == ["asset_0", "asset_1", "asset_2"]

    def test_assets_kwarg_overrides_column_names(self, returns_3):
        override = ["X", "Y", "Z"]
        w = cvar_minimise(returns_3, assets=override)
        assert list(w.index) == override

    def test_minimal_two_row_input(self, returns_2row):
        w = cvar_minimise(returns_2row, alpha=0.95)
        assert abs(w.sum() - 1.0) < TOL_SUM

    # --- Economic sanity ---

    def test_favours_low_vol_asset(self, returns_mixed_vol):
        """CVaR-min must allocate more weight to the low-vol asset."""
        w = cvar_minimise(returns_mixed_vol, alpha=0.95)
        assert w["low"] > w["high"]

    # --- OPTIMAL_INACCURATE emits UserWarning ---

    def test_optimal_inaccurate_emits_warning(self, returns_3):
        """If CVXPY returns OPTIMAL_INACCURATE the function warns but still returns weights.

        Strategy: run the real solver to populate variable values, then
        override the status to OPTIMAL_INACCURATE before the status-check
        code reads it.  This exercises the warning branch without relying
        on CVXPY internal attributes.
        """
        _real_solve = cp.Problem.solve

        def _mock_solve(self_problem, solver=None, verbose=False, **kwargs):
            _real_solve(self_problem, solver=solver, verbose=verbose, **kwargs)
            self_problem._status = cp.OPTIMAL_INACCURATE

        with patch.object(cp.Problem, "solve", _mock_solve):
            with pytest.warns(UserWarning, match="OPTIMAL_INACCURATE"):
                w = cvar_minimise(returns_3, alpha=0.95)
        assert abs(w.sum() - 1.0) < TOL_SUM

    # --- Solver failure raises ValueError ---

    def test_raises_on_solver_failure(self, returns_3):
        """A mocked infeasible status must raise a descriptive ValueError."""
        def _mock_infeasible(self, solver=None, verbose=False, **kwargs):
            self._status = cp.INFEASIBLE

        with patch.object(cp.Problem, "solve", _mock_infeasible):
            with pytest.raises(ValueError, match="CVaR optimisation failed"):
                cvar_minimise(returns_3, alpha=0.95)

    def test_solver_failure_message_contains_alpha(self, returns_3):
        """Error message must report the alpha level used."""
        def _mock_infeasible(self, solver=None, verbose=False, **kwargs):
            self._status = cp.INFEASIBLE

        with patch.object(cp.Problem, "solve", _mock_infeasible):
            with pytest.raises(ValueError, match="0.99"):
                cvar_minimise(returns_3, alpha=0.99)

    # --- Invalid alpha error cases ---

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5, 2.0])
    def test_raises_on_invalid_alpha(self, returns_3, bad_alpha):
        with pytest.raises(ValueError, match="alpha must be strictly in"):
            cvar_minimise(returns_3, alpha=bad_alpha)

    # --- Invalid returns error cases ---

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            cvar_minimise(pd.DataFrame())

    def test_raises_on_1d_array(self):
        with pytest.raises(ValueError, match="2-D"):
            cvar_minimise(np.array([0.01, -0.02, 0.005]))

    def test_raises_on_single_row(self, assets_3):
        one_row = pd.DataFrame([[0.01, -0.02, 0.005]], columns=assets_3)
        with pytest.raises(ValueError, match="at least 2 observations"):
            cvar_minimise(one_row)

    def test_raises_on_any_nan(self, assets_3):
        df = pd.DataFrame(
            [[0.01, np.nan, 0.005], [0.02, -0.01, 0.003]],
            columns=assets_3,
        )
        with pytest.raises(ValueError, match="NaN"):
            cvar_minimise(df)

    def test_raises_on_all_nan(self, assets_3):
        df = pd.DataFrame(
            [[np.nan] * 3, [np.nan] * 3],
            columns=assets_3,
        )
        with pytest.raises(ValueError, match="NaN"):
            cvar_minimise(df)


# ---------------------------------------------------------------------------
# TestSharedBehaviour
# ---------------------------------------------------------------------------

class TestSharedBehaviour:
    """Cross-strategy invariants that must hold regardless of the algorithm."""

    @pytest.mark.parametrize("strategy,kwargs", [
        ("ew",  {}),
        ("mv",  {}),
        ("cv",  {"alpha": 0.95}),
        ("cv",  {"alpha": 0.99}),
    ])
    def test_return_type_is_series(self, returns_3, assets_3, strategy, kwargs):
        if strategy == "ew":
            w = equal_weight(assets_3)
        elif strategy == "mv":
            w = minimum_variance(returns_3, **kwargs)
        else:
            w = cvar_minimise(returns_3, **kwargs)
        assert isinstance(w, pd.Series)

    @pytest.mark.parametrize("strategy,kwargs", [
        ("ew",  {}),
        ("mv",  {}),
        ("cv",  {"alpha": 0.95}),
    ])
    def test_no_nan_weights(self, returns_3, assets_3, strategy, kwargs):
        if strategy == "ew":
            w = equal_weight(assets_3)
        elif strategy == "mv":
            w = minimum_variance(returns_3, **kwargs)
        else:
            w = cvar_minimise(returns_3, **kwargs)
        assert not w.isna().any(), f"{strategy}: weights contain NaN"

    @pytest.mark.parametrize("strategy,kwargs", [
        ("ew",  {}),
        ("mv",  {}),
        ("cv",  {"alpha": 0.95}),
    ])
    def test_sum_to_one(self, returns_3, assets_3, strategy, kwargs):
        if strategy == "ew":
            w = equal_weight(assets_3)
        elif strategy == "mv":
            w = minimum_variance(returns_3, **kwargs)
        else:
            w = cvar_minimise(returns_3, **kwargs)
        assert abs(w.sum() - 1.0) < TOL_SUM, f"{strategy}: sum = {w.sum()}"

    @pytest.mark.parametrize("strategy,kwargs", [
        ("ew",  {}),
        ("mv",  {}),
        ("cv",  {"alpha": 0.95}),
    ])
    def test_long_only(self, returns_3, assets_3, strategy, kwargs):
        if strategy == "ew":
            w = equal_weight(assets_3)
        elif strategy == "mv":
            w = minimum_variance(returns_3, **kwargs)
        else:
            w = cvar_minimise(returns_3, **kwargs)
        assert (w >= 0).all(), f"{strategy}: negative weight found"

    def test_to_weights_series_clips_small_negatives(self):
        """Tiny solver noise negatives must be clipped to zero, not rejected."""
        raw = np.array([-1e-12, 0.5, 0.5 + 1e-12])
        w = to_weights_series(raw, ["a", "b", "c"])
        assert (w >= 0).all()
        assert abs(w.sum() - 1.0) < TOL_SUM

    def test_to_weights_series_rejects_large_negative(self):
        """A genuinely negative weight (e.g. -0.05) should raise after clipping."""
        raw = np.array([-0.05, 0.55, 0.55])   # sum after clip = 1.10, far from 1
        with pytest.raises(ValueError, match="sum to"):
            to_weights_series(raw, ["a", "b", "c"])

    def test_to_weights_series_renormalises(self):
        """Weights slightly off 1.0 due to floating point are renormalised."""
        raw = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])  # may not sum exactly to 1
        w = to_weights_series(raw, ["a", "b", "c"])
        assert abs(w.sum() - 1.0) < TOL_SUM

    def test_to_weights_series_auto_names(self):
        """When assets=None, names fall back to 'asset_0', 'asset_1', …"""
        w = to_weights_series(np.array([0.4, 0.6]), None)
        assert list(w.index) == ["asset_0", "asset_1"]
