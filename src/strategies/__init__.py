"""
Strategy allocation functions for the CVaR portfolio backtest.

Public API
----------
equal_weight(assets)
    Equal-weight (1/N) allocation.

minimum_variance(returns, assets=None, tol=1e-10)
    Minimum-variance portfolio via scipy SLSQP.

cvar_minimise(returns, alpha=0.95, assets=None, solver=None)
    CVaR-minimising portfolio via CVXPY (Rockafellar–Uryasev LP).

All functions return a ``pd.Series`` indexed by asset name with weights
that are long-only (≥ 0) and fully invested (sum = 1).
"""

from .equal_weight import equal_weight
from .min_variance import minimum_variance
from .cvar_min import cvar_minimise

__all__ = ["equal_weight", "minimum_variance", "cvar_minimise"]
