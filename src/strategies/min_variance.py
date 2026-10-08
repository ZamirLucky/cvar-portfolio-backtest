"""Minimum-variance portfolio allocation (long-only, fully invested)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._common import validate_returns, to_weights_series


def minimum_variance(
    returns: pd.DataFrame | np.ndarray,
    assets: list[str] | None = None,
    tol: float = 1e-10,
) -> pd.Series:
    """
    Compute the minimum-variance portfolio weights.

    Solves:

        min   w' Σ w
        s.t.  Σ w_i = 1
              w_i ≥ 0   ∀i

    using the SLSQP method in ``scipy.optimize.minimize``.  The sample
    covariance matrix Σ is estimated from the supplied return history.

    Parameters
    ----------
    returns : pd.DataFrame or np.ndarray, shape (T, n)
        Historical return observations.  T rows = time steps, n cols = assets.
        Minimum 2 rows required.
    assets : list of str, optional
        Asset names.  Inferred from DataFrame column names if *returns* is a
        DataFrame and this argument is omitted.
    tol : float
        Optimiser convergence tolerance (passed as ``ftol`` to SLSQP).

    Returns
    -------
    pd.Series
        Optimal weights indexed by asset name.  Values ≥ 0, sum = 1.

    Raises
    ------
    ValueError
        If the input fails validation or the optimiser does not converge.

    Notes
    -----
    The analytical gradient ``∂(w'Σw)/∂w = 2Σw`` is supplied to the
    optimiser for faster, more accurate convergence.
    """
    arr, names = validate_returns(returns, min_obs=2)

    if assets is not None:
        names = list(assets)
    if names is None:
        names = [f"asset_{i}" for i in range(arr.shape[1])]

    n = arr.shape[1]
    cov: np.ndarray = np.cov(arr.T)  # (n × n) sample covariance

    # Degenerate case: single asset
    if n == 1:
        return pd.Series([1.0], index=names, name="weight")

    def _objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    def _gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * cov @ w

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.full(n, 1.0 / n)  # warm-start from equal weight

    result = minimize(
        _objective,
        w0,
        jac=_gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": 1000},
    )

    if not result.success:
        raise ValueError(
            f"Minimum-variance optimisation did not converge.\n"
            f"  Message : {result.message!r}\n"
            f"  T = {arr.shape[0]}, n = {n}\n"
            "Try increasing the window length or checking for near-constant "
            "return series."
        )

    return to_weights_series(result.x, names)
