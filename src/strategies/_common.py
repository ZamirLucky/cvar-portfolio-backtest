"""
Shared helpers for all strategy modules.

Functions
---------
validate_returns  – parse + validate a returns matrix, return (ndarray, names)
to_weights_series – wrap a weight array in a labelled pd.Series
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_returns(
    returns: pd.DataFrame | np.ndarray,
    min_obs: int = 2,
) -> tuple[np.ndarray, list[str] | None]:
    """
    Parse and validate a returns matrix.

    Parameters
    ----------
    returns : pd.DataFrame or array-like, shape (T, n)
        Historical return observations.  Rows = time steps, columns = assets.
    min_obs : int
        Minimum required number of rows.

    Returns
    -------
    arr : np.ndarray, shape (T, n), dtype float64
    names : list[str] | None
        Asset names extracted from DataFrame columns; None for bare arrays.

    Raises
    ------
    ValueError
        On empty input, wrong dimensionality, too few observations,
        or any NaN values.
    """
    if isinstance(returns, pd.DataFrame):
        names: list[str] | None = list(returns.columns)
        arr = returns.to_numpy(dtype=float)
    else:
        arr = np.asarray(returns, dtype=float)
        names = None

    if arr.size == 0:
        raise ValueError("returns is empty.")

    if arr.ndim != 2:
        raise ValueError(
            f"returns must be 2-D (T × n), got shape {arr.shape}."
        )

    T, n = arr.shape

    if T < min_obs:
        raise ValueError(
            f"Need at least {min_obs} observations (rows), got {T}."
        )

    if n < 1:
        raise ValueError("returns must have at least 1 asset (column).")

    nan_count = int(np.isnan(arr).sum())
    if nan_count:
        raise ValueError(
            f"returns contains {nan_count} NaN value(s). "
            "Drop or fill missing data before calling a strategy function."
        )

    return arr, names


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

def to_weights_series(
    weights: np.ndarray,
    assets: list[str] | None,
) -> pd.Series:
    """
    Wrap raw optimiser weights in a labelled pd.Series.

    Clips tiny solver-noise negatives to zero, then renormalises so that
    weights sum exactly to 1.  Raises if the raw total deviates by more
    than 1 % (indicating a real solver problem, not just floating-point
    noise).

    Parameters
    ----------
    weights : array-like, shape (n,)
    assets  : list of str | None
        Index for the returned Series.  Falls back to "asset_0", … if None.

    Returns
    -------
    pd.Series  indexed by asset names, values ≥ 0, sum = 1.
    """
    w = np.asarray(weights, dtype=float).copy()

    # Clip solver floating-point noise (e.g. -1e-12 → 0)
    w = np.clip(w, 0.0, None)

    total = w.sum()
    if abs(total - 1.0) > 1e-2:
        raise ValueError(
            f"Weights sum to {total:.6f} — expected ≈ 1.0. "
            "The optimiser may have returned an infeasible solution."
        )

    w /= total  # renormalise to machine-precision 1.0

    if assets is None:
        assets = [f"asset_{i}" for i in range(len(w))]

    return pd.Series(w, index=assets, name="weight")
