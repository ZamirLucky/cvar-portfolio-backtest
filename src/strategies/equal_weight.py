"""Equal-weight (1/N) portfolio allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(assets: list[str] | pd.Index) -> pd.Series:
    """
    Return an equal-weight (1/N) portfolio.

    The simplest possible allocation: every asset receives weight 1/N.
    Always long-only and fully invested by construction.

    Parameters
    ----------
    assets : list of str or pd.Index
        Ordered asset names.  Length N determines the weight.

    Returns
    -------
    pd.Series
        Weights indexed by asset name.  All values equal 1/N, sum = 1.

    Raises
    ------
    ValueError
        If ``assets`` is empty.

    Examples
    --------
    >>> equal_weight(["ETF01", "ETF02", "ETF03"])
    ETF01    0.333333
    ETF02    0.333333
    ETF03    0.333333
    Name: weight, dtype: float64
    """
    assets = list(assets)
    n = len(assets)
    if n == 0:
        raise ValueError("assets must be non-empty.")

    w = np.full(n, 1.0 / n)
    return pd.Series(w, index=assets, name="weight")
