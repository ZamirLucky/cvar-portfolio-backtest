"""CVaR-minimising portfolio allocation (long-only, fully invested)."""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pandas as pd

from ._common import validate_returns, to_weights_series

# Solvers that handle the LP well; tried in order if the default fails.
_PREFERRED_SOLVERS = ["CLARABEL", "HIGHS", "SCS"]


def cvar_minimise(
    returns: pd.DataFrame | np.ndarray,
    alpha: float = 0.95,
    assets: list[str] | None = None,
    solver: str | None = None,
) -> pd.Series:
    """
    Compute the CVaR-minimising portfolio weights.

    Uses the Rockafellar–Uryasev (2000) linear-programming formulation:

        min   ζ + 1/(T · (1−α)) · Σ_t u_t
        s.t.  u_t ≥ −(r_t' w) − ζ    ∀ t = 1, …, T
              u_t ≥ 0                  ∀ t
              Σ w_i = 1
              w_i ≥ 0                  ∀ i

    The objective equals CVaR_α (= Expected Shortfall at level α) of the
    portfolio return distribution, expressed as a loss (minimised ⟹ smallest
    expected loss in the worst (1−α) fraction of scenarios).

    Parameters
    ----------
    returns : pd.DataFrame or np.ndarray, shape (T, n)
        Historical return observations.  Rows = scenarios, columns = assets.
        The more scenarios, the better the CVaR estimate.
    alpha : float
        Confidence level in (0, 1).  Common choices:
          - 0.95 → minimise the expected loss in the worst 5 % of days
          - 0.99 → minimise the expected loss in the worst 1 % of days
        Higher alpha → more conservative (tail-focused) allocation.
    assets : list of str, optional
        Asset names.  Inferred from DataFrame column names when omitted.
    solver : str, optional
        CVXPY solver to use (e.g. ``"CLARABEL"``, ``"HIGHS"``, ``"SCS"``).
        Auto-selects from ``_PREFERRED_SOLVERS`` if None.

    Returns
    -------
    pd.Series
        Optimal weights indexed by asset name.  Values ≥ 0, sum = 1.

    Raises
    ------
    ValueError
        If ``alpha`` is outside (0, 1), input validation fails, or the
        CVXPY solver returns an infeasible / unbounded / error status.

    Warns
    -----
    UserWarning
        If the solver returns ``OPTIMAL_INACCURATE`` — weights are usable
        but may be slightly imprecise.

    References
    ----------
    Rockafellar, R. T., & Uryasev, S. (2000).
    Optimization of conditional value-at-risk.
    *Journal of Risk*, 2(3), 21–41.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be strictly in (0, 1), got {alpha}.")

    arr, names = validate_returns(returns, min_obs=2)

    if assets is not None:
        names = list(assets)
    if names is None:
        names = [f"asset_{i}" for i in range(arr.shape[1])]

    T, n = arr.shape

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------
    w = cp.Variable(n, nonneg=True)   # portfolio weights  (n,)
    zeta = cp.Variable()              # VaR threshold ζ    (scalar)
    u = cp.Variable(T, nonneg=True)   # exceedance slack   (T,)

    # ------------------------------------------------------------------
    # Objective: CVaR_α = ζ + 1/(T(1-α)) · Σ u_t
    # ------------------------------------------------------------------
    scale = 1.0 / (T * (1.0 - alpha))
    objective = cp.Minimize(zeta + scale * cp.sum(u))

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    # Loss at scenario t = -(r_t' w).  Combined with u >= 0, this gives:
    #   u_t = max(loss_t - ζ, 0)  at optimality.
    portfolio_loss = -arr @ w   # shape (T,) — positive value = loss

    constraints = [
        u >= portfolio_loss - zeta,
        cp.sum(w) == 1,
    ]

    problem = cp.Problem(objective, constraints)

    # ------------------------------------------------------------------
    # Solve — try preferred solvers in sequence if no explicit choice
    # ------------------------------------------------------------------
    solvers_to_try = [solver] if solver is not None else _PREFERRED_SOLVERS

    last_status: str = "NOT_SOLVED"
    for slv in solvers_to_try:
        problem.solve(solver=slv, verbose=False)
        last_status = problem.status
        if last_status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            break

    # ------------------------------------------------------------------
    # Status handling
    # ------------------------------------------------------------------
    if last_status == cp.OPTIMAL_INACCURATE:
        warnings.warn(
            f"CVaR optimisation returned OPTIMAL_INACCURATE "
            f"(alpha={alpha}, solver={solver or solvers_to_try[-1]}). "
            "Weights are approximate; consider using a different solver or "
            "increasing the return window.",
            UserWarning,
            stacklevel=2,
        )
    elif last_status not in (cp.OPTIMAL,):
        tried = solver if solver else ", ".join(solvers_to_try)
        raise ValueError(
            f"CVaR optimisation failed.\n"
            f"  Status  : {last_status}\n"
            f"  alpha   : {alpha}\n"
            f"  T × n   : {T} × {n}\n"
            f"  Solvers tried: {tried}\n"
            "Possible causes: too few observations for the chosen alpha "
            "(need at least 1/((1-alpha)·T) effective tail scenarios), "
            "near-singular return matrix, or solver numerical failure. "
            "Try lowering alpha, increasing the window length, or specifying "
            "a different solver via the solver= argument."
        )

    return to_weights_series(w.value, names)
