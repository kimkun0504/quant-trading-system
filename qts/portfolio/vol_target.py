"""Portfolio-level volatility targeting helpers.

``portfolio_vol_scalar`` is the plain ``target / predicted`` multiplier used
for *reporting* and for realised-vol overlays.  ``SinglePeriodOptimizer`` does
**not** multiply the book by it.  With the default
``PortfolioConfig.risk_aversion_mode="target_vol"`` it sizes the book by
choosing the risk aversion so that the alpha-aware optimum has the target
ex-ante vol (second-pass audit finding 1).  With ``"fixed"`` gamma,
``vol_target_mode`` either re-solves a projection ``min gamma H (w - s w*)'
Sigma (w - s w*) + c TC(w - w0)`` onto the full constraint set with a risk cap
(``"projection"``), or adds the SOC cap ``sqrt(w' Sigma w) <= target`` to the
alpha solve itself (``"constraint"``, one solve, target is a ceiling).  Every
band holds after targeting in all modes (see ``qts.portfolio.optimizer``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_vol_scalar(
    predicted_vol_annual: float,
    target: float | None,
    max_leverage_scalar: float = 1.0,
    min_scalar: float = 0.0,
) -> float:
    """Multiplier applied to the whole book so that predicted vol <= target.

    ``max_leverage_scalar`` caps scaling *up* (default 1.0 = never lever up;
    the optimizer already chose the risk level, vol targeting only trims).
    Returns 1.0 when there is no target or the prediction is degenerate.
    """
    if target is None or not np.isfinite(target) or target <= 0:
        return 1.0
    if predicted_vol_annual is None or not np.isfinite(predicted_vol_annual) or predicted_vol_annual <= 0:
        return 1.0
    s = target / predicted_vol_annual
    return float(np.clip(s, min_scalar, max_leverage_scalar))


def realized_vol_annual(
    returns: pd.Series, window: int = 63, half_life: int | None = None, periods_per_year: int = 252
) -> float:
    """Realised-vol fallback used when no ex-ante estimate is available
    (EWMA if ``half_life`` is given, else simple rolling window)."""
    r = pd.Series(returns).dropna()
    if len(r) < 5:
        return float("nan")
    if half_life:
        v = r.ewm(halflife=half_life, min_periods=5).var().iloc[-1]
    else:
        v = r.tail(window).var()
    return float(np.sqrt(v * periods_per_year))


def realized_vol_scalar(
    returns: pd.Series, target: float, window: int = 63, max_leverage_scalar: float = 1.0
) -> float:
    return portfolio_vol_scalar(realized_vol_annual(returns, window), target, max_leverage_scalar)
