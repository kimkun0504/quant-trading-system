"""Portfolio-level volatility targeting."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def portfolio_vol_scalar(predicted_vol_annual: float, target: Optional[float],
                         max_leverage_scalar: float = 1.0, min_scalar: float = 0.0) -> float:
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


def realized_vol_annual(returns: pd.Series, window: int = 63, half_life: Optional[int] = None,
                        periods_per_year: int = 252) -> float:
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


def realized_vol_scalar(returns: pd.Series, target: float, window: int = 63,
                        max_leverage_scalar: float = 1.0) -> float:
    return portfolio_vol_scalar(realized_vol_annual(returns, window), target, max_leverage_scalar)
