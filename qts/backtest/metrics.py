"""Performance metrics for daily return series."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from qts.core import stats


def _clean(r: pd.Series) -> pd.Series:
    return pd.Series(r, dtype=float).dropna()


def annualized_return(returns: pd.Series, periods_per_year: int = 252, geometric: bool = True) -> float:
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    if geometric:
        g = float(np.prod(1.0 + r.values))
        if g <= 0:
            return -1.0
        return g ** (periods_per_year / len(r)) - 1.0
    return float(r.mean() * periods_per_year)


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan")


def _autocorr(r: np.ndarray, k: int) -> float:
    if k >= len(r) or k <= 0:
        return 0.0
    x = r - r.mean()
    den = float(x @ x)
    return float(x[k:] @ x[:-k] / den) if den > 0 else 0.0


def annualized_sharpe(
    returns: pd.Series,
    periods_per_year: int = 252,
    rf: float = 0.0,
    lo_adjust: bool = False,
    lags: int | None = None,
) -> float:
    """Sharpe ratio annualised by sqrt(q).  With ``lo_adjust`` the Lo (2002)
    correction for serial correlation is used:
    ``SR_q = SR_1 * q / sqrt(q + 2 * sum_{k=1}^{q-1} (q-k) rho_k)`` (rho_k
    truncated after ``lags`` terms, default min(q-1, 20))."""
    r = _clean(returns) - rf / periods_per_year
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr1 = float(r.mean() / r.std(ddof=1))
    q = periods_per_year
    if not lo_adjust:
        return sr1 * np.sqrt(q)
    L = min(q - 1, lags if lags is not None else 20, len(r) - 2)
    rv = r.values
    s = sum((q - k) * _autocorr(rv, k) for k in range(1, L + 1))
    den = q + 2.0 * s
    if den <= 0:
        return sr1 * np.sqrt(q)
    return sr1 * q / np.sqrt(den)


def sharpe_t_stat(returns: pd.Series) -> float:
    """t-statistic of the Sharpe ratio using Lo's iid standard error
    ``se(SR) = sqrt((1 + SR^2/2) / T)`` on per-period returns."""
    r = _clean(returns)
    T = len(r)
    if T < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = float(r.mean() / r.std(ddof=1))
    return sr / np.sqrt((1.0 + 0.5 * sr**2) / T)


def newey_west_tstat(returns: pd.Series, lags: int | None = None) -> float:
    """t-stat of the mean with a Newey-West (Bartlett kernel) long-run variance
    (:func:`qts.core.stats.newey_west_tstat`, at least 5 observations).
    ``lags`` defaults to ``floor(4 (T/100)^(2/9))``."""
    return stats.newey_west_tstat(_clean(returns).to_numpy(), lags, min_obs=5)


def drawdown_series(returns: pd.Series) -> pd.Series:
    r = _clean(returns)
    nav = (1.0 + r).cumprod()
    return nav / nav.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else float("nan")


def calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return annualized_return(returns, periods_per_year) / abs(mdd)


def sortino(returns: pd.Series, periods_per_year: int = 252, target: float = 0.0) -> float:
    r = _clean(returns) - target / periods_per_year
    if len(r) < 3:
        return float("nan")
    down = np.sqrt(np.mean(np.minimum(r.values, 0.0) ** 2))
    if down == 0:
        return float("nan")
    return float(r.mean() / down * np.sqrt(periods_per_year))


def hit_rate(returns: pd.Series) -> float:
    r = _clean(returns)
    r = r[r != 0]
    return float((r > 0).mean()) if len(r) else float("nan")


def turnover_stats(turnover_series: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    """Two-sided daily turnover statistics (fraction of NAV)."""
    to = _clean(turnover_series)
    if len(to) == 0:
        return {
            "mean_daily": float("nan"),
            "median_daily": float("nan"),
            "annual": float("nan"),
            "one_sided_mean_daily": float("nan"),
        }
    return {
        "mean_daily": float(to.mean()),
        "median_daily": float(to.median()),
        "annual": float(to.mean() * periods_per_year),
        "one_sided_mean_daily": float(to.mean() / 2),
    }


def breakeven_cost_bps(gross_returns: pd.Series, turnover_series: pd.Series) -> float:
    """Cost per side (bps of traded notional) that drives the mean return to
    zero: ``mean(r) / mean(sum_i |x_i|) * 1e4``.  Each unit of two-sided
    turnover is one side of a trade, so a per-side charge ``c`` costs
    ``c * turnover_t`` per day.

    **Which series:** ``r`` must already carry every cost that is *not* a
    per-side trading cost -- pass ``BacktestResult.returns["gross_after_borrow"]``
    (gross minus the day's borrow / pre-trade NAV), not ``"gross"``.  Borrow does
    not scale with turnover, so starting from ``gross`` leaves it out of the gate
    entirely and overstates the breakeven for short-heavy books (audit 2,
    finding 1).  ``turnover_series`` must be positionally aligned with ``r``
    (``BacktestResult.turnover_on_return_dates()``)."""
    g = _clean(gross_returns)
    to = pd.Series(turnover_series, dtype=float).reindex(g.index).dropna()
    g = g.reindex(to.index)
    if len(g) == 0 or to.mean() <= 0:
        return float("inf") if g.mean() > 0 else float("nan")
    return float(g.mean() / to.mean() * 1e4)


def cost_sensitivity(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps_per_side: float = 10.0,
    multipliers: Iterable[float] = (0.5, 1, 2, 3),
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Net Sharpe / CAGR under ``cost_bps_per_side * m`` for each multiplier
    ``m``, charging ``cost * turnover_t`` per day.

    ``gross_returns`` should be ``BacktestResult.returns["gross_after_borrow"]``
    (borrow already deducted, as realised); see :func:`breakeven_cost_bps`.  With
    plain ``"gross"`` the 1x row overstates the net Sharpe by the borrow drag."""
    g = _clean(gross_returns)
    to = pd.Series(turnover, dtype=float).reindex(g.index).fillna(0.0)
    rows = []
    for m in multipliers:
        c = cost_bps_per_side * m / 1e4
        net = g - c * to
        rows.append(
            {
                "multiplier": m,
                "cost_bps_per_side": cost_bps_per_side * m,
                "net_sharpe": annualized_sharpe(net, periods_per_year),
                "net_cagr": annualized_return(net, periods_per_year),
                "mean_daily_cost_bps": float(c * to.mean() * 1e4),
            }
        )
    return pd.DataFrame(rows).set_index("multiplier")
