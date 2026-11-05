"""Derived series computed from a run's stored frames (cheap, on request).

Date axes
---------
* **Return series** (``/series``) are indexed by *realisation* dates (the index of
  ``returns.parquet``): ``nav`` is the NAV after that day's return, ``drawdown`` is
  :func:`qts.backtest.metrics.drawdown_series` of the net returns (so its minimum
  equals ``summary.max_drawdown_net``), ``turnover`` and ``exante_vol`` are the
  decision-date values of the previous close aligned *positionally* (decision ``t``
  -> realisation ``t+1``, as ``BacktestResult.turnover_on_return_dates`` does).
* **Costs** (``/costs``) are indexed by *decision* dates with the pre-trade NAV.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from qts.backtest import metrics as M


def iso_dates(index: Iterable[Any]) -> list[str]:
    return [pd.Timestamp(d).strftime("%Y-%m-%d") for d in index]


def clean(values: Iterable[Any]) -> list[float | None]:
    """Floats with NaN / inf -> None (JSON null)."""
    out: list[float | None] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(None)
            continue
        out.append(f if math.isfinite(f) else None)
    return out


def finite_or_nan(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def positional(values: pd.Series, target: pd.Index) -> pd.Series:
    """``values`` (by decision date) re-indexed positionally onto ``target`` (realisation dates)."""
    n = min(len(values), len(target))
    out = pd.Series(np.nan, index=target, dtype=float)
    if n:
        out.iloc[:n] = values.to_numpy(dtype=float)[:n]
    return out


def return_series_frame(
    nav: pd.Series,
    returns: pd.DataFrame,
    trades: pd.DataFrame,
    exante_vol: pd.Series | None,
) -> pd.DataFrame:
    """All ``RunSeriesName`` columns on the realisation-date axis."""
    if returns is None or not len(returns):
        return pd.DataFrame(columns=["nav", "gross", "net", "drawdown", "turnover", "exante_vol"])
    idx = pd.DatetimeIndex(returns.index)
    net = returns["net"].astype(float) if "net" in returns else pd.Series(np.nan, index=idx)
    gross = returns["gross"].astype(float) if "gross" in returns else pd.Series(np.nan, index=idx)
    dd = M.drawdown_series(net).reindex(idx)
    turnover = (
        positional(trades.abs().sum(axis=1), idx)
        if trades is not None and len(trades)
        else pd.Series(np.nan, index=idx)
    )
    ev = positional(exante_vol, idx) if exante_vol is not None and len(exante_vol) else pd.Series(np.nan, index=idx)
    return pd.DataFrame(
        {
            "nav": nav.reindex(idx).astype(float) if nav is not None and len(nav) else np.nan,
            "gross": gross,
            "net": net,
            "drawdown": dd,
            "turnover": turnover,
            "exante_vol": ev,
        },
        index=idx,
    )


def cost_sensitivity_rows(stored: Any, gross: pd.Series | None, turnover: pd.Series | None, cost_bps: float):
    """Rows of ``summary.json["cost_sensitivity"]`` (``{"0.5": {...}}``); recomputed with
    :func:`qts.backtest.metrics.cost_sensitivity` when absent."""
    rows: list[dict[str, float]] = []
    if isinstance(stored, dict) and stored:
        for k, v in stored.items():
            if not isinstance(v, dict):
                continue
            rows.append(
                {
                    "multiplier": finite_or_nan(k),
                    "cost_bps_per_side": finite_or_nan(v.get("cost_bps_per_side")),
                    "net_sharpe": finite_or_nan(v.get("net_sharpe")),
                    "net_cagr": finite_or_nan(v.get("net_cagr")),
                    "mean_daily_cost_bps": finite_or_nan(v.get("mean_daily_cost_bps")),
                }
            )
    elif gross is not None and turnover is not None and len(gross) and math.isfinite(cost_bps):
        cs = M.cost_sensitivity(gross, turnover.reindex(gross.index).fillna(0.0), cost_bps_per_side=cost_bps)
        for m, r in cs.iterrows():
            rows.append({"multiplier": float(m), **{c: finite_or_nan(r[c]) for c in cs.columns}})
    return sorted(rows, key=lambda r: r["multiplier"])


def month_end_positions(dates: Sequence[pd.Timestamp]) -> list[int]:
    """Positions of the last date of each calendar month."""
    idx = pd.DatetimeIndex(dates)
    if not len(idx):
        return []
    p = idx.to_period("M")
    last = np.r_[p[1:] != p[:-1], True]
    return [int(i) for i in np.flatnonzero(last)]


def moments(net: pd.Series) -> tuple[float, float, int]:
    """(skew, non-excess kurtosis, n_obs) as the pipeline's DSR uses them."""
    r = net.dropna().astype(float)
    return float(r.skew()), float(r.kurt() + 3.0), len(r)


def ic_by_year(ic: pd.Series) -> list[dict[str, Any]]:
    s = ic.dropna().astype(float)
    if not len(s):
        return []
    g = s.groupby(pd.DatetimeIndex(s.index).year)
    return [{"year": int(y), "mean_ic": float(v.mean()), "n": len(v)} for y, v in g]
