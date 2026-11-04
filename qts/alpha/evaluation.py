"""Signal diagnostics (BUILD_PLAN §2.4 Gate 2): IC statistics, decile spreads,
IC decay, signal half-life and incremental IC.  All rank correlations are
vectorised across dates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from qts.core.types import MarketData, Wide

from .targets import forward_returns, residualize_forward_returns


# ----------------------------------------------------------------------------- vectorised rank IC
def _row_corr(a: np.ndarray, b: np.ndarray, min_n: int = 20) -> np.ndarray:
    """Row-wise Pearson correlation of two equally shaped arrays, ignoring NaNs
    (a pair is used only where both are finite)."""
    m = np.isfinite(a) & np.isfinite(b)
    a = np.where(m, a, 0.0)
    b = np.where(m, b, 0.0)
    n = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = a.sum(axis=1) / n
        mb = b.sum(axis=1) / n
        ac = np.where(m, a - ma[:, None], 0.0)
        bc = np.where(m, b - mb[:, None], 0.0)
        r = (ac * bc).sum(axis=1) / np.sqrt((ac ** 2).sum(axis=1) * (bc ** 2).sum(axis=1))
    r[n < min_n] = np.nan
    return r


def rank_ic_series(pred: Wide, target: Wide) -> pd.Series:
    """Per-date Spearman IC between two wide panels (aligned on the intersection)."""
    idx = pred.index.intersection(target.index)
    cols = pred.columns.intersection(target.columns)
    p = pred.loc[idx, cols]
    t = target.loc[idx, cols]
    both = p.notna() & t.notna()
    p = p.where(both)
    t = t.where(both)
    rp = p.rank(axis=1).to_numpy(dtype=float)
    rt = t.rank(axis=1).to_numpy(dtype=float)
    return pd.Series(_row_corr(rp, rt), index=idx, name="rank_ic")


# ----------------------------------------------------------------------------- IC report
def infer_periods_per_year(index: pd.Index, trading_days_per_year: int = 252) -> float:
    """Observations per year implied by the index spacing: ``252 / median gap``
    where the gap is measured in business days (so a series sampled every 20
    trading days gives 12.6, a daily series 252)."""
    idx = pd.DatetimeIndex(index).sort_values()
    if len(idx) < 2:
        return float(trading_days_per_year)
    gaps = np.diff(idx.values).astype("timedelta64[D]").astype(float)
    med_cal = float(np.median(gaps))
    med_bdays = max(1.0, round(med_cal * 5.0 / 7.0)) if med_cal > 1.5 else 1.0
    return trading_days_per_year / med_bdays


def ic_report(pred: Wide, target: Wide, periods_per_year: Optional[float] = None,
              trading_days_per_year: int = 252) -> Dict:
    """Mean rank IC, sd, ICIR (annualised), t-stat, hit rate and by-year table.

    ``periods_per_year`` (for the ICIR annualisation) defaults to the sampling
    frequency *inferred from the IC series' index spacing* (median gap in
    business days -> ``252 / gap``), so a score panel sampled every ``H`` days
    is annualised with ``252 / H`` rather than 252 (audit finding 4).  The
    t-stat uses ``n`` as is and is unaffected.
    """
    ic = rank_ic_series(pred, target).dropna()
    if len(ic) == 0:
        return {"n": 0, "mean_ic": np.nan, "ic_sd": np.nan, "icir": np.nan, "t_stat": np.nan,
                "hit_rate": np.nan, "by_year": pd.DataFrame(), "ic_series": ic, "periods_per_year": np.nan}
    ppy = float(periods_per_year) if periods_per_year is not None else infer_periods_per_year(ic.index, trading_days_per_year)
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    by_year = ic.groupby(ic.index.year).agg(["mean", "std", "count"])
    by_year["t_stat"] = by_year["mean"] / by_year["std"] * np.sqrt(by_year["count"])
    return {
        "n": int(len(ic)),
        "mean_ic": mean,
        "ic_sd": sd,
        "icir": mean / sd * np.sqrt(ppy) if sd > 0 else np.nan,
        "t_stat": mean / sd * np.sqrt(len(ic)) if sd > 0 else np.nan,
        "hit_rate": float((ic > 0).mean()),
        "by_year": by_year,
        "ic_series": ic,
        "periods_per_year": ppy,
    }


# ----------------------------------------------------------------------------- decile spread
@dataclass
class DecileSpreadResult:
    returns: pd.Series            # per-date long-short period return (top minus bottom decile)
    mean: float
    sharpe: float                 # annualised, assuming ``periods_per_year`` non-overlapping periods
    t_stat: float                 # Newey-West (lags = horizon-1) if overlap, else plain
    horizon: int


def decile_spread(pred: Wide, fwd_raw: Wide, n: int = 10, weights: Optional[Wide] = None,
                  horizon: int = 1, periods_per_year: int = 252) -> DecileSpreadResult:
    """Top-minus-bottom decile forward return per formation date.

    ``fwd_raw`` is the (raw) forward return over ``horizon`` days aligned at the
    formation date.  ``weights`` (e.g. market cap) gives value-weighted deciles.
    Because consecutive daily formation dates overlap when ``horizon > 1``, the
    t-stat uses a Newey-West correction with ``horizon - 1`` lags and the
    Sharpe is annualised by ``periods_per_year / horizon``.
    """
    idx = pred.index.intersection(fwd_raw.index)
    cols = pred.columns.intersection(fwd_raw.columns)
    p = pred.loc[idx, cols].where(fwd_raw.loc[idx, cols].notna())
    r = fwd_raw.loc[idx, cols]
    pct = p.rank(axis=1, pct=True)
    top = (pct > 1 - 1 / n)
    bot = (pct <= 1 / n)
    w = pd.DataFrame(1.0, index=idx, columns=cols) if weights is None else weights.reindex(index=idx, columns=cols).fillna(0.0)
    def port(mask):
        ww = (w * mask).where(r.notna()).fillna(0.0)
        return (ww * r.fillna(0.0)).sum(axis=1) / ww.sum(axis=1).replace(0, np.nan)
    spread = (port(top) - port(bot)).dropna()
    spread.name = "decile_spread"
    mean = float(spread.mean())
    sd = float(spread.std(ddof=1))
    ppy = periods_per_year / max(horizon, 1)
    sharpe = mean / sd * np.sqrt(ppy) if sd > 0 else np.nan
    t = _nw_tstat(spread.values, lags=max(horizon - 1, 0))
    return DecileSpreadResult(spread, mean, sharpe, t, horizon)


def _nw_tstat(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    mu = x.mean()
    e = x - mu
    var = (e @ e) / n
    for L in range(1, min(lags, n - 1) + 1):
        wgt = 1 - L / (lags + 1)
        var += 2 * wgt * (e[L:] @ e[:-L]) / n
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else np.nan


# ----------------------------------------------------------------------------- decay / half-life
def ic_decay_curve(score: Wide, md: MarketData, horizons: Sequence[int] = (1, 5, 10, 20, 40, 60, 120),
                   residualized: bool = True, features=None) -> pd.Series:
    """Mean rank IC of ``score`` against forward returns at each horizon
    (cumulative, not marginal).  Residualised targets by default."""
    out = {}
    for h in horizons:
        tgt = residualize_forward_returns(md, h, features=features) if residualized else forward_returns(md, h)
        out[h] = float(rank_ic_series(score, tgt).mean())
    s = pd.Series(out, name="mean_rank_ic")
    s.index.name = "horizon"
    return s


def rank_autocorrelation(score: Wide, lags: Iterable[int]) -> pd.Series:
    """Mean cross-sectional rank correlation between score_t and score_{t-lag}."""
    out = {}
    a = score.rank(axis=1).to_numpy(dtype=float)
    for lag in lags:
        if lag >= len(score):
            out[lag] = np.nan
            continue
        r = _row_corr(a[lag:], a[:-lag] if lag > 0 else a)
        out[lag] = float(np.nanmean(r))
    return pd.Series(out, name="rank_autocorr")


DEFAULT_HL_LAGS = (1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 80, 100, 120)


def signal_half_life(score: Wide, lags: Sequence[int] = DEFAULT_HL_LAGS, max_lag: int = 120) -> float:
    """Half-life (trading days) of the score's cross-sectional rank autocorrelation.

    Fits ``log(rho_lag) = -lambda * lag`` by least squares through the origin on
    lags with ``rho > 0``; ``half_life = ln 2 / lambda``.  Returns +inf if the
    autocorrelation does not decay and NaN if nothing can be estimated.
    """
    lags = [l for l in lags if 0 < l <= max_lag and l < len(score)]
    if not lags:
        return float("nan")
    rho = rank_autocorrelation(score, lags).dropna()
    rho = rho[rho > 0.02]
    if len(rho) < 2:
        return float("nan")
    x = np.asarray(rho.index, float)
    yv = -np.log(rho.values)
    lam = float((x @ yv) / (x @ x))
    if lam <= 1e-8:
        return float("inf")
    return float(np.log(2) / lam)


# ----------------------------------------------------------------------------- incremental IC
def residualize_on(new_score: Wide, existing: Wide) -> Wide:
    """Per-date OLS residual of ``new_score`` on ``existing`` (with intercept), vectorised."""
    idx = new_score.index.intersection(existing.index)
    cols = new_score.columns.intersection(existing.columns)
    a = new_score.loc[idx, cols]
    b = existing.loc[idx, cols]
    m = a.notna() & b.notna()
    a, b = a.where(m), b.where(m)
    am, bm = a.mean(axis=1), b.mean(axis=1)
    ac, bc = a.sub(am, axis=0), b.sub(bm, axis=0)
    beta = (ac * bc).sum(axis=1) / (bc ** 2).sum(axis=1).replace(0, np.nan)
    return ac - bc.mul(beta, axis=0)


def incremental_ic(new_score: Wide, existing_composite: Wide, target: Wide) -> Dict:
    """IC report of ``new_score`` after residualising it on the existing composite
    (Gate 2 hurdle: incremental rank IC > 0.005 and positive in >= 2/3 of folds)."""
    resid = residualize_on(new_score, existing_composite)
    rep = ic_report(resid, target)
    rep["standalone_mean_ic"] = float(rank_ic_series(new_score, target).mean())
    rep["composite_mean_ic"] = float(rank_ic_series(existing_composite, target).mean())
    return rep
