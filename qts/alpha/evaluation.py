"""Signal diagnostics (BUILD_PLAN §2.4 Gate 2): IC statistics, decile spreads,
IC decay, signal half-life and incremental IC.  All rank correlations are
vectorised across dates.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd

from qts.core import stats
from qts.core.types import MarketData, Wide

from .targets import forward_returns, residualize_forward_returns

# ----------------------------------------------------------------------------- vectorised rank IC
#: Row-wise Pearson / per-date Spearman IC: the single implementations live in
#: :mod:`qts.core.stats` (min pairs ``stats.IC_MIN_OBS`` = 20).
_row_corr = stats.row_corr
rank_ic_series = stats.rank_ic_series


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


def newey_west_lags(n_obs: int, overlap: int = 1) -> int:
    """Bartlett-kernel bandwidth for a t-stat on ``n_obs`` observations whose
    labels overlap by ``overlap`` observations (handbook fix #7, audit #15):

        L = max(h, floor(4 * (T / 100) ** (2 / 9)))

    i.e. the Newey-West (1994) automatic rule floored at the overlap ``h``.
    ``L = h - 1`` (the old choice) under-covers an MA(h-1) process because the
    Bartlett weights ``1 - k/(L+1)`` down-weight exactly the lags that carry
    the overlap; flooring at ``h`` keeps every overlapping lag with weight
    > 0 and the automatic term covers residual serial correlation when the
    observations do not overlap (``h = 1``).  Capped at ``n_obs - 1``."""
    T = int(n_obs)
    if T < 2:
        return 0
    auto = stats.default_nw_lags(T)
    return int(min(max(int(overlap), auto), T - 1))


@dataclass
class ICReport:
    """Typed result of :func:`ic_report` (handbook 1.5 "typed results").

    ``t_stat`` is the plain ``mean / sd * sqrt(n)`` (correct for the
    pipeline's non-overlapping samples); ``t_stat_nw`` is the Newey-West t with
    ``nw_lags = newey_west_lags(n, overlap)`` and is the one to read for daily
    IC series of multi-day labels.  ``decay`` is the mean IC against the
    targets passed as ``decay_targets`` (index: horizon in days; the pipeline
    reports ``AlphaConfig.horizon_days`` and ``secondary_horizons``).

    Dict-style access (``rep["mean_ic"]``, ``rep.get(...)``, ``"n" in rep``) is
    kept for backward compatibility during the transition; new code should use
    attributes.
    """

    n: int = 0
    mean_ic: float = float("nan")
    ic_sd: float = float("nan")
    icir: float = float("nan")
    t_stat: float = float("nan")
    t_stat_nw: float = float("nan")
    nw_lags: int = 0
    hit_rate: float = float("nan")
    periods_per_year: float = float("nan")
    by_year: pd.DataFrame = field(default_factory=pd.DataFrame)
    ic_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, name="rank_ic"))
    decay: pd.Series = field(default_factory=lambda: pd.Series(dtype=float, name="mean_rank_ic"))

    # ---- dict-style backward compatibility
    def __getitem__(self, key: str) -> Any:
        if key in self._field_names():
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return key in self._field_names()

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key) if key in self._field_names() else default

    def keys(self) -> Iterator[str]:
        return iter(self._field_names())

    @classmethod
    def _field_names(cls) -> Sequence[str]:
        return [f.name for f in fields(cls)]

    def scalars(self) -> dict[str, float]:
        """JSON-safe scalar fields (no frames/series)."""
        out: dict[str, float] = {}
        for k in self._field_names():
            v = getattr(self, k)
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                out[k] = float(v) if not isinstance(v, (int, np.integer)) else int(v)
        return out


@dataclass
class IncrementalICReport(ICReport):
    """:func:`incremental_ic` result: the IC report of the residualised score plus
    the standalone and composite mean ICs."""

    standalone_mean_ic: float = float("nan")
    composite_mean_ic: float = float("nan")


def ic_report(
    pred: Wide,
    target: Wide,
    periods_per_year: float | None = None,
    trading_days_per_year: int = 252,
    overlap: int = 1,
    decay_targets: Mapping[int, Wide] | None = None,
) -> ICReport:
    """Mean rank IC, sd, ICIR (annualised), t-stats, hit rate, by-year table and
    an optional IC-decay section -> :class:`ICReport`.

    ``periods_per_year`` (for the ICIR annualisation) defaults to the sampling
    frequency *inferred from the IC series' index spacing* (median gap in
    business days -> ``252 / gap``), so a score panel sampled every ``H`` days
    is annualised with ``252 / H`` rather than 252 (audit finding 4).
    ``overlap``: number of consecutive IC observations sharing label days
    (``H`` for a daily IC of ``H``-day labels, 1 for non-overlapping samples);
    it floors the Newey-West bandwidth of ``t_stat_nw`` (:func:`newey_west_lags`).
    ``decay_targets``: ``{horizon: target panel}``; ``decay[h]`` is the mean
    rank IC of ``pred`` against ``decay_targets[h]``.
    """
    ic = rank_ic_series(pred, target).dropna()
    decay = pd.Series(
        {int(h): float(rank_ic_series(pred, t).mean()) for h, t in (decay_targets or {}).items()},
        dtype=float,
        name="mean_rank_ic",
    ).sort_index()
    decay.index.name = "horizon"
    if len(ic) == 0:
        return ICReport(ic_series=ic, decay=decay)
    ppy = (
        float(periods_per_year)
        if periods_per_year is not None
        else infer_periods_per_year(ic.index, trading_days_per_year)
    )
    mean, sd = float(ic.mean()), float(ic.std(ddof=1))
    by_year = ic.groupby(ic.index.year).agg(["mean", "std", "count"])
    by_year["t_stat"] = by_year["mean"] / by_year["std"] * np.sqrt(by_year["count"])
    lags = newey_west_lags(len(ic), overlap)
    return ICReport(
        n=len(ic),
        mean_ic=mean,
        ic_sd=sd,
        icir=mean / sd * np.sqrt(ppy) if sd > 0 else np.nan,
        t_stat=mean / sd * np.sqrt(len(ic)) if sd > 0 else np.nan,
        t_stat_nw=stats.newey_west_tstat(ic.to_numpy(dtype=float), lags=int(lags)),
        nw_lags=lags,
        hit_rate=float((ic > 0).mean()),
        periods_per_year=ppy,
        by_year=by_year,
        ic_series=ic,
        decay=decay,
    )


# ----------------------------------------------------------------------------- decile spread
@dataclass
class DecileSpreadResult:
    returns: pd.Series  # per-date long-short period return (top minus bottom decile)
    mean: float
    sharpe: float  # annualised, assuming ``periods_per_year`` non-overlapping periods
    t_stat: float  # Newey-West, lags = newey_west_lags(T, horizon)
    horizon: int
    nw_lags: int = 0


def decile_spread(
    pred: Wide,
    fwd_raw: Wide,
    n: int = 10,
    weights: Wide | None = None,
    horizon: int = 1,
    periods_per_year: int = 252,
) -> DecileSpreadResult:
    """Top-minus-bottom decile forward return per formation date.

    ``fwd_raw`` is the (raw) forward return over ``horizon`` days aligned at the
    formation date.  ``weights`` (e.g. market cap) gives value-weighted deciles.
    Because consecutive daily formation dates overlap when ``horizon > 1``, the
    t-stat uses a Newey-West correction with
    ``L = max(horizon, floor(4 (T/100)^(2/9)))`` lags (:func:`newey_west_lags`;
    for non-overlapping formation dates pass ``horizon=1`` and the automatic
    term applies) and the Sharpe is annualised by ``periods_per_year / horizon``.
    """
    idx = pred.index.intersection(fwd_raw.index)
    cols = pred.columns.intersection(fwd_raw.columns)
    p = pred.loc[idx, cols].where(fwd_raw.loc[idx, cols].notna())
    r = fwd_raw.loc[idx, cols]
    pct = p.rank(axis=1, pct=True)
    top = pct > 1 - 1 / n
    bot = pct <= 1 / n
    w = (
        pd.DataFrame(1.0, index=idx, columns=cols)
        if weights is None
        else weights.reindex(index=idx, columns=cols).fillna(0.0)
    )

    def port(mask):
        ww = (w * mask).where(r.notna()).fillna(0.0)
        return (ww * r.fillna(0.0)).sum(axis=1) / ww.sum(axis=1).replace(0, np.nan)

    spread = (port(top) - port(bot)).dropna()
    spread.name = "decile_spread"
    mean = float(spread.mean())
    sd = float(spread.std(ddof=1))
    ppy = periods_per_year / max(horizon, 1)
    sharpe = mean / sd * np.sqrt(ppy) if sd > 0 else np.nan
    lags = newey_west_lags(len(spread), max(int(horizon), 1))
    t = stats.newey_west_tstat(spread.to_numpy(dtype=float), lags=int(lags))
    return DecileSpreadResult(spread, mean, sharpe, t, horizon, nw_lags=lags)


# ----------------------------------------------------------------------------- decay / half-life
def ic_decay_curve(
    score: Wide,
    md: MarketData,
    horizons: Sequence[int] = (1, 5, 10, 20, 40, 60, 120),
    residualized: bool = True,
    features=None,
) -> pd.Series:
    """Mean rank IC of ``score`` against forward returns at each horizon
    (cumulative, not marginal).  Residualised targets by default."""
    out = {}
    for h in horizons:
        tgt = (
            residualize_forward_returns(md, h, features=features) if residualized else forward_returns(md, h)
        )
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
    beta = (ac * bc).sum(axis=1) / (bc**2).sum(axis=1).replace(0, np.nan)
    return ac - bc.mul(beta, axis=0)


def incremental_ic(
    new_score: Wide, existing_composite: Wide, target: Wide, overlap: int = 1
) -> IncrementalICReport:
    """IC report of ``new_score`` after residualising it on the existing composite
    (Gate 2 hurdle: incremental rank IC > 0.005 and positive in >= 2/3 of folds)."""
    resid = residualize_on(new_score, existing_composite)
    rep = ic_report(resid, target, overlap=overlap)
    vals = {f.name: getattr(rep, f.name) for f in fields(ICReport)}
    return IncrementalICReport(
        **vals,
        standalone_mean_ic=float(rank_ic_series(new_score, target).mean()),
        composite_mean_ic=float(rank_ic_series(existing_composite, target).mean()),
    )
