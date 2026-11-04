"""Factor exposure construction (Barra/USE4 style).

Factor block layout for every exposure matrix (tickers x factors):

* ``market``      -- the USE4 *Country* factor: unit exposure for every stock.
* one dummy per industry label in ``MarketData.industry`` (0/1, rows sum to 1).
* style factors from ``RiskConfig.styles`` -- standardized Barra-style.

Style definitions (all computed from ``MarketData`` with data ``<= t`` only)
------------------------------------------------------------------------------
beta
    Slope of a rolling 252-day regression of the stock's return on the
    equal-weighted market return, shrunk toward 1 (Blume / Vasicek style
    prior):  ``beta = 1 + (1 - shrink) * (beta_hat - 1)``, ``shrink = 1/3``.
    Rolling cov/var is computed from rolling means of products so the whole
    panel is handled in a few vectorized passes.
size
    ``log(market_cap)``.
momentum
    12-1 month return: ``P_{t-21} / P_{t-252} - 1`` (skips the most recent
    month, which belongs to reversal).
value
    ``log(book_to_price)`` when available, otherwise ``earnings_yield``.
residual_vol
    252-day idiosyncratic volatility relative to the market regression:
    ``sqrt(var(r) - cov(r, m)^2 / var(m))``.
liquidity
    ``log`` of the 63-day mean turnover, where turnover is proxied by
    ``dollar_volume / market_cap`` (= volume / shares outstanding).
short_term_reversal
    21-day return ``P_t / P_{t-21} - 1``.
quality
    ``gross_profitability``.
investment
    ``-asset_growth`` (conservative investment loads positively).
short_interest
    ``short_interest_ratio``.

Standardization (USE4 Methodology Notes, section on descriptor standardization):
cap-weighted mean 0 and equal-weighted standard deviation 1, outliers trimmed
at +-3 standard deviations.  We apply: cap-weighted z-score -> clip to [-3, 3]
-> re-standardize (so the sd is ~1 after trimming) -> fill missing with 0
(a missing style exposure is treated as "average", the USE4 convention).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from qts.core.config import RiskConfig
from qts.core.crosssection import zscore
from qts.core.types import MarketData

MARKET_FACTOR = "market"
BETA_WINDOW = 252
RESVOL_WINDOW = 252
MOMENTUM_LONG = 252
MOMENTUM_SKIP = 21
REVERSAL_WINDOW = 21
LIQUIDITY_WINDOW = 63
BETA_SHRINK = 1.0 / 3.0
WINSOR_SD = 3.0

SUPPORTED_STYLES = [
    "beta", "size", "momentum", "value", "residual_vol", "liquidity",
    "short_term_reversal", "quality", "investment", "short_interest",
]


# --------------------------------------------------------------------------- raw styles
def _rolling_market_regression(returns: pd.DataFrame, window: int, min_periods: int):
    """Vectorized rolling beta and residual variance of each column on the
    equal-weighted market return.  Returns ``(beta_hat, resid_var)`` wide frames."""
    m = returns.mean(axis=1)
    r_mean = returns.rolling(window, min_periods=min_periods).mean()
    m_mean = m.rolling(window, min_periods=min_periods).mean()
    rm_mean = returns.mul(m, axis=0).rolling(window, min_periods=min_periods).mean()
    r_var = returns.rolling(window, min_periods=min_periods).var(ddof=0)
    m_var = m.rolling(window, min_periods=min_periods).var(ddof=0).replace(0.0, np.nan)
    cov = rm_mean - r_mean.mul(m_mean, axis=0)
    beta_hat = cov.div(m_var, axis=0)
    resid_var = (r_var - cov.pow(2).div(m_var, axis=0)).clip(lower=0.0)
    return beta_hat, resid_var


def raw_style_panels(md: MarketData, styles: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """Compute un-standardized style descriptors as wide panels (date x ticker).

    Every descriptor is causal (rolling windows / lags only), so slicing a
    panel at date ``t`` uses data ``<= t`` only."""
    styles = list(styles)
    unknown = set(styles) - set(SUPPORTED_STYLES)
    if unknown:
        raise ValueError(f"unsupported styles: {sorted(unknown)}")
    out: Dict[str, pd.DataFrame] = {}
    returns = md.returns
    need_reg = {"beta", "residual_vol"} & set(styles)
    if need_reg:
        beta_hat, resid_var = _rolling_market_regression(returns, BETA_WINDOW, BETA_WINDOW // 4)
    for s in styles:
        if s == "beta":
            out[s] = 1.0 + (1.0 - BETA_SHRINK) * (beta_hat - 1.0)
        elif s == "size":
            out[s] = np.log(md.market_cap.where(md.market_cap > 0))
        elif s == "momentum":
            out[s] = md.close.shift(MOMENTUM_SKIP) / md.close.shift(MOMENTUM_LONG) - 1.0
        elif s == "value":
            if md.book_to_price is not None:
                out[s] = np.log(md.book_to_price.where(md.book_to_price > 0))
            elif md.earnings_yield is not None:
                out[s] = md.earnings_yield.copy()
            else:
                out[s] = pd.DataFrame(np.nan, index=md.dates, columns=md.tickers)
        elif s == "residual_vol":
            out[s] = np.sqrt(resid_var)
        elif s == "liquidity":
            turnover = (md.dollar_volume / md.market_cap.replace(0.0, np.nan))
            out[s] = np.log(turnover.rolling(LIQUIDITY_WINDOW, min_periods=LIQUIDITY_WINDOW // 3)
                            .mean().where(lambda x: x > 0))
        elif s == "short_term_reversal":
            out[s] = md.close / md.close.shift(REVERSAL_WINDOW) - 1.0
        elif s == "quality":
            out[s] = (md.gross_profitability.copy() if md.gross_profitability is not None
                      else pd.DataFrame(np.nan, index=md.dates, columns=md.tickers))
        elif s == "investment":
            out[s] = (-md.asset_growth if md.asset_growth is not None
                      else pd.DataFrame(np.nan, index=md.dates, columns=md.tickers))
        elif s == "short_interest":
            out[s] = (md.short_interest_ratio.copy() if md.short_interest_ratio is not None
                      else pd.DataFrame(np.nan, index=md.dates, columns=md.tickers))
    return out


def standardize_style(raw: pd.DataFrame, market_cap: pd.DataFrame) -> pd.DataFrame:
    """Barra standardization of a wide descriptor panel, row by row:
    cap-weighted mean 0 / equal-weighted sd 1, trim at +-3 sd, re-standardize,
    fill missing with 0."""
    raw = raw.replace([np.inf, -np.inf], np.nan)
    z = zscore(raw, weights=market_cap).clip(-WINSOR_SD, WINSOR_SD)
    z = zscore(z, weights=market_cap)
    return z.fillna(0.0)


def standardized_style_panels(md: MarketData, styles: Iterable[str]) -> Dict[str, pd.DataFrame]:
    raw = raw_style_panels(md, styles)
    return {k: standardize_style(v, md.market_cap) for k, v in raw.items()}


# --------------------------------------------------------------------------- assembly
def industry_dummies(industry: pd.Series, tickers: pd.Index) -> pd.DataFrame:
    ind = industry.reindex(tickers)
    d = pd.get_dummies(ind, dtype=float)
    d = d.reindex(columns=sorted(d.columns))
    d.index = tickers
    return d


def factor_names(md: MarketData, cfg: RiskConfig) -> List[str]:
    inds = sorted(md.industry.reindex(md.tickers).dropna().unique())
    return [MARKET_FACTOR] + list(inds) + list(cfg.styles)


def exposure_history(md: MarketData, dates: Optional[Iterable[pd.Timestamp]] = None,
                     cfg: Optional[RiskConfig] = None) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Exposure matrices for many dates.

    All rolling statistics are computed *once* on the wide panels (vectorized),
    stacked into a (T x N x K) array and sliced per date -- no per-date
    recomputation.  Because every descriptor is causal this is point-in-time
    correct.  A ticker is included on date ``t`` if it has a finite close and
    a positive market cap on ``t``."""
    cfg = cfg or RiskConfig()
    dates = md.dates if dates is None else pd.DatetimeIndex(list(dates))
    styles = list(cfg.styles)
    panels = standardized_style_panels(md, styles)
    dummies = industry_dummies(md.industry, md.tickers)
    cols = pd.Index([MARKET_FACTOR] + list(dummies.columns) + styles, name="factor")
    tickers_all = md.tickers
    N = len(tickers_all)
    static = np.column_stack([np.ones(N)] + [dummies[c].values for c in dummies.columns])
    style_arr = (np.stack([panels[s].reindex(index=md.dates, columns=tickers_all).values for s in styles], axis=2)
                 if styles else np.zeros((len(md.dates), N, 0)))
    valid = (md.close.notna() & (md.market_cap > 0)).values
    pos = {d: i for i, d in enumerate(md.dates)}
    out: Dict[pd.Timestamp, pd.DataFrame] = {}
    for d in dates:
        i = pos.get(d)
        if i is None:
            continue
        mask = valid[i]
        arr = np.concatenate([static[mask], style_arr[i][mask]], axis=1)
        out[d] = pd.DataFrame(arr, index=tickers_all[mask], columns=cols)
    return out


def build_exposures(md: MarketData, date: pd.Timestamp, cfg: Optional[RiskConfig] = None) -> pd.DataFrame:
    """Exposure matrix (tickers x factors) as of ``date`` using data ``<= date``.

    For many dates prefer :func:`exposure_history`, which shares the rolling
    computations across dates."""
    cfg = cfg or RiskConfig()
    date = pd.Timestamp(date)
    sub = md.slice_until(date)
    hist = exposure_history(sub, [sub.dates[-1]], cfg)
    if not hist:
        raise ValueError(f"no data on or before {date}")
    return hist[sub.dates[-1]]


def split_factor_columns(columns: Iterable[str], cfg: RiskConfig):
    """Return ``(market, industries, styles)`` column lists for an exposure frame."""
    cols = list(columns)
    styles = [c for c in cols if c in cfg.styles]
    market = [c for c in cols if c == MARKET_FACTOR]
    industries = [c for c in cols if c not in styles and c != MARKET_FACTOR]
    return market, industries, styles
