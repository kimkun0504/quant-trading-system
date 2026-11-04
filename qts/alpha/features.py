"""Tier-1 feature construction (BUILD_PLAN §2.3).

Every panel produced by :class:`FeatureBuilder.build` is strictly point-in-time:
the value at date ``t`` is a function of ``MarketData`` rows with timestamp
``<= t`` only.  This is guaranteed structurally — all time-series operations
are ``shift(k >= 0)`` / trailing ``rolling`` windows — and verified by the
leakage test in ``tests/test_alpha.py``.

Processing (:meth:`FeatureBuilder.process`) is per-date (era-wise):
winsorize -> industry demean (selected features) -> rank-gaussianise ->
fill NaN with 0 -> long frame.  The *same* function is used for training and
live scoring (process parity; see ``trainer.train_live_parity_check``).

All rolling statistics are vectorised wide-panel operations; there are no
Python loops over dates.
"""
from __future__ import annotations

import hashlib
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from qts.core.config import AlphaConfig
from qts.core.crosssection import demean_by_group, rank_gauss
from qts.core.types import Long, MarketData, Wide, to_long

# Canonical feature order.  Keep stable: it is part of the live/train parity hash.
FEATURE_ORDER: List[str] = [
    "value_bp", "value_ep",
    "mom_12_1", "mom_6_1", "mom_change", "industry_momentum",
    "quality_gpa", "investment_ag",
    "beta_252", "idio_vol_252", "max_ret_21",
    "turnover_63", "amihud_63", "log_dollar_vol_63", "log_mcap",
    "short_interest", "borrow_fee",
    "strev_21",
]


def feature_signature(columns) -> str:
    """Hash of feature names *and order* — used for train/live parity checks."""
    return hashlib.sha1("|".join(map(str, columns)).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- helpers
def _winsorize_rows(wide: Wide, pct: float) -> Wide:
    """Vectorised per-date winsorisation (same semantics as crosssection.winsorize
    but computed with nanpercentile along axis=1 instead of a row-wise apply)."""
    if pct <= 0:
        return wide
    arr = wide.to_numpy(dtype=float)
    valid = np.isfinite(arr).sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows during warm-up
        lo = np.nanpercentile(arr, 100 * pct, axis=1)
        hi = np.nanpercentile(arr, 100 * (1 - pct), axis=1)
    lo = np.where(valid >= 10, lo, -np.inf)
    hi = np.where(valid >= 10, hi, np.inf)
    out = np.clip(arr, lo[:, None], hi[:, None])
    out[~np.isfinite(arr)] = np.nan
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


def _trailing_return(close: Wide, start_lag: int, end_lag: int) -> Wide:
    """Return from t-start_lag to t-end_lag (both >= 0, start > end)."""
    return close.shift(end_lag) / close.shift(start_lag) - 1.0


def rolling_market_beta(returns: Wide, market: pd.Series, window: int, min_periods: Optional[int] = None):
    """Vectorised rolling OLS beta of each column on ``market`` plus residual vol.

    Returns ``(beta, resid_vol)`` wide panels, both using data ``<= t`` only.
    """
    mp = min_periods or int(window * 0.8)
    rm = returns.mul(market, axis=0)
    mean_r = returns.rolling(window, min_periods=mp).mean()
    mean_m = market.rolling(window, min_periods=mp).mean()
    cov = rm.rolling(window, min_periods=mp).mean() - mean_r.mul(mean_m, axis=0)
    var_m = market.rolling(window, min_periods=mp).var(ddof=0).replace(0.0, np.nan)
    var_r = returns.rolling(window, min_periods=mp).var(ddof=0)
    beta = cov.div(var_m, axis=0)
    resid_var = var_r - (cov ** 2).div(var_m, axis=0)
    resid_vol = np.sqrt(resid_var.clip(lower=0.0))
    return beta, resid_vol


def cap_weighted_group_mean(values: Wide, weights: Wide, groups: pd.Series) -> Wide:
    """Per-date cap-weighted mean of ``values`` within each group, broadcast back
    to every member of the group."""
    g = groups.reindex(values.columns)
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    w = weights.where(values.notna())
    for key, cols in g.groupby(g).groups.items():
        cols = list(cols)
        num = (values[cols] * w[cols]).sum(axis=1, min_count=1)
        den = w[cols].sum(axis=1, min_count=1)
        out[cols] = np.repeat((num / den).to_numpy()[:, None], len(cols), axis=1)
    return out


# ----------------------------------------------------------------------------- builder
class FeatureBuilder:
    """Builds raw Tier-1 feature panels and processes them into a model-ready long frame."""

    def __init__(self, cfg: AlphaConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ raw
    def build(self, md: MarketData) -> Dict[str, Wide]:
        close = md.close
        rets = md.returns
        mcap = md.market_cap
        dvol = md.dollar_volume
        ind = md.industry
        f: Dict[str, Wide] = {}

        # value
        if md.book_to_price is not None:
            f["value_bp"] = np.log(md.book_to_price.clip(lower=1e-6))
        if md.earnings_yield is not None:
            f["value_ep"] = md.earnings_yield.copy()

        # momentum
        f["mom_12_1"] = _trailing_return(close, 252, 21)
        f["mom_6_1"] = _trailing_return(close, 126, 21)
        f["mom_change"] = f["mom_12_1"] - f["mom_12_1"].shift(63)
        f["industry_momentum"] = cap_weighted_group_mean(f["mom_6_1"], mcap, ind)

        # quality / investment
        if md.gross_profitability is not None:
            f["quality_gpa"] = md.gross_profitability.copy()
        if md.asset_growth is not None:
            f["investment_ag"] = -md.asset_growth

        # low risk
        market = rets.mean(axis=1)
        beta, idio = rolling_market_beta(rets, market, 252)
        f["beta_252"] = beta
        f["idio_vol_252"] = idio
        f["max_ret_21"] = rets.rolling(21, min_periods=15).max()

        # liquidity / size
        f["turnover_63"] = (dvol / mcap).rolling(63, min_periods=40).mean()
        amihud = (rets.abs() / dvol.replace(0.0, np.nan)).rolling(63, min_periods=40).mean()
        f["amihud_63"] = np.log(amihud * 1e6 + 1e-12)
        f["log_dollar_vol_63"] = np.log(dvol.rolling(63, min_periods=40).mean().clip(lower=1.0))
        f["log_mcap"] = np.log(mcap.clip(lower=1.0))

        # shorting
        if md.short_interest_ratio is not None:
            f["short_interest"] = md.short_interest_ratio.copy()
        if md.borrow_fee is not None:
            f["borrow_fee"] = md.borrow_fee.copy()

        # short-term residual reversal (timing feature): -(21d return, industry-demeaned)
        r21 = _trailing_return(close, 21, 0)
        f["strev_21"] = -demean_by_group(r21, ind)

        ordered = {k: f[k] for k in FEATURE_ORDER if k in f}
        for k, v in ordered.items():
            v.index.name = "date"
            v.columns.name = "ticker"
        return ordered

    # ------------------------------------------------------------------ processing
    def process(self, features: Dict[str, Wide], md: MarketData) -> Long:
        """Per-date: winsorize -> industry demean (cfg.industry_demean) -> rank_gauss
        -> fill 0.  Returns a long frame indexed by (date, ticker)."""
        cols = []
        demean_set = set(self.cfg.industry_demean)
        for name in FEATURE_ORDER:
            if name not in features:
                continue
            w = _winsorize_rows(features[name], self.cfg.winsor_pct)
            if name in demean_set:
                w = demean_by_group(w, md.industry)
            w = rank_gauss(w)
            cols.append(to_long(w, name))
        out = pd.concat(cols, axis=1).fillna(0.0)
        out = out.reindex(columns=[c for c in FEATURE_ORDER if c in out.columns])
        return out

    def build_and_process(self, md: MarketData) -> Long:
        return self.process(self.build(md), md)
