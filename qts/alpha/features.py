"""Tier-1 feature construction (BUILD_PLAN §2.3).

Every panel produced by :class:`FeatureBuilder.build` is strictly point-in-time:
the value at date ``t`` is a function of ``MarketData`` rows with timestamp
``<= t`` only.  This is guaranteed structurally — all time-series operations
are ``shift(k >= 0)`` / trailing ``rolling`` windows — and verified by the
leakage test in ``tests/test_alpha.py``.

Processing (:meth:`FeatureBuilder.process`) is per-date (era-wise) and
**restricted to the names eligible at that date** (``MarketData.eligibility()``:
in the universe as of ``t`` and with a price at ``t``): winsorize -> industry
demean (selected features) -> rank-gaussianise -> fill NaN with 0 (eligible
names only) -> long frame containing *only* eligible (date, ticker) rows.
Ineligible names are therefore absent — never zero-filled — so neither future
universe entrants nor dead names can sit in the cross-section.  The *same*
function is used for training and live scoring (process parity; see
``trainer.train_live_parity_check``).

All rolling statistics are vectorised wide-panel operations; there are no
Python loops over dates.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from qts.core import stats
from qts.core.config import AlphaConfig
from qts.core.crosssection import demean_by_group
from qts.core.roles import FEATURE_SET_2_FEATURES
from qts.core.stats import rank_gauss
from qts.core.types import Long, MarketData, Wide, to_long

from .features_v2 import EARNINGS_MAX_AGE_DAYS, build_v2_features

# Canonical feature order.  Keep stable: it is part of the live/train parity hash.
FEATURE_ORDER: list[str] = [
    "value_bp",
    "value_ep",
    "mom_12_1",
    "mom_6_1",
    "mom_change",
    "industry_momentum",
    "quality_gpa",
    "investment_ag",
    "beta_252",
    "idio_vol_252",
    "max_ret_21",
    "turnover_63",
    "amihud_63",
    "log_dollar_vol_63",
    "log_mcap",
    "short_interest",
    "borrow_fee",
    "strev_21",
]


#: ``AlphaConfig.feature_set`` -> canonical feature order.  ``"v2"`` appends the
#: eight feature-set-2 features (handbook 04 "Feature set 2"); ``FEATURE_ORDER``
#: (v1) is a prefix, so the v1 columns keep their positions.
FEATURE_SETS: dict[str, list[str]] = {
    "v1": FEATURE_ORDER,
    "v2": [*FEATURE_ORDER, *FEATURE_SET_2_FEATURES],
}
#: v2 features that go through the ordinary chain winsorise -> industry demean ->
#: rank-Gaussian (all but the conditioning variable below), in addition to
#: ``AlphaConfig.industry_demean``.
V2_INDUSTRY_DEMEAN: frozenset[str] = frozenset(FEATURE_SET_2_FEATURES) - {"days_since_earnings"}
#: Conditioning variables: not winsorised, industry-demeaned or ranked.  feature ->
#: (divisor, fill): the processed value is ``raw / divisor`` in [0, 1] and a missing
#: value (no announcement seen yet) is ``fill`` -- the cap, i.e. "no recent earnings".
CONDITIONING_FEATURES: dict[str, tuple[float, float]] = {
    "days_since_earnings": (float(EARNINGS_MAX_AGE_DAYS), 1.0),
}


def feature_order(feature_set: str = "v1") -> list[str]:
    """Canonical feature order of ``AlphaConfig.feature_set``."""
    try:
        return FEATURE_SETS[feature_set]
    except KeyError:
        raise ValueError(f"unknown feature_set {feature_set!r}; known: {sorted(FEATURE_SETS)}") from None


def feature_signature(columns) -> str:
    """Hash of feature names *and order* — used for train/live parity checks."""
    return hashlib.sha1("|".join(map(str, columns)).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- helpers
def _winsorize_rows(wide: Wide, pct: float) -> Wide:
    """Per-date winsorisation: :func:`qts.core.stats.winsorize_rows` (the single
    implementation; rows with < ``stats.WINSOR_MIN_OBS`` finite values unclipped)."""
    return stats.winsorize_rows(wide, pct)


def _trailing_return(close: Wide, start_lag: int, end_lag: int) -> Wide:
    """Return from t-start_lag to t-end_lag (both >= 0, start > end)."""
    return close.shift(end_lag) / close.shift(start_lag) - 1.0


def rolling_market_beta(returns: Wide, market: pd.Series, window: int, min_periods: int | None = None):
    """Rolling OLS beta of each column on ``market`` plus residual vol, data ``<= t``
    only: :func:`qts.core.stats.rolling_beta` with ``min_periods`` defaulting to
    ``0.8 * window`` (the alpha-feature convention).  Returns ``(beta, resid_vol)``."""
    rb = stats.rolling_beta(returns, market, window, min_periods or int(window * 0.8))
    return rb.beta, rb.resid_vol


def industry_segments(md: MarketData) -> list[tuple[pd.DatetimeIndex, pd.Series]]:
    """Point-in-time industry classification as ``[(dates, labels), ...]``:
    contiguous date blocks over which ``md.industry_as_of(d)`` is constant.
    One block (the static map) when ``md.industry_history`` is None."""
    dates = md.dates
    if getattr(md, "industry_history", None) is None:
        return [(dates, md.industry.reindex(md.tickers))]
    hist = md.industry_history
    # a row dated h takes effect on the first trading date >= h
    starts = {0} | {int(dates.searchsorted(pd.Timestamp(h))) for h in hist.index}
    out: list[tuple[pd.DatetimeIndex, pd.Series]] = []
    bounds = [*sorted(b for b in starts if b < len(dates)), len(dates)]
    for i in range(len(bounds) - 1):
        if bounds[i + 1] > bounds[i]:
            seg = dates[bounds[i] : bounds[i + 1]]
            out.append((seg, md.industry_as_of(seg[0])))
    return out


def _by_industry(fn, panel: Wide, segments) -> Wide:
    """Apply a group operation ``fn(sub_panel, labels)`` per PIT industry segment."""
    if len(segments) == 1:
        return fn(panel, segments[0][1])
    parts = [fn(panel.loc[d], lab) for d, lab in segments]
    return pd.concat(parts).reindex(panel.index)


def cap_weighted_group_mean(values: Wide, weights: Wide, groups: pd.Series) -> Wide:
    """Per-date cap-weighted mean of ``values`` within each group, broadcast back
    to every member of the group."""
    g = groups.reindex(values.columns)
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns)
    w = weights.where(values.notna())
    for cols in g.groupby(g).groups.values():
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
    def build(self, md: MarketData) -> dict[str, Wide]:
        close = md.close
        rets = md.returns
        mcap = md.market_cap
        dvol = md.dollar_volume
        segs = industry_segments(md)  # PIT industry labels (md.industry_as_of)
        valid = md.eligibility()  # in-universe & alive, per date (PIT)
        f: dict[str, Wide] = {}

        # value
        if md.book_to_price is not None:
            f["value_bp"] = np.log(md.book_to_price.clip(lower=1e-6))
        if md.earnings_yield is not None:
            f["value_ep"] = md.earnings_yield.copy()

        # momentum
        f["mom_12_1"] = _trailing_return(close, 252, 21)
        f["mom_6_1"] = _trailing_return(close, 126, 21)
        f["mom_change"] = f["mom_12_1"] - f["mom_12_1"].shift(63)
        # cross-sectional statistic: cap weights of names eligible at t only (the
        # level panels carry full history for every listed name, audit 2 #6)
        mcap_elig = mcap.where(valid)
        f["industry_momentum"] = _by_industry(
            lambda v, lab: cap_weighted_group_mean(v, mcap_elig.loc[v.index], lab), f["mom_6_1"], segs
        )

        # quality / investment
        if md.gross_profitability is not None:
            f["quality_gpa"] = md.gross_profitability.copy()
        if md.asset_growth is not None:
            f["investment_ag"] = -md.asset_growth

        # low risk (market proxy = equal-weight mean over the names eligible at t)
        market = rets.where(valid).mean(axis=1)
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

        # short-term reversal: -(21d return, industry-demeaned).  A risk-only style
        # proxy (qts.core.roles: label residualisation / optimizer neutralisation);
        # as a feature it is an ordinary ML input -- it is not used for trade timing.
        r21 = _trailing_return(close, 21, 0).where(valid)  # demean within eligible names only
        f["strev_21"] = -_by_industry(demean_by_group, r21, segs)

        if self.cfg.feature_set == "v2":
            # earnings reaction, abnormal volume, accruals / issuance, 52-week high
            # (qts.alpha.features_v2; raises FeatureSetDataError without the earnings tables)
            f.update(build_v2_features(md, market))

        ordered = {k: f[k] for k in feature_order(self.cfg.feature_set) if k in f}
        for v in ordered.values():
            v.index.name = "date"
            v.columns.name = "ticker"
        return ordered

    # ------------------------------------------------------------------ processing
    def process(self, features: dict[str, Wide], md: MarketData, valid: Wide | None = None) -> Long:
        """Per-date, over eligible names only: winsorize -> industry demean
        (cfg.industry_demean; under ``feature_set = "v2"`` also the seven ranked v2
        features) -> rank_gauss -> fill 0; a conditioning variable
        (``CONDITIONING_FEATURES``: ``days_since_earnings``) is only scaled to
        [0, 1] and filled with its cap.  Returns a long frame
        indexed by (date, ticker) that contains **only** rows eligible at that
        date (``valid`` defaults to ``md.eligibility()``).  Ineligible names are
        dropped, not zero-filled, so they never enter a cross-sectional
        statistic or receive a score."""
        valid = md.eligibility() if valid is None else valid.astype(bool)
        cols = []
        order = feature_order(self.cfg.feature_set)
        demean_set = set(self.cfg.industry_demean)
        if self.cfg.feature_set == "v2":
            demean_set |= V2_INDUSTRY_DEMEAN
        segs = industry_segments(md)
        for name in order:
            if name not in features:
                continue
            panel = features[name].where(valid.reindex_like(features[name]).fillna(False).astype(bool))
            if name in CONDITIONING_FEATURES:
                divisor, fill = CONDITIONING_FEATURES[name]
                cols.append(to_long((panel / divisor).fillna(fill), name))
                continue
            w = _winsorize_rows(panel, self.cfg.winsor_pct)
            if name in demean_set:
                w = _by_industry(demean_by_group, w, segs)
            w = rank_gauss(w)
            cols.append(to_long(w, name))
        out = pd.concat(cols, axis=1)
        keep = to_long(valid, "eligible").reindex(out.index).fillna(False).astype(bool)
        out = out.loc[keep.to_numpy()].fillna(0.0)
        out = out.reindex(columns=[c for c in order if c in out.columns])
        return out

    def build_and_process(self, md: MarketData) -> Long:
        return self.process(self.build(md), md)
