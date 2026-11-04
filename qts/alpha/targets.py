"""Forward-return labels for the alpha model.

**Labels are training targets only.**  Nothing produced here may ever be used
as a feature, and nothing here is consumed by ``AlphaModel.as_of`` for a date
whose forward window has not fully elapsed (the walk-forward trainer enforces
the embargo; see ``trainer.py``).

Primary target (BUILD_PLAN §2.3): ``horizon``-day forward return residualised
cross-sectionally against **risk-only** exposures *known at t* — market beta,
size, idiosyncratic vol, liquidity, short-term reversal and industry dummies —
then z-scored per date.  The alpha themes (momentum, value, quality,
investment, short interest) are *never* residualised out: projecting them out
of the label would define the alpha away (audit finding 2).

Without a risk model, :func:`residualize_forward_returns` uses a lightweight
set of exposures computed from ``MarketData`` (``default_exposures``,
``DEFAULT_STYLE_EXPOSURES``).  Pass ``exposures_fn`` (e.g.
``lambda d: risk_model.exposures_as_of(d)``) to neutralise against a proper
risk model; dates for which it returns ``None`` (risk-model warm-up) fall back
to the same risk-only internal set, so the label definition is consistent.

Eligibility: labels exist only for names in ``md.eligibility()`` at ``t``;
cross-sectional winsorisation, regression and z-scoring use those names only.

Delistings: a name whose last close falls inside ``(t, t+h]`` gets the return to
its last close times ``(1 + delisting_return)`` (default -30 %, see
``AlphaConfig.delisting_return``) instead of NaN, so delisting outcomes reach
the training set and every IC/decile statistic.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from qts.core.crosssection import neutralize, zscore
from qts.core.types import MarketData, Wide

from .features import _winsorize_rows, rolling_market_beta

ExposuresFn = Callable[[pd.Timestamp], pd.DataFrame]

#: Risk-only styles used by the built-in residualiser.  Must never contain an
#: alpha theme (mom_12_1, value_bp, quality_gpa, investment_ag, short_interest).
DEFAULT_STYLE_EXPOSURES = ("beta_252", "log_mcap", "idio_vol_252", "log_dollar_vol_63", "turnover_63", "strev_21")
ALPHA_THEME_FEATURES = frozenset({"value_bp", "value_ep", "mom_12_1", "mom_6_1", "mom_change", "industry_momentum",
                                  "quality_gpa", "investment_ag", "short_interest", "borrow_fee"})
assert not (set(DEFAULT_STYLE_EXPOSURES) & ALPHA_THEME_FEATURES)

DEFAULT_DELISTING_RETURN = -0.30


def forward_returns(md: MarketData, horizon: int, delisting_return: Optional[float] = DEFAULT_DELISTING_RETURN) -> Wide:
    """``close_{t+h} / close_t - 1`` aligned at ``t``.

    Delisting handling: if the name's *last ever* close is inside ``(t, t+h]``
    the return is measured to that last close and multiplied by
    ``(1 + delisting_return)`` (conservative CRSP-style haircut, default -30 %;
    ``None``/0 disables).  A temporary gap in the price series is not a
    delisting (there is a later close) and is simply forward-filled inside the
    window.  NaN where ``close_t`` is NaN or the window extends past the panel.
    ``close.ffill()`` is a *label-only* helper here — never reuse it for features.
    """
    c = md.close
    last = c.ffill()
    fwd = last.shift(-horizon) / c - 1.0
    if delisting_return:
        # dead at date d: no close at d, had one before, and none after (true delisting)
        dead = c.isna() & last.notna() & c.bfill().isna()
        dead_by_end = dead.shift(-horizon).fillna(False).astype(bool)   # dead at t+h  <=> delisted in (t, t+h]
        fwd = fwd.where(~dead_by_end, (1.0 + fwd) * (1.0 + float(delisting_return)) - 1.0)
    return fwd.where(c.notna())


def default_exposures(md: MarketData, features: Optional[Dict[str, Wide]] = None) -> Dict[str, Wide]:
    """Style exposures (wide panels) used by the built-in residualiser.

    Risk-only (``DEFAULT_STYLE_EXPOSURES``): beta, size, idio vol, dollar
    volume, turnover, short-term reversal.  If raw feature panels from
    ``FeatureBuilder.build`` are supplied they are reused, otherwise the needed
    ones are recomputed here.  Each is cross-sectionally z-scored per date over
    the eligible names; NaN -> 0 (= cross-sectional mean) so that early dates
    (before a 252-day beta exists) still get a label.
    """
    f = dict(features or {})
    close = md.close
    valid = md.eligibility()
    rets = md.returns
    if "beta_252" not in f or "idio_vol_252" not in f:
        beta, idio = rolling_market_beta(rets, rets.where(valid).mean(axis=1), 252)
        f.setdefault("beta_252", beta)
        f.setdefault("idio_vol_252", idio)
    if "log_mcap" not in f:
        f["log_mcap"] = np.log(md.market_cap.clip(lower=1.0))
    if "log_dollar_vol_63" not in f or "turnover_63" not in f:
        dvol = md.dollar_volume
        f.setdefault("log_dollar_vol_63", np.log(dvol.rolling(63, min_periods=40).mean().clip(lower=1.0)))
        f.setdefault("turnover_63", (dvol / md.market_cap).rolling(63, min_periods=40).mean())
    if "strev_21" not in f:
        f["strev_21"] = -(close / close.shift(21) - 1.0)
    out = {}
    for k in DEFAULT_STYLE_EXPOSURES:
        if k in f:
            assert k not in ALPHA_THEME_FEATURES, k
            out[k] = zscore(_winsorize_rows(f[k].where(valid), 0.01)).fillna(0.0)
    return out


def industry_dummies(industry: pd.Series, tickers: pd.Index) -> pd.DataFrame:
    """Ticker x (n_industries - 1) dummy matrix (one dropped: intercept present)."""
    d = pd.get_dummies(industry.reindex(tickers), dtype=float)
    return d.iloc[:, 1:] if d.shape[1] > 1 else d.iloc[:, :0]


def residualize_forward_returns(
    md: MarketData,
    horizon: int,
    exposures_fn: Optional[ExposuresFn] = None,
    features: Optional[Dict[str, Wide]] = None,
    winsor_pct: float = 0.01,
    weights: Optional[Wide] = None,
    delisting_return: Optional[float] = DEFAULT_DELISTING_RETURN,
) -> Wide:
    """Per date: neutralise the ``horizon``-day forward return on exposures known
    at ``t`` (intercept + industry dummies + the risk-only styles in
    ``DEFAULT_STYLE_EXPOSURES`` by default; or ``exposures_fn(date)`` -> ticker x
    factor frame), then z-score cross-sectionally.  Only names eligible at ``t``
    (``md.eligibility()``) take part; everything else is NaN.

    ``weights`` (e.g. sqrt market cap) makes the neutralisation WLS.
    Rows (dates) whose forward window is not yet complete are NaN.
    """
    valid = md.eligibility()
    fwd = _winsorize_rows(forward_returns(md, horizon, delisting_return).where(valid), winsor_pct)
    tickers = md.tickers
    dates = md.dates
    out = np.full(fwd.shape, np.nan)
    valid_rows = (fwd.notna().sum(axis=1) >= 20).to_numpy()
    Y = fwd.to_numpy(dtype=float)
    W = None if weights is None else weights.reindex(index=dates, columns=tickers).to_numpy(dtype=float)
    # Always build the internal (fast) exposures: they are the fallback for dates
    # where an external exposures_fn has no model yet (e.g. risk-model warm-up).
    styles = default_exposures(md, features)
    D = industry_dummies(md.industry, tickers).to_numpy(dtype=float)
    S = np.stack([styles[k].reindex(index=dates, columns=tickers).to_numpy(dtype=float)
                  for k in styles], axis=2) if styles else np.zeros((len(dates), len(tickers), 0))
    ones = np.ones((len(tickers), 1))
    rows = np.flatnonzero(valid_rows)
    external = {}
    if exposures_fn is not None:
        for i in rows:
            X = exposures_fn(dates[i])
            if X is not None and len(X):
                external[i] = X
    if exposures_fn is None or not external:
        internal_rows = rows
    else:
        internal_rows = np.array([i for i in rows if i not in external], dtype=int)
    if len(internal_rows):
        # Fast path: static industry dummies + per-date style arrays, pure numpy.
        for i in internal_rows:
            y = Y[i]
            m = np.isfinite(y)
            X = np.hstack([ones, D, S[i]])[m]
            yy = y[m]
            if W is not None:
                sw = np.sqrt(np.nan_to_num(W[i][m], nan=np.nanmedian(W[i])).clip(1e-12))
                X, yy = X * sw[:, None], yy * sw
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            out[i, m] = y[m] - np.hstack([ones, D, S[i]])[m] @ beta
    if external:
        # Hook: external (risk-model) exposures per date via crosssection.neutralize.
        # Drop constant/collinear columns (e.g. the unit "market" column duplicates the intercept).
        for i, X in external.items():
            d = dates[i]
            X = X.reindex(tickers).fillna(0.0)
            X = X.loc[:, X.std(axis=0) > 1e-12]
            w = None if weights is None else weights.loc[d]
            out[i] = neutralize(fwd.iloc[i], X, w).values
    out = zscore(pd.DataFrame(out, index=dates, columns=tickers))
    out.index.name, out.columns.name = "date", "ticker"
    return out
