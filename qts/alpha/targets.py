"""Forward-return labels for the alpha model.

**Labels are training targets only.**  Nothing produced here may ever be used
as a feature, and nothing here is consumed by ``AlphaModel.as_of`` for a date
whose forward window has not fully elapsed (the walk-forward trainer enforces
the embargo; see ``trainer.py``).

Primary target (BUILD_PLAN §2.3): ``horizon``-day forward return residualised
cross-sectionally against risk exposures *known at t* — market beta, size,
industry dummies, momentum and value — then z-scored per date.

Until the Barra-style risk model is wired in, :func:`residualize_forward_returns`
uses a lightweight, dependency-free set of exposures computed from
``MarketData`` (``default_exposures``).  Pass ``exposures_fn`` (e.g.
``lambda d: risk_model.as_of(d).exposures``) to neutralise against a proper
risk model instead.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from qts.core.crosssection import neutralize, zscore
from qts.core.types import MarketData, Wide

from .features import _winsorize_rows, rolling_market_beta

ExposuresFn = Callable[[pd.Timestamp], pd.DataFrame]

DEFAULT_STYLE_EXPOSURES = ("beta_252", "log_mcap", "mom_12_1", "value_bp")


def forward_returns(md: MarketData, horizon: int) -> Wide:
    """``close_{t+h} / close_t - 1`` aligned at ``t``.  NaN for the last ``h`` rows."""
    return md.close.shift(-horizon) / md.close - 1.0


def default_exposures(md: MarketData, features: Optional[Dict[str, Wide]] = None) -> Dict[str, Wide]:
    """Style exposures (wide panels) used by the built-in residualiser.

    If raw feature panels from ``FeatureBuilder.build`` are supplied they are
    reused, otherwise the needed ones are recomputed here.  Each is
    cross-sectionally z-scored per date; NaN -> 0 (= cross-sectional mean) so
    that early dates (before a 252-day beta exists) still get a label.
    """
    f = dict(features or {})
    close = md.close
    if "beta_252" not in f:
        rets = md.returns
        f["beta_252"], _ = rolling_market_beta(rets, rets.mean(axis=1), 252)
    if "log_mcap" not in f:
        f["log_mcap"] = np.log(md.market_cap.clip(lower=1.0))
    if "mom_12_1" not in f:
        f["mom_12_1"] = close.shift(21) / close.shift(252) - 1.0
    if "value_bp" not in f and md.book_to_price is not None:
        f["value_bp"] = np.log(md.book_to_price.clip(lower=1e-6))
    out = {}
    for k in DEFAULT_STYLE_EXPOSURES:
        if k in f:
            out[k] = zscore(_winsorize_rows(f[k], 0.01)).fillna(0.0)
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
) -> Wide:
    """Per date: neutralise the ``horizon``-day forward return on exposures known
    at ``t`` (intercept + market beta + industry dummies + size + momentum +
    value by default; or ``exposures_fn(date)`` -> ticker x factor frame), then
    z-score cross-sectionally.

    ``weights`` (e.g. sqrt market cap) makes the neutralisation WLS.
    Rows (dates) whose forward window is not yet complete are NaN.
    """
    fwd = _winsorize_rows(forward_returns(md, horizon), winsor_pct)
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
    if True:
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
