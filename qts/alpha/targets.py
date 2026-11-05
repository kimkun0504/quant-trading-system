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

Design decision D1 (docs/notes/doc_review_findings.md): **the label is
residualised on exactly the set of styles the optimizer neutralises** —
``qts.core.roles.RISK_ONLY_STYLES`` (beta, size, residual_vol, liquidity,
short_term_reversal) plus market and industries.  Predicting a premium the
book is then forced to hedge away wastes model capacity, so the linear premium
of the low-risk theme (beta / residual vol) is *intentionally* removed from
the ML target even though the rank composite still carries a ``low_risk``
theme.  ``qts.core.roles`` is the single source of truth for both this module
and the pipeline/optimizer; nothing here keeps a local copy.

Without a risk model, :func:`residualize_forward_returns` uses a lightweight
set of exposures computed from ``MarketData`` (``default_exposures``,
``DEFAULT_STYLE_EXPOSURES`` = ``roles.RISK_ONLY_FEATURES``, the alpha-layer
proxies of the risk-only styles).  Pass ``exposures_fn`` (e.g.
``lambda d: risk_model.exposures_as_of(d)``) to neutralise against a proper
risk model; dates for which it returns ``None`` (risk-model warm-up) fall back
to the same risk-only internal set, so the label definition is consistent.

Eligibility: labels exist only for names in ``md.eligibility()`` at ``t``;
cross-sectional winsorisation, regression and z-scoring use those names only.

Delistings: a name whose last close falls inside ``(t, t+h]`` gets the return to
its last close times ``(1 + delisting_return)`` (default -30 %, see
``CostConfig.delisting_haircut``) instead of NaN, so delisting outcomes reach
the training set and every IC/decile statistic.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from qts.core import stats
from qts.core.config import CostConfig
from qts.core.crosssection import neutralize, zscore
from qts.core.roles import ALPHA_THEME_FEATURES, RISK_ONLY_FEATURES, RISK_ONLY_STYLES
from qts.core.types import MarketData, Wide

from .features import _winsorize_rows, industry_segments, rolling_market_beta

ExposuresFn = Callable[[pd.Timestamp], pd.DataFrame]

__all__ = [
    "ALPHA_THEME_FEATURES",
    "DEFAULT_STYLE_EXPOSURES",
    "RISK_ONLY_STYLES",
    "ExposuresFn",
    "default_exposures",
    "forward_returns",
    "industry_dummies",
    "residualize_forward_returns",
]

#: Risk-only style proxies used by the built-in residualiser (D1: the alpha-layer
#: features proxying ``RISK_ONLY_STYLES``).  Never contains an alpha theme;
#: both sets are defined once in ``qts.core.roles``.
DEFAULT_STYLE_EXPOSURES = RISK_ONLY_FEATURES

#: default label delisting haircut = ``CostConfig.delisting_haircut`` (single source)
DEFAULT_DELISTING_RETURN = CostConfig().delisting_haircut


def forward_returns(
    md: MarketData, horizon: int, delisting_return: float | None = DEFAULT_DELISTING_RETURN
) -> Wide:
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
        dead_by_end = dead.shift(-horizon).fillna(False).astype(bool)  # dead at t+h  <=> delisted in (t, t+h]
        fwd = fwd.where(~dead_by_end, (1.0 + fwd) * (1.0 + float(delisting_return)) - 1.0)
    return fwd.where(c.notna())


def default_exposures(md: MarketData, features: dict[str, Wide] | None = None) -> dict[str, Wide]:
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
    """Ticker x (n_industries - 1) dummy matrix (first industry dropped: an
    intercept is present) -- :func:`qts.core.stats.industry_dummies`."""
    return stats.industry_dummies(industry, tickers, drop_first=True)


def residualize_forward_returns(
    md: MarketData,
    horizon: int,
    exposures_fn: ExposuresFn | None = None,
    features: dict[str, Wide] | None = None,
    winsor_pct: float = 0.01,
    weights: Wide | None = None,
    delisting_return: float | None = DEFAULT_DELISTING_RETURN,
    only_dates: pd.DatetimeIndex | None = None,
) -> Wide:
    """Per date: neutralise the ``horizon``-day forward return on exposures known
    at ``t`` (intercept + industry dummies + the risk-only styles in
    ``DEFAULT_STYLE_EXPOSURES`` by default; or ``exposures_fn(date)`` -> ticker x
    factor frame), then z-score cross-sectionally.  Only names eligible at ``t``
    (``md.eligibility()``) take part; everything else is NaN.

    ``weights`` (e.g. sqrt market cap) makes the neutralisation WLS.
    Rows (dates) whose forward window is not yet complete are NaN.
    ``only_dates`` restricts the (per-date) regressions to those rows — every
    other row is NaN; values on the computed rows are identical to the full
    run (used for cheap multi-horizon IC-decay diagnostics).
    """
    valid = md.eligibility()
    fwd = _winsorize_rows(forward_returns(md, horizon, delisting_return).where(valid), winsor_pct)
    tickers = md.tickers
    dates = md.dates
    out = np.full(fwd.shape, np.nan)
    valid_rows = (fwd.notna().sum(axis=1) >= 20).to_numpy().copy()
    if only_dates is not None:
        valid_rows &= dates.isin(pd.DatetimeIndex(only_dates))
    Y = fwd.to_numpy(dtype=float)
    W = None if weights is None else weights.reindex(index=dates, columns=tickers).to_numpy(dtype=float)
    # Always build the internal (fast) exposures: they are the fallback for dates
    # where an external exposures_fn has no model yet (e.g. risk-model warm-up).
    styles = default_exposures(md, features)
    # PIT industry dummies: one matrix per block of constant classification
    # (md.industry_as_of); a single static matrix without industry_history.
    D_of_row: dict[int, np.ndarray] = {}
    for seg_dates, labels in industry_segments(md):
        Dm = industry_dummies(labels, tickers).to_numpy(dtype=float)
        for i in dates.get_indexer(seg_dates):
            D_of_row[int(i)] = Dm
    S = (
        np.stack(
            [styles[k].reindex(index=dates, columns=tickers).to_numpy(dtype=float) for k in styles], axis=2
        )
        if styles
        else np.zeros((len(dates), len(tickers), 0))
    )
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
            D = D_of_row[int(i)]
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
