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

Label timing with an execution lag (``L = ExecutionConfig.signal_lag_days``)
-----------------------------------------------------------------------------
A signal formed at the close of ``s`` is traded at the close of ``s + L`` and
earns from there.  The label of decision date ``s`` is therefore the return
``close_{s+L} -> close_{s+L+h}`` (``label_lag_days=L``), aligned at ``s``; with
``L = 0`` it is the classic ``close_s -> close_{s+h}``.  The label of ``s`` is
realised at the close of ``s + L + h`` (or later, see below), so every consumer
(walk-forward purge, trailing IC) must wait ``h + L`` days, not ``h``.

Delistings vs halts (causal rule; audit 2 finding 1)
----------------------------------------------------
When the name has no close at the window end ``e = s + L + h`` the label is the
return to its last close ``lp`` before ``e``, and it is a *delisting* (times
``1 + delisting_return``, default -30 %, ``CostConfig.delisting_haircut``) or a
*temporary halt* (no haircut).  That decision never uses prices after the date
at which it is made:

* ``md.delisting_dates`` covers the name -> delisted iff its last trading date
  is ``<= date(e)``; decided at ``e``.
* otherwise (causal gap rule, ``delisting_gap_days = G``, default 63) ->
  a halt if a close reappears within ``(lp, lp + G]`` (decided on the day it
  reappears), a delisting if none does (decided at ``max(e, lp + G)``).  While
  neither has happened the cell is *unresolved*: NaN.

:func:`label_realisation` gives, per decision date ``s``, the first date at
which every label of that row (all eligible names) is decided; a row is
consumed (training, trailing IC) only once that date is ``<=`` the consumer's
date.  This makes the residualised labels a consumer sees at ``t`` -- and hence
the ML fit, ``oos_ic_``, ``ic_used`` and ``expected_return`` at ``t`` --
identical whether the data end at ``t`` or later (``tests/test_label_halts.py``).
Without a delisting table a delisting delays its rows by up to ``G`` days; supply
``delisting_dates`` for real data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from qts.core import stats
from qts.core.config import AlphaConfig, CostConfig
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
    "label_realisation",
    "residualize_forward_returns",
]

#: Risk-only style proxies used by the built-in residualiser (D1: the alpha-layer
#: features proxying ``RISK_ONLY_STYLES``).  Never contains an alpha theme;
#: both sets are defined once in ``qts.core.roles``.
DEFAULT_STYLE_EXPOSURES = RISK_ONLY_FEATURES

#: default label delisting haircut = ``CostConfig.delisting_haircut`` (single source)
DEFAULT_DELISTING_RETURN = CostConfig().delisting_haircut
#: default causal gap rule = ``AlphaConfig.delisting_gap_days`` (single source)
DEFAULT_DELISTING_GAP_DAYS = AlphaConfig().delisting_gap_days


def _delist_pos(md: MarketData) -> np.ndarray:
    """Per ticker: position of the known last trading date (``inf`` = known
    alive, ``nan`` = not covered -> causal rule); :meth:`MarketData.delisting_positions`."""
    return md.delisting_positions()


def _forward_returns_resolved(
    md: MarketData,
    horizon: int,
    delisting_return: float | None,
    label_lag_days: int,
    delisting_gap_days: int,
) -> tuple[Wide, np.ndarray]:
    """Labels plus the position at which each cell's value is decided.

    Returns ``(fwd, ready)``: ``fwd`` as in :func:`forward_returns`; ``ready``
    (T x N float) is the index position at which the cell is decided (``inf`` =
    not decidable within the panel; the cell is NaN then).  Label-only helper.
    """
    L, h, G = int(label_lag_days), int(horizon), int(delisting_gap_days)
    if L < 0 or h < 1 or G < 1:
        raise ValueError("need label_lag_days >= 0, horizon >= 1, delisting_gap_days >= 1")
    c = md.close
    T, N = c.shape
    have = c.notna().to_numpy()
    posmat = np.where(have, np.arange(T, dtype=float)[:, None], np.nan)
    pos_df = pd.DataFrame(posmat)
    last_pos = pos_df.ffill().to_numpy()  # last close position <= d
    next_pos = pos_df.bfill().to_numpy()  # first close position >= d (label-only)
    k = L + h
    rows = np.arange(T)
    e = rows + k  # window end per row
    inside = e < T
    ee = np.minimum(e, T - 1)
    entry = c.shift(-L).to_numpy(dtype=float)  # close at s+L (execution close)
    exit_px = c.ffill().to_numpy(dtype=float)[ee]  # last close <= e (label-only ffill)
    fwd = exit_px / entry - 1.0
    fwd[~inside] = np.nan
    fwd[~have] = np.nan  # no close at s: no label
    ready = np.where(inside[:, None], e[:, None].astype(float), np.inf) * np.ones((1, N))
    gap = inside[:, None] & ~have[ee] & np.isfinite(fwd)  # no close at the window end
    dead = np.zeros((T, N), dtype=bool)
    if gap.any():
        lp = last_pos[ee]  # last close before e (>= s+L since the entry close exists)
        nv = np.full((T, N), np.inf)
        nxt = next_pos[ee]
        nv = np.where(np.isfinite(nxt), nxt, np.inf)
        dpos = _delist_pos(md)[None, :] * np.ones((T, 1))
        covered = ~np.isnan(dpos)
        # vendor delisting dates: decided at e
        known_dead = gap & covered & (dpos <= e[:, None])
        # causal rule: halt if a close reappears within G days of lp, else delisted at lp + G
        rule = gap & ~covered
        halt = rule & (nv <= lp + G)
        rule_dead = rule & ~halt & (lp + G <= T - 1)
        unresolved = rule & ~halt & ~rule_dead
        ready = np.where(halt, nv, ready)
        ready = np.where(rule_dead, np.maximum(e[:, None], lp + G), ready)
        ready = np.where(unresolved, np.inf, ready)
        dead = known_dead | rule_dead
        fwd = np.where(unresolved, np.nan, fwd)
    if delisting_return:
        fwd = np.where(dead, (1.0 + fwd) * (1.0 + float(delisting_return)) - 1.0, fwd)
    ready = np.where(np.isnan(fwd), np.inf, ready)
    out = pd.DataFrame(fwd, index=c.index, columns=c.columns)
    return out, ready


def forward_returns(
    md: MarketData,
    horizon: int,
    delisting_return: float | None = DEFAULT_DELISTING_RETURN,
    label_lag_days: int = 0,
    delisting_gap_days: int = DEFAULT_DELISTING_GAP_DAYS,
) -> Wide:
    """``close_{t+L+h} / close_{t+L} - 1`` aligned at ``t`` (``L = label_lag_days``,
    the execution lag; ``L = 0`` -> ``close_{t+h} / close_t - 1``).

    If the name has no close at the window end the return runs to its last close
    before it; a *delisting* (see the module docstring: vendor
    ``md.delisting_dates`` or the causal ``delisting_gap_days`` rule, never a
    look at later prices) is multiplied by ``(1 + delisting_return)``
    (conservative CRSP-style haircut, default -30 %; ``None``/0 disables), a
    halt is not.  NaN where ``close_t`` or ``close_{t+L}`` is NaN, the window
    extends past the panel, or the halt/delisting question is not yet decided
    within the panel.  ``close.ffill()`` is a *label-only* helper here -- never
    reuse it for features.
    """
    fwd, _ = _forward_returns_resolved(md, horizon, delisting_return, label_lag_days, delisting_gap_days)
    return fwd


def label_realisation(
    md: MarketData,
    horizon: int,
    label_lag_days: int = 0,
    delisting_gap_days: int = DEFAULT_DELISTING_GAP_DAYS,
    valid: Wide | None = None,
) -> pd.Series:
    """Per decision date ``s``: the date at which the whole label row is decided
    (``NaT`` = not within the panel).

    The row's residualised labels depend on every eligible (``valid``, default
    ``md.eligibility()``) name's forward return, so the row is ready at the
    latest decision date of its cells (at least ``s + L + h``).  Consumers at
    date ``t`` may use row ``s`` only if ``label_realisation(...)[s] <= t``.
    """
    _, ready = _forward_returns_resolved(md, horizon, 0.0, label_lag_days, delisting_gap_days)
    valid = md.eligibility() if valid is None else valid
    v = valid.reindex(index=md.dates, columns=md.tickers).fillna(False).to_numpy(dtype=bool)
    T = len(md.dates)
    L, h = int(label_lag_days), int(horizon)
    c = md.close.to_numpy(dtype=float)
    entry_ok = np.zeros_like(v)
    if L < T:
        entry_ok[: T - L] = np.isfinite(c[L:])
    relevant = v & entry_ok  # cells that enter the cross-section of row s
    nominal = np.arange(T, dtype=float) + L + h
    nominal[nominal >= T] = np.inf
    r = np.where(relevant, ready, -np.inf).max(axis=1)
    r = np.maximum(r, nominal)
    out = pd.Series(pd.NaT, index=md.dates, dtype="datetime64[ns]", name="label_ready")
    ok = np.isfinite(r)
    out.iloc[np.flatnonzero(ok)] = md.dates[r[ok].astype(int)]
    return out


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


def residualize_forward_returns(
    md: MarketData,
    horizon: int,
    exposures_fn: ExposuresFn | None = None,
    features: dict[str, Wide] | None = None,
    winsor_pct: float = 0.01,
    weights: Wide | None = None,
    delisting_return: float | None = DEFAULT_DELISTING_RETURN,
    only_dates: pd.DatetimeIndex | None = None,
    label_lag_days: int = 0,
    delisting_gap_days: int = DEFAULT_DELISTING_GAP_DAYS,
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
    ``label_lag_days`` / ``delisting_gap_days``: see :func:`forward_returns`;
    rows are usable at :func:`label_realisation` (module docstring).
    """
    valid = md.eligibility()
    raw = forward_returns(md, horizon, delisting_return, label_lag_days, delisting_gap_days)
    fwd = _winsorize_rows(raw.where(valid), winsor_pct)
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
        Dm = stats.industry_dummies(labels, tickers, drop_first=True).to_numpy(dtype=float)
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
