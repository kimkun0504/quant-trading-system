"""Synthetic-data audits for high-capacity models (Nagel 2025 critique).

Nagel (NBER w34104, 2025) shows that flexible return-prediction models fitted
to short panels tend to *collapse* onto a simple implicit strategy — their
predictions are close to a recency-weighted average of past returns (a
momentum/reversal proxy) — and that in-sample "skill" on real data is often
this implicit strategy plus noise.  The recommended audit is to run the model
on synthetic data with **known** dynamics: on pure noise the model should
show IC ~ 0 and its predictions should *not* load strongly on a recency
weighted mean of past returns.

:func:`nagel_collapse_test` builds such a panel (i.i.d. in the mean, with
GARCH-style volatility clustering and a common vol factor so it looks like
real data to a second-moment model, but with **no** serial correlation in
the mean), fits the caller's model on the first half and evaluates on the
second half.

``model_predict_fn(X_train, y_train, X_test) -> np.ndarray`` where ``X`` is
``(n_samples, n_lags)`` of lagged returns (lag 1 first) and ``y`` is the
forward ``horizon``-day return.  A scikit-learn estimator can be wrapped as
``lambda Xtr, ytr, Xte: est.fit(Xtr, ytr).predict(Xte)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from qts.core.stats import rank_ic_series as ic_series


def generate_noise_panel(n_assets: int, n_days: int, seed: int = 0, vol_annual: float = 0.30) -> pd.DataFrame:
    """Zero-mean returns with vol clustering, no momentum, no cross-sectional signal."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n_days)
    cols = [f"N{i:04d}" for i in range(n_assets)]
    vd = vol_annual / np.sqrt(252)
    # common GARCH(1,1) vol factor + idiosyncratic GARCH
    h_c = np.full(n_days, 1.0)
    z_c = rng.standard_normal(n_days)
    for t in range(1, n_days):
        h_c[t] = 0.05 + 0.10 * (z_c[t - 1] ** 2) * h_c[t - 1] + 0.85 * h_c[t - 1]
    scale_i = np.exp(0.3 * rng.standard_normal(n_assets))
    eps = rng.standard_normal((n_days, n_assets))
    h = np.ones((n_days, n_assets))
    for t in range(1, n_days):
        h[t] = 0.05 + 0.08 * (eps[t - 1] ** 2) * h[t - 1] + 0.87 * h[t - 1]
    r = vd * scale_i[None, :] * np.sqrt(h) * np.sqrt(h_c)[:, None] * eps
    return pd.DataFrame(r, index=dates, columns=cols)


def lagged_design(returns: pd.DataFrame, n_lags: int, horizon: int):
    """Long design: rows = (date, asset), X = returns at lags 1..n_lags, y = forward horizon return."""
    R = returns.to_numpy()
    T, N = R.shape
    fwd = (np.exp(np.log1p(returns).rolling(horizon).sum().shift(-horizon)) - 1).to_numpy()
    rows_t, Xs, ys = [], [], []
    for t in range(n_lags, T - horizon):
        Xs.append(R[t - n_lags + 1 : t + 1][::-1].T)  # N x n_lags, lag1 first
        ys.append(fwd[t])
        rows_t.append(np.full(N, t))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys)
    t_idx = np.concatenate(rows_t)
    a_idx = np.tile(np.arange(N), len(Xs))
    return X, y, t_idx, a_idx


def recency_weighted_mean(X_lags: np.ndarray, half_life: float = 10.0) -> np.ndarray:
    """Exponentially weighted mean of lagged returns (lag 1 heaviest)."""
    k = X_lags.shape[1]
    w = 0.5 ** (np.arange(k) / half_life)
    w = w / w.sum()
    return X_lags @ w


def nagel_collapse_test(
    model_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_assets: int = 100,
    n_days: int = 1000,
    seed: int = 0,
    n_lags: int = 60,
    horizon: int = 20,
    half_life: float = 10.0,
    max_abs_ic_t: float = 3.0,
    max_abs_corr: float = 0.5,
) -> dict[str, float]:
    """Fit on the first half of a noise panel, evaluate on the second half.

    Returns ``ic_mean``, ``ic_tstat``, ``corr_with_recency_mean`` (Pearson
    correlation between predictions and the recency-weighted past mean,
    pooled over the test set), ``corr_with_recency_mean_xs`` (average
    cross-sectional rank correlation), and ``passed`` (|IC t| < 3 and
    |corr| < 0.5).  A model that fails ``corr`` has collapsed onto a
    momentum/reversal rule; one that "passes" IC with a high corr has merely
    found a trivial strategy the noise panel does not reward.
    """
    ret = generate_noise_panel(n_assets, n_days, seed)
    X, y, t_idx, a_idx = lagged_design(ret, n_lags, horizon)
    T_cut = (n_lags + (n_days - horizon)) // 2
    tr, te = t_idx < T_cut, t_idx >= T_cut + horizon  # gap so train labels don't reach test
    pred = np.asarray(model_predict_fn(X[tr], y[tr], X[te]), dtype=float).ravel()
    if pred.shape[0] != te.sum():
        raise ValueError("model_predict_fn must return one prediction per test row")

    dates = ret.index

    def _wide(vals):
        arr = np.full(ret.shape, np.nan)
        arr[t_idx[te], a_idx[te]] = vals
        return pd.DataFrame(arr, index=dates, columns=ret.columns)

    pred_w = _wide(pred)
    y_w = _wide(y[te])
    ics = ic_series(pred_w.dropna(how="all"), y_w.dropna(how="all")).dropna()
    # thin to non-overlapping dates for the t-stat
    ics_no = ics.iloc[::horizon]
    ic_mean = float(ics.mean()) if len(ics) else float("nan")
    ic_t = (
        float(ics_no.mean() / ics_no.std() * np.sqrt(len(ics_no)))
        if len(ics_no) > 2 and ics_no.std() > 0
        else 0.0
    )

    rwm = recency_weighted_mean(X[te], half_life)
    sd_p = pred.std()
    corr = float(np.corrcoef(pred, rwm)[0, 1]) if sd_p > 0 and rwm.std() > 0 else 0.0
    rwm_w = _wide(rwm)
    corr_xs = ic_series(pred_w.dropna(how="all"), rwm_w.dropna(how="all")).dropna()
    corr_xs_mean = float(corr_xs.mean()) if len(corr_xs) else 0.0

    return {
        "ic_mean": ic_mean,
        "ic_tstat": ic_t,
        "n_ic_dates": len(ics),
        "corr_with_recency_mean": corr,
        "corr_with_recency_mean_xs": corr_xs_mean,
        "pred_sd": float(sd_p),
        "collapsed": bool(abs(corr) > max_abs_corr or abs(corr_xs_mean) > max_abs_corr),
        "passed": bool(
            abs(ic_t) < max_abs_ic_t and abs(corr) <= max_abs_corr and abs(corr_xs_mean) <= max_abs_corr
        ),
    }
