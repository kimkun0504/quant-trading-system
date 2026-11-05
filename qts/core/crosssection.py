"""Cross-sectional utilities used by alpha, risk and portfolio layers.

The winsorise / rank-gauss / rank-IC helpers here are thin wrappers kept for
backward compatibility; the single vectorised implementations live in
:mod:`qts.core.stats` -- new code should import from there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qts.core import stats as _stats


def winsorize(wide: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
    """Row-wise winsorisation; delegates to :func:`qts.core.stats.winsorize_rows`."""
    return _stats.winsorize_rows(wide, pct)


def zscore(wide: pd.DataFrame, weights: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row-wise z-score.  If ``weights`` (e.g. market cap) is given, the mean is
    weight-weighted while the sd is equal-weighted (Barra convention)."""
    if weights is None:
        mu = wide.mean(axis=1)
    else:
        w = weights.where(wide.notna()).fillna(0.0)
        w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
        mu = (wide.fillna(0.0) * w).sum(axis=1)
    sd = wide.sub(mu, axis=0).std(axis=1, ddof=0).replace(0, np.nan)
    return wide.sub(mu, axis=0).div(sd, axis=0)


def rank_gauss(wide: pd.DataFrame) -> pd.DataFrame:
    """Row-wise rank -> Gaussianised score ``Phi^{-1}((rank - 0.5) / n)``.

    Bounded by ``+-Phi^{-1}(1 - 0.5 / n)`` for a row with ``n`` non-NaN names
    (+-2.58 at n = 100), and never beyond ``+-Phi^{-1}(1 - 1e-4) ~= 3.72``
    because the percentile is clipped to ``[1e-4, 1 - 1e-4]``.  Delegates to
    :func:`qts.core.stats.rank_gauss`.
    """
    return _stats.rank_gauss(wide)


def demean_by_group(wide: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract the per-date mean within each group (e.g. industry)."""
    g = groups.reindex(wide.columns)
    out = wide.copy()
    for cols in g.groupby(g).groups.values():
        cols = [c for c in cols if c in wide.columns]
        sub = wide[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out


def neutralize(target: pd.Series, exposures: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
    """Residualise a cross-section ``target`` on ``exposures`` (ticker x factor) by WLS."""
    idx = target.dropna().index.intersection(exposures.dropna().index)
    y = target.loc[idx].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(idx)), exposures.loc[idx].to_numpy(dtype=np.float64)])
    w: NDArray[np.float64]
    if weights is None:
        w = np.ones(len(idx))
    else:
        w = weights.reindex(idx).fillna(weights.median()).clip(lower=1e-12).to_numpy(dtype=np.float64)
    sw: NDArray[np.float64] = np.sqrt(w)
    lhs: NDArray[np.float64] = X * sw[:, None]
    rhs: NDArray[np.float64] = y * sw
    beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    res = pd.Series(y - X @ beta, index=idx)
    return res.reindex(target.index)


def rank_ic(pred: pd.Series, realized: pd.Series) -> float:
    """Spearman correlation of one cross-section (NaN below 20 pairs)."""
    p = pred.to_frame().T
    r = realized.to_frame().T
    r.index = p.index
    return float(_stats.rank_ic_series(p, r).iloc[0])


def ic_series(pred_wide: pd.DataFrame, realized_wide: pd.DataFrame) -> pd.Series:
    """Per-date rank IC between a wide prediction panel and realized labels.

    Delegates to the vectorised :func:`qts.core.stats.rank_ic_series`
    (dates sorted, unnamed Series for backward compatibility).
    """
    out = _stats.rank_ic_series(pred_wide, realized_wide).sort_index()
    out.name = None
    return out
