"""Cross-sectional utilities used by alpha, risk and portfolio layers."""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_row(row: pd.Series, pct: float = 0.01) -> pd.Series:
    x = row.dropna()
    if len(x) < 10:
        return row
    lo, hi = x.quantile(pct), x.quantile(1 - pct)
    return row.clip(lo, hi)


def winsorize(wide: pd.DataFrame, pct: float = 0.01) -> pd.DataFrame:
    return wide.apply(lambda r: winsorize_row(r, pct), axis=1)


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
    """Row-wise rank -> gaussianised score in (-3, 3)."""
    from scipy.stats import norm
    r = wide.rank(axis=1, pct=True)
    n = wide.notna().sum(axis=1)
    r = r.mul(n, axis=0).sub(0.5).div(n, axis=0)
    return pd.DataFrame(norm.ppf(r.clip(1e-4, 1 - 1e-4)), index=wide.index, columns=wide.columns).where(wide.notna())


def demean_by_group(wide: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract the per-date mean within each group (e.g. industry)."""
    g = groups.reindex(wide.columns)
    out = wide.copy()
    for key, cols in g.groupby(g).groups.items():
        cols = [c for c in cols if c in wide.columns]
        sub = wide[cols]
        out[cols] = sub.sub(sub.mean(axis=1), axis=0)
    return out


def neutralize(target: pd.Series, exposures: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
    """Residualise a cross-section ``target`` on ``exposures`` (ticker x factor) by WLS."""
    idx = target.dropna().index.intersection(exposures.dropna().index)
    y = target.loc[idx].values
    X = np.column_stack([np.ones(len(idx)), exposures.loc[idx].values])
    if weights is None:
        w = np.ones(len(idx))
    else:
        w = weights.reindex(idx).fillna(weights.median()).clip(lower=1e-12).values
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    res = pd.Series(y - X @ beta, index=idx)
    return res.reindex(target.index)


def rank_ic(pred: pd.Series, realized: pd.Series) -> float:
    idx = pred.dropna().index.intersection(realized.dropna().index)
    if len(idx) < 20:
        return np.nan
    return float(pd.Series(pred.loc[idx]).rank().corr(pd.Series(realized.loc[idx]).rank()))


def ic_series(pred_wide: pd.DataFrame, realized_wide: pd.DataFrame) -> pd.Series:
    """Per-date rank IC between a wide prediction panel and realized labels."""
    out = {}
    for d in pred_wide.index.intersection(realized_wide.index):
        out[d] = rank_ic(pred_wide.loc[d], realized_wide.loc[d])
    return pd.Series(out).sort_index()
