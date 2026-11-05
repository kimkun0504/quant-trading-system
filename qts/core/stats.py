"""Shared, vectorised statistics used by the alpha, risk, portfolio, backtest
and validation layers.

One implementation per concept (WORK_PLAN 1.3 / docs/audit/code_quality.md
"Duplication across layers").  Every function here is pure, works on wide
panels (date x ticker) or 1-D arrays, and is checked against a brute-force
reference in ``tests/test_core_stats.py``.

Contents
--------
* :func:`winsorize_rows` -- per-date (row-wise) quantile clipping.
* :func:`rank_gauss` -- per-date rank -> standard-normal score.
* :func:`row_corr`, :func:`rank_ic_series` -- per-date Pearson / Spearman
  correlation of two wide panels.
* :func:`newey_west_se`, :func:`newey_west_tstat` -- HAC (Bartlett kernel)
  standard error / t-stat of a sample mean.
* :func:`rolling_beta` -- rolling OLS beta (and residual variance) of every
  column on one market series.
* :func:`industry_dummies` -- ticker x industry 0/1 matrix.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]
ArrayOrFrame = TypeVar("ArrayOrFrame", NDArray[Any], pd.DataFrame)
#: 1-D numeric input accepted by the Newey-West helpers.
Vector = NDArray[Any] | pd.Series | list[float]

#: Minimum number of finite observations in a cross-section for a winsorisation
#: to be applied (smaller cross-sections are returned unchanged).
WINSOR_MIN_OBS = 10
#: Minimum number of paired observations for a per-date (rank) correlation;
#: dates with fewer pairs get NaN.
IC_MIN_OBS = 20
#: Clip applied to the rank percentile in :func:`rank_gauss`.
RANK_GAUSS_CLIP = 1e-4


# ----------------------------------------------------------------------------- winsorise
def _row_quantiles_sorted(srt: FloatArray, n: NDArray[np.intp], q: float) -> FloatArray:
    """Per-row ``q``-quantile of a row-sorted array whose first ``n[i]`` entries of
    row ``i`` are its finite values (NaNs sorted last).

    Bit-for-bit the ``numpy.nanpercentile(..., method="linear")`` algorithm
    (numpy >= 1.22, Hyndman & Fan type 7): virtual index ``v = (n - 1) q``,
    neighbours ``floor(v)`` and ``floor(v) + 1`` (both clamped to the last
    value when ``v >= n - 1``, to the first when ``v < 0``), and numpy's
    two-sided lerp ``a + (b - a) g`` (``g < 0.5``) / ``b - (b - a)(1 - g)``
    (``g >= 0.5``) with ``g = v - floor(v)``.  Rows with ``n == 0`` give NaN.
    """
    v = (n - 1) * np.float64(q)
    prev = np.floor(v)
    gamma = v - prev
    prev_i = prev.astype(np.intp)
    next_i = prev_i + 1
    above = v >= n - 1
    prev_i[above] = n[above] - 1
    next_i[above] = n[above] - 1
    below = v < 0
    prev_i[below] = 0
    next_i[below] = 0
    rows = np.arange(int(srt.shape[0]))
    a = srt[rows, prev_i]
    b = srt[rows, next_i]
    diff = b - a
    out = a + diff * gamma
    hi = gamma >= 0.5
    out[hi] = b[hi] - diff[hi] * (1 - gamma[hi])
    return out


def winsorize_rows(x: ArrayOrFrame, pct: float, min_obs: int = WINSOR_MIN_OBS) -> ArrayOrFrame:
    """Clip each row to its own ``[pct, 1 - pct]`` quantiles.

    Quantiles are computed over the finite entries of the row with linear
    interpolation -- exactly ``numpy.nanpercentile(row, 100 * pct,
    method="linear")`` (identical to ``pandas.Series.quantile``), but from one
    row-wise ``np.sort`` instead of numpy's per-row Python loop (~25x faster
    on a 3,780 x 1,000 panel; ``tests/test_core_stats.py`` checks equality).
    Rows with fewer than ``min_obs`` finite entries are returned unchanged;
    non-finite entries (NaN, +-inf) become NaN.  ``pct <= 0`` returns the
    input untouched.  Accepts a 2-D ``ndarray`` or a ``DataFrame`` and returns
    the same type (index/columns preserved).
    """
    if pct <= 0:
        return x
    if not 0 < pct < 0.5:
        raise ValueError("pct must be in (0, 0.5)")
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("winsorize_rows expects a 2-D array (rows = dates)")
    finite = np.isfinite(arr)
    valid: NDArray[np.intp] = finite.sum(axis=1)
    work: FloatArray = np.where(finite, arr, np.nan)
    if work.shape[1] == 0:
        return x
    srt = np.sort(work, axis=1)  # NaNs last
    # nanpercentile divides the percentile by 100: reproduce q exactly
    q_lo = float(np.true_divide(100 * pct, 100))
    q_hi = float(np.true_divide(100 * (1 - pct), 100))
    lo = _row_quantiles_sorted(srt, valid, q_lo)
    hi = _row_quantiles_sorted(srt, valid, q_hi)
    lo = np.where(valid >= min_obs, lo, -np.inf)
    hi = np.where(valid >= min_obs, hi, np.inf)
    with np.errstate(invalid="ignore"):
        out = np.clip(work, lo[:, None], hi[:, None])
    out[~finite] = np.nan
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    return out


# ----------------------------------------------------------------------------- rank-gauss
def rank_gauss(wide: pd.DataFrame, clip: float = RANK_GAUSS_CLIP) -> pd.DataFrame:
    """Row-wise rank -> Gaussianised score ``Phi^{-1}((rank - 0.5) / n)``.

    ``rank`` is the average rank (ties share it) among the ``n`` non-NaN names
    of the row.  The percentile ``(rank - 0.5) / n`` lies in
    ``[0.5 / n, 1 - 0.5 / n]``, so the output is bounded by
    ``+-Phi^{-1}(1 - 0.5 / n)`` (e.g. +-2.58 for n = 100, +-3.29 for
    n = 1000), and the percentile is additionally clipped to
    ``[clip, 1 - clip]``, i.e. ``|score| <= Phi^{-1}(1 - 1e-4) ~= 3.72``
    (reached only for n >= 5000).  NaN inputs stay NaN.
    """
    r = wide.rank(axis=1, pct=True)
    n = wide.notna().sum(axis=1)
    r = r.mul(n, axis=0).sub(0.5).div(n, axis=0)
    z = norm.ppf(r.clip(clip, 1 - clip).to_numpy(dtype=float))
    return pd.DataFrame(z, index=wide.index, columns=wide.columns).where(wide.notna())


# ----------------------------------------------------------------------------- correlations
def row_corr(a: NDArray[Any], b: NDArray[Any], min_n: int = IC_MIN_OBS) -> FloatArray:
    """Row-wise Pearson correlation of two equally shaped 2-D arrays.

    A pair enters row ``t`` only where both entries are finite.  Rows with
    fewer than ``min_n`` pairs, or zero variance on either side, give NaN.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("row_corr expects two 2-D arrays of the same shape")
    m = np.isfinite(a) & np.isfinite(b)
    a0 = np.where(m, a, 0.0)
    b0 = np.where(m, b, 0.0)
    n = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = a0.sum(axis=1) / n
        mb = b0.sum(axis=1) / n
        ac = np.where(m, a0 - ma[:, None], 0.0)
        bc = np.where(m, b0 - mb[:, None], 0.0)
        r = (ac * bc).sum(axis=1) / np.sqrt((ac**2).sum(axis=1) * (bc**2).sum(axis=1))
    r = np.where(n >= min_n, r, np.nan)
    return r


def rank_ic_series(pred: pd.DataFrame, target: pd.DataFrame, min_n: int = IC_MIN_OBS) -> pd.Series:
    """Per-date Spearman rank correlation between two wide panels.

    Aligned on the intersection of dates and tickers; on each date only names
    with both a prediction and a target are ranked (average ranks for ties),
    then the Pearson correlation of the ranks is taken -- exactly
    ``scipy.stats.spearmanr`` on the paired, non-NaN cross-section.  Dates with
    fewer than ``min_n`` pairs give NaN.  Fully vectorised (no per-date loop).
    """
    idx = pred.index.intersection(target.index)
    cols = pred.columns.intersection(target.columns)
    p = pred.loc[idx, cols]
    t = target.loc[idx, cols]
    both = p.notna() & t.notna()
    rp = p.where(both).rank(axis=1).to_numpy(dtype=float)
    rt = t.where(both).rank(axis=1).to_numpy(dtype=float)
    return pd.Series(row_corr(rp, rt, min_n=min_n), index=idx, name="rank_ic")


# ----------------------------------------------------------------------------- Newey-West
def default_nw_lags(n: int) -> int:
    """Newey-West (1994) automatic lag ``floor(4 (n / 100)^(2/9))``."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def _finite_1d(x: Vector) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def newey_west_se(x: Vector, lags: int | None = None, min_obs: int = 3) -> float:
    """HAC standard error of the sample mean of ``x`` (Newey-West 1987).

    Long-run variance with the Bartlett kernel,

        S = gamma_0 + 2 * sum_{L=1}^{lags} (1 - L / (lags + 1)) gamma_L,
        gamma_L = (1/n) sum_{t=L}^{n-1} (x_t - xbar)(x_{t-L} - xbar),

    and ``se = sqrt(S / n)``.  Non-finite entries are dropped first.  ``lags``
    defaults to :func:`default_nw_lags` (``floor(4 (n/100)^(2/9))``); lags
    beyond ``n - 1`` contribute nothing.  Returns NaN when fewer than
    ``min_obs`` observations remain or ``S <= 0``.
    """
    v = _finite_1d(x)
    n = len(v)
    if n < max(min_obs, 2):
        return float("nan")
    if lags is None:
        lags = default_nw_lags(n)
    e = v - v.mean()
    s = float(e @ e) / n
    for L in range(1, min(int(lags), n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        s += 2.0 * w * float(e[L:] @ e[:-L]) / n
    if not s > 0:
        return float("nan")
    return float(np.sqrt(s / n))


def newey_west_tstat(x: Vector, lags: int | None = None, min_obs: int = 3) -> float:
    """``mean(x) / newey_west_se(x, lags)`` (NaN if the SE is undefined)."""
    v = _finite_1d(x)
    se = newey_west_se(v, lags=lags, min_obs=min_obs)
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    return float(v.mean() / se)


# ----------------------------------------------------------------------------- rolling beta
class RollingBeta(NamedTuple):
    """Result of :func:`rolling_beta` (both wide panels, date x ticker)."""

    beta: pd.DataFrame
    resid_var: pd.DataFrame

    @property
    def resid_vol(self) -> pd.DataFrame:
        # np.sqrt on a DataFrame returns a DataFrame; numpy's ufunc stubs say ndarray
        return cast(pd.DataFrame, np.sqrt(self.resid_var))


def rolling_beta(returns: pd.DataFrame, market: pd.Series, window: int, min_periods: int) -> RollingBeta:
    """Rolling OLS slope of every column of ``returns`` on ``market`` (with
    intercept) plus the residual variance, using data ``<= t`` only.

        beta_t  = Cov_w(r, m) / Var_w(m)
        resid_t = Var_w(r) - Cov_w(r, m)^2 / Var_w(m)      (clipped at 0)

    with population (ddof = 0) moments over the trailing ``window`` days and
    at least ``min_periods`` observations (pandas ``rolling`` semantics: each
    moment uses the days on which *that* series is non-NaN).  Days with zero
    market variance give NaN.  ``min_periods`` is required on purpose: the
    alpha features historically used ``0.8 * window`` and the risk model
    ``window // 4`` -- choose explicitly.
    """
    if min_periods < 2 or min_periods > window:
        raise ValueError("need 2 <= min_periods <= window")
    m = market.reindex(returns.index)
    w, mp = window, min_periods
    mean_r = returns.rolling(w, min_periods=mp).mean()
    mean_m = m.rolling(w, min_periods=mp).mean()
    mean_rm = returns.mul(m, axis=0).rolling(w, min_periods=mp).mean()
    var_r = returns.rolling(w, min_periods=mp).var(ddof=0)
    var_m = m.rolling(w, min_periods=mp).var(ddof=0).replace(0.0, np.nan)
    cov = mean_rm - mean_r.mul(mean_m, axis=0)
    beta = cov.div(var_m, axis=0)
    resid_var = (var_r - cov.pow(2).div(var_m, axis=0)).clip(lower=0.0)
    return RollingBeta(beta=beta, resid_var=resid_var)


# ----------------------------------------------------------------------------- industry dummies
def industry_dummies(
    labels: pd.Series, tickers: pd.Index | None = None, drop_first: bool = False
) -> pd.DataFrame:
    """Ticker x industry 0/1 float matrix with columns sorted by label.

    ``labels`` maps ticker -> industry label (e.g. ``MarketData.industry`` or
    ``MarketData.industry_as_of(t)``); ``tickers`` (default ``labels.index``)
    fixes the row order.  A ticker with a missing label gets an all-zero row.
    ``drop_first=True`` drops the first (alphabetical) industry, for
    regressions that also carry an intercept.
    """
    t = labels.index if tickers is None else pd.Index(tickers)
    lab = labels.reindex(t)
    cats = sorted(pd.unique(lab.dropna()).tolist())
    codes = pd.Categorical(lab, categories=cats).codes
    mat = np.zeros((len(t), len(cats)))
    ok = codes >= 0
    mat[np.flatnonzero(ok), codes[ok]] = 1.0
    out = pd.DataFrame(mat, index=t, columns=pd.Index(cats))
    if drop_first:
        out = out.iloc[:, 1:]
    return out
