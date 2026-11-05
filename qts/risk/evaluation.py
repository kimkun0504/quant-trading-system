"""Risk-model evaluation (USE4 bias statistics, MZ / QLIKE losses, GMV tests).

* **Bias statistic** (USE4 "Model testing"): ``b = std_t( r_t / sigma_{t-1} )``
  where ``sigma_{t-1}`` is the forecast made before ``r_t`` is realized.  An
  unbiased forecast has ``b = 1``; under normality the 95% confidence band is
  ``1 +- 2 / sqrt(2T)`` (``sqrt(2/T)`` is the sd of a sample sd).
* **Mincer-Zarnowitz**: regress realized squared returns on predicted variance,
  ``r^2 = a + b * sigma^2 + e``; ideal ``a = 0, b = 1``.
* **QLIKE** (Patton 2011): ``mean( r^2/sigma^2 - log(r^2/sigma^2) - 1 )``,
  robust to noise in the volatility proxy and the loss we primarily report.
* **Min-variance OOS test**: compare covariance estimators by the realized
  out-of-sample variance of the global-minimum-variance portfolio
  ``w ∝ Sigma^{-1} 1`` rebalanced every ``rebalance_every`` days (Ledoit &
  Wolf 2017/2022; Reis, Serra & Gama 2025).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from qts.core.types import MarketData


# --------------------------------------------------------------------------- bias statistics
def bias_statistic(predicted_vol: pd.Series, realized_returns: pd.Series) -> float:
    """Std of standardized returns ``r_t / sigma_t`` over the common index."""
    idx = predicted_vol.dropna().index.intersection(realized_returns.dropna().index)
    if len(idx) < 2:
        return np.nan
    z = realized_returns.loc[idx] / predicted_vol.loc[idx].clip(lower=1e-12)
    return float(z.std(ddof=0))


def bias_band(T: int, n_sd: float = 2.0) -> tuple:
    half = n_sd / np.sqrt(2.0 * T)
    return (1.0 - half, 1.0 + half)


def rolling_bias_stats(predicted_vol: pd.Series, realized_returns: pd.Series, window: int = 252) -> pd.Series:
    """Rolling-window bias statistic (USE4 reports 12-month windows and the
    5th/95th percentiles, ~0.66/1.34 under kurtosis)."""
    idx = predicted_vol.dropna().index.intersection(realized_returns.dropna().index)
    z = realized_returns.loc[idx] / predicted_vol.loc[idx].clip(lower=1e-12)
    return z.rolling(window, min_periods=max(20, window // 4)).std(ddof=0)


def mean_rolling_absolute_deviation(rolling_bias: pd.Series) -> float:
    """MRAD: mean |b_window - 1| (USE4 ideal ~0.17 for 12-month windows)."""
    return float((rolling_bias.dropna() - 1.0).abs().mean())


PortfolioSpec = pd.Series | Callable[[pd.Timestamp, pd.Index], pd.Series]


def portfolio_bias_test(
    model,
    md: MarketData,
    portfolios: dict[str, PortfolioSpec],
    dates: Iterable[pd.Timestamp],
    include_alignment: bool = True,
    alpha_fn: Callable[[pd.Timestamp], pd.Series] | None = None,
) -> pd.DataFrame:
    """Bias statistics of several portfolios through time.

    For each ``t`` in ``dates`` the model is evaluated ``as_of(t)``, the
    portfolio weights ``w_t`` are static (``pd.Series``) or produced by a
    callable ``f(date, tickers) -> weights``; the forecast
    ``sigma_t = sqrt(w' Sigma_t w)`` is compared with the realized return of
    the portfolio on the *next* trading day ``t+1``.

    Returns a DataFrame indexed by portfolio name with columns
    ``bias, lower, upper, in_band, T``."""
    dates = pd.DatetimeIndex(list(dates))
    all_dates = md.dates
    rets = md.returns
    pos = {d: i for i, d in enumerate(all_dates)}
    pred = {k: {} for k in portfolios}
    real = {k: {} for k in portfolios}
    for d in dates:
        i = pos.get(d)
        if i is None or i + 1 >= len(all_dates):
            continue
        alpha = alpha_fn(d) if alpha_fn is not None else None
        rm = model.as_of(d, alpha=alpha)
        nxt = all_dates[i + 1]
        r_next = rets.loc[nxt]
        for name, spec in portfolios.items():
            w = spec(d, rm.tickers) if callable(spec) else spec
            w = w.reindex(rm.tickers).fillna(0.0)
            var = rm.portfolio_variance(w, include_alignment=include_alignment)
            pred[name][nxt] = np.sqrt(max(var, 0.0))
            real[name][nxt] = float((w * r_next.reindex(w.index).fillna(0.0)).sum())
    rows = []
    for name in portfolios:
        p, r = pd.Series(pred[name]), pd.Series(real[name])
        T = len(p)
        lo, hi = bias_band(T)
        b = bias_statistic(p, r)
        rows.append(
            {"portfolio": name, "bias": b, "lower": lo, "upper": hi, "in_band": bool(lo <= b <= hi), "T": T}
        )
    return pd.DataFrame(rows).set_index("portfolio")


def random_long_short_weights(rng: np.random.Generator, gross: float = 2.0) -> Callable:
    """Weight generator: random dollar-neutral portfolio with given gross."""

    def gen(date: pd.Timestamp, tickers: pd.Index) -> pd.Series:
        x = rng.standard_normal(len(tickers))
        x = x - x.mean()
        x = x / np.abs(x).sum() * gross
        return pd.Series(x, index=tickers)

    return gen


# --------------------------------------------------------------------------- per-stock losses
def _align(pred_var, realized_sq):
    p = pd.Series(pred_var).astype(float)
    r = pd.Series(realized_sq).astype(float)
    idx = p.dropna().index.intersection(r.dropna().index)
    return p.loc[idx].clip(lower=1e-18).values, r.loc[idx].values


def mincer_zarnowitz(pred_var, realized_sq) -> dict[str, float]:
    """OLS of realized squared returns on predicted variance.  Returns
    ``{"alpha", "beta", "r2", "n"}``; ideal alpha 0, beta 1."""
    p, r = _align(pred_var, realized_sq)
    if len(p) < 3:
        return {"alpha": np.nan, "beta": np.nan, "r2": np.nan, "n": len(p)}
    X = np.column_stack([np.ones(len(p)), p])
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    fit = X @ coef
    tss = ((r - r.mean()) ** 2).sum()
    r2 = 1 - ((r - fit) ** 2).sum() / tss if tss > 0 else np.nan
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "r2": float(r2), "n": len(p)}


def qlike(pred_var, realized_sq) -> float:
    """Patton (2011) QLIKE loss, mean over observations (lower is better, 0 is
    perfect).  Zero realized squared returns are floored at a tiny value."""
    p, r = _align(pred_var, realized_sq)
    if len(p) == 0:
        return np.nan
    ratio = np.clip(r, 1e-18, None) / p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse_variance(pred_var, realized_sq) -> float:
    p, r = _align(pred_var, realized_sq)
    return float(np.mean((r - p) ** 2)) if len(p) else np.nan


# --------------------------------------------------------------------------- GMV test
def gmv_weights(S: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    N = S.shape[0]
    ones = np.ones(N)
    x = np.linalg.solve(S + ridge * np.trace(S) / N * np.eye(N), ones)
    return x / x.sum()


def min_variance_oos_test(
    cov_estimators: dict[str, Callable[[np.ndarray], np.ndarray]],
    returns: pd.DataFrame,
    rebalance_every: int = 21,
    window: int = 252,
    min_window: int | None = None,
) -> pd.DataFrame:
    """Compare covariance estimators by realized out-of-sample variance (and
    turnover) of the GMV portfolio.

    ``cov_estimators`` maps name -> ``f(X) -> Sigma`` where ``X`` is the
    trailing (window x N) return matrix.  Every ``rebalance_every`` days the
    GMV weights are recomputed from the trailing window and held.  Reports
    annualized OOS volatility, mean daily variance and average turnover."""
    R = returns.dropna(axis=1, how="any")
    X = R.values
    T = X.shape[0]
    min_window = min_window or window
    out = {name: {"ret": [], "turn": []} for name in cov_estimators}
    prev_w = dict.fromkeys(cov_estimators)
    t = min_window
    while t < T:
        end = min(t + rebalance_every, T)
        hist = X[max(0, t - window) : t]
        for name, est in cov_estimators.items():
            w = gmv_weights(est(hist))
            if prev_w[name] is not None:
                out[name]["turn"].append(np.abs(w - prev_w[name]).sum())
            prev_w[name] = w
            out[name]["ret"].extend(list(X[t:end] @ w))
        t = end
    rows = []
    for name, d in out.items():
        r = np.array(d["ret"])
        rows.append(
            {
                "estimator": name,
                "oos_daily_var": float(r.var(ddof=1)),
                "oos_vol_annual": float(r.std(ddof=1) * np.sqrt(252)),
                "avg_turnover": float(np.mean(d["turn"])) if d["turn"] else np.nan,
                "n_days": len(r),
            }
        )
    return pd.DataFrame(rows).set_index("estimator").sort_values("oos_daily_var")
