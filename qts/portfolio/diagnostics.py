"""Portfolio diagnostics: transfer coefficient, holding period, factor
exposures, realised-vs-ex-ante risk, factor/specific attribution."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qts.core.types import RiskModelOutput


def transfer_coefficient(alpha: pd.Series, w: pd.Series, cov: pd.DataFrame | None = None,
                         risk: RiskModelOutput | None = None, exact_max_n: int = 500) -> float:
    """Transfer coefficient (Clarke, de Silva & Thorley 2002).

    Approximate (always available): ``corr(w_i * sigma_i, alpha_i / sigma_i)``
    with ``sigma_i`` the specific vol (from ``risk`` if given, else total vol
    from ``cov`` diagonal, else 1).  Exact (N <= ``exact_max_n`` and a
    covariance is available): ``alpha'w / sqrt(w'Σw · alpha'Σ⁻¹alpha)`` i.e.
    the Σ-correlation between ``w`` and the unconstrained Markowitz portfolio.
    Returns the exact version when computable, else the approximation.
    """
    idx = pd.Index(alpha.index).intersection(w.index)
    a = pd.Series(alpha, dtype=float).reindex(idx).fillna(0.0).values
    wv = pd.Series(w, dtype=float).reindex(idx).fillna(0.0).values
    if len(idx) < 3 or not np.any(a) or not np.any(wv):
        return float("nan")
    S = None
    if cov is not None:
        S = cov.reindex(index=idx, columns=idx).fillna(0.0).values
    elif risk is not None and len(idx) <= exact_max_n:
        S = risk.covariance(idx).values
    if S is not None and len(idx) <= exact_max_n:
        try:
            S = S + 1e-12 * np.eye(len(idx))
            Sa = np.linalg.solve(S, a)
            num = float(a @ wv)
            den = float(np.sqrt(max(wv @ S @ wv, 1e-30) * max(a @ Sa, 1e-30)))
            return float(np.clip(num / den, -1.0, 1.0))
        except np.linalg.LinAlgError:
            pass
    if risk is not None:
        sig = np.sqrt(pd.Series(risk.specific_var, dtype=float).reindex(idx)
                      .fillna(risk.specific_var.median()).clip(lower=1e-12).values)
    elif S is not None:
        sig = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    else:
        sig = np.ones(len(idx))
    c = np.corrcoef(wv * sig, a / sig)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def effective_holding_days(turnover_series: pd.Series, gross_series: pd.Series | None = None) -> float:
    """Average holding period in days = average gross book / average one-sided
    daily traded.  ``turnover_series`` is the two-sided daily turnover
    ``sum_i |x_it|`` (fraction of NAV; a round trip of the whole book = 2).
    ``gross_series`` is ``sum_i |w_it|``; if omitted the book is assumed to be
    fully deployed at gross 1, giving ``2 / mean(turnover)``.
    """
    to = pd.Series(turnover_series, dtype=float).dropna()
    if len(to) == 0:
        return float("nan")
    one_sided = to.mean() / 2.0
    gross = 1.0 if gross_series is None else float(pd.Series(gross_series, dtype=float).reindex(to.index).mean())
    if one_sided <= 0:
        return float("inf")
    return float(gross / one_sided)


def factor_exposure_report(w: pd.Series, risk: RiskModelOutput) -> pd.DataFrame:
    """Per-factor exposure ``B'w``, factor vol, and marginal contribution to
    portfolio variance (``f_k * (F f)_k``), plus specific and total rows."""
    t = pd.Index(w.index)
    B = risk.exposures.reindex(t).fillna(0.0)
    F = risk.factor_cov.loc[B.columns, B.columns].values
    wv = pd.Series(w, dtype=float).fillna(0.0).values
    f = B.values.T @ wv
    Ff = F @ f
    contrib = f * Ff
    d = pd.Series(risk.specific_var, dtype=float).reindex(t).fillna(risk.specific_var.median()).values
    spec = float((wv ** 2 * d).sum())
    total = float(f @ Ff + spec)
    out = pd.DataFrame({
        "exposure": f,
        "factor_vol_annual": np.sqrt(np.clip(np.diag(F), 0, None) * 252),
        "var_contribution": contrib,
    }, index=B.columns)
    out["pct_of_var"] = out["var_contribution"] / total if total > 0 else np.nan
    out.loc["__specific__"] = [np.nan, np.nan, spec, spec / total if total > 0 else np.nan]
    out.loc["__total__"] = [np.nan, np.sqrt(total * 252), total, 1.0]
    return out


def realized_vs_exante_risk(returns: pd.Series, exante_var_series: pd.Series,
                            periods_per_year: int = 252) -> dict[str, float]:
    """Compare realised return vol with the model's ex-ante forecast.

    ``exante_var_series[t]`` must be the forecast daily variance of the
    position that earns ``returns[t]``.  Returns the realised/ex-ante vol
    ratio and the bias statistic ``std(r_t / sigma_t)`` (1 = unbiased).
    """
    r = pd.Series(returns, dtype=float)
    v = pd.Series(exante_var_series, dtype=float).reindex(r.index)
    m = r.notna() & v.notna() & (v > 0)
    if m.sum() < 5:
        return {"ratio": float("nan"), "bias_stat": float("nan"), "realized_vol_annual": float("nan"),
                "exante_vol_annual": float("nan")}
    rr, vv = r[m], v[m]
    real = float(rr.std(ddof=1) * np.sqrt(periods_per_year))
    ex = float(np.sqrt(vv.mean() * periods_per_year))
    z = rr / np.sqrt(vv)
    return {"ratio": real / ex if ex > 0 else float("nan"), "bias_stat": float(z.std(ddof=1)),
            "realized_vol_annual": real, "exante_vol_annual": ex}


ExposuresHistory = Mapping[pd.Timestamp, pd.DataFrame] | pd.DataFrame


def brinson_like_attribution(w_history: pd.DataFrame, factor_returns: pd.DataFrame,
                             exposures_history: ExposuresHistory, specific_returns: pd.DataFrame | None = None,
                             total_returns: pd.DataFrame | None = None, lag: int = 1) -> pd.DataFrame:
    """Decompose daily portfolio PnL into factor and specific components.

    ``w_history`` rows are weights *held* at the close of date t; they earn
    the returns dated ``t + lag`` (default 1).  ``exposures_history`` is either
    a mapping date -> (ticker x factor) frame (as of t) or a long frame with a
    (date, ticker) MultiIndex.  Factor PnL on day t+1 is
    ``sum_k (B_t' w_t)_k f_{k,t+1}``; specific PnL is ``w_t' u_{t+1}`` where
    ``u`` is ``specific_returns`` if supplied, else ``total - B f``.
    Output: one column per factor, ``specific`` and ``total``.
    """
    rows = {}
    dates = list(w_history.index)
    fr_idx = factor_returns.index
    for t in dates:
        pos = fr_idx.searchsorted(t, side="right") - 1 + lag
        if pos < 0 or pos >= len(fr_idx):
            continue
        t1 = fr_idx[pos]
        if isinstance(exposures_history, pd.DataFrame):
            if t not in exposures_history.index.get_level_values(0):
                continue
            B = exposures_history.xs(t, level=0)
        else:
            B = exposures_history.get(t)
            if B is None:
                continue
        w = w_history.loc[t].dropna()
        w = w[w != 0]
        if w.empty:
            continue
        Bt = B.reindex(w.index).fillna(0.0)
        f = factor_returns.loc[t1].reindex(Bt.columns).fillna(0.0)
        expo = Bt.values.T @ w.values
        fac = pd.Series(expo * f.values, index=Bt.columns)
        if specific_returns is not None:
            u = specific_returns.loc[t1].reindex(w.index).fillna(0.0).values
        elif total_returns is not None:
            r = total_returns.loc[t1].reindex(w.index).fillna(0.0).values
            u = r - Bt.values @ f.values
        else:
            u = np.zeros(len(w))
        rec = fac.to_dict()
        rec["specific"] = float(w.values @ u)
        rec["total"] = float(fac.sum() + rec["specific"])
        rows[t1] = rec
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()
