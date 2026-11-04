"""Alpha alignment factor (Saxena & Stubbs, "Alpha Alignment Factor: A
Solution to the Underestimation of Risk for Optimized Active Portfolios",
Axioma research; see also Saxena & Stubbs 2012, J. Portfolio Management).

Mechanism
---------
Decompose the alpha into a component spanned by the risk-model exposures and
an orthogonal remainder:

    alpha = B b + alpha_perp,        y = alpha_perp / ||alpha_perp||.

The optimizer sees no systematic risk in ``alpha_perp`` and over-loads on it,
which is why optimized books realize bias statistics well above 1.  The fix
augments the covariance with a rank-one term

    Sigma_aug = B F B' + D + nu * y y'

which keeps the problem convex and (empirically) brings the bias statistic of
the optimized portfolio back toward 1.

Choice of ``nu``
----------------
Saxena-Stubbs leave ``nu`` as a calibration parameter (they tune it so the
bias statistic of the "alignment portfolio" is ~1).  Here ``nu`` is set so
that the alignment factor has the variance of a *typical style factor*:
``nu = nu_scale * median_k F_kk`` over the style factors, **after** rescaling
to the unit-norm vector ``y``.  Note that a style factor with unit-sd
exposures ``x`` has portfolio contribution ``F_kk (x'w)^2``, while the
alignment term contributes ``nu (y'w)^2`` with ``||y|| = 1``; to make the two
comparable we scale by ``N`` (``||x||^2 ≈ N`` for a standardized style):
``nu = nu_scale * N * median_style_var``.  If no factor covariance is
supplied the fallback is ``nu = nu_scale * median(specific_var)``, i.e. the
alignment factor is treated like one more idiosyncratic source (documented
fallback, less principled).

That heuristic is only a starting point.  :func:`calibrate_alignment_nu`
implements the Saxena-Stubbs calibration loop proper: it searches
``nu_scale`` so that the bias statistic of the *optimized* book (forecast
with the alignment term, realised next day) is ~1 over a calibration window.
It is slow (one optimisation per date per bisection step) and is deliberately
**not** on the default ``as_of`` path -- run it offline on the warm-up window
and pass the resulting ``nu_scale`` to ``as_of``.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from qts.core.config import RiskConfig
from qts.core.types import RiskModelOutput
from qts.risk.covariance import style_variances


def alpha_alignment(alpha: pd.Series, exposures: pd.DataFrame, specific_var: pd.Series,
                    nu_scale: float = 1.0, factor_cov: Optional[pd.DataFrame] = None,
                    cfg: Optional[RiskConfig] = None, weighted: bool = False
                    ) -> Tuple[pd.Series, float]:
    """Return ``(y, nu)``: unit-norm alignment vector orthogonal to the
    exposure columns.

    By default the regression is OLS (Euclidean projection), which is what
    Saxena-Stubbs use and what makes ``B'y = 0`` exact.  ``weighted=True``
    uses ``1/specific_var`` WLS weights (the "implied alpha" metric), in which
    case ``y`` is orthogonal to ``B`` in the weighted inner product instead.
    """
    cfg = cfg or RiskConfig()
    idx = alpha.dropna().index.intersection(exposures.index)
    a = alpha.loc[idx].values.astype(float)
    B = exposures.loc[idx].values.astype(float)
    keep = np.abs(B).sum(axis=0) > 0
    B = B[:, keep]
    if weighted:
        w = 1.0 / specific_var.reindex(idx).fillna(specific_var.median()).clip(lower=1e-12).values
        w = w / w.mean()
        sw = np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(B * sw, a * sw[:, 0], rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
    resid = a - B @ coef
    norm = float(np.linalg.norm(resid))
    if norm < 1e-14:
        y = pd.Series(0.0, index=idx)
        return y.reindex(alpha.index).fillna(0.0), 0.0
    y = pd.Series(resid / norm, index=idx)
    N = len(idx)
    if factor_cov is not None:
        sv = style_variances(factor_cov, cfg)
        base = float(np.median(sv.values)) if len(sv) else float(np.median(np.diag(factor_cov.values)))
        nu = nu_scale * N * base
    else:
        nu = nu_scale * float(specific_var.reindex(idx).median())
    return y.reindex(alpha.index).fillna(0.0), float(nu)


def misalignment_gauge(alpha: pd.Series, exposures: pd.DataFrame) -> float:
    """``1 - R^2`` of alpha regressed on the exposures: the fraction of alpha
    variance that the risk model cannot see (BUILD_PLAN section 3 monitor)."""
    idx = alpha.dropna().index.intersection(exposures.index)
    a = alpha.loc[idx].values.astype(float)
    B = exposures.loc[idx].values.astype(float)
    coef, *_ = np.linalg.lstsq(B, a, rcond=None)
    resid = a - B @ coef
    tss = float(((a - a.mean()) ** 2).sum())
    return float((resid ** 2).sum() / tss) if tss > 0 else np.nan


# --------------------------------------------------------------------------- nu calibration
def _default_mean_variance_book(date: pd.Timestamp, rm: RiskModelOutput, alpha: pd.Series,
                                gross: float = 2.0) -> pd.Series:
    """Cheap stand-in for the production optimizer: dollar-neutral
    ``Sigma_aug^{-1} alpha`` scaled to ``gross``.  Loads on ``alpha_perp`` in
    exactly the way an unconstrained optimizer does, which is the effect the
    alignment term is calibrated against."""
    t = rm.tickers
    a = pd.Series(alpha, dtype=float).reindex(t).fillna(0.0).values
    S = rm.covariance(include_alignment=True).values
    S = S + 1e-12 * np.trace(S) / len(t) * np.eye(len(t))
    w = np.linalg.solve(S, a)
    w = w - w.mean()
    g = np.abs(w).sum()
    w = w / g * gross if g > 0 else w
    return pd.Series(w, index=t)


def calibrate_alignment_nu(
    model,
    md,
    alphas_by_date: Union[Mapping[pd.Timestamp, pd.Series], Callable[[pd.Timestamp], pd.Series]],
    dates: Iterable[pd.Timestamp],
    solve_fn: Optional[Callable[[pd.Timestamp, RiskModelOutput, pd.Series], pd.Series]] = None,
    lo: float = 0.05,
    hi: float = 50.0,
    n_iter: int = 10,
    tol: float = 0.02,
) -> Dict[str, object]:
    """Saxena-Stubbs calibration of the alignment variance scale.

    For a candidate ``nu_scale`` the optimized book on every date ``t`` in
    ``dates`` is ``w_t = solve_fn(t, model.as_of(t, alpha_t, nu_scale), alpha_t)``,
    its forecast ``sqrt(w' Sigma_aug w)`` (alignment included) is compared
    with the realised return on ``t+1`` and the bias statistic
    ``sd(realised / forecast)`` is formed.  A larger ``nu_scale`` both raises the
    forecast and pushes the optimizer off ``alpha_perp``, so the bias falls
    monotonically (in expectation); the function bisects on ``log nu_scale``
    in ``[lo, hi]`` until the bias is within ``tol`` of 1 or ``n_iter`` steps
    are used.  ``model.as_of`` is evaluated **once per date** (``nu`` is linear
    in ``nu_scale``: ``alignment_var`` is rescaled in place for the search).

    ``solve_fn`` defaults to a dollar-neutral ``Sigma_aug^{-1} alpha`` book;
    pass a closure around ``SinglePeriodOptimizer.solve`` to calibrate on the
    production constraint set (slower: one convex solve per date per step).

    Returns ``{"nu_scale", "bias", "bias_unaligned", "lower", "upper",
    "in_band", "n_dates", "history": [(nu_scale, bias), ...], "converged"}``.
    Cost is O(n_dates * (1 as_of + (n_iter + 3) solves)); keep ``dates`` to a
    sample of the warm-up window (e.g. every 5th day of a year).
    """
    from qts.risk.evaluation import bias_band, bias_statistic

    dates = pd.DatetimeIndex(list(dates))
    solve = solve_fn or _default_mean_variance_book
    all_dates = md.dates
    pos = {d: i for i, d in enumerate(all_dates)}
    rets = md.returns

    def alpha_at(d):
        return alphas_by_date(d) if callable(alphas_by_date) else alphas_by_date[d]

    base: list = []  # (date, rm_at_nu_scale_1, alpha, r_next)
    for d in dates:
        i = pos.get(d)
        if i is None or i + 1 >= len(all_dates):
            continue
        a = pd.Series(alpha_at(d), dtype=float)
        rm = model.as_of(d, alpha=a, nu_scale=1.0)
        if rm.alignment_vector is None or rm.alignment_var <= 0:
            continue
        base.append((d, rm, a, rets.loc[all_dates[i + 1]]))
    if not base:
        raise ValueError("no dates with an alignment term (is cfg.alignment on?)")

    def bias_for(nu_scale: float) -> float:
        pred, real = [], []
        for d, rm, a, r_next in base:
            rm_s = dataclasses.replace(rm, alignment_var=float(rm.alignment_var) * float(nu_scale))
            w = pd.Series(solve(d, rm_s, a), dtype=float).reindex(rm_s.tickers).fillna(0.0)
            v = rm_s.portfolio_variance(w, include_alignment=True)
            pred.append(np.sqrt(max(v, 0.0)))
            real.append(float((w * r_next.reindex(w.index).fillna(0.0)).sum()))
        return float(bias_statistic(pd.Series(pred), pd.Series(real)))

    history = []
    b0 = bias_for(0.0)
    b_lo, b_hi = bias_for(lo), bias_for(hi)
    history += [(0.0, b0), (lo, b_lo), (hi, b_hi)]
    converged = False
    if b_lo <= 1.0:
        nu_star, b_star = lo, b_lo
    elif b_hi >= 1.0:
        nu_star, b_star = hi, b_hi
    else:
        a_, c_ = np.log(lo), np.log(hi)
        nu_star, b_star = hi, b_hi
        for _ in range(int(n_iter)):
            m_ = 0.5 * (a_ + c_)
            nu_m = float(np.exp(m_))
            b_m = bias_for(nu_m)
            history.append((nu_m, b_m))
            nu_star, b_star = nu_m, b_m
            if abs(b_m - 1.0) <= tol:
                converged = True
                break
            if b_m > 1.0:
                a_ = m_
            else:
                c_ = m_
    lo_band, hi_band = bias_band(len(base))
    return {"nu_scale": float(nu_star), "bias": float(b_star), "bias_unaligned": float(b0),
            "lower": lo_band, "upper": hi_band, "in_band": bool(lo_band <= b_star <= hi_band),
            "n_dates": len(base), "history": history, "converged": converged}
