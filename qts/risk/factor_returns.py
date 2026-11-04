"""Cross-sectional factor-return estimation (USE4 regression).

Each day ``t`` we solve the weighted least squares problem

    r_t = B_{t-1} f_t + u_t,     weights  w_i ∝ sqrt(cap_i)

with the Barra identification constraint on the industry block

    sum_j  c_j f_j = 0,   c_j = (cap of industry j) / (total cap),

so that the ``market`` (Country) factor return equals the cap-weighted
market return and industry returns are *relative* to the market
(USE4 Methodology Notes, "Factor return estimation").  The constraint is
imposed exactly by restricted least squares: we reparameterize ``f = R g``
with ``R`` the (K x K-1) basis of the constraint null-space (pivot on the
largest-cap industry), solve the unconstrained WLS in ``g`` and map back.

Exposures used for day ``t`` are those built from data ``<= t-1`` so the
regression is point-in-time (exposures known before the return is realized).

R^2 is the weighted coefficient of determination
``1 - sum w u^2 / sum w (r - rbar_w)^2`` where ``rbar_w`` is the weighted mean.
"""
from __future__ import annotations

from typing import Dict, Iterable, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from qts.core.config import RiskConfig
from qts.core.types import MarketData
from qts.risk.exposures import MARKET_FACTOR, split_factor_columns


class CrossSectionalFit(NamedTuple):
    """Result of one cross-sectional regression.  Unpacks as a 3-tuple
    ``(factor_returns, specific_returns, r2)``."""
    factor_returns: pd.Series
    specific_returns: pd.Series
    r2: float


def _constraint_nullspace(c: np.ndarray) -> np.ndarray:
    """Basis ``R`` (K x K-1) of ``{f : c'f = 0}``.  If ``c == 0`` returns identity."""
    K = len(c)
    if not np.any(c != 0):
        return np.eye(K)
    p = int(np.argmax(np.abs(c)))
    free = [k for k in range(K) if k != p]
    R = np.zeros((K, K - 1))
    for col, k in enumerate(free):
        R[k, col] = 1.0
        R[p, col] = -c[k] / c[p]
    return R


def restricted_wls(y: np.ndarray, X: np.ndarray, w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Minimise ``sum w_i (y_i - x_i'f)^2`` subject to ``c'f = 0``."""
    R = _constraint_nullspace(c)
    sw = np.sqrt(w)[:, None]
    XR = X @ R
    g, *_ = np.linalg.lstsq(XR * sw, y * sw[:, 0], rcond=None)
    return R @ g


def estimate_factor_returns(returns_t: pd.Series, exposures_tminus1: pd.DataFrame,
                            market_cap_tminus1: pd.Series,
                            cfg: Optional[RiskConfig] = None,
                            constraint_weights: str = "cap",
                            weight_cap_quantile: Optional[float] = None) -> CrossSectionalFit:
    """One day's constrained WLS.  See module docstring.

    Parameters
    ----------
    returns_t : realized simple returns on day t (index: tickers).
    exposures_tminus1 : exposure matrix built as of t-1 (tickers x factors).
    market_cap_tminus1 : market caps as of t-1 (regression weights ∝ sqrt(cap),
        constraint weights ∝ cap).
    constraint_weights : ``"cap"`` (USE4: industry returns sum to zero
        cap-weighted, so ``market`` = cap-weighted market) or ``"regression"``
        (sum to zero under the sqrt-cap regression weights, so ``market`` =
        sqrt-cap-weighted market -- less dominated by mega-caps in a
        concentrated universe).
    weight_cap_quantile : if given, market caps are winsorized at this
        cross-sectional quantile before forming regression and constraint
        weights (Axioma-style mega-cap damping).  ``None`` = pure USE4.

    Returns
    -------
    CrossSectionalFit(factor_returns, specific_returns, r2).  Factor returns of
    industries with no members on the day are reported as 0.0.
    """
    cfg = cfg or RiskConfig()
    idx = (returns_t.dropna().index
           .intersection(exposures_tminus1.index)
           .intersection(market_cap_tminus1.dropna().index))
    B = exposures_tminus1.loc[idx]
    cols = list(B.columns)
    y = returns_t.loc[idx].values.astype(float)
    cap = market_cap_tminus1.loc[idx].clip(lower=1e-9).values.astype(float)
    if weight_cap_quantile is not None:
        cap = np.minimum(cap, np.quantile(cap, weight_cap_quantile))
    X = B.values.astype(float)
    w = np.sqrt(cap)
    w = w / w.sum()
    cw = cap if constraint_weights == "cap" else w

    _, industries, _ = split_factor_columns(cols, cfg)
    c = np.zeros(len(cols))
    if industries:
        ind_pos = [cols.index(j) for j in industries]
        ind_cap = X[:, ind_pos].T @ cw
        c[ind_pos] = ind_cap / cw.sum()
    # drop empty columns (industries with no member today) to keep X full rank
    nonempty = np.abs(X).sum(axis=0) > 0
    f = np.zeros(len(cols))
    f[nonempty] = restricted_wls(y, X[:, nonempty], w, c[nonempty])
    u = y - X @ f
    ybar = float(w @ y)
    sst = float(w @ (y - ybar) ** 2)
    r2 = float(1.0 - (w @ u ** 2) / sst) if sst > 0 else np.nan
    return CrossSectionalFit(pd.Series(f, index=cols, name="factor_return"),
                             pd.Series(u, index=idx, name="specific_return"), r2)


def estimate_factor_return_history(md: MarketData, exposures_by_date: Dict[pd.Timestamp, pd.DataFrame],
                                   cfg: Optional[RiskConfig] = None,
                                   dates: Optional[Iterable[pd.Timestamp]] = None,
                                   **regression_kwargs,
                                   ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Run the daily regression for every date ``t`` whose previous trading
    day ``t-1`` has an exposure matrix in ``exposures_by_date``.

    ``regression_kwargs`` are forwarded to :func:`estimate_factor_returns`
    (``constraint_weights``, ``weight_cap_quantile``).

    Returns ``(factor_returns [date x factor], specific_returns [date x ticker], r2 [date])``.
    """
    cfg = cfg or RiskConfig()
    all_dates = md.dates
    returns = md.returns
    pos = {d: i for i, d in enumerate(all_dates)}
    candidates = all_dates if dates is None else pd.DatetimeIndex(list(dates))
    fr, sr, r2 = {}, {}, {}
    for d in candidates:
        i = pos.get(d)
        if i is None or i == 0:
            continue
        prev = all_dates[i - 1]
        B = exposures_by_date.get(prev)
        if B is None:
            continue
        fit = estimate_factor_returns(returns.iloc[i], B, md.market_cap.iloc[i - 1], cfg, **regression_kwargs)
        fr[d], sr[d], r2[d] = fit.factor_returns, fit.specific_returns, fit.r2
    if not fr:
        empty = pd.DataFrame(index=pd.DatetimeIndex([]))
        return empty, empty.copy(), pd.Series(dtype=float)
    factor_returns = pd.DataFrame(fr).T.sort_index().fillna(0.0)
    specific_returns = pd.DataFrame(sr).T.sort_index().reindex(columns=md.tickers)
    factor_returns.index.name = specific_returns.index.name = "date"
    return factor_returns, specific_returns, pd.Series(r2, name="r2").sort_index()


def cap_weighted_industry_sum(factor_returns_t: pd.Series, exposures_tminus1: pd.DataFrame,
                              market_cap_tminus1: pd.Series, cfg: Optional[RiskConfig] = None) -> float:
    """Diagnostic: ``sum_j c_j f_j`` which the constraint forces to zero."""
    cfg = cfg or RiskConfig()
    _, industries, _ = split_factor_columns(exposures_tminus1.columns, cfg)
    idx = exposures_tminus1.index.intersection(market_cap_tminus1.dropna().index)
    cap = market_cap_tminus1.loc[idx]
    c = exposures_tminus1.loc[idx, industries].mul(cap, axis=0).sum() / cap.sum()
    return float((c * factor_returns_t.reindex(industries).fillna(0.0)).sum())
