"""Statistical (asymptotic principal components) risk model.

Why
---
A fundamental model can only price the factors it can observe.  In the
synthetic-market shakedown, books aligned with a latent factor that had no
observable footprint ran at 1.3-1.8x their ex-ante risk (docs/REVIEW_NOTES.md,
"Integration findings").  The plan (docs/BUILD_PLAN.md section 3, Axioma's
recommendation) is a statistical companion model at the same horizon that
(i) catches transient / unobserved common factors and (ii) flags them.

Method: APCA (Connor & Korajczyk 1986, 1988)
--------------------------------------------
On a demeaned ``T x N`` return panel ``R`` with ``N > T`` the ``N x N`` sample
covariance is singular, but the ``T x T`` cross-product ``Omega = R R' / N``
has the same non-zero spectrum.  Its top-``k`` eigenvectors, scaled so that
``F'F / T = I_k``, are consistent (as ``N -> inf``) estimates of the factor
*returns*; loadings follow by time-series regression of each stock on ``F``
(``B = R'F / T`` when nothing is missing).  Connor-Korajczyk's refinement
corrects for cross-sectional heteroskedasticity: re-scale each stock by its
estimated specific vol, re-extract the factors from the scaled panel, and
re-estimate loadings on the original panel; ``apca_iters`` rounds.  When
``N <= T`` the ``N x N`` covariance is eigen-decomposed instead (identical
solution, cheaper).

Missing returns are demeaned and set to zero for the extraction (the standard
EM-free approximation) and skipped in the per-stock regressions, which use
only observed days.

Factor identification and sign stability
----------------------------------------
Principal components are identified only up to an orthogonal rotation.  To
make the factor-return history interpretable across refits, each new
solution is aligned to the previous one by orthogonal Procrustes on the
overlapping window days (``Q = U V'`` from the SVD of ``F_new' F_prev``);
this fixes signs and small rotations without changing the fitted
systematic covariance ``F B'``.  The first solution is signed so that each
factor's loadings sum to a positive number.

Point-in-time schedule
----------------------
``fit`` precomputes a full APCA solution every ``refit_every`` trading days
on the trailing ``window`` days, with the refit grid anchored at the **first**
eligible date (index ``min_obs - 1``), never at the end of the sample.  That
anchoring is what makes ``as_of(date)`` on a long fit identical to a fit on
``returns.loc[:date]``.  Between refits:

* new daily factor returns are estimated by cross-sectional OLS of the day's
  (demeaned) returns on the last refit's loadings;
* exposures are **refreshed daily** by regressing each stock's trailing
  ``window`` returns on the factor-return history for the same days
  (on a refit date this reproduces the APCA loadings exactly);
* specific variance is the (optionally EWMA-weighted, ``half_life``) residual
  variance over the window;
* the factor covariance is the weighted covariance of the trailing ``window``
  factor returns.

Everything used by ``as_of(date)`` therefore has timestamp ``<= date``.

Factor count
------------
:func:`select_n_factors` implements Bai & Ng (2002) ``IC_p2`` and the
Marchenko-Pastur edge ``lambda_+ = sigma^2 (1 + sqrt(N/T))^2`` (counting
eigenvalues of the standardized sample covariance above the noise edge, with
``sigma^2`` re-estimated from the bulk after removing the spikes and the
dimension taken as the number of non-zero eigenvalues, since residuals of a
``p``-factor cross-sectional regression live in an ``N - p`` subspace).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from qts.core.types import RiskModelOutput, Wide
from qts.risk.covariance import ewma_weights, nearest_psd

STAT_FACTOR_PREFIX = "stat_"


# --------------------------------------------------------------------------- small helpers
def _weights(T: int, half_life: Optional[float]) -> np.ndarray:
    if half_life is None:
        return np.full(T, 1.0 / T)
    return ewma_weights(T, half_life)


def _prepare_panel(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(R, mask, mu)``: observed-mean-demeaned panel with NaN -> 0."""
    A = np.asarray(X, dtype=float)
    mask = np.isfinite(A)
    A0 = np.where(mask, A, 0.0)
    n = mask.sum(axis=0)
    mu = np.where(n > 0, A0.sum(axis=0) / np.maximum(n, 1), 0.0)
    R = np.where(mask, A0 - mu, 0.0)
    return R, mask, mu


def _spectrum(R: np.ndarray) -> np.ndarray:
    """Descending non-zero eigenvalues of the sample covariance ``R'R / T``
    computed on the smaller of the two Gram matrices."""
    T, N = R.shape
    G = (R @ R.T) if N > T else (R.T @ R)
    vals = np.linalg.eigvalsh(G / T)[::-1]
    return np.clip(vals, 0.0, None)


def _extract_factors(Rs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Top-``k`` principal factors of the (possibly rescaled) panel ``Rs``.

    Returns ``(F, eigvals)`` with ``F`` (T x k) scaled so ``F'F / T = I_k`` and
    ``eigvals`` the full descending spectrum of ``Rs'Rs / T``.  Uses the
    ``T x T`` cross-product when ``N > T`` (Connor-Korajczyk), else ``N x N``."""
    T, N = Rs.shape
    k = int(min(k, T - 1, N))
    if N > T:
        Om = Rs @ Rs.T / N
        vals, vecs = np.linalg.eigh(Om)
        order = np.argsort(vals)[::-1]
        F = np.sqrt(T) * vecs[:, order[:k]]
        eig = np.clip(vals[order] * N / T, 0.0, None)      # spectrum of Rs'Rs/T
    else:
        S = Rs.T @ Rs / T
        vals, W = np.linalg.eigh(S)
        order = np.argsort(vals)[::-1]
        lam = np.clip(vals[order[:k]], 1e-30, None)
        F = (Rs @ W[:, order[:k]]) / np.sqrt(lam)           # F'F/T = I
        eig = np.clip(vals[order], 0.0, None)
    # sign convention: loadings of each factor sum to a positive number
    load = Rs.T @ F
    s = np.where(load.sum(axis=0) < 0, -1.0, 1.0)
    return F * s, eig


def _ts_regress(R: np.ndarray, mask: np.ndarray, F: np.ndarray, min_obs: int,
                ridge: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Per-stock OLS of the (zero-filled) columns of ``R`` on ``F`` using only
    observed rows.  Returns ``(B [N x k], ok [N])`` where ``ok`` marks stocks
    with at least ``min_obs`` observations; others get zero loadings."""
    T, N = R.shape
    k = F.shape[1]
    M = mask.astype(float)
    G = np.einsum("ti,tj,tn->nij", F, F, M)                 # N x k x k
    h = (R.T @ F)                                           # N x k (R is 0 where missing)
    n_obs = mask.sum(axis=0)
    ok = n_obs >= max(min_obs, k + 2)
    B = np.zeros((N, k))
    if ok.any():
        Gk = G[ok] + ridge * np.eye(k)[None, :, :] * np.maximum(n_obs[ok], 1)[:, None, None]
        B[ok] = np.linalg.solve(Gk, h[ok][:, :, None])[:, :, 0]
    return B, ok


def _weighted_resid_var(R: np.ndarray, mask: np.ndarray, F: np.ndarray, B: np.ndarray,
                        w: np.ndarray) -> np.ndarray:
    E = np.where(mask, R - F @ B.T, 0.0)
    wm = w[:, None] * mask
    den = wm.sum(axis=0)
    var = (wm * E ** 2).sum(axis=0) / np.where(den > 0, den, np.nan)
    return var


def _procrustes(F_new: np.ndarray, F_ref: np.ndarray) -> np.ndarray:
    """Orthogonal ``Q`` minimising ``||F_new Q - F_ref||_F``."""
    U, _, Vt = np.linalg.svd(F_new.T @ F_ref)
    return U @ Vt


# --------------------------------------------------------------------------- APCA core
@dataclass
class APCAResult:
    F: np.ndarray            # T x k factor returns, F'F/T = I (before rotation)
    B: np.ndarray            # N x k loadings
    specific_var: np.ndarray  # N
    eigvals: np.ndarray      # spectrum of the last (rescaled) panel
    ok: np.ndarray           # N bool: enough observations
    mu: np.ndarray           # N column means removed
    n_iter: int = 0


def apca(X: np.ndarray, n_factors: int, n_iter: int = 3, half_life: Optional[float] = None,
         min_stock_obs: int = 20) -> APCAResult:
    """Asymptotic PCA with heteroskedasticity iteration on a ``T x N`` panel
    (NaN allowed).  See the module docstring."""
    R, mask, mu = _prepare_panel(X)
    T, N = R.shape
    w = _weights(T, half_life)
    sigma = np.ones(N)
    F = B = None
    var = np.ones(N)
    eig = None
    ok = np.ones(N, dtype=bool)
    for it in range(max(1, int(n_iter) + 1)):
        Rs = R / sigma[None, :]
        F, eig = _extract_factors(Rs, n_factors)
        B, ok = _ts_regress(R, mask, F, min_stock_obs)
        var = _weighted_resid_var(R, mask, F, B, w)
        med = np.nanmedian(var[ok]) if ok.any() else 1.0
        var = np.where(np.isfinite(var) & (var > 0), var, med)
        sigma = np.sqrt(np.clip(var, 1e-3 * med, None))
    return APCAResult(F=F, B=B, specific_var=var, eigvals=eig, ok=ok, mu=mu, n_iter=n_iter)


# --------------------------------------------------------------------------- factor count
def marchenko_pastur_edge(eigvals: np.ndarray, N: int, T: int, max_factors: int = 20,
                          n_rounds: int = 10) -> Tuple[int, float, float]:
    """Count spikes above the Marchenko-Pastur edge.

    ``lambda_+ = sigma^2 (1 + sqrt(N/T))^2``; ``sigma^2`` is first the average
    eigenvalue, then re-estimated from the bulk (eigenvalues below the current
    edge, trace shared over the remaining ``N - k`` dimensions) until the
    count is stable.  Returns ``(k, edge, sigma2)``."""
    lam = np.asarray(eigvals, dtype=float)
    # effective dimension: residuals of a p-factor cross-sectional regression
    # live in an (N - p)-dimensional subspace, so exact-zero eigenvalues must
    # not dilute the noise-variance estimate
    tol = 1e-10 * max(float(lam[0]), 1e-300)
    N_eff = int(min(N, (lam > tol).sum()))
    lam = lam[:N_eff] if N_eff > 0 else lam[:1]
    N_eff = max(N_eff, 1)
    gamma = N_eff / float(T)
    sigma2 = float(lam.sum() / N_eff)
    k = 0
    for _ in range(n_rounds):
        edge = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
        k_new = int(min((lam > edge).sum(), max_factors))
        if k_new == k:
            break
        k = k_new
        sigma2 = float(lam[k:].sum() / max(N_eff - k, 1))
    edge = sigma2 * (1.0 + np.sqrt(gamma)) ** 2
    return k, float(edge), sigma2


def bai_ng_ic2(eigvals: np.ndarray, N: int, T: int, max_factors: int = 20) -> Tuple[int, np.ndarray]:
    """Bai & Ng (2002) ``IC_p2(k) = ln V(k) + k (N+T)/(NT) ln(min(N,T))`` with
    ``V(k) = (1/N) sum_{j>k} lambda_j`` (eigenvalues of ``R'R/T``).  Returns
    ``(argmin k, IC values for k = 0..max_factors)``."""
    lam = np.asarray(eigvals, dtype=float)
    kmax = int(min(max_factors, len(lam) - 1))
    tail = np.concatenate([[lam.sum()], lam.sum() - np.cumsum(lam[:kmax])])  # V(k)*N, k=0..kmax
    V = np.clip(tail / N, 1e-300, None)
    ks = np.arange(kmax + 1)
    pen = ks * (N + T) / (N * T) * np.log(min(N, T))
    ic = np.log(V) + pen
    return int(np.argmin(ic)), ic


def select_n_factors(returns, method: str = "bai_ng", max_factors: int = 20,
                     n_factors: Optional[int] = None, standardize: bool = True,
                     return_details: bool = False):
    """Choose the number of statistical factors for a ``T x N`` panel.

    ``method``: ``"bai_ng"`` (IC_p2), ``"marchenko_pastur"`` (spikes above the
    noise edge) or ``"fixed"`` (returns ``n_factors``).  Columns are
    standardized by their observed standard deviation by default so that the
    noise variance is comparable across stocks (``sigma^2 ~ 1``)."""
    if method == "fixed":
        if n_factors is None:
            raise ValueError("method='fixed' needs n_factors")
        return int(n_factors)
    X = returns.values if isinstance(returns, pd.DataFrame) else np.asarray(returns, dtype=float)
    R, mask, _ = _prepare_panel(X)
    T, N = R.shape
    if standardize:
        n = np.maximum(mask.sum(axis=0), 1)
        sd = np.sqrt((R ** 2).sum(axis=0) / n)
        R = R / np.where(sd > 0, sd, 1.0)
    lam = _spectrum(R)
    if method == "bai_ng":
        k, ic = bai_ng_ic2(lam, N, T, max_factors)
        details = {"ic": ic, "eigvals": lam}
    elif method == "marchenko_pastur":
        k, edge, s2 = marchenko_pastur_edge(lam, N, T, max_factors)
        details = {"edge": edge, "sigma2": s2, "eigvals": lam}
    else:
        raise ValueError(f"unknown method {method!r}")
    return (k, details) if return_details else k


# --------------------------------------------------------------------------- model
@dataclass
class _Refit:
    index: int                      # position in the returns index
    rows: slice                     # window rows
    F: np.ndarray                   # window x k (aligned)
    B: np.ndarray                   # N x k (aligned)
    ok: np.ndarray                  # N bool
    specific_var: np.ndarray        # N
    mu: np.ndarray                  # N
    eigvals: np.ndarray


class StatisticalRiskModel:
    """APCA statistical factor model with a point-in-time refit schedule.

    Parameters
    ----------
    n_factors : number of principal factors ``k``.
    window : trailing days used for every estimate.
    half_life : EWMA half-life for the specific variance and the factor
        covariance (``None`` = equal weights over the window).
    min_obs : first date with at least this many rows of history is the first
        date ``as_of`` can serve (and the anchor of the refit grid).
    method : ``"apca"`` (heteroskedasticity-iterated) or ``"pca"``
        (``apca_iters = 0``).
    apca_iters : number of re-scaling rounds.
    refit_every : full re-extraction frequency in trading days.
    min_stock_obs : a stock needs this many observed returns in the window to
        receive loadings; otherwise its loadings are 0 and its specific
        variance is the cross-sectional median.
    """

    def __init__(self, n_factors: int = 10, window: int = 252, half_life: Optional[float] = None,
                 min_obs: int = 200, method: str = "apca", apca_iters: int = 3,
                 refit_every: int = 21, min_stock_obs: Optional[int] = None):
        if method not in ("apca", "pca"):
            raise ValueError("method must be 'apca' or 'pca'")
        self.n_factors = int(n_factors)
        self.window = int(window)
        self.half_life = half_life
        self.min_obs = int(min_obs)
        self.method = method
        self.apca_iters = int(apca_iters) if method == "apca" else 0
        self.refit_every = max(1, int(refit_every))
        self.min_stock_obs = int(min_stock_obs) if min_stock_obs is not None else max(20, self.window // 6)
        self.returns: Optional[Wide] = None
        self.factor_names: List[str] = [f"{STAT_FACTOR_PREFIX}{j + 1:02d}" for j in range(self.n_factors)]
        self._refits: Dict[int, _Refit] = {}
        self._refit_positions: List[int] = []
        self.factor_returns: Optional[pd.DataFrame] = None   # real-time chain
        self._R = self._mask = None
        self._cache: Dict[pd.Timestamp, RiskModelOutput] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, returns: Wide, dates: Optional[Iterable[pd.Timestamp]] = None) -> "StatisticalRiskModel":
        """Precompute the refit grid on ``returns`` (date x ticker).  ``dates``
        optionally restricts how far the grid is computed (its max date); the
        grid itself is always anchored at the first eligible date."""
        self.returns = returns
        self._cache.clear()
        X = returns.values.astype(float)
        self._mask = np.isfinite(X)
        self._R = np.where(self._mask, X, 0.0)
        T_all = len(returns.index)
        i0 = self.min_obs - 1
        if T_all <= i0:
            raise ValueError(f"need at least {self.min_obs} rows, got {T_all}")
        last = T_all - 1
        if dates is not None:
            dts = pd.DatetimeIndex(list(dates))
            if len(dts):
                last = int(returns.index.get_indexer([dts.max()])[0])
                if last < 0:
                    last = T_all - 1
        self._refits.clear()
        self._refit_positions = list(range(i0, last + 1, self.refit_every))
        prev: Optional[_Refit] = None
        for i in self._refit_positions:
            prev = self._refit_at(i, prev)
            self._refits[i] = prev
        self._build_realtime_chain(i0, last)
        return self

    def _window_rows(self, i: int) -> slice:
        return slice(max(0, i - self.window + 1), i + 1)

    def _refit_at(self, i: int, prev: Optional[_Refit]) -> _Refit:
        rows = self._window_rows(i)
        X = self.returns.values[rows]
        res = apca(X, self.n_factors, self.apca_iters, self.half_life, self.min_stock_obs)
        F, B = res.F, res.B
        if prev is not None:
            lo = max(rows.start, prev.rows.start)
            hi = min(rows.stop, prev.rows.stop)
            if hi - lo >= self.n_factors + 5:
                Q = _procrustes(F[lo - rows.start:hi - rows.start], prev.F[lo - prev.rows.start:hi - prev.rows.start])
            else:  # no overlap: align on loadings of common stocks
                both = res.ok & prev.ok
                Q = _procrustes(B[both], prev.B[both]) if both.sum() >= self.n_factors + 5 else np.eye(self.n_factors)
            F, B = F @ Q, B @ Q
        return _Refit(index=i, rows=rows, F=F, B=B, ok=res.ok, specific_var=res.specific_var,
                      mu=res.mu, eigvals=res.eigvals)

    def _cs_factor_return(self, i: int, ref: _Refit) -> np.ndarray:
        """Cross-sectional OLS of day ``i`` demeaned returns on ``ref`` loadings."""
        avail = self._mask[i] & ref.ok
        if avail.sum() < self.n_factors + 2:
            return np.zeros(self.n_factors)
        y = self._R[i, avail] - ref.mu[avail]
        f, *_ = np.linalg.lstsq(ref.B[avail], y, rcond=None)
        return f

    def _last_refit(self, i: int) -> _Refit:
        pos = np.searchsorted(self._refit_positions, i, side="right") - 1
        if pos < 0:
            raise ValueError(f"no statistical fit available at position {i} (first at {self._refit_positions[0]})")
        return self._refits[self._refit_positions[pos]]

    def _build_realtime_chain(self, i0: int, last: int) -> None:
        """Real-time factor-return series: on refit days the last row of that
        refit's APCA solution, in between the cross-sectional estimate from the
        most recent refit's loadings."""
        out = np.full((last - i0 + 1, self.n_factors), np.nan)
        for i in range(i0, last + 1):
            ref = self._last_refit(i)
            out[i - i0] = ref.F[-1] if ref.index == i else self._cs_factor_return(i, ref)
        self.factor_returns = pd.DataFrame(out, index=self.returns.index[i0:last + 1], columns=self.factor_names)
        self.factor_returns.index.name = "date"

    # ------------------------------------------------------------------ as_of
    @property
    def first_date(self) -> pd.Timestamp:
        return self.returns.index[self._refit_positions[0]]

    def available(self, date: pd.Timestamp) -> bool:
        return self.returns is not None and pd.Timestamp(date) >= self.first_date

    def factor_history(self, date: pd.Timestamp) -> pd.DataFrame:
        """Factor returns ``<= date``: the last refit's full-window APCA
        solution, then daily cross-sectional estimates up to ``date``, preceded
        by the real-time chain for older days."""
        i = self._pos(date)
        ref = self._last_refit(i)
        idx = self.returns.index
        parts = []
        i0 = self._refit_positions[0]
        if ref.rows.start > i0:
            parts.append(self.factor_returns.iloc[: ref.rows.start - i0])
        parts.append(pd.DataFrame(ref.F, index=idx[ref.rows], columns=self.factor_names))
        if i > ref.index:
            parts.append(self.factor_returns.loc[idx[ref.index + 1]: idx[i]])
        hist = pd.concat(parts)
        hist.index.name = "date"
        return hist

    def _pos(self, date: pd.Timestamp) -> int:
        if self.returns is None:
            raise RuntimeError("call fit() first")
        date = pd.Timestamp(date)
        loc = self.returns.index.get_indexer([date])[0]
        if loc < 0:
            # most recent available date <= date
            loc = int(self.returns.index.searchsorted(date, side="right")) - 1
            if loc < 0:
                raise KeyError(f"no data on or before {date}")
        if loc < self._refit_positions[0]:
            raise ValueError(f"{date} precedes the first statistical fit date {self.first_date.date()}")
        if loc > self._refit_positions[0] + len(self.factor_returns) - 1:
            raise ValueError(f"{date} is beyond the fitted range; call fit() again")
        return int(loc)

    def as_of(self, date: pd.Timestamp) -> RiskModelOutput:
        """Statistical risk model as of ``date`` (only returns ``<= date``)."""
        date = pd.Timestamp(date)
        if date in self._cache:
            return self._cache[date]
        i = self._pos(date)
        ref = self._last_refit(i)
        hist = self.factor_history(date)
        rows = self._window_rows(i)
        idx = self.returns.index
        Fw = hist.loc[idx[rows.start]: idx[i]].values
        Xw = self.returns.values[rows]
        R, mask, mu = _prepare_panel(Xw)
        B, ok = _ts_regress(R, mask, Fw, self.min_stock_obs)
        w = _weights(R.shape[0], self.half_life)
        var = _weighted_resid_var(R, mask, Fw, B, w)
        med = float(np.nanmedian(var[ok])) if ok.any() else float(np.nanmedian(var))
        var = np.where(ok & np.isfinite(var) & (var > 0), var, med)
        # factor covariance: weighted covariance of the trailing window of factor returns
        Fc = Fw - w @ Fw
        S = nearest_psd((Fc * w[:, None]).T @ Fc)
        tickers = self.returns.columns
        exposures = pd.DataFrame(B, index=tickers, columns=self.factor_names)
        exposures.columns.name = "factor"
        out = RiskModelOutput(
            date=idx[i], exposures=exposures,
            factor_cov=pd.DataFrame(S, index=self.factor_names, columns=self.factor_names),
            specific_var=pd.Series(var, index=tickers, name="specific_var"),
            factor_returns=hist,
            factor_groups={"market": [], "industry": [], "style": [],
                           "statistical": [str(c) for c in self.factor_names], "alignment": []},
        )
        self._cache[date] = out
        return out

    # ------------------------------------------------------------------ diagnostics
    def refit_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([self.returns.index[i] for i in self._refit_positions])

    def refit_factor_returns(self, date: pd.Timestamp) -> pd.DataFrame:
        """The full-window APCA factor returns of the last refit on/before ``date``."""
        ref = self._last_refit(self._pos(date))
        return pd.DataFrame(ref.F, index=self.returns.index[ref.rows], columns=self.factor_names)

    def explained_share(self, date: pd.Timestamp) -> pd.Series:
        """Per-stock share of window variance explained by the factors."""
        out = self.as_of(date)
        B, F = out.exposures.values, out.factor_cov.values
        sysv = np.einsum("ik,kl,il->i", B, F, B)
        return pd.Series(sysv / (sysv + out.specific_var.values), index=out.tickers, name="explained_share")
