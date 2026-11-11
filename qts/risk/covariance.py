"""Covariance estimation for the risk model.

Contents
--------
* :func:`ewma_cov` -- exponentially weighted covariance with Newey-West
  (Bartlett kernel) autocovariance correction (USE4, "Factor covariance").
* :func:`factor_covariance` -- USE4 factor covariance pipeline: separate
  volatility and correlation half-lives, Volatility Regime Adjustment (VRA)
  and Eigenfactor Risk Adjustment.
* :func:`nonlinear_shrinkage` -- Ledoit & Wolf (2020, 2022) analytical
  nonlinear shrinkage ("quadratic-inverse shrinkage", QIS).
* :func:`linear_shrinkage` -- Ledoit & Wolf (2004) linear shrinkage (sklearn).
* :func:`specific_risk` -- USE4 specific-risk pipeline: EWMA+NW time-series
  vol, structural model blend for thin histories, Bayesian shrinkage toward
  the size-decile mean, floor / ceiling relative to the ESTU median, specific VRA.
* :func:`repair_covariance` -- finite / bounded check of the assembled
  covariance (the last line of defence in ``as_of``).

All variances are in **daily** units.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qts.core.config import RiskConfig
from qts.core.risk_fill import (  # noqa: F401  (re-exported)
    OUT_OF_MODEL_FALLBACK_VAR,
    OUT_OF_MODEL_SPECIFIC_VAR_PCTL,
    out_of_model_specific_var,
)
from qts.risk.exposures import split_factor_columns


# --------------------------------------------------------------------------- helpers
def half_life_to_lambda(half_life: float) -> float:
    return 0.5 ** (1.0 / float(half_life))


def ewma_weights(T: int, half_life: float) -> np.ndarray:
    """Normalized weights ``w_t ∝ lambda^(T-1-t)`` (most recent observation last)."""
    lam = half_life_to_lambda(half_life)
    w = lam ** np.arange(T - 1, -1, -1, dtype=float)
    return w / w.sum()


def nearest_psd(S: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Clip negative eigenvalues (Higham-lite).  Newey-West with EWMA weights
    is PSD only asymptotically, so we clip as a safeguard."""
    S = (S + S.T) / 2.0
    vals, vecs = np.linalg.eigh(S)
    if vals.min() >= eps:
        return S
    vals = np.maximum(vals, eps)
    return (vecs * vals) @ vecs.T


def _as_array(X) -> np.ndarray:
    return X.values if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X, dtype=float)


# --------------------------------------------------------------------------- EWMA + Newey-West
def ewma_cov(
    X, half_life: float, newey_west_lags: int = 0, demean: bool = True, ensure_psd: bool = True
) -> np.ndarray:
    """Exponentially weighted covariance of the rows of ``X`` (T x K) with a
    Newey-West (1987) correction for serial correlation:

        Sigma = Gamma_0 + sum_{l=1}^{L} (1 - l/(L+1)) (Gamma_l + Gamma_l')
        Gamma_l = sum_t w_t (x_t - mu)(x_{t-l} - mu)'

    with ``w_t ∝ lambda^(T-1-t)``, ``lambda = 0.5^(1/half_life)``.  The
    Bartlett weights ``1 - l/(L+1)`` are those used in USE4 for the factor
    covariance (5 lags for vol, 2 for correlation).  NaNs are treated as 0
    (missing contribution)."""
    A = _as_array(X).astype(float)
    if A.ndim == 1:
        A = A[:, None]
    A = np.where(np.isfinite(A), A, 0.0)
    T = A.shape[0]
    w = ewma_weights(T, half_life)
    if demean:
        A = A - w @ A
    S = (A * w[:, None]).T @ A
    L = int(newey_west_lags)
    for l in range(1, min(L, T - 1) + 1):
        G = (A[l:] * w[l:, None]).T @ A[:-l]
        S = S + (1.0 - l / (L + 1.0)) * (G + G.T)
    S = (S + S.T) / 2.0
    return nearest_psd(S) if ensure_psd else S


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(S), 1e-30, None))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


def combine_vol_corr(S_vol: np.ndarray, S_corr: np.ndarray) -> np.ndarray:
    """``F = diag(sigma) C diag(sigma)`` with ``sigma`` from the short-half-life
    estimate and ``C`` from the long-half-life estimate."""
    sig = np.sqrt(np.clip(np.diag(S_vol), 0.0, None))
    return nearest_psd(cov_to_corr(S_corr) * np.outer(sig, sig))


# --------------------------------------------------------------------------- VRA
def factor_vra_multiplier(
    F: pd.DataFrame,
    vol_half_life: float,
    vra_half_life: float,
    min_periods: int = 20,
    z2_clip: float | None = None,
    clip: tuple[float, float] | None = None,
) -> float:
    """Volatility Regime Adjustment (USE4 section "Volatility regime adjustment").

    Daily cross-sectional bias statistic of the factors
        B_t = sqrt( mean_k ( f_kt / sigma_{k,t-1} )^2 )
    where ``sigma_{k,t-1}`` is the EWMA (``vol_half_life``) forecast made at
    ``t-1``.  The factor volatility multiplier is
        lambda_F = sqrt( EWMA_{vra_half_life}( B_t^2 ) ),
    and the factor covariance is scaled by ``lambda_F^2``.  Returns 1.0 when
    there is too little history.

    Robustness (``cfg.vra_z2_clip`` / ``vra_min`` / ``vra_max`` via
    :func:`factor_covariance`): each factor's ``z^2`` is clipped at ``z2_clip``
    before the cross-sectional mean (a 5-sigma day counts as 5 sigma, not as
    the squared ratio of a return to a collapsed forecast) and ``lambda_F`` is
    clipped to ``clip``.  ``None`` = off (the pre-2026-11-10 behaviour)."""
    f2 = F.pow(2)
    # columns that are identically zero (industries with no members over the
    # whole history) carry no information; an exact 0.0 on a given day means
    # "no members that day" (the regression reports empty industries as 0) and
    # is excluded from that day's cross-sectional mean rather than counted as
    # a perfectly forecast zero, which would bias B_t (and lambda_F) down.
    f2 = f2.loc[:, (f2 != 0).any(axis=0)]
    if f2.shape[1] == 0:
        return 1.0
    sig2_prev = f2.ewm(halflife=vol_half_life, min_periods=min_periods).mean().shift(1)
    z2 = (f2 / sig2_prev.replace(0.0, np.nan)).where(f2 != 0)
    if z2_clip is not None and np.isfinite(z2_clip):
        z2 = z2.clip(upper=float(z2_clip))
    B2 = z2.mean(axis=1, skipna=True).dropna()
    if len(B2) < min_periods:
        return 1.0
    lam2 = B2.ewm(halflife=vra_half_life).mean().iloc[-1]
    lam = float(np.sqrt(max(lam2, 1e-6)))
    return _clip_multiplier(lam, clip)


def _clip_multiplier(lam: float, clip: tuple[float, float] | None) -> float:
    if clip is None:
        return lam
    lo, hi = float(clip[0]), float(clip[1])
    return float(min(max(lam, lo), hi))


def vra_clip_from(cfg: RiskConfig) -> tuple[float | None, tuple[float, float] | None]:
    """``(z2_clip, (vra_min, vra_max))`` for the VRA functions from ``cfg``
    (``None`` where the config switches the clip off)."""
    z2 = float(getattr(cfg, "vra_z2_clip", np.inf))
    lo = float(getattr(cfg, "vra_min", 0.0))
    hi = float(getattr(cfg, "vra_max", np.inf))
    return (z2 if np.isfinite(z2) else None), ((lo, hi) if (lo > 0 or np.isfinite(hi)) else None)


# --------------------------------------------------------------------------- eigenfactor adjustment
def eigenfactor_bias(
    F0: np.ndarray,
    estimator: Callable[[np.ndarray], np.ndarray],
    T: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """USE4 Eigenfactor Risk Adjustment, Monte-Carlo step.

    1. ``F0 = U D U'``; eigenfactors are uncorrelated with variances ``D``.
    2. For each simulation draw ``T`` iid returns ``b ~ N(0, D)`` of the
       eigenfactors, rotate to factor space ``f = U b`` and re-estimate the
       covariance with the *same* estimator -> ``F_m = U_m D_m U_m'``.
    3. The true variance of the simulated eigenfactor ``k`` is
       ``d~_mk = (U_m' F0 U_m)_kk``; the simulated bias is
       ``v_k^2 = mean_m d~_mk / D_mk``.
    4. The adjusted covariance is ``U diag(v_k^2 D_k) U'`` (sample eigenvariances
       are too small for the small-eigenvalue eigenfactors and too large for
       the large ones; this de-biases them).

    This returns the **raw** simulated ``v_k`` (unclipped), in
    ascending-eigenvalue order; the [0.5, 2] safety clip is applied once, after
    the USE4 smoothing / inflation (second-pass audit finding 9: clipping the
    raw ratio first would change what the inflation acts on).  USE4 then optionally smooths ``v`` across ``k``
    with a parabola and inflates it by the empirical factor ``a`` (1.2):
    :func:`scale_eigen_bias`, applied inside :func:`factor_covariance` from
    ``cfg.eigen_inflation_a`` / ``cfg.eigen_parabola_smoothing``.
    Returns ``v`` (length K)."""
    vals, U = np.linalg.eigh(F0)
    vals = np.clip(vals, 1e-18, None)
    K = len(vals)
    ratio = np.zeros(K)
    sd = np.sqrt(vals)
    for _ in range(n_sims):
        b = rng.standard_normal((T, K)) * sd
        f = b @ U.T
        Fm = estimator(f)
        dm, Um = np.linalg.eigh(Fm)
        dm = np.clip(dm, 1e-18, None)
        d_true = np.einsum("ik,ij,jk->k", Um, F0, Um)
        ratio += d_true / dm
    return np.sqrt(ratio / n_sims)


def scale_eigen_bias(
    v: np.ndarray, a: float = 1.2, parabola: bool = False, clip: tuple[float, float] = (0.5, 2.0)
) -> np.ndarray:
    """USE4 post-processing of the simulated eigenfactor bias ``v`` (one entry
    per eigenfactor, ascending eigenvalue order):

    1. optional parabola smoothing ``v_P(k) = c0 + c1 k + c2 k^2`` (least
       squares over ``k = 0..K-1``), which removes the Monte Carlo noise of the
       individual ratios;
    2. empirical inflation ``v_S(k) = a (v_P(k) - 1) + 1`` -- the simulated
       bias is too small because simulated returns are Gaussian and stationary
       while real ones are not; USE4 calibrates ``a = 1.2``.

    ``a = 1`` and ``parabola=False`` return ``v`` unchanged.  The result is
    clipped to ``clip`` (default [0.5, 2]) and is the *volatility* multiplier:
    eigenvariances are scaled by ``v_S^2``."""
    v = np.asarray(v, dtype=float)
    out = v.copy()
    K = len(v)
    if parabola and K >= 3:
        k = np.arange(K, dtype=float)
        coef = np.polyfit(k, v, 2)
        out = np.polyval(coef, k)
    out = float(a) * (out - 1.0) + 1.0
    return np.clip(out, clip[0], clip[1])


def apply_eigen_adjustment(F0: np.ndarray, v: np.ndarray) -> np.ndarray:
    vals, U = np.linalg.eigh(F0)
    vals = np.clip(vals, 0.0, None) * v**2
    return (U * vals) @ U.T


@dataclass
class FactorCovarianceResult:
    cov: pd.DataFrame
    vra_multiplier: float
    eigen_bias: np.ndarray | None  # raw simulated bias (reusable across dates)
    raw_cov: pd.DataFrame
    eigen_bias_scaled: np.ndarray | None = None  # after parabola / inflation (what was applied)


def factor_covariance(
    factor_returns: pd.DataFrame,
    cfg: RiskConfig | None = None,
    eigen_bias: np.ndarray | None = None,
    seed: int = 0,
    return_details: bool = False,
):
    """USE4 factor covariance as of the last row of ``factor_returns``.

    Steps: (i) EWMA+NW with ``vol_half_life`` / ``newey_west_lags_vol`` gives
    the variances; (ii) EWMA+NW with ``corr_half_life`` / ``newey_west_lags_corr``
    gives the correlations; (iii) combine; (iv) eigenfactor risk adjustment
    (``cfg.eigenfactor_adjust``, ``cfg.n_eigen_sims`` simulations; pass a
    precomputed *raw* ``eigen_bias`` vector to reuse it across dates), with the
    USE4 parabola smoothing / inflation ``a`` applied to the raw bias
    (``cfg.eigen_parabola_smoothing``, ``cfg.eigen_inflation_a``,
    :func:`scale_eigen_bias`); (v) multiply by
    the VRA factor ``lambda_F^2`` (``cfg.vra_half_life``).

    Returns a DataFrame (factor x factor) or, with ``return_details=True``, a
    :class:`FactorCovarianceResult`."""
    cfg = cfg or RiskConfig()
    F = factor_returns.dropna(how="all").fillna(0.0)
    cols = F.columns
    X = F.values
    S_vol = ewma_cov(X, cfg.vol_half_life, cfg.newey_west_lags_vol)
    S_corr = ewma_cov(X, cfg.corr_half_life, cfg.newey_west_lags_corr)
    F0 = combine_vol_corr(S_vol, S_corr)
    raw = pd.DataFrame(F0, index=cols, columns=cols)

    v = None
    v_scaled = None
    if cfg.eigenfactor_adjust:
        if eigen_bias is None:

            def est(Y):
                return combine_vol_corr(
                    ewma_cov(Y, cfg.vol_half_life, 0, demean=False, ensure_psd=False),
                    ewma_cov(Y, cfg.corr_half_life, 0, demean=False, ensure_psd=False),
                )

            T_sim = int(min(len(X), 4 * cfg.corr_half_life))
            v = eigenfactor_bias(F0, est, T_sim, cfg.n_eigen_sims, np.random.default_rng(seed))
        else:
            v = np.asarray(eigen_bias, dtype=float)
        v_scaled = scale_eigen_bias(
            v, getattr(cfg, "eigen_inflation_a", 1.0), getattr(cfg, "eigen_parabola_smoothing", False)
        )
        F0 = apply_eigen_adjustment(F0, v_scaled)

    z2_clip, lam_clip = vra_clip_from(cfg)
    lam = factor_vra_multiplier(F, cfg.vol_half_life, cfg.vra_half_life, z2_clip=z2_clip, clip=lam_clip)
    Fc = nearest_psd(F0 * lam**2)
    cov = pd.DataFrame(Fc, index=cols, columns=cols)
    if return_details:
        return FactorCovarianceResult(
            cov=cov, vra_multiplier=lam, eigen_bias=v, raw_cov=raw, eigen_bias_scaled=v_scaled
        )
    return cov


# --------------------------------------------------------------------------- shrinkage
def nonlinear_shrinkage(X: np.ndarray, assume_centered: bool = False) -> np.ndarray:
    """Analytical nonlinear shrinkage -- quadratic-inverse shrinkage (QIS).

    Port of Ledoit & Wolf, "Quadratic Shrinkage for Large Covariance
    Matrices" (Bernoulli 2022; working paper 2020), MATLAB routine ``QIS.m``.
    ``X`` is a *data matrix* (n observations x p variables); the function
    forms the sample covariance, eigen-decomposes it, estimates the limiting
    spectral density via a kernel estimator of the Stieltjes transform (the
    "smoothed Stein shrinker" ``theta`` and its Hilbert-transform conjugate
    ``Htheta``, bandwidth ``h = min(c^2, 1/c^2)^0.35 / p^0.35``), and replaces
    each sample eigenvalue ``lambda_i`` by

        delta_i = 1 / ( (1-c)^2/lambda_i + 2c(1-c) theta_i/lambda_i + c^2 (theta_i^2 + Htheta_i^2)/lambda_i )

    (case ``p <= n``) -- the asymptotically optimal rotation-equivariant
    estimator under the quadratic-inverse (Frobenius-on-inverse / Stein)
    loss.  Eigenvectors are kept; the trace is preserved.  Handles ``p > n``
    with the singular-case formula from the same paper."""
    Y = np.asarray(X, dtype=float)
    n, p = Y.shape
    if not assume_centered:
        Y = Y - Y.mean(axis=0, keepdims=True)
        n = n - 1
    c = p / n
    sample = (Y.T @ Y) / n
    sample = (sample + sample.T) / 2.0
    lam, u = np.linalg.eigh(sample)  # ascending
    lam = np.clip(lam, 0.0, None)
    h = min(c**2, 1.0 / c**2) ** 0.35 / p**0.35
    nz = lam[max(0, p - n) :]  # non-null eigenvalues
    invlam = 1.0 / np.clip(nz, 1e-300, None)
    Lj = np.tile(invlam, (len(invlam), 1))  # Lj[i, j] = 1/lambda_j
    Lj_i = Lj - Lj.T  # (1/lambda_j) - (1/lambda_i)
    denom = Lj_i**2 + (h**2) * Lj**2
    theta = np.mean(Lj * Lj_i / denom, axis=1)
    Htheta = np.mean(Lj * (h * Lj) / denom, axis=1)
    Atheta2 = theta**2 + Htheta**2
    if p <= n:
        delta = 1.0 / ((1 - c) ** 2 * invlam + 2 * c * (1 - c) * invlam * theta + c**2 * invlam * Atheta2)
    else:
        delta0 = 1.0 / ((c - 1) * np.mean(invlam))
        delta = np.concatenate([np.full(p - n, delta0), 1.0 / (invlam * Atheta2)])
    delta = delta * (lam.sum() / delta.sum())  # preserve trace
    return (u * delta) @ u.T


def linear_shrinkage(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf (2004) linear shrinkage toward scaled identity via sklearn."""
    from sklearn.covariance import LedoitWolf

    return LedoitWolf(assume_centered=False).fit(np.asarray(X, dtype=float)).covariance_


def shrunk_residual_covariance(
    specific_returns: pd.DataFrame,
    cfg: RiskConfig | None = None,
    window: int | None = None,
    method: str = "qis",
) -> pd.DataFrame:
    """Full (non-diagonal) residual covariance of the last ``window`` days of
    specific returns using nonlinear (``method="qis"``, default) or linear
    (``"lw"``) shrinkage.  Missing residuals are set to 0.

    **Evaluation only** (e.g. GMV tests, residual-structure diagnostics):
    ``RiskModelOutput`` keeps only the diagonal ``specific_var`` from
    :func:`specific_risk`, which this function does not influence.  Using the
    QIS diagonal as the thin-history prior instead of the USE4 structural model
    was tested and rejected (out-of-sample QLIKE 0.07 structural vs 0.15 QIS
    with a coverage correction / 0.45 without, synthetic seeds 0-2; finding #16)."""
    cfg = cfg or RiskConfig()
    if method not in ("qis", "lw"):
        raise ValueError("method must be 'qis' or 'lw'")
    window = window or max(4 * cfg.specific_half_life, 252)
    U = specific_returns.tail(window)
    U = U.loc[:, U.notna().sum() > 0].fillna(0.0)
    S = nonlinear_shrinkage(U.values) if method == "qis" else linear_shrinkage(U.values)
    return pd.DataFrame(S, index=U.columns, columns=U.columns)


# --------------------------------------------------------------------------- specific risk
def ewma_nw_variance(U: pd.DataFrame, half_life: float, newey_west_lags: int) -> tuple[pd.Series, pd.Series]:
    """Per-column EWMA variance with Newey-West lags (own autocovariances only).
    Returns ``(variance, n_obs)``; NaNs contribute nothing and weights are
    renormalized per column over observed days.  The NW-adjusted variance is
    floored at 25% of the lag-0 variance so it stays positive."""
    A = U.values.astype(float)
    T = A.shape[0]
    mask = np.isfinite(A)
    A0 = np.where(mask, A, 0.0)
    w = ewma_weights(T, half_life)[:, None]
    wsum = (w * mask).sum(axis=0)
    wsum = np.where(wsum > 0, wsum, np.nan)
    mu = (w * A0).sum(axis=0) / wsum
    D = np.where(mask, A0 - mu, 0.0)
    g0 = (w * D**2).sum(axis=0) / wsum
    var = g0.copy()
    L = int(newey_west_lags)
    for l in range(1, min(L, T - 1) + 1):
        gl = (w[l:] * D[l:] * D[:-l]).sum(axis=0) / wsum
        var = var + 2.0 * (1.0 - l / (L + 1.0)) * gl
    var = np.maximum(var, 0.25 * g0)
    return (pd.Series(var, index=U.columns), pd.Series(mask.sum(axis=0), index=U.columns))


def specific_vra_multiplier(
    U: pd.DataFrame,
    market_cap: pd.Series | pd.DataFrame,
    half_life: float,
    vra_half_life: float,
    min_periods: int = 20,
    z2_clip: float | None = None,
    clip: tuple[float, float] | None = None,
    floor_frac: float = 0.0,
) -> float:
    """Specific-risk VRA (USE4 Methodology Notes, "Specific risk volatility
    regime adjustment"): cap-weighted cross-sectional bias statistic

        B_t^2 = sum_i w_it (u_it / sigma_{i,t-1})^2 / sum_i w_it,

    then ``lambda_S = sqrt(EWMA_{vra}(B_t^2))``.

    ``w_it``: with a ``market_cap`` *panel* (date x ticker) the cap of the
    previous trading day ``t-1`` (USE4 weights each day's statistic by that
    day's caps; ``t-1`` keeps it point-in-time, matching the regression
    convention), falling back to the latest cap where missing; with a Series
    (older callers) today's cap on every day.

    Documented deviation from USE4: ``sigma_{i,t-1}`` is the per-stock
    EWMA(u^2) forecast made at ``t-1``, not the final (structural-blended,
    Bayesian-shrunk) specific forecast -- re-running the full specific-risk
    pipeline for every historical day would multiply the cost of ``as_of`` by
    the VRA window.  The EWMA forecast is the time-series component of the
    final forecast, so the bias statistic measures the same regime shift.

    Robustness (``docs/notes/risk_thin_industry_fix_2026-11-10.md``): a name
    whose residual was exactly zero for a stretch (single-member industry) has
    ``sigma_{i,t-1} ~ 1e-16``; its first real residual gives ``z^2 ~ 1e26`` and
    the cap-weighted average -- applied to *every* name -- explodes.  With
    ``floor_frac > 0`` names whose forecast variance sits below ``floor_frac``
    times that day's cross-sectional median forecast are excluded from the
    statistic; each remaining ``z^2`` is clipped at ``z2_clip`` and
    ``lambda_S`` at ``clip``.  All three default to off (bitwise the earlier
    estimator); :func:`specific_risk` passes the ``RiskConfig`` values."""
    A = U.values.astype(float)
    T = A.shape[0]
    if min_periods + 1 >= T:
        return 1.0
    u2 = pd.DataFrame(A**2)
    s2_prev = u2.ewm(halflife=half_life, min_periods=min_periods, ignore_na=True).mean().shift(1).values
    z2 = A**2 / np.where(s2_prev > 0, s2_prev, np.nan)
    if floor_frac > 0:
        pos = np.where(s2_prev > 0, s2_prev, np.nan)
        med = np.full(T, np.nan)
        has = np.isfinite(pos).any(axis=1)  # all-NaN rows (warm-up) stay NaN without a warning
        if has.any():
            med[has] = np.nanmedian(pos[has], axis=1)
        with np.errstate(invalid="ignore"):
            at_floor = s2_prev < (floor_frac * med)[:, None]
        z2 = np.where(at_floor, np.nan, z2)
    if z2_clip is not None and np.isfinite(z2_clip):
        z2 = np.minimum(z2, float(z2_clip))
    if isinstance(market_cap, pd.DataFrame):
        w = _vra_cap_weights(market_cap, U)
    else:
        w = np.broadcast_to(market_cap.reindex(U.columns).fillna(0.0).clip(lower=0.0).values, A.shape)
    ok = np.isfinite(z2)
    wsum = (ok * w).sum(axis=1)
    B2 = np.where(wsum > 0, (np.where(ok, z2, 0.0) * w).sum(axis=1) / np.where(wsum > 0, wsum, 1.0), np.nan)
    B2 = B2[np.isfinite(B2)]
    if len(B2) < min_periods:
        return 1.0
    lam2 = float(ewma_weights(len(B2), vra_half_life) @ B2)
    return _clip_multiplier(float(np.sqrt(np.clip(lam2, 1e-6, None))), clip)


def _vra_cap_weights_pandas(market_cap: pd.DataFrame, U: pd.DataFrame) -> np.ndarray:
    """Reference (pandas) form of :func:`_vra_cap_weights`; used when the fast
    path's preconditions do not hold and by the equivalence test."""
    latest = market_cap.reindex(columns=U.columns).ffill().iloc[-1] if len(market_cap) else None
    panel = market_cap.reindex(columns=U.columns).shift(1).reindex(U.index)
    if latest is not None:
        panel = panel.fillna(latest)
    return panel.fillna(0.0).clip(lower=0.0).values


def _vra_cap_weights(market_cap: pd.DataFrame, U: pd.DataFrame) -> np.ndarray:
    """Weights ``w_it`` of :func:`specific_vra_multiplier` from a cap panel:
    the cap of the previous row of ``market_cap`` for each row of ``U``, then
    the latest non-missing cap, then 0, floored at 0.

    NumPy form of :func:`_vra_cap_weights_pandas` (bitwise identical, same
    memory layout; the pandas chain forward-filled the *whole* cap history and
    split the frame into one block per column, ~0.2-0.7 s per ``as_of`` at
    300-1000 names).
    Falls back to the pandas form for an empty or non-float64 panel or
    non-unique labels."""
    sub = market_cap.reindex(columns=U.columns)
    if (
        len(market_cap) == 0
        or not market_cap.index.is_unique
        or not U.columns.is_unique
        or not all(dt == np.float64 for dt in sub.dtypes)
    ):
        return _vra_cap_weights_pandas(market_cap, U)
    M = sub.to_numpy(dtype=np.float64)
    n_rows, n_cols = M.shape
    pos = market_cap.index.get_indexer(U.index)
    # shift(1) then reindex(U.index): row of U at position p of the cap index gets
    # cap row p - 1; p == 0 or a label absent from the cap index -> NaN
    prev = pos - 1
    have = pos > 0
    panel = np.full((len(U.index), n_cols), np.nan)
    if have.any():
        panel[have] = M[prev[have]]
    valid = ~np.isnan(M)
    any_valid = valid.any(axis=0)
    last = n_rows - 1 - np.argmax(valid[::-1], axis=0)
    latest = np.where(any_valid, M[last, np.arange(n_cols)], np.nan)
    panel = np.where(np.isnan(panel), latest[None, :], panel)
    panel = np.where(np.isnan(panel), 0.0, panel)
    # Fortran order like DataFrame.values: the row sums downstream are
    # layout-sensitive in the last bit (numpy's pairwise summation)
    return np.asfortranarray(np.where(panel >= 0.0, panel, 0.0))


def structural_specific_vol(
    sigma_ts: pd.Series,
    gamma: pd.Series,
    exposures: pd.DataFrame,
    market_cap: pd.Series,
    clip_pctl: float = 0.0,
) -> pd.Series:
    """USE4 structural model: regress ``ln sigma_TS`` on the factor exposures
    (WLS, weights ∝ sqrt(cap)) over stocks with full history (``gamma == 1``),
    then ``sigma_STR = E0 * exp(X b)`` with ``E0`` the bias correction making
    the fitted values unbiased in levels.

    ``clip_pctl = p > 0`` clips the prediction to the ``[p, 1 - p]`` quantile
    band of the ``sigma_TS`` it was fitted on.  ``exp`` of a linear
    extrapolation has no upper bound: a name whose exposure sits far outside the
    fitted cross-section (a gross descriptor outlier) can be assigned a vol of
    ``1e55`` (docs/notes/risk_new_listing_fix_2026-11-10.md).  The structural
    value is the prior for a name without history; a prior beyond every
    seasoned name is not information.  A non-finite prediction takes the
    fit-set median."""
    idx = sigma_ts.index
    X = exposures.reindex(idx).fillna(0.0).values
    fit_mask = (
        (gamma.reindex(idx).fillna(0.0).values >= 1.0) & np.isfinite(sigma_ts.values) & (sigma_ts.values > 0)
    )
    if fit_mask.sum() < X.shape[1] + 5:
        fit_mask = np.isfinite(sigma_ts.values) & (sigma_ts.values > 0)
    if fit_mask.sum() < 5:
        return pd.Series(np.nanmedian(sigma_ts.values), index=idx)
    y = np.log(sigma_ts.values[fit_mask])
    w = np.sqrt(market_cap.reindex(idx).fillna(market_cap.median()).clip(lower=1e-9).values[fit_mask])
    sw = np.sqrt(w)[:, None]
    b, *_ = np.linalg.lstsq(X[fit_mask] * sw, y * sw[:, 0], rcond=None)
    if clip_pctl > 0:
        with np.errstate(over="ignore", invalid="ignore"):
            fitted = np.exp(X @ b)
    else:
        fitted = np.exp(X @ b)
    E0 = float(np.sum(w * sigma_ts.values[fit_mask]) / np.sum(w * fitted[fit_mask]))
    pred = E0 * fitted
    if clip_pctl > 0:
        obs = sigma_ts.to_numpy(dtype=float)[fit_mask]
        lo, hi = np.quantile(obs, [float(clip_pctl), 1.0 - float(clip_pctl)])
        pred = np.where(np.isfinite(pred), pred, float(np.median(obs)))
        pred = np.clip(pred, lo, hi)
    return pd.Series(pred, index=idx)


def bayesian_shrink_specific(
    sigma: pd.Series, market_cap: pd.Series, q: float, n_deciles: int = 10, return_target: bool = False
):
    """USE4 Bayesian shrinkage of specific risk toward the cap-weighted mean of
    its size decile:

        sigma_SH = v sigma_bar + (1 - v) sigma,
        v = q |sigma - sigma_bar| / ( Delta + q |sigma - sigma_bar| ),

    where ``sigma_bar`` is the cap-weighted mean and ``Delta`` the dispersion of
    ``sigma`` *about that cap-weighted mean* within the decile,
    ``Delta = sqrt(mean_n (sigma_n - sigma_bar)^2)`` (USE4 Methodology Notes,
    "Bayesian shrinkage"; second-pass audit finding 9 -- it was the
    equal-weighted sd, i.e. about the equal-weighted mean).

    ``sigma_bar`` is an arithmetic mean: it has no defence against one huge
    member, which then pulls every other name of the decile to (nearly) the
    same value.  :func:`specific_risk` therefore bounds the input
    (``cfg.specific_var_cap_mult``) before calling this.  With
    ``return_target`` also returns each name's ``sigma_bar``."""
    cap = market_cap.reindex(sigma.index).fillna(market_cap.median()).clip(lower=1e-9).values
    s = sigma.values.astype(float)
    n = len(s)
    k = max(1, min(n_deciles, n // 10))
    order = np.argsort(cap, kind="stable")
    dec = np.empty(n, dtype=int)
    dec[order] = (np.arange(n) * k) // n
    out = s.copy()
    target = np.full(n, np.nan)
    for d in range(k):
        m = dec == d
        if not m.any():
            continue
        sd_, cd = s[m], cap[m]
        sbar = float((sd_ * cd).sum() / cd.sum())
        target[m] = sbar
        delta = float(np.sqrt(np.mean((sd_ - sbar) ** 2)))
        dev = np.abs(sd_ - sbar)
        den = delta + q * dev
        v = np.where(den > 0, q * dev / np.where(den > 0, den, 1.0), 0.0)
        out[m] = v * sbar + (1 - v) * sd_
    shrunk = pd.Series(out, index=sigma.index)
    if return_target:
        return shrunk, pd.Series(target, index=sigma.index)
    return shrunk


def _column_quantiles(U: pd.DataFrame, q: float) -> pd.Series:
    """``U.quantile(q)`` (linear interpolation, NaNs skipped) for a float64
    frame, bitwise identical to pandas: pandas calls ``np.quantile`` once per
    column when any value is missing; here the columns are sorted once (NaN
    last) and every group of columns with the same number of valid values is
    passed to ``np.quantile(..., axis=0)`` together (same order statistics,
    same interpolation arithmetic).  Other dtypes use pandas."""
    if not all(dt == np.float64 for dt in U.dtypes) or U.shape[1] == 0:
        return U.quantile(q)
    A = U.to_numpy(dtype=np.float64)
    n, m = A.shape
    out = np.full(m, np.nan)
    if n:
        cnt = n - np.isnan(A).sum(axis=0)
        S = np.sort(A, axis=0)  # NaN sorted last
        for k in np.unique(cnt):
            if k == 0:
                continue
            cols = np.flatnonzero(cnt == k)
            out[cols] = np.quantile(S[:k, cols], np.asarray([q], dtype=np.float64), axis=0, method="linear")[
                0
            ]
    return pd.Series(out, index=U.columns, name=q)


def specific_risk(
    specific_returns: pd.DataFrame,
    exposures_latest: pd.DataFrame,
    market_cap: pd.Series,
    cfg: RiskConfig | None = None,
    return_details: bool = False,
    market_cap_history: pd.DataFrame | None = None,
) -> pd.Series:
    """USE4 specific-risk forecast (daily **variance**) per ticker.

    1. Time-series: EWMA (``specific_half_life``) + Newey-West
       (``newey_west_lags_vol``) variance of each stock's specific returns.
    2. Thin-history blend: ``gamma_i = clip((h_i - 60)/120, 0, 1) *
       min(1, exp(1 - Z_i))`` where ``h_i`` is the number of observations and
       ``Z_i = |sd_eq / sd_robust - 1|`` with ``sd_robust = IQR / 1.35``
       (USE4 eq. for the "delta" blending weight); ``sigma = gamma sigma_TS +
       (1 - gamma) sigma_STR`` with the structural model of
       :func:`structural_specific_vol`.
    3. Bayesian shrinkage toward the size-decile mean with ``q = specific_shrink_q``.
    4. Specific-variance floor (``cfg.specific_var_floor_frac``): the shrunk
       variance is floored at ``floor_frac`` times its cross-sectional median
       over the names observed in the window (the ESTU); the raw EWMA input of
       step 1 is floored the same way *before* the structural fit, so a
       residual series that is identically zero (single-member industry) enters
       as a small-but-real vol, never as ``1e-16``.
    5. Specific-variance ceiling (``cfg.specific_var_cap_mult``), symmetric to
       the floor: variance ``<= cap_mult`` times the ESTU median, applied to the
       raw EWMA input, to the blended value **before** the Bayesian shrinkage
       (the size-decile target is a cap-weighted *mean*; one unbounded member
       drags every other name of its decile to the same huge value) and to the
       final shrunk forecast.
    6. Specific VRA multiplier ``lambda_S`` (daily cap weights from
       ``market_cap_history`` when given, see :func:`specific_vra_multiplier`;
       names at the floor are excluded from the bias statistic, ``z^2`` clipped
       at ``cfg.vra_z2_clip``, ``lambda_S`` clipped to ``[vra_min, vra_max]``).

    Seasoning: ``cfg.min_specific_obs`` (60) is the ``h_i`` below which
    ``gamma_i = 0`` -- the name takes the structural value alone, clipped to the
    ``[p, 1 - p]`` band of seasoned names (``cfg.structural_clip_pctl``).

    Returns ``(lambda_S * sigma_SH)^2`` indexed like ``exposures_latest``;
    with ``return_details`` also ``n_at_floor`` / ``n_at_cap`` (names whose
    variance was moved to the bound), ``floor_var`` / ``cap_var`` (bounds on
    the shrunk variance, before VRA), ``var_bounds`` (the same bounds on the
    returned variance, i.e. times ``lambda_S^2``; ``None`` where a bound is
    off), ``var_ts_raw`` (EWMA variance before the floor / cap) and
    ``shrink_target`` (each name's size-decile mean vol)."""
    cfg = cfg or RiskConfig()
    tickers = exposures_latest.index
    window = int(max(6 * cfg.specific_half_life, cfg.min_history_days))
    U = specific_returns.reindex(columns=tickers).tail(window)
    var_ts, n_obs = ewma_nw_variance(U, cfg.specific_half_life, cfg.newey_west_lags_vol)
    var_ts_raw = var_ts
    floor_frac = float(getattr(cfg, "specific_var_floor_frac", 0.0))
    cap_mult = float(getattr(cfg, "specific_var_cap_mult", np.inf))
    has_cap = bool(np.isfinite(cap_mult))
    estu = n_obs > 0
    capped = pd.Series(False, index=var_ts.index)  # names the ceiling acted on, at any stage
    if (floor_frac > 0 or has_cap) and estu.any():
        med_ts = float(np.nanmedian(var_ts[estu].to_numpy(dtype=float)))
        if np.isfinite(med_ts) and med_ts > 0:
            if floor_frac > 0:
                var_ts = var_ts.where(~(var_ts < floor_frac * med_ts), floor_frac * med_ts)
            if has_cap:
                capped = (var_ts > cap_mult * med_ts).fillna(False)
                var_ts = var_ts.where(~capped, cap_mult * med_ts)
    sigma_ts = np.sqrt(var_ts)

    # robust-vs-equal sd ratio for outlier damping (USE4: sigma_robust = IQR/1.35,
    # Z = |sigma_eq / sigma_robust - 1|, gamma = clip((h-60)/120) * min(1, exp(1 - Z)))
    q75, q25 = _column_quantiles(U, 0.75), _column_quantiles(U, 0.25)
    sd_robust = ((q75 - q25) / 1.35).replace(0.0, np.nan)
    sd_eq = U.std(ddof=0)
    Z = (sd_eq / sd_robust - 1.0).abs().fillna(0.0)
    min_obs = int(getattr(cfg, "min_specific_obs", 60))
    gamma = ((n_obs - min_obs) / 120.0).clip(0.0, 1.0) * np.minimum(1.0, np.exp(1.0 - Z))
    gamma = gamma.fillna(0.0)

    sigma_str = structural_specific_vol(
        sigma_ts,
        gamma,
        exposures_latest,
        market_cap,
        clip_pctl=float(getattr(cfg, "structural_clip_pctl", 0.0)),
    )
    sigma = gamma * sigma_ts.fillna(0.0) + (1 - gamma) * sigma_str
    sigma = sigma.where(sigma > 0, sigma_str)

    if has_cap and estu.any():
        # bound the shrinkage *input*: the decile target below is a cap-weighted mean
        med_bl = float(np.nanmedian((sigma[estu] ** 2).to_numpy(dtype=float)))
        if np.isfinite(med_bl) and med_bl > 0:
            over = ~(sigma**2 <= cap_mult * med_bl) & sigma.notna()
            if over.any():
                capped = capped | over
                sigma = sigma.where(~over, np.sqrt(cap_mult * med_bl))

    sigma_sh, target = bayesian_shrink_specific(sigma, market_cap, cfg.specific_shrink_q, return_target=True)
    n_at_floor = 0
    floor_var = 0.0
    cap_var = float("inf")
    if (floor_frac > 0 or has_cap) and estu.any():
        med_sh = float(np.nanmedian((sigma_sh[estu] ** 2).to_numpy(dtype=float)))
        if np.isfinite(med_sh) and med_sh > 0:
            if floor_frac > 0:
                floor_var = floor_frac * med_sh
                below = (sigma_sh**2 < floor_var).fillna(False)
                n_at_floor = int(below.sum())
                sigma_sh = sigma_sh.where(~below, np.sqrt(floor_var))
            if has_cap:
                cap_var = cap_mult * med_sh
                above = (sigma_sh**2 > cap_var).fillna(False)
                if above.any():
                    capped = capped | above
                    sigma_sh = sigma_sh.where(~above, np.sqrt(cap_var))
    n_at_cap = int(capped.sum())
    z2_clip, lam_clip = vra_clip_from(cfg)
    lam = specific_vra_multiplier(
        U,
        market_cap_history if market_cap_history is not None else market_cap,
        cfg.specific_half_life,
        cfg.vra_half_life,
        z2_clip=z2_clip,
        clip=lam_clip,
        floor_frac=floor_frac,
    )
    var = (lam * sigma_sh) ** 2
    var.name = "specific_var"
    if return_details:
        return var, {
            "sigma_ts": sigma_ts,
            "sigma_str": sigma_str,
            "gamma": gamma,
            "sigma_shrunk": sigma_sh,
            "vra_multiplier": lam,
            "n_obs": n_obs,
            "n_at_floor": n_at_floor,
            "floor_var": floor_var,
            "n_at_cap": n_at_cap,
            "cap_var": cap_var,
            "var_bounds": (
                floor_var * lam**2 if floor_var > 0 else None,
                cap_var * lam**2 if np.isfinite(cap_var) else None,
            ),
            "var_ts_raw": var_ts_raw,
            "shrink_target": target,
        }
    return var


# --------------------------------------------------------------------------- repair
#: ``median(x^2) / MEDIAN_CHI2_1`` is a Gaussian-consistent variance of a zero-mean ``x``.
MEDIAN_CHI2_1 = 0.4549364
REPAIR_WINDOWS = (252, 63, 21)
REPAIR_MIN_OBS = 20


def robust_factor_variances(
    factor_returns: pd.DataFrame,
    windows: tuple[int, ...] = REPAIR_WINDOWS,
    min_obs: int = REPAIR_MIN_OBS,
) -> pd.Series:
    """Per factor: ``median(f^2) / 0.4549`` over the trailing year, a variance
    estimate that one bad day (or one bad month) cannot move -- and the larger
    of that and the same statistic over the trailing quarter and month, so a
    genuine regime change (a crash raises every factor's vol several-fold) lifts
    the reference within ~11 days instead of half a year.  Exact zeros ("no
    members that day") and NaNs are not observations; a window with fewer than
    ``min(min_obs, 2/3 window)`` of them is skipped and a factor with no usable
    window gets NaN (no reference)."""
    A_all = factor_returns.to_numpy(dtype=float)
    out = np.full(A_all.shape[1], np.nan)
    for win in windows:
        A = A_all[-int(win) :]
        A = np.where(np.isfinite(A) & (A != 0.0), A, np.nan)
        n = np.isfinite(A).sum(axis=0)
        cols = np.flatnonzero(n >= min(int(min_obs), (2 * int(win)) // 3))
        if len(cols):
            v = np.nanmedian(A[:, cols] ** 2, axis=0) / MEDIAN_CHI2_1
            out[cols] = np.fmax(out[cols], v)
    return pd.Series(out, index=factor_returns.columns)


def repair_covariance(
    factor_cov: pd.DataFrame,
    specific_var: pd.Series,
    factor_returns: pd.DataFrame | None = None,
    var_bounds: tuple[float | None, float | None] | None = None,
    factor_var_mult: float = 100.0,
    rtol: float = 1e-9,
) -> tuple[pd.DataFrame, pd.Series, list[dict]]:
    """Last line of defence after the covariance is assembled (``as_of``).

    * every specific variance finite and ``> 0`` (else the median of the finite
      ones) and inside ``var_bounds = (floor, cap)`` (else clipped; ``None`` = no
      bound on that side);
    * every factor-covariance entry finite and every factor variance
      ``<= factor_var_mult`` times :func:`robust_factor_variances` of that
      factor's returns (trailing year; quarter / month if larger).  A factor above the bound has its row and
      column scaled so the variance sits at the bound (correlations kept); a
      non-finite variance is set to the reference itself (0 without one) with
      zero covariances.

    Returns ``(factor_cov, specific_var, repairs)``; ``repairs`` is a list of
    ``{"what": ..., "n": ..., "worst": (label, value)}`` and is empty -- with
    the inputs returned untouched -- when nothing violated a bound.  This is a
    repair, not an estimator: the upstream protections are meant to make it
    unreachable, and every firing is recorded so it cannot hide a cause."""
    repairs: list[dict] = []
    d = specific_var.to_numpy(dtype=float)
    bad = ~np.isfinite(d) | (d <= 0)
    lo, hi = var_bounds if var_bounds is not None else (None, None)
    d_new = d
    if bad.any():
        good = d[~bad]
        fill = float(np.median(good)) if len(good) else OUT_OF_MODEL_FALLBACK_VAR
        k = int(np.flatnonzero(bad)[0])
        repairs.append(
            {
                "what": "specific_var_not_finite",
                "n": int(bad.sum()),
                "worst": (str(specific_var.index[k]), float(d[k])),
            }
        )
        d_new = np.where(bad, fill, d)
    if lo is not None and lo > 0:
        below = d_new < lo * (1.0 - rtol)
        if below.any():
            k = int(np.argmin(np.where(below, d_new, np.inf)))
            repairs.append(
                {
                    "what": "specific_var_below_floor",
                    "n": int(below.sum()),
                    "worst": (str(specific_var.index[k]), float(d_new[k])),
                }
            )
            d_new = np.where(below, lo, d_new)
    if hi is not None and np.isfinite(hi):
        above = d_new > hi * (1.0 + rtol)
        if above.any():
            k = int(np.argmax(np.where(above, d_new, -np.inf)))
            repairs.append(
                {
                    "what": "specific_var_above_cap",
                    "n": int(above.sum()),
                    "worst": (str(specific_var.index[k]), float(d_new[k])),
                }
            )
            d_new = np.where(above, hi, d_new)
    D = specific_var if d_new is d else pd.Series(d_new, index=specific_var.index, name=specific_var.name)

    Fv = factor_cov.to_numpy(dtype=float)
    cols = factor_cov.columns
    ref = (
        robust_factor_variances(factor_returns).reindex(cols).to_numpy(dtype=float)
        if factor_returns is not None and len(factor_returns)
        else np.full(len(cols), np.nan)
    )
    diag = np.diag(Fv).copy()
    nonfinite = ~np.isfinite(Fv)
    has_ref = np.isfinite(ref) & (ref > 0)
    with np.errstate(invalid="ignore"):
        too_big = np.isfinite(diag) & has_ref & (diag > float(factor_var_mult) * ref * (1.0 + rtol))
    if nonfinite.any() or too_big.any():
        F2 = Fv.copy()
        bad_diag = ~np.isfinite(diag) | (diag < 0)
        for k in np.flatnonzero(bad_diag):
            F2[k, :] = 0.0
            F2[:, k] = 0.0
            F2[k, k] = ref[k] if has_ref[k] else 0.0
        if bad_diag.any():
            k = int(np.flatnonzero(bad_diag)[0])
            repairs.append(
                {
                    "what": "factor_var_not_finite",
                    "n": int(bad_diag.sum()),
                    "worst": (str(cols[k]), float(diag[k])),
                }
            )
        if too_big.any():
            ratio = np.where(too_big, diag / np.where(has_ref, ref, 1.0), 0.0)
            k = int(np.argmax(ratio))
            repairs.append(
                {
                    "what": "factor_var_above_bound",
                    "n": int(too_big.sum()),
                    "worst": (str(cols[k]), float(diag[k])),
                    "reference_var": float(ref[k]),
                }
            )
            scale = np.ones(len(cols))
            scale[too_big] = np.sqrt(float(factor_var_mult) * ref[too_big] / diag[too_big])
            F2 = F2 * np.outer(scale, scale)
        left = ~np.isfinite(F2)
        if left.any():
            if not bad_diag.any():
                i, j = (int(x[0]) for x in np.nonzero(left))
                repairs.append(
                    {
                        "what": "factor_cov_not_finite",
                        "n": int(left.sum()),
                        "worst": (f"{cols[i]}/{cols[j]}", float(Fv[i, j])),
                    }
                )
            F2 = np.where(left, 0.0, F2)
        factor_cov = pd.DataFrame(nearest_psd(F2), index=factor_cov.index, columns=cols)
    return factor_cov, D, repairs


# Out-of-model specific variance (second-pass finding 8): the rule
# (OUT_OF_MODEL_SPECIFIC_VAR_PCTL = 90th percentile) lives in qts.core.risk_fill, shared
# with RiskModelOutput.covariance / portfolio_variance; ``out_of_model_specific_var`` is
# re-exported here (and from qts.risk) under its public name.


def style_variances(factor_cov: pd.DataFrame, cfg: RiskConfig) -> pd.Series:
    _, _, styles = split_factor_columns(factor_cov.columns, cfg)
    return pd.Series(np.diag(factor_cov.loc[styles, styles].values), index=styles)
