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
  the size-decile mean, specific VRA.

All variances are in **daily** units.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qts.core.config import RiskConfig
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
    F: pd.DataFrame, vol_half_life: float, vra_half_life: float, min_periods: int = 20
) -> float:
    """Volatility Regime Adjustment (USE4 section "Volatility regime adjustment").

    Daily cross-sectional bias statistic of the factors
        B_t = sqrt( mean_k ( f_kt / sigma_{k,t-1} )^2 )
    where ``sigma_{k,t-1}`` is the EWMA (``vol_half_life``) forecast made at
    ``t-1``.  The factor volatility multiplier is
        lambda_F = sqrt( EWMA_{vra_half_life}( B_t^2 ) ),
    and the factor covariance is scaled by ``lambda_F^2``.  Returns 1.0 when
    there is too little history."""
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
    B2 = z2.mean(axis=1, skipna=True).dropna()
    if len(B2) < min_periods:
        return 1.0
    lam2 = B2.ewm(halflife=vra_half_life).mean().iloc[-1]
    return float(np.sqrt(max(lam2, 1e-6)))


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

    This returns the **raw** simulated ``v_k`` (clipped to [0.5, 2]), in
    ascending-eigenvalue order.  USE4 then optionally smooths ``v`` across ``k``
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
    v = np.sqrt(ratio / n_sims)
    return np.clip(v, 0.5, 2.0)


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

    lam = factor_vra_multiplier(F, cfg.vol_half_life, cfg.vra_half_life)
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
    U: pd.DataFrame, market_cap: pd.Series, half_life: float, vra_half_life: float, min_periods: int = 20
) -> float:
    """Specific-risk VRA: cap-weighted cross-sectional bias statistic
    ``B_t^2 = sum_i w_i (u_it / sigma_{i,t-1})^2`` (USE4 uses this form with
    the structural specific risk forecasts; we use the EWMA forecast made at
    ``t-1``), then ``lambda_S = sqrt(EWMA_{vra}(B_t^2))``."""
    A = U.values.astype(float)
    T = A.shape[0]
    if min_periods + 1 >= T:
        return 1.0
    u2 = pd.DataFrame(A**2)
    s2_prev = u2.ewm(halflife=half_life, min_periods=min_periods, ignore_na=True).mean().shift(1).values
    z2 = A**2 / np.where(s2_prev > 0, s2_prev, np.nan)
    w = market_cap.reindex(U.columns).fillna(0.0).clip(lower=0.0).values
    ok = np.isfinite(z2)
    wsum = (ok * w).sum(axis=1)
    B2 = np.where(wsum > 0, (np.where(ok, z2, 0.0) * w).sum(axis=1) / np.where(wsum > 0, wsum, 1.0), np.nan)
    B2 = B2[np.isfinite(B2)]
    if len(B2) < min_periods:
        return 1.0
    lam2 = float(ewma_weights(len(B2), vra_half_life) @ B2)
    return float(np.sqrt(np.clip(lam2, 1e-6, None)))


def structural_specific_vol(
    sigma_ts: pd.Series, gamma: pd.Series, exposures: pd.DataFrame, market_cap: pd.Series
) -> pd.Series:
    """USE4 structural model: regress ``ln sigma_TS`` on the factor exposures
    (WLS, weights ∝ sqrt(cap)) over stocks with full history (``gamma == 1``),
    then ``sigma_STR = E0 * exp(X b)`` with ``E0`` the bias correction making
    the fitted values unbiased in levels."""
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
    fitted = np.exp(X @ b)
    E0 = float(np.sum(w * sigma_ts.values[fit_mask]) / np.sum(w * fitted[fit_mask]))
    return pd.Series(E0 * fitted, index=idx)


def bayesian_shrink_specific(
    sigma: pd.Series, market_cap: pd.Series, q: float, n_deciles: int = 10
) -> pd.Series:
    """USE4 Bayesian shrinkage of specific risk toward the cap-weighted mean of
    its size decile:

        sigma_SH = v sigma_bar + (1 - v) sigma,
        v = q |sigma - sigma_bar| / ( Delta + q |sigma - sigma_bar| ),

    where ``sigma_bar`` is the cap-weighted mean and ``Delta`` the equal-weighted
    standard deviation of ``sigma`` within the decile."""
    cap = market_cap.reindex(sigma.index).fillna(market_cap.median()).clip(lower=1e-9).values
    s = sigma.values.astype(float)
    n = len(s)
    k = max(1, min(n_deciles, n // 10))
    order = np.argsort(cap, kind="stable")
    dec = np.empty(n, dtype=int)
    dec[order] = (np.arange(n) * k) // n
    out = s.copy()
    for d in range(k):
        m = dec == d
        if not m.any():
            continue
        sd_, cd = s[m], cap[m]
        sbar = float((sd_ * cd).sum() / cd.sum())
        delta = float(sd_.std())
        dev = np.abs(sd_ - sbar)
        den = delta + q * dev
        v = np.where(den > 0, q * dev / np.where(den > 0, den, 1.0), 0.0)
        out[m] = v * sbar + (1 - v) * sd_
    return pd.Series(out, index=sigma.index)


def specific_risk(
    specific_returns: pd.DataFrame,
    exposures_latest: pd.DataFrame,
    market_cap: pd.Series,
    cfg: RiskConfig | None = None,
    return_details: bool = False,
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
    4. Specific VRA multiplier ``lambda_S``.

    Returns ``(lambda_S * sigma_SH)^2`` indexed like ``exposures_latest``."""
    cfg = cfg or RiskConfig()
    tickers = exposures_latest.index
    window = int(max(6 * cfg.specific_half_life, cfg.min_history_days))
    U = specific_returns.reindex(columns=tickers).tail(window)
    var_ts, n_obs = ewma_nw_variance(U, cfg.specific_half_life, cfg.newey_west_lags_vol)
    sigma_ts = np.sqrt(var_ts)

    # robust-vs-equal sd ratio for outlier damping (USE4: sigma_robust = IQR/1.35,
    # Z = |sigma_eq / sigma_robust - 1|, gamma = clip((h-60)/120) * min(1, exp(1 - Z)))
    q75, q25 = U.quantile(0.75), U.quantile(0.25)
    sd_robust = ((q75 - q25) / 1.35).replace(0.0, np.nan)
    sd_eq = U.std(ddof=0)
    Z = (sd_eq / sd_robust - 1.0).abs().fillna(0.0)
    gamma = ((n_obs - 60) / 120.0).clip(0.0, 1.0) * np.minimum(1.0, np.exp(1.0 - Z))
    gamma = gamma.fillna(0.0)

    sigma_str = structural_specific_vol(sigma_ts, gamma, exposures_latest, market_cap)
    sigma = gamma * sigma_ts.fillna(0.0) + (1 - gamma) * sigma_str
    sigma = sigma.where(sigma > 0, sigma_str)

    sigma_sh = bayesian_shrink_specific(sigma, market_cap, cfg.specific_shrink_q)
    lam = specific_vra_multiplier(U, market_cap, cfg.specific_half_life, cfg.vra_half_life)
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
        }
    return var


def style_variances(factor_cov: pd.DataFrame, cfg: RiskConfig) -> pd.Series:
    _, _, styles = split_factor_columns(factor_cov.columns, cfg)
    return pd.Series(np.diag(factor_cov.loc[styles, styles].values), index=styles)
