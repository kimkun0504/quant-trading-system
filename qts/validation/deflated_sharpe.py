"""Probabilistic and Deflated Sharpe Ratio (Bailey & López de Prado 2012/2014)
plus minimum track-record and minimum backtest length.

All Sharpe ratios passed to the core functions are **per-observation**
(e.g. daily SR for daily returns) unless the function name says ``annual``.
Use :func:`annualize_sharpe` / :func:`deannualize_sharpe` to move between the
two.  Skew and kurtosis are those of the per-observation returns; ``kurt`` is
the *raw* (non-excess) kurtosis, i.e. 3 for a Gaussian.

Key formulas
------------
Standard error of an estimated Sharpe ratio over ``n`` observations
(Bailey & LdP 2012, eq. 5; Lo 2002 for the Gaussian case)::

    se(SR) = sqrt( (1 - skew*SR + (kurt-1)/4 * SR^2) / (n - 1) )

Probabilistic Sharpe ratio::

    PSR(SR*) = Phi( (SR - SR*) / se(SR) )

Expected maximum of ``N`` i.i.d. Sharpe estimates with variance ``V`` under
the null of zero true SR (Bailey & LdP 2014, eq. 8)::

    E[max SR] ~= sqrt(V) * ( (1-gamma) Phi^-1(1 - 1/N) + gamma Phi^-1(1 - 1/(N e)) )

with Euler-Mascheroni gamma ~= 0.5772.  The deflated Sharpe ratio is the PSR
evaluated at SR* = E[max SR].

Minimum track record length (Bailey & LdP 2012)::

    MinTRL = 1 + (1 - skew*SR + (kurt-1)/4 SR^2) * (Phi^-1(prob) / (SR - SR*))^2

Minimum backtest length (Bailey et al. 2014, "Pseudo-mathematics...", and
Harvey & Liu's multiple-testing framing): the number of observations such that
the expected maximum Sharpe of ``N`` pure-noise trials is below the target.
Under the null each noise trial's SR has variance ~ 1/n, so
``E[max SR] ~= z_N / sqrt(n)`` and ``n_min = (z_N / target_SR_per_obs)^2``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329
TRADING_DAYS = 252


# --------------------------------------------------------------------------- helpers
def annualize_sharpe(sr_per_obs: float, periods_per_year: int = TRADING_DAYS) -> float:
    return float(sr_per_obs) * math.sqrt(periods_per_year)


def deannualize_sharpe(sr_annual: float, periods_per_year: int = TRADING_DAYS) -> float:
    return float(sr_annual) / math.sqrt(periods_per_year)


def sharpe_from_returns(returns, ddof: int = 1) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    sd = r.std(ddof=ddof)
    return float(r.mean() / sd) if sd > 0 else float("nan")


def moments_from_returns(returns):
    """(sr, n, skew, raw kurtosis) of a return series."""
    from scipy.stats import kurtosis, skew

    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    return sharpe_from_returns(r), len(r), float(skew(r)), float(kurtosis(r, fisher=False))


# --------------------------------------------------------------------------- core
def sharpe_ratio_se(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Standard error of a per-observation Sharpe estimate (non-Gaussian aware)."""
    if n <= 1:
        return float("inf")
    var = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1)
    return math.sqrt(max(var, 1e-18))


def probabilistic_sharpe(
    sr: float, sr_benchmark: float, n: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """P[true SR > sr_benchmark] given the observed per-obs ``sr`` over ``n`` observations."""
    se = sharpe_ratio_se(sr, n, skew, kurt)
    if not np.isfinite(se) or se == 0:
        return float("nan")
    return float(norm.cdf((sr - sr_benchmark) / se))


def expected_max_sharpe(n_trials: float, var_sr: float) -> float:
    """E[max of N trial Sharpes] under the zero-SR null with cross-trial variance ``var_sr``
    (Bailey & Lopez de Prado 2014, eq. 8)::

        E[max SR] = sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))

    ``N`` may be fractional (a Nyholt effective number of trials): it is used as
    is, never truncated (rounding 2.4 down to 2 under-deflates; audit 2 #17).

    The closed form is an extreme-value (large-``N``) approximation.  Below
    ``N`` of about 1.28 it turns negative (at ``N = 1.01`` it gives ``-0.79 sd``),
    which would set a *negative* benchmark Sharpe and inflate the DSR, although
    the true expected maximum of ``N`` zero-mean draws is never below zero and is
    exactly zero at ``N = 1``.  For ``1 < N < 2`` the function therefore
    interpolates linearly between 0 at ``N = 1`` and the closed form at ``N = 2``
    (0.52 sd; the exact value for two standard normals is 0.564 sd), which is
    non-negative, continuous and non-decreasing in ``N``.
    """
    n_trials = float(n_trials)
    if n_trials <= 1:
        return 0.0
    sd = math.sqrt(max(var_sr, 0.0))
    if n_trials < 2.0:
        return float((n_trials - 1.0) * _closed_form_unit(2.0) * sd)
    return float(sd * _closed_form_unit(n_trials))


def _closed_form_unit(n_trials: float) -> float:
    """Bailey & Lopez de Prado eq. 8 with unit variance; valid for ``n_trials >= 2``."""
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe_ratio(
    sr_observed: float,
    n_trials: float,
    var_trials_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """Deflated Sharpe ratio: PSR of the selected trial against E[max SR] of the search.

    Parameters are per-observation (daily SR, variance of daily SRs across
    trials, number of daily observations).  Returns a probability in [0, 1];
    the plan's gate is DSR > 0.95.
    """
    sr_star = expected_max_sharpe(n_trials, var_trials_sr)
    return probabilistic_sharpe(sr_observed, sr_star, n_obs, skew, kurt)


def deflated_sharpe_ratio_annual(
    sr_observed_annual: float,
    n_trials: float,
    var_trials_sr_annual: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Convenience wrapper taking annualized SR and annualized cross-trial variance."""
    return deflated_sharpe_ratio(
        deannualize_sharpe(sr_observed_annual, periods_per_year),
        n_trials,
        var_trials_sr_annual / periods_per_year,
        n_obs,
        skew,
        kurt,
    )


def min_track_record_length(
    sr: float, sr_benchmark: float, skew: float = 0.0, kurt: float = 3.0, prob: float = 0.95
) -> float:
    """Observations needed for PSR(sr_benchmark) >= prob given the per-obs ``sr``.  inf if sr <= benchmark."""
    if sr <= sr_benchmark:
        return float("inf")
    z = norm.ppf(prob)
    return float(1.0 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) * (z / (sr - sr_benchmark)) ** 2)


def min_backtest_length(
    n_trials: int, target_sr_annual: float, periods_per_year: int = TRADING_DAYS
) -> float:
    """Observations such that E[max SR] of ``n_trials`` noise strategies < ``target_sr_annual``.

    Returns the number of *observations* (days for daily data); divide by
    ``periods_per_year`` for years.  With N=1 there is no selection effect
    and the answer is 0.
    """
    n_trials = int(n_trials)
    if n_trials <= 1:
        return 0.0
    target = deannualize_sharpe(target_sr_annual, periods_per_year)
    if target <= 0:
        return float("inf")
    z_n = expected_max_sharpe(n_trials, 1.0)  # E[max] with unit variance
    return float((z_n / target) ** 2)


def min_backtest_length_years(
    n_trials: int, target_sr_annual: float, periods_per_year: int = TRADING_DAYS
) -> float:
    return min_backtest_length(n_trials, target_sr_annual, periods_per_year) / periods_per_year


def dsr_from_trial_returns(trial_returns: np.ndarray, selected: int | None = None) -> dict:
    """Given a (T x N) matrix of per-period trial returns, compute the DSR of the
    best (or ``selected``) trial using the empirical cross-trial SR variance."""
    R = np.asarray(trial_returns, dtype=float)
    srs = np.array([sharpe_from_returns(R[:, j]) for j in range(R.shape[1])])
    j = int(np.nanargmax(srs)) if selected is None else int(selected)
    sr, n, sk, ku = moments_from_returns(R[:, j])
    var_sr = float(np.nanvar(srs, ddof=1)) if R.shape[1] > 1 else 0.0
    return {
        "selected": j,
        "sr_per_obs": sr,
        "sr_annual": annualize_sharpe(sr),
        "n_obs": n,
        "n_trials": R.shape[1],
        "var_trials_sr": var_sr,
        "expected_max_sr": expected_max_sharpe(R.shape[1], var_sr),
        "dsr": deflated_sharpe_ratio(sr, R.shape[1], var_sr, n, sk, ku),
    }
