"""Synthetic market generator with known factor structure and planted alphas.

Purpose: every layer of the system can be developed and unit-tested offline,
and the alpha/validation layers can be checked against *known* truth (we know
exactly which signals carry information and how much).

Design
------
Daily returns for N stocks over T days:

    r_it = beta_i * m_t + sum_k x_ik f_kt + alpha_it + eps_it

* ``m_t``        market factor, annual vol ~16%, with GARCH-like vol clustering.
* ``f_kt``       K style factor returns (size, value, momentum, quality,
                 investment, low_risk) each with small positive premium.
* ``x_ik``       slowly mean-reverting exposures (AR(1) in monthly steps).
* ``alpha_it``   planted predictable component = ``ic_true * sigma_resid *
                 z_it`` where ``z_it`` is a latent score that the generator
                 also emits through observable characteristics with noise.
                 This is how the test-suite checks the alpha model recovers
                 roughly ``ic_true`` rank-IC and not more.
* ``eps_it``     idiosyncratic noise, per-stock vol drawn log-normally.

Also generated: shares outstanding -> market cap, volume (log-normal around a
size-dependent ADV), spreads (decreasing in size), short interest ratio,
borrow fee (increasing in short interest), book-to-price, earnings yield,
gross profitability, asset growth, industry labels (12 groups).

Everything is seeded and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from qts.core.types import MarketData

# Latent style factors.  Every one of them is *observable* through a generated
# characteristic (size -> market cap, value -> book_to_price / earnings_yield,
# quality -> gross_profitability, investment -> asset_growth), so a correctly
# built fundamental risk model can explain their variance.  Earlier versions
# also carried latent "momentum" and "low_risk" factors that were unrelated to
# past returns or volatility; no risk model built from observables could see
# them, which made optimized books look 2x riskier than ex-ante.  Real-world
# unobserved factors are what the plan's APCA companion model is for.
STYLE_FACTORS = ["size", "value", "quality", "investment"]


@dataclass
class SyntheticSpec:
    n_stocks: int = 300
    n_days: int = 252 * 6
    start: str = "2018-01-02"
    n_industries: int = 12
    market_vol_annual: float = 0.16
    style_vol_annual: float = 0.04
    style_premium_annual: float = 0.02
    idio_vol_annual_median: float = 0.30
    ic_true: float = 0.04          # planted rank-IC of the latent score vs 20d resid return
    alpha_horizon_days: int = 20
    characteristic_noise: float = 1.0  # noise sd added to latent score in observables
    seed: int = 0


def _ar1_path(n_steps: int, n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros((n_steps, n))
    x[0] = rng.standard_normal(n)
    s = np.sqrt(1 - phi ** 2)
    for t in range(1, n_steps):
        x[t] = phi * x[t - 1] + s * rng.standard_normal(n)
    return x


def generate(spec: Optional[SyntheticSpec] = None) -> MarketData:
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    N, T = spec.n_stocks, spec.n_days
    dates = pd.bdate_range(spec.start, periods=T)
    tickers = pd.Index([f"S{i:04d}" for i in range(N)], name="ticker")
    K = len(STYLE_FACTORS)

    # ---- static-ish attributes
    industry = pd.Series(rng.integers(0, spec.n_industries, N), index=tickers).map(lambda k: f"IND{k:02d}")
    log_size0 = rng.normal(0, 1.2, N)
    beta = np.clip(1.0 + 0.35 * rng.standard_normal(N) + 0.1 * (-log_size0), 0.2, 2.5)
    idio_vol = spec.idio_vol_annual_median * np.exp(0.35 * rng.standard_normal(N) - 0.15 * log_size0)
    idio_vol = np.clip(idio_vol, 0.10, 1.2) / np.sqrt(252)

    # ---- monthly-varying exposures and latent alpha score
    n_months = T // 21 + 2
    expo_m = _ar1_path(n_months, N * K, 0.90, rng).reshape(n_months, N, K)
    expo_m[:, :, 0] = log_size0 / 1.2  # size exposure = standardized log market cap (observable)
    z_m = _ar1_path(n_months, N, 0.80, rng)              # latent alpha score
    month_of_day = np.arange(T) // 21
    expo = expo_m[month_of_day]                           # T x N x K
    z = z_m[month_of_day]                                 # T x N

    # ---- factor returns with vol clustering in the market
    mkt_vol_d = spec.market_vol_annual / np.sqrt(252)
    h = np.full(T, mkt_vol_d ** 2)
    m = np.zeros(T)
    for t in range(1, T):
        h[t] = 0.05 * mkt_vol_d ** 2 + 0.10 * m[t - 1] ** 2 + 0.85 * h[t - 1]
        m[t] = np.sqrt(h[t]) * rng.standard_normal() + 0.06 / 252
    style_vol_d = spec.style_vol_annual / np.sqrt(252)
    f = style_vol_d * rng.standard_normal((T, K)) + spec.style_premium_annual / 252
    # occasional market-wide factor shocks (fat tails for the risk model to cope with)
    if T > 300:
        crash_days = rng.choice(np.arange(252, T), size=max(1, T // 600), replace=False)
        for cd in crash_days:
            m[cd:cd + 10] -= 0.004
            f[cd:cd + 10, 1] -= 0.003   # value drawdown

    # ---- planted alpha: ic_true * sigma_resid * z lagged so z at t predicts the next H days
    H = spec.alpha_horizon_days
    per_day_alpha = spec.ic_true * idio_vol[None, :] * z / np.sqrt(H)
    eps = idio_vol[None, :] * rng.standard_normal((T, N))
    r = beta[None, :] * m[:, None] + np.einsum("tnk,tk->tn", expo, f) + per_day_alpha + eps
    r = np.clip(r, -0.5, 1.0)

    close = 100 * np.exp(np.cumsum(np.log1p(r), axis=0))
    close = pd.DataFrame(close, index=dates, columns=tickers)

    shares = np.exp(log_size0 + 17.2)            # median mcap ≈ $3B at $100; range ≈ $100M–$300B
    market_cap = close * shares[None, :]
    # daily turnover ≈ 0.8% of shares outstanding (US mid/large-cap typical), log-normal dispersion
    adv_share = shares * np.exp(-4.8 + 0.4 * rng.standard_normal(N))
    volume = adv_share[None, :] * np.exp(0.5 * rng.standard_normal((T, N)) - 0.125)
    volume = pd.DataFrame(volume, index=dates, columns=tickers).round()

    size_rank = pd.Series(log_size0, index=tickers).rank(pct=True)
    spread = (1.0 + 12.0 * (1 - size_rank) ** 2).values[None, :] * np.exp(0.2 * rng.standard_normal((T, N)))
    spread_bps = pd.DataFrame(spread, index=dates, columns=tickers)

    si_m = 0.03 + 0.03 * np.exp(0.8 * _ar1_path(n_months, N, 0.95, rng))
    si = pd.DataFrame(si_m[month_of_day], index=dates, columns=tickers).clip(0.001, 0.6)
    borrow = (0.003 + 2.0 * np.maximum(si - 0.08, 0) ** 1.5).clip(0.003, 1.0)

    # observable characteristics: exposures + noisy versions of the latent score
    def obs(k: int, w_alpha: float) -> pd.DataFrame:
        # measurement noise is persistent (monthly), like a reported fundamental
        noise_m = rng.standard_normal((n_months, N))
        base = expo[:, :, k] + w_alpha * z + spec.characteristic_noise * 0.5 * noise_m[month_of_day]
        return pd.DataFrame(base, index=dates, columns=tickers)

    book_to_price = np.exp(0.5 * obs(1, 0.4))
    earnings_yield = 0.05 + 0.03 * obs(1, 0.4)
    gross_profitability = 0.3 + 0.1 * obs(2, 0.4)
    asset_growth = 0.1 - 0.1 * obs(3, 0.4)

    md = MarketData(
        close=close, volume=volume, market_cap=market_cap, industry=industry,
        spread_bps=spread_bps, short_interest_ratio=si, borrow_fee=borrow,
        book_to_price=book_to_price, earnings_yield=earnings_yield,
        gross_profitability=gross_profitability, asset_growth=asset_growth,
        extra={
            "true_latent_score": pd.DataFrame(z, index=dates, columns=tickers),
            "true_beta": pd.DataFrame(np.tile(beta, (T, 1)), index=dates, columns=tickers),
            "true_market_return": pd.DataFrame(np.tile(m[:, None], (1, N)), index=dates, columns=tickers),
            "true_style_returns": pd.DataFrame(np.tile(f[:, :1], (1, N)), index=dates, columns=tickers),
        },
    )
    md.validate()
    return md


def true_factor_returns(spec: SyntheticSpec) -> pd.DataFrame:
    """Re-generate the same draw and return market + style factor returns (for tests)."""
    # regenerate deterministically by calling generate and reading extras is awkward;
    # keep it simple: tests use md.extra["true_market_return"].iloc[:, 0].
    raise NotImplementedError("use generate(spec).extra['true_market_return']")
