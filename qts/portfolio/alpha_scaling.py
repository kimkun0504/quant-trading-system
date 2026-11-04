"""Score -> alpha conversion and multi-sleeve combination.

* Grinold's rule turns a standardised score into an expected residual return:
  ``alpha = IC * sigma_resid * z``.  ``sigma_resid`` must be the residual vol
  over the *same horizon* as the alpha (daily vol * sqrt(H) for an H-day
  alpha).  Missing scores map to zero alpha (MSCI "Converting scores into
  alphas").
* Garleanu-Pedersen shrinkage: in GP's closed form the aim portfolio weights
  each signal by ``1 / (1 + phi_k / rho)`` where ``phi_k`` is the signal's
  mean-reversion speed and ``rho`` the trading speed implied by costs.  With
  ``phi_k = ln 2 / half_life_k`` and ``rho = ln 2 / holding_days`` this is
  ``1 / (1 + holding_days / half_life_k)``: the longer the cost-implied holding
  period relative to a signal's half-life, the more that signal is shrunk.
  Fast signals are down-weighted as costs (hence holding periods) rise.  We
  apply this in *alpha space* (one combined alpha vector into a single-period
  optimizer) which reproduces the GP trajectory to first order and never
  mixes portfolios.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd


def shrink_ic(ic_est: float, shrink: float = 0.5, floor: float = 0.0, cap: float = 0.1) -> float:
    """Shrink an estimated IC toward zero (default 50%) and clip to [floor, cap].
    NaN estimates become ``floor``."""
    if ic_est is None or not np.isfinite(ic_est):
        return float(floor)
    return float(np.clip((1.0 - shrink) * ic_est, floor, cap))


def grinold_alpha(score: pd.Series, resid_vol: pd.Series, ic: float) -> pd.Series:
    """``alpha_i = IC * sigma_i * z_i`` with ``z`` re-standardised cross-sectionally.

    ``resid_vol`` is per-horizon residual volatility (same horizon as the
    returned alpha).  Names with missing score get alpha 0; names with missing
    vol get the cross-sectional median vol.
    """
    z = pd.Series(score, dtype=float).copy()
    valid = z.notna()
    if valid.sum() >= 2:
        sd = z[valid].std(ddof=0)
        z[valid] = (z[valid] - z[valid].mean()) / (sd if sd > 0 else 1.0)
    z = z.fillna(0.0)
    sig = pd.Series(resid_vol, dtype=float).reindex(z.index)
    med = sig.median()
    sig = sig.fillna(med if np.isfinite(med) else 0.0).clip(lower=0.0)
    out = ic * sig * z
    out.name = "alpha"
    return out


def gp_sleeve_weights(half_lives: Mapping[str, float], cost_implied_holding_days: float) -> pd.Series:
    """Raw GP weights ``1 / (1 + H / hl_k)`` normalised to sum to one."""
    hl = pd.Series(half_lives, dtype=float).clip(lower=1e-6)
    H = max(float(cost_implied_holding_days), 0.0)
    raw = 1.0 / (1.0 + H / hl)
    return raw / raw.sum()


def combine_in_forecast_space(scores: Mapping[str, pd.Series], weights: Mapping[str, float]) -> pd.Series:
    """Weighted sum of per-sleeve forecasts (alphas or scores) on the union of
    tickers; missing sleeve values count as zero.  Weights are used as given
    (normalise upstream if required).  Combine forecasts, never portfolios."""
    if not scores:
        return pd.Series(dtype=float)
    idx = None
    for s in scores.values():
        idx = s.index if idx is None else idx.union(s.index)
    out = pd.Series(0.0, index=idx)
    for k, s in scores.items():
        w = float(weights.get(k, 0.0))
        if w == 0.0:
            continue
        out = out + w * pd.Series(s, dtype=float).reindex(idx).fillna(0.0)
    out.name = "combined"
    return out


def garleanu_pedersen_shrink(
    alphas: Dict[str, pd.Series],
    half_lives: Dict[str, float],
    cost_implied_holding_days: float,
    sleeve_weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Combine per-sleeve alphas with GP half-life shrinkage.

    weight_k ∝ prior_k / (1 + holding_days / half_life_k), normalised to sum
    to one so the combined alpha stays on the scale of a single sleeve.
    ``prior_k`` (``sleeve_weights``) defaults to equal.
    """
    keys = [k for k in alphas if k in half_lives]
    if not keys:
        raise ValueError("no sleeves with a half-life")
    gp = gp_sleeve_weights({k: half_lives[k] for k in keys}, cost_implied_holding_days)
    if sleeve_weights:
        pri = pd.Series({k: float(sleeve_weights.get(k, 1.0)) for k in keys})
        gp = gp * pri
        gp = gp / gp.sum()
    return combine_in_forecast_space({k: alphas[k] for k in keys}, gp.to_dict())


def cost_implied_holding_days(expected_alpha_per_day: float, round_trip_cost: float,
                              min_days: float = 1.0, max_days: float = 252.0) -> float:
    """Crude GP trading-speed proxy: the holding period at which one round-trip
    cost equals the alpha earned, ``H = cost / alpha_per_day``."""
    if expected_alpha_per_day <= 0:
        return float(max_days)
    return float(np.clip(round_trip_cost / expected_alpha_per_day, min_days, max_days))
