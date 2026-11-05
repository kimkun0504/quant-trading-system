"""Intraday scheduling helpers: volume curves, VWAP / POV schedules, urgency
mapping and an Almgren-Chriss closed-form trajectory for reference.

Caveat (BUILD_PLAN sections 4 and 6): under the square-root law the impact of
a metaorder is approximately independent of how it is split across child
orders and time at modest participation (Bouchaud).  Scheduling therefore buys
little expected cost; its value is in controlling participation (stay under
the 5% ADV step), timing risk and, for auction-centric flow, simply getting to
the close.  Almgren-Chriss assumes *linear* temporary impact and is kept here
only as a reference trajectory and for the urgency intuition.
"""

from __future__ import annotations

import numpy as np

# 13 half-hour buckets 09:30-16:00.  U-shape: heavy open, heavier close.
DEFAULT_U_SHAPE = np.array(
    [0.110, 0.075, 0.060, 0.055, 0.050, 0.048, 0.048, 0.050, 0.055, 0.065, 0.080, 0.104, 0.200]
)
BUCKET_LABELS = [
    "09:30",
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
]


def intraday_volume_curve(n_buckets: int = 13, close_share: float | None = None) -> np.ndarray:
    """Fraction of daily volume per bucket, summing to 1.

    Default is a 13-bucket U-shape whose last bucket (15:30-16:00, which
    contains the closing auction) carries 20% of the day.  ``close_share``
    rescales the last bucket (e.g. 0.30 on index-rebalance days) and the rest
    proportionally.  Other ``n_buckets`` are produced by interpolating the
    default shape.
    """
    curve = DEFAULT_U_SHAPE.copy()
    if n_buckets != 13:
        xs = np.linspace(0, 1, 13)
        xt = np.linspace(0, 1, n_buckets)
        curve = np.interp(xt, xs, curve)
    curve = curve / curve.sum()
    if close_share is not None:
        rest = curve[:-1] / curve[:-1].sum() * (1 - close_share)
        curve = np.append(rest, close_share)
    return curve


def vwap_schedule(qty: float, curve: np.ndarray | None = None) -> np.ndarray:
    """Child-order sizes proportional to the volume curve (sum == qty)."""
    curve = intraday_volume_curve() if curve is None else np.asarray(curve, float)
    sched = qty * curve / curve.sum()
    return sched


def pov_schedule(qty: float, expected_volume_curve: np.ndarray, pct: float) -> np.ndarray:
    """Percentage-of-volume schedule: trade ``pct`` of expected volume in each
    bucket until ``qty`` is done.  ``expected_volume_curve`` is in shares per
    bucket.  Any residual after the last bucket is left unfilled (returned sum
    may be < qty)."""
    ev = np.asarray(expected_volume_curve, float)
    out = np.zeros_like(ev)
    remaining = float(qty)
    for i, v in enumerate(ev):
        take = min(remaining, pct * v)
        out[i] = take
        remaining -= take
        if remaining <= 0:
            break
    return out


def urgency_from_half_life(
    half_life_days: float, min_part: float = 0.01, max_part: float = 0.25, ref_days: float = 5.0
) -> float:
    """Map signal half-life to a target participation rate.

    Heuristic: a signal with half-life ``h`` loses ``1 - 2^(-1/h)`` of its
    value per day of delay; participation should scale with that decay.
    ``ref_days`` is the half-life at which participation equals the geometric
    mean of ``min_part`` and ``max_part``.  Output clipped to [min, max].
    """
    h = max(float(half_life_days), 1e-6)
    decay = 1 - 2 ** (-1 / h)
    ref_decay = 1 - 2 ** (-1 / ref_days)
    mid = np.sqrt(min_part * max_part)
    part = mid * (decay / ref_decay)
    return float(np.clip(part, min_part, max_part))


def almgren_chriss_schedule(
    qty: float, T_buckets: int, sigma: float, eta: float, lam: float, tau: float = 1.0
) -> dict:
    """Closed-form Almgren-Chriss (2000) optimal liquidation trajectory.

    Minimises ``E[cost] + lam * Var[cost]`` with linear temporary impact
    ``eta * v`` (price units per share per unit time) and price volatility
    ``sigma`` (price units per sqrt(time)).  With ``kappa_tilde^2 = lam*sigma^2/eta``
    and ``kappa = acosh(kappa_tilde^2 * tau^2 / 2 + 1) / tau``::

        x_j = qty * sinh(kappa * (T - t_j)) / sinh(kappa * T),  t_j = j*tau

    Returns ``{"holdings": x (T+1,), "trades": n (T,), "kappa": float}``.
    ``lam -> 0`` gives the straight-line (TWAP) solution.  Units must be
    consistent (per-bucket time ``tau``).  Reference only; see module docstring
    for why square-root impact makes the schedule nearly irrelevant for cost.
    """
    T = int(T_buckets)
    t = np.arange(T + 1) * tau
    if lam <= 0 or eta <= 0 or sigma <= 0:
        x = qty * (1 - t / (T * tau))
        kappa = 0.0
    else:
        kt2 = lam * sigma**2 / eta
        kappa = np.arccosh(kt2 * tau**2 / 2 + 1) / tau
        x = qty * np.sinh(kappa * (T * tau - t)) / np.sinh(kappa * T * tau)
    n = -np.diff(x)
    return {"holdings": x, "trades": n, "kappa": float(kappa)}
