"""Holding costs: stock-borrow fees on short positions.

``w`` are weights (fraction of NAV, shorts negative).  The daily holding cost
as a fraction of NAV is ``sum(neg(w) * fee_annual / 252) * days``.  The fee is
the per-name annualised borrow rate (IBKR SLB indicative rate logged daily in
production; ``md.borrow_fee`` in simulation; ``cfg.default_borrow_fee_annual``
as a GC placeholder).  Interest earned on short proceeds is ignored
(conservative).  Long positions carry no holding cost here; margin interest on
leverage above 1x is left to the portfolio layer's financing term.

Borrow-fee proxy (PLACEHOLDER)
------------------------------
Until the daily IBKR SLB indicative-rate history exists, :func:`borrow_fee_proxy`
maps short-interest ratio (and optionally days-to-cover and market cap) to an
annual fee:

* SI ratio below ``cfg.borrow_proxy_si_gc`` (5%): general collateral,
  ``cfg.default_borrow_fee_annual`` (0.5%).
* Between 5% and ``cfg.borrow_proxy_si_htb`` (20%): rises **convexly**
  (quadratic in the normalised distance) from GC to ``cfg.htb_fee_cap_annual``
  (25%), so 10% SI maps to ~3.2%/yr and 15% SI to ~11%/yr, in line with the
  5-50%/yr range typical of hard-to-borrow names.
* At/above 20% SI: ``htb_fee_cap_annual``.
* ``days_to_cover`` (SI shares / ADV shares), when supplied, is an alternative
  tightness measure: DTC of 2 maps to the GC end and DTC of 10 to the HTB end;
  the proxy takes the tighter of the two signals.
* Market cap below ``cfg.small_cap_threshold_usd`` ($2B): the fee is
  multiplied by ``cfg.small_cap_borrow_multiplier`` (2.0) **after** the curve
  (so a small-cap HTB name can reach 50%/yr; small-cap GC is 1%/yr).  The
  BUILD_PLAN asks for a conservative small-cap haircut in backtests.

The proxy is deliberately pessimistic for the ``short_interest`` alpha sleeve
(it shorts exactly the high-SI names).  Replace with logged SLB rates as soon
as they exist; ``snapshot["borrow_fee_source"]`` records which was used.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from qts.core.config import CostConfig

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None

TRADING_DAYS = 252


def borrow_fee_proxy(short_interest_ratio, days_to_cover=None, cfg: Optional[CostConfig] = None,
                     market_cap=None):
    """Annual borrow-fee proxy from short interest (see module docstring).

    ``short_interest_ratio`` is short interest / shares outstanding (fraction).
    Accepts scalars, arrays or Series (Series in -> Series out, same index).
    NaN short interest -> GC default (``cfg.default_borrow_fee_annual``).
    ``days_to_cover`` and ``market_cap`` (USD) are optional and broadcast.
    """
    cfg = cfg or CostConfig()
    gc = float(cfg.default_borrow_fee_annual)
    cap = float(getattr(cfg, "htb_fee_cap_annual", 0.25))
    lo = float(getattr(cfg, "borrow_proxy_si_gc", 0.05))
    hi = float(getattr(cfg, "borrow_proxy_si_htb", 0.20))
    index = short_interest_ratio.index if isinstance(short_interest_ratio, pd.Series) else None
    si = np.asarray(short_interest_ratio, dtype=float)
    with np.errstate(invalid="ignore"):
        u = np.clip((si - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    if days_to_cover is not None:
        dtc = np.asarray(days_to_cover, dtype=float)
        with np.errstate(invalid="ignore"):
            u_dtc = np.clip((dtc - 2.0) / 8.0, 0.0, 1.0)
        u = np.fmax(u, u_dtc)                       # fmax ignores NaN on one side
    u = np.where(np.isfinite(u), u, 0.0)
    fee = gc + (cap - gc) * u ** 2                  # convex rise GC -> cap
    if market_cap is not None:
        mc = np.asarray(market_cap, dtype=float)
        small = np.isfinite(mc) & (mc < float(getattr(cfg, "small_cap_threshold_usd", 2e9)))
        fee = np.where(small, fee * float(getattr(cfg, "small_cap_borrow_multiplier", 2.0)), fee)
    if index is not None:
        return pd.Series(fee, index=index, dtype=float)
    return fee if np.ndim(fee) else float(fee)


def borrow_fee_vector(index: pd.Index, borrow_fee_annual: Optional[pd.Series],
                      cfg: Optional[CostConfig] = None) -> pd.Series:
    """Per-name annual borrow fee aligned to ``index``; missing -> default."""
    cfg = cfg or CostConfig()
    if borrow_fee_annual is None:
        return pd.Series(cfg.default_borrow_fee_annual, index=index, dtype=float)
    return borrow_fee_annual.reindex(index).fillna(cfg.default_borrow_fee_annual).astype(float)


def holding_cost_fraction(w: pd.Series, borrow_fee_annual: Optional[pd.Series] = None,
                          cfg: Optional[CostConfig] = None, days: int = 1) -> float:
    """Borrow cost on short positions over ``days`` trading days, as a fraction of NAV:
    ``sum(|w_short| * fee / 252) * days``."""
    fee = borrow_fee_vector(w.index, borrow_fee_annual, cfg)
    short = np.minimum(w.astype(float).values, 0.0)
    return float(-(short * fee.values).sum() / TRADING_DAYS * days)


def holding_cost_breakdown(w: pd.Series, borrow_fee_annual: Optional[pd.Series] = None,
                           cfg: Optional[CostConfig] = None, days: int = 1) -> pd.Series:
    """Per-name version of :func:`holding_cost_fraction`."""
    fee = borrow_fee_vector(w.index, borrow_fee_annual, cfg)
    short = np.minimum(w.astype(float).values, 0.0)
    return pd.Series(-short * fee.values / TRADING_DAYS * days, index=w.index)


def cvx_holding_expression(w_post, borrow_fee_vector_annual: Union[np.ndarray, pd.Series, float],
                           days: int = 1):
    """cvxpy expression for daily borrow cost (fraction of NAV): ``sum(fee/252 * neg(w_post))``.
    Convex in ``w_post`` since ``neg`` is convex and the fee is non-negative."""
    if cp is None:  # pragma: no cover
        raise ImportError("cvxpy required")
    fee = np.asarray(borrow_fee_vector_annual, dtype=float)
    fee = np.broadcast_to(fee, (w_post.shape[0],))
    return cp.sum(cp.multiply(fee / TRADING_DAYS * days, cp.neg(w_post)))
