"""Holding costs: stock-borrow fees on short positions.

``w`` are weights (fraction of NAV, shorts negative).  The daily holding cost
as a fraction of NAV is ``sum(neg(w) * fee_annual / 252) * days``.  The fee is
the per-name annualised borrow rate (IBKR SLB indicative rate logged daily in
production; ``md.borrow_fee`` in simulation; ``cfg.default_borrow_fee_annual``
as a GC placeholder).  Interest earned on short proceeds is ignored
(conservative).  Long positions carry no holding cost here; margin interest on
leverage above 1x is left to the portfolio layer's financing term.
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
