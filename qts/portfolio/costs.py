"""Cost interface used by the optimizer and the simulator.

The contract lives in :mod:`qts.core.costs` (``LiquiditySnapshot``,
``CostEstimate``, ``CostModel`` protocol); the production implementation is
:class:`qts.tcost.TransactionCostModel`.  This module keeps the portfolio-layer
names stable:

* ``CostTerms`` -- alias of :class:`qts.core.costs.CostModel`.
* ``Snapshot`` -- a :class:`LiquiditySnapshot` (or, for backward compatibility,
  an old ``dict`` snapshot; every cost-model method accepts both).
* :func:`snapshot_from_market` -- the single snapshot builder
  (``TransactionCostModel(cfg).snapshot``).
* :func:`align_snapshot` -- reindex a snapshot to the optimizer's tickers
  (median fills; a ``LiquiditySnapshot`` stays typed, a dict stays a dict).
* :class:`SimpleCostModel` -- **test double, not used in production**.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from qts.core.config import CostConfig
from qts.core.costs import VENUE_ROUTED, CostEstimate, CostModel, LiquiditySnapshot, SnapshotLike, as_snapshot
from qts.core.types import MarketData

CostTerms = CostModel
Snapshot = LiquiditySnapshot | dict[str, Any]
#: per-ticker keys the optimizer reads from a (dict) snapshot
SNAPSHOT_KEYS = ("adv_usd", "sigma_daily", "half_spread", "price", "borrow_fee")


def _tcost_model(cfg: CostConfig):
    from qts.tcost.model import TransactionCostModel  # local import: no import cycle at package load
    return TransactionCostModel(cfg)


def snapshot_from_market(
    md: MarketData,
    date: pd.Timestamp,
    adv_window: int = 63,
    vol_window: int = 63,
    cfg: CostConfig | None = None,
) -> LiquiditySnapshot:
    """Point-in-time :class:`LiquiditySnapshot` (only data with timestamp <= date):
    ``TransactionCostModel(cfg).snapshot(md, date, adv_window, vol_window)``.
    Supports ``snap["adv_usd"]``-style access for older callers."""
    return _tcost_model(cfg or CostConfig()).snapshot(md, pd.Timestamp(date), adv_window=adv_window,
                                                       vol_window=vol_window)


def align_snapshot(snapshot: Snapshot, tickers: pd.Index, cfg: CostConfig | None = None) -> Snapshot:
    """Reindex a snapshot to ``tickers``.

    ``LiquiditySnapshot`` -> ``snapshot.reindex(tickers)`` (median fills, floors).
    Old ``dict`` -> dict whose core keys (:data:`SNAPSHOT_KEYS`) are reindexed
    with median fills and floors (ADV >= $1,000, sigma >= 1e-4, price >= $0.01);
    other per-ticker Series are reindexed, scalars (``date``, ``nav``,
    ``auction_share``) pass through, and ``half_spread_bps`` is kept equal to
    ``1e4 * half_spread``.
    """
    if isinstance(snapshot, LiquiditySnapshot):
        return snapshot.reindex(tickers)
    cfg = cfg or CostConfig()
    defaults = {
        "adv_usd": 1e6, "sigma_daily": 0.02, "half_spread": cfg.default_half_spread_bps / 1e4,
        "price": 100.0, "borrow_fee": cfg.default_borrow_fee_annual,
    }
    src = dict(snapshot)
    if src.get("half_spread") is None and src.get("half_spread_bps") is not None:
        src["half_spread"] = pd.Series(src["half_spread_bps"], dtype=float) / 1e4
    out: dict[str, Any] = {}
    for k in SNAPSHOT_KEYS:
        s = src.get(k)
        if s is None:
            out[k] = pd.Series(defaults[k], index=tickers, dtype=float)
            continue
        s = pd.Series(s).reindex(tickers).astype(float)
        med = s.median()
        out[k] = s.fillna(med if np.isfinite(med) else defaults[k])
    out["adv_usd"] = out["adv_usd"].clip(lower=1e3)
    out["sigma_daily"] = out["sigma_daily"].clip(lower=1e-4)
    out["price"] = out["price"].clip(lower=0.01)
    for k, v in src.items():
        if k in SNAPSHOT_KEYS:
            continue
        out[k] = v.reindex(tickers) if isinstance(v, pd.Series) else v
    if "half_spread_bps" in out:
        out["half_spread_bps"] = out["half_spread"] * 1e4
    return out


@dataclass
class SimpleCostModel:
    """**Test double, not used in production.**

    ``zero=True``: free trading (every cost 0) -- isolates optimizer / simulator
    mechanics in tests.  Otherwise every method delegates to
    ``qts.tcost.TransactionCostModel(cfg)``, so tests that use it price exactly
    the production model.  Production code (pipeline, scripts) uses
    ``TransactionCostModel`` directly.
    """

    cfg: CostConfig = None  # type: ignore[assignment]
    zero: bool = False
    _tcm: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.cfg is None:
            self.cfg = CostConfig()
        if not self.zero:
            self._tcm = _tcost_model(self.cfg)

    @property
    def delegated(self) -> bool:
        return self._tcm is not None

    def snapshot(self, md: MarketData, date, **kw) -> LiquiditySnapshot:
        return _tcost_model(self.cfg).snapshot(md, date, **kw) if self.zero else self._tcm.snapshot(md, date, **kw)

    def cvx_terms(self, x, w_post, nav, snapshot: SnapshotLike, venue: str | None = VENUE_ROUTED,
                  tickers: pd.Index | None = None):
        if self.zero:
            z = cp.Constant(0.0)
            return z, z
        return self._tcm.cvx_terms(x, w_post, nav, snapshot, venue=venue, tickers=tickers)

    def estimate(self, trades_usd: pd.Series, snapshot: SnapshotLike, venue: str | None = VENUE_ROUTED,
                 streak: pd.Series | None = None) -> CostEstimate:
        if self.zero:
            return CostEstimate.zero(trades_usd.index, venue or VENUE_ROUTED)
        return self._tcm.estimate(trades_usd, snapshot, venue=venue, streak=streak)

    def holding_cost_usd(self, w_post: pd.Series, nav: float, snapshot: SnapshotLike) -> pd.Series:
        if self.zero:
            return pd.Series(0.0, index=w_post.index)
        return self._tcm.holding_cost_usd(w_post, nav, snapshot)


__all__ = [
    "SNAPSHOT_KEYS",
    "CostTerms",
    "SimpleCostModel",
    "Snapshot",
    "align_snapshot",
    "as_snapshot",
    "snapshot_from_market",
]
