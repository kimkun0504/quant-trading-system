"""Order dataclass and target-weight -> order-list conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ORDER_TYPES = ("MOC", "LOC", "LMT", "MKT", "ALGO")
SIDES = ("BUY", "SELL")


@dataclass
class Order:
    ticker: str
    side: str  # "BUY" | "SELL"
    qty: float  # shares, positive
    order_type: str = "MOC"  # one of ORDER_TYPES
    limit_px: float | None = None
    algo: str | None = None  # IB algoStrategy: "Vwap" | "ArrivalPx" | "ClosePx" | "Adaptive" | "Twap"
    algo_params: dict[str, Any] = field(default_factory=dict)
    tif: str = "DAY"
    venue_hint: str | None = None  # "auction" | "continuous" | "open"
    order_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # sleeve, decision_px, routing_reason, ...

    def __post_init__(self):
        self.side = self.side.upper()
        if self.side not in SIDES:
            raise ValueError(f"side must be BUY/SELL, got {self.side}")
        if self.order_type not in ORDER_TYPES:
            raise ValueError(f"order_type must be in {ORDER_TYPES}")
        if self.qty < 0:
            raise ValueError("qty must be positive; use side for direction")

    @property
    def sign(self) -> int:
        return 1 if self.side == "BUY" else -1

    @property
    def signed_qty(self) -> float:
        return self.sign * self.qty

    def notional(self, px: float) -> float:
        return self.qty * px

    def with_(self, **changes) -> Order:
        from dataclasses import replace

        return replace(self, **changes)


def targets_to_orders(
    w_target: pd.Series,
    w_current: pd.Series,
    nav: float,
    prices: pd.Series,
    lot_size: int = 1,
    min_notional: float = 200.0,
    decision_ts=None,
    meta: dict[str, dict[str, Any]] | None = None,
) -> list[Order]:
    """Convert weight targets to share orders.

    ``w_target``/``w_current`` are weights (fraction of NAV); missing tickers
    count as 0.  Shares are rounded down to ``lot_size`` multiples; orders with
    notional below ``min_notional`` (dust) are skipped.  ``meta`` optionally maps
    ticker -> dict merged into each order's ``meta`` (e.g. ``{"sleeve":
    "reversal", "half_life_days": 3}``).  Each order's meta also gets
    ``decision_px`` (the price used) and ``target_w``/``current_w``.
    """
    idx = w_target.index.union(w_current.index)
    wt = w_target.reindex(idx).fillna(0.0).astype(float)
    wc = w_current.reindex(idx).fillna(0.0).astype(float)
    px = prices.reindex(idx).astype(float)
    dw = wt - wc
    orders: list[Order] = []
    for t in idx:
        p = px.loc[t]
        if not np.isfinite(p) or p <= 0:
            continue
        dollars = dw.loc[t] * float(nav)
        shares = int(np.floor(abs(dollars) / p / lot_size)) * lot_size
        if shares <= 0 or shares * p < min_notional:
            continue
        side = "BUY" if dollars > 0 else "SELL"
        m = {"decision_px": float(p), "target_w": float(wt.loc[t]), "current_w": float(wc.loc[t])}
        if decision_ts is not None:
            m["decision_ts"] = pd.Timestamp(decision_ts)
        if meta and t in meta:
            m.update(meta[t])
        # order_id is unique across days when decision_ts is given (TCALedger keys on it)
        prefix = f"{pd.Timestamp(decision_ts):%Y%m%d}-" if decision_ts is not None else ""
        orders.append(
            Order(
                ticker=str(t),
                side=side,
                qty=float(shares),
                order_type="MOC",
                venue_hint="auction",
                order_id=f"{prefix}{t}-{side}-{shares}",
                meta=m,
            )
        )
    return orders
