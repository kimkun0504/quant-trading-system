"""IBKR payload builders and a dry-run broker.  **No network code here.**

Real connectivity is a later phase and will go through ``ib_async``
(``ib_async.IB().placeOrder(contract, order)``).  The dicts built here mirror
the ``ib_async.Order`` attribute names (``orderType``, ``lmtPrice``, ``tif``,
``algoStrategy``, ``algoParams`` as a list of ``TagValue(tag, value)``) so the
adapter is ``Order(**payload)`` plus ``TagValue`` wrapping.

IMPORTANT: algo names and parameter tags below follow the TWS API reference
(``Vwap``, ``ArrivalPx``, ``ClosePx``, ``Adaptive``, ``Twap``, ``PctVol``) with
tags ``maxPctVol, startTime, endTime, noTakeLiq, riskAversion, adaptivePriority,
forceCompletion, allowPastEndTime, speedUp``.  The GitHub reference is marked
deprecated in favour of IBKR Campus; **verify every tag and value range against
the current IBKR Campus docs before live use** (plan section 6).

Values: ``maxPctVol`` 0.01-0.5; ``riskAversion`` in
{"Get Done","Aggressive","Neutral","Passive"}; ``adaptivePriority`` in
{"Urgent","Normal","Patient"}; times "hh:mm:ss TMZ".  ``ClosePx`` also takes
``maxPctVol`` and ``riskAversion``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from qts.core.config import CostConfig
from qts.execution.orders import Order
from qts.tcost.impact import VENUE_AUCTION, VENUE_CONTINUOUS, SquareRootImpact

NY_TZ = "America/New_York"
KNOWN_ALGOS = {"Vwap", "ArrivalPx", "ClosePx", "Adaptive", "Twap", "PctVol", "DarkIce", "AD"}
ALGO_PARAM_ORDER = ["maxPctVol", "startTime", "endTime", "noTakeLiq", "riskAversion", "adaptivePriority",
                    "forceCompletion", "allowPastEndTime", "speedUp", "pctVol", "displaySize"]


def _tag_value(tag: str, value: Any) -> Dict[str, str]:
    if isinstance(value, bool):
        v = "1" if value else "0"
    else:
        v = str(value)
    return {"tag": tag, "value": v}


def to_ib_order_dict(order: Order, account: Optional[str] = None, transmit: bool = True,
                     outside_rth: bool = False) -> Dict[str, Any]:
    """Build an ib_async-style ``Order`` kwargs dict from an :class:`Order`.

    Mapping: MOC -> orderType "MOC"; LOC -> "LOC" + lmtPrice; LMT -> "LMT" +
    lmtPrice; MKT -> "MKT"; ALGO -> orderType "LMT" if limit_px else "MKT", plus
    ``algoStrategy`` and ``algoParams`` list of ``{"tag","value"}`` dicts.
    Also returns a ``contract`` sub-dict (STK/SMART/USD) and ``orderRef`` set to
    the internal order id.
    """
    payload: Dict[str, Any] = {
        "action": order.side,
        "totalQuantity": float(order.qty),
        "tif": order.tif,
        "outsideRth": outside_rth,
        "transmit": transmit,
        "orderRef": order.order_id or "",
        "contract": {"symbol": order.ticker, "secType": "STK", "exchange": "SMART",
                     "primaryExchange": "", "currency": "USD"},
    }
    if account:
        payload["account"] = account
    ot = order.order_type
    if ot in ("MOC", "MKT"):
        payload["orderType"] = ot
    elif ot in ("LOC", "LMT"):
        if order.limit_px is None:
            raise ValueError(f"{ot} order needs limit_px")
        payload["orderType"] = ot
        payload["lmtPrice"] = float(order.limit_px)
    elif ot == "ALGO":
        if not order.algo:
            raise ValueError("ALGO order needs algo strategy")
        if order.algo not in KNOWN_ALGOS:
            raise ValueError(f"unknown IB algoStrategy {order.algo!r}")
        payload["orderType"] = "LMT" if order.limit_px is not None else "MKT"
        if order.limit_px is not None:
            payload["lmtPrice"] = float(order.limit_px)
        payload["algoStrategy"] = order.algo
        params = dict(order.algo_params)
        ordered = [k for k in ALGO_PARAM_ORDER if k in params] + [k for k in params if k not in ALGO_PARAM_ORDER]
        payload["algoParams"] = [_tag_value(k, params[k]) for k in ordered]
    else:  # pragma: no cover
        raise ValueError(ot)
    return payload


def auction_deadline(date, exchange: str = "NYSE") -> pd.Timestamp:
    """Latest time to enter/modify/cancel MOC/LOC orders.  NYSE: 15:50 ET
    (after which only contra-imbalance orders are accepted).  Nasdaq: 15:50 ET
    for MOC/LOC entry (imbalance-only orders until 15:55).  Returns a
    tz-aware ``America/New_York`` timestamp."""
    d = pd.Timestamp(date)
    d = d.tz_localize(None) if d.tzinfo is not None else d
    cutoff = {"NYSE": "15:50", "NASDAQ": "15:50", "ARCA": "15:59"}.get(exchange.upper(), "15:50")
    hh, mm = (int(v) for v in cutoff.split(":"))
    return (d.normalize() + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(NY_TZ)


def is_before_auction_deadline(now, date=None, exchange: str = "NYSE") -> bool:
    """``now`` naive -> interpreted as ET; aware -> converted to ET."""
    now = pd.Timestamp(now)
    now = now.tz_localize(NY_TZ) if now.tzinfo is None else now.tz_convert(NY_TZ)
    day = pd.Timestamp(date) if date is not None else now.tz_localize(None).normalize()
    return now < auction_deadline(day, exchange)


@dataclass
class DryRunBroker:
    """Accepts orders and simulates fills at a provided reference price plus
    square-root impact.  For paper-trading plumbing tests only (the realistic
    simulator lives in :mod:`qts.execution.simulator_fills`)."""

    cfg_cost: CostConfig = field(default_factory=CostConfig)
    submitted: List[Dict[str, Any]] = field(default_factory=list)
    fills: List[Dict[str, Any]] = field(default_factory=list)
    _next_id: int = 1

    def __post_init__(self):
        self.impact = SquareRootImpact(self.cfg_cost)

    def place_order(self, order: Order) -> int:
        payload = to_ib_order_dict(order)
        oid = self._next_id
        self._next_id += 1
        self.submitted.append({"ib_order_id": oid, "order": order, "payload": payload, "status": "Submitted"})
        return oid

    def simulate_fill(self, ib_order_id: int, ref_px: float, adv_shares: float, sigma_daily: float,
                      ts=None, venue: Optional[str] = None) -> Dict[str, Any]:
        rec = next(r for r in self.submitted if r["ib_order_id"] == ib_order_id)
        o: Order = rec["order"]
        venue = venue or (VENUE_AUCTION if o.order_type in ("MOC", "LOC") else VENUE_CONTINUOUS)
        vol_sh = adv_shares * (0.09 if venue == VENUE_AUCTION else 1.0)
        bps = float(self.impact.cost_bps(o.qty * ref_px, vol_sh * ref_px, sigma_daily, venue=venue))
        px = ref_px * (1 + o.sign * bps / 1e4)
        filled = o.qty
        if o.order_type in ("LOC", "LMT") and o.limit_px is not None and (
                (o.side == "BUY" and px > o.limit_px) or (o.side == "SELL" and px < o.limit_px)):
            filled = 0.0
        fill = {"ib_order_id": ib_order_id, "order_id": o.order_id, "ticker": o.ticker, "side": o.side,
                "qty": filled, "px": px if filled else np.nan, "ts": pd.Timestamp(ts) if ts is not None else pd.Timestamp.now(),
                "venue": venue, "impact_bps": bps}
        rec["status"] = "Filled" if filled else "Cancelled"
        self.fills.append(fill)
        return fill

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.fills)
