"""Order router implementing the tactic table of BUILD_PLAN section 6.

| Situation                                                  | Tactic                                  |
|------------------------------------------------------------|-----------------------------------------|
| order <= auction_max_pct x expected closing-auction volume | LOC, indicative +/- loc_offset_bps      |
|   (buy limit above / sell limit below the indicative,      |   submitted before the 15:50 ET cutoff  |
|    i.e. "through" the price on the order's side)           |                                         |
| larger / less liquid                                       | ALGO ClosePx (low urgency) or Vwap,     |
|                                                            |   maxPctVol = urgency_from_half_life    |
|                                                            |   clipped to [1%, algo_max_pct_adv]     |
| signal half-life < urgency threshold                       | ALGO ArrivalPx (or Adaptive) next open  |
| reversal-sleeve order (``meta['sleeve']=='reversal'`` or   | passive LMT at/inside the indicative,   |
|   ``meta['reversal']``)                                    |   never marketable                      |
| order > algo_max_pct_adv x ADV                             | ALGO capped; flagged as multi-day       |

The snapshot passed to :meth:`ExecutionRouter.route` is a dict of per-ticker
Series: ``adv_shares, auction_volume_est, sigma_daily, half_spread_bps,
indicative_close_px``.  Missing ``auction_volume_est`` is derived from
``adv_shares * auction_share_for_date(date, CostConfig.auction_volume_share_default)``
(date-aware: ~20% of ADV on quad-witching, ~34% on Russell days; finding #21).

Participation (``maxPctVol``) for algo orders comes from the signal half-life
through :func:`qts.execution.scheduler.urgency_from_half_life` (faster decay ->
higher participation), clipped to ``[1%, cfg.algo_max_pct_adv]``; unknown
half-life -> ``algo_max_pct_adv``.  An order larger than ``maxPctVol`` x ADV is
flagged ``multi_day``.

IBKR tags: ``ClosePx`` takes ``maxPctVol, riskAversion, startTime,
forceCompletion`` only -- ``noTakeLiq`` is not a documented ClosePx parameter
(verified 2026-11-04 against the TWS API IB Algos reference and the IBKR
Campus Close Price lesson; finding #22) and is no longer sent.  ``Vwap`` does
document ``noTakeLiq``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from qts.core.config import CostConfig, ExecutionConfig
from qts.core.costs import auction_share_for_date
from qts.execution.orders import Order
from qts.execution.scheduler import urgency_from_half_life

ALGO_CLOSE = "ClosePx"
ALGO_VWAP = "Vwap"
ALGO_ARRIVAL = "ArrivalPx"
ALGO_ADAPTIVE = "Adaptive"


class ExecutionRouter:
    """``loc_offset_impact_mult``: the LOC offset is
    ``max(cfg.loc_offset_bps, loc_offset_impact_mult * expected_auction_impact_bps)``
    where the expected impact is ``b_auction * sigma * sqrt(qty / auction_volume)``.
    A fixed few-bp offset is only consistent with the cost model for orders that
    are tiny relative to the auction; without this floor the limit would be
    violated by the order's own expected impact.  Set to 0 to use the fixed offset.
    """

    def __init__(self, cfg: ExecutionConfig | None = None, cost_cfg: CostConfig | None = None,
                 loc_offset_impact_mult: float = 1.5):
        self.cfg = cfg or ExecutionConfig()
        self.cost_cfg = cost_cfg or CostConfig()
        self.loc_offset_impact_mult = loc_offset_impact_mult

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _get(snapshot: dict, key: str, ticker: str, default=np.nan) -> float:
        s = snapshot.get(key)
        if s is None:
            return default
        if isinstance(s, (pd.Series, dict)):
            v = s.get(ticker, default)
        else:
            v = s
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return v if np.isfinite(v) else default

    @staticmethod
    def is_reversal(order: Order) -> bool:
        m = order.meta or {}
        return bool(m.get("reversal")) or str(m.get("sleeve", "")).lower() in ("reversal", "str", "short_term_reversal")

    def loc_offset_bps(self, qty: float, auction_volume: float, sigma_daily: float) -> float:
        """Offset in bps through the indicative: see class docstring."""
        off = float(self.cfg.loc_offset_bps)
        if self.loc_offset_impact_mult > 0 and np.isfinite(auction_volume) and auction_volume > 0 and np.isfinite(sigma_daily):
            e = float(self.cost_cfg.impact_exponent)
            exp_imp = 1e4 * self.cost_cfg.impact_b_auction * sigma_daily * (qty / auction_volume) ** (e - 1.0)
            off = max(off, self.loc_offset_impact_mult * exp_imp)
        return off

    @staticmethod
    def tick_size(px: float) -> float:
        """Reg NMS minimum increment: $0.01 at/above $1, $0.0001 below."""
        return 0.01 if px >= 1.0 else 0.0001

    @classmethod
    def round_to_tick(cls, px: float, direction: str) -> float:
        """Round ``px`` to the tick, ``direction`` "up" (ceil) or "down" (floor)."""
        tick = cls.tick_size(px)
        n = px / tick
        n = round(n, 6)                       # kill float noise (7.0021/0.01 = 700.21 exactly)
        k = math.ceil(n) if direction == "up" else math.floor(n)
        return round(k * tick, 4 if tick < 0.01 else 2)

    def loc_limit(self, side: str, indicative_px: float, offset_bps: float | None = None) -> float:
        """Limit ``offset_bps`` *through* the indicative on the order's side: buys
        above, sells below, so the order is marketable in the auction but
        capped against a dislocated print.  Rounded to the tick in the
        favourable (through) direction -- ceil for buys, floor for sells -- so
        a small offset on a low-priced stock never rounds back onto the
        indicative (a 3 bp offset on a $7 stock is $0.0021, i.e. one tick)."""
        off = (self.cfg.loc_offset_bps if offset_bps is None else offset_bps) / 1e4
        if side == "BUY":
            return self.round_to_tick(indicative_px * (1 + off), "up")
        return self.round_to_tick(indicative_px * (1 - off), "down")

    def passive_limit(self, side: str, indicative_px: float, half_spread_bps: float) -> float:
        """Passive limit for reversal trades: buy at/below the bid, sell at/above
        the ask.  Rounded away from the indicative (buys floor, sells ceil) so
        the order stays passive after rounding."""
        hs = max(float(half_spread_bps), 0.0) / 1e4
        if side == "BUY":
            return self.round_to_tick(indicative_px * (1 - hs), "down")
        return self.round_to_tick(indicative_px * (1 + hs), "up")

    def participation(self, half_life_days: float | None) -> float:
        """``maxPctVol`` for an algo order: ``urgency_from_half_life(h)`` clipped
        to ``[0.01, cfg.algo_max_pct_adv]`` (``algo_max_pct_adv`` if ``h`` unknown)."""
        cap = float(min(self.cfg.algo_max_pct_adv, 0.5))
        if half_life_days is None or not np.isfinite(float(half_life_days)):
            return cap
        return float(np.clip(urgency_from_half_life(float(half_life_days)), min(0.01, cap), cap))

    # --------------------------------------------------------------- route
    def route(self, orders: list[Order], snapshot: dict, signal_half_life_days: float | None = None,
              date=None) -> list[Order]:
        """Return new Order objects with order_type/limit/algo set and
        ``meta['routing_reason']`` populated.  Per-order ``meta['half_life_days']``
        overrides ``signal_half_life_days``."""
        cfg = self.cfg
        share = auction_share_for_date(date, self.cost_cfg.auction_volume_share_default)
        routed: list[Order] = []
        for o in orders:
            t = o.ticker
            adv = self._get(snapshot, "adv_shares", t)
            auc = self._get(snapshot, "auction_volume_est", t)
            if not np.isfinite(auc) and np.isfinite(adv):
                auc = adv * share
            ind = self._get(snapshot, "indicative_close_px", t, o.meta.get("decision_px", np.nan))
            hs = self._get(snapshot, "half_spread_bps", t, 2.0)
            sig = self._get(snapshot, "sigma_daily", t, np.nan)
            hl = o.meta.get("half_life_days", signal_half_life_days)
            pct_auc = o.qty / auc if (np.isfinite(auc) and auc > 0) else np.inf
            pct_adv = o.qty / adv if (np.isfinite(adv) and adv > 0) else np.inf
            meta = dict(o.meta)
            meta.update(pct_of_auction_volume=pct_auc, pct_of_adv=pct_adv)

            # 1) reversal sleeve: passive, never marketable
            if self.is_reversal(o):
                lim = self.passive_limit(o.side, ind, hs) if np.isfinite(ind) else None
                meta["routing_reason"] = (f"reversal sleeve -> passive LMT at {lim} "
                                          f"(liquidity provision; never cross the spread)")
                routed.append(o.with_(order_type="LMT", limit_px=lim, algo=None, algo_params={},
                                      tif="DAY", venue_hint="continuous", meta=meta))
                continue

            # 2) fast-decaying signal: arrival-price algo at next open
            if hl is not None and float(hl) < cfg.urgency_half_life_days_threshold:
                params = {"maxPctVol": self.participation(hl), "riskAversion": "Aggressive",
                          "forceCompletion": False, "allowPastEndTime": True}
                meta["routing_reason"] = (f"half-life {float(hl):.1f}d < {cfg.urgency_half_life_days_threshold}d "
                                          f"-> ALGO ArrivalPx at next open, maxPctVol {params['maxPctVol']:.1%}")
                routed.append(o.with_(order_type="ALGO", algo=ALGO_ARRIVAL, algo_params=params, limit_px=None,
                                      venue_hint="open", meta=meta))
                continue

            # 3) small relative to the closing auction: LOC through the indicative
            if pct_auc <= cfg.auction_max_pct_of_auction_volume and np.isfinite(ind):
                off = self.loc_offset_bps(o.qty, auc, sig)
                lim = self.loc_limit(o.side, ind, off)
                meta["loc_offset_bps"] = off
                meta["routing_reason"] = (f"{pct_auc:.2%} of est. auction volume <= "
                                          f"{cfg.auction_max_pct_of_auction_volume:.0%} -> LOC "
                                          f"{'above' if o.side == 'BUY' else 'below'} indicative {ind:.2f} "
                                          f"by {off:.1f}bp; submit before 15:50 ET")
                routed.append(o.with_(order_type="LOC", limit_px=lim, algo=None, algo_params={},
                                      venue_hint="auction", meta=meta))
                continue
            if pct_auc <= cfg.auction_max_pct_of_auction_volume:  # no indicative price available
                meta["routing_reason"] = (f"{pct_auc:.2%} of est. auction volume but no indicative price -> MOC")
                routed.append(o.with_(order_type="MOC", limit_px=None, venue_hint="auction", meta=meta))
                continue

            # 4) larger: Close Price algo (low urgency) or VWAP, participation-capped
            max_pct = self.participation(hl)
            low_urgency = hl is None or float(hl) >= 2 * cfg.urgency_half_life_days_threshold
            algo = ALGO_CLOSE if low_urgency else ALGO_VWAP
            params = {"maxPctVol": max_pct}
            if algo == ALGO_VWAP:     # noTakeLiq is documented for Vwap, not for ClosePx (#22)
                params.update(startTime="13:00:00 US/Eastern", endTime="16:00:00 US/Eastern", noTakeLiq=False)
            reason = (f"{pct_auc:.1%} of est. auction volume > {cfg.auction_max_pct_of_auction_volume:.0%} "
                      f"-> ALGO {algo}, maxPctVol {max_pct:.1%}")
            if pct_adv > max_pct:
                reason += (f"; WARNING order is {pct_adv:.1%} of ADV > {max_pct:.1%} participation cap: "
                           f"expect partial fill / multi-day working")
                meta["multi_day"] = True
            meta["routing_reason"] = reason
            routed.append(o.with_(order_type="ALGO", algo=algo, algo_params=params, limit_px=None,
                                  venue_hint="continuous", meta=meta))
        return routed

    def routing_table(self, routed: list[Order]) -> pd.DataFrame:
        rows = [{"ticker": o.ticker, "side": o.side, "qty": o.qty, "order_type": o.order_type,
                 "algo": o.algo, "limit_px": o.limit_px, "venue_hint": o.venue_hint,
                 "pct_auction": o.meta.get("pct_of_auction_volume"), "pct_adv": o.meta.get("pct_of_adv"),
                 "reason": o.meta.get("routing_reason")} for o in routed]
        return pd.DataFrame(rows)
