"""Daily-bar fill simulator for routed orders (plan section 6: a daily-bar
backtester with the cvxportfolio-style cost model using realized volume
suffices for an auction-centric system).

Realized data per ticker (dict of Series or a DataFrame indexed by ticker)
with columns ``open, close, next_open, volume, sigma_daily`` and optionally
``auction_volume`` (shares; default ``volume * auction_share``, where the
share is ``auction_share`` if given, else :func:`qts.core.costs.auction_share_for_date`
of the trade date with ``CostConfig.auction_volume_share_default``),
``half_spread_bps`` and ``next_close`` (TCA reversion reference).

Fill rules
----------
* **MOC**: fills in full at ``close * (1 + s * impact_auction)`` where the
  impact uses ``b_auction`` and ``V = auction_volume``.
* **LOC**: same fill price; fills only if the limit is not violated (buy:
  fill_px <= limit; sell: fill_px >= limit).  Otherwise unfilled.
* **ALGO** (Vwap/ClosePx/ArrivalPx/Adaptive): fills at a VWAP proxy
  ``(open + close) / 2`` (ArrivalPx/Adaptive use ``open``) plus continuous
  impact (``b_continuous``, ``V = full-day realized volume``) with spread
  crossing; partial fill if ``qty > maxPctVol * volume``.
* **LMT** (passive reversal): fills at the limit, no impact or spread, if the
  close crossed the limit (buy: close <= limit; sell: close >= limit).
  Crude; a probabilistic fill model (NautilusTrader-style) is a later step.
* **MKT**: fills at ``open`` with continuous impact and spread.

Fees are computed from :class:`FeeSchedule` on filled shares.  If a
:class:`TCALedger` is supplied, each order is recorded (decision px from
``order.meta['decision_px']`` else the open; arrival = open for continuous /
``close`` for auction orders, approximating the 3:50 mid), fills appended and
the order closed with the realized ADV / sigma so :meth:`TCALedger.decompose`
works immediately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qts.core.config import CostConfig
from qts.core.costs import auction_share_for_date
from qts.execution.orders import Order
from qts.tcost.fees import FeeSchedule
from qts.tcost.impact import VENUE_AUCTION, VENUE_CONTINUOUS, SquareRootImpact
from qts.tcost.tca import TCALedger

FILL_COLS = [
    "order_id",
    "ticker",
    "side",
    "order_type",
    "algo",
    "qty",
    "filled_qty",
    "fill_px",
    "venue",
    "impact_bps",
    "spread_bps",
    "fees_usd",
    "status",
    "reason",
]


class FillSimulator:
    def __init__(
        self,
        cfg_cost: CostConfig | None = None,
        auction_share: float | None = None,
        default_max_pct_vol: float = 0.05,
    ):
        self.cfg = cfg_cost or CostConfig()
        self.fees = FeeSchedule(self.cfg)
        self.impact = SquareRootImpact(self.cfg)
        self.auction_share = auction_share
        self.default_max_pct_vol = default_max_pct_vol

    @staticmethod
    def _row(realized: pd.DataFrame | dict, ticker: str) -> dict[str, float]:
        if isinstance(realized, pd.DataFrame):
            r = realized.loc[ticker].to_dict()
        else:
            r = {k: (v.get(ticker) if isinstance(v, (pd.Series, dict)) else v) for k, v in realized.items()}
        return r

    def simulate(
        self,
        orders: list[Order],
        realized: pd.DataFrame | dict,
        date=None,
        ledger: TCALedger | None = None,
        decision_ts=None,
    ) -> pd.DataFrame:
        date = pd.Timestamp(date) if date is not None else pd.Timestamp.today().normalize()
        share = (
            float(self.auction_share)
            if self.auction_share is not None
            else auction_share_for_date(date, self.cfg.auction_volume_share_default)
        )
        rows = []
        for o in orders:
            r = self._row(realized, o.ticker)
            close = float(r["close"])
            open_ = float(r.get("open", np.nan))
            if not np.isfinite(open_):
                open_ = close
            next_open = float(r.get("next_open", np.nan))
            if not np.isfinite(next_open):
                next_open = close
            vol = float(r.get("volume", np.nan))
            sigma = float(r.get("sigma_daily", 0.02))
            hs_bps = float(r.get("half_spread_bps", self.cfg.default_half_spread_bps))
            auc_vol = float(r.get("auction_volume", np.nan))
            if not np.isfinite(auc_vol):
                auc_vol = vol * share
            s = o.sign
            filled, fill_px, impact_bps, spread_bps, venue, status, reason = 0.0, np.nan, 0.0, 0.0, "", "", ""

            if o.order_type in ("MOC", "LOC"):
                venue = VENUE_AUCTION
                impact_bps = float(self.impact.cost_bps(o.qty * close, auc_vol * close, sigma, venue=venue))
                fill_px = close * (1 + s * impact_bps / 1e4)
                if (
                    o.order_type == "LOC"
                    and o.limit_px is not None
                    and ((s > 0 and fill_px > o.limit_px) or (s < 0 and fill_px < o.limit_px))
                ):
                    filled, status, reason = (
                        0.0,
                        "unfilled",
                        f"LOC limit {o.limit_px} violated by auction px {fill_px:.4f}",
                    )
                    fill_px = np.nan
                else:
                    filled, status, reason = o.qty, "filled", "auction fill"
            elif o.order_type == "ALGO":
                venue = VENUE_CONTINUOUS
                max_pct = float(o.algo_params.get("maxPctVol", self.default_max_pct_vol))
                cap = max_pct * vol if np.isfinite(vol) else o.qty
                filled = float(min(o.qty, cap))
                ref = open_ if o.algo in ("ArrivalPx", "Adaptive") else 0.5 * (open_ + close)
                impact_bps = (
                    float(self.impact.cost_bps(filled * ref, vol * ref, sigma, venue=venue))
                    if filled > 0
                    else 0.0
                )
                spread_bps = hs_bps
                fill_px = ref * (1 + s * (impact_bps + spread_bps) / 1e4) if filled > 0 else np.nan
                status = "filled" if filled >= o.qty else ("partial" if filled > 0 else "unfilled")
                reason = f"{o.algo} at VWAP proxy, participation cap {max_pct:.0%} of {vol:.0f}"
            elif o.order_type == "LMT":
                venue = VENUE_CONTINUOUS
                lim = o.limit_px if o.limit_px is not None else close
                crossed = (s > 0 and close <= lim) or (s < 0 and close >= lim)
                if crossed:
                    filled, fill_px, status, reason = (
                        o.qty,
                        float(lim),
                        "filled",
                        "passive limit crossed by close",
                    )
                else:
                    status, reason = "unfilled", "passive limit not reached"
            elif o.order_type == "MKT":
                venue = VENUE_CONTINUOUS
                impact_bps = float(self.impact.cost_bps(o.qty * open_, vol * open_, sigma, venue=venue))
                spread_bps = hs_bps
                fill_px = open_ * (1 + s * (impact_bps + spread_bps) / 1e4)
                filled, status, reason = o.qty, "filled", "market at open"

            fees_usd = (
                self.fees.explicit_cost(filled, fill_px, o.side, date, venue=venue)["total"]
                if filled > 0
                else 0.0
            )
            rows.append(
                {
                    "order_id": o.order_id,
                    "ticker": o.ticker,
                    "side": o.side,
                    "order_type": o.order_type,
                    "algo": o.algo,
                    "qty": o.qty,
                    "filled_qty": filled,
                    "fill_px": fill_px,
                    "venue": venue,
                    "impact_bps": impact_bps,
                    "spread_bps": spread_bps,
                    "fees_usd": fees_usd,
                    "status": status,
                    "reason": reason,
                }
            )

            if ledger is not None:
                oid = o.order_id or f"{date:%Y%m%d}-{o.ticker}-{o.side}"
                if oid in ledger.orders:  # same id reused on a later day -> date prefix
                    oid = f"{date:%Y%m%d}-{oid}"
                dec_px = float(o.meta.get("decision_px", open_))
                dec_ts = o.meta.get("decision_ts", decision_ts if decision_ts is not None else date)
                arrival = close if venue == VENUE_AUCTION else open_
                arr_ts = date + (
                    pd.Timedelta(hours=15, minutes=50)
                    if venue == VENUE_AUCTION
                    else pd.Timedelta(hours=9, minutes=30)
                )
                ledger.record_order(
                    oid,
                    o.ticker,
                    o.side,
                    o.qty,
                    dec_px,
                    dec_ts,
                    arrival,
                    arr_ts,
                    venue=venue,
                    order_type=o.order_type,
                    algo=o.algo,
                )
                if filled > 0:
                    ledger.record_fill(oid, filled, fill_px, date + pd.Timedelta(hours=16), venue)
                adv = float(r.get("adv_shares", np.nan))
                if not np.isfinite(adv):
                    adv = vol  # fallback: day's realised volume (audit #16)
                # V the fill was priced on: realised auction volume / realised day volume
                venue_vol = auc_vol if venue == VENUE_AUCTION else vol
                nc = r.get("next_close", np.nan)
                nc = float(nc) if nc is not None and np.isfinite(float(nc)) else None
                ledger.close_order(
                    oid,
                    close,
                    next_open,
                    adv,
                    sigma,
                    fees_usd,
                    half_spread_bps=spread_bps,
                    venue_volume=venue_vol,
                    next_close_px=nc,
                )
        return pd.DataFrame(rows, columns=FILL_COLS)
