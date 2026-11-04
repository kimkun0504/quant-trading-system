"""Explicit (regulatory + broker) fee schedule, IBKR Pro tiered, 2026.

Sources are listed in docs/BUILD_PLAN.md section 4.  Everything here is a
*linear* function of shares or dollars traded, which is why the optimizer can
absorb it as a linear ``a|x|`` term (see :mod:`qts.tcost.impact`).

Components
----------
* Commission: ``commission_per_share`` per share, floor ``commission_min_per_order``
  per order, cap ``commission_max_pct_value`` of trade value (IBKR Pro tiered,
  first 300k shares/month tier; the lower tiers are not modelled).
* Clearing (NSCC/DTC): per share, capped at 0.5% of value.
* SEC Section 31: dollar-based, **sells only** ($20.60 per $1M from 2026-04-04).
* FINRA TAF: per share, **sells only**, capped per trade; **$0 during the
  2026-10-01..2026-12-31 fee holiday** (FINRA Schedule A).
* CAT fee: per share, both sides.

Exchange rebates / pass-through fees for adding or removing liquidity are not
modelled; they are small and sign-ambiguous for a mixed auction/algo flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import pandas as pd

from qts.core.config import CostConfig

DateLike = Union[str, pd.Timestamp, None]

# FINRA TAF holiday window (inclusive).  Documented in FINRA Schedule A for 2026.
TAF_HOLIDAY_START = pd.Timestamp("2026-10-01")
TAF_HOLIDAY_END = pd.Timestamp("2026-12-31")
# SEC Section 31 rate was $0 through 2026-04-03 (fee rate advisory 2026-2).
SEC_FEE_EFFECTIVE = pd.Timestamp("2026-04-04")
CLEARING_MAX_PCT_VALUE = 0.005


def _norm_side(side: str) -> str:
    s = str(side).lower()
    if s in ("buy", "b", "long", "cover"):
        return "buy"
    if s in ("sell", "s", "short", "sell_short", "ss"):
        return "sell"
    raise ValueError(f"unknown side {side!r}")


@dataclass
class FeeSchedule:
    cfg: CostConfig

    # ------------------------------------------------------------------ hooks
    @staticmethod
    def fee_holiday(date: DateLike) -> Dict[str, float]:
        """Return fee overrides active on ``date``.

        Currently only the FINRA TAF holiday: TAF per-share rate is 0 for
        trades dated 2026-10-01 through 2026-12-31 inclusive.  Returns a dict
        of ``{component: rate}`` overrides (empty when no holiday applies).
        ``date=None`` means "unknown date" -> no overrides.
        """
        if date is None:
            return {}
        d = pd.Timestamp(date).normalize()
        out: Dict[str, float] = {}
        if TAF_HOLIDAY_START <= d <= TAF_HOLIDAY_END:
            out["finra_taf_per_share_sold"] = 0.0
        if d < SEC_FEE_EFFECTIVE:
            out["sec_fee_per_dollar_sold"] = 0.0
        return out

    # ------------------------------------------------------------- per order
    def explicit_cost(self, shares: float, price: float, side: str,
                      date: DateLike = None) -> Dict[str, float]:
        """Dollar fees for one order of ``shares`` at ``price`` on ``side``.

        Returns a dict with keys ``commission, clearing, sec, taf, cat, total``
        (all >= 0 dollars).  ``shares`` may be negative; its absolute value is
        used.  ``date`` activates the fee-holiday overrides.
        """
        c = self.cfg
        q = abs(float(shares))
        value = q * float(price)
        side = _norm_side(side)
        over = self.fee_holiday(date)
        taf_rate = over.get("finra_taf_per_share_sold", c.finra_taf_per_share_sold)
        sec_rate = over.get("sec_fee_per_dollar_sold", c.sec_fee_per_dollar_sold)

        if q == 0:
            return {"commission": 0.0, "clearing": 0.0, "sec": 0.0, "taf": 0.0, "cat": 0.0, "total": 0.0}

        commission = q * c.commission_per_share
        commission = max(commission, c.commission_min_per_order)
        commission = min(commission, c.commission_max_pct_value * value)
        clearing = min(q * c.clearing_per_share, CLEARING_MAX_PCT_VALUE * value)
        cat = q * c.cat_fee_per_share
        sec = taf = 0.0
        if side == "sell":
            sec = value * sec_rate
            taf = min(q * taf_rate, c.finra_taf_max_per_trade)
        total = commission + clearing + sec + taf + cat
        return {"commission": commission, "clearing": clearing, "sec": sec,
                "taf": taf, "cat": cat, "total": total}

    def explicit_bps(self, notional: float, price: float, side: str,
                     date: DateLike = None) -> float:
        """Explicit fees in bps of ``notional`` for a trade of ``notional`` dollars
        in a stock priced ``price``."""
        notional = abs(float(notional))
        if notional == 0:
            return 0.0
        shares = notional / float(price)
        return 1e4 * self.explicit_cost(shares, price, side, date)["total"] / notional

    def round_trip_bps(self, notional: float, price: float, date: DateLike = None) -> float:
        """Buy + sell explicit cost in bps of one-way notional."""
        return (self.explicit_bps(notional, price, "buy", date)
                + self.explicit_bps(notional, price, "sell", date))

    # ------------------------------------------------------- linear approx
    def marginal_fraction(self, price: float, side: str, date: DateLike = None) -> float:
        """Marginal explicit cost per dollar traded (fraction), ignoring the
        per-order minimum and per-trade caps.  This is the linear coefficient
        used inside the convex optimizer."""
        c = self.cfg
        side = _norm_side(side)
        over = self.fee_holiday(date)
        taf_rate = over.get("finra_taf_per_share_sold", c.finra_taf_per_share_sold)
        sec_rate = over.get("sec_fee_per_dollar_sold", c.sec_fee_per_dollar_sold)
        per_share = c.commission_per_share + c.clearing_per_share + c.cat_fee_per_share
        frac = per_share / float(price)
        if side == "sell":
            frac += taf_rate / float(price) + sec_rate
        return frac
