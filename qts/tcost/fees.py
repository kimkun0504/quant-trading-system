"""Explicit (regulatory + broker + exchange) fee schedule, IBKR Pro tiered, 2026.

Sources are listed in docs/BUILD_PLAN.md section 4.  Everything here is a
*linear* function of shares or dollars traded, which is why the optimizer can
absorb it as a linear ``a|x|`` term (see :mod:`qts.tcost.impact`).

Components
----------
* Commission: ``commission_per_share`` per share, floor ``commission_min_per_order``
  per order, cap ``commission_max_pct_value`` of trade value (IBKR Pro tiered,
  first 300k shares/month tier; the lower tiers are not modelled).
* Clearing (NSCC/DTC): per share, capped at 0.5% of value.
* SEC Section 31: dollar-based, **sells only**.  Dated table
  :data:`SEC_SECTION31_RATES` (``$20.60`` per $1M from 2026-04-04, ``$0`` from
  2025-05-14, ``$27.80`` from 2024-05-22, ...).  ``date=None`` uses
  ``cfg.sec_fee_per_dollar_sold``.
* FINRA TAF: per share, **sells only**, capped per trade; dated table
  :data:`FINRA_TAF_RATES`; **$0 during the 2026-10-01..2026-12-31 fee holiday**
  (SR-FINRA-2026-021).
* CAT fee: per share, both sides.
* Closing-auction exchange fee (NYSE/Nasdaq MOC/LOC): per share, both sides,
  **auction venue only** (``cfg.auction_exchange_fee_per_share``).

Other exchange rebates / pass-through fees for adding or removing liquidity in
the continuous session are not modelled; they are small and sign-ambiguous
for a mixed auction/algo flow.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from qts.core.config import CostConfig

DateLike = str | pd.Timestamp | None

# FINRA TAF holiday window (inclusive).  SR-FINRA-2026-021 Exhibit 5A.
TAF_HOLIDAY_START = pd.Timestamp("2026-10-01")
TAF_HOLIDAY_END = pd.Timestamp("2026-12-31")
# Kept for backward compatibility: first day of the current (FY2026) non-zero rate.
SEC_FEE_EFFECTIVE = pd.Timestamp("2026-04-04")
CLEARING_MAX_PCT_VALUE = 0.005

# ---------------------------------------------------------------------------
# SEC Section 31 transaction-fee rate, dollars per $1M of covered sales, by
# effective date (charge date = trade date for OTC / FINRA-reported sales).
# Each row was verified against the SEC fee-rate advisory or the FINRA
# Information Notice announcing it unless marked otherwise.  A date before the
# first row uses the first row's rate (the then-current rate is unknown here
# but was never zero before FY2025), never 0.
SEC_SECTION31_RATES: list[tuple[pd.Timestamp, float]] = [
    (pd.Timestamp("2016-02-16"), 21.80),   # FY2016 (rate confirmed as the prior rate in FINRA IN 6/9/17; date unverified)
    (pd.Timestamp("2017-07-04"), 23.10),   # FY2017 #3, FINRA IN 6/9/17
    (pd.Timestamp("2018-05-22"), 13.00),   # FY2018 #3, FINRA IN 5/7/18
    (pd.Timestamp("2019-04-16"), 20.70),   # FY2019 #2, FINRA IN 4/3/19
    (pd.Timestamp("2020-02-18"), 22.10),   # FY2020 #3, FINRA IN 2/3/20
    (pd.Timestamp("2021-02-25"), 5.10),    # FY2021 #3, FINRA IN 2/16/21
    (pd.Timestamp("2022-05-14"), 22.90),   # FY2022 #2, SEC press release 2022-60
    (pd.Timestamp("2023-02-27"), 8.00),    # FY2023 #3, FINRA IN 2/27/23
    (pd.Timestamp("2024-05-22"), 27.80),   # FY2024 #2, SEC fee rate advisory 2024-2
    (pd.Timestamp("2025-05-14"), 0.00),    # FY2025 #2, SEC fee rate advisory 2025-2
    (pd.Timestamp("2026-04-04"), 20.60),   # FY2026 #2, SEC fee rate advisory 2026-2
]
_SEC_DATES = [d for d, _ in SEC_SECTION31_RATES]

# FINRA Trading Activity Fee for covered equity securities: (effective date,
# per share sold, per-trade cap).  2024-2026 rows verified against the
# SR-FINRA-2024-019 fee-adjustment schedule; SR-FINRA-2026-020 keeps the 2026
# rate in place through 2028-12-31 (per SR-FINRA-2026-021), superseding the
# 2027/2028 step-ups in the 2024 schedule.  Pre-2024 rows are from memory and
# unverified (TODO 2026-11-04: confirm against FINRA Schedule A history).
FINRA_TAF_RATES: list[tuple[pd.Timestamp, float, float]] = [
    (pd.Timestamp("2012-01-01"), 0.000119, 5.95),   # unverified
    (pd.Timestamp("2022-01-01"), 0.000130, 6.49),   # unverified
    (pd.Timestamp("2023-01-01"), 0.000145, 7.27),   # unverified
    (pd.Timestamp("2024-01-01"), 0.000166, 8.30),   # verified (also 2025, no change)
    (pd.Timestamp("2026-01-01"), 0.000195, 9.79),   # verified
]
_TAF_DATES = [d for d, _, _ in FINRA_TAF_RATES]


def sec_section31_rate(date: DateLike, cfg: CostConfig | None = None) -> float:
    """SEC Section 31 fee per **dollar** sold in force on ``date``.

    Picks the last :data:`SEC_SECTION31_RATES` row with effective date ``<=
    date``; dates before the first row use the first row's rate.  ``date=None``
    (unknown) returns ``cfg.sec_fee_per_dollar_sold``.
    """
    if date is None:
        return float((cfg or CostConfig()).sec_fee_per_dollar_sold)
    d = pd.Timestamp(date).normalize()
    i = bisect.bisect_right(_SEC_DATES, d) - 1
    rate_per_million = SEC_SECTION31_RATES[max(i, 0)][1]
    return rate_per_million / 1e6


def finra_taf_rate(date: DateLike, cfg: CostConfig | None = None) -> tuple[float, float]:
    """``(per_share_sold, per_trade_cap)`` FINRA TAF in force on ``date``.

    Applies the 2026-Q4 fee holiday (both 0).  ``date=None`` returns the config
    rates; dates before the first table row use the first row.
    """
    c = cfg or CostConfig()
    if date is None:
        return float(c.finra_taf_per_share_sold), float(c.finra_taf_max_per_trade)
    d = pd.Timestamp(date).normalize()
    if TAF_HOLIDAY_START <= d <= TAF_HOLIDAY_END:
        return 0.0, 0.0
    i = bisect.bisect_right(_TAF_DATES, d) - 1
    _, per_share, cap = FINRA_TAF_RATES[max(i, 0)]
    return float(per_share), float(cap)


def _norm_side(side: str) -> str:
    s = str(side).lower()
    if s in ("buy", "b", "long", "cover"):
        return "buy"
    if s in ("sell", "s", "short", "sell_short", "ss"):
        return "sell"
    raise ValueError(f"unknown side {side!r}")


def _is_auction(venue: str | None) -> bool:
    return venue is not None and str(venue).lower() == "auction"


@dataclass
class FeeSchedule:
    cfg: CostConfig

    # ------------------------------------------------------------------ hooks
    def fee_holiday(self, date: DateLike) -> dict[str, float]:
        """Return the dated fee overrides active on ``date``.

        Keys: ``finra_taf_per_share_sold``, ``finra_taf_max_per_trade`` and
        ``sec_fee_per_dollar_sold`` from the dated tables (TAF is 0 for trades
        dated 2026-10-01 through 2026-12-31 inclusive).  ``date=None`` means
        "unknown date" -> no overrides (config rates apply).
        """
        if date is None:
            return {}
        taf, cap = finra_taf_rate(date, self.cfg)
        return {"finra_taf_per_share_sold": taf, "finra_taf_max_per_trade": cap,
                "sec_fee_per_dollar_sold": sec_section31_rate(date, self.cfg)}

    def _rates(self, date: DateLike) -> tuple[float, float, float]:
        c = self.cfg
        over = self.fee_holiday(date)
        return (over.get("finra_taf_per_share_sold", c.finra_taf_per_share_sold),
                over.get("finra_taf_max_per_trade", c.finra_taf_max_per_trade),
                over.get("sec_fee_per_dollar_sold", c.sec_fee_per_dollar_sold))

    # ------------------------------------------------------------- per order
    def explicit_cost(self, shares: float, price: float, side: str,
                      date: DateLike = None, venue: str | None = None) -> dict[str, float]:
        """Dollar fees for one order of ``shares`` at ``price`` on ``side``.

        Returns a dict with keys ``commission, clearing, sec, taf, cat, exchange,
        total`` (all >= 0 dollars).  ``shares`` may be negative; its absolute
        value is used.  ``date`` selects the dated SEC/TAF rates; ``venue ==
        "auction"`` adds the closing-auction exchange fee.
        """
        c = self.cfg
        q = abs(float(shares))
        value = q * float(price)
        side = _norm_side(side)
        taf_rate, taf_cap, sec_rate = self._rates(date)
        zero = {"commission": 0.0, "clearing": 0.0, "sec": 0.0, "taf": 0.0, "cat": 0.0,
                "exchange": 0.0, "total": 0.0}
        if q == 0:
            return zero

        commission = q * c.commission_per_share
        commission = max(commission, c.commission_min_per_order)
        commission = min(commission, c.commission_max_pct_value * value)
        clearing = min(q * c.clearing_per_share, CLEARING_MAX_PCT_VALUE * value)
        cat = q * c.cat_fee_per_share
        exchange = q * getattr(c, "auction_exchange_fee_per_share", 0.0) if _is_auction(venue) else 0.0
        sec = taf = 0.0
        if side == "sell":
            sec = value * sec_rate
            taf = min(q * taf_rate, taf_cap)
        total = commission + clearing + sec + taf + cat + exchange
        return {"commission": commission, "clearing": clearing, "sec": sec,
                "taf": taf, "cat": cat, "exchange": exchange, "total": total}

    def explicit_bps(self, notional: float, price: float, side: str,
                     date: DateLike = None, venue: str | None = None) -> float:
        """Explicit fees in bps of ``notional`` for a trade of ``notional`` dollars
        in a stock priced ``price``."""
        notional = abs(float(notional))
        if notional == 0:
            return 0.0
        shares = notional / float(price)
        return 1e4 * self.explicit_cost(shares, price, side, date, venue)["total"] / notional

    def round_trip_bps(self, notional: float, price: float, date: DateLike = None,
                       venue: str | None = None) -> float:
        """Buy + sell explicit cost in bps of one-way notional."""
        return (self.explicit_bps(notional, price, "buy", date, venue)
                + self.explicit_bps(notional, price, "sell", date, venue))

    # ------------------------------------------------------------ vectorised
    def explicit_cost_vec(self, shares, price, is_sell, date: DateLike = None,
                          is_auction=False) -> np.ndarray:
        """Vectorised :meth:`explicit_cost` ``["total"]``: dollar fees for one
        order per element (``shares`` >= 0 or signed -- abs is used; ``is_sell``
        / ``is_auction`` boolean arrays or scalars).  Same formulas, same dated
        SEC/TAF rates, same per-order minimum and caps; elements with zero
        shares or non-positive price cost 0.  ~O(N) numpy instead of N Python
        calls building dicts (audit performance item 9)."""
        c = self.cfg
        q = np.abs(np.asarray(shares, dtype=float))
        p = np.asarray(price, dtype=float)
        q, p = np.broadcast_arrays(q, p)
        sell = np.broadcast_to(np.asarray(is_sell, dtype=bool), q.shape)
        auc = np.broadcast_to(np.asarray(is_auction, dtype=bool), q.shape)
        taf_rate, taf_cap, sec_rate = self._rates(date)
        value = q * p
        commission = np.minimum(np.maximum(q * c.commission_per_share, c.commission_min_per_order),
                                c.commission_max_pct_value * value)
        clearing = np.minimum(q * c.clearing_per_share, CLEARING_MAX_PCT_VALUE * value)
        cat = q * c.cat_fee_per_share
        exchange = np.where(auc, q * getattr(c, "auction_exchange_fee_per_share", 0.0), 0.0)
        sec = np.where(sell, value * sec_rate, 0.0)
        taf = np.where(sell, np.minimum(q * taf_rate, taf_cap), 0.0)
        total = commission + clearing + cat + exchange + sec + taf
        ok = (q > 0) & np.isfinite(p) & (p > 0)
        return np.where(ok, total, 0.0)

    def marginal_fraction_vec(self, price, is_sell, date: DateLike = None, is_auction=False) -> np.ndarray:
        """Vectorised :meth:`marginal_fraction` (0 where price <= 0)."""
        c = self.cfg
        p = np.asarray(price, dtype=float)
        sell = np.broadcast_to(np.asarray(is_sell, dtype=bool), p.shape)
        auc = np.broadcast_to(np.asarray(is_auction, dtype=bool), p.shape)
        taf_rate, _, sec_rate = self._rates(date)
        per_share = c.commission_per_share + c.clearing_per_share + c.cat_fee_per_share \
            + np.where(auc, getattr(c, "auction_exchange_fee_per_share", 0.0), 0.0)
        per_share = per_share + np.where(sell, taf_rate, 0.0)
        ok = np.isfinite(p) & (p > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = per_share / p + np.where(sell, sec_rate, 0.0)
        return np.where(ok, frac, 0.0)

    # ------------------------------------------------------- linear approx
    def marginal_fraction(self, price: float, side: str, date: DateLike = None,
                          venue: str | None = None) -> float:
        """Marginal explicit cost per dollar traded (fraction), ignoring the
        per-order minimum and per-trade caps.  This is the linear coefficient
        used inside the convex optimizer."""
        c = self.cfg
        side = _norm_side(side)
        taf_rate, _, sec_rate = self._rates(date)
        per_share = c.commission_per_share + c.clearing_per_share + c.cat_fee_per_share
        if _is_auction(venue):
            per_share += getattr(c, "auction_exchange_fee_per_share", 0.0)
        frac = per_share / float(price)
        if side == "sell":
            frac += taf_rate / float(price) + sec_rate
        return frac
