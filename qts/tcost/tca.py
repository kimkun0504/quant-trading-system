"""Transaction cost analysis ledger and the monthly ``b`` calibration job.

Every parent order is logged with its decision price (signal time), arrival
price (submission), fills, the day's close, next open, ADV and sigma.  The
Perold implementation-shortfall decomposition (BUILD_PLAN section 6) is, with
``s = +1`` for buys and ``-1`` for sells, ``F`` filled shares, ``U`` unfilled
shares, ``Q = F + U`` and all quantities in **bps of decision notional**
``Q * P_decision``, signed so that **positive = cost**::

    delay        = s * F * (P_arrival - P_decision)
    impact       = s * F * (VWAP_fill - P_arrival)
    opportunity  = s * U * (P_close   - P_decision)
    fees         = explicit dollar fees
    total_is     = delay + impact + opportunity + fees
                 = s * [F*(VWAP - P_dec) + U*(P_close - P_dec)] + fees

    reversion    = s * F * (VWAP_fill - P_ref)          (diagnostic; NOT in total)

``P_ref`` is the **next close** by default (BUILD_PLAN convention;
``CostConfig.tca_reversion_reference = "next_close"``, supplied through
``close_order(next_close_px=)``) or the next open (``"next_open"``).  A
missing reference price gives NaN reversion, never a silent 0.
Reversion is positive when the price moved back against the fill after the
trade (a buy followed by a lower reference price), i.e. the part of the impact
that turned out to be temporary.  It is reported but, per the plan, never
credited as a benefit in the optimizer.

Calibration regresses, pooled across orders, ``y = s*(VWAP - P_arr)/P_arr -
half_spread`` (return units) on ``x = sigma * (F / V)^(e-1)`` through the
origin, and shrinks the OLS slope toward the prior ``b`` using
``prior_weight`` pseudo-observations.  ``e = cost_cfg.impact_exponent``: the
cost model's integrated impact of ``Q`` dollars is ``C(Q) = b*sigma*Q^e/V^(e-1)``
(USD, ``V`` in USD), so the *average* impact per dollar filled -- what the fill
VWAP measures against arrival, in return units -- is
``C(Q)/Q = b*sigma*(Q/V)^(e-1)``; the participation ``Q/V`` is the same in
shares or dollars.  ``e = 1.5`` gives the familiar ``sigma*sqrt(F/V)``
(Almgren et al. 2005; Toth et al. 2011).  Regressing on ``sqrt`` with another
``e`` would bias ``b`` by orders of magnitude (audit 2, finding 2).

**Venue volume.**  ``V`` is the volume the cost model uses for the order's
venue: full-day ADV for continuous-session orders and ``auction_share * ADV``
for closing-auction orders (``b_auction`` is defined against auction volume;
regressing auction fills on full ADV would recover ``b_auction/sqrt(share)``,
3.3x too large at a 9% share).  The venue comes from ``record_order(venue=)``
or ``meta['order_type'] in {MOC, LOC}``; the auction share from
``close_order(auction_share=)``, else :func:`qts.core.costs.auction_share_for_date`
on the decision date with ``cost_cfg.auction_volume_share_default``; ``close_order(venue_volume=)`` overrides ``V`` outright
(e.g. the realised auction print).  Use :meth:`TCALedger.calibrate_b_by_venue`
to update ``impact_b_auction`` and ``impact_b_continuous`` separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
import pandas as pd

from qts.core.config import CostConfig
from qts.core.costs import auction_share_for_date

REVERSION_REFERENCES = ("next_close", "next_open")

ADV_BUCKETS = [0.0, 0.01, 0.05, 0.10, np.inf]
ADV_LABELS = ["<1%ADV", "1-5%ADV", "5-10%ADV", ">10%ADV"]
VOL_BUCKETS = [0.0, 0.015, 0.03, np.inf]
VOL_LABELS = ["low_vol", "mid_vol", "high_vol"]

ORDER_COLUMNS = [
    "order_id",
    "ticker",
    "side",
    "qty",
    "decision_px",
    "decision_ts",
    "arrival_px",
    "arrival_ts",
    "close_px",
    "next_open_px",
    "next_close_px",
    "adv_shares",
    "sigma_daily",
    "fees",
    "half_spread_bps",
    "venue",
    "auction_share",
    "venue_volume",
    "closed",
    "meta",
]
VENUE_AUCTION = "auction"
VENUE_CONTINUOUS = "continuous"
AUCTION_ORDER_TYPES = ("MOC", "LOC")
FILL_COLUMNS = ["order_id", "qty", "px", "ts", "venue"]


def _sign(side: str) -> int:
    s = str(side).lower()
    if s in ("buy", "b", "cover", "long"):
        return 1
    if s in ("sell", "s", "short", "sell_short", "ss"):
        return -1
    raise ValueError(f"unknown side {side!r}")


class CalibrationDiagnostics(TypedDict):
    """Diagnostics of :meth:`TCALedger.calibrate_b`."""

    n: int
    b_ols: float
    r2: float
    se: float
    prior_b: float
    prior_weight: float
    venue: str | None


class ImpactCalibration(NamedTuple):
    """:meth:`TCALedger.calibrate_b` result; unpacks as ``(b, diagnostics)``."""

    b: float
    diagnostics: CalibrationDiagnostics


@dataclass
class TCALedger:
    orders: dict[str, dict] = field(default_factory=dict)
    fills: list[dict] = field(default_factory=list)
    #: priors (``calibrate_b_by_venue``), default auction share and the
    #: reversion reference (``tca_reversion_reference``) come from here
    cost_cfg: CostConfig = field(default_factory=CostConfig)

    # ---------------------------------------------------------------- record
    def record_order(
        self,
        order_id: str,
        ticker: str,
        side: str,
        qty: float,
        decision_px: float,
        decision_ts,
        arrival_px: float,
        arrival_ts,
        venue: str | None = None,
        **meta,
    ) -> None:
        """``venue`` is "auction" or "continuous"; when omitted it is derived from
        ``meta['order_type']`` (MOC/LOC -> auction) and defaults to continuous.
        ``order_id`` must be unique across days (prefix it with the date)."""
        if order_id in self.orders:
            raise KeyError(f"order {order_id} already recorded")
        if venue is None:
            venue = (
                VENUE_AUCTION
                if str(meta.get("order_type", "")).upper() in AUCTION_ORDER_TYPES
                else VENUE_CONTINUOUS
            )
        self.orders[order_id] = {
            "order_id": order_id,
            "ticker": ticker,
            "side": side,
            "qty": abs(float(qty)),
            "decision_px": float(decision_px),
            "decision_ts": pd.Timestamp(decision_ts),
            "arrival_px": float(arrival_px),
            "arrival_ts": pd.Timestamp(arrival_ts),
            "close_px": np.nan,
            "next_open_px": np.nan,
            "next_close_px": np.nan,
            "adv_shares": np.nan,
            "sigma_daily": np.nan,
            "fees": 0.0,
            "half_spread_bps": 0.0,
            "venue": str(venue),
            "auction_share": np.nan,
            "venue_volume": np.nan,
            "closed": False,
            "meta": dict(meta),
        }

    def record_fill(self, order_id: str, qty: float, px: float, ts, venue: str = "") -> None:
        if order_id not in self.orders:
            raise KeyError(f"unknown order {order_id}")
        self.fills.append(
            {
                "order_id": order_id,
                "qty": abs(float(qty)),
                "px": float(px),
                "ts": pd.Timestamp(ts),
                "venue": venue,
            }
        )

    def close_order(
        self,
        order_id: str,
        close_px: float,
        next_open_px: float,
        adv_shares: float,
        sigma_daily: float,
        fees: float = 0.0,
        half_spread_bps: float = 0.0,
        auction_share: float | None = None,
        venue_volume: float | None = None,
        next_close_px: float | None = None,
    ) -> None:
        """``adv_shares`` is the 63-day ADV (participation denominator).
        ``auction_share`` (auction orders) sets ``V = auction_share * adv_shares``;
        ``venue_volume`` (shares) overrides ``V`` directly for either venue.
        ``next_close_px`` is the reversion reference under the default
        ``tca_reversion_reference = "next_close"``."""
        o = self.orders[order_id]
        o.update(
            close_px=float(close_px),
            next_open_px=float(next_open_px),
            next_close_px=np.nan if next_close_px is None else float(next_close_px),
            adv_shares=float(adv_shares),
            sigma_daily=float(sigma_daily),
            fees=float(fees),
            half_spread_bps=float(half_spread_bps),
            closed=True,
            auction_share=np.nan if auction_share is None else float(auction_share),
            venue_volume=np.nan if venue_volume is None else float(venue_volume),
        )

    # ----------------------------------------------------------------- views
    def orders_frame(self) -> pd.DataFrame:
        if not self.orders:
            return pd.DataFrame(columns=ORDER_COLUMNS).set_index("order_id")
        df = pd.DataFrame(list(self.orders.values()))
        return df.set_index("order_id")

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=FILL_COLUMNS)
        return pd.DataFrame(self.fills)

    def _fill_stats(self) -> pd.DataFrame:
        f = self.fills_frame()
        if f.empty:
            return pd.DataFrame(columns=["filled_qty", "fill_vwap"])
        f = f.assign(notional=f["qty"] * f["px"])
        g = f.groupby("order_id").agg(filled_qty=("qty", "sum"), notional=("notional", "sum"))
        g["fill_vwap"] = g["notional"] / g["filled_qty"]
        return g[["filled_qty", "fill_vwap"]]

    # ------------------------------------------------------------ decompose
    def decompose(self, closed_only: bool = True) -> pd.DataFrame:
        """Per-order cost decomposition in bps of decision notional (positive = cost).

        Columns: ticker, side, qty, filled_qty, fill_rate, decision_notional,
        delay_bps, impact_bps, opportunity_bps, fees_bps, total_is_bps,
        reversion_bps (vs ``cost_cfg.tca_reversion_reference``), arrival_slippage_bps (unweighted VWAP vs arrival),
        participation (filled / ADV), sigma_daily, adv_bucket, vol_bucket, venue,
        venue_volume (V: ADV for continuous, auction_share*ADV for auction),
        venue_participation (F/V), impact_x (= sigma*(F/V)^(e-1), the calibration
        regressor).
        """
        o = self.orders_frame()
        if closed_only:
            o = o[o["closed"].astype(bool)]
        if o.empty:
            return pd.DataFrame()
        for col, default in (
            ("venue", VENUE_CONTINUOUS),
            ("auction_share", np.nan),
            ("venue_volume", np.nan),
            ("next_close_px", np.nan),
        ):
            if col not in o.columns:  # ledgers saved before venue tracking
                o[col] = default
        fs = self._fill_stats().reindex(o.index)
        filled = fs["filled_qty"].fillna(0.0)
        vwap = fs["fill_vwap"]
        s = o["side"].map(_sign).astype(float)
        qty = o["qty"].astype(float)
        unfilled = (qty - filled).clip(lower=0.0)
        pdec, parr, pclose, pnext = o["decision_px"], o["arrival_px"], o["close_px"], o["next_open_px"]
        notional = qty * pdec
        k = 1e4 / notional

        delay = s * filled * (parr - pdec) * k
        impact = (s * filled * (vwap - parr) * k).fillna(0.0)
        opportunity = s * unfilled * (pclose - pdec) * k
        fees = o["fees"].astype(float) * k
        total = delay + impact + opportunity + fees
        ref = str(self.cost_cfg.tca_reversion_reference)
        if ref not in REVERSION_REFERENCES:
            raise ValueError(f"tca_reversion_reference must be one of {REVERSION_REFERENCES}, got {ref!r}")
        pref = o["next_close_px"].astype(float) if ref == "next_close" else pnext
        reversion = (s * filled * (vwap - pref) * k).where(filled > 0, 0.0)
        arr_slip = s * (vwap - parr) / parr * 1e4
        adv = o["adv_shares"].astype(float)
        participation = filled / adv
        venue = o["venue"].fillna(VENUE_CONTINUOUS).astype(str)
        is_auc = venue == VENUE_AUCTION
        share = o["auction_share"].astype(float)
        sd = float(self.cost_cfg.auction_volume_share_default)
        share_default = pd.Series([auction_share_for_date(ts, sd) for ts in o["decision_ts"]], index=o.index)
        share = share.fillna(share_default)
        venue_vol = adv.where(~is_auc, adv * share)
        venue_vol = o["venue_volume"].astype(float).fillna(venue_vol)
        e = float(self.cost_cfg.impact_exponent)
        with np.errstate(invalid="ignore", divide="ignore"):
            # average impact per unit return of C(Q) = b sigma Q^e / V^(e-1): b sigma (Q/V)^(e-1)
            impact_x = o["sigma_daily"].astype(float) * (filled / venue_vol) ** (e - 1.0)

        out = pd.DataFrame(
            {
                "ticker": o["ticker"],
                "side": o["side"],
                "qty": qty,
                "filled_qty": filled,
                "fill_rate": filled / qty,
                "decision_notional": notional,
                "delay_bps": delay,
                "impact_bps": impact,
                "opportunity_bps": opportunity,
                "fees_bps": fees,
                "total_is_bps": total,
                "reversion_bps": reversion,
                "arrival_slippage_bps": arr_slip,
                "participation": participation,
                "sigma_daily": o["sigma_daily"].astype(float),
                "half_spread_bps": o["half_spread_bps"].astype(float),
                "venue": venue,
                "venue_volume": venue_vol,
                "venue_participation": filled / venue_vol,
                "impact_x": impact_x,
                "decision_ts": o["decision_ts"],
            }
        )
        out["adv_bucket"] = pd.cut(
            out["participation"].fillna(0.0), ADV_BUCKETS, labels=ADV_LABELS, include_lowest=True, right=False
        ).astype(str)
        out["vol_bucket"] = pd.cut(
            out["sigma_daily"].fillna(0.0), VOL_BUCKETS, labels=VOL_LABELS, include_lowest=True, right=False
        ).astype(str)
        return out

    def summary(self, by: Sequence[str] = ("adv_bucket", "vol_bucket")) -> pd.DataFrame:
        """Notional-weighted mean of each cost component (bps) by bucket, plus counts."""
        d = self.decompose()
        if d.empty:
            return pd.DataFrame()
        cols = [
            "delay_bps",
            "impact_bps",
            "opportunity_bps",
            "fees_bps",
            "total_is_bps",
            "reversion_bps",
            "arrival_slippage_bps",
        ]
        w = d["decision_notional"]

        def agg(g: pd.DataFrame) -> pd.Series:
            ww = w.loc[g.index]
            res = {}
            for c in cols:
                ok = g[c].notna()  # e.g. reversion without a reference price: excluded, not 0
                wc = ww[ok]
                res[c] = float(np.average(g.loc[ok, c], weights=wc)) if wc.sum() > 0 else np.nan
            res["n_orders"] = len(g)
            res["notional"] = float(ww.sum())
            res["fill_rate"] = float(np.average(g["fill_rate"], weights=ww)) if ww.sum() > 0 else np.nan
            return pd.Series(res)

        return d.groupby(list(by), observed=True).apply(agg, include_groups=False)

    # ------------------------------------------------------------ calibrate
    def calibrate_b(
        self,
        prior_b: float = 1.0,
        prior_weight: float = 20.0,
        min_fill_rate: float = 0.5,
        venue: str | None = None,
    ) -> ImpactCalibration:
        """Through-the-origin regression of arrival slippage on ``sigma*(F/V)^(e-1)``.

        ``y_i = s_i*(VWAP_i - P_arr_i)/P_arr_i - half_spread_i`` (return units),
        ``x_i = sigma_i * (F_i / V_i)^(e-1)`` with ``e = cost_cfg.impact_exponent``
        (module docstring: average impact per dollar of ``C(Q) = b sigma Q^e/V^(e-1)``;
        ``sqrt`` at e = 1.5) and ``V_i`` the **venue** volume
        (ADV for continuous, auction_share*ADV for auction orders).
        ``b_ols = sum(xy)/sum(x^2)``.  Shrinkage:
        ``b = (n*b_ols + prior_weight*prior_b) / (n + prior_weight)``.

        ``venue=None`` pools all orders (each on its own ``V``; only meaningful
        if ``b`` is believed common across venues); ``venue="auction"`` /
        ``"continuous"`` restricts to that venue -- that is what should feed
        ``impact_b_auction`` / ``impact_b_continuous`` (see
        :meth:`calibrate_b_by_venue`).

        Returns an :class:`ImpactCalibration` ``(b, diagnostics)`` (a named
        tuple: ``b, diag = ...`` still works) where diagnostics has ``n, b_ols,
        r2, se, prior_b, prior_weight, venue``.  With no usable orders ``b =
        prior_b``.
        """
        empty: CalibrationDiagnostics = {
            "n": 0,
            "b_ols": np.nan,
            "r2": np.nan,
            "se": np.nan,
            "prior_b": float(prior_b),
            "prior_weight": float(prior_weight),
            "venue": venue,
        }
        d = self.decompose()
        if d.empty:
            return ImpactCalibration(float(prior_b), empty)
        d = d[
            (d["filled_qty"] > 0)
            & (d["fill_rate"] >= min_fill_rate)
            & np.isfinite(d["impact_x"])
            & (d["impact_x"] > 0)
        ]
        if venue is not None:
            d = d[d["venue"] == venue]
        y = (d["arrival_slippage_bps"] - d["half_spread_bps"]).values / 1e4
        x = d["impact_x"].values
        n = len(x)
        if n == 0:
            return ImpactCalibration(float(prior_b), empty)
        sxx = float(x @ x)
        b_ols = float(x @ y) / sxx
        resid = y - b_ols * x
        dof = max(n - 1, 1)
        se = float(np.sqrt((resid @ resid) / dof / sxx))
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = float(1 - (resid @ resid) / ss_tot) if ss_tot > 0 else np.nan
        b = (n * b_ols + prior_weight * prior_b) / (n + prior_weight)
        return ImpactCalibration(
            float(b),
            {
                "n": int(n),
                "b_ols": b_ols,
                "r2": r2,
                "se": se,
                "prior_b": float(prior_b),
                "prior_weight": float(prior_weight),
                "venue": venue,
            },
        )

    def calibrate_b_by_venue(
        self, prior_b: dict[str, float] | None = None, prior_weight: float = 20.0, min_fill_rate: float = 0.5
    ) -> dict[str, ImpactCalibration]:
        """:meth:`calibrate_b` once per venue.  ``prior_b`` maps venue -> prior
        (default ``self.cost_cfg.impact_b_auction`` / ``impact_b_continuous``).
        Returns ``{"auction": (b, diag), "continuous": (b, diag)}``; the two
        numbers are the updates for ``impact_b_auction`` and ``impact_b_continuous``."""
        if prior_b is None:
            c = self.cost_cfg
            prior_b = {VENUE_AUCTION: c.impact_b_auction, VENUE_CONTINUOUS: c.impact_b_continuous}
        return {
            v: self.calibrate_b(
                prior_b=prior_b.get(v, 1.0), prior_weight=prior_weight, min_fill_rate=min_fill_rate, venue=v
            )
            for v in (VENUE_AUCTION, VENUE_CONTINUOUS)
        }

    # ------------------------------------------------------------ persist
    def save(self, path: str | Path, fmt: str = "parquet") -> None:
        """Write ``<path>_orders.<fmt>`` and ``<path>_fills.<fmt>``."""
        import json

        path = Path(path)
        o = self.orders_frame().reset_index()
        o["meta"] = o["meta"].apply(lambda m: json.dumps(m or {}, default=str))
        f = self.fills_frame()
        if fmt == "parquet":
            o.to_parquet(path.with_name(path.stem + "_orders.parquet"), index=False)
            f.to_parquet(path.with_name(path.stem + "_fills.parquet"), index=False)
        elif fmt == "csv":
            o.to_csv(path.with_name(path.stem + "_orders.csv"), index=False)
            f.to_csv(path.with_name(path.stem + "_fills.csv"), index=False)
        else:
            raise ValueError(fmt)

    @classmethod
    def load(cls, path: str | Path, fmt: str = "parquet", cost_cfg: CostConfig | None = None) -> TCALedger:
        import json

        path = Path(path)
        if fmt == "parquet":
            o = pd.read_parquet(path.with_name(path.stem + "_orders.parquet"))
            f = pd.read_parquet(path.with_name(path.stem + "_fills.parquet"))
        else:
            o = pd.read_csv(
                path.with_name(path.stem + "_orders.csv"), parse_dates=["decision_ts", "arrival_ts"]
            )
            f = pd.read_csv(path.with_name(path.stem + "_fills.csv"), parse_dates=["ts"])
        led = cls(cost_cfg=cost_cfg or CostConfig())
        for rec in o.to_dict("records"):
            m = rec.get("meta", "{}")
            try:
                rec["meta"] = json.loads(m) if isinstance(m, str) else (m or {})
            except (TypeError, ValueError):
                rec["meta"] = {}
            rec["decision_ts"] = pd.Timestamp(rec["decision_ts"])
            rec["arrival_ts"] = pd.Timestamp(rec["arrival_ts"])
            led.orders[rec["order_id"]] = rec
        for rec in f.to_dict("records"):
            rec["ts"] = pd.Timestamp(rec["ts"])
            led.fills.append(rec)
        return led
