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

    reversion    = s * F * (VWAP_fill - P_next_open)   (diagnostic; NOT in total)

Reversion is positive when the price moved back against the fill after the
trade (a buy followed by a lower next open), i.e. the part of the impact that
turned out to be temporary.  It is reported but, per the plan, never credited
as a benefit in the optimizer.

Calibration regresses, pooled across orders, ``y = s*(VWAP - P_arr)/P_arr -
half_spread`` (return units) on ``x = sigma * sqrt(F / V)`` through the
origin, and shrinks the OLS slope toward the prior ``b`` using
``prior_weight`` pseudo-observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

ADV_BUCKETS = [0.0, 0.01, 0.05, 0.10, np.inf]
ADV_LABELS = ["<1%ADV", "1-5%ADV", "5-10%ADV", ">10%ADV"]
VOL_BUCKETS = [0.0, 0.015, 0.03, np.inf]
VOL_LABELS = ["low_vol", "mid_vol", "high_vol"]

ORDER_COLUMNS = ["order_id", "ticker", "side", "qty", "decision_px", "decision_ts", "arrival_px",
                 "arrival_ts", "close_px", "next_open_px", "adv_shares", "sigma_daily", "fees",
                 "half_spread_bps", "closed", "meta"]
FILL_COLUMNS = ["order_id", "qty", "px", "ts", "venue"]


def _sign(side: str) -> int:
    s = str(side).lower()
    if s in ("buy", "b", "cover", "long"):
        return 1
    if s in ("sell", "s", "short", "sell_short", "ss"):
        return -1
    raise ValueError(f"unknown side {side!r}")


@dataclass
class TCALedger:
    orders: Dict[str, dict] = field(default_factory=dict)
    fills: List[dict] = field(default_factory=list)

    # ---------------------------------------------------------------- record
    def record_order(self, order_id: str, ticker: str, side: str, qty: float, decision_px: float,
                     decision_ts, arrival_px: float, arrival_ts, **meta) -> None:
        if order_id in self.orders:
            raise KeyError(f"order {order_id} already recorded")
        self.orders[order_id] = dict(
            order_id=order_id, ticker=ticker, side=side, qty=abs(float(qty)),
            decision_px=float(decision_px), decision_ts=pd.Timestamp(decision_ts),
            arrival_px=float(arrival_px), arrival_ts=pd.Timestamp(arrival_ts),
            close_px=np.nan, next_open_px=np.nan, adv_shares=np.nan, sigma_daily=np.nan,
            fees=0.0, half_spread_bps=0.0, closed=False, meta=dict(meta),
        )

    def record_fill(self, order_id: str, qty: float, px: float, ts, venue: str = "") -> None:
        if order_id not in self.orders:
            raise KeyError(f"unknown order {order_id}")
        self.fills.append(dict(order_id=order_id, qty=abs(float(qty)), px=float(px),
                               ts=pd.Timestamp(ts), venue=venue))

    def close_order(self, order_id: str, close_px: float, next_open_px: float, adv_shares: float,
                    sigma_daily: float, fees: float = 0.0, half_spread_bps: float = 0.0) -> None:
        o = self.orders[order_id]
        o.update(close_px=float(close_px), next_open_px=float(next_open_px),
                 adv_shares=float(adv_shares), sigma_daily=float(sigma_daily),
                 fees=float(fees), half_spread_bps=float(half_spread_bps), closed=True)

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
        reversion_bps, arrival_slippage_bps (unweighted VWAP vs arrival),
        participation (filled / ADV), sigma_daily, adv_bucket, vol_bucket,
        impact_x (= sigma*sqrt(F/ADV), the calibration regressor).
        """
        o = self.orders_frame()
        if closed_only:
            o = o[o["closed"].astype(bool)]
        if o.empty:
            return pd.DataFrame()
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
        reversion = (s * filled * (vwap - pnext) * k).fillna(0.0)
        arr_slip = (s * (vwap - parr) / parr * 1e4)
        participation = filled / o["adv_shares"].astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            impact_x = o["sigma_daily"].astype(float) * np.sqrt(participation)

        out = pd.DataFrame({
            "ticker": o["ticker"], "side": o["side"], "qty": qty, "filled_qty": filled,
            "fill_rate": filled / qty, "decision_notional": notional,
            "delay_bps": delay, "impact_bps": impact, "opportunity_bps": opportunity,
            "fees_bps": fees, "total_is_bps": total, "reversion_bps": reversion,
            "arrival_slippage_bps": arr_slip, "participation": participation,
            "sigma_daily": o["sigma_daily"].astype(float), "half_spread_bps": o["half_spread_bps"].astype(float),
            "impact_x": impact_x, "decision_ts": o["decision_ts"],
        })
        out["adv_bucket"] = pd.cut(out["participation"].fillna(0.0), ADV_BUCKETS, labels=ADV_LABELS,
                                   include_lowest=True, right=False).astype(str)
        out["vol_bucket"] = pd.cut(out["sigma_daily"].fillna(0.0), VOL_BUCKETS, labels=VOL_LABELS,
                                   include_lowest=True, right=False).astype(str)
        return out

    def summary(self, by: Sequence[str] = ("adv_bucket", "vol_bucket")) -> pd.DataFrame:
        """Notional-weighted mean of each cost component (bps) by bucket, plus counts."""
        d = self.decompose()
        if d.empty:
            return pd.DataFrame()
        cols = ["delay_bps", "impact_bps", "opportunity_bps", "fees_bps", "total_is_bps",
                "reversion_bps", "arrival_slippage_bps"]
        w = d["decision_notional"]

        def agg(g: pd.DataFrame) -> pd.Series:
            ww = w.loc[g.index]
            res = {c: float(np.average(g[c].fillna(0.0), weights=ww)) if ww.sum() > 0 else np.nan for c in cols}
            res["n_orders"] = len(g)
            res["notional"] = float(ww.sum())
            res["fill_rate"] = float(np.average(g["fill_rate"], weights=ww)) if ww.sum() > 0 else np.nan
            return pd.Series(res)

        return d.groupby(list(by), observed=True).apply(agg, include_groups=False)

    # ------------------------------------------------------------ calibrate
    def calibrate_b(self, prior_b: float = 1.0, prior_weight: float = 20.0,
                    min_fill_rate: float = 0.5) -> Tuple[float, Dict[str, float]]:
        """Pooled through-the-origin regression of arrival slippage on ``sigma*sqrt(F/V)``.

        ``y_i = s_i*(VWAP_i - P_arr_i)/P_arr_i - half_spread_i`` (return units),
        ``x_i = sigma_i * sqrt(F_i / ADV_i)``.  ``b_ols = sum(xy)/sum(x^2)``.
        Shrinkage: ``b = (n*b_ols + prior_weight*prior_b) / (n + prior_weight)``.

        Returns ``(b, diagnostics)`` where diagnostics has ``n, b_ols, r2, se,
        prior_b, prior_weight``.  With no usable orders returns ``prior_b``.
        """
        d = self.decompose()
        if d.empty:
            return float(prior_b), {"n": 0, "b_ols": np.nan, "r2": np.nan, "se": np.nan,
                                    "prior_b": prior_b, "prior_weight": prior_weight}
        d = d[(d["filled_qty"] > 0) & (d["fill_rate"] >= min_fill_rate) & np.isfinite(d["impact_x"]) & (d["impact_x"] > 0)]
        y = (d["arrival_slippage_bps"] - d["half_spread_bps"]).values / 1e4
        x = d["impact_x"].values
        n = len(x)
        if n == 0:
            return float(prior_b), {"n": 0, "b_ols": np.nan, "r2": np.nan, "se": np.nan,
                                    "prior_b": prior_b, "prior_weight": prior_weight}
        sxx = float(x @ x)
        b_ols = float(x @ y) / sxx
        resid = y - b_ols * x
        dof = max(n - 1, 1)
        se = float(np.sqrt((resid @ resid) / dof / sxx))
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = float(1 - (resid @ resid) / ss_tot) if ss_tot > 0 else np.nan
        b = (n * b_ols + prior_weight * prior_b) / (n + prior_weight)
        return float(b), {"n": int(n), "b_ols": b_ols, "r2": r2, "se": se,
                          "prior_b": float(prior_b), "prior_weight": float(prior_weight)}

    # ------------------------------------------------------------ persist
    def save(self, path: Union[str, Path], fmt: str = "parquet") -> None:
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
    def load(cls, path: Union[str, Path], fmt: str = "parquet") -> "TCALedger":
        import json
        path = Path(path)
        if fmt == "parquet":
            o = pd.read_parquet(path.with_name(path.stem + "_orders.parquet"))
            f = pd.read_parquet(path.with_name(path.stem + "_fills.parquet"))
        else:
            o = pd.read_csv(path.with_name(path.stem + "_orders.csv"), parse_dates=["decision_ts", "arrival_ts"])
            f = pd.read_csv(path.with_name(path.stem + "_fills.csv"), parse_dates=["ts"])
        led = cls()
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
