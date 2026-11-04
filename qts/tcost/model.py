"""``TransactionCostModel``: facade over fees, impact and borrow for the
portfolio layer (cvxpy terms), the backtester (point estimates) and reporting.

Snapshot contract
-----------------
A *snapshot* is a ``dict`` of per-ticker ``pd.Series`` (shared index), built by
:meth:`TransactionCostModel.snapshot_from_market` using data ``<= date`` only:

``adv_usd``        63d mean dollar volume
``adv_shares``     63d mean share volume
``sigma_daily``    63d standard deviation of daily simple returns
``half_spread_bps`` quoted half-spread proxy in bps (``md.spread_bps`` or default)
``half_spread``    the same as a fraction
``price``          last close
``borrow_fee``     annual borrow fee (``md.borrow_fee`` or default)
``venue``          "auction" (default) or "continuous" per ticker
``date``           the as-of timestamp (scalar)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from qts.core.config import CostConfig
from qts.core.types import MarketData
from qts.tcost.borrow import borrow_fee_vector, cvx_holding_expression, holding_cost_breakdown
from qts.tcost.fees import FeeSchedule
from qts.tcost.impact import VENUE_AUCTION, SquareRootImpact

SNAPSHOT_KEYS = ("adv_usd", "adv_shares", "sigma_daily", "half_spread_bps", "half_spread",
                 "price", "borrow_fee", "venue")


class TransactionCostModel:
    def __init__(self, cfg: Optional[CostConfig] = None):
        self.cfg = cfg or CostConfig()
        self.fees = FeeSchedule(self.cfg)
        self.impact = SquareRootImpact(self.cfg)

    # ------------------------------------------------------------ snapshot
    def snapshot_from_market(self, md: MarketData, date, cfg_adv_window: int = 63,
                             default_venue: str = VENUE_AUCTION,
                             auction_share: Optional[float] = None) -> Dict[str, pd.Series]:
        """Build the per-ticker cost snapshot as of ``date`` (point-in-time).

        ``adv_usd`` for the ``venue`` used by :meth:`estimate` is full-day ADV;
        auction-venue names use ``auction_share * adv`` internally.
        """
        date = pd.Timestamp(date)
        close = md.close.loc[:date].tail(cfg_adv_window + 1)
        vol = md.volume.loc[:date].tail(cfg_adv_window)
        if close.empty:
            raise ValueError(f"no data on or before {date}")
        dv = (close.tail(cfg_adv_window) * vol).mean(axis=0, skipna=True)
        adv_sh = vol.mean(axis=0, skipna=True)
        rets = close.pct_change(fill_method=None).tail(cfg_adv_window)
        sigma = rets.std(axis=0, ddof=1, skipna=True)
        price = close.iloc[-1]
        tickers = md.tickers
        if md.spread_bps is not None:
            hs_bps = md.spread_bps.loc[:date].iloc[-1].reindex(tickers)
            hs_bps = hs_bps.fillna(self.cfg.default_half_spread_bps)
        else:
            hs_bps = pd.Series(self.cfg.default_half_spread_bps, index=tickers)
        if md.borrow_fee is not None:
            bf = md.borrow_fee.loc[:date].iloc[-1].reindex(tickers).fillna(self.cfg.default_borrow_fee_annual)
        else:
            bf = pd.Series(self.cfg.default_borrow_fee_annual, index=tickers)
        snap = {
            "adv_usd": dv.reindex(tickers).astype(float),
            "adv_shares": adv_sh.reindex(tickers).astype(float),
            "sigma_daily": sigma.reindex(tickers).astype(float),
            "half_spread_bps": hs_bps.astype(float),
            "half_spread": hs_bps.astype(float) / 1e4,
            "price": price.reindex(tickers).astype(float),
            "borrow_fee": bf.astype(float),
            "venue": pd.Series(default_venue, index=tickers),
            "date": date,
        }
        if auction_share is not None:
            snap["auction_share"] = float(auction_share)
        return snap

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _series(snap: Dict, key: str, index: pd.Index, default) -> pd.Series:
        v = snap.get(key)
        if v is None:
            return pd.Series(default, index=index)
        if isinstance(v, pd.Series):
            return v.reindex(index).fillna(default)
        return pd.Series(v, index=index)

    def _venue_volume(self, snap: Dict, index: pd.Index, venue_override: Optional[str] = None) -> Tuple[pd.Series, pd.Series]:
        """Return (volume_for_venue_usd, b_per_name) given per-name venue."""
        adv = self._series(snap, "adv_usd", index, np.nan).astype(float)
        venue = self._series(snap, "venue", index, VENUE_AUCTION) if venue_override is None \
            else pd.Series(venue_override, index=index)
        share = float(snap.get("auction_share", self.impact.auction_share_for_date(snap.get("date"))))
        is_auc = (venue == VENUE_AUCTION)
        vol = adv.where(~is_auc, adv * share)
        b = pd.Series(np.where(is_auc, self.cfg.impact_b_auction, self.cfg.impact_b_continuous), index=index)
        return vol, b

    # ------------------------------------------------------------ estimate
    def estimate(self, trades_usd: pd.Series, md_snapshot: Dict, venue: Optional[str] = None,
                 date=None, w_post: Optional[pd.Series] = None) -> pd.DataFrame:
        """Per-ticker expected cost of ``trades_usd`` (signed dollars).

        Columns: ``trade_usd, venue, spread_bps, impact_bps, explicit_bps, total_bps,
        spread_usd, impact_usd, explicit_usd, total_usd`` and, if ``w_post`` is
        given (weights) together with a NAV in the snapshot (``md_snapshot['nav']``),
        ``borrow_usd`` (one day of holding cost).  ``venue`` overrides the
        per-name venue in the snapshot.
        """
        idx = trades_usd.index
        x = trades_usd.astype(float)
        sigma = self._series(md_snapshot, "sigma_daily", idx, np.nan).astype(float)
        price = self._series(md_snapshot, "price", idx, np.nan).astype(float)
        hs_bps = self._series(md_snapshot, "half_spread_bps", idx, self.cfg.default_half_spread_bps).astype(float)
        vol, b = self._venue_volume(md_snapshot, idx, venue)
        vname = self._series(md_snapshot, "venue", idx, VENUE_AUCTION) if venue is None else pd.Series(venue, index=idx)
        d = date if date is not None else md_snapshot.get("date")

        impact_bps = pd.Series(
            [float(self.impact.cost_bps(q, v, s, b=bb)) if q != 0 else 0.0
             for q, v, s, bb in zip(x.values, vol.values, sigma.values, b.values)], index=idx)
        # auction fills clear at one price: no spread crossing
        spread_bps = hs_bps.where(vname != VENUE_AUCTION, 0.0).where(x != 0, 0.0)
        explicit_bps = pd.Series(
            [self.fees.explicit_bps(q, p, "buy" if q > 0 else "sell", d) if (q != 0 and p > 0) else 0.0
             for q, p in zip(x.values, price.values)], index=idx)
        total_bps = spread_bps + impact_bps + explicit_bps
        aq = x.abs()
        out = pd.DataFrame({
            "trade_usd": x, "venue": vname, "spread_bps": spread_bps, "impact_bps": impact_bps,
            "explicit_bps": explicit_bps, "total_bps": total_bps,
            "spread_usd": spread_bps * aq / 1e4, "impact_usd": impact_bps * aq / 1e4,
            "explicit_usd": explicit_bps * aq / 1e4, "total_usd": total_bps * aq / 1e4,
        })
        # aliases consumed by qts.backtest.simulator (spread/impact/fees USD)
        out["fees_usd"] = out["explicit_usd"]
        if w_post is not None and "nav" in md_snapshot:
            bf = borrow_fee_vector(idx, md_snapshot.get("borrow_fee"), self.cfg)
            out["borrow_usd"] = holding_cost_breakdown(w_post.reindex(idx).fillna(0.0), bf, self.cfg) * float(md_snapshot["nav"])
        return out

    def holding_cost_usd(self, w_post: pd.Series, nav: float, snapshot: Dict) -> pd.Series:
        """One day of borrow cost in USD per ticker (shorts only); simulator hook."""
        idx = w_post.index
        bf = borrow_fee_vector(idx, snapshot.get("borrow_fee"), self.cfg)
        return holding_cost_breakdown(w_post.fillna(0.0), bf, self.cfg) * float(nav)

    def total_cost_usd(self, trades_usd: pd.Series, md_snapshot: Dict, **kw) -> float:
        return float(self.estimate(trades_usd, md_snapshot, **kw)["total_usd"].sum())

    # ------------------------------------------------------------ convex
    def cvx_terms(self, x, w_post, nav: float, snapshot: Dict, venue: Optional[str] = None,
                  tickers: Optional[pd.Index] = None, include_explicit: bool = True):
        """``(trade_cost_expr, holding_cost_expr)`` as fractions of NAV, for the optimizer.

        ``x`` and ``w_post`` are cvxpy expressions over ``tickers`` (defaults to
        ``snapshot['adv_usd'].index``).  Trade cost uses the per-name venue in
        the snapshot (auction -> b=1.5 and V = auction share x ADV) unless
        ``venue`` overrides it.  Because cvxpy needs one coefficient vector, the
        3/2-term coefficient is ``b_i*sigma_i/sqrt(V_i/nav)`` per name.
        """
        import cvxpy as cp

        idx = tickers if tickers is not None else snapshot["adv_usd"].index
        sigma = self._series(snapshot, "sigma_daily", idx, np.nan).astype(float).fillna(0.0).values
        hs = self._series(snapshot, "half_spread", idx, self.cfg.default_half_spread_bps / 1e4).astype(float).values
        price = self._series(snapshot, "price", idx, np.nan).astype(float)
        vol, b = self._venue_volume(snapshot, idx, venue)
        vname = self._series(snapshot, "venue", idx, VENUE_AUCTION) if venue is None else pd.Series(venue, index=idx)
        hs = np.where(vname.values == VENUE_AUCTION, 0.0, hs)
        vol = np.maximum(vol.fillna(vol.median() if vol.notna().any() else 1.0).values, 1e-9)
        coef = b.values * sigma / np.sqrt(vol / float(nav))
        ax = cp.abs(x)
        trade = cp.sum(cp.multiply(hs, ax)) + cp.sum(cp.multiply(coef, cp.power(ax, 1.5)))
        if include_explicit:
            d = snapshot.get("date")
            fb = np.array([self.fees.marginal_fraction(p, "buy", d) if p > 0 else 0.0 for p in price.values])
            fs = np.array([self.fees.marginal_fraction(p, "sell", d) if p > 0 else 0.0 for p in price.values])
            trade = trade + cp.sum(cp.multiply(fb, cp.pos(x))) + cp.sum(cp.multiply(fs, cp.neg(x)))
        bf = borrow_fee_vector(idx, snapshot.get("borrow_fee"), self.cfg).values
        hold = cvx_holding_expression(w_post, bf)
        return trade, hold

    def participation_cap(self, snapshot: Dict, nav: float, max_part: Optional[float] = None) -> pd.Series:
        return self.impact.participation_cap(snapshot["adv_usd"], nav, max_part)
