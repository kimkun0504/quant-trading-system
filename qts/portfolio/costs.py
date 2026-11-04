"""Cost interface used by the optimizer and the simulator, plus a built-in
stand-in model so both run without ``qts.tcost``.

Protocol
--------
``CostTerms`` is the only thing the portfolio and backtest layers know about a
cost model.  ``qts.tcost.TransactionCostModel`` plugs in by satisfying the
same two methods (structural typing; no inheritance required):

* ``cvx_terms(x, w_post, nav, snapshot, venue) -> (trade_cost, holding_cost)``
  Both are convex cvxpy expressions in **fraction of NAV**.  ``trade_cost`` is
  the one-off cost of the trade vector ``x``; ``holding_cost`` is the cost of
  holding ``w_post`` for **one day** (the optimizer multiplies it by the alpha
  horizon so all objective terms are in horizon units).
* ``estimate(trades_usd, snapshot) -> pd.DataFrame`` indexed by ticker with
  columns ``spread``, ``impact``, ``fees`` in USD (the simulator also accepts a
  ``total`` column or a plain Series and books it as ``impact``).

``snapshot`` is ``dict[str, pd.Series]`` indexed by ticker with keys
``adv_usd``, ``sigma_daily``, ``half_spread`` (fraction), ``price`` and
``borrow_fee`` (annual fraction); extra keys written by
``qts.tcost`` (``half_spread_bps``, ``adv_shares``, ``venue``, scalar ``date``)
are passed through untouched.  ``snapshot_from_market`` builds one from a
``MarketData`` as of a date using only data ``<= date``.

One cost model, not two
-----------------------
``SimpleCostModel`` **delegates** to ``qts.tcost.TransactionCostModel`` when
that package is importable (lazy import, so ``qts.portfolio`` never depends on
``qts.tcost`` at import time).  The optimizer's convex terms, the simulator's
point estimates and the snapshot builder therefore price the *same* model
whichever entry point a caller uses.  The native formulas below are only the
fallback when ``qts.tcost`` is absent, plus the ``zero=True`` free-trading
path used by tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import cvxpy as cp
import numpy as np
import pandas as pd

from qts.core.config import CostConfig
from qts.core.types import MarketData

Snapshot = Dict[str, Any]
SNAPSHOT_KEYS = ("adv_usd", "sigma_daily", "half_spread", "price", "borrow_fee")


@runtime_checkable
class CostTerms(Protocol):
    def cvx_terms(
        self,
        x: cp.Expression,
        w_post: cp.Expression,
        nav: float,
        snapshot: Snapshot,
        venue: str,
    ) -> Tuple[cp.Expression, cp.Expression]: ...

    def estimate(self, trades_usd: pd.Series, snapshot: Snapshot) -> pd.DataFrame: ...


def _tcost_model(cfg: CostConfig):
    """``qts.tcost.TransactionCostModel(cfg)`` or ``None`` when the package is
    not importable (lazy import: avoids an import cycle and keeps the
    portfolio layer usable on its own)."""
    try:
        from qts.tcost.model import TransactionCostModel  # local import by design
    except Exception:  # pragma: no cover - only without qts.tcost
        return None
    return TransactionCostModel(cfg)


def _fill(s: pd.Series, default: float) -> pd.Series:
    s = pd.Series(s, dtype=float)
    med = s.median()
    return s.fillna(med if np.isfinite(med) else default)


def _native_snapshot(md: MarketData, date: pd.Timestamp, adv_window: int, vol_window: int,
                     cfg: CostConfig) -> Snapshot:
    close = md.close.loc[:date]
    tickers = md.tickers
    dv = (close * md.volume.loc[:date]).tail(adv_window)
    adv = dv.mean().reindex(tickers)
    rets = close.pct_change(fill_method=None).tail(vol_window)
    sigma = rets.std(ddof=1).reindex(tickers)
    price = close.iloc[-1].reindex(tickers)
    if md.spread_bps is not None:
        hs = (md.spread_bps.loc[:date].iloc[-1] / 1e4).reindex(tickers)
        hs = hs.fillna(cfg.default_half_spread_bps / 1e4)
    else:
        hs = pd.Series(cfg.default_half_spread_bps / 1e4, index=tickers)
    if md.borrow_fee is not None:
        bf = md.borrow_fee.loc[:date].iloc[-1].reindex(tickers).fillna(cfg.default_borrow_fee_annual)
    else:
        bf = pd.Series(cfg.default_borrow_fee_annual, index=tickers)
    return {"adv_usd": adv, "sigma_daily": sigma, "half_spread": hs, "price": price, "borrow_fee": bf,
            "date": pd.Timestamp(date)}


def snapshot_from_market(
    md: MarketData,
    date: pd.Timestamp,
    adv_window: int = 63,
    vol_window: int = 63,
    cfg: Optional[CostConfig] = None,
) -> Snapshot:
    """Point-in-time liquidity snapshot (only data with timestamp <= date).

    Delegates to ``TransactionCostModel.snapshot_from_market`` when
    ``qts.tcost`` is importable (so the simulator and the optimizer see the
    same ``adv_usd`` / ``sigma`` / spread arithmetic and the extra keys it
    writes), then applies the portfolio-layer fill policy: missing ADV /
    sigma / price -> cross-sectional median, floors on each.  ``vol_window``
    different from ``adv_window`` recomputes ``sigma_daily`` over
    ``vol_window`` (tcost uses a single window)."""
    cfg = cfg or CostConfig()
    date = pd.Timestamp(date)
    tcm = _tcost_model(cfg)
    if tcm is not None:
        snap: Snapshot = dict(tcm.snapshot_from_market(md, date, cfg_adv_window=adv_window))
        if vol_window != adv_window:
            rets = md.close.loc[:date].pct_change(fill_method=None).tail(vol_window)
            snap["sigma_daily"] = rets.std(ddof=1).reindex(md.tickers)
    else:
        snap = _native_snapshot(md, date, adv_window, vol_window, cfg)
    tickers = md.tickers
    snap["adv_usd"] = _fill(snap["adv_usd"].reindex(tickers), 1e6).clip(lower=1e3)
    snap["sigma_daily"] = _fill(snap["sigma_daily"].reindex(tickers), 0.02).clip(lower=1e-4)
    snap["price"] = _fill(snap["price"].reindex(tickers), 100.0).clip(lower=0.01)
    snap["half_spread"] = pd.Series(snap["half_spread"], dtype=float).reindex(tickers) \
        .fillna(cfg.default_half_spread_bps / 1e4).clip(lower=0.0)
    if "half_spread_bps" in snap:
        snap["half_spread_bps"] = snap["half_spread"] * 1e4
    snap["borrow_fee"] = pd.Series(snap["borrow_fee"], dtype=float).reindex(tickers) \
        .fillna(cfg.default_borrow_fee_annual).clip(lower=0.0)
    for k in ("adv_usd", "sigma_daily", "price"):
        snap[k] = snap[k].astype(float)
    return snap


def align_snapshot(snapshot: Snapshot, tickers: pd.Index, cfg: Optional[CostConfig] = None) -> Snapshot:
    """Reindex every snapshot series to ``tickers`` with sensible fills.

    The five core keys are filled with the cross-sectional median (or a
    default); any other per-ticker Series (``venue``, ``half_spread_bps``,
    ``adv_shares``...) is reindexed and left for the cost model's own defaults,
    and scalars (``date``, ``nav``, ``auction_share``) pass through."""
    cfg = cfg or CostConfig()
    defaults = {
        "adv_usd": 1e6, "sigma_daily": 0.02, "half_spread": cfg.default_half_spread_bps / 1e4,
        "price": 100.0, "borrow_fee": cfg.default_borrow_fee_annual,
    }
    out: Snapshot = {}
    for k in SNAPSHOT_KEYS:
        s = snapshot.get(k)
        if s is None:
            out[k] = pd.Series(defaults[k], index=tickers, dtype=float)
            continue
        s = pd.Series(s).reindex(tickers).astype(float)
        med = s.median()
        out[k] = s.fillna(med if np.isfinite(med) else defaults[k])
    out["adv_usd"] = out["adv_usd"].clip(lower=1e3)
    out["sigma_daily"] = out["sigma_daily"].clip(lower=1e-4)
    out["price"] = out["price"].clip(lower=0.01)
    for k, v in snapshot.items():
        if k in SNAPSHOT_KEYS:
            continue
        out[k] = v.reindex(tickers) if isinstance(v, pd.Series) else v
    # keep the bps spread consistent with the (possibly filled) fraction
    if "half_spread_bps" in out:
        out["half_spread_bps"] = out["half_spread"] * 1e4
    return out


@dataclass
class SimpleCostModel:
    """Cost model used by tests and stand-alone portfolio work.

    When ``qts.tcost`` is importable every method delegates to
    ``TransactionCostModel(cfg)`` (same convex terms, same point estimates,
    same auction treatment), so the optimizer and the simulator never price
    different models.  Without it the native stand-in applies: half-spread +
    explicit fees (linear), square-root impact in integrated form
    ``b * sigma * |x|^1.5 / sqrt(ADV / nav)`` and a linear borrow holding cost
    on shorts; all quantities are fractions of NAV and ``venue`` picks the
    impact prefactor.  ``zero=True`` is a free-trading model for tests and
    ``delegate=False`` forces the native formulas.
    """

    cfg: CostConfig = None  # type: ignore[assignment]
    zero: bool = False
    delegate: bool = True
    _tcm: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.cfg is None:
            self.cfg = CostConfig()
        if self.delegate and not self.zero:
            self._tcm = _tcost_model(self.cfg)

    @property
    def delegated(self) -> bool:
        return self._tcm is not None

    # ----------------------------------------------------------- coefficients
    def _b(self, venue: str) -> float:
        return self.cfg.impact_b_auction if venue == "auction" else self.cfg.impact_b_continuous

    def fee_fraction(self, price: pd.Series, side: str = "both") -> pd.Series:
        """Explicit fees per dollar traded.  ``side`` in {"buy", "sell", "both"}
        (``both`` = average of buy and sell, used inside the convex term)."""
        c = self.cfg
        per_share = c.commission_per_share + c.clearing_per_share + c.cat_fee_per_share
        buy = per_share / price
        sell = buy + c.sec_fee_per_dollar_sold + c.finra_taf_per_share_sold / price
        buy = buy.clip(upper=c.commission_max_pct_value)
        sell = sell.clip(upper=c.commission_max_pct_value + c.sec_fee_per_dollar_sold)
        if side == "buy":
            return buy
        if side == "sell":
            return sell
        return 0.5 * (buy + sell)

    def linear_coef(self, snapshot: Snapshot, venue: str) -> pd.Series:
        return snapshot["half_spread"] + self.fee_fraction(snapshot["price"])

    def impact_coef(self, snapshot: Snapshot, nav: float, venue: str) -> pd.Series:
        """Coefficient ``k_i`` so that impact_i = k_i * |x_i|^1.5 (fraction of NAV)."""
        return self._b(venue) * snapshot["sigma_daily"] / np.sqrt(snapshot["adv_usd"] / max(nav, 1.0))

    @staticmethod
    def _with_bps(snap: Snapshot) -> Snapshot:
        """tcost reads ``half_spread_bps``; derive it from ``half_spread`` if absent."""
        if "half_spread_bps" not in snap and "half_spread" in snap:
            snap = dict(snap)
            snap["half_spread_bps"] = pd.Series(snap["half_spread"], dtype=float) * 1e4
        return snap

    # ------------------------------------------------------------------ convex
    def cvx_terms(self, x, w_post, nav, snapshot, venue="auction"):
        if self.zero:
            z = cp.Constant(0.0)
            return z, z
        tick = snapshot["adv_usd"].index
        snap = align_snapshot(snapshot, tick, self.cfg)
        if self._tcm is not None:
            return self._tcm.cvx_terms(x, w_post, nav, self._with_bps(snap), venue=venue, tickers=tick)
        lin = self.linear_coef(snap, venue).values
        k = self.impact_coef(snap, nav, venue).values
        ax = cp.abs(x)
        trade = lin @ ax + k @ cp.power(ax, self.cfg.impact_exponent)
        hold = (snap["borrow_fee"].values / 252.0) @ cp.neg(w_post)
        return trade, hold

    # ------------------------------------------------------------- estimates
    def estimate(self, trades_usd: pd.Series, snapshot: Snapshot, venue: str = "auction",
                 nav: Optional[float] = None) -> pd.DataFrame:
        """Realised (expected) cost of executing ``trades_usd`` in USD per ticker
        (columns ``spread``, ``impact``, ``fees``)."""
        t = trades_usd.index
        if self.zero:
            z = pd.Series(0.0, index=t)
            return pd.DataFrame({"spread": z, "impact": z, "fees": z})
        snap = align_snapshot(snapshot, t, self.cfg)
        if self._tcm is not None:
            est = self._tcm.estimate(trades_usd.astype(float), self._with_bps(snap), venue=venue)
            return pd.DataFrame({"spread": est["spread_usd"].astype(float),
                                 "impact": est["impact_usd"].astype(float),
                                 "fees": est["explicit_usd"].astype(float)}, index=t)
        q = trades_usd.abs().astype(float)
        spread = q * snap["half_spread"]
        part = q / snap["adv_usd"]
        impact = q * self._b(venue) * snap["sigma_daily"] * np.sqrt(part)
        fees = np.where(trades_usd.values < 0,
                        q * self.fee_fraction(snap["price"], "sell"),
                        q * self.fee_fraction(snap["price"], "buy"))
        fees = pd.Series(fees, index=t)
        # per-order minimum commission
        n_orders = (q > 0).astype(float)
        fees = np.maximum(fees, n_orders * self.cfg.commission_min_per_order) * (q > 0)
        return pd.DataFrame({"spread": spread, "impact": impact, "fees": fees})

    def holding_cost_usd(self, w_post: pd.Series, nav: float, snapshot: Snapshot) -> pd.Series:
        """One day of borrow cost on short positions, USD per ticker."""
        if self.zero:
            return pd.Series(0.0, index=w_post.index)
        snap = align_snapshot(snapshot, w_post.index, self.cfg)
        if self._tcm is not None:
            return pd.Series(self._tcm.holding_cost_usd(w_post, nav, snap), dtype=float).reindex(w_post.index)
        return (-w_post).clip(lower=0.0) * nav * snap["borrow_fee"] / 252.0
