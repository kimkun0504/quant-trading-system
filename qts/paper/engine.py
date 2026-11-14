"""The paper book's daily step: marks, fills, corporate events, targets, orders.

Timeline (``L = execution.signal_lag_days = 1``)
-------------------------------------------------
The job for trading day ``t`` runs after the vendor has published day ``t``'s
prices (07:05 ET on ``t+1``) and sees data ``<= t`` only::

        close t-1                 close t                       close t+1
    ------|-------------------------|-----------------------------|------>
          | orders O(t-1) generated  | job "t" (next morning):     |
          | from the signal at t-1   |  1. mark t-1 -> t            |
          |                          |  2. fill O(t-1) at close t   |
          |                          |  3. signal at t -> O(t)      | O(t) filled by job "t+1"

* **Mark-to-market** ``t-1 -> t`` exactly as ``DailyBarSimulator``: each position earns
  the day's total return from the *adjusted* close (``close_t / last close - 1``; a
  halted name earns 0 and the whole halt-period move on resumption; data-error closes
  of ``qts.core.returns.flag_bad_prices`` are halts), a missing close is a halt or a
  delisting by the vendor date or the causal gap rule (``alpha.delisting_gap_days``),
  a delisting books its reason's return on the stock and is liquidated at spread +
  fees on the previous day's liquidity (continuous venue), and the previous day's
  borrow accrual is charged.
* **Prices and corporate events.**  Positions are valued at the *as-traded* close
  (``close_raw``) times an implied share count, and the day's P&L is the total
  return from the adjusted pair ``(t-1, t)`` read from the **same** snapshot.  The
  implied share count is ``value / close_raw``: it is multiplied by ``k`` at a ``k:1``
  split and grows by the yield at a dividend (the distribution is reinvested at the
  ex-date close, the backtest's total-return convention).  So the NAV, the weights
  and the P&L are the simulator's, Sharadar's restatement of the whole adjusted
  history at every dividend cannot move them (only the ratio of two closes of one
  snapshot is used), and splits need no corporate-action table (the event factor
  ``(1 + r_adj) / (1 + r_raw)`` is recorded as a split when it is >= 1.25 or <= 0.8,
  as a distribution otherwise).  A real account receives the dividend as cash on the
  pay date: same NAV, weight lower by the yield until the next rebalance.
* **Fills** at close ``t``, priced with the cost model on day ``t``'s liquidity at the
  routed venue (``TransactionCostModel.estimate`` with the simulator's multi-day
  impact streak), and one day of borrow on the post-fill shorts is accrued for the
  next mark (``holding_cost_usd``).  Costs are taken from cash at the fill, so the
  reported NAV is after the day's trading costs; ``nav_pre`` (after the mark, before
  the day's fills) is the simulator's NAV of that date.
* **Targets** for close ``t+1`` from the day-``t`` signal: on a rebalance day
  (``portfolio.rebalance_every_days`` = 5 execution dates, counted from the book's
  first execution date) the optimizer is called exactly as the simulator's target
  step calls it (same alpha/risk/eligibility/no-trade/forced-exit/snapshot inputs);
  on the other days nothing is traded except names that left the universe, which
  keep unwinding on their linear ``exit_ineligible_days`` schedule.

Two sizing conventions (``BookHeader.sizing``)
----------------------------------------------
``moc`` (default; what live trading does).  Orders are whole shares, sized at the
job's close ``t``: ``round_toward_zero((w_target - w_now) * NAV_t / close_raw_t)``
(a full exit sells the exact holding), filled at close ``t+1`` whatever the price
did in between.  An order for a name with no close on the fill day expires
unfilled (an MOC order is a day order; the next rebalance re-targets the name);
an order for a name that splits before the fill is scaled by the split ratio; an
order for a name that delists is cancelled.  The optimizer sees the book as it
stands at close ``t``.

``replica`` (the backtest's convention; the pipeline check).  The target for close
``t`` is solved with the book marked *at* close ``t`` and filled there in fractional
shares, from the signal of ``t-1``: the job reproduces ``DailyBarSimulator`` day by
day (same weights and NAV within solver tolerance).  It cannot be traded live
(the order size depends on the fill's own close); it exists so a backfill can be
reconciled with a batch backtest exactly.

The difference between the two is the sizing drift of one day (the book moves
with the market between the close that sizes an order and the close that fills it)
and whole-share rounding; ``tests/test_paper.py`` measures it.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qts.backtest.cost_adapter import cost_frame
from qts.core.config import SystemConfig
from qts.core.costs import VENUE_CONTINUOUS, LiquiditySnapshot
from qts.core.delisting import cost_position_returns, reasons_by_name, returns_by_name
from qts.core.returns import flag_bad_prices
from qts.core.types import MarketData, RiskModelOutput
from qts.data.universe import apply_mask, build_universe_mask
from qts.paper.book import PaperError, iso
from qts.paper.data import Calendar
from qts.paper.serving import SignalFit, SignalProvider
from qts.portfolio.optimizer import OK_STATUSES, SinglePeriodOptimizer
from qts.tcost import TransactionCostModel

log = logging.getLogger(__name__)

_TRADE_EPS = 1e-10  # |trade weight| below this is solver noise (simulator's _TRADE_EPS)
_HELD_EPS = 1e-12  # |weight| above this counts as held for forced exits (simulator's _HELD_EPS)
#: event factor (1 + r_adj) / (1 + r_raw) outside [SPLIT_DOWN, SPLIT_UP] is a split
SPLIT_UP = 1.25
SPLIT_DOWN = 0.8
#: event factors closer to 1 than this are no event (float noise)
EVENT_EPS = 1e-6
#: a held name whose as-traded close has not changed for this many sessions is flagged stale
STALE_DAYS = 5
SIZINGS = ("moc", "replica")


# --------------------------------------------------------------------------- state
@dataclass
class BookState:
    """Everything carried from one trading day to the next (JSON-serialisable;
    stored in each journal record, so the journal alone restores the book)."""

    date: str | None  #: last processed job day (None before the first)
    nav: float  #: NAV at the close after the day's fills and their costs
    nav_pre: float  #: NAV after the mark, before the day's fills (the simulator's NAV)
    cash: float
    shares: dict[str, float] = field(default_factory=dict)  #: implied as-traded shares
    px: dict[str, float] = field(default_factory=dict)  #: as-traded valuation price
    px_date: dict[str, str] = field(default_factory=dict)  #: date of that price
    stale: dict[str, int] = field(default_factory=dict)  #: sessions with an unchanged price
    pending: list[dict[str, Any]] = field(default_factory=list)  #: orders for the next close (moc)
    pending_borrow: float = 0.0  #: borrow accrued on the post-fill book, charged at the next mark
    streak: dict[str, list[float]] = field(default_factory=dict)  #: [days, usd, sign] per name
    forced: dict[str, dict[str, float]] = field(default_factory=dict)  #: forced exits {w0, k}
    halts_open: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_solve: str | None = None  #: execution date of the last optimizer solve
    last_risk_date: str | None = None  #: signal date of the persisted risk output

    @classmethod
    def initial(cls, nav0: float) -> BookState:
        return cls(date=None, nav=float(nav0), nav_pre=float(nav0), cash=float(nav0))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BookState:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in known})

    def values(self) -> dict[str, float]:
        return {k: float(self.shares[k]) * float(self.px[k]) for k in self.shares}


@dataclass
class DayResult:
    """Output of :meth:`PaperEngine.run_day`."""

    state: BookState
    record: dict[str, Any]
    positions: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    last_risk: RiskModelOutput | None
    solve: dict[str, Any] | None


@dataclass(frozen=True)
class UniverseView:
    """The market as the job sees it on one day: data through ``t`` with the PIT
    universe mask attached (``apply_mask``), the cleaned close and helpers."""

    md: MarketData  #: md_u (eligibility attached)
    close_clean: pd.DataFrame  #: adjusted close with data-error prints blanked
    t: pd.Timestamp


def universe_view(md: MarketData, cfg: SystemConfig, fit: SignalFit | None = None) -> UniverseView:
    """``apply_mask(md, build_universe_mask(md))`` (reused from ``fit`` when it was
    fitted on the same data) plus the simulator's cleaned close."""
    t = pd.Timestamp(md.dates[-1])
    if fit is not None and fit.asof == t:
        md_u = fit.md_u
    else:
        md_u = apply_mask(md, build_universe_mask(md, cfg.universe))
    close = md_u.close
    if cfg.risk.clean_returns:
        close, _ = flag_bad_prices(
            md_u.close, cap_up=float(cfg.risk.return_abs_cap_up), cap_down=float(cfg.risk.return_abs_cap_down)
        )
    return UniverseView(md=md_u, close_clean=close, t=t)


def _optimizer_snapshot(snap_sig: Any, auction_share: float) -> Any:
    """Signal-date liquidity with the execution date's auction share (the simulator's
    ``_optimizer_snapshot``)."""
    if isinstance(snap_sig, LiquiditySnapshot):
        if snap_sig.auction_share == float(auction_share):
            return snap_sig
        return dataclasses.replace(snap_sig, auction_share=float(auction_share))
    out = dict(snap_sig)
    out["auction_share"] = auction_share
    return out


def _round_toward_zero(q: np.ndarray) -> np.ndarray:
    return np.trunc(np.round(q, 9))


# --------------------------------------------------------------------------- engine
class PaperEngine:
    """Runs one trading day of a paper book (module docstring)."""

    def __init__(
        self,
        cfg: SystemConfig,
        *,
        sizing: str,
        first_execution: pd.Timestamp,
        calendar: Calendar,
        venue: str = "routed",
        cost_model: TransactionCostModel | None = None,
    ):
        if sizing not in SIZINGS:
            raise PaperError(f"unknown sizing {sizing!r} (moc | replica)")
        self.cfg = cfg
        self.sizing = sizing
        self.E = pd.Timestamp(first_execution)
        self.cal = calendar
        self.venue = venue
        self.cost = cost_model or TransactionCostModel(cfg.cost, cfg.execution)
        self.optimizer = SinglePeriodOptimizer(
            cfg.portfolio, self.cost, horizon_days=cfg.alpha.horizon_days, exec_cfg=cfg.execution
        )
        self.k = max(int(cfg.portfolio.rebalance_every_days), 1)
        self.lag = int(getattr(cfg.execution, "signal_lag_days", 1) or 0)
        if self.lag != 1:
            raise PaperError(
                f"the paper book assumes execution.signal_lag_days = 1 (orders from the close of t fill at "
                f"the close of t+1); the config has {self.lag}"
            )
        self.exit_days = max(int(getattr(cfg.portfolio, "exit_ineligible_days", 5)), 1)
        self.gap_days = int(getattr(cfg.alpha, "delisting_gap_days", 63))
        from qts.core.roles import RISK_ONLY_STYLES

        self.neutral_styles = [s for s in cfg.risk.styles if s in RISK_ONLY_STYLES]

    # ------------------------------------------------------------------ schedule
    def exec_index(self, d: pd.Timestamp) -> int:
        """Position of execution date ``d`` among the sessions from the first execution date."""
        if d < self.E:
            return -1
        return len(self.cal.sessions(self.E, d)) - 1

    def is_rebalance(self, exec_date: pd.Timestamp) -> bool:
        i = self.exec_index(exec_date)
        return i >= 0 and i % self.k == 0

    def decision_dates(self, t: pd.Timestamp) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """``(exec_date, signal_date)`` of the decision the job for day ``t`` makes:
        ``moc``: the order for the next session from the signal of ``t``;
        ``replica``: the trade at ``t`` from the signal of the previous session."""
        if self.sizing == "moc":
            return self.cal.next_session(t), t
        if t < self.E:
            return None, None
        prev = self.cal.sessions(t - pd.Timedelta(days=15), t)
        prev = prev[prev < t]
        return t, (pd.Timestamp(prev[-1]) if len(prev) else None)

    # ------------------------------------------------------------------ day
    def run_day(
        self,
        state: BookState,
        t: pd.Timestamp,
        md_t: MarketData,
        signals: SignalProvider,
        *,
        last_risk: RiskModelOutput | None = None,
        halted: bool = False,
        expected_distributions: Mapping[str, float] | None = None,
    ) -> DayResult:
        """Process job day ``t`` (``md_t``: data through ``t``)."""
        t = pd.Timestamp(t)
        if len(md_t.dates) == 0 or md_t.dates[-1] != t:
            raise PaperError(f"data for the job day {iso(t)} must end on that day")
        if state.date is not None and pd.Timestamp(state.date) >= t:
            raise PaperError(f"day {iso(t)} is not after the book's last day {state.date}")
        cfg = self.cfg
        exec_date, sig_date = self.decision_dates(t)
        solve_today = exec_date is not None and sig_date is not None and self.is_rebalance(exec_date)
        fit: SignalFit | None = None
        if solve_today and not halted and self.sizing == "moc":
            fit = signals.fit_for(t)  # the moc decision's signal is today's; reuse its universe
        view = universe_view(md_t, cfg, fit)
        md_u = view.md
        warnings: list[str] = []

        held_names = [k for k, v in state.shares.items() if v != 0]
        missing = [k for k in held_names if k not in md_u.tickers]
        pending_names = [str(o["ticker"]) for o in state.pending]
        extra = list(dict.fromkeys(missing + [p for p in pending_names if p not in md_u.tickers]))
        tickers = md_u.tickers.append(pd.Index(extra)) if extra else md_u.tickers
        if missing:
            warnings.append(
                f"{len(missing)} held name(s) missing from the store: {missing[:10]} (frozen at the last price)"
            )
        N = len(tickers)
        ix = {str(k): i for i, k in enumerate(tickers)}
        dates = md_u.dates
        pt = int(dates.searchsorted(t, side="left"))  # t is a date of md_u (checked above)
        close = view.close_clean.reindex(columns=tickers)
        raw_all = md_u.trade_price().reindex(columns=tickers)
        c_t = close.iloc[pt].to_numpy(dtype=float)
        raw_t = raw_all.iloc[pt].to_numpy(dtype=float)
        alive = np.isfinite(c_t)

        rec: dict[str, Any] = {"date": iso(t), "sizing": self.sizing}
        events: dict[str, list[dict[str, Any]]] = {
            "delistings": [],
            "halts_opened": [],
            "halts_closed": [],
            "splits": [],
            "distributions": [],
            "forced_exits": [],
            "unfilled": [],
        }

        # ------------------------------------------------------------ 1. mark t_prev -> t
        new = BookState(
            date=iso(t),
            nav=state.nav,
            nav_pre=state.nav,
            cash=state.cash,
            shares=dict(state.shares),
            px=dict(state.px),
            px_date=dict(state.px_date),
            stale=dict(state.stale),
            pending=[],
            pending_borrow=0.0,
            streak={k: list(v) for k, v in state.streak.items()},
            forced={k: dict(v) for k, v in state.forced.items()},
            halts_open={k: dict(v) for k, v in state.halts_open.items()},
            last_solve=state.last_solve,
            last_risk_date=state.last_risk_date,
        )
        V = np.zeros(N)
        for k, v in state.values().items():
            V[ix[k]] = v
        r_used = np.zeros(N)
        pnl_positions = 0.0
        liq_cost = {"spread": 0.0, "fees": 0.0}
        liquidate = np.zeros(N, dtype=bool)
        borrow_charged = float(state.pending_borrow)
        delist_pnl = 0.0
        if state.date is not None:
            t_prev = pd.Timestamp(state.date)
            nav_prev_pre = float(state.nav_pre)
            held = V != 0
            r = self._returns(view, close, t_prev, pt)
            nan_r = ~np.isfinite(r)
            r[nan_r] = 0.0
            miss = held & nan_r
            if miss.any():
                liquidate = self._classify_missing(md_u, close, tickers, pt, miss, new, state)
                frozen = miss & ~liquidate
                for j in np.where(frozen)[0]:
                    tk = str(tickers[int(j)])
                    ep = new.halts_open.get(tk)
                    if ep is None:
                        ep = {"ticker": tk, "start": iso(t), "weight": float(V[j] / nav_prev_pre), "days": 0}
                        new.halts_open[tk] = ep
                        events["halts_opened"].append(dict(ep))
                    ep["days"] = int(ep["days"]) + 1
            for tk in list(new.halts_open):
                j = ix.get(tk)
                if j is None:
                    continue
                if liquidate[j]:
                    events["halts_closed"].append(
                        {**new.halts_open.pop(tk), "end": iso(t), "outcome": "delisted"}
                    )
                elif not nan_r[j] or not held[j]:
                    events["halts_closed"].append(
                        {**new.halts_open.pop(tk), "end": iso(t), "outcome": "resumed"}
                    )
            if liquidate.any():
                delist_pnl, liq_cost = self._liquidate(
                    md_u, tickers, liquidate, V, r, nav_prev_pre, t, t_prev, new, events
                )
            pnl_positions = float(V @ r)
            V_mark = V * (1.0 + r)
            proceeds = float(V_mark[liquidate].sum())
            V_mark[liquidate] = 0.0
            new.cash = float(state.cash + proceeds - borrow_charged - liq_cost["spread"] - liq_cost["fees"])
            r_used = r
            self._update_prices(tickers, V_mark, raw_t, alive, t, state, new, events, expected_distributions)
            for tk in [str(tickers[int(j)]) for j in np.where(liquidate)[0]]:
                for d in (new.shares, new.px, new.px_date, new.stale):
                    d.pop(tk, None)
        else:
            V_mark = V
        nav_pre = float(new.cash + V_mark.sum())
        new.nav_pre = nav_pre
        w_pre = V_mark / nav_pre if nav_pre > 0 else np.zeros(N)

        # ------------------------------------------------------------ 2. fills at close t
        snap_t = self.cost.snapshot(md_u, t)
        fills_x = np.zeros(N)  # trade, fraction of nav_pre
        fill_kind = np.full(N, "", dtype=object)
        solve_info: dict[str, Any] | None = None
        risk_out = last_risk
        if self.sizing == "replica":
            if exec_date is not None and sig_date is not None and not halted:
                w_target, solve_info, risk_out = self._decide(
                    view,
                    tickers,
                    exec_date,
                    sig_date,
                    w_pre,
                    nav_pre,
                    alive,
                    snap_t,
                    new,
                    signals,
                    events,
                    risk_out,
                    solve_today,
                )
                x = w_target - w_pre
                x[np.abs(x) < _TRADE_EPS] = 0.0
                fills_x = x
                fill_kind[x != 0] = "rebalance" if solve_today else "forced_exit"
            elif halted and exec_date is not None:
                warnings.append("STOP status: no trades (acknowledge with `qts paper ack`)")
        else:
            fills_x, fill_kind = self._fill_pending(
                state, tickers, ix, alive, raw_t, V_mark, nav_pre, r_used, liquidate, events
            )
        tc, borrow_next, x_final = self._cost_step(tickers, fills_x, w_pre, nav_pre, snap_t, new)
        fill_usd = x_final * nav_pre
        V_post = V_mark + fill_usd
        new.cash = float(new.cash - fill_usd.sum() - tc["total"])
        new.nav = float(new.cash + V_post.sum())
        new.pending_borrow = float(borrow_next)
        self._set_positions(tickers, V_post, raw_t, alive, t, new, fill_usd)
        if self.sizing == "replica" and solve_today and not halted and exec_date is not None:
            new.last_solve = iso(exec_date)

        # ------------------------------------------------------------ 3. decision for t+1 (moc)
        orders = _empty_orders()
        nav_end = new.nav
        w_post = V_post / nav_end if nav_end > 0 else np.zeros(N)
        if self.sizing == "moc" and exec_date is not None and sig_date is not None:
            if halted:
                warnings.append("STOP status: no orders generated (acknowledge with `qts paper ack`)")
            else:
                w_target, solve_info, risk_out = self._decide(
                    view,
                    tickers,
                    exec_date,
                    sig_date,
                    w_post,
                    nav_end,
                    alive,
                    snap_t,
                    new,
                    signals,
                    events,
                    risk_out,
                    solve_today,
                    fit=fit,
                )
                orders = self._make_orders(
                    tickers, w_target, w_post, nav_end, raw_t, new, exec_date, snap_t, solve_today
                )
                new.pending = [{str(k): v for k, v in rec.items()} for rec in orders.to_dict("records")]
                if solve_today:
                    new.last_solve = iso(exec_date)
        if solve_info is not None and solve_info.get("signal_date"):
            new.last_risk_date = solve_info["signal_date"]

        # ------------------------------------------------------------ 4. record
        exante_vol = float("nan")
        if risk_out is not None and np.abs(w_post).sum() > 0:
            wv = pd.Series(w_post, index=tickers)
            exante_vol = math.sqrt(max(risk_out.portfolio_variance(wv), 0.0) * cfg.trading_days_per_year)
        traded = float(np.abs(fill_usd).sum())
        n_alive_held = int(((V_post != 0) & alive).sum())
        stale_names = sorted(k for k, n in new.stale.items() if n >= STALE_DAYS and new.shares.get(k, 0) != 0)
        abs_v = {str(tickers[int(j)]): abs(float(V_post[j])) for j in np.flatnonzero(V_post)}
        missing_gross = sum(abs_v.get(k, 0.0) for k in missing) / nav_end if nav_end > 0 else 0.0
        stale_gross = sum(abs_v.get(k, 0.0) for k in stale_names) / nav_end if nav_end > 0 else 0.0
        if stale_names:
            warnings.append(
                f"{len(stale_names)} held name(s) with an unchanged close for >= {STALE_DAYS} sessions"
            )
        if solve_info is not None and not solve_info.get("ok", True):
            warnings.append(f"optimizer solve failed ({solve_info.get('status')}): book held")
        long_v = float(V_post[V_post > 0].sum())
        short_v = float(-V_post[V_post < 0].sum())
        rec.update(
            {
                "nav": new.nav,
                "nav_pre": nav_pre,
                "cash": new.cash,
                "gross": (long_v + short_v) / nav_end if nav_end > 0 else float("nan"),
                "net": (long_v - short_v) / nav_end if nav_end > 0 else float("nan"),
                "long_usd": long_v,
                "short_usd": short_v,
                "n_long": int((V_post > 0).sum()),
                "n_short": int((V_post < 0).sum()),
                "pnl": {
                    "positions": pnl_positions,
                    "delisting": delist_pnl,
                    "spread": tc["spread"],
                    "impact": tc["impact"],
                    "fees": tc["fees"],
                    "trading_cost": tc["spread"] + tc["impact"] + tc["fees"],
                    "borrow": borrow_charged,
                    "liquidation_cost": liq_cost["spread"] + liq_cost["fees"],
                    "total": pnl_positions
                    - borrow_charged
                    - liq_cost["spread"]
                    - liq_cost["fees"]
                    - (tc["spread"] + tc["impact"] + tc["fees"]),
                },
                "ret": (new.nav / state.nav - 1.0) if state.date is not None else 0.0,
                "ret_sim": (nav_pre / state.nav_pre - 1.0) if state.date is not None else 0.0,
                "traded_usd": traded,
                "turnover": traded / nav_pre if nav_pre > 0 else float("nan"),
                "n_fills": int((fill_usd != 0).sum()),
                "n_orders": len(orders),
                "orders_for": None if exec_date is None or self.sizing != "moc" else iso(exec_date),
                "exante_vol": exante_vol,
                "solve": solve_info,
                "rebalance_decision": bool(solve_today),
                "events": events,
                "warnings": warnings,
                "data_quality": {
                    "missing_held": missing,
                    "stale_held": stale_names,
                    "missing_gross": missing_gross,
                    "stale_gross": stale_gross,
                    "n_halted_held": len(new.halts_open),
                    "n_names": len(md_u.tickers),
                    "n_eligible": int(md_u.eligibility().iloc[pt].sum()),
                    "n_alive_held": n_alive_held,
                },
                "borrow_accrued_next": borrow_next,
            }
        )
        positions = positions_frame(new)
        fills = self._fills_frame(tickers, fill_usd, raw_t, fill_kind, tc, t)
        rec["fills"] = _columns(fills)
        rec["orders"] = _columns(orders)
        return DayResult(new, rec, positions, fills, orders, risk_out, solve_info)

    # ------------------------------------------------------------------ helpers: mark
    def _returns(self, view: UniverseView, close: pd.DataFrame, t_prev: pd.Timestamp, pt: int) -> np.ndarray:
        """Day-``t`` return from the last cleaned close in ``[first_execution, t_prev]``
        (the simulator's ``close / close.ffill().shift(1) - 1`` over the run's dates)."""
        dates = view.md.dates
        p0 = int(dates.searchsorted(self.E, side="left"))
        window = close.iloc[p0:pt]
        if len(window) == 0:
            return np.full(close.shape[1], np.nan)
        last = window.ffill().iloc[-1].to_numpy(dtype=float)
        c_t = close.iloc[pt].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = c_t / last - 1.0
        r[~np.isfinite(c_t)] = np.nan
        return r

    def _classify_missing(
        self,
        md_u: MarketData,
        close: pd.DataFrame,
        tickers: pd.Index,
        pt: int,
        miss: np.ndarray,
        new: BookState,
        state: BookState,
    ) -> np.ndarray:
        """Halt or delisting for held names without a close on ``t`` (the simulator's
        ``_classify_missing``): the vendor's delisting date when it covers the name,
        else the causal gap rule -- never a later price.  A name missing from the
        store is measured from the date of its last price in the book."""
        vendor = np.full(len(tickers), np.nan)
        vp = md_u.delisting_positions()
        n_md = len(md_u.tickers)
        vendor[:n_md] = vp
        last_pos = np.full(len(tickers), np.nan)
        cols = np.flatnonzero(miss[:n_md])
        if len(cols):  # last close position (<= t) of the names that need it only
            have = close.iloc[: pt + 1, cols].notna().to_numpy()[::-1]
            anyv = have.any(axis=0)
            last_pos[cols] = np.where(anyv, pt - have.argmax(axis=0), np.nan)
        dates = md_u.dates
        for j in range(n_md, len(tickers)):  # missing from the store: last price date in the book
            d = state.px_date.get(str(tickers[int(j)]))
            last_pos[j] = float(dates.searchsorted(pd.Timestamp(d), side="right") - 1) if d else np.nan
        covered = ~np.isnan(vendor)
        by_vendor = covered & (vendor < pt)
        by_gap = ~covered & np.isfinite(last_pos) & (pt - last_pos >= self.gap_days)
        return miss & (by_vendor | by_gap)

    def _liquidate(
        self,
        md_u: MarketData,
        tickers: pd.Index,
        liquidate: np.ndarray,
        V: np.ndarray,
        r: np.ndarray,
        nav_prev_pre: float,
        t: pd.Timestamp,
        t_prev: pd.Timestamp,
        new: BookState,
        events: dict[str, list[dict[str, Any]]],
    ) -> tuple[float, dict[str, float]]:
        """Book the reason's delisting return and price the forced liquidation at
        spread + fees on the previous day's liquidity (continuous venue, no impact),
        as ``DailyBarSimulator._liquidate_delisted``."""
        n_md = len(md_u.tickers)
        d_ret = np.full(len(tickers), float(self.cfg.cost.delisting_returns().get("unknown", -0.3)))
        reason = np.full(len(tickers), "unknown", dtype=object)
        d_ret[:n_md] = returns_by_name(md_u, self.cfg.cost.delisting_returns())
        reason[:n_md] = reasons_by_name(md_u)
        w = V / nav_prev_pre
        r[liquidate] = cost_position_returns(self.cfg.cost, d_ret[liquidate], w[liquidate])
        x_liq = np.zeros(len(tickers))
        x_liq[liquidate] = -w[liquidate] * (1.0 + r[liquidate])
        snap_prev = self.cost.snapshot(md_u, t_prev)
        idx_md = md_u.tickers
        x_md = pd.Series(x_liq[:n_md] * nav_prev_pre, index=idx_md)
        cf = cost_frame(self.cost.estimate(x_md, snap_prev, venue=VENUE_CONTINUOUS, streak=None), idx_md)
        cf["impact"] = 0.0
        spread = float(cf["spread"].sum())
        fees = float(cf["fees"].sum())
        pnl = 0.0
        for j in np.where(liquidate)[0]:
            tk = str(tickers[int(j)])
            c = float(cf.iloc[j].sum()) if j < n_md else 0.0
            pnl += float(V[j] * r[j])
            events["delistings"].append(
                {
                    "date": iso(t),
                    "ticker": tk,
                    "weight": float(w[j]),
                    "reason": str(reason[j]),
                    "side": "long" if w[j] > 0 else "short",
                    "stock_return": float(d_ret[j]),
                    "haircut_return": float(r[j]),
                    "pnl": float(w[j] * r[j]),
                    "pnl_usd": float(V[j] * r[j]),
                    "liquidation_usd": float(abs(x_liq[j]) * nav_prev_pre),
                    "cost_usd": c,
                }
            )
            new.streak.pop(tk, None)
            new.forced.pop(tk, None)
        return pnl, {"spread": spread, "fees": fees}

    def _update_prices(
        self,
        tickers: pd.Index,
        V_mark: np.ndarray,
        raw_t: np.ndarray,
        alive: np.ndarray,
        t: pd.Timestamp,
        state: BookState,
        new: BookState,
        events: dict[str, list[dict[str, Any]]],
        expected_distributions: Mapping[str, float] | None,
    ) -> None:
        """Implied as-traded share counts after the mark; split / distribution events
        from the factor ``(1 + r_adj) / (1 + r_raw)`` of held names."""
        exp = dict(expected_distributions or {})
        for j, tk in enumerate(tickers):
            tk = str(tk)
            if tk not in state.shares or state.shares[tk] == 0:
                continue
            px_prev = float(state.px[tk])
            if not (alive[j] and np.isfinite(raw_t[j]) and raw_t[j] > 0):
                continue
            v_prev = float(state.shares[tk]) * px_prev
            e = (V_mark[j] / v_prev) / (raw_t[j] / px_prev) if v_prev != 0 and px_prev > 0 else 1.0
            if np.isfinite(e) and abs(e - 1.0) > EVENT_EPS:
                if e >= SPLIT_UP or e <= SPLIT_DOWN:
                    events["splits"].append(
                        {"ticker": tk, "ratio": float(e), "shares_before": state.shares[tk]}
                    )
                else:
                    events["distributions"].append(
                        {"ticker": tk, "per_share": float((e - 1.0) * raw_t[j]), "factor": float(e)}
                    )
            elif tk in exp:
                events["distributions"].append(
                    {
                        "ticker": tk,
                        "per_share": 0.0,
                        "factor": 1.0,
                        "missing": True,
                        "note": f"corporate_actions lists a {exp[tk]:g} distribution on {iso(t)} but the "
                        "adjusted close does not reflect it (stale adjusted history: run a full sync)",
                    }
                )

    # ------------------------------------------------------------------ helpers: fills
    def _fill_pending(
        self,
        state: BookState,
        tickers: pd.Index,
        ix: dict[str, int],
        alive: np.ndarray,
        raw_t: np.ndarray,
        V_mark: np.ndarray,
        nav_pre: float,
        r: np.ndarray,
        liquidate: np.ndarray,
        events: dict[str, list[dict[str, Any]]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """``moc``: fill the orders generated by the previous job at today's
        as-traded close (module docstring for halts, splits, delistings)."""
        x = np.zeros(len(tickers))
        kind = np.full(len(tickers), "", dtype=object)
        if nav_pre <= 0:
            return x, kind
        for o in state.pending:
            tk = str(o["ticker"])
            j = ix[tk]
            q = float(o["shares"])
            if liquidate[j]:
                events["unfilled"].append({"ticker": tk, "shares": q, "reason": "delisted"})
                continue
            if not (alive[j] and np.isfinite(raw_t[j]) and raw_t[j] > 0):
                events["unfilled"].append(
                    {"ticker": tk, "shares": q, "reason": "no close (halted / missing)"}
                )
                continue
            if bool(o.get("close_out")):
                usd = -V_mark[j]
            else:
                px_dec = float(o["px"])
                e = (1.0 + r[j]) / (raw_t[j] / px_dec) if px_dec > 0 else 1.0
                if np.isfinite(e) and (e >= SPLIT_UP or e <= SPLIT_DOWN):
                    q = float(_round_toward_zero(np.array([q * e]))[0])
                usd = q * raw_t[j]
            x[j] += usd / nav_pre
            kind[j] = str(o.get("kind", "rebalance"))
        return x, kind

    def _cost_step(
        self,
        tickers: pd.Index,
        x: np.ndarray,
        w_pre: np.ndarray,
        nav_pre: float,
        snap_t: Any,
        new: BookState,
    ) -> tuple[dict[str, float], float, np.ndarray]:
        """The simulator's cost step: same-direction streak, ``estimate`` at the venue
        with ``prior_streak_usd``, one day of borrow on the post-trade book."""
        x = np.where(np.abs(x) < _TRADE_EPS, 0.0, x)
        N = len(tickers)
        streak = np.zeros(N)
        s_usd = np.zeros(N)
        last_sign = np.zeros(N)
        pos = {str(k): i for i, k in enumerate(tickers)}
        for tk, (k, usd, sg) in new.streak.items():
            j = pos.get(tk)
            if j is not None:
                streak[j], s_usd[j], last_sign[j] = k, usd, sg
        sgn = np.sign(x)
        same = (sgn != 0) & (sgn == last_sign)
        streak = np.where(sgn != 0, np.where(same, streak + 1.0, 1.0), 0.0)
        prior = np.where(same, s_usd, 0.0)
        s_usd = np.where(sgn != 0, prior + np.abs(x) * nav_pre, 0.0)
        new.streak = {
            str(tickers[int(j)]): [float(streak[j]), float(s_usd[j]), float(sgn[j])]
            for j in np.flatnonzero(sgn)
        }
        w_post = w_pre + x
        n_md = len(snap_t.tickers) if isinstance(snap_t, LiquiditySnapshot) else N
        idx = tickers[:n_md]
        if np.any(x[n_md:] != 0):
            raise PaperError(
                "a trade in a name missing from the store"
            )  # pragma: no cover - guarded by alive
        est = self.cost.estimate(
            pd.Series(x[:n_md] * nav_pre, index=idx),
            snap_t,
            venue=self.venue,
            streak=pd.Series(streak[:n_md], index=idx),
            prior_streak_usd=pd.Series(prior[:n_md], index=idx),
        )
        cf = cost_frame(est, idx)
        borrow = self.cost.holding_cost_usd(pd.Series(w_post[:n_md], index=idx), nav_pre, snap_t)
        tc = {
            "spread": float(cf["spread"].sum()),
            "impact": float(cf["impact"].sum()),
            "fees": float(cf["fees"].sum()),
        }
        tc["total"] = tc["spread"] + tc["impact"] + tc["fees"]
        self._last_cost_frame = cf
        return tc, float(np.nansum(np.asarray(borrow, dtype=float))), x

    def _set_positions(
        self,
        tickers: pd.Index,
        V_post: np.ndarray,
        raw_t: np.ndarray,
        alive: np.ndarray,
        t: pd.Timestamp,
        new: BookState,
        fill_usd: np.ndarray,
    ) -> None:
        for j in np.flatnonzero((V_post != 0) | (fill_usd != 0)):
            tk = str(tickers[int(j)])
            if V_post[j] == 0:
                for d in (new.shares, new.px, new.px_date, new.stale):
                    d.pop(tk, None)
                continue
            if alive[j] and np.isfinite(raw_t[j]) and raw_t[j] > 0:
                px_old = new.px.get(tk)
                new.stale[tk] = new.stale.get(tk, 0) + 1 if px_old is not None and px_old == raw_t[j] else 0
                new.px[tk] = float(raw_t[j])
                new.px_date[tk] = iso(t)
            elif tk not in new.px:  # pragma: no cover - a new position always has a price
                raise PaperError(f"cannot value {tk}: no price")
            new.shares[tk] = float(V_post[j] / new.px[tk])

    # ------------------------------------------------------------------ helpers: decision
    def _universe_step(
        self,
        view: UniverseView,
        tickers: pd.Index,
        sig_date: pd.Timestamp,
        exec_date: pd.Timestamp,
        w_cur: np.ndarray,
        alive: np.ndarray,
        new: BookState,
        events: dict[str, list[dict[str, Any]]],
    ) -> tuple[np.ndarray, list[str], dict[str, float]]:
        """Eligibility as of the signal date and the forced-exit schedule
        (``DailyBarSimulator._universe_step``)."""
        md_u = view.md
        elig_panel = md_u.eligible
        if elig_panel is not None:
            row = elig_panel.reindex(columns=tickers).loc[sig_date].fillna(False).to_numpy(dtype=bool)
        else:  # pragma: no cover - apply_mask always sets it
            row = np.ones(len(tickers), dtype=bool)
        eligible = row & alive
        inel = alive & ~eligible
        held = np.abs(w_cur) > _HELD_EPS
        forced = new.forced
        for j in np.where(inel & held)[0]:
            tk = str(tickers[int(j)])
            if tk not in forced:
                forced[tk] = {"w0": float(abs(w_cur[j])), "k": 0.0}
                events["forced_exits"].append(
                    {"date": iso(exec_date), "ticker": tk, "weight": float(w_cur[j])}
                )
        pos = {str(k): i for i, k in enumerate(tickers)}
        for tk in list(forced):
            j = pos.get(tk)
            if j is None or eligible[j] or not held[j]:
                forced.pop(tk)
        caps: dict[str, float] = {}
        for tk, st in forced.items():
            st["k"] = float(st["k"]) + 1.0
            caps[tk] = max(float(st["w0"]) * (1.0 - float(st["k"]) / self.exit_days), 0.0)
        no_trade = [str(x) for x in tickers[~alive]] + [str(x) for x in tickers[inel & ~held]]
        return inel, no_trade, caps

    def _decide(
        self,
        view: UniverseView,
        tickers: pd.Index,
        exec_date: pd.Timestamp,
        sig_date: pd.Timestamp,
        w_cur: np.ndarray,
        nav: float,
        alive: np.ndarray,
        snap_t: Any,
        new: BookState,
        signals: SignalProvider,
        events: dict[str, list[dict[str, Any]]],
        last_risk: RiskModelOutput | None,
        solve_today: bool,
        fit: SignalFit | None = None,
    ) -> tuple[np.ndarray, dict[str, Any] | None, RiskModelOutput | None]:
        """Target weights for ``exec_date`` (``DailyBarSimulator._target_step``)."""
        inel, no_trade, caps = self._universe_step(
            view, tickers, sig_date, exec_date, w_cur, alive, new, events
        )
        w = w_cur.copy()
        if not solve_today:
            pos = {str(k): i for i, k in enumerate(tickers)}
            for tk, cap in caps.items():
                j = pos[tk]
                if alive[j] and abs(w[j]) > cap:
                    w[j] = np.sign(w[j]) * cap
            return w, None, last_risk
        if fit is None or fit.asof != sig_date:
            fit = signals.fit_for(sig_date)
        alpha = fit.alpha_as_of(sig_date)
        exp_ret = pd.Series(alpha.expected_return, dtype=float).reindex(tickers)
        exp_ret[~alive] = 0.0
        exp_ret[inel] = 0.0
        exp_ret = exp_ret.fillna(0.0)
        risk = fit.risk_as_of(sig_date, exp_ret)
        md_u = view.md
        if self.sizing == "replica":
            snap_sig = self.cost.snapshot(md_u, sig_date)
            share = snap_t.auction_share if isinstance(snap_t, LiquiditySnapshot) else snap_t["auction_share"]
        else:
            snap_sig = snap_t
            share = self.cost.impact.auction_share_for_date(exec_date)
        snapshot = _optimizer_snapshot(snap_sig, share)
        kw: dict[str, Any] = {}
        hl = getattr(alpha, "half_life_days", None)
        if hl is not None:
            kw["alpha_half_life_days"] = float(hl)
        n_md = len(md_u.tickers)
        idx = tickers[:n_md]
        res = self.optimizer.solve(
            alpha=exp_ret.iloc[:n_md],
            risk=risk,
            w_current=pd.Series(w_cur[:n_md], index=idx),
            nav=nav,
            snapshot=snapshot,
            venue=self.venue,
            neutral_styles=self.neutral_styles,
            no_trade=[x for x in no_trade if x in set(idx)],
            horizon_days=getattr(alpha, "horizon_days", None) or self.optimizer.horizon_days,
            force_to_zero=caps or None,
            **kw,
        )
        ok = res.status in OK_STATUSES
        w = np.array(res.w_target.reindex(tickers).fillna(0.0).to_numpy(dtype=float), copy=True)
        w[~alive] = w_cur[~alive]
        limit = float(getattr(self.cfg.validation, "max_exante_vol", 2.0))
        ref_w = pd.Series(w_cur, index=tickers) if np.abs(w_cur).sum() > _HELD_EPS else None
        incoming_vol = (
            math.sqrt(max(risk.portfolio_variance(ref_w), 0.0) * self.cfg.trading_days_per_year)
            if ref_w is not None
            else float("nan")
        )
        fold = fit.fold_for(sig_date)
        info = {
            "exec_date": iso(exec_date),
            "signal_date": iso(sig_date),
            "status": res.status,
            "ok": ok,
            "solver_errors": list(getattr(res, "solver_errors", []) or []),
            "risk_aversion": float(getattr(res, "risk_aversion", float("nan"))),
            "vol_target_binding": str(getattr(res, "vol_target_binding", "") or ""),
            "n_solves": int(getattr(res, "n_solves", 0) or 0),
            "expected_alpha": float(res.expected_alpha),
            "expected_risk": float(res.expected_risk),
            "expected_cost": float(res.expected_cost),
            "transfer_coefficient": float(res.transfer_coefficient),
            "n_small_trades_dropped": int(getattr(res, "n_small_trades_dropped", 0) or 0),
            "ic_estimate": float(alpha.ic_estimate),
            "half_life_days": float(alpha.half_life_days),
            "n_scored": len(alpha.score),
            "incoming_exante_vol": incoming_vol,
            "exante_vol_alert": bool(np.isfinite(incoming_vol) and incoming_vol > limit),
            "fit_asof": iso(fit.asof),
            "fold": fold,
            "fit_seconds": round(fit.seconds, 2),
            "folds_trained": list(fit.trained),
            "fit_warnings": list(fit.warnings),
        }
        if not ok:
            log.warning(
                "optimizer solve failed for %s: status=%s (holding the book)", iso(exec_date), res.status
            )
        return w, info, risk

    def _make_orders(
        self,
        tickers: pd.Index,
        w_target: np.ndarray,
        w_cur: np.ndarray,
        nav: float,
        raw_t: np.ndarray,
        new: BookState,
        exec_date: pd.Timestamp,
        snap_t: Any,
        solve_today: bool,
    ) -> pd.DataFrame:
        """``moc`` orders for ``exec_date``: whole shares toward zero at today's
        as-traded close; a target of zero sells the exact holding; each order with
        the cost model's estimate (today's liquidity, the fill day's auction share)."""
        dw = w_target - w_cur
        dw[np.abs(dw) < _TRADE_EPS] = 0.0
        rows = []
        for j in np.flatnonzero(dw):
            tk = str(tickers[int(j)])
            px = float(raw_t[j])
            if not (np.isfinite(px) and px > 0):
                continue
            close_out = bool(w_target[j] == 0.0 and w_cur[j] != 0.0)
            if close_out:
                q = -float(new.shares.get(tk, 0.0))
            else:
                q = float(_round_toward_zero(np.array([dw[j] * nav / px]))[0])
            if q == 0:
                continue
            rows.append(
                {
                    "ticker": tk,
                    "side": "BUY" if q > 0 else "SELL",
                    "shares": q,
                    "px": px,
                    "usd": q * px,
                    "target_w": float(w_target[j]),
                    "current_w": float(w_cur[j]),
                    "kind": "close_out" if close_out else ("rebalance" if solve_today else "forced_exit"),
                    "close_out": close_out,
                    "exec_date": iso(exec_date),
                }
            )
        if not rows:
            return _empty_orders()
        out = pd.DataFrame(rows)
        n_md = len(snap_t.tickers) if isinstance(snap_t, LiquiditySnapshot) else len(tickers)
        idx = tickers[:n_md]
        usd = out.set_index("ticker")["usd"].reindex(idx).fillna(0.0)
        streak = pd.Series(0.0, index=idx)
        prior = pd.Series(0.0, index=idx)
        for tk, (k, s_usd, sg) in new.streak.items():
            if tk in usd.index and np.sign(usd[tk]) == sg and sg != 0:
                streak[tk], prior[tk] = k + 1.0, s_usd
        streak[(usd != 0) & (streak == 0)] = 1.0
        snap = _optimizer_snapshot(snap_t, self.cost.impact.auction_share_for_date(exec_date))
        est = self.cost.estimate(usd, snap, venue=self.venue, streak=streak, prior_streak_usd=prior)
        tot = est.frame["total_usd"].reindex(out["ticker"]).to_numpy(dtype=float)
        out["est_cost_usd"] = tot
        with np.errstate(divide="ignore", invalid="ignore"):
            out["est_cost_bps"] = np.where(out["usd"] != 0, tot / out["usd"].abs() * 1e4, 0.0)
        return out

    # ------------------------------------------------------------------ helpers: views
    def _fills_frame(
        self,
        tickers: pd.Index,
        fill_usd: np.ndarray,
        raw_t: np.ndarray,
        kind: np.ndarray,
        tc: dict[str, float],
        t: pd.Timestamp,
    ) -> pd.DataFrame:
        cf = getattr(self, "_last_cost_frame", None)
        rows = []
        for j in np.flatnonzero(fill_usd):
            tk = str(tickers[int(j)])
            c = cf.loc[tk] if cf is not None and tk in cf.index else None
            rows.append(
                {
                    "ticker": tk,
                    "side": "BUY" if fill_usd[j] > 0 else "SELL",
                    "shares": float(fill_usd[j] / raw_t[j]),
                    "px": float(raw_t[j]),
                    "usd": float(fill_usd[j]),
                    "spread": 0.0 if c is None else float(c["spread"]),
                    "impact": 0.0 if c is None else float(c["impact"]),
                    "fees": 0.0 if c is None else float(c["fees"]),
                    "kind": str(kind[j]) or "rebalance",
                    "date": iso(t),
                }
            )
        cols = ["ticker", "side", "shares", "px", "usd", "spread", "impact", "fees", "kind", "date"]
        return pd.DataFrame(rows, columns=cols)


POSITION_COLUMNS = ["ticker", "shares", "px", "px_date", "value", "weight"]


def positions_frame(state: BookState) -> pd.DataFrame:
    """Positions at the close of ``state.date``: implied as-traded shares, valuation
    price (and its date: earlier than the book date for a halted name), value and
    weight of the NAV after the day's fills."""
    rows = []
    nav = float(state.nav)
    for tk in sorted(state.shares):
        q = float(state.shares[tk])
        if q == 0:
            continue
        px = float(state.px[tk])
        rows.append(
            {
                "ticker": tk,
                "shares": q,
                "px": px,
                "px_date": state.px_date.get(tk),
                "value": q * px,
                "weight": q * px / nav if nav > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


ORDER_COLUMNS = [
    "ticker",
    "side",
    "shares",
    "px",
    "usd",
    "target_w",
    "current_w",
    "kind",
    "close_out",
    "exec_date",
    "est_cost_usd",
    "est_cost_bps",
]


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in ORDER_COLUMNS})


def _columns(frame: pd.DataFrame) -> dict[str, list[Any]]:
    """Column-oriented dict of a small frame (compact JSON in the journal)."""
    return {str(c): frame[c].tolist() for c in frame.columns}


def frame_from_columns(cols: Mapping[str, list[Any]] | None) -> pd.DataFrame:
    return pd.DataFrame({k: list(v) for k, v in (cols or {}).items()})
