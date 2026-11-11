"""Daily-bar backtest simulator (closing-auction rebalancer).

Timeline for each simulation date ``t`` (close of ``t``), with
``L = signal_lag_days`` (``cfg.execution.signal_lag_days``, default 1; design
decision D4) and ``s`` = the trading day ``L`` days before ``t`` in ``md.dates``:

1. ``alpha_fn(s)`` and ``risk_fn(s, alpha)`` see only data ``<= s``; the
   optimizer prices costs on the snapshot as of ``s`` (ADV, sigma, spread,
   borrow: data ``<= s``) **with the auction share of the execution date**
   ``t`` -- the closing-auction calendar (quad witching, Russell
   reconstitution) is published in advance, so pricing a heavy day ``t`` as
   the normal day ``s`` would only make the optimizer under-use a cheap
   auction (audit 2, finding 14).  With ``L = 0``,
   ``s = t`` (close-as-signal).  With ``L >= 1`` the decision uses data that
   is ``L`` days stale -- the honest setting for a strategy that cannot see
   the close it trades at; the Sharpe difference between 0 and 1 bounds the
   close-as-signal bias.  On the first ``L`` days of ``md`` there is no signal
   date and the book is held.  Universe eligibility is read as of ``s``;
   whether a name *can* trade (has a price) is read as of ``t``.  The optimizer
   sees the current book ``w_pre`` and NAV marked at ``t`` (positions are
   known; only the signal inputs are lagged).
2. The optimizer turns them into target weights ``w_post`` (fraction of the
   NAV at the close of ``t``, before costs).
3. Trades ``x = w_post - w_pre`` execute at the close of ``t`` and are charged
   ``cost_model.estimate(x * nav_t, snapshot_t, venue)`` -- the snapshot as of
   the *execution* date ``t`` (actual liquidity), not ``s``.  The default
   venue is ``"routed"`` (design decision D3): each name's trade up to
   ``ExecutionConfig.auction_max_pct_of_auction_volume`` of the closing
   auction is priced at the auction, the rest in the continuous session
   (``qts.tcost.model`` docstring).
   Multi-day impact persistence: per name, the simulator carries the USD
   already traded in the current same-direction streak (``Q_prev``; reset
   when the trade sign flips or a day has no trade) and passes it to
   ``estimate(prior_streak_usd=)`` when the model accepts it (exact streak
   increment ``C(Q_prev + q) - C(Q_prev)``, ``qts.tcost.model`` docstring),
   besides the streak length ``streak``.
4. Positions earn close-to-close returns ``t -> t+1``; shorts pay one day of
   borrow ``cost_model.holding_cost_usd(w_post, nav_t, snapshot_t)``.
5. NAV identity (exact by construction, tested):
   ``nav_{t+1} = nav_t * (1 + sum_i w_post_i r_{i,t+1}) - costs_t`` where
   ``costs_t`` is the USD total of spread + impact + fees + borrow booked on
   ``t``.  Pre-trade weights on ``t+1`` are the drifted post-trade weights
   re-based to ``nav_{t+1}`` (cash absorbs costs).

Code layout: :meth:`DailyBarSimulator.run` is a short loop over days; each
day is a fixed pipeline of small steps, one method each, in this order:

==========================  ==============================================
step                        method
==========================  ==============================================
day view, signal date       :meth:`DailyBarSimulator._day`
eligibility, forced exits   :meth:`DailyBarSimulator._universe_step`
signal -> target weights    :meth:`DailyBarSimulator._target_step`
trade costs (fills)         :meth:`DailyBarSimulator._cost_step`
mark-to-market, halts,      :meth:`DailyBarSimulator._mark_to_market_step`
delistings
accounting (NAV, drift)     :meth:`DailyBarSimulator._accounting_step`
diagnostics                 :class:`_Recorder` / :class:`_EventLog`
==========================  ==============================================

Run inputs are a frozen :class:`~qts.backtest.spec.BacktestSpec`; the state
carried from day to day is :class:`_BookState`.

Cost model: any :class:`qts.core.costs.CostModel` (``snapshot`` /
``estimate`` / ``holding_cost_usd``).  A duck-typed legacy model without
``snapshot`` is wrapped once, at construction, by
:class:`~qts.backtest.cost_adapter._LegacyCostModel`; there is no silent
fallback between paths.

Halts vs delistings (same classification as the labels, ``qts.alpha.targets``):
a held name with no close on ``t+1`` is either *halted* or *delisted*, decided
without looking at later prices:

* ``md.delisting_dates`` covers the name -> delisted iff the vendor's last
  trading date is before ``t+1``; otherwise (``NaT`` or a later date) halted;
* otherwise the causal gap rule (``cfg.alpha.delisting_gap_days = G``) ->
  delisted on the first date ``t+1`` with no close that is ``G`` trading days
  (``md.dates``) after the name's last close; before that, halted.

A *halted* name is frozen at its last price: return 0 on each missing day, no
trading (it is on the optimizer's no-trade list), weight carried (drifting only
with NAV); borrow on a frozen short is still charged.  When a close reappears
the return is measured from the last close before the halt, so the whole
halt-period move is realised on the resumption day.  Frozen name-days are
counted in ``diagnostics["n_halted_name_days"]`` and episodes recorded in
``diagnostics["halts"]`` (``outcome`` resumed / delisted / open).

A *delisted* name earns ``cfg.cost.delisting_haircut`` (default -30%, against
the position's sign) on the decision day ``t+1`` (relative to its last, frozen
price), is force-liquidated at spread + explicit fees (continuous venue, booked
as a trade on decision row ``t`` and counted in turnover) and the event is
recorded in ``BacktestResult.diagnostics["delistings"]``.  Names with no price
on ``t`` are placed on the optimizer's no-trade list.

Eligibility: ``md.eligible`` (wide bool panel, if present) or the
``eligible_fn(date) -> Index`` kwarg marks the investable universe as of a
date.  An ineligible name gets expected return 0; if it is not held it goes on
the no-trade list (nothing can be opened), and if it *is* held it enters a
forced-exit set that the optimizer receives as ``force_to_zero`` with the
allowed ``|w|`` shrinking linearly to 0 over
``cfg.portfolio.exit_ineligible_days`` days (recorded in
``diagnostics["forced_exits"]``).

Data-error prices (``cfg.risk.clean_returns``; :mod:`qts.core.returns`): a
close that would produce a return above ``cfg.risk.return_abs_cap_up`` (+300 %)
or below ``cfg.risk.return_abs_cap_down`` (-95 %) is a bad print, not tradable
P&L -- nobody earned +50,000 % on a restated close, and nobody lost 99.8 % on
its reversal the next day.  Such closes are blanked before the run
(:func:`qts.core.returns.flag_bad_prices`), so the name takes the *halt* path
above on that day (frozen, no return, no trade) and the next day's return is
measured from the last good close.  Blanked closes inside the run are counted
in ``diagnostics["n_data_error_returns"]`` and listed (first 100) in
``diagnostics["data_error_returns"]``.  Nothing else is cleaned here: a genuine
-80 % day is P&L.

Risk-model sanity (``cfg.validation.max_exante_vol``): on every rebalance the
annualised ex-ante vol of the *incoming* book under the day's risk model (or,
when the book is empty, of a sqrt-cap-weighted long-only reference book) is
checked; above the limit the date, the vol and the top-5 contributing names and
factors are logged, appended to ``diagnostics["exante_vol_alerts"]`` and
counted in ``diagnostics["n_exante_vol_alerts"]`` -- the ``risk_model_sanity``
hurdle gate fails when alerts exceed ``max_exante_vol_alert_frac`` of rebalance
solves.  A target-vol optimizer facing a 10,000 % ex-ante vol liquidates
everything and the book then stays empty for many covariance half-lives; the
gate makes that failure visible instead of reporting a quiet flat NAV.

Failed solves: a rebalance whose solve status is not ``optimal`` /
``optimal_inaccurate`` holds the current book; it is counted in
``diagnostics["n_failed_solves"]`` and listed in
``diagnostics["failed_solves"]`` (date, status, solver errors).
``diagnostics["n_rebalance_solves"]`` is the denominator of the
``failed_solves`` hurdle gate (:mod:`qts.validation.hurdles`).

Recorded panels are indexed by the decision date ``t``: ``weights`` (post-
trade), ``trades``, ``costs``, ``exante_var``; ``returns`` and ``nav`` are
indexed by the date the return is realised (``t+1``).  ``returns`` columns:
``gross`` (``sum_i w_post_i r_{i,t+1}``, before every cost), ``net``
(``nav_{t+1}/nav_t - 1``) and ``gross_after_borrow`` (``gross - borrow_t /
nav_t``: after the holding cost but before trading costs).  Per-side trading
cost gates (breakeven cost, cost sensitivity) must start from
``gross_after_borrow``: borrow is not a per-side cost, and charging it nowhere
makes the gate optimistic for short-heavy books (audit 2, finding 1).
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any

import numpy as np
import pandas as pd

from qts.backtest.cost_adapter import _LegacyCostModel, cost_frame
from qts.backtest.result import COST_COLUMNS, BacktestResult
from qts.backtest.spec import BacktestSpec
from qts.core.config import SystemConfig
from qts.core.costs import VENUE_CONTINUOUS, VENUE_ROUTED, LiquiditySnapshot
from qts.core.returns import flag_bad_prices
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput
from qts.portfolio.costs import Snapshot
from qts.portfolio.optimizer import OK_STATUSES, SinglePeriodOptimizer

log = logging.getLogger(__name__)

AlphaFn = Callable[[pd.Timestamp], AlphaOutput]
RiskFn = Callable[[pd.Timestamp, "pd.Series | None"], RiskModelOutput]
EligibleFn = Callable[[pd.Timestamp], pd.Index]

_TRADE_EPS = 1e-10  # |trade weight| below this is solver noise: no trade
_HELD_EPS = 1e-12  # |weight| above this counts as held for forced exits

__all__ = [
    "COST_COLUMNS",
    "BacktestResult",
    "BacktestSpec",
    "DailyBarSimulator",
    "cost_frame",
]


# --------------------------------------------------------------------------- run context
class _SnapshotCache:
    """Point-in-time cost snapshots by date; keeps only the ``lag + 3`` most
    recent dates (the window the signal lag still needs)."""

    def __init__(self, build: Callable[[pd.Timestamp], Snapshot], lag: int):
        self._build = build
        self._lag = lag
        self._cache: dict[pd.Timestamp, Snapshot] = {}

    def __call__(self, d: pd.Timestamp) -> Snapshot:
        if d not in self._cache:
            if len(self._cache) > self._lag + 2:  # keep only the window still needed
                self._cache.pop(next(iter(self._cache)))
            self._cache[d] = self._build(d)
        return self._cache[d]


@dataclass(frozen=True, eq=False)
class _RunContext:
    """Everything fixed for the whole run (built once by :meth:`DailyBarSimulator._context`)."""

    md: MarketData
    dates: pd.DatetimeIndex
    tickers: pd.Index
    lag: int
    md_pos: np.ndarray  # positions of ``dates`` in ``md.dates``
    R: pd.DataFrame  # return realised on the row date, from the last close before it
    price_ok: pd.DataFrame  # has a close on the row date
    last_close_md: np.ndarray  # (len(md.dates), N) position of the last close so far, nan before
    vendor_pos: np.ndarray  # (N,) vendor delisting position in md.dates, nan = not covered
    gap_days: int
    elig_md: np.ndarray | None  # (len(md.dates), N) eligibility panel, rows = md.dates
    eligible_fn: EligibleFn | None
    exit_days: int
    haircut: float
    venue: str
    rebalance_every: int
    neutral_styles: list[str] | None
    progress: bool
    solve_accepts_hl: bool
    snap_at: _SnapshotCache

    @property
    def n_names(self) -> int:
        return len(self.tickers)


@dataclass(frozen=True, eq=False)
class _Day:
    """One simulation date: decision date ``t``, realisation date ``t1``, signal date."""

    i: int
    t: pd.Timestamp
    t1: pd.Timestamp
    sig_date: pd.Timestamp | None  # None during the first ``lag`` days of md
    alive: np.ndarray  # (N,) has a price on t (can trade)
    snap_exec: Snapshot  # liquidity as of the execution date t

    @property
    def rebalance(self) -> bool:
        return self.sig_date is not None


@dataclass
class _BookState:
    """State carried from one day to the next."""

    nav: float
    w_pre: np.ndarray  # pre-trade weights at the close of t
    streak: np.ndarray  # consecutive same-direction trading days per name
    streak_usd: np.ndarray  # |USD| traded in the current streak, through the last trade
    last_sign: np.ndarray
    last_risk: RiskModelOutput | None = None
    last_alpha: AlphaOutput | None = None


@dataclass
class _Costs:
    """USD costs booked on one decision date."""

    spread: float
    impact: float
    fees: float
    borrow: float

    @property
    def total(self) -> float:
        return self.spread + self.impact + self.fees + self.borrow

    def as_row(self) -> tuple[float, float, float, float, float]:
        return (self.spread, self.impact, self.fees, self.borrow, self.total)


@dataclass
class _EventLog:
    """Event diagnostics accumulated over the run."""

    halts_open: dict[str, dict[str, Any]] = field(default_factory=dict)
    halts: list[dict[str, object]] = field(default_factory=list)
    n_halted: int = 0
    forced: dict[str, dict[str, float]] = field(default_factory=dict)  # ticker -> {"w0", "k"}
    forced_log: list[dict[str, object]] = field(default_factory=list)
    n_out_of_model: int = 0
    n_ineligible_days: int = 0
    n_no_signal: int = 0
    delistings: list[dict[str, object]] = field(default_factory=list)
    solve_log: list[dict[str, object]] = field(default_factory=list)
    failed_solves: list[dict[str, object]] = field(default_factory=list)
    exante_vol_alerts: list[dict[str, object]] = field(default_factory=list)
    n_data_error_returns: int = 0
    data_error_returns: list[dict[str, object]] = field(default_factory=list)

    @property
    def n_failed(self) -> int:
        return len(self.failed_solves)

    def all_halts(self) -> list[dict[str, object]]:
        return self.halts + [{**ep, "end": None, "outcome": "open"} for ep in self.halts_open.values()]


class _Recorder:
    """Per-day output arrays; :meth:`truncate` when the run stops early."""

    def __init__(self, T: int, N: int, nav0: float):
        self.T = T
        self.W = np.zeros((T, N))
        self.X = np.zeros((T, N))
        self.C = np.zeros((T, 5))
        self.EXV = np.full(T, np.nan)
        self.TO = np.zeros(T)
        self.TC = np.full(T, np.nan)
        self.navs = np.empty(T + 1)
        self.navs[0] = nav0
        self.gross = np.empty(T)
        self.net = np.empty(T)
        self.borrow_ret = np.empty(T)

    def record(
        self,
        i: int,
        w_post: np.ndarray,
        x: np.ndarray,
        costs: _Costs,
        nav: float,
        nav1: float,
        g_ret: float,
        n_ret: float,
        exante_var: float | None,
    ) -> None:
        self.W[i] = w_post
        self.X[i] = x
        self.C[i] = costs.as_row()
        self.TO[i] = float(np.abs(x).sum())
        if exante_var is not None:
            self.EXV[i] = exante_var
        self.gross[i] = g_ret
        self.net[i] = n_ret
        self.borrow_ret[i] = costs.borrow / nav
        self.navs[i + 1] = nav1

    def truncate(self, T_eff: int) -> None:
        self.W, self.X, self.C = self.W[:T_eff], self.X[:T_eff], self.C[:T_eff]
        self.EXV, self.TO, self.TC = self.EXV[:T_eff], self.TO[:T_eff], self.TC[:T_eff]
        self.navs, self.gross, self.net = self.navs[: T_eff + 1], self.gross[:T_eff], self.net[:T_eff]
        self.borrow_ret = self.borrow_ret[:T_eff]
        self.T = T_eff


def _reference_weights(market_cap: pd.Series, tickers: pd.Index) -> pd.Series:
    """Long-only sqrt-cap book (gross 1) over ``tickers``; equal weights without caps."""
    cap = market_cap.reindex(tickers).to_numpy(dtype=float)
    w = pd.Series(np.sqrt(np.where(np.isfinite(cap), np.clip(cap, 0.0, None), 0.0)), index=tickers)
    tot = float(w.sum())
    if tot <= 0:
        return pd.Series(1.0 / max(len(tickers), 1), index=tickers)
    return w / tot


def _exante_contributors(
    risk: RiskModelOutput, w: pd.Series, periods_per_year: float
) -> tuple[float, list[tuple[str, float]], list[tuple[str, float]]]:
    """Annualised ex-ante vol of ``w`` under ``risk`` and the top-5 annualised
    variance contributions of names (specific, ``w_i^2 d_i``) and factors
    (``(B'w)_k (F B'w)_k``)."""
    t = w.index
    B = risk.exposures.reindex(t).fillna(0.0).to_numpy(dtype=float)
    F = risk.factor_cov.loc[risk.factors, risk.factors].to_numpy(dtype=float)
    d = risk.specific_var.reindex(t).to_numpy(dtype=float)
    d = np.where(np.isfinite(d), d, 0.0)
    wv = w.to_numpy(dtype=float)
    fw = B.T @ wv
    fac = fw * (F @ fw)
    spec = wv**2 * d
    var = float(fac.sum() + spec.sum()) + (
        risk.alignment_var * float(risk.alignment_vector.reindex(t).fillna(0.0).to_numpy() @ wv) ** 2
        if risk.alignment_vector is not None and risk.alignment_var > 0
        else 0.0
    )
    vol = float(np.sqrt(max(var, 0.0) * periods_per_year))
    kn = np.argsort(-spec)[:5]
    kf = np.argsort(-np.abs(fac))[:5]
    names = [(str(t[k]), float(spec[k] * periods_per_year)) for k in kn if spec[k] > 0]
    factors = [(str(risk.factors[k]), float(fac[k] * periods_per_year)) for k in kf]
    return vol, names, factors


# --------------------------------------------------------------------------- simulator
class DailyBarSimulator:
    def __init__(self, cfg: SystemConfig, cost_model: Any, adv_window: int = 63, vol_window: int = 63):
        """``cost_model``: a :class:`qts.core.costs.CostModel` (has ``snapshot``),
        or a legacy duck-typed model (wrapped by :class:`_LegacyCostModel`).
        ``adv_window`` / ``vol_window`` only apply to the legacy built-in snapshot."""
        self.cfg = cfg
        self.cost_model = cost_model
        self.adv_window = adv_window
        self.vol_window = vol_window
        if callable(getattr(cost_model, "snapshot", None)):
            self._cm = cost_model
        else:
            log.info(
                "DailyBarSimulator: %s has no snapshot(); using the legacy cost-model adapter",
                type(cost_model).__name__,
            )
            self._cm = _LegacyCostModel(cost_model, cfg, adv_window, vol_window)
        try:
            params = inspect.signature(self._cm.estimate).parameters
            self._pass_prior_streak = "prior_streak_usd" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            self._pass_prior_streak = False
        self._bad_prices: list[dict[str, object]] = []  # filled per run by _context

    _cost_frame = staticmethod(cost_frame)

    def _snapshot(self, md: MarketData, t: pd.Timestamp) -> Snapshot:
        """Point-in-time cost snapshot as of ``t`` (``cost_model.snapshot``)."""
        return self._cm.snapshot(md, pd.Timestamp(t))

    @staticmethod
    def _optimizer_snapshot(snap_sig: Snapshot, snap_exec: Snapshot) -> Snapshot:
        """The optimizer's cost snapshot: signal-date liquidity (``snap_sig``) with the
        execution date's auction share (``snap_exec``) -- module docstring, step 1."""
        if snap_sig is snap_exec:
            return snap_sig
        share = snap_exec.get("auction_share") if hasattr(snap_exec, "get") else None
        if share is None:
            return snap_sig
        if isinstance(snap_sig, LiquiditySnapshot):
            if snap_sig.auction_share == float(share):
                return snap_sig
            return dc_replace(snap_sig, auction_share=float(share))
        out = dict(snap_sig)
        out["auction_share"] = share
        return out

    # ----------------------------------------------------------------- public API
    def run(
        self,
        md: MarketData,
        dates: pd.DatetimeIndex | BacktestSpec,
        alpha_fn: AlphaFn,
        risk_fn: RiskFn,
        optimizer: SinglePeriodOptimizer,
        nav0: float = 1e6,
        rebalance_every: int = 1,
        venue: str = VENUE_ROUTED,
        neutral_styles: Sequence[str] | None = None,
        warm_start_weights: pd.Series | None = None,
        progress: bool = False,
        eligible_fn: EligibleFn | None = None,
        exit_ineligible_days: int | None = None,
        signal_lag_days: int | None = None,
    ) -> BacktestResult:
        """Simulate ``dates`` (a :class:`BacktestSpec`, or the dates plus keyword
        arguments, which are packed into one).

        ``eligible_fn(date)`` returns the tickers investable as of ``date``;
        when ``md.eligible`` (wide bool panel) is present it takes precedence
        and ``eligible_fn`` is ignored.  ``exit_ineligible_days`` overrides
        ``cfg.portfolio.exit_ineligible_days`` (linear unwind horizon).
        ``venue`` in {"routed" (default), "auction", "continuous"} prices every
        trade (and is passed to the optimizer's cost terms).
        ``signal_lag_days`` overrides ``cfg.execution.signal_lag_days`` (module
        docstring, step 1)."""
        if isinstance(dates, BacktestSpec):
            spec = dates
        else:
            spec = BacktestSpec.create(
                dates,
                nav0=nav0,
                rebalance_every=rebalance_every,
                venue=venue,
                neutral_styles=neutral_styles,
                warm_start_weights=warm_start_weights,
                progress=progress,
                eligible_fn=eligible_fn,
                exit_ineligible_days=exit_ineligible_days,
                signal_lag_days=signal_lag_days,
            )
        return self.run_spec(md, spec, alpha_fn, risk_fn, optimizer)

    def run_spec(
        self,
        md: MarketData,
        spec: BacktestSpec,
        alpha_fn: AlphaFn,
        risk_fn: RiskFn,
        optimizer: SinglePeriodOptimizer,
    ) -> BacktestResult:
        """The day loop: one fixed step pipeline per date (module docstring)."""
        t_start = time.perf_counter()
        ctx = self._context(md, spec, optimizer)
        state = self._initial_state(ctx, spec)
        events = _EventLog()
        rec = _Recorder(len(ctx.dates) - 1, ctx.n_names, state.nav)
        for i in range(rec.T):
            day = self._day(ctx, i)
            inel, no_trade, force_caps = self._universe_step(ctx, day, state, events)
            w_post = self._target_step(
                ctx, day, state, events, rec, alpha_fn, risk_fn, optimizer, inel, no_trade, force_caps
            )
            x = w_post - state.w_pre
            x[np.abs(x) < _TRADE_EPS] = 0.0
            w_post = state.w_pre + x
            costs = self._cost_step(ctx, day, state, x, w_post)
            r, liquidate, x = self._mark_to_market_step(ctx, day, state, events, w_post, x, costs)
            if not self._accounting_step(ctx, day, state, rec, w_post, x, r, liquidate, costs):
                break
        return self._result(ctx, spec, events, rec, t_start)

    # ----------------------------------------------------------------- set-up
    def _context(self, md: MarketData, spec: BacktestSpec, optimizer: SinglePeriodOptimizer) -> _RunContext:
        dates = pd.DatetimeIndex(spec.dates)
        if len(dates) < 2:
            raise ValueError("need at least two dates")
        if not dates.isin(md.dates).all():
            raise ValueError("dates must be a subset of md.dates")
        lag = int(
            spec.signal_lag_days
            if spec.signal_lag_days is not None
            else getattr(self.cfg.execution, "signal_lag_days", 0)
        )
        if lag < 0:
            raise ValueError("signal_lag_days must be >= 0")
        tickers = md.tickers
        close_md, self._bad_prices = self._clean_close(md, dates)
        close = close_md.loc[dates]
        # return realised on the row date, measured from the last close before it:
        # equals pct_change on consecutive closes; after a halt it carries the whole
        # halt-period move (module docstring, "Halts vs delistings")
        R = close / close.ffill().shift(1) - 1.0
        R = R.where(close.notna())
        # halt / delisting classification inputs (positions in md.dates)
        have_md = close_md.notna().to_numpy()
        last_close_md = (
            pd.DataFrame(np.where(have_md, np.arange(len(md.dates), dtype=float)[:, None], np.nan))
            .ffill()
            .to_numpy()
        )
        elig_panel = getattr(md, "eligible", None)
        elig_md: np.ndarray | None = None  # rows aligned with md.dates
        eligible_fn = spec.eligible_fn
        if elig_panel is not None:
            elig_md = elig_panel.reindex(index=md.dates, columns=tickers).fillna(False).astype(bool).values
            eligible_fn = None
        exit_days = int(
            spec.exit_ineligible_days
            if spec.exit_ineligible_days is not None
            else getattr(self.cfg.portfolio, "exit_ineligible_days", 5)
        )
        return _RunContext(
            md=md,
            dates=dates,
            tickers=tickers,
            lag=lag,
            md_pos=md.dates.get_indexer(dates),
            R=R,
            price_ok=close.notna(),
            last_close_md=last_close_md,
            vendor_pos=md.delisting_positions(),
            gap_days=int(getattr(self.cfg.alpha, "delisting_gap_days", 63)),
            elig_md=elig_md,
            eligible_fn=eligible_fn,
            exit_days=max(exit_days, 1),
            haircut=float(self.cfg.cost.delisting_haircut),
            venue=spec.venue,
            rebalance_every=spec.rebalance_every,
            neutral_styles=None if spec.neutral_styles is None else list(spec.neutral_styles),
            progress=spec.progress,
            # signal half-life (handbook fix #5): AlphaOutput.half_life_days drives the
            # optimizer's Garleanu-Pedersen shrink; passed only when solve() accepts it
            solve_accepts_hl="alpha_half_life_days" in inspect.signature(optimizer.solve).parameters,
            snap_at=_SnapshotCache(lambda d: self._snapshot(md, d), lag),
        )

    def _clean_close(
        self, md: MarketData, dates: pd.DatetimeIndex
    ) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        """``md.close`` with data-error prints blanked (module docstring "Data-error
        prices"), plus the list of blanked ``(date, ticker)`` cells inside ``dates``
        (raw close kept for the record).  Raw ``md.close`` and an empty list when
        ``cfg.risk.clean_returns`` is off."""
        rcfg = self.cfg.risk
        if not getattr(rcfg, "clean_returns", False):
            return md.close, []
        clean, bad = flag_bad_prices(
            md.close, cap_up=float(rcfg.return_abs_cap_up), cap_down=float(rcfg.return_abs_cap_down)
        )
        inside = bad.loc[dates]
        raw_close = md.close.loc[dates].to_numpy(dtype=float)
        cells = np.argwhere(inside.to_numpy())
        out: list[dict[str, object]] = [
            {
                "date": pd.Timestamp(dates[int(i)]),
                "ticker": str(inside.columns[int(j)]),
                "close": float(raw_close[int(i), int(j)]),
            }
            for i, j in cells
        ]
        if out:
            log.warning(
                "backtest: %d data-error closes blanked (treated as halts; first: %s %s close=%g)",
                len(out),
                pd.Timestamp(dates[int(cells[0][0])]).date(),
                out[0]["ticker"],
                out[0]["close"],
            )
        return clean, out

    @staticmethod
    def _initial_state(ctx: _RunContext, spec: BacktestSpec) -> _BookState:
        N = ctx.n_names
        w_pre = np.zeros(N)
        if spec.warm_start_weights is not None:
            w_pre = (
                pd.Series(spec.warm_start_weights, dtype=float).reindex(ctx.tickers).fillna(0.0).to_numpy()
            )
        return _BookState(
            nav=float(spec.nav0),
            w_pre=w_pre,
            streak=np.zeros(N),
            streak_usd=np.zeros(N),
            last_sign=np.zeros(N),
        )

    @staticmethod
    def _day(ctx: _RunContext, i: int) -> _Day:
        t = ctx.dates[i]
        snap_exec = ctx.snap_at(t)
        p_sig = int(ctx.md_pos[i]) - ctx.lag
        return _Day(
            i=i,
            t=t,
            t1=ctx.dates[i + 1],
            sig_date=ctx.md.dates[p_sig] if p_sig >= 0 else None,
            alive=ctx.price_ok.iloc[i].to_numpy(),
            snap_exec=snap_exec,
        )

    # ----------------------------------------------------------------- step 1: universe
    @staticmethod
    def _universe_step(
        ctx: _RunContext, day: _Day, state: _BookState, events: _EventLog
    ) -> tuple[np.ndarray, list[str], dict[str, float]]:
        """Eligibility as of the signal date and the forced-exit schedule.

        Returns ``(inel, no_trade, force_caps)``: alive-but-ineligible mask, the
        optimizer's no-trade list (no price, or ineligible and not held) and the
        per-name ``|w|`` caps of names being unwound."""
        tickers, alive = ctx.tickers, day.alive
        elig_date = day.sig_date if day.sig_date is not None else day.t
        if ctx.elig_md is not None:
            eligible = ctx.elig_md[ctx.md.dates.get_loc(elig_date)] & alive
        elif ctx.eligible_fn is not None:
            eligible = tickers.isin(pd.Index(ctx.eligible_fn(elig_date))) & alive
        else:
            eligible = alive.copy()
        inel = alive & ~eligible
        events.n_ineligible_days += int(inel.sum())
        held = np.abs(state.w_pre) > _HELD_EPS
        forced = events.forced
        for j in np.where(inel & held)[0]:
            tk = tickers[int(j)]
            if tk not in forced:
                forced[tk] = {"w0": float(abs(state.w_pre[j])), "k": 0.0}
                events.forced_log.append({"date": day.t, "ticker": tk, "weight": float(state.w_pre[j])})
        for tk in list(forced):
            j = tickers.get_loc(tk)
            if eligible[j] or not held[j]:
                forced.pop(tk)  # back in the universe, or already flat
        force_caps: dict[str, float] = {}
        for tk, st in forced.items():
            st["k"] += 1.0
            force_caps[tk] = max(st["w0"] * (1.0 - st["k"] / ctx.exit_days), 0.0)
        no_trade = list(tickers[~alive]) + list(tickers[inel & ~held])
        return inel, no_trade, force_caps

    # ----------------------------------------------------------------- step 2: signal -> target
    def _target_step(
        self,
        ctx: _RunContext,
        day: _Day,
        state: _BookState,
        events: _EventLog,
        rec: _Recorder,
        alpha_fn: AlphaFn,
        risk_fn: RiskFn,
        optimizer: SinglePeriodOptimizer,
        inel: np.ndarray,
        no_trade: list[str],
        force_caps: dict[str, float],
    ) -> np.ndarray:
        """Post-trade target weights: the current book on non-rebalance days and
        before the first signal date, else the optimizer's solution (names
        without a price keep their weight)."""
        w_post = state.w_pre.copy()
        if day.i % ctx.rebalance_every != 0:
            return w_post
        if not day.rebalance:
            events.n_no_signal += 1
            return w_post
        sig_date = day.sig_date
        assert sig_date is not None
        tickers, alive = ctx.tickers, day.alive
        state.last_alpha = alpha_fn(sig_date)
        exp_ret = pd.Series(state.last_alpha.expected_return, dtype=float).reindex(tickers)
        exp_ret[~alive] = 0.0
        exp_ret[inel] = 0.0
        exp_ret = exp_ret.fillna(0.0)
        state.last_risk = risk_fn(sig_date, exp_ret)
        self._check_exante_vol(ctx, day, state, events, sig_date)
        solve_kw: dict[str, Any] = {}
        hl = getattr(state.last_alpha, "half_life_days", None)
        if ctx.solve_accepts_hl and hl is not None:
            solve_kw["alpha_half_life_days"] = float(hl)
        res = optimizer.solve(
            alpha=exp_ret,
            risk=state.last_risk,
            w_current=pd.Series(state.w_pre, index=tickers),
            nav=state.nav,
            snapshot=self._optimizer_snapshot(ctx.snap_at(sig_date), day.snap_exec),
            venue=ctx.venue,
            neutral_styles=ctx.neutral_styles,
            no_trade=no_trade,
            horizon_days=getattr(state.last_alpha, "horizon_days", None) or optimizer.horizon_days,
            force_to_zero=force_caps or None,
            **solve_kw,
        )
        if res.status not in OK_STATUSES:
            events.failed_solves.append(
                {
                    "date": day.t,
                    "status": res.status,
                    "solver_errors": list(getattr(res, "solver_errors", []) or []),
                }
            )
            log.warning(
                "optimizer solve failed on %s: status=%s (holding current book)", day.t.date(), res.status
            )
        events.n_out_of_model += int(getattr(res, "n_out_of_model", 0) or 0)
        w_post = np.array(res.w_target.reindex(tickers).fillna(0.0).values, dtype=float, copy=True)
        w_post[~alive] = state.w_pre[~alive]  # cannot trade names without a price
        rec.TC[day.i] = res.transfer_coefficient
        events.solve_log.append(_solve_log_entry(day, res))
        return w_post

    def _check_exante_vol(
        self, ctx: _RunContext, day: _Day, state: _BookState, events: _EventLog, sig_date: pd.Timestamp
    ) -> None:
        """Risk-model sanity guard (module docstring): annualised ex-ante vol of the
        incoming book (or of the sqrt-cap reference book when it is empty) under
        ``state.last_risk``; above ``cfg.validation.max_exante_vol`` the alert is
        logged with the top-5 contributing names and factors and recorded."""
        risk = state.last_risk
        if risk is None:
            return
        limit = float(getattr(self.cfg.validation, "max_exante_vol", 2.0))
        ppy = float(self.cfg.trading_days_per_year)
        tickers = ctx.tickers
        held = np.abs(state.w_pre).sum() > _HELD_EPS
        if held:
            w = pd.Series(state.w_pre, index=tickers)
            book = "incoming"
        else:
            cap = ctx.md.market_cap.loc[:sig_date]
            cap_row = cap.iloc[-1] if len(cap) else pd.Series(dtype=float)
            w = _reference_weights(cap_row, risk.tickers)
            book = "reference"
        vol, top_names, top_factors = _exante_contributors(risk, w, ppy)
        if not np.isfinite(vol) or vol <= limit:
            return
        entry: dict[str, object] = {
            "date": day.t,
            "signal_date": sig_date,
            "book": book,
            "exante_vol": vol,
            "limit": limit,
            "top_names": top_names,
            "top_factors": top_factors,
        }
        events.exante_vol_alerts.append(entry)
        log.warning(
            "backtest %s: risk model ex-ante vol of the %s book = %.1f%% (limit %.0f%%); "
            "top names %s; top factors %s",
            day.t.date(),
            book,
            100 * vol,
            100 * limit,
            [(n, round(v, 4)) for n, v in top_names],
            [(f, round(v, 4)) for f, v in top_factors],
        )

    # ----------------------------------------------------------------- step 3: costs (fills)
    def _cost_step(
        self, ctx: _RunContext, day: _Day, state: _BookState, x: np.ndarray, w_post: np.ndarray
    ) -> _Costs:
        """USD trading costs on the execution date's liquidity at the run's venue,
        plus one day of borrow.  Updates the per-name same-direction streak that
        drives multi-day impact persistence."""
        tickers, nav = ctx.tickers, state.nav
        sgn = np.sign(x)
        same = (sgn != 0) & (sgn == state.last_sign)
        state.streak = np.where(sgn != 0, np.where(same, state.streak + 1.0, 1.0), 0.0)
        prior_usd = np.where(same, state.streak_usd, 0.0)  # Q_{k-1}: earlier days of this streak
        state.streak_usd = np.where(sgn != 0, prior_usd + np.abs(x) * nav, 0.0)
        state.last_sign = sgn
        est_kw: dict[str, Any] = {"venue": ctx.venue, "streak": pd.Series(state.streak, index=tickers)}
        if self._pass_prior_streak:
            est_kw["prior_streak_usd"] = pd.Series(prior_usd, index=tickers)
        est = self._cm.estimate(pd.Series(x * nav, index=tickers), day.snap_exec, **est_kw)
        cf = cost_frame(est, tickers)
        borrow = self._cm.holding_cost_usd(pd.Series(w_post, index=tickers), nav, day.snap_exec)
        return _Costs(
            spread=float(cf["spread"].sum()),
            impact=float(cf["impact"].sum()),
            fees=float(cf["fees"].sum()),
            borrow=float(np.nansum(np.asarray(borrow, dtype=float))),
        )

    # ----------------------------------------------------------------- step 4: mark to market
    def _mark_to_market_step(
        self,
        ctx: _RunContext,
        day: _Day,
        state: _BookState,
        events: _EventLog,
        w_post: np.ndarray,
        x: np.ndarray,
        costs: _Costs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns ``t -> t+1`` per name with halts frozen at 0 and delistings at
        the haircut; delisted names are force-liquidated (cost added to
        ``costs``, liquidation added to the trade vector).  Returns
        ``(r, liquidate, x)``."""
        N = ctx.n_names
        r = ctx.R.iloc[day.i + 1].to_numpy(copy=True)
        held = w_post != 0
        nan_r = ~np.isfinite(r)
        r[nan_r] = 0.0  # unheld names: irrelevant; halted (frozen) names: 0
        missing = held & nan_r
        liquidate = np.zeros(N, dtype=bool)
        if missing.any():
            liquidate = self._classify_missing(ctx, day, events, w_post, missing)
        self._close_halt_episodes(ctx, day, events, liquidate, nan_r, held)
        if liquidate.any():
            x = x + self._liquidate_delisted(ctx, day, state, events, w_post, r, liquidate, costs)
        return r, liquidate, x

    @staticmethod
    def _classify_missing(
        ctx: _RunContext, day: _Day, events: _EventLog, w_post: np.ndarray, missing: np.ndarray
    ) -> np.ndarray:
        """Halt or delisting for held names with no close on ``t+1`` (module
        docstring): the vendor date if covered, else the causal G-day gap rule --
        never a later price.  Opens / extends halt episodes; returns the
        delisted mask."""
        p1 = int(ctx.md_pos[day.i + 1])
        lp = ctx.last_close_md[p1]
        covered = ~np.isnan(ctx.vendor_pos)
        by_vendor = covered & (ctx.vendor_pos < p1)
        by_gap = ~covered & np.isfinite(lp) & (p1 - lp >= ctx.gap_days)
        liquidate = missing & (by_vendor | by_gap)
        frozen = missing & ~liquidate
        events.n_halted += int(frozen.sum())
        for j in np.where(frozen)[0]:
            tk = ctx.tickers[int(j)]
            ep = events.halts_open.get(tk)
            if ep is None:
                ep = {"ticker": tk, "start": day.t1, "weight": float(w_post[j]), "days": 0}
                events.halts_open[tk] = ep
            ep["days"] += 1
        return liquidate

    @staticmethod
    def _close_halt_episodes(
        ctx: _RunContext,
        day: _Day,
        events: _EventLog,
        liquidate: np.ndarray,
        nan_r: np.ndarray,
        held: np.ndarray,
    ) -> None:
        for tk in list(events.halts_open):
            j = ctx.tickers.get_loc(tk)
            if liquidate[j]:
                events.halts.append({**events.halts_open.pop(tk), "end": day.t1, "outcome": "delisted"})
            elif not nan_r[j] or not held[j]:
                events.halts.append({**events.halts_open.pop(tk), "end": day.t1, "outcome": "resumed"})

    def _liquidate_delisted(
        self,
        ctx: _RunContext,
        day: _Day,
        state: _BookState,
        events: _EventLog,
        w_post: np.ndarray,
        r: np.ndarray,
        liquidate: np.ndarray,
        costs: _Costs,
    ) -> np.ndarray:
        """Delisting: apply ``cfg.cost.delisting_haircut`` against the position's
        sign (a loss for longs and shorts alike -- conservative; written into
        ``r``), force-liquidate the remaining value at spread + explicit fees
        (continuous venue, no impact), add the cost to ``costs`` and record the
        event.  Returns the liquidation trade vector (booked on this decision row)."""
        tickers, nav = ctx.tickers, state.nav
        r[liquidate] = ctx.haircut * np.sign(w_post[liquidate])
        x_liq = np.zeros(ctx.n_names)
        x_liq[liquidate] = -w_post[liquidate] * (1.0 + r[liquidate])
        liq_cf = cost_frame(
            self._cm.estimate(
                pd.Series(x_liq * nav, index=tickers), day.snap_exec, venue=VENUE_CONTINUOUS, streak=None
            ),
            tickers,
        )
        liq_cf["impact"] = 0.0
        for j in np.where(liquidate)[0]:
            events.delistings.append(
                {
                    "date": day.t1,
                    "ticker": tickers[int(j)],
                    "weight": float(w_post[j]),
                    "haircut_return": float(r[j]),
                    "liquidation_usd": float(abs(x_liq[j]) * nav),
                    "cost_usd": float(liq_cf.iloc[j].sum()),
                }
            )
        costs.spread += float(liq_cf["spread"].sum())
        costs.fees += float(liq_cf["fees"].sum())
        return x_liq

    # ----------------------------------------------------------------- step 5: accounting
    @staticmethod
    def _accounting_step(
        ctx: _RunContext,
        day: _Day,
        state: _BookState,
        rec: _Recorder,
        w_post: np.ndarray,
        x: np.ndarray,
        r: np.ndarray,
        liquidate: np.ndarray,
        costs: _Costs,
    ) -> bool:
        """NAV identity, recording, and drift to the pre-trade weights of
        ``t+1`` (cash absorbs costs).  Returns False when NAV hits zero (the
        run stops and the recorder is truncated)."""
        nav = state.nav
        g_ret = float(w_post @ r)
        nav1 = nav * (1.0 + g_ret) - costs.total
        n_ret = nav1 / nav - 1.0
        exv = (
            float(state.last_risk.portfolio_variance(pd.Series(w_post, index=ctx.tickers)))
            if state.last_risk is not None
            else None
        )
        rec.record(day.i, w_post, x, costs, nav, nav1, g_ret, n_ret, exv)
        w_next = w_post * (1.0 + r) * nav / nav1 if nav1 > 0 else np.zeros(ctx.n_names)
        w_next[liquidate] = 0.0
        state.w_pre = w_next
        state.nav = nav1
        if nav1 <= 0:
            log.error("NAV <= 0 on %s; stopping", day.t1)
            rec.truncate(day.i + 1)
            return False
        if ctx.progress and day.i % 50 == 0:
            log.info("%s nav=%.0f gross=%.4f to=%.3f", day.t.date(), nav1, g_ret, rec.TO[day.i])
        return True

    # ----------------------------------------------------------------- step 6: diagnostics
    def _result(
        self, ctx: _RunContext, spec: BacktestSpec, events: _EventLog, rec: _Recorder, t_start: float
    ) -> BacktestResult:
        T = rec.T
        dates = ctx.dates[: T + 1]
        dec = dates[:T]
        real = dates[1 : T + 1]
        tickers = ctx.tickers
        res_obj = BacktestResult(
            nav=pd.Series(rec.navs, index=dates, name="nav"),
            returns=pd.DataFrame(
                {"gross": rec.gross, "net": rec.net, "gross_after_borrow": rec.gross - rec.borrow_ret},
                index=real,
            ),
            weights=pd.DataFrame(rec.W, index=dec, columns=tickers),
            trades=pd.DataFrame(rec.X, index=dec, columns=tickers),
            costs=pd.DataFrame(rec.C, index=dec, columns=COST_COLUMNS),
            exante_var=pd.Series(rec.EXV, index=dec, name="exante_var_daily"),
            turnover=pd.Series(rec.TO, index=dec, name="turnover"),
            transfer_coefficient=pd.Series(rec.TC, index=dec, name="transfer_coefficient"),
            periods_per_year=self.cfg.trading_days_per_year,
            diagnostics={
                "delistings": events.delistings,
                "halts": events.all_halts(),
                "n_halted_name_days": events.n_halted,
                "solve_log": events.solve_log,
                "n_failed_solves": events.n_failed,
                "failed_solves": events.failed_solves,
                "n_rebalance_solves": len(events.solve_log),
                "n_exante_vol_alerts": len(events.exante_vol_alerts),
                "exante_vol_alerts": events.exante_vol_alerts,
                "n_data_error_returns": len(self._bad_prices),
                "data_error_returns": self._bad_prices[:100],
                "run_time_sec": time.perf_counter() - t_start,
                "rebalance_every": spec.rebalance_every,
                "venue": spec.venue,
                "signal_lag_days": ctx.lag,
                "n_no_signal_days": events.n_no_signal,
                "forced_exits": events.forced_log,
                "n_ineligible_name_days": events.n_ineligible_days,
                # sum over solves of OptimizationResult.n_out_of_model: names outside the
                # risk model still held *after* the solve (forced exits in progress); names
                # with alpha but no position are excluded, not counted (n_excluded_alpha)
                "n_out_of_model_name_days": events.n_out_of_model,
            },
        )
        if events.n_failed:
            log.warning("backtest: %d failed solves (see diagnostics['failed_solves'])", events.n_failed)
        if events.exante_vol_alerts:
            log.warning(
                "backtest: %d rebalances with ex-ante vol above %.0f%% (see diagnostics['exante_vol_alerts'])",
                len(events.exante_vol_alerts),
                100 * float(getattr(self.cfg.validation, "max_exante_vol", 2.0)),
            )
        if events.n_out_of_model:
            log.warning(
                "backtest: %d name-days with alpha/position outside the risk-model universe "
                "(priced at 90th-pct specific variance, zero factor exposure)",
                events.n_out_of_model,
            )
        return res_obj


def _solve_log_entry(day: _Day, res: Any) -> dict[str, object]:
    """One ``diagnostics["solve_log"]`` row from an :class:`OptimizationResult`."""
    return {
        "date": day.t,
        "signal_date": day.sig_date,
        "status": res.status,
        "solve_time": res.solve_time,
        "binding": res.binding_constraints,
        "vol_scalar": res.vol_scalar,
        "expected_alpha": res.expected_alpha,
        "expected_risk": res.expected_risk,
        "expected_cost": res.expected_cost,
        "notes": res.notes,
        "n_out_of_model": getattr(res, "n_out_of_model", 0),
        # sizing (risk_aversion_mode); absent on optimizers without them
        "risk_aversion": float(getattr(res, "risk_aversion", float("nan"))),
        "vol_target_binding": str(getattr(res, "vol_target_binding", "") or ""),
        "n_solves": int(getattr(res, "n_solves", 0) or 0),
        "n_excluded_alpha": int(getattr(res, "n_excluded_alpha", 0) or 0),
    }
