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

Cost model: any :class:`qts.core.costs.CostModel` (``snapshot`` /
``estimate`` / ``holding_cost_usd``).  A duck-typed legacy model without
``snapshot`` (a test double with ``snapshot_from_market`` returning a dict and
``estimate`` returning a DataFrame) is wrapped once, at construction, by
:class:`_LegacyCostModel`; there is no silent fallback between paths.

Delistings: a held name whose return on ``t+1`` is NaN earns
``cfg.cost.delisting_haircut`` (default -30%, against the position's sign) on
its final day, is force-liquidated at spread + explicit fees (continuous venue,
booked as a trade on decision row ``t`` and counted in turnover) and the event
is recorded in ``BacktestResult.diagnostics["delistings"]``.  Names with no
price on ``t`` are placed on the optimizer's no-trade list.

Eligibility: ``md.eligible`` (wide bool panel, if present) or the
``eligible_fn(date) -> Index`` kwarg marks the investable universe as of a
date.  An ineligible name gets expected return 0; if it is not held it goes on
the no-trade list (nothing can be opened), and if it *is* held it enters a
forced-exit set that the optimizer receives as ``force_to_zero`` with the
allowed ``|w|`` shrinking linearly to 0 over
``cfg.portfolio.exit_ineligible_days`` days (recorded in
``diagnostics["forced_exits"]``).

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
from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Any

import numpy as np
import pandas as pd

from qts.backtest import metrics as M
from qts.core.config import SystemConfig
from qts.core.costs import VENUE_CONTINUOUS, VENUE_ROUTED, CostEstimate, LiquiditySnapshot
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput, Wide
from qts.portfolio.costs import Snapshot, align_snapshot, snapshot_from_market
from qts.portfolio.diagnostics import effective_holding_days, realized_vs_exante_risk
from qts.portfolio.optimizer import SinglePeriodOptimizer

log = logging.getLogger(__name__)

COST_COLUMNS = ["spread", "impact", "fees", "borrow", "total"]


@dataclass
class BacktestResult:
    nav: pd.Series  # indexed by date (first entry = nav0 on dates[0])
    returns: pd.DataFrame  # columns: gross, net, gross_after_borrow (indexed by realisation date)
    weights: Wide  # post-trade weights by decision date
    trades: Wide  # weight changes by decision date
    costs: pd.DataFrame  # USD by decision date: spread, impact, fees, borrow, total
    diagnostics: dict[str, object] = field(default_factory=dict)
    exante_var: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    turnover: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    transfer_coefficient: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    periods_per_year: int = 252

    # ----------------------------------------------------------------- views
    @property
    def gross_returns(self) -> pd.Series:
        return self.returns["gross"]

    @property
    def net_returns(self) -> pd.Series:
        return self.returns["net"]

    @property
    def gross_after_borrow_returns(self) -> pd.Series:
        """``gross - borrow/nav_pre`` (module docstring); computed from ``costs``
        and ``nav`` for results built without the column."""
        if "gross_after_borrow" in self.returns:
            return self.returns["gross_after_borrow"]
        n = min(len(self.costs), len(self.returns))
        nav_pre = self.nav.reindex(self.costs.index).to_numpy(dtype=float)[:n]
        borrow = self.costs["borrow"].to_numpy(dtype=float)[:n] if "borrow" in self.costs else np.zeros(n)
        out = self.returns["gross"].astype(float).copy()
        out.iloc[:n] = out.to_numpy(dtype=float)[:n] - borrow / nav_pre
        return out.rename("gross_after_borrow")

    def cost_sensitivity(
        self,
        cost_bps_per_side: float | None = None,
        multipliers: tuple[float, ...] = (0.5, 1, 2, 3),
    ) -> pd.DataFrame:
        """:func:`qts.backtest.metrics.cost_sensitivity` on ``gross_after_borrow``
        (borrow charged as realised, trading cost scaled); default cost = the
        realised ``avg_cost_bps_per_side``."""
        r = self.gross_after_borrow_returns.dropna()
        c = self.summary()["avg_cost_bps_per_side"] if cost_bps_per_side is None else cost_bps_per_side
        c = float(c) if np.isfinite(c) else 10.0
        return M.cost_sensitivity(
            r, self.turnover_on_return_dates().reindex(r.index), c, multipliers, self.periods_per_year
        )

    @property
    def cost_bps_of_nav(self) -> pd.Series:
        nav_pre = self.nav.reindex(self.costs.index)
        return (self.costs["total"] / nav_pre * 1e4).rename("cost_bps_of_nav")

    @property
    def traded_usd(self) -> pd.Series:
        nav_pre = self.nav.reindex(self.trades.index)
        return (self.trades.abs().sum(axis=1) * nav_pre).rename("traded_usd")

    def summary(self) -> dict[str, float]:
        q = self.periods_per_year
        g, n = self.gross_returns.dropna(), self.net_returns.dropna()
        gb = self.gross_after_borrow_returns.dropna()
        traded = float(self.traded_usd.sum())
        trading_cost = float((self.costs["spread"] + self.costs["impact"] + self.costs["fees"]).sum())
        rv = (
            realized_vs_exante_risk(g, self.exante_var.shift(1).reindex(g.index), q)
            if len(self.exante_var)
            else {"ratio": float("nan")}
        )
        gross_w = self.weights.abs().sum(axis=1)
        out = {
            "start": str(self.nav.index[0].date()) if len(self.nav) else "",
            "end": str(self.nav.index[-1].date()) if len(self.nav) else "",
            "n_days": len(n),
            "final_nav": float(self.nav.iloc[-1]) if len(self.nav) else float("nan"),
            "cagr_gross": M.annualized_return(g, q),
            "cagr_net": M.annualized_return(n, q),
            "ann_vol_gross": M.annualized_vol(g, q),
            "ann_vol_net": M.annualized_vol(n, q),
            "sharpe_gross": M.annualized_sharpe(g, q),
            "sharpe_net": M.annualized_sharpe(n, q),
            "sharpe_net_lo": M.annualized_sharpe(n, q, lo_adjust=True),
            "sharpe_t_stat_net": M.sharpe_t_stat(n),
            "newey_west_t_net": M.newey_west_tstat(n),
            "max_drawdown_net": M.max_drawdown(n),
            "calmar_net": M.calmar(n, q),
            "sortino_net": M.sortino(n, q),
            "hit_rate_net": M.hit_rate(n),
            "avg_gross_exposure": float(gross_w.mean()) if len(gross_w) else float("nan"),
            "avg_net_exposure": float(self.weights.sum(axis=1).mean()) if len(self.weights) else float("nan"),
            "avg_daily_turnover": float(self.turnover.mean()) if len(self.turnover) else float("nan"),
            "effective_holding_days": effective_holding_days(self.turnover, gross_w),
            "avg_cost_bps_per_side": trading_cost / traded * 1e4 if traded > 0 else float("nan"),
            "avg_daily_cost_bps_of_nav": float(self.cost_bps_of_nav.mean())
            if len(self.costs)
            else float("nan"),
            "total_costs_usd": float(self.costs["total"].sum()),
            "transfer_coefficient_mean": float(self.transfer_coefficient.mean())
            if len(self.transfer_coefficient.dropna())
            else float("nan"),
            "realized_vs_exante_vol_ratio": rv["ratio"],
            # per-side breakeven net of borrow (borrow is not a per-side cost; audit 2 #1)
            "breakeven_cost_bps": M.breakeven_cost_bps(gb, self.turnover_on_return_dates().reindex(gb.index)),
            "n_delistings": len(self.diagnostics.get("delistings", [])),
            "n_failed_solves": int(self.diagnostics.get("n_failed_solves", 0)),
            **self.sizing_summary(),
        }
        return out

    def sizing_summary(self) -> dict[str, float]:
        """From ``diagnostics["solve_log"]``: ``vol_target_caps_share`` = share of the
        target-vol solves (non-empty ``vol_target_binding``) whose book was capped below
        the vol target by the gross / position / ADV limits (``"caps"``);
        ``mean_risk_aversion`` = mean annual gamma of the returned books.  NaN when the
        log has no such entries (fixed gamma without a search, or no solves)."""
        log_ = self.diagnostics.get("solve_log") or []
        entries = [e for e in log_ if isinstance(e, dict)] if isinstance(log_, list) else []
        binding = [str(e.get("vol_target_binding") or "") for e in entries]
        searched = [b for b in binding if b]
        gam = np.array([float(e.get("risk_aversion", np.nan)) for e in entries], dtype=float)
        gam = gam[np.isfinite(gam)]
        return {
            "vol_target_caps_share": float(np.mean([b == "caps" for b in searched]))
            if searched
            else float("nan"),
            "mean_risk_aversion": float(gam.mean()) if len(gam) else float("nan"),
        }

    def turnover_on_return_dates(self) -> pd.Series:
        """Turnover on decision date t re-indexed to the realisation date t+1
        (positionally aligned with ``returns``)."""
        n = min(len(self.turnover), len(self.returns))
        return pd.Series(self.turnover.values[:n], index=self.returns.index[:n], name="turnover")


def cost_frame(est, tickers: pd.Index) -> pd.DataFrame:
    """Normalise a cost estimate into ``spread``/``impact``/``fees`` USD per ticker.

    Accepted: a :class:`CostEstimate`; ``None`` (zero cost); a Series (booked
    as impact); a DataFrame / dict with columns ``spread``, ``impact``,
    ``fees`` (aliases ``<k>_usd`` / ``<k>_cost``) or a single ``total`` column
    (booked as impact).  Anything else raises ``ValueError`` naming the
    columns, and so does any non-finite cost -- a renamed column or a NaN input
    must never make the backtest silently free.
    """
    out = pd.DataFrame(0.0, index=tickers, columns=["spread", "impact", "fees"])
    if est is None:
        return out
    if isinstance(est, CostEstimate):
        f = est.frame
        for k in ("spread", "impact", "fees"):
            out[k] = f[k + "_usd"].reindex(tickers).fillna(0.0).astype(float)
        return out
    if isinstance(est, pd.Series):
        v = est.reindex(tickers).astype(float)
        missing = v.index[~np.isfinite(v.values)]
        if len(missing):
            raise ValueError(f"cost estimate has non-finite values for {list(missing[:5])}")
        out["impact"] = v
        return out
    if isinstance(est, dict):
        est = pd.DataFrame(est)
    if isinstance(est, pd.DataFrame):
        cols = {c.lower(): c for c in est.columns}
        matched = []
        for k in ("spread", "impact", "fees"):
            for alias in (k, k + "_usd", k + "_cost"):
                if alias in cols:
                    out[k] = est[cols[alias]].reindex(tickers).astype(float)
                    matched.append(cols[alias])
                    break
        if not matched:
            if "total" in cols:
                out["impact"] = est[cols["total"]].reindex(tickers).astype(float)
                matched.append(cols["total"])
            else:
                raise ValueError(
                    "estimate() returned no recognised cost column "
                    f"(expected spread/impact/fees[_usd|_cost] or total); got {list(est.columns)}"
                )
        bad = {
            c: list(out.index[~np.isfinite(out[c].values)][:5])
            for c in out.columns
            if not np.isfinite(out[c].values).all()
        }
        if bad:
            raise ValueError(f"estimate() returned non-finite costs (source columns {matched}): {bad}")
        return out
    raise TypeError(f"unsupported cost estimate type {type(est)}")


class _LegacyCostModel:
    """Adapter giving a duck-typed pre-protocol cost model (no ``snapshot``
    method) the :class:`~qts.core.costs.CostModel` interface the simulator uses.

    * ``snapshot``: the model's ``snapshot_from_market(md, t)`` dict if it has
      one, else the built-in builder; reindexed to ``md.tickers`` with
      :func:`align_snapshot` (dict kept) and ``date`` set.
    * ``estimate``: the model's ``estimate(trades, snap, [venue], [streak])``
      (only the keyword arguments its signature accepts), normalised by
      :func:`cost_frame` into a :class:`CostEstimate`.
    * ``holding_cost_usd``: the model's method if present, else
      ``max(-w, 0) * nav * snapshot["borrow_fee"] / trading_days``.
    """

    def __init__(self, model: Any, cfg: SystemConfig, adv_window: int, vol_window: int):
        self.model = model
        self.cfg = cfg
        self.adv_window = adv_window
        self.vol_window = vol_window
        try:
            self._params = set(inspect.signature(model.estimate).parameters)
        except (TypeError, ValueError):
            self._params = set()

    def snapshot(self, md: MarketData, t: pd.Timestamp) -> Snapshot:
        fn = getattr(self.model, "snapshot_from_market", None)
        raw = (
            fn(md, t)
            if callable(fn)
            else snapshot_from_market(md, t, self.adv_window, self.vol_window, self.cfg.cost).to_dict()
        )
        out = align_snapshot(dict(raw), md.tickers, self.cfg.cost)
        out.setdefault("date", pd.Timestamp(t))
        return out

    def estimate(
        self,
        trades_usd: pd.Series,
        snap: Snapshot,
        venue: str | None = None,
        streak: pd.Series | None = None,
        prior_streak_usd: pd.Series | None = None,
    ) -> CostEstimate:
        kw: dict[str, object] = {}
        if "venue" in self._params and venue is not None:
            kw["venue"] = venue
        if "streak" in self._params and streak is not None:
            kw["streak"] = streak
        if "prior_streak_usd" in self._params and prior_streak_usd is not None:
            kw["prior_streak_usd"] = prior_streak_usd
        cf = cost_frame(self.model.estimate(trades_usd, snap, **kw), trades_usd.index)
        return CostEstimate.from_components(
            trades_usd.index,
            venue or VENUE_ROUTED,
            trades_usd,
            cf["spread"].to_numpy(),
            cf["impact"].to_numpy(),
            cf["fees"].to_numpy(),
        )

    def holding_cost_usd(self, w_post: pd.Series, nav: float, snap: Snapshot) -> pd.Series:
        fn = getattr(self.model, "holding_cost_usd", None)
        if callable(fn):
            return pd.Series(fn(w_post, nav, snap), dtype=float).reindex(w_post.index).fillna(0.0)
        fee = (
            pd.Series(snap["borrow_fee"])
            .reindex(w_post.index)
            .fillna(self.cfg.cost.default_borrow_fee_annual)
        )
        return (-w_post).clip(lower=0.0) * nav * fee / self.cfg.trading_days_per_year


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

    # ----------------------------------------------------------------- run
    def run(
        self,
        md: MarketData,
        dates: pd.DatetimeIndex,
        alpha_fn: Callable[[pd.Timestamp], AlphaOutput],
        risk_fn: Callable[[pd.Timestamp, pd.Series | None], RiskModelOutput],
        optimizer: SinglePeriodOptimizer,
        nav0: float = 1e6,
        rebalance_every: int = 1,
        venue: str = VENUE_ROUTED,
        neutral_styles: list[str] | None = None,
        warm_start_weights: pd.Series | None = None,
        progress: bool = False,
        eligible_fn: Callable[[pd.Timestamp], pd.Index] | None = None,
        exit_ineligible_days: int | None = None,
        signal_lag_days: int | None = None,
    ) -> BacktestResult:
        """``eligible_fn(date)`` returns the tickers investable as of ``date``;
        when ``md.eligible`` (wide bool panel) is present it takes precedence
        and ``eligible_fn`` is ignored.  ``exit_ineligible_days`` overrides
        ``cfg.portfolio.exit_ineligible_days`` (linear unwind horizon).
        ``venue`` in {"routed" (default), "auction", "continuous"} prices every
        trade (and is passed to the optimizer's cost terms).
        ``signal_lag_days`` overrides ``cfg.execution.signal_lag_days`` (module
        docstring, step 1)."""
        t_start = time.perf_counter()
        dates = pd.DatetimeIndex(dates)
        if len(dates) < 2:
            raise ValueError("need at least two dates")
        if not dates.isin(md.dates).all():
            raise ValueError("dates must be a subset of md.dates")
        lag = int(
            signal_lag_days
            if signal_lag_days is not None
            else getattr(self.cfg.execution, "signal_lag_days", 0)
        )
        if lag < 0:
            raise ValueError("signal_lag_days must be >= 0")
        tickers = md.tickers
        N = len(tickers)
        md_pos = md.dates.get_indexer(dates)
        close = md.close.loc[dates]
        R = close.pct_change(fill_method=None)  # return realised on row date
        price_ok = close.notna()
        elig_panel = getattr(md, "eligible", None)
        elig_md: np.ndarray | None = None  # rows aligned with md.dates
        if elig_panel is not None:
            elig_md = elig_panel.reindex(index=md.dates, columns=tickers).fillna(False).astype(bool).values
            eligible_fn = None
        exit_days = int(
            exit_ineligible_days
            if exit_ineligible_days is not None
            else getattr(self.cfg.portfolio, "exit_ineligible_days", 5)
        )
        exit_days = max(exit_days, 1)
        forced: dict[str, dict[str, float]] = {}  # ticker -> {"w0": |w| at entry, "k": days elapsed}
        forced_log: list[dict[str, object]] = []
        n_out_of_model = 0
        n_ineligible_days = 0
        n_no_signal = 0
        haircut = float(self.cfg.cost.delisting_haircut)
        snap_cache: dict[pd.Timestamp, Snapshot] = {}

        def snap_at(d: pd.Timestamp) -> Snapshot:
            if d not in snap_cache:
                if len(snap_cache) > lag + 2:  # keep only the window still needed
                    snap_cache.pop(next(iter(snap_cache)))
                snap_cache[d] = self._snapshot(md, d)
            return snap_cache[d]

        nav = float(nav0)
        w_pre = np.zeros(N)
        if warm_start_weights is not None:
            w_pre = pd.Series(warm_start_weights, dtype=float).reindex(tickers).fillna(0.0).values

        T = len(dates) - 1
        W = np.zeros((T, N))
        X = np.zeros((T, N))
        C = np.zeros((T, 5))
        EXV = np.full(T, np.nan)
        TO = np.zeros(T)
        TC = np.full(T, np.nan)
        navs = np.empty(T + 1)
        navs[0] = nav
        gross = np.empty(T)
        net = np.empty(T)
        streak = np.zeros(N)  # consecutive same-direction trading days per name
        streak_usd = np.zeros(N)  # |USD| traded in the current streak, through the last trade
        last_sign = np.zeros(N)
        borrow_ret = np.empty(T)
        delistings: list[dict[str, object]] = []
        solve_log: list[dict[str, object]] = []
        n_failed = 0
        last_risk: RiskModelOutput | None = None
        last_alpha: AlphaOutput | None = None
        # Signal half-life (handbook fix #5): AlphaOutput.half_life_days drives the
        # optimizer's Garleanu-Pedersen shrink; passed only when solve() accepts it.
        solve_accepts_hl = "alpha_half_life_days" in inspect.signature(optimizer.solve).parameters

        for i in range(T):
            t = dates[i]
            t1 = dates[i + 1]
            snap_exec = snap_at(t)
            p_sig = int(md_pos[i]) - lag
            sig_date: pd.Timestamp | None = md.dates[p_sig] if p_sig >= 0 else None
            elig_date = sig_date if sig_date is not None else t
            alive = price_ok.iloc[i].values
            # ---- eligibility as of the signal date
            if elig_md is not None:
                eligible = elig_md[md.dates.get_loc(elig_date)] & alive
            elif eligible_fn is not None:
                eligible = tickers.isin(pd.Index(eligible_fn(elig_date))) & alive
            else:
                eligible = alive.copy()
            inel = alive & ~eligible
            n_ineligible_days += int(inel.sum())
            held = np.abs(w_pre) > 1e-12
            for j in np.where(inel & held)[0]:
                tk = tickers[j]
                if tk not in forced:
                    forced[tk] = {"w0": float(abs(w_pre[j])), "k": 0.0}
                    forced_log.append({"date": t, "ticker": tk, "weight": float(w_pre[j])})
            for tk in list(forced):
                j = tickers.get_loc(tk)
                if eligible[j] or not held[j]:
                    forced.pop(tk)  # back in the universe, or already flat
            force_caps: dict[str, float] = {}
            for tk, st in forced.items():
                st["k"] += 1.0
                force_caps[tk] = max(st["w0"] * (1.0 - st["k"] / exit_days), 0.0)
            dead = list(tickers[~alive]) + list(tickers[inel & ~held])
            w_post = w_pre.copy()
            if i % rebalance_every == 0 and sig_date is None:
                n_no_signal += 1
            elif i % rebalance_every == 0:
                last_alpha = alpha_fn(sig_date)
                exp_ret = pd.Series(last_alpha.expected_return, dtype=float).reindex(tickers)
                exp_ret[~alive] = 0.0
                exp_ret[inel] = 0.0
                exp_ret = exp_ret.fillna(0.0)
                last_risk = risk_fn(sig_date, exp_ret)
                solve_kw: dict[str, Any] = {}
                hl = getattr(last_alpha, "half_life_days", None)
                if solve_accepts_hl and hl is not None:
                    solve_kw["alpha_half_life_days"] = float(hl)
                res = optimizer.solve(
                    alpha=exp_ret,
                    risk=last_risk,
                    w_current=pd.Series(w_pre, index=tickers),
                    nav=nav,
                    snapshot=self._optimizer_snapshot(snap_at(sig_date), snap_exec),
                    venue=venue,
                    neutral_styles=neutral_styles,
                    no_trade=dead,
                    horizon_days=getattr(last_alpha, "horizon_days", None) or optimizer.horizon_days,
                    force_to_zero=force_caps or None,
                    **solve_kw,
                )
                if res.status not in ("optimal", "optimal_inaccurate"):
                    n_failed += 1
                    log.warning(
                        "optimizer solve failed on %s: status=%s (holding current book)", t.date(), res.status
                    )
                n_out_of_model += int(getattr(res, "n_out_of_model", 0) or 0)
                w_post = np.array(res.w_target.reindex(tickers).fillna(0.0).values, dtype=float, copy=True)
                w_post[~alive] = w_pre[~alive]  # cannot trade names without a price
                TC[i] = res.transfer_coefficient
                solve_log.append(
                    {
                        "date": t,
                        "signal_date": sig_date,
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
                )
            x = w_post - w_pre
            x[np.abs(x) < 1e-10] = 0.0
            w_post = w_pre + x

            # ---- costs in USD on the execution date's liquidity, at the requested venue;
            # same-direction streak per name drives multi-day impact persistence
            sgn = np.sign(x)
            same = (sgn != 0) & (sgn == last_sign)
            streak = np.where(sgn != 0, np.where(same, streak + 1.0, 1.0), 0.0)
            prior_usd = np.where(same, streak_usd, 0.0)  # Q_{k-1}: earlier days of this streak
            streak_usd = np.where(sgn != 0, prior_usd + np.abs(x) * nav, 0.0)
            last_sign = sgn
            est_kw: dict[str, Any] = {"venue": venue, "streak": pd.Series(streak, index=tickers)}
            if self._pass_prior_streak:
                est_kw["prior_streak_usd"] = pd.Series(prior_usd, index=tickers)
            est = self._cm.estimate(pd.Series(x * nav, index=tickers), snap_exec, **est_kw)
            cf = cost_frame(est, tickers)
            borrow = self._cm.holding_cost_usd(pd.Series(w_post, index=tickers), nav, snap_exec)
            c_spread = float(cf["spread"].sum())
            c_impact = float(cf["impact"].sum())
            c_fees = float(cf["fees"].sum())
            c_borrow = float(np.nansum(np.asarray(borrow, dtype=float)))
            c_total = c_spread + c_impact + c_fees + c_borrow

            # ---- returns t -> t+1
            r = R.iloc[i + 1].values.copy()
            held = w_post != 0
            nan_r = ~np.isfinite(r)
            liquidate = held & nan_r
            r[nan_r] = 0.0
            if liquidate.any():
                # Delisting / permanent halt: the name has no close on t+1.  Apply
                # cfg.cost.delisting_haircut against the position's sign (a loss
                # for longs and shorts alike -- conservative), force-liquidate the
                # remaining value at spread + explicit fees (continuous venue, no
                # impact), book the exit as a trade on this decision row (with its
                # cost) and record it in diagnostics["delistings"].
                r[liquidate] = haircut * np.sign(w_post[liquidate])
                x_liq = np.zeros(N)
                x_liq[liquidate] = -w_post[liquidate] * (1.0 + r[liquidate])
                liq_cf = cost_frame(
                    self._cm.estimate(
                        pd.Series(x_liq * nav, index=tickers), snap_exec, venue=VENUE_CONTINUOUS, streak=None
                    ),
                    tickers,
                )
                liq_cf["impact"] = 0.0
                for j in np.where(liquidate)[0]:
                    delistings.append(
                        {
                            "date": t1,
                            "ticker": tickers[j],
                            "weight": float(w_post[j]),
                            "haircut_return": float(r[j]),
                            "liquidation_usd": float(abs(x_liq[j]) * nav),
                            "cost_usd": float(liq_cf.iloc[j].sum()),
                        }
                    )
                c_spread += float(liq_cf["spread"].sum())
                c_fees += float(liq_cf["fees"].sum())
                c_total = c_spread + c_impact + c_fees + c_borrow
                x = x + x_liq  # recorded in trades / turnover below
            g_ret = float(w_post @ r)
            nav1 = nav * (1.0 + g_ret) - c_total
            n_ret = nav1 / nav - 1.0

            # ---- record
            W[i] = w_post
            X[i] = x
            C[i] = (c_spread, c_impact, c_fees, c_borrow, c_total)
            TO[i] = float(np.abs(x).sum())
            if last_risk is not None:
                EXV[i] = float(last_risk.portfolio_variance(pd.Series(w_post, index=tickers)))
            gross[i] = g_ret
            net[i] = n_ret
            borrow_ret[i] = c_borrow / nav
            navs[i + 1] = nav1

            # ---- drift to pre-trade weights of t+1 (cash absorbs costs)
            w_next = w_post * (1.0 + r) * nav / nav1 if nav1 > 0 else np.zeros(N)
            w_next[liquidate] = 0.0
            w_pre = w_next
            nav = nav1
            if nav <= 0:
                log.error("NAV <= 0 on %s; stopping", t1)
                T_eff = i + 1
                W, X, C, EXV, TO, TC = W[:T_eff], X[:T_eff], C[:T_eff], EXV[:T_eff], TO[:T_eff], TC[:T_eff]
                navs, gross, net = navs[: T_eff + 1], gross[:T_eff], net[:T_eff]
                borrow_ret = borrow_ret[:T_eff]
                dates = dates[: T_eff + 1]
                T = T_eff
                break
            if progress and i % 50 == 0:
                log.info("%s nav=%.0f gross=%.4f to=%.3f", t.date(), nav, g_ret, TO[i])

        dec = dates[:T]
        real = dates[1 : T + 1]
        res_obj = BacktestResult(
            nav=pd.Series(navs, index=dates[: T + 1], name="nav"),
            returns=pd.DataFrame(
                {"gross": gross, "net": net, "gross_after_borrow": gross - borrow_ret}, index=real
            ),
            weights=pd.DataFrame(W, index=dec, columns=tickers),
            trades=pd.DataFrame(X, index=dec, columns=tickers),
            costs=pd.DataFrame(C, index=dec, columns=COST_COLUMNS),
            exante_var=pd.Series(EXV, index=dec, name="exante_var_daily"),
            turnover=pd.Series(TO, index=dec, name="turnover"),
            transfer_coefficient=pd.Series(TC, index=dec, name="transfer_coefficient"),
            periods_per_year=self.cfg.trading_days_per_year,
            diagnostics={
                "delistings": delistings,
                "solve_log": solve_log,
                "n_failed_solves": n_failed,
                "run_time_sec": time.perf_counter() - t_start,
                "rebalance_every": rebalance_every,
                "venue": venue,
                "signal_lag_days": lag,
                "n_no_signal_days": n_no_signal,
                "forced_exits": forced_log,
                "n_ineligible_name_days": n_ineligible_days,
                # sum over solves of OptimizationResult.n_out_of_model: names outside the
                # risk model still held *after* the solve (forced exits in progress); names
                # with alpha but no position are excluded, not counted (n_excluded_alpha)
                "n_out_of_model_name_days": n_out_of_model,
            },
        )
        if n_failed:
            log.warning("backtest: %d failed solves (see diagnostics['solve_log'])", n_failed)
        if n_out_of_model:
            log.warning(
                "backtest: %d name-days with alpha/position outside the risk-model universe "
                "(priced at 90th-pct specific variance, zero factor exposure)",
                n_out_of_model,
            )
        return res_obj
