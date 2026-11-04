"""Daily-bar backtest simulator (closing-auction rebalancer).

Timeline for each simulation date ``t`` (close of ``t``):

1. ``alpha_fn(t)`` and ``risk_fn(t, alpha)`` see only data ``<= t``.
2. The optimizer turns them into target weights ``w_post`` (fraction of the
   NAV at the close of ``t``, before costs).
3. Trades ``x = w_post - w_pre`` execute at the close of ``t`` (auction) and
   are charged ``cost_model.estimate(x * nav_t, snapshot_t)`` (spread, impact,
   explicit fees, in USD).
4. Positions earn close-to-close returns ``t -> t+1``; shorts pay one day of
   borrow on ``|w_post| * nav_t``.
5. NAV identity (exact by construction, tested):
   ``nav_{t+1} = nav_t * (1 + sum_i w_post_i r_{i,t+1}) - costs_t`` where
   ``costs_t`` is the USD total of spread + impact + fees + borrow booked on
   ``t``.  Pre-trade weights on ``t+1`` are the drifted post-trade weights
   re-based to ``nav_{t+1}`` (cash absorbs costs).

Delistings: a held name whose return on ``t+1`` is NaN earns
``cfg.cost.delisting_haircut`` (default -30%, against the position's sign) on
its final day, is force-liquidated at spread + explicit fees (booked as a trade
on decision row ``t`` and counted in turnover) and the event is recorded in
``BacktestResult.diagnostics["delistings"]``.  Names with no price on ``t``
are placed on the optimizer's no-trade list.

Eligibility: ``md.eligible`` (wide bool panel, if present) or the
``eligible_fn(t) -> Index`` kwarg marks the investable universe as of ``t``.
An ineligible name gets expected return 0; if it is not held it goes on the
no-trade list (nothing can be opened), and if it *is* held it enters a
forced-exit set that the optimizer receives as ``force_to_zero`` with the
allowed ``|w|`` shrinking linearly to 0 over
``cfg.portfolio.exit_ineligible_days`` days (recorded in
``diagnostics["forced_exits"]``).

Recorded panels are indexed by the decision date ``t``: ``weights`` (post-
trade), ``trades``, ``costs``, ``exante_var``; ``returns`` (gross/net) and
``nav`` are indexed by the date the return is realised (``t+1``).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from qts.backtest import metrics as M
from qts.core.config import SystemConfig
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput, Wide
from qts.portfolio.costs import CostTerms, Snapshot, align_snapshot, snapshot_from_market
from qts.portfolio.diagnostics import effective_holding_days, realized_vs_exante_risk
from qts.portfolio.optimizer import SinglePeriodOptimizer

log = logging.getLogger(__name__)

COST_COLUMNS = ["spread", "impact", "fees", "borrow", "total"]


@dataclass
class BacktestResult:
    nav: pd.Series                       # indexed by date (first entry = nav0 on dates[0])
    returns: pd.DataFrame                # columns: gross, net (indexed by realisation date)
    weights: Wide                        # post-trade weights by decision date
    trades: Wide                         # weight changes by decision date
    costs: pd.DataFrame                  # USD by decision date: spread, impact, fees, borrow, total
    diagnostics: Dict[str, object] = field(default_factory=dict)
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
    def cost_bps_of_nav(self) -> pd.Series:
        nav_pre = self.nav.reindex(self.costs.index)
        return (self.costs["total"] / nav_pre * 1e4).rename("cost_bps_of_nav")

    @property
    def traded_usd(self) -> pd.Series:
        nav_pre = self.nav.reindex(self.trades.index)
        return (self.trades.abs().sum(axis=1) * nav_pre).rename("traded_usd")

    def summary(self) -> Dict[str, float]:
        q = self.periods_per_year
        g, n = self.gross_returns.dropna(), self.net_returns.dropna()
        traded = float(self.traded_usd.sum())
        trading_cost = float((self.costs["spread"] + self.costs["impact"] + self.costs["fees"]).sum())
        rv = realized_vs_exante_risk(g, self.exante_var.shift(1).reindex(g.index), q) \
            if len(self.exante_var) else {"ratio": float("nan")}
        gross_w = self.weights.abs().sum(axis=1)
        out = {
            "start": str(self.nav.index[0].date()) if len(self.nav) else "",
            "end": str(self.nav.index[-1].date()) if len(self.nav) else "",
            "n_days": int(len(n)),
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
            "avg_daily_cost_bps_of_nav": float(self.cost_bps_of_nav.mean()) if len(self.costs) else float("nan"),
            "total_costs_usd": float(self.costs["total"].sum()),
            "transfer_coefficient_mean": float(self.transfer_coefficient.mean())
            if len(self.transfer_coefficient.dropna()) else float("nan"),
            "realized_vs_exante_vol_ratio": rv["ratio"],
            "breakeven_cost_bps": M.breakeven_cost_bps(g, self.turnover_on_return_dates().reindex(g.index)),
            "n_delistings": len(self.diagnostics.get("delistings", [])),
            "n_failed_solves": int(self.diagnostics.get("n_failed_solves", 0)),
        }
        return out

    def turnover_on_return_dates(self) -> pd.Series:
        """Turnover on decision date t re-indexed to the realisation date t+1
        (positionally aligned with ``returns``)."""
        n = min(len(self.turnover), len(self.returns))
        return pd.Series(self.turnover.values[:n], index=self.returns.index[:n], name="turnover")


class DailyBarSimulator:
    def __init__(self, cfg: SystemConfig, cost_model: CostTerms, adv_window: int = 63, vol_window: int = 63):
        self.cfg = cfg
        self.cost_model = cost_model
        self.adv_window = adv_window
        self.vol_window = vol_window

    # ------------------------------------------------------------- helpers
    # keys the cost model needs beyond the optimizer's SNAPSHOT_KEYS; passed through
    # (per-ticker Series reindexed, scalars untouched) instead of being dropped
    SNAPSHOT_PASSTHROUGH = ("venue", "date", "half_spread_bps", "adv_shares", "auction_share",
                            "borrow_fee_source", "nav")

    def _snapshot(self, md: MarketData, t: pd.Timestamp) -> Snapshot:
        """Point-in-time cost snapshot for ``t``.

        Uses ``cost_model.snapshot_from_market`` when the model has one (the
        ``qts.tcost`` facade), else the built-in :func:`snapshot_from_market`.
        Only the per-ticker Series the optimizer knows about go through
        :func:`align_snapshot`; ``venue``, ``date``, ``half_spread_bps``,
        ``adv_shares`` and ``auction_share`` are preserved so that ``estimate()``
        prices the requested venue with real spreads and the dated fee /
        auction-share schedules.  When neither path supplies a borrow datum and
        ``md.short_interest_ratio`` exists, the short-interest borrow-fee proxy
        replaces the flat GC default (placeholder until IBKR SLB history exists).
        """
        fn = getattr(self.cost_model, "snapshot_from_market", None)
        raw: Optional[Snapshot] = None
        if callable(fn):
            try:
                snap = fn(md, t)
                if isinstance(snap, dict) and "adv_usd" in snap:
                    raw = snap
            except Exception as e:  # fall back to the built-in snapshot
                if not getattr(self, "_warned_snapshot", False):
                    log.warning("cost_model.snapshot_from_market failed (%s); using built-in snapshot", e)
                    self._warned_snapshot = True
        if raw is None:
            raw = snapshot_from_market(md, t, self.adv_window, self.vol_window, self.cfg.cost)
            raw["date"] = pd.Timestamp(t)
            if md.borrow_fee is None and md.short_interest_ratio is not None:
                raw["borrow_fee"] = self._proxy_borrow_fee(md, t, md.tickers)
                raw["borrow_fee_source"] = "proxy"
        out = align_snapshot(raw, md.tickers, self.cfg.cost)
        if "half_spread_bps" not in raw and "half_spread" in out:
            out["half_spread_bps"] = out["half_spread"] * 1e4
        for k in self.SNAPSHOT_PASSTHROUGH:
            v = raw.get(k)
            if v is None:
                continue
            if isinstance(v, pd.Series):
                v = v.reindex(md.tickers)
                if k == "half_spread_bps":
                    v = v.astype(float).fillna(out["half_spread"] * 1e4)
                elif k == "adv_shares":
                    v = v.astype(float)
                    med = v.median()
                    v = v.fillna(med if np.isfinite(med) else np.nan)
                elif k == "venue":
                    v = v.fillna("auction")
            out[k] = v
        out.setdefault("date", pd.Timestamp(t))
        return out

    def _proxy_borrow_fee(self, md: MarketData, t: pd.Timestamp, tickers: pd.Index) -> pd.Series:
        from qts.tcost.borrow import borrow_fee_proxy
        si = md.short_interest_ratio.loc[:t]
        default = self.cfg.cost.default_borrow_fee_annual
        if si.empty:
            return pd.Series(default, index=tickers, dtype=float)
        si = si.iloc[-1].reindex(tickers).astype(float)
        mc = md.market_cap.loc[:t].iloc[-1].reindex(tickers).astype(float) if md.market_cap is not None else None
        if not getattr(self, "_warned_proxy", False):
            log.warning("md.borrow_fee absent: shorts financed with the short-interest borrow-fee PROXY "
                        "(qts.tcost.borrow.borrow_fee_proxy; placeholder until IBKR SLB history exists)")
            self._warned_proxy = True
        return borrow_fee_proxy(si, cfg=self.cfg.cost, market_cap=mc).where(si.notna(), default)

    @staticmethod
    def _cost_frame(est, tickers: pd.Index) -> pd.DataFrame:
        """Normalise whatever ``estimate`` returned into spread/impact/fees USD.

        Accepted: ``None`` (zero cost), a Series (booked as impact), or a
        DataFrame / dict with columns ``spread``, ``impact``, ``fees`` (aliases
        ``<k>_usd`` / ``<k>_cost``) or a single ``total`` column (booked as
        impact).  Anything else raises ``ValueError`` naming the columns, and so
        does any non-finite cost -- a renamed column or a NaN input must never
        make the backtest silently free.
        """
        out = pd.DataFrame(0.0, index=tickers, columns=["spread", "impact", "fees"])
        if est is None:
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
                    raise ValueError("estimate() returned no recognised cost column "
                                     f"(expected spread/impact/fees[_usd|_cost] or total); got {list(est.columns)}")
            bad = {c: list(out.index[~np.isfinite(out[c].values)][:5]) for c in out.columns
                   if not np.isfinite(out[c].values).all()}
            if bad:
                raise ValueError(f"estimate() returned non-finite costs (source columns {matched}): {bad}")
            return out
        raise TypeError(f"unsupported cost estimate type {type(est)}")

    def _borrow_usd(self, w_post: pd.Series, nav: float, snap: Snapshot) -> pd.Series:
        fn = getattr(self.cost_model, "holding_cost_usd", None)
        if callable(fn):
            try:
                return pd.Series(fn(w_post, nav, snap), dtype=float).reindex(w_post.index).fillna(0.0)
            except Exception as e:
                if not getattr(self, "_warned_borrow", False):
                    log.warning("cost_model.holding_cost_usd failed (%s); using snapshot borrow_fee", e)
                    self._warned_borrow = True
        fee = snap["borrow_fee"].reindex(w_post.index).fillna(self.cfg.cost.default_borrow_fee_annual)
        return (-w_post).clip(lower=0.0) * nav * fee / self.cfg.trading_days_per_year

    def _estimate_kwargs(self, venue: str, date: pd.Timestamp, streak: Optional[np.ndarray],
                         tickers: pd.Index) -> Dict[str, object]:
        """Keyword arguments ``cost_model.estimate`` accepts among
        ``venue`` / ``date`` / ``streak`` (structural typing: SimpleCostModel
        takes ``venue`` only, the tcost facade takes all three)."""
        import inspect
        params = getattr(self, "_estimate_params", None)
        if params is None:
            try:
                params = set(inspect.signature(self.cost_model.estimate).parameters)
            except (TypeError, ValueError):
                params = set()
            self._estimate_params = params
        kw: Dict[str, object] = {}
        if "venue" in params:
            kw["venue"] = venue
        if "date" in params:
            kw["date"] = date
        if "streak" in params and streak is not None:
            kw["streak"] = pd.Series(streak, index=tickers)
        return kw

    # ----------------------------------------------------------------- run
    def run(
        self,
        md: MarketData,
        dates: pd.DatetimeIndex,
        alpha_fn: Callable[[pd.Timestamp], AlphaOutput],
        risk_fn: Callable[[pd.Timestamp, Optional[pd.Series]], RiskModelOutput],
        optimizer: SinglePeriodOptimizer,
        nav0: float = 1e6,
        rebalance_every: int = 1,
        venue: str = "auction",
        neutral_styles: Optional[List[str]] = None,
        warm_start_weights: Optional[pd.Series] = None,
        progress: bool = False,
        eligible_fn: Optional[Callable[[pd.Timestamp], pd.Index]] = None,
        exit_ineligible_days: Optional[int] = None,
    ) -> BacktestResult:
        """``eligible_fn(t)`` returns the tickers investable as of ``t``; when
        ``md.eligible`` (wide bool panel) is present it takes precedence and
        ``eligible_fn`` is ignored.  ``exit_ineligible_days`` overrides
        ``cfg.portfolio.exit_ineligible_days`` (linear unwind horizon)."""
        t_start = time.perf_counter()
        dates = pd.DatetimeIndex(dates)
        if len(dates) < 2:
            raise ValueError("need at least two dates")
        if not dates.isin(md.dates).all():
            raise ValueError("dates must be a subset of md.dates")
        tickers = md.tickers
        N = len(tickers)
        close = md.close.loc[dates]
        R = close.pct_change(fill_method=None)            # return realised on row date
        price_ok = close.notna()
        elig_panel = getattr(md, "eligible", None)
        elig_rows: Optional[np.ndarray] = None
        if elig_panel is not None:
            e = elig_panel.reindex(index=dates, columns=tickers)
            elig_rows = e.fillna(False).astype(bool).values
            eligible_fn = None
        exit_days = int(exit_ineligible_days if exit_ineligible_days is not None
                        else getattr(self.cfg.portfolio, "exit_ineligible_days", 5))
        exit_days = max(exit_days, 1)
        forced: Dict[str, Dict[str, float]] = {}   # ticker -> {"w0": |w| at entry, "k": days elapsed}
        forced_log: List[Dict[str, object]] = []
        n_out_of_model = 0
        n_ineligible_days = 0

        nav = float(nav0)
        w_pre = np.zeros(N)
        if warm_start_weights is not None:
            w_pre = pd.Series(warm_start_weights, dtype=float).reindex(tickers).fillna(0.0).values

        T = len(dates) - 1
        W = np.zeros((T, N)); X = np.zeros((T, N))
        C = np.zeros((T, 5)); EXV = np.full(T, np.nan); TO = np.zeros(T); TC = np.full(T, np.nan)
        navs = np.empty(T + 1); navs[0] = nav
        gross = np.empty(T); net = np.empty(T)
        delistings: List[Dict[str, object]] = []
        solve_log: List[Dict[str, object]] = []
        n_failed = 0
        last_risk: Optional[RiskModelOutput] = None
        last_alpha: Optional[AlphaOutput] = None

        for i in range(T):
            t = dates[i]
            t1 = dates[i + 1]
            snap = self._snapshot(md, t)
            alive = price_ok.iloc[i].values
            # ---- eligibility as of t
            if elig_rows is not None:
                eligible = elig_rows[i] & alive
            elif eligible_fn is not None:
                eligible = tickers.isin(pd.Index(eligible_fn(t))) & alive
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
                    forced.pop(tk)          # back in the universe, or already flat
            force_caps: Dict[str, float] = {}
            for tk, st in forced.items():
                st["k"] += 1.0
                force_caps[tk] = max(st["w0"] * (1.0 - st["k"] / exit_days), 0.0)
            dead = list(tickers[~alive]) + [tk for tk in tickers[inel & ~held]]
            w_post = w_pre.copy()
            if i % rebalance_every == 0:
                last_alpha = alpha_fn(t)
                exp_ret = pd.Series(last_alpha.expected_return, dtype=float).reindex(tickers)
                exp_ret[~alive] = 0.0
                exp_ret[inel] = 0.0
                exp_ret = exp_ret.fillna(0.0)
                last_risk = risk_fn(t, exp_ret)
                res = optimizer.solve(
                    alpha=exp_ret, risk=last_risk, w_current=pd.Series(w_pre, index=tickers), nav=nav,
                    snapshot=snap, venue=venue, neutral_styles=neutral_styles, no_trade=dead,
                    horizon_days=getattr(last_alpha, "horizon_days", None) or optimizer.horizon_days,
                    force_to_zero=force_caps or None,
                )
                if res.status not in ("optimal", "optimal_inaccurate"):
                    n_failed += 1
                    log.warning("optimizer solve failed on %s: status=%s (holding current book)",
                                t.date(), res.status)
                n_out_of_model += int(getattr(res, "n_out_of_model", 0) or 0)
                w_post = np.array(res.w_target.reindex(tickers).fillna(0.0).values, dtype=float, copy=True)
                w_post[~alive] = w_pre[~alive]  # cannot trade names without a price
                TC[i] = res.transfer_coefficient
                solve_log.append({"date": t, "status": res.status, "solve_time": res.solve_time,
                                  "binding": res.binding_constraints, "vol_scalar": res.vol_scalar,
                                  "expected_alpha": res.expected_alpha, "expected_risk": res.expected_risk,
                                  "expected_cost": res.expected_cost, "notes": res.notes,
                                  "n_out_of_model": getattr(res, "n_out_of_model", 0)})
            x = w_post - w_pre
            x[np.abs(x) < 1e-10] = 0.0
            w_post = w_pre + x

            # costs in USD (priced at the venue this run was asked for, dated t)
            # -- same-direction streak per name for multi-day impact persistence
            # (loop-local state; initialised on the first step, carried across iterations)
            if i == 0:
                streak = np.zeros(N, dtype=float)
                last_sign = np.zeros(N, dtype=float)
            sgn = np.sign(x)
            streak = np.where(sgn != 0, np.where(sgn == last_sign, streak + 1.0, 1.0), 0.0)
            last_sign = sgn
            est = self.cost_model.estimate(pd.Series(x * nav, index=tickers), snap,
                                           **self._estimate_kwargs(venue, t, streak, tickers))
            cf = self._cost_frame(est, tickers)
            borrow = self._borrow_usd(pd.Series(w_post, index=tickers), nav, snap)
            c_spread, c_impact, c_fees, c_borrow = (float(cf["spread"].sum()), float(cf["impact"].sum()),
                                                    float(cf["fees"].sum()), float(borrow.sum()))
            c_total = c_spread + c_impact + c_fees + c_borrow

            # returns t -> t+1
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
                h = float(getattr(self.cfg.cost, "delisting_haircut", -0.30))
                r[liquidate] = h * np.sign(w_post[liquidate])
                x_liq = np.zeros(N)
                x_liq[liquidate] = -w_post[liquidate] * (1.0 + r[liquidate])
                liq_kw = self._estimate_kwargs("continuous", t, None, tickers)
                try:
                    liq_cf = self._cost_frame(self.cost_model.estimate(pd.Series(x_liq * nav, index=tickers), snap,
                                                                       **liq_kw), tickers)
                    liq_cf["impact"] = 0.0
                except Exception as e:  # pragma: no cover - cost model without spread/fees split
                    log.warning("delisting liquidation cost failed (%s); charging default spread only", e)
                    liq_cf = pd.DataFrame(0.0, index=tickers, columns=["spread", "impact", "fees"])
                    liq_cf["spread"] = np.abs(x_liq) * nav * self.cfg.cost.default_half_spread_bps / 1e4
                for j in np.where(liquidate)[0]:
                    delistings.append({"date": t1, "ticker": tickers[j], "weight": float(w_post[j]),
                                       "haircut_return": float(r[j]),
                                       "liquidation_usd": float(abs(x_liq[j]) * nav),
                                       "cost_usd": float(liq_cf.iloc[j].sum())})
                c_spread += float(liq_cf["spread"].sum()); c_fees += float(liq_cf["fees"].sum())
                c_total = c_spread + c_impact + c_fees + c_borrow
                x = x + x_liq                     # recorded in trades / turnover below
            g_ret = float(w_post @ r)
            nav1 = nav * (1.0 + g_ret) - c_total
            n_ret = nav1 / nav - 1.0

            # record
            W[i] = w_post; X[i] = x; C[i] = (c_spread, c_impact, c_fees, c_borrow, c_total)
            TO[i] = float(np.abs(x).sum())
            if last_risk is not None:
                EXV[i] = float(last_risk.portfolio_variance(pd.Series(w_post, index=tickers)))
            gross[i] = g_ret; net[i] = n_ret; navs[i + 1] = nav1

            # drift to pre-trade weights of t+1 (cash absorbs costs)
            w_next = w_post * (1.0 + r) * nav / nav1 if nav1 > 0 else np.zeros(N)
            w_next[liquidate] = 0.0
            w_pre = w_next
            nav = nav1
            if nav <= 0:
                log.error("NAV <= 0 on %s; stopping", t1)
                T_eff = i + 1
                W, X, C, EXV, TO, TC = W[:T_eff], X[:T_eff], C[:T_eff], EXV[:T_eff], TO[:T_eff], TC[:T_eff]
                navs, gross, net = navs[:T_eff + 1], gross[:T_eff], net[:T_eff]
                dates = dates[:T_eff + 1]
                T = T_eff
                break
            if progress and i % 50 == 0:
                log.info("%s nav=%.0f gross=%.4f to=%.3f", t.date(), nav, g_ret, TO[i])

        dec = dates[:T]
        real = dates[1:T + 1]
        res_obj = BacktestResult(
            nav=pd.Series(navs, index=dates[:T + 1], name="nav"),
            returns=pd.DataFrame({"gross": gross, "net": net}, index=real),
            weights=pd.DataFrame(W, index=dec, columns=tickers),
            trades=pd.DataFrame(X, index=dec, columns=tickers),
            costs=pd.DataFrame(C, index=dec, columns=COST_COLUMNS),
            exante_var=pd.Series(EXV, index=dec, name="exante_var_daily"),
            turnover=pd.Series(TO, index=dec, name="turnover"),
            transfer_coefficient=pd.Series(TC, index=dec, name="transfer_coefficient"),
            periods_per_year=self.cfg.trading_days_per_year,
            diagnostics={"delistings": delistings, "solve_log": solve_log, "n_failed_solves": n_failed,
                         "run_time_sec": time.perf_counter() - t_start, "rebalance_every": rebalance_every,
                         "venue": venue, "forced_exits": forced_log,
                         "n_ineligible_name_days": n_ineligible_days,
                         "n_out_of_model_name_days": n_out_of_model},
        )
        if n_failed:
            log.warning("backtest: %d failed solves (see diagnostics['solve_log'])", n_failed)
        if n_out_of_model:
            log.warning("backtest: %d name-days with alpha/position outside the risk-model universe "
                        "(priced at 90th-pct specific variance, zero factor exposure)", n_out_of_model)
        return res_obj
