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

Delistings: a held name whose return on ``t+1`` is NaN is liquidated at its
last price (return 0), the position is dropped and the event recorded in
``BacktestResult.diagnostics["delistings"]``.  Names with no price on ``t``
are placed on the optimizer's no-trade list.

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
    def _snapshot(self, md: MarketData, t: pd.Timestamp) -> Snapshot:
        fn = getattr(self.cost_model, "snapshot_from_market", None)
        if callable(fn):
            try:
                snap = fn(md, t)
                if isinstance(snap, dict) and "adv_usd" in snap:
                    return align_snapshot(snap, md.tickers, self.cfg.cost)
            except Exception as e:  # fall back to the built-in snapshot
                log.debug("cost_model.snapshot_from_market failed (%s); using built-in", e)
        return snapshot_from_market(md, t, self.adv_window, self.vol_window, self.cfg.cost)

    @staticmethod
    def _cost_frame(est, tickers: pd.Index) -> pd.DataFrame:
        """Normalise whatever ``estimate`` returned into spread/impact/fees USD."""
        out = pd.DataFrame(0.0, index=tickers, columns=["spread", "impact", "fees"])
        if est is None:
            return out
        if isinstance(est, pd.Series):
            out["impact"] = est.reindex(tickers).fillna(0.0)
            return out
        if isinstance(est, dict):
            est = pd.DataFrame(est)
        if isinstance(est, pd.DataFrame):
            cols = {c.lower(): c for c in est.columns}
            matched = False
            for k in ("spread", "impact", "fees"):
                for alias in (k, k + "_usd", k + "_cost"):
                    if alias in cols:
                        out[k] = est[cols[alias]].reindex(tickers).fillna(0.0).astype(float)
                        matched = True
                        break
            if not matched and "total" in cols:
                out["impact"] = est[cols["total"]].reindex(tickers).fillna(0.0).astype(float)
            return out
        raise TypeError(f"unsupported cost estimate type {type(est)}")

    def _borrow_usd(self, w_post: pd.Series, nav: float, snap: Snapshot) -> pd.Series:
        fn = getattr(self.cost_model, "holding_cost_usd", None)
        if callable(fn):
            try:
                return pd.Series(fn(w_post, nav, snap), dtype=float).reindex(w_post.index).fillna(0.0)
            except Exception:
                pass
        fee = snap["borrow_fee"].reindex(w_post.index).fillna(self.cfg.cost.default_borrow_fee_annual)
        return (-w_post).clip(lower=0.0) * nav * fee / self.cfg.trading_days_per_year

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
    ) -> BacktestResult:
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
            dead = list(tickers[~alive])
            w_post = w_pre.copy()
            if i % rebalance_every == 0:
                last_alpha = alpha_fn(t)
                exp_ret = pd.Series(last_alpha.expected_return, dtype=float).reindex(tickers)
                exp_ret[~alive] = 0.0
                exp_ret = exp_ret.fillna(0.0)
                last_risk = risk_fn(t, exp_ret)
                res = optimizer.solve(
                    alpha=exp_ret, risk=last_risk, w_current=pd.Series(w_pre, index=tickers), nav=nav,
                    snapshot=snap, venue=venue, neutral_styles=neutral_styles, no_trade=dead,
                    horizon_days=getattr(last_alpha, "horizon_days", None) or optimizer.horizon_days,
                )
                if res.status not in ("optimal", "optimal_inaccurate"):
                    n_failed += 1
                w_post = np.array(res.w_target.reindex(tickers).fillna(0.0).values, dtype=float, copy=True)
                w_post[~alive] = w_pre[~alive]  # cannot trade names without a price
                TC[i] = res.transfer_coefficient
                solve_log.append({"date": t, "status": res.status, "solve_time": res.solve_time,
                                  "binding": res.binding_constraints, "vol_scalar": res.vol_scalar,
                                  "expected_alpha": res.expected_alpha, "expected_risk": res.expected_risk,
                                  "expected_cost": res.expected_cost, "notes": res.notes})
            x = w_post - w_pre
            x[np.abs(x) < 1e-10] = 0.0
            w_post = w_pre + x

            # costs in USD
            est = self.cost_model.estimate(pd.Series(x * nav, index=tickers), snap)
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
            for j in np.where(liquidate)[0]:
                delistings.append({"date": t1, "ticker": tickers[j], "weight": float(w_post[j])})
            r[nan_r] = 0.0
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
                         "venue": venue},
        )
        return res_obj
