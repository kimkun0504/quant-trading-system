"""Backtest output: :class:`BacktestResult` (panels + diagnostics) and its
derived views (summary metrics, cost sensitivity, sizing summary).

Recorded panels are indexed by the decision date ``t``: ``weights`` (post-
trade), ``trades``, ``costs``, ``exante_var``; ``returns`` and ``nav`` are
indexed by the date the return is realised (``t+1``).  See
:mod:`qts.backtest.simulator` for the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, cast

import numpy as np
import pandas as pd

from qts.backtest import metrics as M
from qts.core.types import Wide
from qts.portfolio.diagnostics import effective_holding_days, realized_vs_exante_risk

COST_COLUMNS = ["spread", "impact", "fees", "borrow", "total"]


class SizingSummary(TypedDict):
    """:meth:`BacktestResult.sizing_summary`."""

    vol_target_caps_share: float
    mean_risk_aversion: float


class BacktestSummary(SizingSummary):
    """:meth:`BacktestResult.summary` (persisted as ``summary.json`` and served by
    the API as ``BacktestSummary``; keys are part of the run-directory contract)."""

    start: str
    end: str
    n_days: int
    final_nav: float
    cagr_gross: float
    cagr_net: float
    ann_vol_gross: float
    ann_vol_net: float
    sharpe_gross: float
    sharpe_net: float
    sharpe_net_lo: float
    sharpe_t_stat_net: float
    newey_west_t_net: float
    max_drawdown_net: float
    calmar_net: float
    sortino_net: float
    hit_rate_net: float
    avg_gross_exposure: float
    avg_net_exposure: float
    avg_daily_turnover: float
    effective_holding_days: float
    avg_cost_bps_per_side: float
    avg_daily_cost_bps_of_nav: float
    total_costs_usd: float
    transfer_coefficient_mean: float
    realized_vs_exante_vol_ratio: float
    breakeven_cost_bps: float
    n_delistings: int
    n_failed_solves: int


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

    def summary(self) -> BacktestSummary:
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
        out: BacktestSummary = {
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
            "n_delistings": len(cast(list[object], self.diagnostics.get("delistings", []))),
            "n_failed_solves": int(cast(int, self.diagnostics.get("n_failed_solves", 0))),
            **self.sizing_summary(),
        }
        return out

    def sizing_summary(self) -> SizingSummary:
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

    def failed_solve_fraction(self) -> float | None:
        """Failed rebalance solves / rebalance solves (``diagnostics``), or
        ``None`` when no solve was attempted (nothing to judge)."""
        n = self.diagnostics.get("n_rebalance_solves")
        if n is None:  # results built without the counter: count the solve log
            log_ = self.diagnostics.get("solve_log")
            n = len(log_) if isinstance(log_, list) else 0
        n = int(cast(int, n))
        if n <= 0:
            return None
        return int(cast(int, self.diagnostics.get("n_failed_solves", 0))) / n

    def turnover_on_return_dates(self) -> pd.Series:
        """Turnover on decision date t re-indexed to the realisation date t+1
        (positionally aligned with ``returns``)."""
        n = min(len(self.turnover), len(self.returns))
        return pd.Series(self.turnover.values[:n], index=self.returns.index[:n], name="turnover")
