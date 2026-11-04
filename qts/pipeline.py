"""End-to-end research pipeline: universe -> risk -> alpha -> cost -> portfolio -> backtest -> validation.

This is the "thin vertical slice" the review asked for: every layer is wired
together on a daily-bar simulator so integration problems surface before any
single layer is deepened.  It runs on the synthetic market out of the box and
on a :class:`~qts.data.store.ParquetStore` once real data has been loaded.

Usage::

    from qts.pipeline import ResearchPipeline
    from qts.data.synthetic import generate, SyntheticSpec
    md = generate(SyntheticSpec(n_stocks=300, n_days=1512))
    res = ResearchPipeline().run(md)
    print(res.summary)          # backtest statistics
    print(res.hurdles.format()) # plan gates (phase 1 / 2)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from qts.alpha import AlphaModel
from qts.alpha.evaluation import ic_report, rank_ic_series
from qts.alpha.targets import residualize_forward_returns
from qts.backtest import DailyBarSimulator, BacktestResult
from qts.backtest import metrics as M
from qts.core.config import SystemConfig
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput
from qts.data.universe import apply_mask, build_universe_mask
from qts.portfolio import SinglePeriodOptimizer
from qts.risk import BarraStyleRiskModel
from qts.tcost import TransactionCostModel
from qts.validation import HurdleReport, TrialLedger, deflated_sharpe_ratio_annual

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    backtest: BacktestResult
    summary: Dict[str, float]
    ic_report: Dict[str, object]
    hurdles: object
    cost_sensitivity: pd.DataFrame
    fold_sharpes: pd.Series
    timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class ResearchPipeline:
    """Fit all five models on one MarketData and backtest them out-of-sample.

    Parameters
    ----------
    cfg : SystemConfig
    use_ml : bool
        Include the walk-forward ML sleeve (LightGBM/MLP/ridge).  Phase 1 of
        the plan is ``use_ml=False`` (rank composite only).
    phase : int
        Which hurdle set to apply (1: composite baseline, 2: + ML).
    ledger_path : str | None
        Trial ledger (JSONL) to append this run to; needed for an honest DSR.
    """

    def __init__(self, cfg: Optional[SystemConfig] = None, use_ml: bool = True, phase: Optional[int] = None,
                 ledger_path: Optional[str] = None, composite_weight: float = 0.5,
                 ml_factory: Optional[Callable[[], object]] = None, rebalance_every: int = 1,
                 venue: str = "auction", nav0: float = 1_000_000.0, warmup_days: Optional[int] = None):
        self.cfg = cfg or SystemConfig()
        self.use_ml = use_ml
        self.phase = phase if phase is not None else (2 if use_ml else 1)
        self.ledger = TrialLedger(ledger_path) if ledger_path else None
        self.composite_weight = composite_weight
        self.ml_factory = ml_factory
        self.rebalance_every = rebalance_every
        self.venue = venue
        self.nav0 = nav0
        self.warmup_days = warmup_days

        # fitted components (exposed for inspection)
        self.universe_mask: Optional[pd.DataFrame] = None
        self.md_u: Optional[MarketData] = None
        self.risk: Optional[BarraStyleRiskModel] = None
        self.alpha: Optional[AlphaModel] = None
        self.cost: Optional[TransactionCostModel] = None
        self.optimizer: Optional[SinglePeriodOptimizer] = None

    # ------------------------------------------------------------------ fit
    def fit(self, md: MarketData) -> "ResearchPipeline":
        cfg = self.cfg
        t: Dict[str, float] = {}

        t0 = time.perf_counter()
        self.universe_mask = build_universe_mask(md, cfg.universe)
        self.md_u = apply_mask(md, self.universe_mask)
        t["universe"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.risk = BarraStyleRiskModel(cfg.risk, warmup_days=self.warmup_days).fit(self.md_u)
        t["risk_fit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.alpha = AlphaModel(cfg.alpha, composite_weight=self.composite_weight,
                                ml_factory=self.ml_factory, use_ml=self.use_ml)
        # residualize labels on the *risk model's* exposures (plan 2.3: targets)
        self.alpha.fit(self.md_u, risk_exposures_fn=self._exposures_fn)
        t["alpha_fit"] = time.perf_counter() - t0

        self.cost = TransactionCostModel(cfg.cost)
        self.optimizer = SinglePeriodOptimizer(cfg.portfolio, self.cost, horizon_days=cfg.alpha.horizon_days)
        self.timings_ = t
        return self

    #: risk-model styles that are *not* alpha themes; labels are residualised
    #: on market + industries + these only, so theme premia remain in the target
    #: (residualising on the alpha themes themselves would define the alpha away).
    RISK_ONLY_STYLES = ("beta", "size", "residual_vol", "liquidity", "short_term_reversal")

    def _exposures_fn(self, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        try:
            X = self.risk.exposures_as_of(date)
        except Exception:
            return None
        alpha_themes = {s for s in self.cfg.risk.styles if s not in self.RISK_ONLY_STYLES}
        return X.loc[:, [c for c in X.columns if c not in alpha_themes]]

    # ------------------------------------------------------------- backtest
    def backtest_dates(self) -> pd.DatetimeIndex:
        """Out-of-sample dates: after the risk warm-up and the first alpha fit."""
        dates = self.md_u.dates
        fr = self.risk.factor_returns.dropna(how="all")
        k = min(len(fr) - 1, self.cfg.risk.min_history_days)
        start_risk = fr.index[k]
        # first date with an out-of-sample ML prediction (if ML is on) else composite start
        start_alpha = None
        ml = getattr(self.alpha, "ml_score_", None)
        if ml is not None:
            valid = ml.notna().any(axis=1)
            if valid.any():
                start_alpha = valid[valid].index[0]
        start = max(start_risk, start_alpha) if start_alpha is not None else start_risk
        return dates[dates >= start][:-1]

    def run(self, md: MarketData, dates: Optional[pd.DatetimeIndex] = None, progress: bool = False) -> PipelineResult:
        if self.alpha is None:
            self.fit(md)
        cfg = self.cfg
        dates = self.backtest_dates() if dates is None else pd.DatetimeIndex(dates)
        sim = DailyBarSimulator(cfg, self.cost)

        def alpha_fn(d: pd.Timestamp) -> AlphaOutput:
            return self.alpha.as_of(d)

        def risk_fn(d: pd.Timestamp, alpha: Optional[pd.Series]) -> RiskModelOutput:
            return self.risk.as_of(d, alpha=alpha if cfg.risk.alignment else None)

        neutral_styles = [s for s in cfg.risk.styles if s in self.RISK_ONLY_STYLES]
        t0 = time.perf_counter()
        bt = sim.run(self.md_u, dates, alpha_fn, risk_fn, self.optimizer, nav0=self.nav0,
                     rebalance_every=self.rebalance_every, venue=self.venue,
                     neutral_styles=neutral_styles, progress=progress)
        timings = dict(self.timings_, backtest=time.perf_counter() - t0)
        return self.evaluate(bt, timings)

    # ------------------------------------------------------------- evaluate
    def evaluate(self, bt: BacktestResult, timings: Optional[Dict[str, float]] = None) -> PipelineResult:
        cfg = self.cfg
        summ = bt.summary()
        net = bt.returns["net"].dropna()
        gross = bt.returns["gross"].dropna()

        # IC of the live alpha score vs the residualised target over the backtest window
        H = cfg.alpha.horizon_days
        target = residualize_forward_returns(self.md_u, H, exposures_fn=self._exposures_fn)
        scores = pd.DataFrame({d: self.alpha.as_of(d).score for d in bt.weights.index[::H]}).T
        scores.index.name = "date"
        icr = ic_report(scores, target.reindex(scores.index))

        # fold-wise net Sharpe (calendar-year folds) for the >=2/3-of-folds gates
        fold_sharpes = net.groupby(net.index.year).apply(lambda r: M.annualized_sharpe(r, cfg.trading_days_per_year))
        fold_sharpes = fold_sharpes[net.groupby(net.index.year).size() >= 60]

        cs = M.cost_sensitivity(gross, bt.turnover.reindex(gross.index).fillna(0.0),
                                cost_bps_per_side=summ.get("avg_cost_bps_per_side", 10.0))
        breakeven = M.breakeven_cost_bps(gross, bt.turnover.reindex(gross.index).fillna(0.0))

        n_trials = self.ledger.n_trials() + 1 if self.ledger else 1
        dsr = deflated_sharpe_ratio_annual(summ["sharpe_net"], n_trials=n_trials, var_trials_sr_annual=0.0 if n_trials == 1 else 0.25,
                                           n_obs=len(net), skew=float(net.skew()), kurt=float(net.kurt() + 3),
                                           periods_per_year=cfg.trading_days_per_year)
        hurdles = HurdleReport(cfg.validation).evaluate_signal(
            ic_by_fold=(icr["by_year"]["mean"].astype(float) if isinstance(icr.get("by_year"), pd.DataFrame)
                        and len(icr["by_year"]) else None),
            decile_tstat=float(icr.get("t_stat", np.nan)),
            net_sharpe_delta_by_fold=fold_sharpes,          # vs. zero baseline for the standalone run
            breakeven_cost_bps=breakeven, modeled_cost_bps=summ.get("avg_cost_bps_per_side"),
            dsr=dsr, pbo=None, phase=self.phase, net_sharpe=summ["sharpe_net"], standalone=True)

        if self.ledger is not None:
            self.ledger.log(name=f"pipeline_phase{self.phase}", config={"use_ml": self.use_ml, "phase": self.phase},
                            metrics={"sharpe_annual": summ["sharpe_net"], "n_obs": len(net),
                                     "skew": float(net.skew()), "kurt": float(net.kurt() + 3),
                                     "mean_ic": icr.get("mean_ic")},
                            notes="ResearchPipeline.run")

        notes = []
        if summ["sharpe_net"] > 2:
            notes.append("net Sharpe > 2: treat as suspect until leakage/cost/selection ruled out (plan 2.4)")
        ehd = summ.get("effective_holding_days", np.nan)
        if np.isfinite(ehd) and ehd < 0.6 * cfg.portfolio.effective_holding_days_target:
            notes.append(f"effective holding {ehd:.0f}d < target {cfg.portfolio.effective_holding_days_target}d: raise trade_aversion")
        return PipelineResult(backtest=bt, summary=summ, ic_report=icr, hurdles=hurdles,
                              cost_sensitivity=cs, fold_sharpes=fold_sharpes, timings=timings or {}, notes=notes)


def format_result(res: PipelineResult) -> str:
    s = res.summary
    lines = [
        "=== Backtest summary ===",
        f"period            {s['start']} -> {s['end']}  ({s['n_days']} days)",
        f"Sharpe gross/net  {s['sharpe_gross']:.2f} / {s['sharpe_net']:.2f}   (NW t = {s['newey_west_t_net']:.2f})",
        f"CAGR gross/net    {100*s['cagr_gross']:.2f}% / {100*s['cagr_net']:.2f}%   vol {100*s['ann_vol_net']:.1f}%",
        f"max drawdown      {100*s['max_drawdown_net']:.1f}%   calmar {s['calmar_net']:.2f}",
        f"turnover/day      {100*s['avg_daily_turnover']:.2f}%   effective holding {s['effective_holding_days']:.0f} d",
        f"cost per side     {s['avg_cost_bps_per_side']:.1f} bp   gross exp {s['avg_gross_exposure']:.2f}  net exp {s['avg_net_exposure']:+.2f}",
        f"transfer coeff    {s.get('transfer_coefficient_mean', float('nan')):.2f}   realized/ex-ante vol {s.get('realized_vs_exante_vol_ratio', float('nan')):.2f}",
        "",
        "=== Alpha IC (score vs residualised %dd target) ===" % 20,
        f"mean rank IC {res.ic_report.get('mean_ic', float('nan')):.4f}  ICIR {res.ic_report.get('icir', float('nan')):.2f}  t {res.ic_report.get('t_stat', float('nan')):.2f}",
        "",
        "=== Fold (calendar-year) net Sharpe ===",
        res.fold_sharpes.round(2).to_string(),
        "",
        "=== Cost sensitivity (net Sharpe at cost multiples) ===",
        res.cost_sensitivity.round(3).to_string(),
        "",
        "=== Plan hurdles ===",
        HurdleReport.format(res.hurdles),
    ]
    if res.notes:
        lines += ["", "NOTES: " + " | ".join(res.notes)]
    if res.timings:
        lines += ["", "timings (s): " + ", ".join(f"{k}={v:.1f}" for k, v in res.timings.items())]
    return "\n".join(lines)
