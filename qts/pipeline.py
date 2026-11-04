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

Eligibility contract
--------------------
``apply_mask`` stores the point-in-time universe on ``md.eligible``; the alpha
layer scores only names in ``md.eligibility()`` at each date (ineligible and
dead names are *absent* from ``AlphaOutput.score``, never zero-filled).  The
pipeline additionally wraps ``alpha_fn`` so that nothing outside the eligible
set reaches the optimizer with a non-zero expected return, and forwards the
eligibility panel to ``DailyBarSimulator.run`` when the simulator exposes an
``eligible_fn`` / ``eligibility`` keyword (see :meth:`ResearchPipeline.run`).

Label contract
--------------
Labels are residualised on the risk model's **risk-only** exposures (market +
industries + beta / size / residual vol / liquidity / short-term reversal) —
never on the alpha themes.  During the risk model's warm-up the same risk-only
style set is built internally (``qts.alpha.targets.default_exposures``); any
other failure of the exposures hook raises instead of silently changing the
label definition.

Hurdle inputs (plan §2.4)
-------------------------
* ``t_stat``: Newey–West t of the **top-minus-bottom decile** of raw forward
  returns on non-overlapping formation dates (every ``H`` days), eligible names
  only, value-weighted.
* ``net_sharpe_delta``: for a standalone run the baseline is zero, so the gate
  reads "net Sharpe by calendar-year fold > +0.05 in >= 2/3 of folds".
* ``dsr``: ``n_trials`` and the cross-trial SR variance come from the trial
  ledger family of this pipeline (``effective_trials_for`` when >= 3 trials;
  single-trial PSR when the ledger is empty / absent).  Identical re-runs
  (same config hash + data hash) update the ledger record instead of adding a
  trial.
* ``pbo``: only computed when ``SystemConfig.pipeline_pbo_configs`` lists a
  configuration grid (CSCV over the grid's net return streams); otherwise the
  gate is reported as *skipped*, which does not fail ``overall``.
* Required gates (fail when missing): ``phase_sharpe``, ``suspect``, ``dsr``.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd

from qts.alpha import AlphaModel
from qts.alpha.evaluation import DecileSpreadResult, decile_spread, ic_report
from qts.alpha.targets import forward_returns
from qts.backtest import BacktestResult, DailyBarSimulator
from qts.backtest import metrics as M
from qts.core.config import SystemConfig
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput
from qts.data.universe import apply_mask, build_universe_mask
from qts.portfolio import SinglePeriodOptimizer
from qts.risk import BarraStyleRiskModel, HybridRiskModel, build_risk_model
from qts.tcost import TransactionCostModel
from qts.validation import HurdleReport, TrialLedger, deflated_sharpe_ratio_annual
from qts.validation.pbo import probability_of_backtest_overfitting

log = logging.getLogger(__name__)

RiskModel = Union[BarraStyleRiskModel, HybridRiskModel]


@dataclass
class PipelineResult:
    backtest: BacktestResult
    summary: Dict[str, float]
    ic_report: Dict[str, object]
    hurdles: Dict[str, object]
    cost_sensitivity: pd.DataFrame
    fold_sharpes: pd.Series
    timings: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    decile: Optional[DecileSpreadResult] = None
    dsr: float = float("nan")
    dsr_inputs: Dict[str, float] = field(default_factory=dict)
    pbo: Optional[float] = None
    config_hash: str = ""
    data_hash: str = ""


def data_hash_of(md: MarketData) -> str:
    """Short SHA-256 of the close panel bytes (plus shape/date range) — the
    data identity written to the trial ledger."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(md.close.to_numpy(dtype=float)).tobytes())
    h.update(f"{md.close.shape}|{md.dates[0]}|{md.dates[-1]}".encode())
    return h.hexdigest()[:16]


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
        Trial ledger (JSONL) to log this run to; needed for an honest DSR.
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
        self.risk: Optional[RiskModel] = None
        self.alpha: Optional[AlphaModel] = None
        self.cost: Optional[TransactionCostModel] = None
        self.optimizer: Optional[SinglePeriodOptimizer] = None
        self.timings_: Dict[str, float] = {}
        self.data_hash_: str = ""
        self._md_input: Optional[MarketData] = None
        self._risk_first_date: Optional[pd.Timestamp] = None
        self._warned_warmup = False

    # ------------------------------------------------------------------ fit
    def fit(self, md: MarketData) -> "ResearchPipeline":
        cfg = self.cfg
        t: Dict[str, float] = {}
        self._md_input = md
        self.data_hash_ = data_hash_of(md)

        t0 = time.perf_counter()
        self.universe_mask = build_universe_mask(md, cfg.universe)
        self.md_u = apply_mask(md, self.universe_mask)
        t["universe"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.risk = build_risk_model(cfg.risk, warmup_days=self.warmup_days).fit(self.md_u)
        exp_dates = list(self.risk.exposures.keys()) if isinstance(self.risk.exposures, dict) else []
        self._risk_first_date = min(exp_dates) if exp_dates else None
        self._warned_warmup = False
        t["risk_fit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.alpha = AlphaModel(cfg.alpha, composite_weight=self.composite_weight,
                                ml_factory=self.ml_factory, use_ml=self.use_ml)
        # residualize labels on the *risk model's* risk-only exposures (plan 2.3: targets)
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
        """Risk-only exposures of the fitted risk model as of ``date``.

        Returns ``None`` **only** for dates before the risk model's first
        exposure date (its rolling-descriptor warm-up); the label builder then
        uses its internal risk-only style set for those dates (logged once).
        Any other failure propagates — a silent fallback would change the
        label definition mid-sample (audit finding 2).
        """
        assert self.risk is not None, "fit() first"
        date = cast(pd.Timestamp, pd.Timestamp(date))
        first = self._risk_first_date
        if first is None or bool(date < first):
            if not self._warned_warmup:
                log.info("labels before %s (risk-model warm-up) use the internal risk-only exposures "
                         "(beta, size, idio vol, liquidity, reversal + industries)",
                         None if self._risk_first_date is None else self._risk_first_date.date())
                self._warned_warmup = True
            return None
        X = self.risk.exposures_as_of(date)          # raises on anything unexpected
        alpha_themes = {s for s in self.cfg.risk.styles if s not in self.RISK_ONLY_STYLES}
        return X.loc[:, [c for c in X.columns if c not in alpha_themes]]

    # ------------------------------------------------------------- backtest
    def _require_fitted(self) -> None:
        if self.alpha is None or self.risk is None or self.md_u is None or self.optimizer is None:
            raise RuntimeError("call fit() first")

    def backtest_dates(self) -> pd.DatetimeIndex:
        """Out-of-sample dates: after the risk warm-up and the first alpha fit."""
        self._require_fitted()
        assert self.md_u is not None and self.risk is not None and self.alpha is not None
        dates = self.md_u.dates
        fr_all = self.risk.factor_returns
        if fr_all is None:
            raise RuntimeError("risk model has no factor returns; fit() first")
        fr = fr_all.dropna(how="all")
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

    def _eligible_alpha_fn(self) -> Callable[[pd.Timestamp], AlphaOutput]:
        """``alpha_fn`` for the simulator that emits scores / expected returns for
        names eligible at ``d`` only.  ``AlphaModel.as_of`` already does this;
        the wrapper is the pipeline-level guarantee (and the place where a
        different alpha object would be constrained).  Names the simulator
        reindexes to that are absent here get expected return 0 (``fillna(0)``
        in ``DailyBarSimulator.run``)."""
        assert self.alpha is not None and self.md_u is not None
        elig = self.md_u.eligibility()
        alpha = self.alpha

        def alpha_fn(d: pd.Timestamp) -> AlphaOutput:
            out = alpha.as_of(d)
            ok = elig.loc[out.date]
            keep = out.score.index[ok.reindex(out.score.index).fillna(False).astype(bool).to_numpy()]
            if len(keep) == len(out.score):
                return out
            return AlphaOutput(date=out.date, score=out.score.loc[keep], expected_return=out.expected_return.loc[keep],
                               horizon_days=out.horizon_days, half_life_days=out.half_life_days,
                               components={k: v.reindex(keep) for k, v in out.components.items()},
                               ic_estimate=out.ic_estimate)
        return alpha_fn

    def run(self, md: MarketData, dates: Optional[pd.DatetimeIndex] = None, progress: bool = False,
            max_days: Optional[int] = None) -> PipelineResult:
        """Fit (if needed), backtest over ``dates`` (default: all out-of-sample
        dates; ``max_days`` keeps only the last ``max_days`` of them) and evaluate."""
        if self.alpha is None:
            self.fit(md)
        self._require_fitted()
        assert self.md_u is not None and self.risk is not None and self.cost is not None and self.optimizer is not None
        cfg = self.cfg
        dates = self.backtest_dates() if dates is None else pd.DatetimeIndex(dates)
        if max_days is not None:
            dates = dates[-int(max_days) - 1:] if len(dates) > int(max_days) + 1 else dates
        sim = DailyBarSimulator(cfg, self.cost)
        risk = self.risk
        alpha_fn = self._eligible_alpha_fn()

        def risk_fn(d: pd.Timestamp, alpha: Optional[pd.Series]) -> RiskModelOutput:
            return risk.as_of(d, alpha=alpha if cfg.risk.alignment else None)

        neutral_styles = [s for s in cfg.risk.styles if s in self.RISK_ONLY_STYLES]
        # Eligibility hook.  ``DailyBarSimulator.run`` reads ``md.eligible`` directly
        # when present (it is: ``apply_mask`` sets it) and also accepts
        # ``eligible_fn(t) -> Index`` of investable tickers; both are supplied so an
        # older/other simulator without the panel logic still gets the hook.
        # Without either, ineligible-but-alive names would reach the optimizer with
        # expected return 0 (they are absent from AlphaOutput) and be unwound by
        # the optimizer's trade-off rather than forced out; dead names are ``no_trade``.
        elig = self.md_u.eligibility()
        extra: Dict[str, Any] = {}
        params = inspect.signature(sim.run).parameters
        if "eligible_fn" in params:
            extra["eligible_fn"] = lambda d: elig.columns[elig.loc[d].to_numpy()]
        elif "eligibility" in params:
            extra["eligibility"] = elig
        t0 = time.perf_counter()
        bt = sim.run(self.md_u, dates, alpha_fn, risk_fn, self.optimizer, nav0=self.nav0,
                     rebalance_every=self.rebalance_every, venue=self.venue,
                     neutral_styles=neutral_styles, progress=progress, **extra)
        timings = dict(self.timings_, backtest=time.perf_counter() - t0)
        return self.evaluate(bt, timings, dates=dates)

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _turnover_on_return_dates(bt: BacktestResult, ret_index: pd.Index) -> pd.Series:
        """Turnover decided at ``t`` aligned to the return realised at ``t+1``:
        use the simulator's own view when present, else align positionally.
        Never a date join (decision and realisation dates differ by one day)."""
        fn = getattr(bt, "turnover_on_return_dates", None)
        if callable(fn):
            to = fn()
        else:
            n = min(len(bt.turnover), len(bt.returns))
            to = pd.Series(bt.turnover.to_numpy()[:n], index=bt.returns.index[:n], name="turnover")
        return to.reindex(ret_index).fillna(0.0)

    def _dsr_inputs(self, family: str, n_obs: int) -> Dict[str, float]:
        """``n_trials`` / cross-trial SR variance for the DSR from the ledger family.

        * no ledger or a single trial -> ``n_trials=1``, variance 0 (PSR against 0);
        * 2 trials -> raw count and their sample variance;
        * >= 3 trials -> Nyholt ``effective_trials_for`` (correlation of the
          stored monthly return streams) and the family's annual SR variance.
        The variance is floored at the pure sampling-noise variance of an annual
        SR over ``n_obs`` days (``252 / n_obs``) whenever more than one trial exists.
        """
        ppy = self.cfg.trading_days_per_year
        if self.ledger is None:
            return {"n_trials": 1.0, "var_trials_sr_annual": 0.0, "n_trials_raw": 1.0}
        n_raw = self.ledger.n_trials(family)
        if n_raw <= 1:
            return {"n_trials": 1.0, "var_trials_sr_annual": 0.0, "n_trials_raw": float(n_raw)}
        n_eff = self.ledger.effective_trials_for(family) if n_raw >= 3 else float(n_raw)
        var = self.ledger.family_sharpe_variance(family, annual=True, periods_per_year=ppy)
        var = max(var, ppy / max(n_obs, 2))
        return {"n_trials": float(n_eff), "var_trials_sr_annual": float(var), "n_trials_raw": float(n_raw)}

    def _pbo_from_grid(self, dates: pd.DatetimeIndex, base_net: pd.Series) -> Optional[float]:
        """CSCV PBO over ``cfg.pipeline_pbo_configs`` (list of ``{"section.field": value}``
        overrides) plus the base configuration; ``None`` when no grid is set."""
        grid = list(self.cfg.pipeline_pbo_configs or [])
        if not grid or self._md_input is None:
            return None
        cols = {"base": base_net}
        for i, overrides in enumerate(grid):
            cfg_i = copy.deepcopy(self.cfg)
            cfg_i.pipeline_pbo_configs = []
            for key, val in overrides.items():
                section, _, fld = key.partition(".")
                target = getattr(cfg_i, section) if fld else cfg_i
                setattr(target, fld or section, val)
            pipe = ResearchPipeline(cfg_i, use_ml=self.use_ml, phase=self.phase, composite_weight=self.composite_weight,
                                    ml_factory=self.ml_factory, rebalance_every=self.rebalance_every,
                                    venue=self.venue, nav0=self.nav0, warmup_days=self.warmup_days)
            res = pipe.run(self._md_input, dates=dates)
            cols[f"cfg{i}"] = res.backtest.returns["net"]
        Mx = pd.DataFrame(cols).dropna(how="any")
        n_groups = 8 if len(Mx) < 320 else 16
        if len(Mx) < 2 * n_groups:
            return None
        try:
            return float(probability_of_backtest_overfitting(Mx, n_groups=n_groups)["pbo"])
        except ValueError as e:           # identical columns etc.
            log.warning("PBO grid degenerate: %s", e)
            return None

    # ------------------------------------------------------------- evaluate
    def evaluate(self, bt: BacktestResult, timings: Optional[Dict[str, float]] = None,
                 dates: Optional[pd.DatetimeIndex] = None) -> PipelineResult:
        self._require_fitted()
        assert self.md_u is not None and self.alpha is not None
        cfg = self.cfg
        summ = bt.summary()
        net = bt.returns["net"].dropna()
        gross = bt.returns["gross"].dropna()
        elig = self.md_u.eligibility()

        # IC / decile spread of the live alpha score on non-overlapping formation
        # dates (every H days), eligible names only, vs the fitted (risk-only
        # residualised) target and the raw forward return respectively
        H = cfg.alpha.horizon_days
        target = self.alpha.target_
        sample_dates = bt.weights.index[::H]
        scores = pd.DataFrame({d: self.alpha.as_of(d).score for d in sample_dates}).T
        scores.index.name = "date"
        scores = scores.reindex(columns=self.md_u.tickers).where(elig.reindex(scores.index))
        icr = ic_report(scores, target.reindex(scores.index))          # ICIR annualised with inferred 252/H
        fwd_raw = forward_returns(self.md_u, H, cfg.alpha.delisting_return).where(elig).reindex(scores.index)
        ds = decile_spread(scores, fwd_raw, horizon=1,                 # formation dates do not overlap
                           weights=self.md_u.market_cap.reindex(scores.index),
                           periods_per_year=max(1, int(round(cfg.trading_days_per_year / H))))

        # fold-wise net Sharpe (calendar-year folds); standalone baseline is zero
        fold_sharpes = net.groupby(net.index.year).apply(lambda r: M.annualized_sharpe(r, cfg.trading_days_per_year))
        fold_sharpes = fold_sharpes[net.groupby(net.index.year).size() >= 60]

        to = self._turnover_on_return_dates(bt, gross.index)
        cs = M.cost_sensitivity(gross, to, cost_bps_per_side=summ.get("avg_cost_bps_per_side", 10.0))
        breakeven = M.breakeven_cost_bps(gross, to)

        # trial ledger: log first (identical re-runs update in place), then read the family statistics
        family = f"pipeline_phase{self.phase}"
        config_hash = cfg.hash()
        skew, kurt = float(net.skew()), float(net.kurt() + 3)
        if self.ledger is not None:
            monthly = (1.0 + net).groupby(net.index.to_period("M")).prod() - 1.0
            monthly.index = monthly.index.to_timestamp()
            self.ledger.log(name=family, family=family, config=cfg.to_dict(), config_hash=config_hash,
                            data_hash=self.data_hash_,
                            metrics={"sharpe_annual": summ["sharpe_net"], "n_obs": len(net), "skew": skew, "kurt": kurt,
                                     "mean_ic": icr.get("mean_ic"), "decile_t": ds.t_stat,
                                     "use_ml": self.use_ml, "phase": self.phase},
                            returns_monthly=monthly, notes="ResearchPipeline.run", replace_identical=True)
        dsr_in = self._dsr_inputs(family, len(net))
        dsr = deflated_sharpe_ratio_annual(summ["sharpe_net"], n_trials=max(int(round(dsr_in["n_trials"])), 1),
                                           var_trials_sr_annual=dsr_in["var_trials_sr_annual"], n_obs=len(net),
                                           skew=skew, kurt=kurt, periods_per_year=cfg.trading_days_per_year)

        pbo = self._pbo_from_grid(dates if dates is not None else bt.weights.index, net)

        by_year = icr.get("by_year")
        hurdles = HurdleReport(cfg.validation).evaluate_signal(
            ic_by_fold=(by_year["mean"].astype(float) if isinstance(by_year, pd.DataFrame) and len(by_year) else None),
            decile_tstat=ds.t_stat,
            net_sharpe_delta_by_fold=fold_sharpes,          # standalone: baseline 0 -> "fold net Sharpe > +0.05"
            breakeven_cost_bps=breakeven, modeled_cost_bps=summ.get("avg_cost_bps_per_side"),
            dsr=dsr, pbo=pbo, phase=self.phase, net_sharpe=summ["sharpe_net"], standalone=True)

        notes = []
        if summ["sharpe_net"] > 2:
            notes.append("net Sharpe > 2: treat as suspect until leakage/cost/selection ruled out (plan 2.4)")
        ehd = summ.get("effective_holding_days", np.nan)
        if np.isfinite(ehd) and ehd < 0.6 * cfg.portfolio.effective_holding_days_target:
            notes.append(f"effective holding {ehd:.0f}d < target {cfg.portfolio.effective_holding_days_target}d: raise trade_aversion")
        if pbo is None:
            notes.append("PBO skipped: set SystemConfig.pipeline_pbo_configs to run a CSCV grid")
        return PipelineResult(backtest=bt, summary=summ, ic_report=icr, hurdles=hurdles,
                              cost_sensitivity=cs, fold_sharpes=fold_sharpes, timings=timings or {}, notes=notes,
                              decile=ds, dsr=float(dsr), dsr_inputs=dsr_in, pbo=pbo,
                              config_hash=config_hash, data_hash=self.data_hash_)


def format_result(res: PipelineResult) -> str:
    s = res.summary
    icr = res.ic_report
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
        "=== Alpha IC (score vs residualised target, sampled every horizon) ===",
        f"mean rank IC {icr.get('mean_ic', float('nan')):.4f}  ICIR {icr.get('icir', float('nan')):.2f} "
        f"(annualised at {icr.get('periods_per_year', float('nan')):.1f}/yr)  t {icr.get('t_stat', float('nan')):.2f}  n {icr.get('n', 0)}",
    ]
    if res.decile is not None:
        lines.append(f"decile L/S (VW)  mean {100*res.decile.mean:.2f}%/period  Sharpe {res.decile.sharpe:.2f}  NW t {res.decile.t_stat:.2f}")
    lines += [
        f"DSR {res.dsr:.3f}  (n_trials {res.dsr_inputs.get('n_trials', 1):.1f}, var SR {res.dsr_inputs.get('var_trials_sr_annual', 0):.3f})"
        + (f"  PBO {res.pbo:.3f}" if res.pbo is not None else "  PBO skipped"),
        f"config {res.config_hash}  data {res.data_hash}",
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
