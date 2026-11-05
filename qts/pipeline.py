"""End-to-end research pipeline: universe -> risk -> alpha -> cost -> portfolio -> backtest -> validation.

This is the "thin vertical slice" the review asked for: every layer is wired
together on a daily-bar simulator so integration problems surface before any
single layer is deepened.  It runs on the synthetic market out of the box and
on a :class:`~qts.data.store.ParquetStore` once real data has been loaded.

Usage::

    from qts.pipeline import ResearchPipeline, format_result, load_run
    from qts.data.synthetic import generate, SyntheticSpec

    md = generate(SyntheticSpec(n_stocks=300, n_days=1512))
    res = ResearchPipeline(runs_root="runs").run(md)
    print(format_result(res))  # backtest statistics + plan gates
    rec = load_run(res.run_dir)  # typed re-load of the run directory

or from the shell: ``python -m qts.cli demo`` (see :mod:`qts.cli`).

Eligibility contract
--------------------
``apply_mask`` stores the point-in-time universe on ``md.eligible``; the alpha
layer scores only names in ``md.eligibility()`` at each date (ineligible and
dead names are *absent* from ``AlphaOutput.score``, never zero-filled).  The
pipeline additionally wraps ``alpha_fn`` so that nothing outside the eligible
set reaches the optimizer with a non-zero expected return, and forwards the
eligibility panel to ``DailyBarSimulator.run`` when the simulator exposes an
``eligible_fn`` / ``eligibility`` keyword (see :meth:`ResearchPipeline.run`).
``ExecutionConfig.signal_lag_days`` (design decision D4) is forwarded the same
way (``signal_lag_days`` keyword, detected by signature inspection).  The
simulator prices trades at the ``"routed"`` venue (D3) with
``TransactionCostModel(cfg.cost, cfg.execution)``.

Signal half-life (fix #5): ``DailyBarSimulator.run`` passes
``AlphaOutput.half_life_days`` to
``SinglePeriodOptimizer.solve(alpha_half_life_days=...)``, where it drives the
Garleanu-Pedersen shrink when ``PortfolioConfig.use_gp_shrink`` is true.

Label contract (design decision D1)
-----------------------------------
Labels are residualised on **exactly the set the optimizer neutralises**:
market + industries + ``qts.core.roles.RISK_ONLY_STYLES`` (beta, size,
residual vol, liquidity, short-term reversal) — never on the alpha themes
(``roles.ALPHA_THEME_STYLES``).  The same constant drives
``neutral_styles`` passed to the simulator/optimizer, so the label and the
hedge cannot drift apart.  A consequence (intended): the low-risk theme's
linear premium is removed from the ML target, because the beta-neutral,
residual-vol-banded book could not harvest it.  During the risk model's
warm-up the same risk-only style set is built internally
(``qts.alpha.targets.default_exposures`` over ``roles.RISK_ONLY_FEATURES``);
any other failure of the exposures hook raises instead of silently changing
the label definition.

Hurdle inputs (BUILD_PLAN §2.4)
-------------------------------
* ``t_stat``: Newey–West t (lags ``max(1, floor(4 (T/100)^(2/9)))``) of the
  **top-minus-bottom decile** of raw forward returns on non-overlapping
  formation dates (every ``H`` days), eligible names only, value-weighted.
* ``ic_sign``: mean sampled IC per ``PurgedWalkForward`` test block (the same
  blocks as ``net_sharpe_delta``, below) > 0 in >= 2/3 of blocks.
* ``net_sharpe_delta``: for a standalone run the baseline is zero, so the gate
  reads "net Sharpe by fold >= +0.05 in >= 2/3 of folds".  Folds are the test
  blocks of ``PurgedWalkForward(n_splits=cfg.validation.n_splits,
  embargo_days=cfg.validation.embargo_days, label_horizon=H, min_train_days=0)``
  over the backtest's return dates (handbook fix #27; previously calendar
  years).  Only the contiguous test blocks are used: every backtest date is
  already out-of-sample (walk-forward ML with embargo), so the purge/embargo
  of the splitter's *train* side has nothing to act on here.  Blocks shorter
  than ``cfg.validation.min_fold_days`` are avoided by reducing the number of
  splits (none -> gate skipped).
* ``dsr``: ``n_trials`` and the cross-trial SR variance come from the trial
  ledger family ``pipeline_phase{p}:{data_hash[:12]}`` (``effective_trials_for``
  when >= 3 trials, used unrounded; single-trial PSR when the ledger is empty /
  absent).  A re-run is de-duplicated (counted once) only when name, config
  hash, data hash, a *clean* code version (git sha; ``+dirty`` / ``unknown``
  never de-duplicate) and the hash of :meth:`ResearchPipeline.run_spec`
  (``use_ml``, ``ml_factory`` identity, ``composite_weight``,
  ``rebalance_every``, ``venue``, ``nav0``, ``warmup_days``, backtest dates,
  signal lag) all match; see :class:`qts.validation.ledger.TrialLedger`.
* ``pbo``: only computed when ``SystemConfig.pipeline_pbo_configs`` lists a
  configuration grid (CSCV over the grid's net return streams); otherwise the
  gate is reported as *skipped*, which does not fail ``overall``.  Every grid
  configuration is logged to the ledger as a trial of the same family before
  the DSR inputs are read.

Signal lag and the evaluation timeline
--------------------------------------
With ``L = cfg.execution.signal_lag_days`` the alpha labels are aligned to the
trade: the label of decision date ``s`` is ``close_{s+L} -> close_{s+L+h}``,
the walk-forward purge and trailing-IC windows grow by ``L``
(``qts.alpha.targets``).  :meth:`ResearchPipeline.evaluate` scores, at every
sampled execution date ``t``, the score that was traded there
(``alpha.as_of(t - L)``) against that signal date's label / raw forward return
(which start at ``close_t``); IC, decile and hurdle frames are indexed by ``t``.
The decile Sharpe and the ICIR share one inferred sampling frequency.
* Required gates (fail when missing): ``phase_sharpe``, ``suspect``, ``dsr``.

IC decay
--------
``PipelineResult.ic_report.decay`` holds the mean rank IC of the sampled score
against risk-only residualised labels at ``horizon_days`` and every
``AlphaConfig.secondary_horizons`` entry (default 5, 20, 60 days); the
secondary labels are computed on the sampled dates only.

Run directory layout (run registry; backs the future UI API)
------------------------------------------------------------
With ``runs_root`` set, :meth:`ResearchPipeline.run` writes
``<runs_root>/<run_id>/`` (``run_id = <UTC timestamp>_<config_hash[:8]>``)::

    manifest.json     RunManifest: run_id, created_at, git sha/dirty, config_hash,
                      data_hash, library versions, python, platform, timings, seeds
    config.yaml       SystemConfig.to_yaml() (reload: SystemConfig.from_yaml)
    summary.json      backtest summary + IC scalars + decile + DSR/PBO + fold Sharpes
                      + IC decay + notes + flags (use_ml, phase, signal_lag_days)
    nav.parquet       column "nav", DatetimeIndex
    weights.parquet   date x ticker post-trade weights (decision dates)
    trades.parquet    date x ticker weight changes (decision dates)
    costs.parquet     date x {spread, impact, fees, borrow, total} USD
    returns.parquet   date x {gross, net, gross_after_borrow} (realisation dates;
                      gross_after_borrow = gross - borrow / NAV, the base of the
                      per-side cost gates)
    ic.parquet        column "rank_ic": sampled per-date rank IC
    hurdles.json      HurdleResult.to_dict()

Written next to them by :func:`qts.runs.artifacts.write_extra_artifacts` (best
effort, for the workbench API; layout in that module's docstring)::

    extras.json         run facts (data kind, universe size, market days, seed,
                        execution venue, label), DSR moments, IC by calendar year,
                        factor roles
    exante_vol.parquet  annualised ex-ante vol of the post-trade book
    exposures.parquet   decision date x factor net book exposure
    holdings.parquet    month-end holdings (weight, alpha, specific vol, industry)

``summary.json`` is written right after ``manifest.json``: a directory without it
is an interrupted run (``qts runs list`` shows it as unreadable).

:func:`load_run` returns a typed :class:`RunRecord`; :func:`list_runs` lists
run directories newest first.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from qts.alpha import AlphaModel
from qts.alpha.evaluation import (
    DecileSpreadResult,
    ICReport,
    decile_spread,
    ic_report,
    infer_periods_per_year,
)
from qts.alpha.targets import forward_returns, residualize_forward_returns
from qts.backtest import BacktestResult, DailyBarSimulator
from qts.backtest import metrics as M
from qts.core.config import SystemConfig
from qts.core.manifest import RunManifest, hash_market_data
from qts.core.roles import ALPHA_THEME_STYLES, RISK_ONLY_STYLES
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput
from qts.data.universe import apply_mask, build_universe_mask, masked_levels
from qts.portfolio import SinglePeriodOptimizer
from qts.risk import BarraStyleRiskModel, HybridRiskModel, build_risk_model
from qts.tcost import TransactionCostModel
from qts.validation import (
    HurdleReport,
    HurdleResult,
    PurgedWalkForward,
    TrialLedger,
    deflated_sharpe_ratio_annual,
)
from qts.validation.ledger import callable_identity
from qts.validation.pbo import probability_of_backtest_overfitting

log = logging.getLogger(__name__)

RiskModel = BarraStyleRiskModel | HybridRiskModel


@dataclass
class PipelineResult:
    """Everything one :meth:`ResearchPipeline.run` produced (typed).

    ``summary``: ``BacktestResult.summary()`` (flat str -> float/str dict).
    ``ic_report``: :class:`~qts.alpha.evaluation.ICReport` of the sampled score.
    ``hurdles``: :class:`~qts.validation.hurdles.HurdleResult`.
    ``fold_sharpes``: annualised net Sharpe per ``PurgedWalkForward`` test block,
    indexed by block start date; ``folds``: start/end/n_days/sharpe per block.
    """

    backtest: BacktestResult
    summary: dict[str, Any]
    ic_report: ICReport
    hurdles: HurdleResult
    cost_sensitivity: pd.DataFrame
    fold_sharpes: pd.Series
    folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    decile: DecileSpreadResult | None = None
    dsr: float = float("nan")
    dsr_inputs: dict[str, float] = field(default_factory=dict)
    pbo: float | None = None
    breakeven_cost_bps: float = float("nan")
    config_hash: str = ""
    data_hash: str = ""
    git_sha: str = "unknown"
    use_ml: bool = False
    phase: int = 1
    signal_lag_days: int = 0
    manifest: RunManifest | None = None
    run_dir: str | None = None


def data_hash_of(md: MarketData) -> str:
    """Data identity written to the trial ledger and the run manifest: SHA-256
    over **all** ``MarketData`` panels (incl. optional ones and ``extra``) and the
    eligibility panel (handbook fix #37) — see :func:`qts.core.manifest.hash_market_data`."""
    return hash_market_data(md)


@dataclass(frozen=True)
class WarmupEstimate:
    """Trading days of a market panel consumed before the first out-of-sample
    backtest date (:func:`estimate_warmup_days`)."""

    days: int  #: total warm-up = max(risk_days, ml_days) + signal_lag_days
    risk_days: int  #: risk exposures warm-up + factor-return history (exact)
    ml_days: int | None  #: universe lead + ML training minimum + embargo (None without ML)
    signal_lag_days: int  #: D4 execution lag added at the end
    exact: bool  #: True when the risk term binds (exact); ML term is a conservative estimate


def estimate_warmup_days(
    cfg: SystemConfig, use_ml: bool, risk_warmup_days: int | None = None
) -> WarmupEstimate:
    """Warm-up implied by the config, i.e. the position in ``md.dates`` of the
    first date :meth:`ResearchPipeline.backtest_dates` returns on a panel whose
    names all have data from its first day.

    With ``m = cfg.risk.min_history_days``, ``W`` = the risk model's exposure
    warm-up (``risk_warmup_days``, default ``m``), ``L =
    cfg.execution.signal_lag_days``, ``h = cfg.alpha.embargo_days or
    cfg.alpha.horizon_days``:

    * risk: exposures start at position ``W``; factor returns one day later
      (the first return needs the previous day's exposures); the backtest needs
      ``m`` of them -> ``risk_days = W + m + 1`` (exact; 505 for the defaults).
    * ML (``use_ml``): the first walk-forward fit is ``train_min_days + h + L``
      feature dates after the first feature date
      (:meth:`qts.alpha.trainer.WalkForwardTrainer.schedule`); the first feature
      date is the first universe date, at most ``max(5, adv_window // 2) - 1 +
      refresh_days`` positions in (the ADV window's ``min_periods``, then up to
      one refresh period) -> ``ml_days`` is an upper bound (828 for the
      defaults with ``L = 1``, total 829; the synthetic market starts at 820).
    * total ``days = max(risk_days, ml_days) + L`` (the decision at ``t`` reads
      the signal of ``t - L``, which must itself be out of sample).

    Data that start later per name (late listings, gaps) only move the start
    further; this is the floor a launch form should enforce.
    """
    m = int(cfg.risk.min_history_days)
    w = m if risk_warmup_days is None else int(risk_warmup_days)
    lag = int(getattr(cfg.execution, "signal_lag_days", 0) or 0)
    risk_days = w + m + 1
    ml_days: int | None = None
    if use_ml:
        u = cfg.universe
        lead = max(5, int(u.adv_window) // 2) - 1 + int(u.refresh_days)
        h = int(cfg.alpha.embargo_days or cfg.alpha.horizon_days)
        ml_days = lead + int(cfg.alpha.train_min_days) + h + lag
    days = max(risk_days, ml_days or 0) + lag
    return WarmupEstimate(
        days=days,
        risk_days=risk_days,
        ml_days=ml_days,
        signal_lag_days=lag,
        exact=ml_days is None or risk_days >= ml_days,
    )


def purged_fold_blocks(
    dates: pd.DatetimeIndex, n_splits: int, embargo_days: int, label_horizon: int, min_fold_days: int = 20
) -> list[pd.DatetimeIndex]:
    """Contiguous test blocks of ``PurgedWalkForward`` over ``dates``.

    ``n_splits`` is reduced until every block has at least ``min_fold_days``
    dates; returns ``[]`` if even one block would be shorter."""
    n = len(dates)
    k = min(int(n_splits), n // max(int(min_fold_days), 1))
    if k < 1:
        return []
    pwf = PurgedWalkForward(
        n_splits=k, embargo_days=int(embargo_days), label_horizon=int(label_horizon), min_train_days=0
    )
    return [dates[t0 : t1 + 1] for t0, t1 in pwf.test_blocks(dates)]


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
    composite_weight : float | None
        Override of ``cfg.alpha.composite_weight``.
    venue : str
        Simulator pricing venue; default ``"routed"`` (design decision D3: the
        part of each trade up to the auction cap at the close auction, the rest
        in the continuous session, as the router would execute it).
    runs_root : str | None
        When set, every :meth:`run` writes a run directory under it (layout in
        the module docstring).
    """

    #: D1: the single risk-only set (``qts.core.roles``); kept as a class
    #: attribute for backward compatibility.
    RISK_ONLY_STYLES = RISK_ONLY_STYLES

    def __init__(
        self,
        cfg: SystemConfig | None = None,
        use_ml: bool = True,
        phase: int | None = None,
        ledger_path: str | None = None,
        composite_weight: float | None = None,
        ml_factory: Callable[[], object] | None = None,
        rebalance_every: int = 1,
        venue: str = "routed",
        nav0: float = 1_000_000.0,
        warmup_days: int | None = None,
        runs_root: str | None = None,
    ):
        self.cfg = cfg or SystemConfig()
        self.use_ml = use_ml
        self.phase = phase if phase is not None else (2 if use_ml else 1)
        self.ledger_path = ledger_path
        self.ledger = TrialLedger(ledger_path) if ledger_path else None
        self.composite_weight = composite_weight
        self.ml_factory = ml_factory
        self.rebalance_every = rebalance_every
        self.venue = venue
        self.nav0 = nav0
        self.warmup_days = warmup_days
        self.runs_root = runs_root

        # fitted components (exposed for inspection)
        self.universe_mask: pd.DataFrame | None = None
        self.md_u: MarketData | None = None
        self.risk: RiskModel | None = None
        self.alpha: AlphaModel | None = None
        self.cost: TransactionCostModel | None = None
        self.optimizer: SinglePeriodOptimizer | None = None
        self.timings_: dict[str, float] = {}
        self.data_hash_: str = ""
        self._md_input: MarketData | None = None
        self._risk_first_date: pd.Timestamp | None = None
        self._warned_warmup = False

    # ------------------------------------------------------------------ fit
    def fit(self, md: MarketData) -> ResearchPipeline:
        cfg = self.cfg
        t: dict[str, float] = {}
        self._md_input = md
        t0 = time.perf_counter()
        self.data_hash_ = data_hash_of(md)
        t["data_hash"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.universe_mask = build_universe_mask(md, cfg.universe)
        self.md_u = apply_mask(md, self.universe_mask)
        t["universe"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # apply_mask keeps full level history (audit 2 #6); the risk model's
        # estimation universe is still "has a market cap", so it gets the view with
        # levels NaN outside the universe (unchanged risk-model behaviour)
        self.risk = build_risk_model(cfg.risk, warmup_days=self.warmup_days).fit(masked_levels(self.md_u))
        self._risk_first_date = self._exposures_first_date()
        self._warned_warmup = False
        t["risk_fit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.alpha = AlphaModel(
            cfg.alpha,
            composite_weight=self.composite_weight,
            ml_factory=self.ml_factory,
            use_ml=self.use_ml,
            delisting_return=cfg.cost.delisting_haircut,
            # labels / purge / trailing IC aligned to what is traded (execution lag L)
            signal_lag_days=self.signal_lag_days(),
        )
        # residualize labels on the *risk model's* risk-only exposures (D1; plan 2.3: targets)
        self.alpha.fit(self.md_u, risk_exposures_fn=self._exposures_fn)
        t["alpha_fit"] = time.perf_counter() - t0

        self.cost = TransactionCostModel(cfg.cost, cfg.execution)  # routed-venue auction cap (D3)
        self.optimizer = SinglePeriodOptimizer(
            cfg.portfolio, self.cost, horizon_days=cfg.alpha.horizon_days, exec_cfg=cfg.execution
        )
        self.timings_ = t
        return self

    def _exposures_first_date(self) -> pd.Timestamp | None:
        """First date of the risk model's exposure history (``ExposureHistory.first_date``)."""
        assert self.risk is not None
        ex = getattr(self.risk, "exposures", None)
        fd = getattr(ex, "first_date", None)
        if fd is not None or ex is None:
            return None if fd is None else pd.Timestamp(fd)
        keys = list(ex.keys()) if isinstance(ex, dict) else []
        return min(keys) if keys else None

    def _alpha_theme_styles(self) -> set:
        """Styles never used for label residualisation: the alpha themes of
        ``roles`` plus any configured risk style that is not risk-only."""
        return set(ALPHA_THEME_STYLES) | {s for s in self.cfg.risk.styles if s not in RISK_ONLY_STYLES}

    def neutral_styles(self) -> list[str]:
        """Styles the optimizer neutralises (D1: the same set the label is residualised on)."""
        return [s for s in self.cfg.risk.styles if s in RISK_ONLY_STYLES]

    def _exposures_fn(self, date: pd.Timestamp) -> pd.DataFrame | None:
        """Risk-only exposures (market + industries + ``RISK_ONLY_STYLES``) of the
        fitted risk model as of ``date``.

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
                log.info(
                    "labels before %s (risk-model warm-up) use the internal risk-only exposures "
                    "(beta, size, idio vol, liquidity, reversal + industries)",
                    None if self._risk_first_date is None else self._risk_first_date.date(),
                )
                self._warned_warmup = True
            return None
        X = self.risk.exposures_as_of(date)  # raises on anything unexpected
        drop = self._alpha_theme_styles()
        return X.loc[:, [c for c in X.columns if c not in drop]]

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
        # D4: with signal_lag_days = L the decision at t reads alpha/risk as of the
        # trading day L days earlier, which must itself be an out-of-sample date
        # (otherwise risk.as_of raises for L >= 2 -- lag investigation 2026-11-05).
        lag = self.signal_lag_days()
        return dates[dates >= start][lag:][:-1]

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
            return dataclasses.replace(
                out,
                score=out.score.loc[keep],
                expected_return=out.expected_return.loc[keep],
                components={k: v.reindex(keep) for k, v in out.components.items()},
            )

        return alpha_fn

    def signal_lag_days(self) -> int:
        """D4: ``cfg.execution.signal_lag_days`` (0 when the field does not exist yet)."""
        return int(getattr(self.cfg.execution, "signal_lag_days", 0) or 0)

    def run(
        self,
        md: MarketData,
        dates: pd.DatetimeIndex | None = None,
        progress: bool = False,
        max_days: int | None = None,
    ) -> PipelineResult:
        """Fit (if needed), backtest over ``dates`` (default: all out-of-sample
        dates; ``max_days`` keeps only the last ``max_days`` of them), evaluate
        and — when ``runs_root`` is set — write the run directory."""
        t_run = time.perf_counter()
        if self.alpha is None or self.alpha.signal_lag_days != self.signal_lag_days():
            # labels / purge / IC windows are aligned to the execution lag at fit
            # time: a changed cfg.execution.signal_lag_days needs a refit
            self.fit(md)
        self._require_fitted()
        assert (
            self.md_u is not None
            and self.risk is not None
            and self.cost is not None
            and self.optimizer is not None
        )
        cfg = self.cfg
        dates = self.backtest_dates() if dates is None else pd.DatetimeIndex(dates)
        if max_days is not None:
            dates = dates[-int(max_days) - 1 :] if len(dates) > int(max_days) + 1 else dates
        sim = DailyBarSimulator(cfg, self.cost)
        risk = self.risk
        # half-life wiring (fix #5): DailyBarSimulator.run passes AlphaOutput.half_life_days
        # to optimizer.solve(alpha_half_life_days=...)
        alpha_fn = self._eligible_alpha_fn()

        def risk_fn(d: pd.Timestamp, alpha: pd.Series | None) -> RiskModelOutput:
            return risk.as_of(d, alpha=alpha if cfg.risk.alignment else None)

        # Eligibility hook.  ``DailyBarSimulator.run`` reads ``md.eligible`` directly
        # when present (it is: ``apply_mask`` sets it) and also accepts
        # ``eligible_fn(t) -> Index`` of investable tickers; both are supplied so an
        # older/other simulator without the panel logic still gets the hook.
        # Without either, ineligible-but-alive names would reach the optimizer with
        # expected return 0 (they are absent from AlphaOutput) and be unwound by
        # the optimizer's trade-off rather than forced out; dead names are ``no_trade``.
        elig = self.md_u.eligibility()
        extra: dict[str, Any] = {}
        params = inspect.signature(sim.run).parameters
        if "eligible_fn" in params:
            extra["eligible_fn"] = lambda d: elig.columns[elig.loc[d].to_numpy()]
        elif "eligibility" in params:
            extra["eligibility"] = elig
        # D4 execution delay: forwarded when the simulator supports it.
        lag = self.signal_lag_days()
        lag_note = None
        if "signal_lag_days" in params:
            extra["signal_lag_days"] = lag
        elif lag:
            lag_note = (
                f"signal_lag_days={lag} requested but DailyBarSimulator.run has no signal_lag_days "
                "argument: ran with lag 0"
            )
            log.warning(lag_note)
            lag = 0
        t0 = time.perf_counter()
        bt = sim.run(
            self.md_u,
            dates,
            alpha_fn,
            risk_fn,
            self.optimizer,
            nav0=self.nav0,
            rebalance_every=self.rebalance_every,
            venue=self.venue,
            neutral_styles=self.neutral_styles(),
            progress=progress,
            **extra,
        )
        timings = dict(self.timings_, backtest=time.perf_counter() - t0)
        res = self.evaluate(bt, timings, dates=dates, signal_lag_days=lag)
        if lag_note:
            res.notes.append(lag_note)
        res.timings["total_run"] = time.perf_counter() - t_run
        res.manifest = RunManifest.create(
            config_hash=res.config_hash, data_hash=res.data_hash, timings=res.timings, seeds=self._seeds()
        )
        res.git_sha = res.manifest.git_sha
        if self.runs_root:
            res.run_dir = str(save_run(res, self.cfg, self.runs_root))
            __import__("qts.runs.artifacts", fromlist=["_"]).write_extra_artifacts(res.run_dir, res, self)
        return res

    def _seeds(self) -> dict[str, int]:
        seeds = {
            "system": int(self.cfg.seed),
            "model_base_seed": 0,
            "model_seeds": int(self.cfg.alpha.model_seeds),
        }
        return seeds

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

    def ledger_family(self) -> str:
        """DSR trial family: ``pipeline_phase{p}:{data_hash[:12]}`` (one search per
        phase *and* data set; signal-lag and config variants pool within it)."""
        return f"pipeline_phase{self.phase}:{(self.data_hash_ or 'nodata')[:12]}"

    def run_spec(self, dates: pd.Index | None, signal_lag_days: int) -> dict[str, Any]:
        """Everything that changes a run's numbers but is not in ``SystemConfig``
        (constructor knobs, backtest dates, the effective signal lag).  Hashed into
        the trial ledger's de-duplication key (audit 2 #2)."""
        d = pd.DatetimeIndex([] if dates is None else dates)
        return {
            "use_ml": bool(self.use_ml),
            "phase": int(self.phase),
            "ml_factory": callable_identity(self.ml_factory),
            "composite_weight": None if self.composite_weight is None else float(self.composite_weight),
            "rebalance_every": int(self.rebalance_every),
            "venue": str(self.venue),
            "nav0": float(self.nav0),
            "warmup_days": None if self.warmup_days is None else int(self.warmup_days),
            "dates": [str(d[0].date()), str(d[-1].date()), len(d)] if len(d) else [],
            "signal_lag_days": int(signal_lag_days),
        }

    def _log_trial(
        self,
        family: str,
        config_hash: str,
        net: pd.Series,
        summ: dict[str, Any],
        mean_ic: float,
        decile_t: float,
        signal_lag_days: int,
        dates: pd.Index | None,
        bt: BacktestResult,
    ) -> None:
        from qts.core.manifest import code_version

        assert self.ledger is not None
        monthly = (1.0 + net).groupby(net.index.to_period("M")).prod() - 1.0
        monthly.index = monthly.index.to_timestamp()
        self.ledger.log(
            name=family,
            family=family,
            config=self.cfg.to_dict(),
            config_hash=config_hash,
            data_hash=self.data_hash_,
            code_version=code_version(),
            metrics={
                "sharpe_annual": summ["sharpe_net"],
                "n_obs": len(net),
                "skew": float(net.skew()),
                "kurt": float(net.kurt() + 3),
                "mean_ic": mean_ic,
                "decile_t": decile_t,
                "use_ml": self.use_ml,
                "phase": self.phase,
                "signal_lag_days": int(signal_lag_days),
            },
            returns_monthly=monthly,
            notes="ResearchPipeline.run",
            replace_identical=True,
            run_spec=self.run_spec(dates if dates is not None else bt.weights.index, signal_lag_days),
        )

    @staticmethod
    def _ic_by_fold(ic: pd.Series, blocks: list[pd.DatetimeIndex]) -> pd.Series:
        """Mean sampled IC per purged walk-forward block (indexed by block start);
        blocks without a finite IC are dropped."""
        ic = ic.dropna()
        out = {}
        for b in blocks:
            sel = ic[(ic.index >= b[0]) & (ic.index <= b[-1])]
            if len(sel):
                out[b[0]] = float(sel.mean())
        return pd.Series(out, dtype=float, name="mean_ic")

    def _dsr_inputs(self, family: str, n_obs: int) -> dict[str, float]:
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

    def _pbo_from_grid(self, dates: pd.DatetimeIndex, base_net: pd.Series) -> float | None:
        """CSCV PBO over ``cfg.pipeline_pbo_configs`` (list of ``{"section.field": value}``
        overrides) plus the base configuration; ``None`` when no grid is set."""
        grid = list(self.cfg.pipeline_pbo_configs or [])
        if not grid or self._md_input is None:
            return None
        cols = {"base": base_net}
        for i, overrides in enumerate(grid):
            cfg_i = self.cfg.with_overrides(dict(overrides, pipeline_pbo_configs=[]))
            pipe = ResearchPipeline(
                cfg_i,
                use_ml=self.use_ml,
                phase=self.phase,
                composite_weight=self.composite_weight,
                ml_factory=self.ml_factory,
                rebalance_every=self.rebalance_every,
                venue=self.venue,
                nav0=self.nav0,
                warmup_days=self.warmup_days,
                # every grid configuration is a trial of the same search (audit 2 #11)
                ledger_path=self.ledger_path,
            )
            res = pipe.run(self._md_input, dates=dates)
            cols[f"cfg{i}"] = res.backtest.returns["net"]
        Mx = pd.DataFrame(cols).dropna(how="any")
        n_groups = 8 if len(Mx) < 320 else 16
        if len(Mx) < 2 * n_groups:
            return None
        try:
            return float(probability_of_backtest_overfitting(Mx, n_groups=n_groups)["pbo"])
        except ValueError as e:  # identical columns etc.
            log.warning("PBO grid degenerate: %s", e)
            return None

    def _decay_targets(self, sample_dates: pd.DatetimeIndex) -> dict[int, pd.DataFrame]:
        """Risk-only residualised labels at ``horizon_days`` and each secondary
        horizon, computed on ``sample_dates`` (signal dates) only (D1 exposures),
        aligned to the execution lag like the training label."""
        assert self.alpha is not None and self.md_u is not None
        acfg = self.cfg.alpha
        out: dict[int, pd.DataFrame] = {int(acfg.horizon_days): self.alpha.target_}
        for h in sorted({int(x) for x in (acfg.secondary_horizons or [])} - {int(acfg.horizon_days)}):
            if h <= 0:
                continue
            out[h] = residualize_forward_returns(
                self.md_u,
                h,
                exposures_fn=self._exposures_fn,
                features=self.alpha.raw_features_,
                winsor_pct=acfg.winsor_pct,
                delisting_return=self.cfg.cost.delisting_haircut,
                only_dates=sample_dates,
                label_lag_days=self.alpha.signal_lag_days,
                delisting_gap_days=acfg.delisting_gap_days,
            )
        return out

    # ------------------------------------------------------------- evaluate
    def evaluate(
        self,
        bt: BacktestResult,
        timings: dict[str, float] | None = None,
        dates: pd.DatetimeIndex | None = None,
        signal_lag_days: int = 0,
    ) -> PipelineResult:
        self._require_fitted()
        assert self.md_u is not None and self.alpha is not None
        cfg = self.cfg
        summ = bt.summary()
        net = bt.returns["net"].dropna()
        gross = bt.returns["gross"].dropna()
        elig = self.md_u.eligibility()
        t0 = time.perf_counter()

        # IC / decile spread of the *traded* alpha score on non-overlapping formation
        # dates (every H execution dates), eligible names only, vs the fitted
        # (risk-only residualised) target and the raw forward return respectively.
        # With signal lag L the book executed at t holds the score as_of(t - L);
        # the label of that signal date (targets.py) is the return from close_t,
        # so score, label and raw return all refer to the trade made at t.  Frames
        # are indexed by the execution date t (L = 0: unchanged).
        H = cfg.alpha.horizon_days
        L = int(signal_lag_days)
        target = self.alpha.target_
        exec_dates = pd.DatetimeIndex(bt.weights.index[::H])
        dpos = self.md_u.dates.get_indexer(exec_dates)
        if (dpos < L).any():
            raise ValueError("evaluate(): an execution date has no signal date L days earlier")
        sig_dates = self.md_u.dates[dpos - L]

        def at_signal(panel: pd.DataFrame) -> pd.DataFrame:
            out = panel.reindex(index=sig_dates, columns=self.md_u.tickers)
            out.index = exec_dates
            out.index.name = "date"
            return out

        scores = pd.DataFrame(
            {t: self.alpha.as_of(s).score for t, s in zip(exec_dates, sig_dates, strict=True)}
        ).T
        scores.index.name = "date"
        scores = scores.reindex(columns=self.md_u.tickers).where(at_signal(elig).fillna(False).astype(bool))
        # one sampling frequency for the ICIR and the decile Sharpe (audit 2 #15)
        ppy = infer_periods_per_year(scores.index, cfg.trading_days_per_year)
        decay_targets = {h: at_signal(t) for h, t in self._decay_targets(pd.DatetimeIndex(sig_dates)).items()}
        icr = ic_report(
            scores,
            at_signal(target),
            periods_per_year=ppy,
            trading_days_per_year=cfg.trading_days_per_year,
            overlap=1,
            decay_targets=decay_targets,
        )
        fwd_raw = at_signal(
            forward_returns(
                self.md_u,
                H,
                cfg.cost.delisting_haircut,
                label_lag_days=L,
                delisting_gap_days=cfg.alpha.delisting_gap_days,
            ).where(elig)
        )
        ds = decile_spread(
            scores,
            fwd_raw,
            horizon=1,  # formation dates do not overlap
            weights=at_signal(self.md_u.market_cap),
            periods_per_year=ppy,
        )

        # fold-wise net Sharpe on PurgedWalkForward test blocks (#27); standalone baseline is zero
        v = cfg.validation
        blocks = purged_fold_blocks(
            pd.DatetimeIndex(net.index), v.n_splits, v.embargo_days, H, getattr(v, "min_fold_days", 20)
        )
        rows = [
            {
                "start": b[0],
                "end": b[-1],
                "n_days": len(b),
                "sharpe": M.annualized_sharpe(net.loc[b], cfg.trading_days_per_year),
            }
            for b in blocks
        ]
        folds = pd.DataFrame(rows, columns=["start", "end", "n_days", "sharpe"])
        fold_sharpes = pd.Series(
            folds["sharpe"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(folds["start"], name="fold_start"),
            name="net_sharpe",
        )

        # breakeven / cost sensitivity on gross returns net of borrow (borrow is not a
        # per-side cost; consistent with BacktestResult.summary())
        gab = getattr(bt, "gross_after_borrow_returns", None)
        gross_cs = gab.dropna() if isinstance(gab, pd.Series) and len(gab.dropna()) else gross
        to = self._turnover_on_return_dates(bt, gross_cs.index)
        cs = M.cost_sensitivity(gross_cs, to, cost_bps_per_side=summ.get("avg_cost_bps_per_side", 10.0))
        breakeven = M.breakeven_cost_bps(gross_cs, to)

        # trial ledger: log this run first, then the PBO grid's configurations (each
        # a trial of the same search, audit 2 #11), then read the family statistics.
        # Family = phase + data hash: trials on other data (seeds, real data) are a
        # different search (audit 2 #17).
        family = self.ledger_family()
        config_hash = cfg.hash()
        skew, kurt = float(net.skew()), float(net.kurt() + 3)
        if self.ledger is not None:
            self._log_trial(
                family, config_hash, net, summ, icr.mean_ic, ds.t_stat, signal_lag_days, dates, bt
            )
        pbo = self._pbo_from_grid(dates if dates is not None else bt.weights.index, net)
        dsr_in = self._dsr_inputs(family, len(net))
        dsr = deflated_sharpe_ratio_annual(
            summ["sharpe_net"],
            n_trials=max(float(dsr_in["n_trials"]), 1.0),  # effective count, not rounded (audit 2 #17)
            var_trials_sr_annual=dsr_in["var_trials_sr_annual"],
            n_obs=len(net),
            skew=skew,
            kurt=kurt,
            periods_per_year=cfg.trading_days_per_year,
        )

        # ic_sign gate on the same purged walk-forward blocks as the Sharpe gate
        # (audit 2 #16): mean sampled IC of the execution dates inside each block
        ic_by_fold = self._ic_by_fold(icr.ic_series, blocks)
        hurdles = HurdleReport(cfg.validation).evaluate_signal(
            ic_by_fold=ic_by_fold if len(ic_by_fold) else None,
            decile_tstat=ds.t_stat,
            net_sharpe_delta_by_fold=fold_sharpes,  # standalone: baseline 0 -> "fold net Sharpe >= +0.05"
            breakeven_cost_bps=breakeven,
            modeled_cost_bps=summ.get("avg_cost_bps_per_side"),
            dsr=dsr,
            pbo=pbo,
            phase=self.phase,
            net_sharpe=summ["sharpe_net"],
            standalone=True,
        )

        notes = []
        if summ["sharpe_net"] > 2:
            notes.append("net Sharpe > 2: treat as suspect until leakage/cost/selection ruled out (plan 2.4)")
        ehd = summ.get("effective_holding_days", np.nan)
        if np.isfinite(ehd) and ehd < 0.6 * cfg.portfolio.effective_holding_days_target:
            notes.append(
                f"effective holding {ehd:.0f}d < target {cfg.portfolio.effective_holding_days_target}d: raise trade_aversion"
            )
        if pbo is None:
            notes.append("PBO skipped: set SystemConfig.pipeline_pbo_configs to run a CSCV grid")
        if not blocks:
            notes.append(f"fold-Sharpe gate skipped: {len(net)} return days < validation.min_fold_days")
        timings = dict(timings or {})
        timings["evaluate"] = time.perf_counter() - t0
        return PipelineResult(
            backtest=bt,
            summary=summ,
            ic_report=icr,
            hurdles=hurdles,
            cost_sensitivity=cs,
            fold_sharpes=fold_sharpes,
            folds=folds,
            timings=timings,
            notes=notes,
            decile=ds,
            dsr=float(dsr),
            dsr_inputs=dsr_in,
            pbo=pbo,
            breakeven_cost_bps=float(breakeven),
            config_hash=config_hash,
            data_hash=self.data_hash_,
            use_ml=self.use_ml,
            phase=self.phase,
            signal_lag_days=int(signal_lag_days),
        )


# ============================================================================ run registry
def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else str(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _from_json_scalar(v: Any) -> Any:
    if isinstance(v, str) and v in ("nan", "inf", "-inf"):
        return float(v)
    if isinstance(v, dict):
        return {k: _from_json_scalar(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_from_json_scalar(x) for x in v]
    return v


def run_summary_dict(res: PipelineResult) -> dict[str, Any]:
    """The ``summary.json`` payload of a run directory."""
    d = res.decile
    return _json_safe(
        {
            "backtest": res.summary,
            "ic": res.ic_report.scalars(),
            "ic_decay": {str(k): v for k, v in res.ic_report.decay.items()},
            "decile": None
            if d is None
            else {
                "mean": d.mean,
                "sharpe": d.sharpe,
                "t_stat": d.t_stat,
                "horizon": d.horizon,
                "nw_lags": d.nw_lags,
            },
            "dsr": res.dsr,
            "dsr_inputs": res.dsr_inputs,
            "pbo": res.pbo,
            "breakeven_cost_bps": res.breakeven_cost_bps,
            "folds": [
                {
                    "start": str(pd.Timestamp(r.start).date()),
                    "end": str(pd.Timestamp(r.end).date()),
                    "n_days": int(r.n_days),
                    "sharpe": r.sharpe,
                }
                for r in res.folds.itertuples()
            ],
            "cost_sensitivity": {
                str(i): {str(c): res.cost_sensitivity.loc[i, c] for c in res.cost_sensitivity.columns}
                for i in res.cost_sensitivity.index
            },
            "overall": bool(res.hurdles.overall),
            "use_ml": res.use_ml,
            "phase": res.phase,
            "signal_lag_days": res.signal_lag_days,
            "config_hash": res.config_hash,
            "data_hash": res.data_hash,
            "git_sha": res.git_sha,
            "timings": res.timings,
            "notes": res.notes,
        }
    )


def _write_parquet(obj: pd.DataFrame | pd.Series, path: Path, name: str = "value") -> None:
    df = obj.to_frame(name=obj.name or name) if isinstance(obj, pd.Series) else obj.copy()
    df.columns = [str(c) for c in df.columns]
    df.to_parquet(path)


def save_run(res: PipelineResult, cfg: SystemConfig, runs_root: str | Path) -> Path:
    """Write ``res`` to ``<runs_root>/<run_id>/`` (layout: module docstring)."""
    if res.manifest is None:
        res.manifest = RunManifest.create(
            config_hash=res.config_hash, data_hash=res.data_hash, timings=res.timings
        )
    run_dir = Path(runs_root) / res.manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    bt = res.backtest
    res.manifest.save(run_dir)
    (run_dir / "config.yaml").write_text(
        cfg.to_yaml(header=f"config of run {res.manifest.run_id} (hash {res.config_hash})")
    )
    (run_dir / "summary.json").write_text(json.dumps(run_summary_dict(res), indent=2, sort_keys=True))
    _write_parquet(bt.nav.rename("nav"), run_dir / "nav.parquet")
    _write_parquet(bt.weights, run_dir / "weights.parquet")
    _write_parquet(bt.trades, run_dir / "trades.parquet")
    _write_parquet(bt.costs, run_dir / "costs.parquet")
    _write_parquet(bt.returns, run_dir / "returns.parquet")
    _write_parquet(res.ic_report.ic_series.rename("rank_ic"), run_dir / "ic.parquet")
    (run_dir / "hurdles.json").write_text(json.dumps(res.hurdles.to_dict(), indent=2))
    return run_dir


@dataclass
class RunRecord:
    """A run directory re-loaded (typed)."""

    run_dir: Path
    manifest: RunManifest
    config: SystemConfig
    summary: dict[str, Any]
    hurdles: HurdleResult
    nav: pd.Series
    returns: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    costs: pd.DataFrame
    ic: pd.Series

    @property
    def run_id(self) -> str:
        return self.manifest.run_id


def load_run(run_dir: str | Path) -> RunRecord:
    """Load a run directory written by :func:`save_run`."""
    p = Path(run_dir)
    if not (p / "manifest.json").exists():
        raise FileNotFoundError(f"{p} is not a run directory (no manifest.json)")

    def _pq(name: str) -> pd.DataFrame:
        f = p / name
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()

    nav = _pq("nav.parquet")
    ic = _pq("ic.parquet")
    return RunRecord(
        run_dir=p,
        manifest=RunManifest.load(p),
        config=SystemConfig.from_yaml(p / "config.yaml"),
        summary=_from_json_scalar(json.loads((p / "summary.json").read_text())),
        hurdles=HurdleResult.from_dict(json.loads((p / "hurdles.json").read_text())),
        nav=nav["nav"] if "nav" in nav else pd.Series(dtype=float, name="nav"),
        returns=_pq("returns.parquet"),
        weights=_pq("weights.parquet"),
        trades=_pq("trades.parquet"),
        costs=_pq("costs.parquet"),
        ic=ic["rank_ic"] if "rank_ic" in ic else pd.Series(dtype=float, name="rank_ic"),
    )


def list_runs(runs_root: str | Path) -> list[Path]:
    """Run directories under ``runs_root`` (those with a manifest), newest first."""
    root = Path(runs_root)
    if not root.exists():
        return []
    return sorted(
        (d for d in root.iterdir() if (d / "manifest.json").exists()), key=lambda d: d.name, reverse=True
    )


# ============================================================================ report
def format_result(res: PipelineResult) -> str:
    s = res.summary
    icr = res.ic_report
    lines = [
        "=== Backtest summary ===",
        f"period            {s['start']} -> {s['end']}  ({s['n_days']} days)"
        + (f"   signal lag {res.signal_lag_days}d" if res.signal_lag_days else ""),
        f"Sharpe gross/net  {s['sharpe_gross']:.2f} / {s['sharpe_net']:.2f}   (NW t = {s['newey_west_t_net']:.2f})",
        f"CAGR gross/net    {100 * s['cagr_gross']:.2f}% / {100 * s['cagr_net']:.2f}%   vol {100 * s['ann_vol_net']:.1f}%",
        f"max drawdown      {100 * s['max_drawdown_net']:.1f}%   calmar {s['calmar_net']:.2f}",
        f"turnover/day      {100 * s['avg_daily_turnover']:.2f}%   effective holding {s['effective_holding_days']:.0f} d",
        f"cost per side     {s['avg_cost_bps_per_side']:.1f} bp   gross exp {s['avg_gross_exposure']:.2f}  net exp {s['avg_net_exposure']:+.2f}",
        f"transfer coeff    {s.get('transfer_coefficient_mean', float('nan')):.2f}   realized/ex-ante vol {s.get('realized_vs_exante_vol_ratio', float('nan')):.2f}",
        f"sizing            mean gamma {s.get('mean_risk_aversion', float('nan')):.2f}   "
        f"vol target capped on {100 * s.get('vol_target_caps_share', float('nan')):.0f}% of solves",
        "",
        "=== Alpha IC (score vs residualised target, sampled every horizon) ===",
        f"mean rank IC {icr.mean_ic:.4f}  ICIR {icr.icir:.2f} "
        f"(annualised at {icr.periods_per_year:.1f}/yr)  t {icr.t_stat:.2f}  NW t {icr.t_stat_nw:.2f} "
        f"(L={icr.nw_lags})  n {icr.n}",
    ]
    if len(icr.decay):
        lines.append(
            "IC decay (horizon d: mean IC)  " + "  ".join(f"{int(h)}d: {v:.4f}" for h, v in icr.decay.items())
        )
    if res.decile is not None:
        lines.append(
            f"decile L/S (VW)  mean {100 * res.decile.mean:.2f}%/period  Sharpe {res.decile.sharpe:.2f}  "
            f"NW t {res.decile.t_stat:.2f} (L={res.decile.nw_lags})"
        )
    lines += [
        f"DSR {res.dsr:.3f}  (n_trials {res.dsr_inputs.get('n_trials', 1):.1f}, var SR {res.dsr_inputs.get('var_trials_sr_annual', 0):.3f})"
        + (f"  PBO {res.pbo:.3f}" if res.pbo is not None else "  PBO skipped"),
        f"config {res.config_hash}  data {res.data_hash}  git {res.git_sha[:12]}"
        + (f"  run {res.manifest.run_id}" if res.manifest is not None else ""),
        "",
        "=== Fold (purged walk-forward test block) net Sharpe ===",
        (
            res.folds.assign(
                start=res.folds["start"].astype(str).str[:10], end=res.folds["end"].astype(str).str[:10]
            )
            .round(2)
            .to_string(index=False)
            if len(res.folds)
            else "(no folds)"
        ),
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
    if res.run_dir:
        lines += [f"run directory: {res.run_dir}"]
    return "\n".join(lines)
