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
  ledger family ``pipeline_phase{p}:{market_hash[:12]}`` (the *research family*:
  ``market_hash`` = :func:`qts.core.manifest.hash_market`, the adjusted-close
  panel only, so auxiliary tables do not start a new family; the full
  ``data_hash`` stays on the record and in the manifest;
  ``effective_trials_for`` when >= 3 trials, used unrounded; single-trial PSR
  when the ledger is empty / absent).  A re-run is de-duplicated (counted once) only when name, config
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
                      data_hash, market_hash, ledger_family, library versions, python,
                      platform, timings, seeds
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
    delistings.parquet one row per delisting of a held name: date, ticker, weight,
                      reason, side, stock_return, haircut_return (booked), pnl
                      (weight x booked return), pnl_usd, liquidation_usd, cost_usd
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
import hashlib
import inspect
import logging
import time
from collections.abc import Callable, Mapping
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
from qts.alpha.features_v2 import require_v2_inputs
from qts.alpha.targets import forward_returns, residualize_forward_returns
from qts.backtest import BacktestResult, BacktestSpec, BacktestSummary, DailyBarSimulator
from qts.backtest import metrics as M
from qts.backtest.result import INVESTED_GROSS_EPS, MIN_INVESTED_DAYS_FOR_VOL_RATIO
from qts.core.config import SystemConfig
from qts.core.manifest import (
    RunManifest,
    chain_reference_hash,
    hash_market,
    hash_market_data,
    reference_hash,
)
from qts.core.roles import ALPHA_THEME_STYLES, RISK_ONLY_STYLES
from qts.core.types import AlphaOutput, MarketData, RiskModelOutput
from qts.data.universe import apply_mask, build_universe_mask, masked_levels
from qts.portfolio import SinglePeriodOptimizer
from qts.risk import BarraStyleRiskModel, HybridRiskModel, build_risk_model

# The run registry and the report card live in qts.runs; re-exported here because
# ``from qts.pipeline import save_run, load_run, format_result, ...`` is public API.
from qts.runs.cache import DEFAULT_CACHE_DIRNAME, STAGE_EXCLUDED_DATA, StageCache
from qts.runs.context import RunContext
from qts.runs.registry import (
    RunRecord,
    _from_json_scalar,
    _json_safe,
    _write_parquet,
    list_runs,
    load_run,
    run_summary_dict,
    save_run,
)
from qts.runs.report import format_result
from qts.tcost import TransactionCostModel
from qts.validation import (
    HurdleReport,
    HurdleResult,
    PurgedWalkForward,
    TrialLedger,
    deflated_sharpe_ratio_annual,
)
from qts.validation.ledger import callable_identity, research_family
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
    summary: BacktestSummary
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
    #: price-history identity (:func:`qts.core.manifest.hash_market`) and the trial
    #: family the DSR counted in (``pipeline_phase{p}:{market_hash[:12]}``)
    market_hash: str = ""
    ledger_family: str = ""
    git_sha: str = "unknown"
    use_ml: bool = False
    phase: int = 1
    signal_lag_days: int = 0
    manifest: RunManifest | None = None
    run_dir: str | None = None
    #: seal / family / record kind of the run (:class:`qts.runs.context.RunContext`)
    context: RunContext = field(default_factory=RunContext)


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
    context : RunContext | None
        Holdout seal, ledger family override and record kind of the run
        (:mod:`qts.runs.context`); ``None`` = the default (no seal fields, the
        default research family, a counted trial).  Changes no number.
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
        rebalance_every: int | None = None,
        venue: str = "routed",
        nav0: float = 1_000_000.0,
        warmup_days: int | None = None,
        runs_root: str | None = None,
        context: RunContext | None = None,
    ):
        self.cfg = cfg or SystemConfig()
        self.context = context if context is not None else RunContext()
        self.use_ml = use_ml
        self.phase = phase if phase is not None else (2 if use_ml else 1)
        self.ledger_path = ledger_path
        self.ledger = TrialLedger(ledger_path) if ledger_path else None
        self.composite_weight = composite_weight
        self.ml_factory = ml_factory
        #: solve every k-th backtest date; ``None`` -> ``cfg.portfolio.rebalance_every_days``
        self.rebalance_every = int(
            self.cfg.portfolio.rebalance_every_days if rebalance_every is None else rebalance_every
        )
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
        self.cache_info_: dict[str, str] = {}
        self.data_hash_: str = ""
        self._market_hash: tuple[int, str] | None = None  # (id of the input md, hash)
        self._md_input: MarketData | None = None
        self._risk_first_date: pd.Timestamp | None = None
        self._warned_warmup = False

    # ------------------------------------------------------------------ fit
    def fit(self, md: MarketData) -> ResearchPipeline:
        """Universe -> risk model -> alpha model -> cost model / optimizer.

        With ``cfg.cache.enabled`` the risk and alpha fits go through the
        content-addressed stage cache (:mod:`qts.runs.cache`): a hit loads the
        fitted model built from the *identical* data snapshot, config sections
        and code (bitwise the same object a fresh fit builds); a miss fits and
        stores.  ``self.cache_info_`` records ``hit`` / ``stored`` / ``miss``
        per stage."""
        cfg = self.cfg
        if cfg.alpha.feature_set == "v2":
            # fail before the risk fit, naming the command that builds the missing tables
            require_v2_inputs(md)
        t: dict[str, float] = {}
        self._md_input = md
        cache = self._stage_cache()
        self.cache_info_ = {}
        t0 = time.perf_counter()
        if cache is None:
            self.data_hash_ = data_hash_of(md)
            full_hash = ""
        else:
            # panel hash shared by both stages (reference tables one stage reads are
            # keyed on that stage, see _cache_parts); the manifest's data hash chains
            # them on top without hashing the panels again
            full_hash = hash_market_data(md, n_chars=64, exclude=STAGE_EXCLUDED_DATA)
            self.data_hash_ = chain_reference_hash(full_hash, md)[:16]  # == data_hash_of(md)
        t["data_hash"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.universe_mask = build_universe_mask(md, cfg.universe)
        self.md_u = apply_mask(md, self.universe_mask)
        t["universe"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # apply_mask keeps full level history (audit 2 #6); the risk model's
        # estimation universe is still "has a market cap", so it gets the view with
        # levels NaN outside the universe (unchanged risk-model behaviour)
        md_risk = masked_levels(self.md_u)
        risk_key = self._cache_key(cache, "risk_fit", md, full_hash)
        risk = self._cache_load(cache, "risk_fit", risk_key, md, full_hash, {"md": md_risk, "pipeline": self})
        if risk is None:
            risk = build_risk_model(cfg.risk, warmup_days=self.warmup_days).fit(md_risk)
            self._cache_store(
                cache, "risk_fit", risk_key, risk, md, full_hash, {"md": md_risk, "pipeline": self}
            )
        self.risk = risk
        self._risk_first_date = self._exposures_first_date()
        self._warned_warmup = False
        t["risk_fit"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        alpha_key = self._cache_key(cache, "alpha_fit", md, full_hash, parent=risk_key)
        alpha_refs = {"md": self.md_u, "ml_factory": self.ml_factory, "pipeline": self}
        alpha = self._cache_load(cache, "alpha_fit", alpha_key, md, full_hash, alpha_refs)
        if alpha is None:
            alpha = AlphaModel(
                cfg.alpha,
                composite_weight=self.composite_weight,
                ml_factory=self.ml_factory,
                use_ml=self.use_ml,
                delisting_return=cfg.cost.delisting_returns(),
                # labels / purge / trailing IC aligned to what is traded (execution lag L)
                signal_lag_days=self.signal_lag_days(),
                bad_price_caps=(
                    (cfg.risk.return_abs_cap_up, cfg.risk.return_abs_cap_down)
                    if cfg.risk.clean_returns
                    else None
                ),
            )
            # residualize labels on the *risk model's* risk-only exposures (D1; plan 2.3: targets)
            alpha.fit(self.md_u, risk_exposures_fn=self._exposures_fn)
            self._cache_store(cache, "alpha_fit", alpha_key, alpha, md, full_hash, alpha_refs)
        self.alpha = alpha
        t["alpha_fit"] = time.perf_counter() - t0

        self.cost = TransactionCostModel(cfg.cost, cfg.execution)  # routed-venue auction cap (D3)
        self.optimizer = SinglePeriodOptimizer(
            cfg.portfolio, self.cost, horizon_days=cfg.alpha.horizon_days, exec_cfg=cfg.execution
        )
        self.timings_ = t
        return self

    # ------------------------------------------------------------- stage cache
    def _stage_cache(self) -> StageCache | None:
        """The run's :class:`~qts.runs.cache.StageCache`, or ``None`` when
        ``cfg.cache.enabled`` is off."""
        cc = self.cfg.cache
        if not cc.enabled:
            return None
        root = cc.root or str(Path(self.runs_root or "runs") / DEFAULT_CACHE_DIRNAME)
        return StageCache(root)

    def _cache_parts(self, stage: str, md: MarketData, full_hash: str) -> dict[str, Any]:
        """Everything (besides code) the stage's result depends on.  The data
        span (first / last date, shape) is listed explicitly next to the full
        data hash: an entry is never served for another as-of span or snapshot."""
        cfg = self.cfg
        parts: dict[str, Any] = {
            "data_hash": full_hash,
            "data_span": self._data_span(md),
            "universe": cfg.universe.hash(),
            "risk": cfg.risk.hash(),
            "warmup_days": self.warmup_days,
            "fit_glue": self._fit_glue_hash(),
        }
        if stage == "alpha_fit":
            parts.update(
                alpha=cfg.alpha.hash(),
                composite_weight=self.composite_weight,
                use_ml=bool(self.use_ml),
                ml_factory=callable_identity(self.ml_factory),
                # labels: return on the stock by delisting reason + the reasons themselves
                # (delisting_short_gain_haircut / delisting_against_position only
                # change the simulator's booking, so they are not part of the key)
                delisting_returns=cfg.cost.delisting_returns(),
                delisting_reasons=reference_hash(md, "delisting_reasons"),
                signal_lag_days=self.signal_lag_days(),
            )
            if cfg.alpha.feature_set != "v1":
                # the earnings tables are read by the v2 features only: a v1 fit
                # stays valid when they are added or re-derived
                parts.update(
                    earnings_dates=reference_hash(md, "earnings_dates"),
                    earnings_fundamentals=reference_hash(md, "earnings_fundamentals"),
                )
        return parts

    @classmethod
    def _fit_glue_hash(cls) -> str:
        """SHA-256 of the source of the pipeline methods that build the two fits
        (``qts/pipeline.py`` as a whole is not part of the stage code hash, see
        :mod:`qts.runs.cache`); ``""`` when the source is unavailable."""
        h = hashlib.sha256()
        for name in ("fit", "_exposures_fn", "_exposures_first_date", "_alpha_theme_styles"):
            fn = getattr(cls, name, None)
            try:
                h.update(inspect.getsource(fn).encode() if fn is not None else b"")
            except (OSError, TypeError):
                return ""
        return h.hexdigest()

    @staticmethod
    def _data_span(md: MarketData) -> list[Any]:
        d = md.dates
        return [str(d[0]) if len(d) else None, str(d[-1]) if len(d) else None, len(d), len(md.tickers)]

    def _cache_key(
        self, cache: StageCache | None, stage: str, md: MarketData, full_hash: str, parent: str | None = None
    ) -> str:
        if cache is None:
            return ""
        parts = self._cache_parts(stage, md, full_hash)
        if parent is not None:
            parts["parent"] = parent  # alpha labels are residualised on the risk model
        return cache.key(stage, parts)

    def _cache_load(
        self,
        cache: StageCache | None,
        stage: str,
        key: str,
        md: MarketData,
        full_hash: str,
        refs: Mapping[str, Any],
    ) -> Any:
        if cache is None or not key or stage not in self.cfg.cache.stages:
            return None
        obj = cache.load(stage, key, {"data_hash": full_hash, "data_span": self._data_span(md)}, refs)
        self.cache_info_[stage] = "hit" if obj is not None else "miss"
        return obj

    def _cache_store(
        self,
        cache: StageCache | None,
        stage: str,
        key: str,
        obj: Any,
        md: MarketData,
        full_hash: str,
        refs: Mapping[str, Any],
    ) -> None:
        if cache is None or not key or stage not in self.cfg.cache.stages:
            return
        expect = {"data_hash": full_hash, "data_span": self._data_span(md)}
        stored = cache.store(stage, key, obj, expect, refs, parts=self._cache_parts(stage, md, full_hash))
        self.cache_info_[stage] = "stored" if stored is not None else "miss"

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

        # Eligibility: ``DailyBarSimulator`` reads ``md.eligible`` directly when
        # present (it is: ``apply_mask`` sets it); ``eligible_fn`` is the same
        # information as a hook and is used only when the panel is absent.  Dead
        # names (no price) go on the optimizer's no-trade list.
        elig = self.md_u.eligibility()
        lag = self.signal_lag_days()  # D4 execution delay
        spec = BacktestSpec.create(
            dates,
            nav0=self.nav0,
            rebalance_every=self.rebalance_every,
            venue=self.venue,
            neutral_styles=self.neutral_styles(),
            progress=progress,
            eligible_fn=lambda d: elig.columns[elig.loc[d].to_numpy()],
            signal_lag_days=lag,
        )
        t0 = time.perf_counter()
        bt = sim.run_spec(self.md_u, spec, alpha_fn, risk_fn, self.optimizer)
        timings = dict(self.timings_, backtest=time.perf_counter() - t0)
        res = self.evaluate(bt, timings, dates=dates, signal_lag_days=lag)
        res.timings["total_run"] = time.perf_counter() - t_run
        res.manifest = RunManifest.create(
            config_hash=res.config_hash,
            data_hash=res.data_hash,
            timings=res.timings,
            seeds=self._seeds(),
            market_hash=res.market_hash,
            ledger_family=res.ledger_family,
            sealed_end=self.context.sealed_end or "",
            unsealed=bool(self.context.unsealed),
            holdout_start=self.context.holdout_start or "",
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

    def market_hash(self) -> str:
        """:func:`qts.core.manifest.hash_market` of the fitted input (the adjusted-close
        panel: values, dates, ids); ``""`` before :meth:`fit`.  Computed once per input."""
        md = self._md_input
        if md is None:
            return ""
        if self._market_hash is None or self._market_hash[0] != id(md):
            self._market_hash = (id(md), hash_market(md))
        return self._market_hash[1]

    def ledger_family(self) -> str:
        """DSR trial family (the *research family*): ``pipeline_phase{p}:{market_hash[:12]}``
        -- one search per phase and price history.  Config, signal-lag and
        auxiliary-data variants (delisting reasons, fundamentals, industries, the
        universe rules) all pool within it; see :mod:`qts.validation.ledger`.
        ``context.family`` overrides it (a run under the holdout seal logs into the
        family of the full-history panel; the mid-cap confirmation into its own)."""
        if self.context.family:
            return str(self.context.family)
        return research_family(self.phase, self.market_hash())

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
        summ: Mapping[str, Any],
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
            market_hash=self.market_hash(),
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
            extra=self.context.ledger_fields() or None,
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
        """``n_trials`` / cross-trial SR variance for the DSR from the ledger family
        (:meth:`qts.validation.ledger.TrialLedger.dsr_inputs`; no ledger ->
        ``n_trials=1``, variance 0, i.e. the PSR against 0)."""
        if self.ledger is None:
            return {"n_trials": 1.0, "var_trials_sr_annual": 0.0, "n_trials_raw": 1.0}
        return self.ledger.dsr_inputs(family, n_obs, self.cfg.trading_days_per_year)

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
                context=self.context,
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
                delisting_return=self.cfg.cost.delisting_returns(),
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
        """The report card of a backtest, in this order: IC and decile spread of
        the traded score (:meth:`_signal_quality`), fold Sharpes on purged
        walk-forward blocks (:meth:`_fold_sharpes`), breakeven / cost
        sensitivity (:meth:`_cost_gates`), trial ledger + PBO + deflated Sharpe
        (:meth:`_selection_stats`), hurdle gates and notes."""
        self._require_fitted()
        cfg = self.cfg
        summ = bt.summary()
        net = bt.returns["net"].dropna()
        gross = bt.returns["gross"].dropna()
        t0 = time.perf_counter()
        icr, ds = self._signal_quality(bt, int(signal_lag_days))
        blocks, folds, fold_sharpes = self._fold_sharpes(net)
        cs, breakeven = self._cost_gates(bt, gross, summ)
        config_hash = cfg.hash()
        dsr, dsr_in, pbo = self._selection_stats(bt, net, summ, icr, ds, config_hash, signal_lag_days, dates)

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
            failed_solve_frac=bt.failed_solve_fraction(),
            exante_vol_alert_frac=bt.exante_vol_alert_fraction(),
        )
        notes = self._notes(summ, pbo, blocks, len(net))
        notes += self._ledger_notes()
        if self.context.holdout_already_used:
            notes.insert(
                0,
                "holdout already used: an earlier unseal of this configuration looked at the sealed "
                "period, so this full-history result is not out-of-sample",
            )
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
            market_hash=self.market_hash(),
            ledger_family=self.ledger_family(),
            use_ml=self.use_ml,
            phase=self.phase,
            signal_lag_days=int(signal_lag_days),
            context=self.context,
        )

    def _signal_quality(self, bt: BacktestResult, L: int) -> tuple[ICReport, DecileSpreadResult]:
        """IC / decile spread of the *traded* alpha score on non-overlapping formation
        dates (every H execution dates), eligible names only, vs the fitted
        (risk-only residualised) target and the raw forward return respectively.
        With signal lag L the book executed at t holds the score as_of(t - L);
        the label of that signal date (targets.py) is the return from close_t,
        so score, label and raw return all refer to the trade made at t.  Frames
        are indexed by the execution date t (L = 0: unchanged)."""
        assert self.md_u is not None and self.alpha is not None
        md_u, alpha, cfg = self.md_u, self.alpha, self.cfg
        elig = md_u.eligibility()
        H = cfg.alpha.horizon_days
        target = alpha.target_
        exec_dates = pd.DatetimeIndex(bt.weights.index[::H])
        dpos = md_u.dates.get_indexer(exec_dates)
        if (dpos < L).any():
            raise ValueError("evaluate(): an execution date has no signal date L days earlier")
        sig_dates = md_u.dates[dpos - L]

        def at_signal(panel: pd.DataFrame) -> pd.DataFrame:
            out = panel.reindex(index=sig_dates, columns=md_u.tickers)
            out.index = exec_dates
            out.index.name = "date"
            return out

        scores = pd.DataFrame({t: alpha.as_of(s).score for t, s in zip(exec_dates, sig_dates, strict=True)}).T
        scores.index.name = "date"
        scores = scores.reindex(columns=md_u.tickers).where(at_signal(elig).fillna(False).astype(bool))
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
                md_u,
                H,
                cfg.cost.delisting_returns(),
                label_lag_days=L,
                delisting_gap_days=cfg.alpha.delisting_gap_days,
            ).where(elig)
        )
        ds = decile_spread(
            scores,
            fwd_raw,
            horizon=1,  # formation dates do not overlap
            weights=at_signal(md_u.market_cap),
            periods_per_year=ppy,
        )
        return icr, ds

    def _fold_sharpes(self, net: pd.Series) -> tuple[list[pd.DatetimeIndex], pd.DataFrame, pd.Series]:
        """Fold-wise net Sharpe on PurgedWalkForward test blocks (#27); the
        standalone baseline is zero.  Returns ``(blocks, folds, fold_sharpes)``."""
        cfg = self.cfg
        v = cfg.validation
        blocks = purged_fold_blocks(
            pd.DatetimeIndex(net.index),
            v.n_splits,
            v.embargo_days,
            cfg.alpha.horizon_days,
            getattr(v, "min_fold_days", 20),
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
        return blocks, folds, fold_sharpes

    def _cost_gates(
        self, bt: BacktestResult, gross: pd.Series, summ: BacktestSummary
    ) -> tuple[pd.DataFrame, float]:
        """Breakeven / cost sensitivity on gross returns net of borrow (borrow is not a
        per-side cost; consistent with BacktestResult.summary())."""
        gab = getattr(bt, "gross_after_borrow_returns", None)
        gross_cs = gab.dropna() if isinstance(gab, pd.Series) and len(gab.dropna()) else gross
        to = self._turnover_on_return_dates(bt, gross_cs.index)
        cs = M.cost_sensitivity(gross_cs, to, cost_bps_per_side=summ.get("avg_cost_bps_per_side", 10.0))
        return cs, M.breakeven_cost_bps(gross_cs, to)

    def _selection_stats(
        self,
        bt: BacktestResult,
        net: pd.Series,
        summ: BacktestSummary,
        icr: ICReport,
        ds: DecileSpreadResult,
        config_hash: str,
        signal_lag_days: int,
        dates: pd.DatetimeIndex | None,
    ) -> tuple[float, dict[str, float], float | None]:
        """Trial ledger: log this run first, then the PBO grid's configurations (each
        a trial of the same search, audit 2 #11), then read the family statistics
        for the deflated Sharpe.  Family = phase + market hash (the price history):
        trials on another market (seeds, real data) are a different search (audit 2
        #17), trials on the same prices with other auxiliary data are not.  Returns
        ``(dsr, dsr_inputs, pbo)``."""
        cfg = self.cfg
        family = self.ledger_family()
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
        return dsr, dsr_in, pbo

    def _ledger_notes(self) -> list[str]:
        """Warn when the ledger holds records of this run's price history that are
        still filed under a data-hash family (written before 2026-11-12) and were
        therefore not counted in this run's DSR."""
        if self.ledger is None:
            return []
        from qts.validation.ledger import rekeyed_family

        family = self.ledger_family()
        mapping = {**self.ledger.known_market_hashes(), self.data_hash_: self.market_hash()}
        n = sum(1 for r in self.ledger.raw_records() if (rekeyed_family(r, mapping) or ("",))[0] == family)
        if not n:
            return []
        return [
            f"trial ledger: {n} earlier record(s) on this price history are filed under a data-hash "
            f"family and were NOT counted in this DSR; run `qts ledger rekey --path {self.ledger_path}` "
            "and then `qts ledger restate <run_dir>`"
        ]

    def _notes(
        self, summ: BacktestSummary, pbo: float | None, blocks: list[pd.DatetimeIndex], n_days: int
    ) -> list[str]:
        cfg = self.cfg
        notes = []
        if summ["sharpe_net"] > 2:
            notes.append("net Sharpe > 2: treat as suspect until leakage/cost/selection ruled out (plan 2.4)")
        ehd = summ.get("effective_holding_days", np.nan)
        if np.isfinite(ehd) and ehd < 0.6 * cfg.portfolio.effective_holding_days_target:
            notes.append(
                f"effective holding {ehd:.0f}d < target {cfg.portfolio.effective_holding_days_target}d: raise trade_aversion"
            )
        inv = summ.get("invested_day_frac", np.nan)
        if np.isfinite(inv) and inv < 0.5:
            notes.append(
                f"book empty (gross <= {INVESTED_GROSS_EPS:g}) on {100 * (1 - inv):.0f}% of days: "
                "check the alpha / IC floor; realized/ex-ante vol is over invested days only"
            )
        if not np.isfinite(summ.get("realized_vs_exante_vol_ratio", np.nan)) and np.isfinite(inv):
            notes.append(
                f"realized/ex-ante vol undefined: < {MIN_INVESTED_DAYS_FOR_VOL_RATIO} invested days with a forecast"
            )
        if pbo is None:
            notes.append("PBO skipped: set SystemConfig.pipeline_pbo_configs to run a CSCV grid")
        if not blocks:
            notes.append(f"fold-Sharpe gate skipped: {n_days} return days < validation.min_fold_days")
        return notes


__all__ = [
    "PipelineResult",
    "ResearchPipeline",
    "RunRecord",
    "WarmupEstimate",
    "_from_json_scalar",
    "_json_safe",
    "_write_parquet",
    "data_hash_of",
    "estimate_warmup_days",
    "format_result",
    "list_runs",
    "load_run",
    "purged_fold_blocks",
    "run_summary_dict",
    "save_run",
]
