"""Pydantic v2 models of the API contract.

They mirror ``ui/src/shared/api/schema.ts`` name for name (snake_case fields,
same optionality, same enums); ``ui/src/shared/api/contract.check.ts`` type-checks
the UI types against the TypeScript generated from this module's OpenAPI schema,
so drift fails ``npm run typecheck``.

Conventions
-----------
* Response models have **no defaults**: every field is always present (``null``
  where the schema allows it), and the OpenAPI schema marks it required.
* Floats that are not finite in Python (NaN / inf, e.g. an undefined ratio) are
  serialised as ``null``; the UI's formatters render them as "n/a".
* Optional properties of the JSON-Schema shapes (``title?`` etc.) are typed
  ``str = None`` (not ``str | None``) so the OpenAPI type stays ``string`` and the
  route drops unset ones (``response_model_exclude_none``).
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

RunKind = Literal["synthetic", "store"]
RunStatus = Literal["running", "done", "failed"]
GateStatus = Literal["PASS", "FAIL", "skipped"]
OverallGate = Literal["PASS", "FAIL"]
Phase = Literal[1, 2]
SignalLag = Literal[0, 1]
GateName = Literal[
    "t_stat",
    "ic_sign",
    "incremental_ic",
    "net_sharpe_delta",
    "breakeven_cost",
    "dsr",
    "pbo",
    "phase_sharpe",
    "suspect",
    "failed_solves",
    "risk_model_sanity",
]
RunSeriesName = Literal["nav", "gross", "net", "drawdown", "turnover", "exante_vol", "gross_exposure", "net_exposure"]
CostComponent = Literal["spread", "impact", "fees", "borrow"]
ExposureRole = Literal["alpha_theme", "risk_only", "market"]
JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
Venue = Literal["routed", "auction", "continuous"]
#: A scalar leaf of a flattened ``SystemConfig`` (dotted path -> value); lists and
#: nested structures that are not sections are serialised as JSON strings.
ConfigLeaf = Union[bool, int, float, str, None]  # noqa: UP007

SERIES_NAMES: tuple[RunSeriesName, ...] = (
    "nav",
    "gross",
    "net",
    "drawdown",
    "turnover",
    "exante_vol",
    "gross_exposure",
    "net_exposure",
)
COST_COMPONENTS: tuple[CostComponent, ...] = ("spread", "impact", "fees", "borrow")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- the seal of a run
SealState = Literal["sealed", "unsealed", "exempt", "none"]


class RunLoadCoverage(BaseModel):
    """The store load-coverage guard's verdict for the run's load (``qts.research.coverage``;
    the ``load_coverage`` field of its ledger record, or ``launch.json`` for API runs)."""

    model_config = ConfigDict(extra="ignore")

    prefilter: str | None = None
    method: str | None = Field(default=None, description="by construction | measured | no prefilter | skipped")
    ok: bool | None = Field(default=None, description="false: the load missed names the universe picks (flagged)")
    share_dates_full: float | None = None
    detail: str | None = None


class RunSeal(_Model):
    """Where a run stands relative to the research holdout seal (``summary.json``)."""

    state: SealState = Field(
        description="sealed: data through sealed_end; unsealed: looked at the holdout; exempt: full history by "
        "design (mid-cap check); none: no seal was in force"
    )
    holdout_start: str | None
    sealed_end: str | None = Field(description="last date the run loaded under the seal")
    unsealed: bool
    unseal_reason: str | None
    holdout_already_used: bool = Field(description="an earlier unseal of this configuration already looked")
    seal_exempt: str | None
    record_kind: str = Field(description="ledger record kind: trial, capacity, calibration, ...")
    reference: bool = Field(description="a re-run of an already counted configuration: logged, not counted")
    load_coverage: RunLoadCoverage | None = Field(
        default=None, description="store runs: the load-coverage guard's verdict (null: not recorded)"
    )


# ----------------------------------------------------------------------------- runs list
class RunHeadline(_Model):
    """Headline numbers shown in the runs list."""

    sharpe_net: float
    sharpe_gross: float
    cagr_net: float
    max_drawdown: float
    avg_cost_bps_per_side: float
    dsr: float
    invested_day_frac: float = Field(description="share of decision days with gross exposure > 1e-4 of NAV")
    overall_gate: OverallGate


class RunSummary(_Model):
    id: str
    created_at: str
    label: str
    kind: RunKind
    phase: Phase
    use_ml: bool
    config_hash: str
    data_hash: str
    ledger_family: str | None = Field(
        description="trial family the run's DSR counted in (the manifest's); null for older or unfinished runs"
    )
    git_sha: str
    status: RunStatus
    headline: RunHeadline | None = Field(description="null while running or after a failure")
    config_leaves: dict[str, ConfigLeaf] = Field(
        description="flattened SystemConfig (dotted path -> scalar); empty while running or after a failure"
    )
    seal: RunSeal


# ----------------------------------------------------------------------------- run detail
class BacktestSummary(_Model):
    """``BacktestResult.summary()`` (qts/backtest/result.py)."""

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
    realized_vs_exante_vol_ratio: float = Field(
        description="realised / ex-ante vol over invested days only; NaN (null) with too few"
    )
    invested_day_frac: float = Field(description="share of decision days with gross exposure > 1e-4 of NAV")
    effective_n_names: float = Field(
        description="mean over invested days of 1 / sum p_i^2, p_i = |w_i| / gross (null for runs recorded before it)"
    )
    specific_var_share: float = Field(
        description="mean specific share of the risk model's ex-ante variance of the book (null when not recorded)"
    )
    exante_vol_at_gross2: float = Field(
        description="mean ex-ante annual vol scaled to gross exposure 2: vol x 2 / gross (null when not recorded)"
    )
    breakeven_cost_bps: float
    n_delistings: int
    delisting_pnl_ann: float = Field(
        description="annualised sum over delistings of held names of weight x booked delisting return "
        "(return on the stock by delisting reason; null for runs recorded before it)"
    )
    n_failed_solves: int


class RunManifest(_Model):
    run_id: str
    created_at: str
    git_sha: str
    git_dirty: bool | None = Field(description="null when git was unavailable")
    config_hash: str
    data_hash: str
    ledger_family: str | None = Field(
        description="trial family the run's DSR counted in, as the manifest records it; null for older runs"
    )
    python: str
    platform: str
    versions: dict[str, str]
    timings: dict[str, float]
    seeds: dict[str, int]


class GateResult(_Model):
    name: GateName
    status: GateStatus
    value: float | str | None
    threshold: float | str | None
    reason: str
    required: bool


class Hurdles(_Model):
    overall: OverallGate
    phase: Phase
    required: list[GateName]
    gates: list[GateResult]


class IcByYear(_Model):
    year: int
    mean_ic: float
    n: int


class IcReport(_Model):
    mean_ic: float
    ic_sd: float
    icir: float
    t_stat: float
    hit_rate: float
    periods_per_year: float
    n: int
    horizon_days: int
    by_year: list[IcByYear]


class DecileSpread(_Model):
    mean: float
    sharpe: float
    t_stat: float


class DsrInputs(_Model):
    n_trials: float
    var_trials_sr_annual: float
    skew: float
    kurtosis: float
    n_obs: int


class FoldSharpe(_Model):
    label: str
    sharpe_net: float


class RunSource(_Model):
    """Where a run's market data came from and how the run was invoked: what the run
    builder needs to start a new run "from" this one.  Read from ``launch.json`` (runs
    launched through the API), else from the CLI command recorded in the manifest
    (``qts demo --stocks/--days/--seed``, ``qts run --store/--start/--end/--max-names/
    --min-mcap-ever``, ``--ml``, ``--cache``, ``--max-days``); ``recorded`` is false when
    neither says (older runs): the fields are then null and the builder asks."""

    recorded: bool
    origin: Literal["api", "cli", "unknown"]
    kind: RunKind
    stocks: int | None = Field(description="synthetic market: number of stocks")
    days: int | None = Field(description="synthetic market: number of days")
    store_path: str | None = Field(
        description="ParquetStore root as recorded (CLI --store or the API's QTS_STORE_PATH)"
    )
    start: str | None = Field(description="store slice start, YYYY-MM-DD")
    end: str | None = Field(description="store slice end, YYYY-MM-DD")
    max_names: int | None = Field(description="store prefilter: the N ids with the largest peak market cap")
    min_mcap_ever: float | None = Field(description="store prefilter: peak market cap floor, USD")
    max_rank: int | None = Field(
        default=None, description="store prefilter: every id ranked <= K by screened market cap on some date"
    )
    max_days: int | None = Field(description="backtest only the last N out-of-sample days")
    cache: bool = Field(description="stage cache on")
    venue: Venue | None
    command: str = Field(description="the recorded command line ('' when none)")


class RunConfig(_Model):
    """Subset of ``SystemConfig`` the UI reads directly; the full YAML is in ``config_yaml``."""

    kind: RunKind
    use_ml: bool
    phase: Phase
    stocks: int | None = Field(description="universe size of the input data; null when not recorded")
    days: int | None = Field(description="days of market data; null when not recorded")
    seed: int
    signal_lag_days: int = Field(description="ExecutionConfig.signal_lag_days (the launch form offers 0 or 1)")
    config_yaml: str
    source: RunSource


class RunDetail(_Model):
    id: str
    label: str
    status: RunStatus
    summary: BacktestSummary
    manifest: RunManifest
    config: RunConfig
    hurdles: Hurdles
    ic_report: IcReport
    decile: DecileSpread | None
    dsr: float
    dsr_inputs: DsrInputs
    pbo: float | None
    fold_sharpes: list[FoldSharpe]
    execution_venue: str | None = Field(
        description='routing venue of the execution layer, e.g. "closing auction (LOC/MOC)"; '
        "null for runs written before it was recorded"
    )
    notes: list[str]
    seal: RunSeal


# ----------------------------------------------------------------------------- compare
class RunCompareRun(_Model):
    """One run of a side-by-side comparison."""

    id: str
    label: str
    created_at: str
    phase: Phase
    start: str
    end: str
    headline: RunHeadline
    n_trials: float = Field(description="DSR trial count from the ledger family")
    fold_sharpes: list[FoldSharpe]
    hurdles: Hurdles
    config_leaves: dict[str, ConfigLeaf]


class RunCompare(_Model):
    """``GET /api/runs/compare?ids=``: NAV series of several runs on one shared date axis
    (the union of their realisation dates, thinned to at most ``max_points``), plus the
    fold tables, gates and flattened configs the compare view diffs."""

    dates: list[str]
    nav: dict[str, list[float | None]] = Field(description="run id -> NAV aligned to `dates` (null where absent)")
    runs: list[RunCompareRun]
    downsampled: bool = Field(description="true when the shared axis was thinned to max_points")


# ----------------------------------------------------------------------------- series
class ColumnarSeries(_Model):
    """Columnar time series: ``series[name][i]`` belongs to ``dates[i]``."""

    dates: list[str]
    series: dict[str, list[float | None]]


class CostValues(_Model):
    spread: list[float | None]
    impact: list[float | None]
    fees: list[float | None]
    borrow: list[float | None]


class CostSeries(_Model):
    """``ColumnarSeries<CostComponent>``: USD per decision date by component."""

    dates: list[str]
    series: CostValues


class CostSensitivityRow(_Model):
    multiplier: float
    cost_bps_per_side: float
    net_sharpe: float
    net_cagr: float
    mean_daily_cost_bps: float


class RunCosts(_Model):
    daily: CostSeries
    nav: list[float | None] = Field(description="pre-trade NAV (USD) per date of `daily`")
    cost_sensitivity: list[CostSensitivityRow]


class ExposureSnapshotRow(_Model):
    factor: str
    exposure: float
    role: ExposureRole


class ExposureSnapshot(_Model):
    date: str
    rows: list[ExposureSnapshotRow]


class RunExposures(_Model):
    factors: list[str]
    roles: dict[str, ExposureRole]
    history: ColumnarSeries
    latest: ExposureSnapshot | None = Field(description="null when the run has no exposure history")


class Holding(_Model):
    ticker: str
    weight: float
    alpha: float | None
    specific_vol: float | None
    industry: str | None


class RunHoldings(_Model):
    date: str
    available_dates: list[str]
    positions: list[Holding]


class IcDecayPoint(_Model):
    horizon_days: int
    mean_ic: float


class RankIcValues(_Model):
    rank_ic: list[float | None]


class RankIcSeries(_Model):
    """``ColumnarSeries<'rank_ic'>``."""

    dates: list[str]
    series: RankIcValues


class RunIc(_Model):
    ic: RankIcSeries
    decay: list[IcDecayPoint]
    horizon_days: int


# ----------------------------------------------------------------------------- labs (steps 4-5)
class AlphaRefitWindow(_Model):
    """One walk-forward refit of the ML sleeve (``alpha_refits.json``)."""

    refit_date: str = Field(description="first date the refit's model predicts for")
    train_start: str
    train_end: str = Field(description="last label date used; the embargo separates it from refit_date")
    valid_start: str | None
    pred_end: str
    n_train: int
    n_valid: int
    blend_weights: dict[str, float] = Field(
        description="ensemble member weights by sleeve (sum 1); empty for a bare model"
    )


class AlphaFeatureIc(_Model):
    feature: str
    sign: float = Field(description="+1 / -1 as the composite uses the feature; +1 for ML-only features")
    theme: str | None
    mean_ic: float | None = Field(
        description="full-sample mean daily rank IC of sign x feature against the training label"
    )
    t_stat: float | None = Field(description="Newey-West t with the label overlap as the bandwidth floor")
    n: int


class AlphaRefits(_Model):
    """Learned weights over time: raw per-sleeve feature importance per refit (the UI
    normalises each refit to sum 1), the refit windows and the feature IC table."""

    refits: list[AlphaRefitWindow]
    sleeves: list[str] = Field(description="lgbm / ridge / mlp / ensemble, those recorded")
    features: list[str]
    importance: dict[str, dict[str, list[float | None]]] = Field(
        description="sleeve -> feature -> one raw value per refit (aligned with `refits`); null where not recorded"
    )
    feature_ic: list[AlphaFeatureIc]
    embargo_days: int
    refit_frequency_days: int
    notes: list[str]
    recorded: bool = Field(description="false for runs written before the refits were recorded")


class AlphaThemeWeights(_Model):
    """The composite's theme weights by decision date (``theme_weights.parquet``)."""

    dates: list[str]
    series: dict[str, list[float | None]]
    themes: list[str]
    static: dict[str, float] = Field(description="`alpha.theme_weights` of the run's config")
    dynamic: bool = Field(description="`alpha.dynamic_weights`: whether the weights move")
    downsampled: bool


class IcByYearCell(_Model):
    year: int
    mean_ic: float | None
    t_stat: float | None
    n: int


class IcByYearRow(_Model):
    sleeve: str = Field(description="combined, composite, ml or theme_<name>")
    cells: list[IcByYearCell] = Field(description="one per year of `years`, in order")


class AlphaIcByYear(_Model):
    years: list[int]
    rows: list[IcByYearRow]
    source: Literal["daily_sleeves", "sampled_combined"] = Field(
        description="daily_sleeves: daily rank IC per sleeve with Newey-West t; sampled_combined: the sampled "
        "ic.parquet (combined only, plain t) for runs written before the sleeve IC was recorded"
    )
    overlap_days: int = Field(description="label overlap (horizon + signal lag) the t-statistics allow for")


class RunConstraints(_Model):
    """Per-rebalance constraint log (``constraints.parquet``); empty `dates` for older runs."""

    dates: list[str]
    names: list[str] = Field(description="constraint names that ever bound, in display order")
    binding: dict[str, list[int]] = Field(description="name -> binding-entry count per rebalance")
    status: list[str]
    vol_target_binding: list[str] = Field(
        description="target / caps / bracket / max_solves / no_alpha / solve_failed:* / ''"
    )
    transfer_coefficient: list[float | None]
    n_traded: list[int] = Field(description="names with a non-zero trade")
    n_solves: list[int]
    share_binding: dict[str, float] = Field(description="share of rebalances on which each constraint bound")
    vol_target_reasons: list[str] = Field(description="distinct non-empty vol_target_binding values")
    n_rebalances: int


# ----------------------------------------------------------------------------- launch / jobs
class LaunchRunRequest(_Model):
    """Strict types (pydantic strict mode): ``"stocks": "60"`` or ``"ml": "true"`` are
    rejected, not coerced; the label may not contain control characters."""

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: RunKind
    ml: bool
    phase: Phase
    stocks: int = Field(ge=50, le=3000)
    days: int = Field(
        ge=300,
        le=6000,
        description=(
            "synthetic market length: at least the effective warm-up (GET /api/config/warmup) "
            "plus one year; checked against the run's own config at launch"
        ),
    )
    seed: int = Field(ge=0, le=2**31 - 1)
    signal_lag_days: SignalLag
    config_overrides: dict[str, Union[bool, int, float, str, None]] = Field(  # noqa: UP007
        description=(
            'dotted-path overrides, e.g. {"portfolio.trade_aversion": 2}; applied on top of the '
            "config of `base_run_id` when given, else of the SystemConfig defaults; null only for nullable fields"
        )
    )
    base_run_id: str | None = Field(
        default=None,
        max_length=128,
        description="start from this run's config.yaml (run builder: clone and edit); default: SystemConfig()",
    )
    start: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="store slice start")
    end: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="store slice end")
    max_names: int | None = Field(
        default=None, ge=10, le=20_000, description="store prefilter: the N ids with the largest peak market cap"
    )
    min_mcap_ever: float | None = Field(
        default=None, ge=0, strict=False, description="store prefilter: peak market cap floor in [start, end], USD"
    )
    max_rank: int | None = Field(
        default=None,
        ge=10,
        le=20_000,
        description=(
            "store prefilter (`qts run --max-rank K`): every id that ranked <= K by market cap among the names "
            "passing the universe's price / ADV screens on some date; covers a top-N universe when "
            "K >= N / (1 - exclude_bottom_size_quantile). Not with max_names / min_mcap_ever"
        ),
    )
    allow_partial_coverage: bool | None = Field(
        default=None,
        description=(
            "store runs: load although the prefilter misses names the universe picks (look-ahead; flagged on the "
            "ledger record; `qts run --allow-partial-coverage`). Default: such a load is refused"
        ),
    )
    cache: bool | None = Field(default=None, description="reuse fitted models from the stage cache (qts --cache)")
    label: str | None = Field(default=None, max_length=120, description="run label (default: derived)")
    max_days: int | None = Field(
        default=None, ge=20, description="backtest only the last N out-of-sample days (quick runs)"
    )
    venue: Venue | None = Field(default=None, description="simulator pricing venue (default routed)")
    unseal_reason: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "store runs: load the sealed holdout (`qts run --unseal REASON`). Once per configuration, logged "
            "in the trial ledger before the run starts; cannot be undone"
        ),
    )
    force: bool | None = Field(
        default=None,
        description="with unseal_reason: unseal a configuration whose holdout was already used (flagged)",
    )

    @field_validator("label", "unseal_reason")
    @classmethod
    def _printable_label(cls, v: str | None) -> str | None:
        if v is not None and any(ord(ch) < 32 or 0x7F <= ord(ch) < 0xA0 for ch in v):
            raise ValueError("must not contain control characters")
        return v


class Warmup(_Model):
    """Effective warm-up of a run (``qts.pipeline.estimate_warmup_days``): trading
    days of the market panel used before the first out-of-sample backtest date."""

    ml: bool
    signal_lag_days: int
    warmup_days: int = Field(description="max(risk_days, ml_days) + signal_lag_days")
    min_days: int = Field(description="warmup_days + one year (trading_days_per_year): the launch floor")
    risk_days: int = Field(description="risk exposures warm-up + min_history_days of factor returns")
    ml_days: int | None = Field(
        description="universe lead + alpha.train_min_days + embargo (upper bound); null without ML"
    )
    exact: bool = Field(description="true when the (exact) risk term binds")
    trading_days_per_year: int


class LaunchRunResponse(_Model):
    job_id: str
    run_id: str


class ValidationIssue(_Model):
    loc: list[str | int]
    msg: str
    type: str


class ConfigValidation(_Model):
    """``POST /api/config/validate``: what a launch with this body would do, without
    queueing anything.  ``errors`` are the same issues a launch answers with 422."""

    valid: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue] = Field(description="launchable, but worth a look before spending a trial")
    config_hash: str | None = Field(description="SystemConfig.hash() of the resulting config; null when invalid")
    base_config_hash: str | None = Field(description="hash of the base (the run's config or the defaults)")
    warmup_days: int | None = Field(description="estimated warm-up of the resulting config (trading days)")
    min_days: int | None = Field(description="warm-up plus one year: the floor for a synthetic market's days")
    overrides_applied: int = Field(description="number of override keys applied on the base")


JobTask = Literal[
    "pipeline",
    "signal_test",
    "sweep_member",
    "sweep_finalize",
    "capacity",
    "risk_calibration",
    "midcap_check",
    "paper_run",
    "paper_reconcile",
    "market_hash",
]
JobErrorKind = Literal["stale_data", "book_locked", "paper", "seal", "coverage", "refused", "error"]
#: A flat result value of a task job (ids, a headline number, a verdict word).
ScalarValue = Union[bool, int, float, str, None]  # noqa: UP007


class Job(_Model):
    task: JobTask = Field(description="pipeline: a run; otherwise a research or paper-trading task")
    subject: str | None = Field(description="what the task is about: sweep id, book name, signal, ...")
    command: str | None = Field(description="the equivalent `qts` command line")
    result: dict[str, ScalarValue] | None = Field(description="a finished task's summary (ids, headline numbers)")
    error_kind: JobErrorKind | None = Field(
        description="failed jobs: stale_data (exit code 3: the vendor has not published the session yet), "
        "book_locked (exit code 4), paper, seal, coverage (the store prefilter misses names the universe picks: "
        "use --max-rank), refused (a once-only or seal rule), error"
    )
    exit_code: int | None = Field(description="failed paper jobs: the `qts paper` exit code")
    job_id: str
    run_id: str
    status: JobStatus
    progress: float
    message: str
    label: str
    kind: RunKind
    ml: bool
    phase: Phase
    config_hash: str
    base_run_id: str | None
    n_overrides: int = Field(description="override keys of the request (0 for an unchanged base)")
    created_at: str
    started_at: str | None
    finished_at: str | None
    position: int | None = Field(description="0-based place in the queue; null unless queued")
    stage: str = Field(description="pipeline stage of a running job (short name); '' otherwise")
    last_log: str | None = Field(description="last line of the job's log; null before the first line")
    eta_seconds: float | None = Field(
        description="running: estimated seconds left, from finished jobs of similar size; null when there are none"
    )
    estimated_seconds: float | None = Field(description="estimated total duration from similar finished jobs")


class JobQueue(_Model):
    """``GET /api/jobs``: running first, then the queue in order, then the most recent
    finished, failed and cancelled jobs."""

    jobs: list[Job]
    max_workers: int
    max_queued: int
    counts: dict[str, int] = Field(description="jobs by status (all on disk, not only the listed ones)")


class JobMove(_Model):
    model_config = ConfigDict(extra="forbid", strict=True)

    position: int = Field(ge=0, description="new 0-based place in the queue (clamped to its end)")


class JobLog(_Model):
    job_id: str
    lines: list[str] = Field(description="the last `tail` lines, oldest first")
    total_lines: int
    truncated: bool = Field(description="true when the log has more lines than returned")


# ----------------------------------------------------------------------------- ledger
class LedgerTrial(_Model):
    id: str
    ts: str
    name: str
    family: str
    config_hash: str
    data_hash: str
    sharpe_annual: float
    n_obs: int
    mean_ic: float | None
    n_reruns: int
    kind: str = Field(description="record kind: trial (absent = trial), unseal, capacity, calibration, ...")
    reference: bool = Field(description="a re-run of an already counted configuration")
    counted: bool = Field(description="enters the family's N and V (an ordinary trial that is not a reference)")
    note: str | None = Field(description="why the record is not counted, in plain words; null when counted")
    holdout_already_used: bool
    unsealed: bool


class LedgerFamily(_Model):
    family: str
    n_trials: int
    effective_trials: float
    var_sr_annual: float
    n_not_counted: int = Field(description="reference re-runs and records that are not trials")


class Ledger(_Model):
    trials: list[LedgerTrial]
    families: list[LedgerFamily]


# ----------------------------------------------------------------------------- config schema
class JsonSchemaNumber(_Model):
    type: Literal["number", "integer"]
    title: str = None  # type: ignore[assignment]
    description: str = None  # type: ignore[assignment]
    nullable: bool = None  # type: ignore[assignment]  # the field accepts null (Optional scalar)
    default: float = None  # type: ignore[assignment]
    minimum: float = None  # type: ignore[assignment]
    maximum: float = None  # type: ignore[assignment]
    exclusiveMinimum: float = None  # type: ignore[assignment]  # JSON Schema keyword
    exclusiveMaximum: float = None  # type: ignore[assignment]


class JsonSchemaBoolean(_Model):
    type: Literal["boolean"]
    title: str = None  # type: ignore[assignment]
    description: str = None  # type: ignore[assignment]
    nullable: bool = None  # type: ignore[assignment]  # the field accepts null (Optional scalar)
    default: bool = None  # type: ignore[assignment]


class JsonSchemaString(_Model):
    type: Literal["string"]
    title: str = None  # type: ignore[assignment]
    description: str = None  # type: ignore[assignment]
    nullable: bool = None  # type: ignore[assignment]  # the field accepts null (Optional scalar)
    default: str = None  # type: ignore[assignment]
    enum: list[str] = None  # type: ignore[assignment]


class JsonSchemaItems(_Model):
    type: Literal["string"]


class JsonSchemaArray(_Model):
    """A multi-select field (``alpha.disabled_features``): the allowed names are
    ``choices[<value of the field at choices_key>]``; as an override it takes a
    comma-separated string of names (``--set`` syntax), ``""`` for none."""

    type: Literal["array"]
    title: str = None  # type: ignore[assignment]
    description: str = None  # type: ignore[assignment]
    items: JsonSchemaItems
    uniqueItems: bool = None  # type: ignore[assignment]  # JSON Schema keyword
    multiple: bool = None  # type: ignore[assignment]  # render as a multi-select
    default: list[str] = None  # type: ignore[assignment]
    choices: dict[str, list[str]] = Field(
        default=None,  # type: ignore[arg-type]
        description="allowed names keyed by the value of the field at choices_key",
    )
    choices_key: str = Field(
        default=None,  # type: ignore[arg-type]
        description="dotted path of the field whose value picks the list in choices (alpha.feature_set)",
    )


class JsonSchemaObject(_Model):
    type: Literal["object"]
    title: str = None  # type: ignore[assignment]
    description: str = None  # type: ignore[assignment]
    properties: dict[
        str,
        Union[JsonSchemaNumber, JsonSchemaBoolean, JsonSchemaString, JsonSchemaArray, JsonSchemaObject],  # noqa: UP007
    ]


JsonSchemaObject.model_rebuild()


# ----------------------------------------------------------------------------- errors / health
class ApiErrorBody(_Model):
    """RFC 7807 problem details (``application/problem+json``); ``detail`` is always a string."""

    type: str = None  # type: ignore[assignment]
    title: str = None  # type: ignore[assignment]
    status: int = None  # type: ignore[assignment]
    detail: str
    errors: list[ValidationIssue] = None  # type: ignore[assignment]


class JobCounts(_Model):
    queued: int
    running: int


# ----------------------------------------------------------------------------- files
RunFileKind = Literal["json", "yaml", "parquet", "text", "other"]


class RunFile(_Model):
    """One file of the run directory (top level only)."""

    name: str
    size_bytes: int
    modified_at: str = Field(description="RFC 3339, UTC")
    kind: RunFileKind
    description: str = Field(description="what the file holds, for known artifacts; empty otherwise")
    download_url: str = Field(description="``/api/runs/{id}/files/{name}``")


class RunFiles(_Model):
    run_id: str
    files: list[RunFile]


class Health(_Model):
    store_path: str | None = Field(description="QTS_STORE_PATH: the ParquetStore store runs read; null when unset")
    status: Literal["ok"]
    version: str
    qts_version: str
    runs_root: str
    runs_root_exists: bool
    n_runs: int
    jobs: JobCounts


# ============================================================================= research tools
# The payloads below are the library's own JSON (``to_dict()`` of the research
# dataclasses, docs/notes/research_tools_2026-11-13.md "Library entry points"), served
# as written.  ``_Lib`` models ignore keys they do not name, so a field the library adds
# later does not break the API; a number that is not finite is null (``F``).
F = Union[float, None]  # noqa: UP007
DATE_RE = r"^\d{4}-\d{2}-\d{2}$"
#: Config files a research task may read: under configs/, or a run's config.yaml.
CONFIG_PATH_RE = r"^(configs/[A-Za-z0-9_.-]+\.ya?ml|runs/[A-Za-z0-9][A-Za-z0-9._-]{0,127}/config\.yaml)$"


class _Lib(BaseModel):
    model_config = ConfigDict(extra="ignore")


class TaskLaunchResponse(_Model):
    job_id: str
    label: str
    command: str = Field(description="the equivalent `qts` command line (recorded on the job)")


class StoreSlice(_Model):
    """The store slice and prefilter of a research command (``qts ... --store`` flags; the
    store itself is the API server's ``QTS_STORE_PATH``)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    start: str | None = Field(default=None, pattern=DATE_RE)
    end: str | None = Field(default=None, pattern=DATE_RE)
    max_names: int | None = Field(default=None, ge=10, le=20_000, description="peak-market-cap prefilter: N ids")
    min_mcap_ever: float | None = Field(default=None, ge=0, strict=False, description="peak market cap floor (USD)")
    max_rank: int | None = Field(
        default=None, ge=10, le=20_000, description="screened-rank prefilter (not with max_names / min_mcap_ever)"
    )


class StoreSpecView(_Lib):
    store: str
    start: str | None = None
    end: str | None = None
    max_names: int | None = None
    min_mcap_ever: F = None
    max_rank: int | None = None
    min_price: F = None
    min_adv_usd: F = None
    adv_window: int | None = None


# ----------------------------------------------------------------------------- seal
class SealInfo(_Lib):
    holdout_start: str
    load_end: str = Field(description="the last date a sealed load may contain (the day before holdout_start)")
    reason: str
    sealed_by: str
    sealed_at: str
    store_end: str | None
    inherit_family: list[str]
    path: str


class SealUnseal(_Lib):
    ts: str | None = None
    config_hash: str | None = None
    family: str | None = None
    phase: int | None = None
    unseal_reason: str | None = None
    forced: bool | None = None
    n_previous_unseals: int | None = None


class SealExemptRun(_Lib):
    ts: str | None = None
    family: str | None = None
    seal_exempt: str | None = None
    config_hash: str | None = None


class SealStatus(_Model):
    """``qts seal status``: the seal, every unseal, the research runs made under it."""

    seal: SealInfo | None = Field(description="null: no seal in force")
    data_through: str | None = Field(description="research loads end here (seal.load_end)")
    unseals: list[SealUnseal]
    n_forced: int = Field(description="unseals that re-used a holdout already looked at")
    runs_under_seal: int
    by_family: dict[str, dict[str, int]] = Field(description="family -> {kind: count} of the runs under the seal")
    inherited_families: dict[str, int | None] = Field(
        description="seal.inherit_family -> its trial count (null: not in this ledger)"
    )
    exempt_runs: list[SealExemptRun]
    ledger_path: str | None


class UnsealCheck(_Model):
    """Earlier unseals of one configuration (same normalised config hash and phase)."""

    config_hash: str
    phase: Phase
    sealed: bool = Field(description="a seal is in force")
    previous: list[SealUnseal]
    requires_force: bool = Field(description="true: this configuration's holdout is already used")


# ----------------------------------------------------------------------------- trial impact
class TrialImpact(_Lib):
    """``qts.research.sweep.trial_impact``: the effect of new trials on a family's DSR."""

    family: str
    n_trials_raw: int
    n_eff_before: float
    n_new: int
    n_reference: int
    n_eff_after: float
    var_sr_annual: float
    exp_max_sr_before: F
    exp_max_sr_after: F
    best_sharpe: F
    best_n_obs: int | None
    dsr_best_before: F
    dsr_best_after: F


class FamilyResolution(_Model):
    """The trial family a research command on a store slice logs into, when it can be
    known without reading the store (the full-history market hash is cached)."""

    known: bool
    market_hash_full: str | None
    reason: str = Field(description="why the family is not known yet; '' when it is")


# ----------------------------------------------------------------------------- features
class FeatureInfo(_Lib):
    name: str
    theme: str = Field(description='composite theme it belongs to, or "ml_only"')
    sign: float
    role: str = Field(description="risk_only_proxy, alpha_theme or model_input")
    feature_set: str = Field(description="first feature set that has it (v1 / v2)")
    description: str


class FeatureTheme(_Model):
    theme: str
    members: list[str]


class FeatureList(_Model):
    feature_set: str
    features: list[FeatureInfo]
    themes: list[FeatureTheme] = Field(description="composite themes and their members in this feature set")


# ----------------------------------------------------------------------------- signal tests
class SignalDef(_Lib):
    name: str
    theme: str
    sign: float
    role: str
    feature_set: str = Field(description="v1, v2 or custom")
    description: str
    industry_demean: bool
    source: str = Field(description="builtin or custom")


class IcStats(_Lib):
    mean_ic: F
    sd: F
    t_nw: F
    nw_lags: int
    n: int
    hit: F


class IcYear(_Lib):
    mean_ic: F
    n: int


class IcFold(_Lib):
    start: str
    end: str
    mean_ic: F
    n: int


class SignalDecay(_Lib):
    ic_by_horizon: dict[str, F]
    lag_ic: dict[str, F]
    half_life_days: F


class PaperBookStats(_Lib):
    sharpe_gross: F
    sharpe_net: F
    ann_return_gross: F
    ann_vol: F
    turnover_per_day: F
    cost_bps_per_side: F
    breakeven_cost_bps: F
    n_days: F


class SignalIncremental(_Lib):
    weight: float
    ic_by_horizon: dict[str, IcStats]
    resid_var_share: F
    baseline_book: PaperBookStats
    blend_book: PaperBookStats
    delta_sharpe_net: F
    delta_sharpe_gross: F
    delta_t_nw: F


class SignalFamilyTest(_Lib):
    signal: str | None = None
    t: F = None
    ts: str | None = None
    id: str | None = None
    p: F = None
    p_holm: F = None


class SignalTrial(_Lib):
    family: str
    logged: bool
    n_tests: int
    t: F
    p: F
    p_holm: F
    hurdle: float
    passes_hurdle: bool
    tests: list[SignalFamilyTest]


class SignalBaseline(_Lib):
    config_hash: str
    horizon_days: int
    signal_lag_days: int
    use_ml: bool
    cache: dict[str, str]
    sealed_end: str | None


class SignalTestResult(_Lib):
    """``runs/_signals/<id>.json`` (``SignalTestResult.to_dict()``)."""

    id: str
    signal: SignalDef
    horizons: list[int]
    primary_horizon: int
    step: int
    window: list[str]
    baseline: SignalBaseline
    coverage_by_year: dict[str, F]
    ic_by_horizon: dict[str, IcStats]
    ic_by_year: dict[str, IcYear]
    ic_by_fold: list[IcFold]
    decay: SignalDecay
    paper: PaperBookStats
    correlations: dict[str, F]
    incremental: SignalIncremental
    trial: SignalTrial
    timings: dict[str, F]
    notes: list[str]
    json_path: str | None = None


class SignalFamilyNow(_Model):
    """The family's multiple-testing state now (later tests raise an earlier test's
    Holm-adjusted p): every test logged so far, Holm over all of them."""

    family: str
    n_tests: int
    hurdle: float
    tests: list[SignalFamilyTest]
    p_holm: F = Field(description="this test's Holm-adjusted p over the family as it is now")


class SignalTestDetail(_Model):
    result: SignalTestResult
    family_now: SignalFamilyNow


class SignalTestSummary(_Model):
    id: str
    signal: str
    theme: str
    source: str
    primary_horizon: int
    t: F
    p: F
    p_holm_at_test: F
    p_holm_now: F
    n_tests_now: int
    passes_hurdle: bool
    logged: bool
    family: str
    window: list[str]
    sealed_end: str | None
    mean_ic_primary: F
    sharpe_net_paper: F


class SignalTestList(_Model):
    tests: list[SignalTestSummary]


class SignalTestRequest(_Model):
    model_config = ConfigDict(extra="forbid", strict=True)

    feature: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_]+$")
    store: StoreSlice = Field(default_factory=StoreSlice)
    horizons: list[int] = Field(min_length=1, max_length=8, description="label horizons, trading days")
    step: int = Field(default=5, ge=1, le=60, description="sampling step and paper-book holding, days")
    cost_bps: float | None = Field(default=None, ge=0, le=200, strict=False, description="default: baseline run")
    baseline: str = Field(default="configs/baseline.yaml", pattern=CONFIG_PATH_RE)
    use_ml: bool = True
    cache: bool = Field(default=True, description="the baseline fit from the stage cache (--no-cache when false)")
    log_trial: bool = Field(default=True, description="false: --no-log (the test is not counted!)")

    @field_validator("horizons")
    @classmethod
    def _positive(cls, v: list[int]) -> list[int]:
        if any(h < 1 or h > 252 for h in v) or len(set(v)) != len(v):
            raise ValueError("horizons must be distinct whole numbers of days between 1 and 252")
        return sorted(v)


class SignalTestPreview(_Model):
    """Before a test: which family it joins and how many tests that family has."""

    family: str | None
    resolution: FamilyResolution
    n_tests: int = Field(description="tests already logged in the family")
    n_tests_after: int
    adds: int = Field(description="1, or 0 with log_trial false (then the test is not counted)")
    hurdle: float
    command: str


# ----------------------------------------------------------------------------- sweeps
class SweepParam(_Lib):
    key: str
    values: list[ScalarValue]


class SweepMemberMetrics(_Lib):
    sharpe_net: F = None
    sharpe_gross: F = None
    cagr_net: F = None
    avg_daily_turnover: F = None
    avg_cost_bps_per_side: F = None
    max_drawdown_net: F = None
    mean_ic: F = None
    dsr: F = None
    n_trials: F = None
    seconds: F = None
    cache: dict[str, str] | None = None


class SweepMember(_Lib):
    index: int
    overrides: dict[str, ScalarValue]
    config_hash: str
    status: Literal["pending", "running", "done", "failed"]
    run_id: str | None
    run_dir: str | None
    reference: bool
    error: str | None
    metrics: SweepMemberMetrics
    started_at: str | None
    finished_at: str | None


class SweepPbo(_Lib):
    pbo: F = None
    reason: str | None = None
    n_blocks: int | None = None
    n_days: int | None = None
    n_configs: int | None = None
    n_combinations: int | None = None
    prob_oos_loss: F = None
    degradation_slope: F = None
    degradation_r2: F = None
    mean_oos_rank_pct: F = None
    logits: list[F] | None = None
    partial: bool | None = None


class SweepManifest(_Lib):
    """``runs/_sweeps/<id>.json`` (``SweepPlan.to_dict()``)."""

    id: str
    base_config: str
    base_config_hash: str
    params: list[SweepParam]
    mode: str
    store: StoreSpecView | None
    use_ml: bool
    nav: float
    max_days: int | None
    members: list[SweepMember]
    family: str
    impact: TrialImpact | None
    created_at: str
    pbo: SweepPbo | None
    runs_root: str
    n_members: int
    status: str


class SweepParamRequest(_Model):
    model_config = ConfigDict(extra="forbid", strict=True)

    key: str = Field(min_length=3, max_length=120, pattern=r"^[a-z_][a-z0-9_.]*$")
    values: list[Union[bool, int, float, str]] = Field(min_length=1, max_length=12)  # noqa: UP007


class SweepRequest(_Model):
    model_config = ConfigDict(extra="forbid", strict=True)

    params: list[SweepParamRequest] = Field(min_length=1, max_length=4)
    mode: Literal["grid", "zip"] = "grid"
    base_config: str = Field(default="configs/baseline.yaml", pattern=CONFIG_PATH_RE)
    store: StoreSlice = Field(default_factory=StoreSlice)
    use_ml: bool = True
    nav: float = Field(default=5e6, gt=0, strict=False)
    max_days: int | None = Field(default=None, ge=20)
    max_runs: int = Field(default=12, ge=1, le=12)


class SweepPlanView(_Model):
    """``plan_sweep``: the members, the family and the DSR haircut; nothing written."""

    plan: SweepManifest
    exists: bool = Field(description="a manifest with this id exists: launching resumes it")
    n_pending: int = Field(description="members that would run (not done)")
    command: str


class SweepCreate(SweepRequest):
    expected_id: str = Field(
        pattern=r"^sw_[0-9a-f]{12}$", description="the id of the confirmed preview: refused if the plan changed"
    )


class SweepLaunchResponse(_Model):
    sweep_id: str
    job_ids: list[str]
    finalize_job_id: str | None
    n_queued: int
    command: str


class SweepSummary(_Model):
    id: str
    created_at: str
    base_config: str
    params: list[SweepParam]
    mode: str
    n_members: int
    n_done: int
    status: str
    family: str
    n_new_trials: int | None
    pbo: F
    partial: bool | None
    best_sharpe_net: F


class SweepList(_Model):
    sweeps: list[SweepSummary]


class SweepDetail(_Model):
    sweep: SweepManifest
    jobs: list[Job] = Field(description="this sweep's member and finalize jobs, newest first")
    active: int = Field(description="queued or running jobs of this sweep")
    resumable: int = Field(description="members not done and not queued or running")
    command: str


# ----------------------------------------------------------------------------- capacity
class CapacityPoint(_Lib):
    nav: float
    sharpe_gross: F
    sharpe_net: F
    cagr_net: F
    cost_bps_per_side: F
    turnover_per_day: F
    traded_at_adv_cap: F
    solves_with_adv_cap: F
    breakeven_cost_bps: F
    impact_bps_per_side: F = None
    spread_bps_per_side: F = None
    fees_bps_per_side: F = None
    run_id: str | None = None
    seconds: F = None


class CapacityResult(_Lib):
    """``runs/_capacity/<id>.json``.  Not a trial: logged with ``kind: capacity``."""

    id: str
    config_hash: str
    family: str
    points: list[CapacityPoint]
    half_sharpe_nav: F
    reference_sharpe_net: F
    cache: dict[str, str]
    sealed_end: str | None
    notes: list[str]
    json_path: str | None = None


class CapacitySummary(_Model):
    id: str
    config_hash: str
    family: str
    navs: list[float]
    half_sharpe_nav: F
    reference_sharpe_net: F
    sealed_end: str | None


class CapacityList(_Model):
    results: list[CapacitySummary]


class CapacityRequest(_Model):
    model_config = ConfigDict(extra="forbid", strict=True)

    store: StoreSlice = Field(default_factory=StoreSlice)
    navs: list[float] = Field(min_length=1, max_length=8, description="NAVs in USD")
    baseline: str = Field(default="configs/baseline.yaml", pattern=CONFIG_PATH_RE)
    max_days: int | None = Field(default=None, ge=20)
    use_ml: bool = True

    @field_validator("navs", mode="before")
    @classmethod
    def _navs(cls, v: object) -> object:
        if isinstance(v, list):
            out = [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if len(out) != len(v) or any(x < 1e4 or x > 1e12 for x in out) or len(set(out)) != len(out):
                raise ValueError("navs must be distinct amounts between 1e4 and 1e12 USD")
            return sorted(out)
        return v


# ----------------------------------------------------------------------------- risk calibration
class VolStats(_Lib):
    realised_vol: F
    exante_vol: F
    ratio: F
    bias_stat: F
    n_days: F = None


class NuResult(_Lib):
    nu_scale: float
    overall: VolStats
    by_year: dict[str, VolStats]
    sharpe_net: F
    run_id: str | None = None


class RiskCalibrationResult(_Lib):
    """``runs/_risk/<id>.json``.  A report: adopting a value is one logged trial."""

    id: str
    config_hash: str
    current_nu_scale: float
    results: list[NuResult]
    recommended_nu_scale: F
    rule: str
    family: str
    sealed_end: str | None
    cache: dict[str, str]
    notes: list[str]
    json_path: str | None = None


class RiskCalibrationSummary(_Model):
    id: str
    config_hash: str
    nus: list[float]
    current_nu_scale: float
    recommended_nu_scale: F
    sealed_end: str | None


class RiskCalibrationList(_Model):
    results: list[RiskCalibrationSummary]


class RiskCalibrationRequest(_Model):
    model_config = ConfigDict(extra="forbid", strict=True)

    store: StoreSlice = Field(default_factory=StoreSlice)
    nu: list[float] = Field(min_length=1, max_length=8, description="risk.alignment_nu_scale values")
    nav: float = Field(default=5e6, gt=0, strict=False)
    baseline: str = Field(default="configs/baseline.yaml", pattern=CONFIG_PATH_RE)
    max_days: int | None = Field(default=None, ge=20)
    use_ml: bool = True

    @field_validator("nu", mode="before")
    @classmethod
    def _nu(cls, v: object) -> object:
        if isinstance(v, list):
            out = [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if len(out) != len(v) or any(x < 0 or x > 100 for x in out) or len(set(out)) != len(out):
                raise ValueError("nu values must be distinct numbers between 0 and 100")
            return sorted(out)
        return v


# ----------------------------------------------------------------------------- mid-cap check
class MidcapCriterion(_Lib):
    id: str
    name: str
    value: str
    threshold: str
    passed: bool | None
    mark: str


class MidcapSideBySide(_Lib):
    metric: str
    midcap: str
    largecap: str
    midcap_value: F = None
    largecap_value: F = None


class MidcapSecondary(_Lib):
    net_sharpe_cost_x2: F = None
    breakeven_cost_bps: F = None
    cost_bps_per_side: F = None
    psr: F = None
    traded_at_cap: F = None
    solves_with_cap: F = None
    median_universe: F = None
    ic_by_fold: dict[str, F] | None = None
    sharpe_by_fold: dict[str, F] | None = None


class MidcapVerdict(_Lib):
    outcome: str = Field(description='"PASS", "PASS, cost-fragile", "FAIL" or "INCONCLUSIVE"')
    criteria: list[MidcapCriterion]
    validity: list[MidcapCriterion]
    secondary: MidcapSecondary
    side_by_side: list[MidcapSideBySide]
    reference_run: str | None
    run_id: str | None
    notes: list[str]


class CoverageWorstDate(_Lib):
    date: str
    coverage: F
    loaded: int


class MidcapCoverage(_Lib):
    rank: int
    prefilter: str
    n_loaded_ids: int
    n_dates: int
    share_dates_full: F
    min_coverage: F
    worst_dates: list[CoverageWorstDate]
    min_loaded: int
    median_loaded: F
    needed_loaded: int
    est_load_gb: F = None
    min_store: int | None = None
    n_short_dates: int | None = None
    min_share: F = None
    ok: bool


class MidcapPreviousRun(_Lib):
    ts: str | None = None
    family: str | None = None
    config_hash: str | None = None
    sharpe_annual: F = None
    forced: bool = False


class MidcapCheckResult(_Lib):
    """``runs/_checks/midcap_<run_id>.json`` / ``<run>/midcap_check.json``."""

    config_hash: str
    config_path: str
    store: StoreSpecView
    coverage: MidcapCoverage | None
    preregistration: str
    preregistration_sha256: str
    previous_runs: list[MidcapPreviousRun]
    dry_run: bool
    forced: bool
    family: str | None
    run_dir: str | None
    verdict: MidcapVerdict | None
    json_path: str | None = None


class MidcapRule(_Lib):
    id: str
    rule: str


class MidcapResultSummary(_Model):
    run_id: str
    outcome: str | None
    forced: bool
    family: str | None
    created_at: str | None
    preregistration_sha256: str
    sha_matches: bool = Field(description="the result's pre-registration hash equals the note's hash now")
    source: str = Field(description="runs/_checks/<file> or <run>/midcap_check.json")


class MidcapStatus(_Model):
    criteria: list[MidcapRule]
    config_hash: str = Field(description="baseline + universe.rank_offset=1000")
    config_path: str
    preregistration: str
    preregistration_sha256: str = Field(description="'' when the note is missing")
    previous_runs: list[MidcapPreviousRun] = Field(description="ledger records of this configuration's check")
    results: list[MidcapResultSummary]
    can_launch: bool
    blocked_reason: str | None = Field(description="why the launch button is disabled")
    needs_force: bool = Field(description="the check already ran: a repeat needs force + the typed phrase")
    force_phrase: str
    store_configured: bool
    reference_run_id: str
    last_dry_run: MidcapCheckResult | None
    jobs: list[Job]


class MidcapResultDetail(_Model):
    result: MidcapCheckResult
    run: RunSummary | None = Field(description="the mid-cap run in the registry (null when not on this machine)")
    reference: RunSummary | None = Field(description="the large-cap reference run")
    sha_matches: bool


class MidcapPreregistration(_Model):
    path: str
    exists: bool
    sha256: str = Field(description="SHA-256 of the text above the '## Result' heading")
    text: str


class MidcapRequest(_Model):
    model_config = ConfigDict(extra="forbid", strict=True)

    store: StoreSlice = Field(
        default_factory=StoreSlice, description="default prefilter: --max-rank 2750 (screened market-cap rank)"
    )
    dry_run: bool = False
    force: bool = False
    confirm: str | None = Field(default=None, max_length=64, description="with force: the typed force_phrase")
    cache: bool = True
    nav: float = Field(default=5e6, gt=0, strict=False)


# ============================================================================= paper trading
PaperLevel = Literal["OK", "WATCH", "STOP"]


class PaperDataSpec(_Lib):
    """How the book loads market data (``qts.paper.book.DataSpec``): ``max_rank`` is the
    screened-rank prefilter (default 1,500); a book written before it existed keeps its
    peak-cap ``max_names`` / ``min_mcap_ever``, which the load-coverage guard refuses."""

    kind: str | None = None
    store: str | None = None
    start: str | None = None
    max_names: int | None = None
    min_mcap_ever: F = None
    max_rank: int | None = None


class PaperHeader(_Lib):
    name: str
    created_at: str
    start: str
    first_execution: str
    nav0: float
    sizing: str
    use_ml: bool
    config_source: str
    config_file: str
    config_hash: str
    venue: str
    calendar: str
    reference_run: str | None
    expectations_sha256: str
    code_version: str
    notes: list[str]
    data: PaperDataSpec | None = None


class PaperBookSummary(_Model):
    name: str
    sizing: str = Field(description="moc (live-tradable) or replica (batch-identical pipeline check)")
    nav0: float
    start: str
    first_execution: str
    use_ml: bool
    config_hash: str
    reference_run: str | None
    n_days: int
    last_date: str | None
    last_mode: str | None = Field(description="live or backfill: only live days are paper-trading evidence")
    level: PaperLevel
    trading_halted: bool
    nav: F


class PaperBooks(_Model):
    paper_root: str
    paper_root_flag: str | None = Field(
        description="the --paper-root value the CLI needs for these books; null when it is the default (paper/)"
    )
    books: list[PaperBookSummary]


class PaperCheck(_Lib):
    rule: str
    level: PaperLevel
    value: F
    threshold: F
    message: str


class PaperAck(_Lib):
    date: str
    ts: str
    level: str | None
    rules: list[str]
    reasons: list[str]
    note: str
    user: str


class PaperCone(_Lib):
    expected: F = None
    lower: F = None
    upper: F = None
    median: F = None


class PaperMetrics(_Lib):
    n_days: int | None = None
    cum_return: F = None
    cum_log: F = None
    max_drawdown: F = None
    drawdown: F = None
    sharpe: F = None
    ann_vol: F = None
    cone: PaperCone | None = None
    bootstrap_cone: PaperCone | None = None
    vol_ratio: F = None
    vol_ratio_backtest: F = None
    turnover: F = None
    cost_bps_per_side: F = None
    gross: F = None
    boundary_levels: dict[str, F] | None = None


class PaperStatusBlock(_Lib):
    level: PaperLevel
    reasons: list[str]
    stop_rules: list[str]
    suppressed: list[str]
    acknowledged: bool
    trading_halted: bool = Field(description="STOP and not acknowledged: no orders until `qts paper ack`")
    checks: list[PaperCheck]
    metrics: PaperMetrics


class PaperLatest(_Lib):
    date: str
    data_asof: str | None = None
    mode: str | None = None
    nav: F = None
    nav_pre: F = None
    cash: F = None
    gross: F = None
    net: F = None
    n_long: int | None = None
    n_short: int | None = None
    ret: F = None
    orders_for: str | None = None
    n_orders: int = 0
    exante_vol: F = None
    warnings: list[str] = Field(default_factory=list)
    code_version: str | None = None
    config_hash: str | None = None


class PaperAnalyticCone(_Model):
    expected: list[F]
    lower: list[F]
    upper: list[F]


class PaperBootstrapCone(_Model):
    lower: list[F]
    median: list[F]
    upper: list[F]
    max_dd_q01: list[F]
    max_dd_q05: list[F]


class PaperBoundaries(_Model):
    """The calibrated stop-rule boundaries at each horizon (``boundary_levels`` quantiles
    of the reference run's bootstrap): drawdown and Sharpe, WATCH and STOP."""

    dd_watch: list[F]
    dd_stop: list[F]
    sharpe_watch: list[F]
    sharpe_stop: list[F]
    levels: dict[str, F]


class PaperSeries(_Model):
    date: list[str]
    mode: list[str | None]
    nav: list[F]
    ret: list[F]
    cum_log: list[F]
    drawdown: list[F]
    gross: list[F]
    net: list[F]
    turnover: list[F]
    exante_vol: list[F]
    realised_vol_21d: list[F]
    status: list[str | None]
    cone_analytic: PaperAnalyticCone | None
    cone_bootstrap: PaperBootstrapCone | None
    boundaries: PaperBoundaries | None


class PaperReconcileDay(_Lib):
    date: str
    paper_nav_pre: F
    batch_nav: F
    weight_diff: F = None


class PaperReconcile(_Lib):
    sizing: str
    dates: list[Union[str, int]]  # noqa: UP007
    seconds: F = None
    max_abs_weight_diff: F
    max_rel_nav_diff: F
    final_nav_paper_pre: F = None
    final_nav_batch: F = None
    cum_return_paper: F
    cum_return_batch: F
    daily_return_corr: F
    tracking_error_ann: F
    max_abs_daily_return_diff: F = None
    batch_summary: dict[str, F] = Field(default_factory=dict)
    per_day: list[PaperReconcileDay]
    note: str


class PaperStatus(_Model):
    """``qts.paper.status`` with the series completed (calibrated boundaries per horizon,
    the live / backfill mode per day), the book's recent jobs and its last reconciliation."""

    book: str
    root: str
    header: PaperHeader
    n_days: int
    last_date: str | None
    acks: list[PaperAck]
    latest: PaperLatest | None
    status: PaperStatusBlock
    series: PaperSeries
    jobs: list[Job]
    last_reconcile: PaperReconcile | None
    stop_rules: dict[str, ScalarValue] = Field(description="the pre-registered rules frozen at init")


class PaperHistoryRow(_Lib):
    date: str
    mode: str | None = None
    nav: F = None
    ret: F = None
    gross: F = None
    net: F = None
    n_fills: int | None = None
    n_orders: int = 0
    orders_for: str | None = None
    pnl_positions: F = None
    trading_cost: F = None
    borrow: F = None
    turnover: F = None
    exante_vol: F = None
    status: str | None = None
    n_warnings: int = 0


class PaperHistory(_Model):
    book: str
    rows: list[PaperHistoryRow]


class PaperOrder(_Lib):
    ticker: str
    side: str
    shares: F
    px: F
    usd: F
    target_w: F = None
    current_w: F = None
    kind: str | None = None
    close_out: bool | None = None
    exec_date: str | None = None
    est_cost_usd: F = None
    est_cost_bps: F = None


class PaperFill(_Lib):
    ticker: str
    side: str
    shares: F
    px: F
    usd: F
    spread: F = None
    impact: F = None
    fees: F = None
    kind: str | None = None
    date: str | None = None


class PaperOrders(_Lib):
    date: str
    orders_for: str | None
    mode: str | None
    status: str | None
    orders: list[PaperOrder]
    fills: list[PaperFill]
    n_orders: int
    buy_usd: F
    sell_usd: F
    est_cost_usd: F
    cancelled_by_stop: int = 0


class PaperHolding(_Model):
    ticker: str
    shares: float
    px: F
    usd: F
    weight: F


class PaperHoldings(_Model):
    date: str | None
    nav: F
    holdings: list[PaperHolding]


class PaperSchedule(_Lib):
    label: str
    plist: str
    crontab: str
    log: str
    install: str


class PaperAckRequest(_Model):
    model_config = ConfigDict(extra="forbid", strict=True)

    note: str = Field(min_length=3, max_length=1000, description="why it is safe to continue")


class PaperRunRequest(_Model):
    model_config = ConfigDict(extra="forbid", strict=True)

    date: str | None = Field(default=None, pattern=DATE_RE, description="process through this day")
    until: str | None = Field(default=None, pattern=DATE_RE, description="backfill through this date")
    no_sync: bool = False
    dry_run: bool = False
    redo_last: bool = False
    signal_mode: Literal["daily", "single"] = "daily"
