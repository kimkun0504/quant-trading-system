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
JobStatus = Literal["queued", "running", "done", "failed"]
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
    git_sha: str
    status: RunStatus
    headline: RunHeadline | None = Field(description="null while running or after a failure")
    config_leaves: dict[str, ConfigLeaf] = Field(
        description="flattened SystemConfig (dotted path -> scalar); empty while running or after a failure"
    )


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
    n_failed_solves: int


class RunManifest(_Model):
    run_id: str
    created_at: str
    git_sha: str
    git_dirty: bool | None = Field(description="null when git was unavailable")
    config_hash: str
    data_hash: str
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
    config_overrides: dict[str, Union[bool, int, float, str]] = Field(  # noqa: UP007
        description='dotted-path overrides of SystemConfig, e.g. {"portfolio.trade_aversion": 2}'
    )
    label: str | None = Field(default=None, max_length=120, description="run label (default: derived)")
    max_days: int | None = Field(
        default=None, ge=20, description="backtest only the last N out-of-sample days (quick runs)"
    )
    venue: Venue | None = Field(default=None, description="simulator pricing venue (default routed)")

    @field_validator("label")
    @classmethod
    def _printable_label(cls, v: str | None) -> str | None:
        if v is not None and any(ord(ch) < 32 or 0x7F <= ord(ch) < 0xA0 for ch in v):
            raise ValueError("label must not contain control characters")
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


class Job(_Model):
    job_id: str
    run_id: str
    status: JobStatus
    progress: float
    message: str


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


class LedgerFamily(_Model):
    family: str
    n_trials: int
    effective_trials: float
    var_sr_annual: float


class Ledger(_Model):
    trials: list[LedgerTrial]
    families: list[LedgerFamily]


# ----------------------------------------------------------------------------- config schema
class JsonSchemaNumber(_Model):
    type: Literal["number", "integer"]
    title: str = None  # type: ignore[assignment]
    description: str = None  # type: ignore[assignment]
    default: float = None  # type: ignore[assignment]
    minimum: float = None  # type: ignore[assignment]
    maximum: float = None  # type: ignore[assignment]
    exclusiveMinimum: float = None  # type: ignore[assignment]  # JSON Schema keyword
    exclusiveMaximum: float = None  # type: ignore[assignment]


class JsonSchemaBoolean(_Model):
    type: Literal["boolean"]
    title: str = None  # type: ignore[assignment]
    description: str = None  # type: ignore[assignment]
    default: bool = None  # type: ignore[assignment]


class JsonSchemaString(_Model):
    type: Literal["string"]
    title: str = None  # type: ignore[assignment]
    description: str = None  # type: ignore[assignment]
    default: str = None  # type: ignore[assignment]
    enum: list[str] = None  # type: ignore[assignment]


class JsonSchemaObject(_Model):
    type: Literal["object"]
    title: str = None  # type: ignore[assignment]
    description: str = None  # type: ignore[assignment]
    properties: dict[
        str,
        Union[JsonSchemaNumber, JsonSchemaBoolean, JsonSchemaString, JsonSchemaObject],  # noqa: UP007
    ]


JsonSchemaObject.model_rebuild()


# ----------------------------------------------------------------------------- errors / health
class ValidationIssue(_Model):
    loc: list[str | int]
    msg: str
    type: str


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
    status: Literal["ok"]
    version: str
    qts_version: str
    runs_root: str
    runs_root_exists: bool
    n_runs: int
    jobs: JobCounts
