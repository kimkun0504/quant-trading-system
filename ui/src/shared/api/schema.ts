/**
 * API contract between the workbench and the `qts` run-registry API (`api/`).
 *
 * The backend's Pydantic models (`api/qts_api/schemas.py`) mirror these types name for
 * name; `generated.ts` is `openapi-typescript` output of the API's OpenAPI document
 * (`npm run api:types`) and `contract.check.ts` type-checks every type here against it,
 * so a drift on either side fails `npm run typecheck`. The types stay hand-written
 * because the UI narrows a few generic shapes (`ColumnarSeries<N>`) that OpenAPI cannot
 * express.
 *
 * Field names follow the Python side verbatim (snake_case): `BacktestResult.summary()`,
 * `HurdleResult.to_dict()`, `ICReport.scalars()`, `RunManifest`.
 *
 * Numbers: a value that is not finite in Python (NaN / inf, e.g. an undefined ratio)
 * arrives as `null` even where the type says `number`; the formatters in
 * `shared/lib/format` render it as "n/a".
 */

export type RunId = string;
export type IsoDate = string; // YYYY-MM-DD
export type IsoDateTime = string; // RFC 3339

export type RunKind = 'synthetic' | 'store';
export type RunStatus = 'running' | 'done' | 'failed';
export type GateStatus = 'PASS' | 'FAIL' | 'skipped';
export type OverallGate = 'PASS' | 'FAIL';
export type Phase = 1 | 2;

export type GateName =
  | 't_stat'
  | 'ic_sign'
  | 'incremental_ic'
  | 'net_sharpe_delta'
  | 'breakeven_cost'
  | 'dsr'
  | 'pbo'
  | 'phase_sharpe'
  | 'suspect'
  | 'failed_solves'
  | 'risk_model_sanity';

/** Headline numbers shown in the runs list. `null` while running or after a failure. */
export interface RunHeadline {
  sharpe_net: number;
  sharpe_gross: number;
  cagr_net: number;
  max_drawdown: number;
  avg_cost_bps_per_side: number;
  dsr: number;
  /** Share of decision days with gross exposure > 1e-4 of NAV. */
  invested_day_frac: number;
  overall_gate: OverallGate;
}

/** A scalar leaf of a flattened `SystemConfig`; lists arrive as compact JSON strings. */
export type ConfigLeaf = number | boolean | string | null;
/** `{"portfolio.trade_aversion": 1.0, ...}`: dotted path to scalar. */
export type ConfigLeaves = Record<string, ConfigLeaf>;

export interface RunSummary {
  id: RunId;
  created_at: IsoDateTime;
  label: string;
  kind: RunKind;
  phase: Phase;
  use_ml: boolean;
  config_hash: string;
  data_hash: string;
  git_sha: string;
  status: RunStatus;
  headline: RunHeadline | null;
  /** Flattened config; empty while running or after a failure. */
  config_leaves: ConfigLeaves;
}

/** `BacktestResult.summary()` (qts/backtest/result.py). */
export interface BacktestSummary {
  start: IsoDate;
  end: IsoDate;
  n_days: number;
  final_nav: number;
  cagr_gross: number;
  cagr_net: number;
  ann_vol_gross: number;
  ann_vol_net: number;
  sharpe_gross: number;
  sharpe_net: number;
  sharpe_net_lo: number;
  sharpe_t_stat_net: number;
  newey_west_t_net: number;
  max_drawdown_net: number;
  calmar_net: number;
  sortino_net: number;
  hit_rate_net: number;
  avg_gross_exposure: number;
  avg_net_exposure: number;
  avg_daily_turnover: number;
  effective_holding_days: number;
  avg_cost_bps_per_side: number;
  avg_daily_cost_bps_of_nav: number;
  total_costs_usd: number;
  transfer_coefficient_mean: number;
  /** Realised / ex-ante vol over invested days only (null with too few). */
  realized_vs_exante_vol_ratio: number;
  /** Share of decision days with gross exposure > 1e-4 of NAV. */
  invested_day_frac: number;
  breakeven_cost_bps: number;
  n_delistings: number;
  n_failed_solves: number;
}

export interface RunManifest {
  run_id: RunId;
  created_at: IsoDateTime;
  git_sha: string;
  /** null when git was unavailable to the run. */
  git_dirty: boolean | null;
  config_hash: string;
  data_hash: string;
  python: string;
  platform: string;
  versions: Record<string, string>;
  timings: Record<string, number>;
  seeds: Record<string, number>;
}

export interface GateResult {
  name: GateName;
  status: GateStatus;
  /** Numeric where the gate compares a number; a string such as "4/5" for fold fractions. */
  value: number | string | null;
  threshold: number | string | null;
  reason: string;
  required: boolean;
}

export interface Hurdles {
  overall: OverallGate;
  phase: Phase;
  required: GateName[];
  gates: GateResult[];
}

export interface IcByYear {
  year: number;
  mean_ic: number;
  n: number;
}

export interface IcReport {
  mean_ic: number;
  ic_sd: number;
  icir: number;
  t_stat: number;
  hit_rate: number;
  periods_per_year: number;
  n: number;
  horizon_days: number;
  by_year: IcByYear[];
}

export interface DecileSpread {
  mean: number;
  sharpe: number;
  t_stat: number;
}

export interface DsrInputs {
  n_trials: number;
  var_trials_sr_annual: number;
  skew: number;
  kurtosis: number;
  n_obs: number;
}

export interface FoldSharpe {
  label: string;
  sharpe_net: number;
}

/** Subset of `SystemConfig` the UI reads directly; the full YAML is in `config_yaml`. */
export interface RunConfig {
  kind: RunKind;
  use_ml: boolean;
  phase: Phase;
  /** Universe size / days of market data of the input; null when the run did not record them. */
  stocks: number | null;
  days: number | null;
  seed: number;
  /** `ExecutionConfig.signal_lag_days` (the launch form offers 0 or 1). */
  signal_lag_days: number;
  config_yaml: string;
}

export interface RunDetail {
  id: RunId;
  label: string;
  status: RunStatus;
  summary: BacktestSummary;
  manifest: RunManifest;
  config: RunConfig;
  hurdles: Hurdles;
  ic_report: IcReport;
  decile: DecileSpread | null;
  dsr: number;
  dsr_inputs: DsrInputs;
  pbo: number | null;
  fold_sharpes: FoldSharpe[];
  /** Routing venue used by the execution layer, e.g. "closing auction (LOC/MOC)"; null for runs written before it was recorded. */
  execution_venue: string | null;
  notes: string[];
}

/** One run of `GET /api/runs/compare`. */
export interface RunCompareRun {
  id: RunId;
  label: string;
  created_at: IsoDateTime;
  phase: Phase;
  start: IsoDate;
  end: IsoDate;
  headline: RunHeadline;
  /** DSR trial count from the ledger family. */
  n_trials: number;
  fold_sharpes: FoldSharpe[];
  hurdles: Hurdles;
  config_leaves: ConfigLeaves;
}

/**
 * Several runs side by side: NAV on one shared date axis (the union of the runs' dates,
 * thinned to ≤ 2,000 points), fold Sharpes, gates and flattened configs.
 */
export interface RunCompare {
  dates: IsoDate[];
  /** run id → NAV aligned to `dates` (null where the run has no value). */
  nav: Record<RunId, (number | null)[]>;
  runs: RunCompareRun[];
  downsampled: boolean;
}

/** Columnar time series: `series[name][i]` belongs to `dates[i]`. */
export interface ColumnarSeries<N extends string = string> {
  dates: IsoDate[];
  series: Record<N, (number | null)[]>;
}

/**
 * `gross` / `net` are daily returns; `gross_exposure` / `net_exposure` are the post-trade
 * book's Σ|w| and Σw (fraction of NAV) aligned like `turnover` (decision t → realisation t+1).
 */
export type RunSeriesName =
  | 'nav'
  | 'gross'
  | 'net'
  | 'drawdown'
  | 'turnover'
  | 'exante_vol'
  | 'gross_exposure'
  | 'net_exposure';

export type CostComponent = 'spread' | 'impact' | 'fees' | 'borrow';

export interface CostSensitivityRow {
  multiplier: number;
  cost_bps_per_side: number;
  net_sharpe: number;
  net_cagr: number;
  mean_daily_cost_bps: number;
}

export interface RunCosts {
  /** USD per day by component. */
  daily: ColumnarSeries<CostComponent>;
  /** Pre-trade NAV per day (USD), to convert to bp of NAV. */
  nav: (number | null)[];
  cost_sensitivity: CostSensitivityRow[];
}

export interface ExposureSnapshotRow {
  factor: string;
  exposure: number;
  role: 'alpha_theme' | 'risk_only' | 'market';
}

export interface ExposureSnapshot {
  date: IsoDate;
  rows: ExposureSnapshotRow[];
}

/** Empty (`factors: []`, `latest: null`) for runs written without an exposure history. */
export interface RunExposures {
  factors: string[];
  roles: Record<string, ExposureSnapshotRow['role']>;
  history: ColumnarSeries;
  latest: ExposureSnapshot | null;
}

/** `alpha`, `specific_vol` and `industry` are null for runs written before they were stored. */
export interface Holding {
  ticker: string;
  weight: number;
  alpha: number | null;
  specific_vol: number | null;
  industry: string | null;
}

export interface RunHoldings {
  date: IsoDate;
  available_dates: IsoDate[];
  positions: Holding[];
}

export interface IcDecayPoint {
  horizon_days: number;
  mean_ic: number;
}

export interface RunIc {
  ic: ColumnarSeries<'rank_ic'>;
  decay: IcDecayPoint[];
  horizon_days: number;
}

// ----------------------------------------------------------------------------- labs
/** One walk-forward refit of the ML sleeve (`alpha_refits.json`). */
export interface AlphaRefitWindow {
  /** First date the refit's model predicts for. */
  refit_date: IsoDate;
  train_start: IsoDate;
  /** Last label date used; the embargo separates it from `refit_date`. */
  train_end: IsoDate;
  valid_start: IsoDate | null;
  pred_end: IsoDate;
  n_train: number;
  n_valid: number;
  /** Ensemble member weights by sleeve (sum 1); empty for a bare model. */
  blend_weights: Record<string, number>;
}

export interface AlphaFeatureIc {
  feature: string;
  /** +1 / −1 as the composite uses the feature; +1 for ML-only features. */
  sign: number;
  theme: string | null;
  mean_ic: number | null;
  /** Newey–West t with the label overlap as the bandwidth floor. */
  t_stat: number | null;
  n: number;
}

/**
 * Learned weights over time: raw per-sleeve feature importance per refit (normalise each
 * refit to sum 1 before drawing), the refit windows and the full-sample feature IC table.
 * `recorded` is false for runs written before the refits were recorded.
 */
export interface AlphaRefits {
  refits: AlphaRefitWindow[];
  /** lgbm / ridge / mlp / ensemble, those recorded. */
  sleeves: string[];
  features: string[];
  /** sleeve → feature → one raw value per refit (aligned with `refits`); null where not recorded. */
  importance: Record<string, Record<string, (number | null)[]>>;
  feature_ic: AlphaFeatureIc[];
  embargo_days: number;
  refit_frequency_days: number;
  notes: string[];
  recorded: boolean;
}

/** The composite's theme weights by decision date; empty `dates` for older runs. */
export interface AlphaThemeWeights {
  dates: IsoDate[];
  series: Record<string, (number | null)[]>;
  themes: string[];
  /** `alpha.theme_weights` of the run's config. */
  static: Record<string, number>;
  dynamic: boolean;
  downsampled: boolean;
}

export interface IcByYearCell {
  year: number;
  mean_ic: number | null;
  t_stat: number | null;
  n: number;
}

export interface IcByYearRow {
  /** combined, composite, ml or theme_<name>. */
  sleeve: string;
  /** One per year of `years`, in order. */
  cells: IcByYearCell[];
}

export interface AlphaIcByYear {
  years: number[];
  rows: IcByYearRow[];
  /** daily_sleeves: daily rank IC per sleeve with Newey–West t; sampled_combined: older runs. */
  source: 'daily_sleeves' | 'sampled_combined';
  overlap_days: number;
}

/** Per-rebalance constraint log (`constraints.parquet`); empty `dates` for older runs. */
export interface RunConstraints {
  dates: IsoDate[];
  /** Constraint names that ever bound. */
  names: string[];
  /** name → binding-entry count per rebalance. */
  binding: Record<string, number[]>;
  status: string[];
  /** target / caps / bracket / max_solves / no_alpha / solve_failed:* / ''. */
  vol_target_binding: string[];
  transfer_coefficient: (number | null)[];
  /** Names with a non-zero trade. */
  n_traded: number[];
  n_solves: number[];
  /** Share of rebalances on which each constraint bound. */
  share_binding: Record<string, number>;
  vol_target_reasons: string[];
  n_rebalances: number;
}

export type Venue = 'routed' | 'auction' | 'continuous';

export interface LaunchRunRequest {
  kind: RunKind;
  ml: boolean;
  phase: Phase;
  stocks: number;
  days: number;
  seed: number;
  signal_lag_days: 0 | 1;
  /** Dotted-path overrides of SystemConfig, e.g. {"portfolio.trade_aversion": 2}. */
  config_overrides: Record<string, number | boolean | string>;
  /** Run label (default: derived from the model and seed). */
  label?: string | null;
  /** Backtest only the last N out-of-sample days (quick runs). */
  max_days?: number | null;
  /** Simulator pricing venue (default "routed"). */
  venue?: Venue | null;
}

/** `GET /api/config/warmup` (`qts.pipeline.estimate_warmup_days`). */
export interface Warmup {
  ml: boolean;
  signal_lag_days: number;
  /** max(risk_days, ml_days) + signal_lag_days. */
  warmup_days: number;
  /** warmup_days + one year: the launch floor for `days`. */
  min_days: number;
  risk_days: number;
  /** Upper bound; null without ML. */
  ml_days: number | null;
  /** True when the (exact) risk term binds. */
  exact: boolean;
  trading_days_per_year: number;
}

export interface LaunchRunResponse {
  job_id: string;
  run_id: RunId;
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed';

export interface Job {
  job_id: string;
  run_id: RunId;
  status: JobStatus;
  progress: number;
  message: string;
}

export interface LedgerTrial {
  id: string;
  ts: IsoDateTime;
  name: string;
  family: string;
  config_hash: string;
  data_hash: string;
  sharpe_annual: number;
  n_obs: number;
  mean_ic: number | null;
  n_reruns: number;
}

export interface LedgerFamily {
  family: string;
  n_trials: number;
  effective_trials: number;
  var_sr_annual: number;
}

export interface Ledger {
  trials: LedgerTrial[];
  families: LedgerFamily[];
}

/** JSON Schema (draft 2020-12 subset) of SystemConfig, as served by /api/config/schema. */
export interface JsonSchemaNumber {
  type: 'number' | 'integer';
  title?: string;
  description?: string;
  default?: number;
  minimum?: number;
  maximum?: number;
  /** Open bounds (value must be strictly greater / less); the API rejects values outside them. */
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
}

export interface JsonSchemaBoolean {
  type: 'boolean';
  title?: string;
  description?: string;
  default?: boolean;
}

export interface JsonSchemaString {
  type: 'string';
  title?: string;
  description?: string;
  default?: string;
  enum?: string[];
}

export type JsonSchemaLeaf = JsonSchemaNumber | JsonSchemaBoolean | JsonSchemaString;

export interface JsonSchemaObject {
  type: 'object';
  title?: string;
  description?: string;
  properties: Record<string, JsonSchemaLeaf | JsonSchemaObject>;
}

export type ConfigSchema = JsonSchemaObject;

export interface ValidationIssue {
  loc: (string | number)[];
  msg: string;
  type: string;
}

/** RFC 7807 problem details (`application/problem+json`); `detail` is always a string. */
export interface ApiErrorBody {
  type?: string;
  title?: string;
  status?: number;
  detail: string;
  /** Per-field issues of a 422. */
  errors?: ValidationIssue[];
}

export type RunFileKind = 'json' | 'yaml' | 'parquet' | 'text' | 'other';

/** One file of the run directory (`GET /api/runs/{id}/files`). */
export interface RunFile {
  name: string;
  size_bytes: number;
  /** RFC 3339, UTC. */
  modified_at: IsoDateTime;
  kind: RunFileKind;
  /** What the file holds, for known artifacts; empty otherwise. */
  description: string;
  /** `/api/runs/{id}/files/{name}` (relative to the API origin). */
  download_url: string;
}

export interface RunFiles {
  run_id: RunId;
  files: RunFile[];
}

export interface Health {
  status: 'ok';
  version: string;
  qts_version: string;
  runs_root: string;
  runs_root_exists: boolean;
  n_runs: number;
  jobs: { queued: number; running: number };
}
