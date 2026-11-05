/**
 * API contract between the workbench and the `qts` run-registry API.
 *
 * Hand-written for now; the backend (FastAPI) will publish an OpenAPI schema and this
 * file is meant to be replaced by `openapi-typescript` output. Keep the names below
 * stable (they are re-exported through `entities/*`), so swapping the source only
 * touches this file.
 *
 * Field names follow the Python side verbatim (snake_case): `BacktestResult.summary()`,
 * `HurdleResult.to_dict()`, `ICReport.scalars()`, `RunManifest`.
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
  | 'suspect';

/** Headline numbers shown in the runs list. `null` while running or after a failure. */
export interface RunHeadline {
  sharpe_net: number;
  sharpe_gross: number;
  cagr_net: number;
  max_drawdown: number;
  avg_cost_bps_per_side: number;
  dsr: number;
  overall_gate: OverallGate;
}

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
}

/** `BacktestResult.summary()` (qts/backtest/simulator.py). */
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
  realized_vs_exante_vol_ratio: number;
  breakeven_cost_bps: number;
  n_delistings: number;
  n_failed_solves: number;
}

export interface RunManifest {
  run_id: RunId;
  created_at: IsoDateTime;
  git_sha: string;
  git_dirty: boolean;
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
  stocks: number;
  days: number;
  seed: number;
  signal_lag_days: 0 | 1;
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
  /** Routing venue used by the execution layer, e.g. "closing auction (LOC/MOC)". */
  execution_venue: string;
  notes: string[];
}

/** Columnar time series: `series[name][i]` belongs to `dates[i]`. */
export interface ColumnarSeries<N extends string = string> {
  dates: IsoDate[];
  series: Record<N, (number | null)[]>;
}

export type RunSeriesName = 'nav' | 'gross' | 'net' | 'drawdown' | 'turnover' | 'exante_vol';

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
  /** NAV per day (USD), to convert to bp of NAV. */
  nav: number[];
  cost_sensitivity: CostSensitivityRow[];
}

export interface ExposureSnapshotRow {
  factor: string;
  exposure: number;
  role: 'alpha_theme' | 'risk_only' | 'market';
}

export interface RunExposures {
  factors: string[];
  roles: Record<string, ExposureSnapshotRow['role']>;
  history: ColumnarSeries;
  latest: { date: IsoDate; rows: ExposureSnapshotRow[] };
}

export interface Holding {
  ticker: string;
  weight: number;
  alpha: number;
  specific_vol: number;
  industry: string;
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

export interface Ledger {
  trials: LedgerTrial[];
  families: {
    family: string;
    n_trials: number;
    effective_trials: number;
    var_sr_annual: number;
  }[];
}

/** JSON Schema (draft 2020-12 subset) of SystemConfig, as served by /api/config/schema. */
export interface JsonSchemaNumber {
  type: 'number' | 'integer';
  title?: string;
  description?: string;
  default?: number;
  minimum?: number;
  maximum?: number;
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

export interface ApiErrorBody {
  detail: string;
}
