/**
 * Deterministic synthetic runs for the mock API. Every number is derived from the
 * generated series (Sharpe, drawdown, costs, gates), so the fixtures are internally
 * consistent: the metrics table, the charts and the gates all agree.
 */
import type {
  AlphaIcByYear,
  AlphaRefits,
  AlphaThemeWeights,
  CostSensitivityRow,
  ExposureSnapshotRow,
  GateResult,
  Holding,
  Hurdles,
  IcByYear,
  Phase,
  RunConstraints,
  RunCosts,
  RunDetail,
  RunExposures,
  RunFile,
  RunHoldings,
  RunIc,
  RunKind,
  RunSeriesName,
  RunSeal,
  RunSource,
  RunStatus,
  RunSummary,
  BacktestSummary,
  ConfigLeaf,
  ConfigLeaves,
  FoldSharpe,
  Warmup,
} from '@/shared/api';
import {
  annualizedSharpe,
  annualizedVol,
  cagr,
  drawdownPath,
  hitRate,
  maxDrawdown,
  mean,
  sd,
  sortino,
} from '@/shared/lib/stats';
import { mockAlphaRefits, mockConstraints, mockIcByYear, mockThemeWeights } from './labs';
import { deflatedSharpe, hashString, mulberry32, normal, standardise } from './random';
import type { Rng } from './random';

export interface RunSpec {
  id: string;
  label: string;
  created_at: string;
  kind: RunKind;
  phase: Phase;
  use_ml: boolean;
  stocks: number;
  /** Simulated market length; the backtest starts after the effective warm-up (`mockWarmup`). */
  days: number;
  seed: number;
  signal_lag_days: 0 | 1;
  status: RunStatus;
  /** Targets the generator calibrates to. */
  sharpeGross: number;
  sharpeNet: number;
  meanIc: number;
  icirAnnual: number;
  decileT: number;
  nTrials: number;
  varTrials: number;
  pbo: number | null;
  startDate?: string;
  /** Dotted-path overrides of the mock's default config (`portfolio.trade_aversion: 0`). */
  overrides?: Record<string, ConfigLeaf>;
  /** Share of decision days the book was invested (default 1). */
  investedDayFrac?: number;
  /** Scales the lab fixtures' IC story (1 = the baseline story). */
  icLevel?: number;
  /** Recorded data source beyond kind / stocks / days (store slice, prefilter, flags). */
  source?: Partial<RunSource>;
  /** The run's seal fields (default: no seal in force when it ran). */
  seal?: Partial<RunSeal>;
}

/** The seal fields of `summary.json` (`RunContext.summary_fields`); none by default. */
export function mockSeal(spec: Pick<RunSpec, 'seal'>): RunSeal {
  return {
    state: 'none',
    holdout_start: null,
    sealed_end: null,
    unsealed: false,
    unseal_reason: null,
    holdout_already_used: false,
    seal_exempt: null,
    record_kind: 'trial',
    reference: false,
    ...spec.seal,
  };
}

export interface GeneratedRun {
  spec: RunSpec;
  summary: RunSummary;
  detail: RunDetail;
  series: Record<RunSeriesName, number[]>;
  dates: string[];
  costs: RunCosts;
  exposures: RunExposures;
  ic: RunIc;
  holdingsFor: (date: string | undefined) => RunHoldings;
  alphaRefits: AlphaRefits;
  themeWeights: AlphaThemeWeights;
  icByYear: AlphaIcByYear;
  constraints: RunConstraints;
}

/**
 * `GET /api/config/warmup` for the default SystemConfig, mirroring
 * `qts.pipeline.estimate_warmup_days`: risk = 252 exposure warm-up + 252 days of
 * factor returns + 1; ML = universe lead 51 + train_min_days 756 + horizon 20 + lag;
 * total = max(risk, ML) + lag.
 */
export function mockWarmup(ml: boolean, lag: 0 | 1): Warmup {
  const riskDays = 252 + 252 + 1;
  const mlDays = ml ? 51 + 756 + 20 + lag : null;
  const warmupDays = Math.max(riskDays, mlDays ?? 0) + lag;
  return {
    ml,
    signal_lag_days: lag,
    warmup_days: warmupDays,
    min_days: warmupDays + 252,
    risk_days: riskDays,
    ml_days: mlDays,
    exact: mlDays === null || riskDays >= mlDays,
    trading_days_per_year: 252,
  };
}
const NAV0 = 5_000_000;
const DAILY_VOL = 0.066 / Math.sqrt(252);

export function businessDays(start: string, n: number): string[] {
  const out: string[] = [];
  const d = new Date(`${start}T00:00:00Z`);
  while (out.length < n) {
    const wd = d.getUTCDay();
    if (wd !== 0 && wd !== 6) out.push(d.toISOString().slice(0, 10));
    d.setUTCDate(d.getUTCDate() + 1);
  }
  return out;
}

function hex(rng: Rng, n: number): string {
  let s = '';
  for (let i = 0; i < n; i++) s += Math.floor(rng() * 16).toString(16);
  return s;
}

/** Mock data hash: same market spec, same hash (like `qts.core.manifest.hash_market_data`). */
export function mockDataHash(spec: Pick<RunSpec, 'kind' | 'seed' | 'stocks' | 'days'>): string {
  return hex(mulberry32(hashString(`${spec.kind}${spec.seed}${spec.stocks}${spec.days}`)), 16);
}

/**
 * Trial family the pipeline logs under: `pipeline_phase{p}:{data_hash[:12]}` (one DSR
 * search per phase and data set); records without a data hash are `legacy`.
 */
export function ledgerFamily(phase: number, dataHash: string): string {
  return dataHash ? `pipeline_phase${phase}:${dataHash.slice(0, 12)}` : 'legacy';
}

/** Fat-tailed shocks with volatility clustering, standardised to mean 0 / sd 1. */
function shocks(rng: Rng, n: number): number[] {
  let h = 1;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    h = Math.max(0.4, 0.94 * h + 0.06 * (1 + 0.6 * normal(rng)));
    const tail = rng() < 0.03 ? 2.2 : 1;
    out.push(normal(rng) * Math.sqrt(h) * tail);
  }
  return standardise(out);
}

function ar1(rng: Rng, n: number, mu: number, phi: number, sigma: number): number[] {
  const out: number[] = [];
  let x = mu;
  for (let i = 0; i < n; i++) {
    x = mu + phi * (x - mu) + sigma * normal(rng);
    out.push(x);
  }
  return out;
}

function skewKurt(xs: number[]): { skew: number; kurtosis: number } {
  const m = mean(xs);
  const s = sd(xs);
  let m3 = 0;
  let m4 = 0;
  for (const x of xs) {
    m3 += ((x - m) / s) ** 3;
    m4 += ((x - m) / s) ** 4;
  }
  return { skew: m3 / xs.length, kurtosis: m4 / xs.length };
}

function gate(
  name: GateResult['name'],
  status: GateResult['status'],
  value: GateResult['value'],
  threshold: GateResult['threshold'],
  reason: string,
  required = false,
): GateResult {
  return { name, status, value, threshold, reason, required };
}

const f2 = (x: number) => x.toFixed(2);

export function evaluateHurdles(args: {
  phase: Phase;
  decileT: number;
  icByFold: number[];
  foldSharpes: FoldSharpe[];
  breakeven: number;
  costBps: number;
  dsr: number;
  pbo: number | null;
  sharpeNet: number;
  /** failed rebalance solves / rebalance solves (`BacktestResult.failed_solve_fraction()`) */
  failedSolveFrac: number;
}): Hurdles {
  const {
    phase,
    decileT,
    icByFold,
    foldSharpes,
    breakeven,
    costBps,
    dsr,
    pbo,
    sharpeNet,
    failedSolveFrac,
  } = args;
  const posIcFolds = icByFold.filter((x) => x > 0).length;
  const posFolds = foldSharpes.filter((f) => f.sharpe_net >= 0.05).length;
  const bar = phase === 2 ? 0.5 : 0.3;
  const gates: GateResult[] = [
    gate(
      't_stat',
      decileT > 3 ? 'PASS' : 'FAIL',
      decileT,
      3,
      `decile L/S t = ${f2(decileT)} ${decileT > 3 ? '>' : '<='} 3.0`,
    ),
    gate(
      'ic_sign',
      posIcFolds / icByFold.length >= 2 / 3 ? 'PASS' : 'FAIL',
      `${posIcFolds}/${icByFold.length}`,
      '2/3',
      `IC > 0 in ${posIcFolds}/${icByFold.length} folds (need >= 0.67)`,
    ),
    gate('incremental_ic', 'skipped', null, 0.005, 'not applied (standalone signal)'),
    gate(
      'net_sharpe_delta',
      posFolds / foldSharpes.length >= 2 / 3 ? 'PASS' : 'FAIL',
      `${posFolds}/${foldSharpes.length}`,
      '2/3',
      `net Sharpe by fold (vs zero baseline) >= +0.05 in ${posFolds}/${foldSharpes.length} folds`,
    ),
    gate(
      'breakeven_cost',
      breakeven >= 2 * costBps ? 'PASS' : 'FAIL',
      breakeven,
      2 * costBps,
      `breakeven ${breakeven.toFixed(1)} bps ${breakeven >= 2 * costBps ? '>=' : '<'} 2x modeled ${costBps.toFixed(1)} bps (= ${(2 * costBps).toFixed(1)} bps)`,
    ),
    gate(
      'dsr',
      dsr > 0.95 ? 'PASS' : 'FAIL',
      dsr,
      0.95,
      `DSR = ${dsr.toFixed(3)} ${dsr > 0.95 ? '>' : '<='} 0.95`,
      true,
    ),
    pbo === null
      ? gate(
          'pbo',
          'skipped',
          null,
          0.5,
          'PBO not supplied (no configuration grid run; set pipeline_pbo_configs)',
        )
      : gate(
          'pbo',
          pbo < 0.5 ? 'PASS' : 'FAIL',
          pbo,
          0.5,
          `PBO = ${f2(pbo)} ${pbo < 0.5 ? '<' : '>='} 0.5`,
        ),
    gate(
      'phase_sharpe',
      sharpeNet >= bar ? 'PASS' : 'FAIL',
      sharpeNet,
      bar,
      `phase-${phase} net Sharpe ${f2(sharpeNet)} ${sharpeNet >= bar ? '>=' : '<'} ${bar}`,
      true,
    ),
    gate(
      'suspect',
      sharpeNet > 2 ? 'FAIL' : 'PASS',
      sharpeNet,
      2,
      sharpeNet > 2
        ? `net Sharpe ${f2(sharpeNet)} > 2: suspect until leakage/cost/selection ruled out`
        : 'net Sharpe within plausible range',
      true,
    ),
    gate(
      'failed_solves',
      failedSolveFrac <= 0.02 ? 'PASS' : 'FAIL',
      failedSolveFrac,
      0.02,
      `failed solves ${(failedSolveFrac * 100).toFixed(1)}% of rebalances ${failedSolveFrac <= 0.02 ? '<=' : '>'} 2.0%`,
    ),
    gate('risk_model_sanity', 'PASS', 0, 0.005, 'ex-ante vol alerts 0.0% of rebalances <= 0.5%'),
  ];
  const overall = gates.every((g) => g.status !== 'FAIL') ? 'PASS' : 'FAIL';
  return { overall, phase, required: ['phase_sharpe', 'suspect', 'dsr'], gates };
}

const ALPHA_THEMES = ['value', 'momentum', 'quality', 'investment', 'low_risk', 'short_interest'];
const RISK_ONLY = ['beta', 'size', 'residual_vol', 'liquidity', 'short_term_reversal'];
const INDUSTRIES = [
  'Software',
  'Banks',
  'Pharmaceuticals',
  'Retail',
  'Utilities',
  'Oil & gas',
  'Semiconductors',
  'Insurance',
  'Machinery',
  'Media',
];

type ConfigTree = { [key: string]: ConfigLeaf | ConfigTree };

/** The mock's SystemConfig: the sections and leaves the compare view diffs. */
function configTree(spec: RunSpec): ConfigTree {
  const tree: ConfigTree = {
    seed: spec.seed,
    universe: {
      top_n_by_mcap: Math.min(1000, spec.stocks),
      min_price: 5.0,
      min_adv_usd: 2e6,
      rank_offset: 0,
    },
    alpha: {
      feature_set: 'v1',
      // a list leaf is flattened to compact JSON, as the API's config_leaves does
      disabled_features: '[]',
      horizon_days: 20,
      use_ml: spec.use_ml,
      ic_window: 756,
      score_smoothing_halflife_days: 10.0,
      sample_weight_halflife_years: 0.0,
      embargo_days: null,
      themes: '["value","momentum","quality","low_risk"]',
      lgbm: { n_estimators: 400, learning_rate: 0.03, num_leaves: 15 },
    },
    risk: { n_factors: 12, halflife_days: 90, shrinkage: 'ledoit_wolf' },
    cost: {
      impact_coefficient: 0.3,
      spread_model: 'quoted',
      borrow_bps: 50,
      delisting_short_gain_haircut: 1.0,
    },
    portfolio: {
      book: 'long_short',
      gross_leverage_max: 2.0,
      risk_aversion_mode: 'target_vol',
      risk_aversion: 2.0,
      target_vol_annual: 0.1,
      trade_aversion: spec.label.includes('aversion 2') ? 2.0 : 1.0,
      max_position_weight: 0.02,
      rebalance_every_days: 5,
      beta_neutral: true,
    },
    execution: { signal_lag_days: spec.signal_lag_days, venue: 'routed' },
    validation: { n_splits: 6, dsr_threshold: 0.95, pbo_threshold: 0.5 },
  };
  for (const [path, value] of Object.entries(spec.overrides ?? {})) {
    const parts = path.split('.');
    let node: ConfigTree = tree;
    for (const part of parts.slice(0, -1)) {
      const next = node[part];
      if (typeof next !== 'object' || next === null) {
        const fresh: ConfigTree = {};
        node[part] = fresh;
        node = fresh;
      } else node = next;
    }
    node[parts[parts.length - 1] ?? path] = value;
  }
  return tree;
}

/** The config hash of a spec: same config leaves, same hash (like `SystemConfig.hash()`). */
export function mockConfigHash(spec: RunSpec): string {
  return hex(mulberry32(hashString(JSON.stringify(flattenConfig(configTree(spec))))), 16);
}

/** `RunConfig.source` of a spec: what `qts demo` / `qts run --store` would have recorded. */
export function mockSource(spec: RunSpec): RunSource {
  const synthetic = spec.kind === 'synthetic';
  const base: RunSource = {
    recorded: true,
    origin: 'cli',
    kind: spec.kind,
    stocks: synthetic ? spec.stocks : null,
    days: synthetic ? spec.days : null,
    store_path: synthetic ? null : MOCK_STORE_PATH,
    start: synthetic ? null : (spec.startDate ?? null),
    end: null,
    max_names: null,
    min_mcap_ever: null,
    max_rank: synthetic ? null : 1500,
    max_days: null,
    cache: false,
    venue: null,
    command: '',
    ...spec.source,
  };
  if (base.command) return base;
  const parts = synthetic
    ? [
        'qts',
        'demo',
        '--stocks',
        String(spec.stocks),
        '--days',
        String(spec.days),
        '--seed',
        String(spec.seed),
      ]
    : [
        'qts',
        'run',
        '--store',
        base.store_path ?? MOCK_STORE_PATH,
        ...(base.start ? ['--start', base.start] : []),
        ...(base.end ? ['--end', base.end] : []),
        ...(base.max_names !== null ? ['--max-names', String(base.max_names)] : []),
        ...(base.min_mcap_ever !== null ? ['--min-mcap-ever', String(base.min_mcap_ever)] : []),
        ...(base.max_rank != null ? ['--max-rank', String(base.max_rank)] : []),
      ];
  if (!synthetic && spec.seal?.load_coverage?.ok === false) parts.push('--allow-partial-coverage');
  if (spec.use_ml) parts.push('--ml');
  for (const [k, v] of Object.entries(spec.overrides ?? {}))
    parts.push('--set', `${k}=${String(v)}`);
  return { ...base, command: parts.join(' ') };
}

/** The store path the mock API "runs" with (`Health.store_path`). */
export const MOCK_STORE_PATH = 'data/store';

/** Dotted-path leaves of a config tree, like the API's `config_leaves`. */
export function flattenConfig(tree: ConfigTree, prefix = ''): ConfigLeaves {
  const out: ConfigLeaves = {};
  for (const [k, v] of Object.entries(tree)) {
    if (typeof v === 'object' && v !== null) Object.assign(out, flattenConfig(v, `${prefix}${k}.`));
    else out[`${prefix}${k}`] = v;
  }
  return out;
}

function yamlValue(v: ConfigLeaf): string {
  return v === null ? 'null' : String(v);
}

function toYaml(tree: ConfigTree, indent = ''): string[] {
  const lines: string[] = [];
  for (const [k, v] of Object.entries(tree)) {
    if (typeof v === 'object' && v !== null)
      lines.push(`${indent}${k}:`, ...toYaml(v, `${indent}  `));
    else lines.push(`${indent}${k}: ${yamlValue(v)}`);
  }
  return lines;
}

function configYaml(spec: RunSpec, tree: ConfigTree): string {
  return [`# config of run ${spec.id}`, ...toYaml(tree)].join('\n');
}

export function generateRun(spec: RunSpec): GeneratedRun {
  const rng = mulberry32(spec.seed * 7919 + hashString(spec.id));
  const n = Math.max(120, spec.days - mockWarmup(spec.use_ml, spec.signal_lag_days).warmup_days);
  const dates = businessDays(spec.startDate ?? '2019-12-10', n);

  // --- returns: gross calibrated exactly to the target Sharpe, costs to the net target
  const z = shocks(rng, n);
  const gross = z.map((x) => DAILY_VOL * (spec.sharpeGross / Math.sqrt(252) + x));
  const turnover = ar1(rng, n, 0.069, 0.85, 0.008).map((t) => Math.max(0.01, t));
  const meanCost = (DAILY_VOL * (spec.sharpeGross - spec.sharpeNet)) / Math.sqrt(252);
  const borrowShare = 0.18;
  const costPerSide = (meanCost * (1 - borrowShare)) / mean(turnover); // fraction per $ traded
  const borrowDaily = meanCost * borrowShare;
  const tradeCost = turnover.map((t) => t * costPerSide * (0.85 + 0.3 * rng()));
  const borrow = turnover.map(() => borrowDaily * (0.9 + 0.2 * rng()));
  const net = gross.map((g, i) => g - (tradeCost[i] ?? 0) - (borrow[i] ?? 0));

  const nav: number[] = [];
  let v = NAV0;
  for (const r of net) {
    v *= 1 + r;
    nav.push(v);
  }
  const navPrev = [NAV0, ...nav.slice(0, -1)];
  const drawdown = drawdownPath(net);
  const exanteVol = ar1(rng, n, 0.066 / 1.04, 0.97, 0.0015).map((x) => Math.max(0.03, x));

  // --- costs (USD per day)
  const spreadW = 0.46;
  const impactW = 0.39;
  const costsDaily = {
    spread: tradeCost.map((c, i) => c * spreadW * (navPrev[i] ?? NAV0)),
    impact: tradeCost.map((c, i) => c * impactW * (navPrev[i] ?? NAV0)),
    fees: tradeCost.map((c, i) => c * (1 - spreadW - impactW) * (navPrev[i] ?? NAV0)),
    borrow: borrow.map((b, i) => b * (navPrev[i] ?? NAV0)),
  };
  const tradedUsd = turnover.reduce((a, t, i) => a + t * (navPrev[i] ?? NAV0), 0);
  const tradingUsd = tradeCost.reduce((a, c, i) => a + c * (navPrev[i] ?? NAV0), 0);
  const costBps = (tradingUsd / tradedUsd) * 1e4;
  const breakeven = (mean(gross) / mean(turnover)) * 1e4;
  const grossExposure = 2.0 - 0.02 * rng();
  // --- book exposure by day: gross near the target, net near zero; the days the book was
  // empty (invested_day_frac < 1) come as a few contiguous gaps so the dossier can draw them
  const investedFrac = spec.investedDayFrac ?? 1;
  const emptyDays = new Set<number>();
  if (investedFrac < 1) {
    const total = Math.round(n * (1 - investedFrac));
    const gaps = Math.max(1, Math.min(3, Math.round(total / 4)));
    for (let g = 0; g < gaps; g++) {
      const len = Math.max(1, Math.round(total / gaps));
      const start = Math.floor(((g + 0.5) / gaps) * n) - Math.floor(len / 2);
      for (let k = 0; k < len && start + k < n; k++) emptyDays.add(Math.max(0, start + k));
    }
  }
  const grossExposureSeries = ar1(rng, n, grossExposure, 0.9, 0.015).map((x, i) =>
    emptyDays.has(i) ? 0 : x,
  );
  const netExposureSeries = ar1(rng, n, 0.0, 0.9, 0.006).map((x, i) => (emptyDays.has(i) ? 0 : x));

  // --- IC (sampled every H = 20 days)
  const H = 20;
  const icDates = dates.filter((_, i) => i % H === 0);
  const nIc = icDates.length;
  const ppy = 252 / H;
  const icirPeriod = spec.icirAnnual / Math.sqrt(ppy);
  const icSd = spec.meanIc / icirPeriod;
  const icZ = standardise(icDates.map(() => normal(rng)));
  const icSeries = icZ.map((x) => spec.meanIc + icSd * x);
  const byYearMap = new Map<number, number[]>();
  icDates.forEach((d, i) => {
    const y = Number(d.slice(0, 4));
    byYearMap.set(y, [...(byYearMap.get(y) ?? []), icSeries[i] ?? 0]);
  });
  const by_year: IcByYear[] = [...byYearMap.entries()].map(([year, xs]) => ({
    year,
    mean_ic: mean(xs),
    n: xs.length,
  }));
  const decay = [1, 5, 20, 60].map((h) => ({
    horizon_days: h,
    mean_ic: spec.meanIc * Math.min(1.15, Math.sqrt(h / H)) * Math.exp(-Math.max(0, h - H) / 400),
  }));

  // --- folds: validation.n_splits (6) contiguous purged walk-forward test blocks over the
  // backtest dates (qts.pipeline.purged_fold_blocks), labelled "YYYY-MM–YYYY-MM" like the API
  const N_SPLITS = 6;
  const edges = Array.from({ length: N_SPLITS + 1 }, (_, k) => Math.floor((k * n) / N_SPLITS));
  const blocks = edges.slice(0, -1).map((lo, k) => [lo, (edges[k + 1] ?? n) - 1] as const);
  const fold_sharpes: FoldSharpe[] = blocks.map(([lo, hi]) => ({
    label: `${(dates[lo] ?? '').slice(0, 7)}–${(dates[hi] ?? '').slice(0, 7)}`,
    sharpe_net: annualizedSharpe(net.slice(lo, hi + 1)),
  }));
  // mean sampled IC per block: the input of the ic_sign gate
  const icByFold = blocks
    .map(([lo, hi]) => {
      const a = dates[lo] ?? '';
      const b = dates[hi] ?? '';
      const xs = icDates.flatMap((d, i) => (d >= a && d <= b ? [icSeries[i] ?? 0] : []));
      return xs.length ? mean(xs) : null;
    })
    .filter((x): x is number => x !== null);

  // --- summary (all from the series)
  const sharpeNet = annualizedSharpe(net);
  const sharpeGross = annualizedSharpe(gross);
  const volNet = annualizedVol(net);
  const srd = sharpeNet / Math.sqrt(252);
  const mdd = maxDrawdown(net);
  const cagrNet = cagr(net);
  const { skew, kurtosis } = skewKurt(net);
  const dsr = deflatedSharpe({
    sharpeAnnual: sharpeNet,
    nObs: n,
    nTrials: spec.nTrials,
    varTrialsAnnual: spec.varTrials,
    skew,
    kurtosis,
  });
  const summary: BacktestSummary = {
    start: dates[0] ?? '',
    end: dates[n - 1] ?? '',
    n_days: n,
    final_nav: nav[n - 1] ?? NAV0,
    cagr_gross: cagr(gross),
    cagr_net: cagrNet,
    ann_vol_gross: annualizedVol(gross),
    ann_vol_net: volNet,
    sharpe_gross: sharpeGross,
    sharpe_net: sharpeNet,
    sharpe_net_lo: sharpeNet * 0.95,
    sharpe_t_stat_net: srd / Math.sqrt((1 + (srd * srd) / 2) / n),
    newey_west_t_net: sharpeNet * Math.sqrt(n / 252) * 1.04,
    max_drawdown_net: mdd,
    calmar_net: cagrNet / Math.abs(mdd),
    sortino_net: sortino(net),
    hit_rate_net: hitRate(net),
    avg_gross_exposure: mean(grossExposureSeries),
    avg_net_exposure: mean(netExposureSeries),
    avg_daily_turnover: mean(turnover),
    effective_holding_days: grossExposure / (mean(turnover) / 2),
    avg_cost_bps_per_side: costBps,
    avg_daily_cost_bps_of_nav: meanCost * 1e4,
    total_costs_usd:
      costsDaily.spread.reduce((a, b) => a + b, 0) +
      costsDaily.impact.reduce((a, b) => a + b, 0) +
      costsDaily.fees.reduce((a, b) => a + b, 0) +
      costsDaily.borrow.reduce((a, b) => a + b, 0),
    transfer_coefficient_mean: 0.38 + 0.12 * rng(),
    realized_vs_exante_vol_ratio: annualizedVol(gross) / mean(exanteVol),
    invested_day_frac: 1 - emptyDays.size / n,
    // fixed (no rng draw: the random stream of the fields below must not shift)
    effective_n_names: spec.signal_lag_days === 0 ? 268 : 312,
    specific_var_share: spec.signal_lag_days === 0 ? 0.61 : 0.56,
    exante_vol_at_gross2: (mean(exanteVol) * 2) / Math.max(grossExposure, 1e-9),
    breakeven_cost_bps: breakeven,
    n_delistings: Math.floor(rng() * 9),
    // fixed (no rng draw: the random stream of the fields below must not shift)
    delisting_pnl_ann: spec.signal_lag_days === 0 ? -0.0031 : -0.0018,
    n_failed_solves: spec.signal_lag_days === 0 ? 2 : 0,
  };

  const icMean = mean(icSeries);
  const ic_report = {
    mean_ic: icMean,
    ic_sd: sd(icSeries),
    icir: (icMean / sd(icSeries)) * Math.sqrt(ppy),
    t_stat: (icMean / sd(icSeries)) * Math.sqrt(nIc),
    hit_rate: icSeries.filter((x) => x > 0).length / nIc,
    periods_per_year: ppy,
    n: nIc,
    horizon_days: H,
    by_year,
  };

  const hurdles = evaluateHurdles({
    phase: spec.phase,
    decileT: spec.decileT,
    icByFold,
    foldSharpes: fold_sharpes,
    breakeven,
    costBps,
    dsr,
    pbo: spec.pbo,
    sharpeNet,
    failedSolveFrac: summary.n_failed_solves / Math.max(summary.n_days, 1),
  });

  const tree = configTree(spec);
  const config_leaves = flattenConfig(tree);
  const configHash = hex(mulberry32(hashString(JSON.stringify(config_leaves))), 16);
  const dataHash = mockDataHash(spec);
  const gitSha = hex(mulberry32(hashString(spec.created_at)), 40);

  const notes: string[] = [];
  if (sharpeNet > 2)
    notes.push(
      'net Sharpe > 2: treat as suspect until leakage/cost/selection ruled out (plan 2.4)',
    );
  if (summary.effective_holding_days < 50)
    notes.push(
      `effective holding ${summary.effective_holding_days.toFixed(0)}d < target 50d: raise trade_aversion`,
    );
  if (spec.pbo === null)
    notes.push('PBO skipped: set SystemConfig.pipeline_pbo_configs to run a CSCV grid');
  if (spec.signal_lag_days === 0)
    notes.push('signal_lag_days = 0: trades at the close that produced the signal (optimistic)');

  const cost_sensitivity: CostSensitivityRow[] = [0.5, 1, 2, 3].map((m) => {
    const nr = gross.map((g, i) => g - (m * costBps * (turnover[i] ?? 0)) / 1e4);
    return {
      multiplier: m,
      cost_bps_per_side: m * costBps,
      net_sharpe: annualizedSharpe(nr),
      net_cagr: cagr(nr),
      mean_daily_cost_bps: m * costBps * mean(turnover),
    };
  });

  const detail: RunDetail = {
    id: spec.id,
    label: spec.label,
    status: spec.status,
    summary,
    manifest: {
      run_id: spec.id,
      created_at: spec.created_at,
      git_sha: gitSha,
      git_dirty: false,
      config_hash: configHash,
      data_hash: dataHash,
      ledger_family: ledgerFamily(spec.phase, dataHash),
      python: '3.12.7',
      platform: 'Linux-6.8-x86_64',
      versions: {
        qts: '0.4.0',
        numpy: '2.3.1',
        pandas: '2.3.3',
        cvxpy: '1.7.2',
        lightgbm: '4.6.0',
      },
      timings: {
        universe: 0.2,
        risk_fit: 6.5,
        alpha_fit: spec.use_ml ? 214.3 : 9.8,
        backtest: 236.9 * (n / 1000),
      },
      seeds: { market: spec.seed, models: spec.seed + 1 },
    },
    config: {
      kind: spec.kind,
      use_ml: spec.use_ml,
      phase: spec.phase,
      stocks: spec.stocks,
      days: spec.days,
      seed: spec.seed,
      signal_lag_days: spec.signal_lag_days,
      config_yaml: configYaml(spec, tree),
      source: mockSource(spec),
    },
    hurdles,
    ic_report,
    decile: {
      mean: 0.0052 * (spec.decileT / 0.74),
      sharpe: spec.decileT * 0.51,
      t_stat: spec.decileT,
    },
    dsr,
    dsr_inputs: {
      n_trials: spec.nTrials,
      var_trials_sr_annual: spec.varTrials,
      skew,
      kurtosis,
      n_obs: n,
    },
    pbo: spec.pbo,
    fold_sharpes,
    execution_venue: 'Closing auction (LOC/MOC), IBKR ClosePx algo above 1% of ADV',
    notes,
    seal: mockSeal(spec),
  };

  const summaryRow: RunSummary = {
    id: spec.id,
    created_at: spec.created_at,
    label: spec.label,
    kind: spec.kind,
    phase: spec.phase,
    use_ml: spec.use_ml,
    config_hash: configHash,
    data_hash: dataHash,
    ledger_family: ledgerFamily(spec.phase, dataHash),
    git_sha: gitSha,
    status: spec.status,
    seal: mockSeal(spec),
    headline:
      spec.status === 'done'
        ? {
            sharpe_net: sharpeNet,
            sharpe_gross: sharpeGross,
            cagr_net: cagrNet,
            max_drawdown: mdd,
            avg_cost_bps_per_side: costBps,
            dsr,
            invested_day_frac: summary.invested_day_frac,
            overall_gate: hurdles.overall,
          }
        : null,
    config_leaves: spec.status === 'done' ? config_leaves : {},
  };

  // --- exposures: alpha themes tilt, risk-only styles stay inside their bands
  const factors = [...ALPHA_THEMES, ...RISK_ONLY];
  const roles: Record<string, ExposureSnapshotRow['role']> = {};
  const history: Record<string, number[]> = {};
  for (const f of factors) {
    const isTheme = ALPHA_THEMES.includes(f);
    roles[f] = isTheme ? 'alpha_theme' : 'risk_only';
    const mu = isTheme ? 0.08 + 0.3 * rng() * (f === 'low_risk' ? 0.4 : 1) : 0;
    history[f] = ar1(rng, n, mu, 0.985, isTheme ? 0.012 : 0.004).map((x) =>
      isTheme ? x : Math.max(-0.05, Math.min(0.05, x)),
    );
  }
  const latest = dates[n - 1] ?? '';
  const exposures: RunExposures = {
    factors,
    roles,
    history: { dates, series: history },
    latest: {
      date: latest,
      rows: factors.map((f) => ({
        factor: f,
        exposure: history[f]?.[n - 1] ?? 0,
        role: roles[f] ?? 'risk_only',
      })),
    },
  };

  const monthEnds = dates.filter((d, i) => dates[i + 1]?.slice(0, 7) !== d.slice(0, 7)).slice(-12);
  const holdingsFor = (date: string | undefined): RunHoldings => {
    const d = date && monthEnds.includes(date) ? date : latest;
    const hr = mulberry32(hashString(`${spec.id}${d}`));
    const positions: Holding[] = [];
    const used = new Set<string>();
    for (let i = 0; i < 120; i++) {
      const long = i < 60;
      const alpha = (long ? 1 : -1) * (0.002 + 0.01 * hr());
      let ticker = `SYN${String(Math.floor(hr() * 300)).padStart(3, '0')}`;
      while (used.has(ticker)) ticker = `SYN${String(Math.floor(hr() * 300)).padStart(3, '0')}`;
      used.add(ticker);
      positions.push({
        ticker,
        weight: (long ? 1 : -1) * (0.0105 + 0.0093 * hr()) * (grossExposure / 2),
        alpha,
        specific_vol: 0.18 + 0.25 * hr(),
        industry: INDUSTRIES[Math.floor(hr() * INDUSTRIES.length)] ?? 'Other',
      });
    }
    positions.sort((a, b) => b.weight - a.weight);
    return { date: d, available_dates: monthEnds, positions };
  };

  return {
    spec,
    summary: summaryRow,
    detail,
    dates,
    series: {
      nav,
      gross,
      net,
      drawdown,
      turnover,
      exante_vol: exanteVol,
      gross_exposure: grossExposureSeries,
      net_exposure: netExposureSeries,
    },
    costs: { daily: { dates, series: costsDaily }, nav: navPrev, cost_sensitivity },
    exposures,
    ic: { ic: { dates: icDates, series: { rank_ic: icSeries } }, decay, horizon_days: H },
    holdingsFor,
    alphaRefits: mockAlphaRefits({ id: spec.id, dates, useMl: spec.use_ml, level: spec.icLevel }),
    themeWeights: mockThemeWeights({ id: spec.id, dates, dynamic: true }),
    icByYear: mockIcByYear({ id: spec.id, dates, useMl: spec.use_ml, level: spec.icLevel }),
    constraints: mockConstraints({
      id: spec.id,
      dates,
      turnover,
      emptyDays,
      tradeAversion: Number(flattenConfig(tree)['portfolio.trade_aversion'] ?? 1),
      volTargetMode: 'target',
    }),
  };
}

/** `GET /api/runs/{id}/files`: the run directory as the pipeline writes it. */
const RUN_FILES: readonly {
  name: string;
  kind: RunFile['kind'];
  description: string;
  size: number;
}[] = [
  {
    name: 'config.yaml',
    kind: 'yaml',
    description: 'The SystemConfig the run used',
    size: 3_912,
  },
  {
    name: 'costs.parquet',
    kind: 'parquet',
    description: 'Spread, impact, fees and borrow in USD by decision date',
    size: 61_440,
  },
  {
    name: 'exante_vol.parquet',
    kind: 'parquet',
    description: "The risk model's ex-ante portfolio volatility by decision date",
    size: 19_874,
  },
  {
    name: 'exposures.parquet',
    kind: 'parquet',
    description: 'Net book exposure to each style factor by decision date',
    size: 84_215,
  },
  {
    name: 'extras.json',
    kind: 'json',
    description: 'Run facts recorded at save time (label, universe size, venue, exposure roles)',
    size: 1_204,
  },
  {
    name: 'holdings.parquet',
    kind: 'parquet',
    description: 'Month-end positions with alpha, specific vol and industry',
    size: 112_330,
  },
  {
    name: 'hurdles.json',
    kind: 'json',
    description: 'Hurdle gates with values, thresholds and reasons',
    size: 2_871,
  },
  { name: 'ic.parquet', kind: 'parquet', description: 'Rank IC per formation date', size: 5_120 },
  {
    name: 'manifest.json',
    kind: 'json',
    description: 'Run manifest: git sha, config and data hashes, library versions, timings, seeds',
    size: 1_588,
  },
  {
    name: 'nav.parquet',
    kind: 'parquet',
    description: 'Net asset value by realisation date',
    size: 14_336,
  },
  {
    name: 'returns.parquet',
    kind: 'parquet',
    description: 'Gross and net daily returns',
    size: 22_118,
  },
  {
    name: 'summary.json',
    kind: 'json',
    description:
      "Backtest summary, IC report, DSR inputs, cost sensitivity and the pipeline's notes",
    size: 9_733,
  },
  {
    name: 'trades.parquet',
    kind: 'parquet',
    description: 'Trades by decision date (fraction of NAV)',
    size: 1_480_204,
  },
  {
    name: 'weights.parquet',
    kind: 'parquet',
    description: 'Post-trade weights by decision date (fraction of NAV)',
    size: 1_902_577,
  },
];

export function runFiles(run: GeneratedRun): RunFile[] {
  const n = run.dates.length;
  return RUN_FILES.map((f) => ({
    name: f.name,
    kind: f.kind,
    description: f.description,
    size_bytes: f.kind === 'parquet' ? Math.round((f.size * n) / 1000) : f.size,
    modified_at: run.spec.created_at.replace('Z', '+00:00'),
    download_url: `/api/runs/${run.spec.id}/files/${f.name}`,
  }));
}

/** The downloadable body of a run file; JSON/YAML are real, parquet is a placeholder. */
export function runFileBody(
  run: GeneratedRun,
  name: string,
): { text: string; type: string } | null {
  const f = RUN_FILES.find((x) => x.name === name);
  if (!f) return null;
  if (name === 'manifest.json')
    return { text: JSON.stringify(run.detail.manifest, null, 2), type: 'application/json' };
  if (name === 'hurdles.json')
    return { text: JSON.stringify(run.detail.hurdles, null, 2), type: 'application/json' };
  if (name === 'summary.json')
    return {
      text: JSON.stringify(
        { backtest: run.detail.summary, ic_report: run.detail.ic_report, notes: run.detail.notes },
        null,
        2,
      ),
      type: 'application/json',
    };
  if (name === 'extras.json')
    return {
      text: JSON.stringify({ run: { label: run.spec.label, venue: run.detail.execution_venue } }),
      type: 'application/json',
    };
  if (name === 'config.yaml')
    return { text: run.detail.config.config_yaml, type: 'application/yaml' };
  return { text: `PAR1 (mock placeholder for ${name})`, type: 'application/vnd.apache.parquet' };
}
