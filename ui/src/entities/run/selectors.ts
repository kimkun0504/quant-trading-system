import type { CostComponent, RunCosts, RunDetail, RunSummary } from '@/shared/api';
import type { MetricContext } from '@/entities/metric/definitions';
import { annualizedSharpe } from '@/shared/lib/stats';

/** Facts about a run used by metric interpretations. */
export function metricContext(run: RunDetail): MetricContext {
  const s = run.summary;
  const ic = run.ic_report;
  return {
    phase: run.config.phase,
    nDays: s.n_days,
    sharpeNet: s.sharpe_net,
    sharpeGross: s.sharpe_gross,
    cagrGross: s.cagr_gross,
    annVolNet: s.ann_vol_net,
    costBps: s.avg_cost_bps_per_side,
    breakevenBps: s.breakeven_cost_bps,
    icHorizonDays: ic.horizon_days,
    icN: ic.n,
    icPeriodsPerYear: ic.periods_per_year,
    icir: ic.icir,
    icTStat: ic.t_stat,
    icHitRate: ic.hit_rate,
    nTrials: run.dsr_inputs.n_trials,
  };
}

/** "Phase 1 · composite" style descriptor, in words. */
export function runDescriptor(r: Pick<RunSummary, 'phase' | 'use_ml' | 'kind'>): string {
  const model = r.use_ml ? 'composite + ML' : 'composite';
  const data = r.kind === 'synthetic' ? 'synthetic market' : 'data store';
  return `Phase ${r.phase}, ${model}, ${data}`;
}

export interface MonthCell {
  year: number;
  month: number; // 0-11
  ret: number;
}

export interface MonthlyReturns {
  years: number[];
  cells: MonthCell[];
  annual: Map<number, number>;
}

/** Compound daily returns into calendar months and years. */
export function monthlyReturns(
  dates: readonly string[],
  rets: readonly (number | null)[],
): MonthlyReturns {
  const acc = new Map<string, { year: number; month: number; g: number }>();
  const yearAcc = new Map<number, number>();
  dates.forEach((d, i) => {
    const r = rets[i];
    if (r === null || r === undefined || !Number.isFinite(r)) return;
    const year = Number(d.slice(0, 4));
    const month = Number(d.slice(5, 7)) - 1;
    const key = `${year}-${month}`;
    const cur = acc.get(key) ?? { year, month, g: 1 };
    cur.g *= 1 + r;
    acc.set(key, cur);
    yearAcc.set(year, (yearAcc.get(year) ?? 1) * (1 + r));
  });
  const cells = [...acc.values()].map((c) => ({ year: c.year, month: c.month, ret: c.g - 1 }));
  const years = [...yearAcc.keys()].sort((a, b) => a - b);
  const annual = new Map([...yearAcc.entries()].map(([y, g]) => [y, g - 1]));
  return { years, cells, annual };
}

/** Rolling annualised Sharpe over `window` observations; null until the window fills. */
export function rollingSharpe(rets: readonly (number | null)[], window = 126): (number | null)[] {
  const out: (number | null)[] = [];
  const buf: number[] = [];
  for (const r of rets) {
    if (r !== null && Number.isFinite(r)) buf.push(r);
    if (buf.length > window) buf.shift();
    out.push(buf.length === window ? annualizedSharpe(buf) : null);
  }
  return out;
}

export const COST_COMPONENTS: readonly CostComponent[] = ['spread', 'impact', 'fees', 'borrow'];

export interface CostBar {
  label: string;
  values: Record<CostComponent, number>;
}

/** Daily cost by component in bp of that day's NAV. */
export function dailyCostBars(costs: RunCosts): CostBar[] {
  return costs.daily.dates.map((d, i) => {
    const nav = costs.nav[i] ?? Number.NaN;
    const values = {} as Record<CostComponent, number>;
    for (const c of COST_COMPONENTS) values[c] = ((costs.daily.series[c][i] ?? 0) / nav) * 1e4;
    return { label: d, values };
  });
}

/** Costs summed by calendar month, in bp of NAV. */
export function monthlyCostBars(costs: RunCosts): CostBar[] {
  const out = new Map<string, CostBar>();
  for (const bar of dailyCostBars(costs)) {
    const key = bar.label.slice(0, 7);
    const cur = out.get(key) ?? {
      label: key,
      values: { spread: 0, impact: 0, fees: 0, borrow: 0 },
    };
    for (const c of COST_COMPONENTS) cur.values[c] += bar.values[c];
    out.set(key, cur);
  }
  return [...out.values()];
}

/** Share of total cost by component over the whole run. */
export function costShares(costs: RunCosts): Record<CostComponent, number> {
  const tot = { spread: 0, impact: 0, fees: 0, borrow: 0 };
  for (const c of COST_COMPONENTS) for (const v of costs.daily.series[c]) tot[c] += v ?? 0;
  const all = tot.spread + tot.impact + tot.fees + tot.borrow;
  return {
    spread: tot.spread / all,
    impact: tot.impact / all,
    fees: tot.fees / all,
    borrow: tot.borrow / all,
  };
}
