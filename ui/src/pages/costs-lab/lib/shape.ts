import type { RunConstraints, RunCosts } from '@/shared/api';
import {
  groupIndices,
  histogram,
  meanAt,
  monthKey,
  quantile,
  shareByGroup,
} from '@/shared/lib/series';
import type { Bin } from '@/shared/lib/series';

/** Constraint names in display order with a short label. */
export const CONSTRAINT_LABELS: Record<string, string> = {
  gross: 'Gross leverage',
  position: 'Position cap',
  participation: 'ADV participation',
  style_bands: 'Style bands',
  industry: 'Industry bands',
  beta: 'Beta band',
  dollar_neutral: 'Dollar neutral',
  no_trade: 'No-trade names',
  force_to_zero: 'Forced exits',
  vol_cap: 'Vol cap',
  turnover: 'Turnover cap',
};

export function constraintLabel(name: string): string {
  if (name in CONSTRAINT_LABELS) return CONSTRAINT_LABELS[name] ?? name;
  const s = name.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export const VOL_REASON_LABELS: Record<string, string> = {
  target: 'At target',
  caps: 'Capped below target',
  bracket: 'Gamma at bracket end',
  max_solves: 'Search budget hit',
  no_alpha: 'No alpha (empty book)',
};

export function volReasonLabel(reason: string): string {
  if (reason in VOL_REASON_LABELS) return VOL_REASON_LABELS[reason] ?? reason;
  if (reason.startsWith('solve_failed')) return 'Solve failed';
  return reason || 'Not vol-targeted';
}

/** Period key: calendar month, or quarter when the run spans more than `maxMonths` months. */
export function periodKeys(
  dates: readonly string[],
  maxMonths = 120,
): { keys: string[]; grain: 'month' | 'quarter' } {
  const months = new Set(dates.map(monthKey));
  if (months.size <= maxMonths) return { keys: dates.map(monthKey), grain: 'month' };
  return {
    keys: dates.map((d) => `${d.slice(0, 4)} Q${Math.floor((Number(d.slice(5, 7)) - 1) / 3) + 1}`),
    grain: 'quarter',
  };
}

export interface BindingGrid {
  rows: string[];
  rowKeys: string[];
  cols: string[];
  /** Share of the period's rebalances on which the constraint bound. */
  values: (number | null)[][];
  grain: 'month' | 'quarter';
}

/** Constraint × period grid of the share of solves where each constraint bound. */
export function bindingGrid(c: RunConstraints): BindingGrid {
  const { keys, grain } = periodKeys(c.dates);
  const groups = groupIndices(keys);
  const names = [...c.names].sort((a, b) => (c.share_binding[b] ?? 0) - (c.share_binding[a] ?? 0));
  const values = names.map((n) => {
    const counts = c.binding[n] ?? [];
    return groups.map((g) => {
      let bound = 0;
      let solved = 0;
      for (const i of g.indices) {
        if (c.status[i] === 'no_alpha') continue;
        solved++;
        if ((counts[i] ?? 0) > 0) bound++;
      }
      return solved ? bound / solved : null;
    });
  });
  return {
    rows: names.map(constraintLabel),
    rowKeys: names,
    cols: groups.map((g) => g.key),
    values,
    grain,
  };
}

/** Monthly (or quarterly) share of each vol-target binding reason, for a 0–1 stacked area. */
export function volReasonShares(c: RunConstraints): {
  dates: string[];
  reasons: string[];
  series: Record<string, number[]>;
} {
  const { keys } = periodKeys(c.dates);
  const groups = groupIndices(keys);
  const reasons = [...c.vol_target_reasons].map((r) =>
    r.startsWith('solve_failed') ? 'solve_failed' : r,
  );
  const uniq = [...new Set(reasons)];
  const order = ['target', 'caps', 'bracket', 'max_solves', 'no_alpha', 'solve_failed'];
  const ordered = uniq.sort(
    (a, b) => ((order.indexOf(a) + 99) % 99) - ((order.indexOf(b) + 99) % 99),
  );
  const category = c.vol_target_binding.map((v) =>
    v.startsWith('solve_failed') ? 'solve_failed' : v,
  );
  const shares = shareByGroup(groups, category, ordered);
  const series: Record<string, number[]> = Object.fromEntries(
    ordered.map((r) => [r, shares.map((s) => s.shares[r] ?? 0)]),
  );
  const dates = groups.map((g) => c.dates[g.indices[0] ?? 0] ?? '');
  return { dates, reasons: ordered, series };
}

/** Participation-cap usage: the share of traded names at the ADV cap, by period. */
export function participationUsage(c: RunConstraints): {
  dates: string[];
  share: (number | null)[];
} {
  const counts = c.binding.participation;
  if (!counts) return { dates: [], share: [] };
  const perDay = counts.map((k, i) => {
    const n = c.n_traded[i] ?? 0;
    return n > 0 ? Math.min(1, k / n) : null;
  });
  const { keys } = periodKeys(c.dates);
  const groups = groupIndices(keys);
  return {
    dates: groups.map((g) => c.dates[g.indices[0] ?? 0] ?? ''),
    share: groups.map((g) => meanAt(perDay, g.indices)),
  };
}

/** Daily trading cost per dollar traded in bp (spread + impact + fees; borrow is a holding cost). */
export function costPerSideDaily(
  costs: RunCosts,
  turnover: readonly (number | null)[],
  dates: readonly string[],
): (number | null)[] {
  const byDate = new Map(dates.map((d, i) => [d, turnover[i] ?? null]));
  return costs.daily.dates.map((d, i) => {
    const nav = costs.nav[i];
    const to = byDate.get(d);
    if (nav === null || nav === undefined || to === null || to === undefined || to <= 1e-6)
      return null;
    const usd =
      (costs.daily.series.spread[i] ?? 0) +
      (costs.daily.series.impact[i] ?? 0) +
      (costs.daily.series.fees[i] ?? 0);
    return (usd / (to * nav)) * 1e4;
  });
}

export interface CostDistribution {
  bins: Bin[];
  median: number | null;
  p90: number | null;
  mean: number | null;
  n: number;
}

export function costDistribution(
  perSide: readonly (number | null)[],
  nBins = 24,
): CostDistribution {
  const xs = perSide.filter((v): v is number => v !== null && Number.isFinite(v));
  const lo = quantile(xs, 0.01) ?? 0;
  const hi = quantile(xs, 0.99) ?? 0;
  const bins = histogram(xs, nBins, [lo, Math.max(hi, lo + 1e-6)]);
  return {
    bins,
    median: quantile(xs, 0.5),
    p90: quantile(xs, 0.9),
    mean: xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null,
    n: xs.length,
  };
}

/** Period means of a daily series (for a long series drawn as monthly/quarterly points). */
export function periodMeans(
  dates: readonly string[],
  values: readonly (number | null)[],
): { dates: string[]; values: (number | null)[] } {
  const { keys } = periodKeys(dates);
  const groups = groupIndices(keys);
  return {
    dates: groups.map((g) => dates[g.indices[0] ?? 0] ?? ''),
    values: groups.map((g) => meanAt(values, g.indices)),
  };
}
