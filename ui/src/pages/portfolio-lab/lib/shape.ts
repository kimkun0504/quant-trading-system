import type { Holding, RunExposures, RunHoldings } from '@/shared/api';
import { rollingMean, rollingSd, thinIndices } from '@/shared/lib/series';

/** Gross exposure below this share of NAV is an empty book (`INVESTED_GROSS_EPS`). */
export const EMPTY_BOOK = 1e-4;
export const MAX_INDUSTRIES = 7;
export const OTHER = 'Other';

export interface IndustryStack {
  dates: string[];
  industries: string[];
  /** `long[industry][i]` ≥ 0 and `short[industry][i]` ≤ 0: gross weight per side. */
  long: Record<string, number[]>;
  short: Record<string, number[]>;
}

/**
 * Long and short gross weight by industry on each holdings date. The `MAX_INDUSTRIES`
 * largest industries by mean gross weight keep their name; the rest fold into Other.
 */
export function industryStack(snapshots: readonly RunHoldings[]): IndustryStack {
  const dates = snapshots.map((s) => s.date);
  const gross = new Map<string, number>();
  for (const s of snapshots)
    for (const p of s.positions) {
      const k = p.industry ?? OTHER;
      gross.set(k, (gross.get(k) ?? 0) + Math.abs(p.weight));
    }
  const ranked = [...gross.entries()]
    .filter(([k]) => k !== OTHER)
    .sort((a, b) => b[1] - a[1])
    .map(([k]) => k);
  const kept = ranked.slice(0, MAX_INDUSTRIES);
  const industries = ranked.length > MAX_INDUSTRIES || gross.has(OTHER) ? [...kept, OTHER] : kept;
  const long: Record<string, number[]> = Object.fromEntries(
    industries.map((k) => [k, dates.map(() => 0)]),
  );
  const short: Record<string, number[]> = Object.fromEntries(
    industries.map((k) => [k, dates.map(() => 0)]),
  );
  snapshots.forEach((s, i) => {
    for (const p of s.positions) {
      const raw = p.industry ?? OTHER;
      const k = kept.includes(raw) ? raw : OTHER;
      const side = p.weight >= 0 ? long : short;
      const arr = side[k];
      if (arr) arr[i] = (arr[i] ?? 0) + p.weight;
    }
  });
  return { dates, industries, long, short };
}

/** The dates of `available` to request for the industry stack: at most `max`, evenly spaced, the last always. */
export function stackDates(available: readonly string[], max = 36): string[] {
  return thinIndices(available.length, max).map((i) => available[i] ?? '');
}

/** Annualised trailing realised vol of the net return next to the ex-ante forecast. */
export function realisedVsExante(
  net: readonly (number | null)[],
  exante: readonly (number | null)[],
  window = 63,
  periodsPerYear = 252,
): { realised: (number | null)[]; exante: (number | null)[] } {
  const sd = rollingSd(net, window, Math.max(20, Math.floor(window * 0.8)));
  return {
    realised: sd.map((v) => (v === null ? null : v * Math.sqrt(periodsPerYear))),
    exante: [...exante],
  };
}

/** Trailing effective holding period: mean gross / (mean two-sided turnover / 2), in days. */
export function holdingPeriodSeries(
  turnover: readonly (number | null)[],
  gross: readonly (number | null)[],
  window = 63,
): (number | null)[] {
  const to = rollingMean(turnover, window, Math.max(10, Math.floor(window / 2)));
  const g = rollingMean(gross, window, Math.max(10, Math.floor(window / 2)));
  return to.map((t, i) => {
    const gi = g[i];
    if (t === null || gi === null || gi === undefined || t <= 0 || gi <= EMPTY_BOOK) return null;
    return gi / (t / 2);
  });
}

/** Largest `n` longs and shorts of a snapshot. */
export function topPositions(
  positions: readonly Holding[],
  n = 10,
): { longs: Holding[]; shorts: Holding[] } {
  const sorted = [...positions].sort((a, b) => b.weight - a.weight);
  return {
    longs: sorted.filter((p) => p.weight > 0).slice(0, n),
    shorts: sorted
      .filter((p) => p.weight < 0)
      .sort((a, b) => a.weight - b.weight)
      .slice(0, n),
  };
}

export interface SnapshotTotals {
  n: number;
  nLong: number;
  nShort: number;
  gross: number;
  net: number;
  /** Largest |weight|. */
  maxAbs: number;
}

export function snapshotTotals(positions: readonly Holding[]): SnapshotTotals {
  let gross = 0;
  let net = 0;
  let maxAbs = 0;
  let nLong = 0;
  for (const p of positions) {
    gross += Math.abs(p.weight);
    net += p.weight;
    maxAbs = Math.max(maxAbs, Math.abs(p.weight));
    if (p.weight > 0) nLong++;
  }
  return { n: positions.length, nLong, nShort: positions.length - nLong, gross, net, maxAbs };
}

/** Alpha-theme factors first (bold), then risk-only, market last. */
export function orderedFactors(x: RunExposures): string[] {
  const rank = (f: string) =>
    x.roles[f] === 'alpha_theme' ? 0 : x.roles[f] === 'risk_only' ? 1 : 2;
  return [...x.factors].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

/** Month-end positions of a date list (the last date of each calendar month). */
export function monthEndIndices(dates: readonly string[]): number[] {
  const out: number[] = [];
  for (let i = 0; i < dates.length; i++)
    if (dates[i + 1]?.slice(0, 7) !== dates[i]?.slice(0, 7)) out.push(i);
  return out;
}
