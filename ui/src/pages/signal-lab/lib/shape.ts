import type { AlphaIcByYear, AlphaRefits, RunIc } from '@/shared/api';
import { rollingMean } from '@/shared/lib/series';

/** "theme_low_risk" → "Low risk", "lgbm" → "LightGBM", "combined" → "Combined". */
export function sleeveLabel(sleeve: string): string {
  const named: Record<string, string> = {
    combined: 'Combined',
    composite: 'Composite',
    ml: 'ML sleeve',
    lgbm: 'LightGBM (gain)',
    ridge: 'Ridge (|coef|)',
    mlp: 'MLP',
    ensemble: 'Ensemble (mean)',
  };
  if (sleeve in named) return named[sleeve] ?? sleeve;
  const s = sleeve.replace(/^theme_/, '').replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export interface HeatGrid {
  rows: string[];
  rowKeys: string[];
  cols: string[];
  values: (number | null)[][];
  /** Companion grid (t-stats) in the same layout, when the source has them. */
  t?: (number | null)[][];
  /** Largest |value|, for a shared tint scale. */
  maxAbs: number;
}

/** Sleeve × year grid of mean IC (and t) from `/alpha/ic-by-year`. */
export function icByYearGrid(data: AlphaIcByYear): HeatGrid {
  const rows = data.rows.map((r) => sleeveLabel(r.sleeve));
  const cols = data.years.map(String);
  const values = data.rows.map((r) =>
    data.years.map((y) => r.cells.find((c) => c.year === y)?.mean_ic ?? null),
  );
  const t = data.rows.map((r) =>
    data.years.map((y) => r.cells.find((c) => c.year === y)?.t_stat ?? null),
  );
  return {
    rows,
    rowKeys: data.rows.map((r) => r.sleeve),
    cols,
    values,
    t,
    maxAbs: maxAbsOf(values),
  };
}

export function maxAbsOf(values: readonly (readonly (number | null)[])[]): number {
  let m = 0;
  for (const row of values)
    for (const v of row) if (v !== null && Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  return m;
}

/** Each column (refit) scaled to sum 1 over the finite values; an all-null column stays null. */
export function normaliseColumns(
  values: readonly (readonly (number | null)[])[],
): (number | null)[][] {
  const nRows = values.length;
  const nCols = values[0]?.length ?? 0;
  const out: (number | null)[][] = values.map((r) => r.map(() => null));
  for (let j = 0; j < nCols; j++) {
    let sum = 0;
    for (let i = 0; i < nRows; i++) {
      const v = values[i]?.[j];
      if (v !== null && v !== undefined && Number.isFinite(v)) sum += Math.max(0, v);
    }
    for (let i = 0; i < nRows; i++) {
      const v = values[i]?.[j];
      const row = out[i];
      if (row && v !== null && v !== undefined && Number.isFinite(v))
        row[j] = sum > 0 ? Math.max(0, v) / sum : 0;
    }
  }
  return out;
}

/**
 * Features × refit dates of one sleeve's importance, each refit normalised to sum 1 and
 * the features ordered by their mean share (largest first) so the heavy rows sit on top.
 */
export function refitGrid(data: AlphaRefits, sleeve: string): HeatGrid | null {
  const table = data.importance[sleeve];
  if (!table || !data.refits.length) return null;
  const features = data.features.filter((f) => f in table);
  const raw = features.map((f) => data.refits.map((_, j) => table[f]?.[j] ?? null));
  const norm = normaliseColumns(raw);
  const meanShare = norm.map((r) => {
    const xs = r.filter((v): v is number => v !== null);
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : -1;
  });
  const order = features.map((_, i) => i).sort((a, b) => (meanShare[b] ?? 0) - (meanShare[a] ?? 0));
  const values = order.map((i) => norm[i] ?? []);
  return {
    rows: order.map((i) => features[i] ?? ''),
    rowKeys: order.map((i) => features[i] ?? ''),
    cols: data.refits.map((r) => r.refit_date),
    values,
    maxAbs: maxAbsOf(values),
  };
}

/** Year labels for refit-date columns: the year where it first appears, else hidden. */
export function yearLabel(cols: readonly string[]): (col: string, index: number) => string | null {
  return (col, j) => {
    const y = col.slice(0, 4);
    return j === 0 || cols[j - 1]?.slice(0, 4) !== y ? y : null;
  };
}

/** The IC series with its trailing one-year mean (window = 252 / horizon sampled points). */
export function icWithRollingMean(ic: RunIc): {
  dates: string[];
  raw: (number | null)[];
  rolling: (number | null)[];
  window: number;
} {
  const window = Math.max(2, Math.round(252 / Math.max(1, ic.horizon_days)));
  const raw = ic.ic.series.rank_ic;
  return {
    dates: ic.ic.dates,
    raw: [...raw],
    rolling: rollingMean(raw, window, Math.ceil(window / 2)),
    window,
  };
}

/** Rows of the feature IC table, strongest |t| first. */
export function featureIcRows(data: AlphaRefits) {
  return [...data.feature_ic].sort((a, b) => Math.abs(b.t_stat ?? 0) - Math.abs(a.t_stat ?? 0));
}

/** One sentence on how the by-year story reads: which theme faded and which rose. */
export function storyLine(data: AlphaIcByYear): string {
  const themes = data.rows.filter((r) => r.sleeve.startsWith('theme_'));
  if (data.years.length < 4 || !themes.length) return '';
  const half = Math.floor(data.years.length / 2);
  const delta = themes.map((r) => {
    const first = mean(r.cells.slice(0, half).map((c) => c.mean_ic));
    const last = mean(r.cells.slice(half).map((c) => c.mean_ic));
    return { label: sleeveLabel(r.sleeve).toLowerCase(), delta: last - first, first, last };
  });
  const faded = [...delta].sort((a, b) => a.delta - b.delta)[0];
  const rose = [...delta].sort((a, b) => b.delta - a.delta)[0];
  if (!faded || !rose) return '';
  const fmt = (v: number) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3);
  return `Over the two halves of the sample ${faded.label} moved most against the book (${fmt(faded.first)} to ${fmt(faded.last)}) and ${rose.label} most in its favour (${fmt(rose.first)} to ${fmt(rose.last)}).`;
}

function mean(xs: readonly (number | null)[]): number {
  const v = xs.filter((x): x is number => x !== null && Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
}
