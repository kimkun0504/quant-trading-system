import type { SweepManifest, SweepMember } from '@/shared/api';
import type { Field } from '@/features/run-builder/lib/fields';
import { bps, num, pct, sharpe } from '@/shared/lib/format';
import type { BadgeTone } from '@/shared/ui';

/** A sweep parameter as the grid editor holds it: a config path and the typed values text. */
export interface ParamDraft {
  key: string;
  text: string;
}

export type SweepValue = boolean | number | string;

/** The members a sweep may have (`qts sweep --max-runs`; the API refuses more). */
export const MAX_MEMBERS = 12;
export const MAX_PARAMS = 4;

/**
 * The fields a sweep can vary: numbers, enums and switches. Free text and the feature
 * multi-select are left out (a sweep over feature lists is a family of its own).
 */
export function sweepableFields(fields: readonly Field[]): Field[] {
  return fields.filter(
    (f) =>
      f.type === 'number' ||
      f.type === 'integer' ||
      f.type === 'boolean' ||
      (f.type === 'string' && f.enum !== undefined),
  );
}

export type ParsedValues = { ok: true; values: SweepValue[] } | { ok: false; error: string };

function inBounds(f: Field, n: number): string | null {
  if (f.minimum !== undefined && n < f.minimum)
    return `${String(n)} is below the minimum ${String(f.minimum)}.`;
  if (f.maximum !== undefined && n > f.maximum)
    return `${String(n)} is above the maximum ${String(f.maximum)}.`;
  if (f.exclusiveMinimum !== undefined && n <= f.exclusiveMinimum)
    return `${String(n)} must be above ${String(f.exclusiveMinimum)}.`;
  if (f.exclusiveMaximum !== undefined && n >= f.exclusiveMaximum)
    return `${String(n)} must be below ${String(f.exclusiveMaximum)}.`;
  return null;
}

/** Parse "0.5, 1, 2" for a field: typed, in bounds, distinct, at most 12. */
export function parseValues(f: Field | undefined, text: string): ParsedValues {
  if (!f) return { ok: false, error: 'Pick a setting.' };
  const items = text
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean);
  if (!items.length) return { ok: false, error: 'List at least one value, comma separated.' };
  if (items.length > MAX_MEMBERS) return { ok: false, error: `At most ${MAX_MEMBERS} values.` };
  const out: SweepValue[] = [];
  for (const it of items) {
    if (f.type === 'boolean') {
      if (it !== 'true' && it !== 'false') return { ok: false, error: 'Use true or false.' };
      out.push(it === 'true');
    } else if (f.type === 'number' || f.type === 'integer') {
      const n = Number(it);
      if (!Number.isFinite(n)) return { ok: false, error: `${it} is not a number.` };
      if (f.type === 'integer' && !Number.isInteger(n))
        return { ok: false, error: `${it} is not a whole number.` };
      const b = inBounds(f, n);
      if (b) return { ok: false, error: b };
      out.push(n);
    } else {
      if (f.enum && !f.enum.includes(it))
        return { ok: false, error: `${it} is not one of ${f.enum.join(', ')}.` };
      out.push(it);
    }
  }
  if (new Set(out.map(String)).size !== out.length) return { ok: false, error: 'Each value once.' };
  return { ok: true, values: out };
}

export type GridCheck =
  | { ok: true; params: { key: string; values: SweepValue[] }[]; members: number }
  | { ok: false; error: string; members: number | null };

/** The grid's parameters and member count, or why it cannot be planned. */
export function checkGrid(
  drafts: readonly ParamDraft[],
  fields: readonly Field[],
  mode: 'grid' | 'zip',
): GridCheck {
  const byPath = new Map(fields.map((f) => [f.path, f]));
  const params: { key: string; values: SweepValue[] }[] = [];
  const keys = new Set<string>();
  for (const d of drafts) {
    if (!d.key) return { ok: false, error: 'Pick a setting for every row.', members: null };
    if (keys.has(d.key)) return { ok: false, error: `${d.key} is listed twice.`, members: null };
    keys.add(d.key);
    const p = parseValues(byPath.get(d.key), d.text);
    if (!p.ok) return { ok: false, error: `${d.key}: ${p.error}`, members: null };
    params.push({ key: d.key, values: p.values });
  }
  if (!params.length) return { ok: false, error: 'Add a setting to vary.', members: null };
  let members: number;
  if (mode === 'zip') {
    const n = params[0]?.values.length ?? 0;
    if (params.some((p) => p.values.length !== n))
      return {
        ok: false,
        error: 'Zip mode pairs the lists: give every row the same number of values.',
        members: null,
      };
    members = n;
  } else members = params.reduce((a, p) => a * p.values.length, 1);
  if (members > MAX_MEMBERS)
    return {
      ok: false,
      error: `${members} members: a sweep has at most ${MAX_MEMBERS}. Narrow a list or use zip.`,
      members,
    };
  return { ok: true, params, members };
}

export const METRICS = {
  sharpe_net: { label: 'Net Sharpe', kind: 'ratio' },
  sharpe_gross: { label: 'Gross Sharpe', kind: 'ratio' },
  dsr: { label: 'DSR', kind: 'prob' },
  cagr_net: { label: 'Net CAGR', kind: 'pct' },
  max_drawdown_net: { label: 'Max drawdown', kind: 'pct' },
  avg_daily_turnover: { label: 'Turnover / day', kind: 'pct' },
  avg_cost_bps_per_side: { label: 'Cost per side', kind: 'bps' },
  mean_ic: { label: 'Mean IC', kind: 'ic' },
} as const;

export type MetricKey = keyof typeof METRICS;

export function metricValue(m: SweepMember, k: MetricKey): number | null {
  const v = m.metrics[k];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export function valueText(v: unknown): string {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number' || typeof v === 'string') return String(v);
  return '';
}

/**
 * The metric against the first parameter's values, one line per value of the second
 * parameter (a zip or a one-parameter sweep has one line). Members that have not finished
 * leave a gap.
 */
export function metricLines(
  s: Pick<SweepManifest, 'params' | 'members' | 'mode'>,
  k: MetricKey,
): { labels: string[]; series: { label: string; values: (number | null)[] }[]; xKey: string } {
  const [x, by] = s.params;
  if (!x) return { labels: [], series: [], xKey: '' };
  const labels = x.values.map(valueText);
  const find = (xv: unknown, bv?: unknown) =>
    s.members.find(
      (m) =>
        valueText(m.overrides[x.key]) === valueText(xv) &&
        (by === undefined || bv === undefined || valueText(m.overrides[by.key]) === valueText(bv)),
    );
  if (!by || s.mode === 'zip')
    return { xKey: x.key, labels, series: [{ label: METRICS[k].label, values: x.values.map((v) => { const m = find(v); return m ? metricValue(m, k) : null; }) }] }; // prettier-ignore
  return {
    xKey: x.key,
    labels,
    series: by.values.map((bv) => ({
      label: `${by.key.split('.').pop() ?? by.key} = ${valueText(bv)}`,
      values: x.values.map((xv) => {
        const m = find(xv, bv);
        return m ? metricValue(m, k) : null;
      }),
    })),
  };
}

/** "trade_aversion = 0.5, risk_aversion = 2" */
export function overridesText(o: Record<string, unknown>): string {
  return Object.entries(o)
    .map(([k, v]) => `${k.split('.').pop() ?? k} = ${valueText(v)}`)
    .join(', ');
}

export const SWEEP_TONE: Record<string, BadgeTone> = {
  done: 'pass',
  partial: 'caution',
  running: 'accent',
  pending: 'neutral',
  failed: 'fail',
};

/** A sweep metric in its unit. */
export function formatMetric(k: MetricKey, v: number | null): string {
  switch (METRICS[k].kind) {
    case 'ratio':
      return sharpe(v, { signed: true });
    case 'prob':
      return num(v, { digits: 3 });
    case 'pct':
      return pct(v, { digits: 1 });
    case 'bps':
      return bps(v);
    case 'ic':
      return num(v, { digits: 4, signed: true });
  }
}
