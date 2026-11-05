import type { LaunchRunRequest, Phase, RunKind } from '@/shared/api';

/** Form state: numbers stay strings until validated so partial input is not lost. */
export interface LaunchFormValues {
  kind: RunKind;
  phase: Phase;
  ml: boolean;
  stocks: string;
  days: string;
  seed: string;
  signal_lag_days: '0' | '1';
  overrides: Record<string, string | boolean>;
}

export const WARMUP_DAYS = 512;

export const DEFAULT_VALUES: LaunchFormValues = {
  kind: 'synthetic',
  phase: 1,
  ml: false,
  stocks: '300',
  days: '1512',
  seed: '0',
  signal_lag_days: '1',
  overrides: {},
};

export type FieldName = 'stocks' | 'days' | 'seed' | 'ml' | `override:${string}`;
export type LaunchErrors = Partial<Record<FieldName, string>>;

export interface OverrideBounds {
  path: string;
  title: string;
  type: 'number' | 'integer' | 'boolean' | 'string';
  minimum?: number;
  maximum?: number;
}

function intInRange(raw: string, min: number, max: number, label: string): string | undefined {
  if (raw.trim() === '') return `Enter ${label}.`;
  const n = Number(raw);
  if (!Number.isInteger(n))
    return `${label.charAt(0).toUpperCase()}${label.slice(1)} must be a whole number.`;
  if (n < min || n > max)
    return `Use between ${min.toLocaleString('en-US')} and ${max.toLocaleString('en-US')}.`;
  return undefined;
}

export function validateLaunch(
  v: LaunchFormValues,
  overrides: readonly OverrideBounds[] = [],
): LaunchErrors {
  const e: LaunchErrors = {};
  const stocks = intInRange(v.stocks, 50, 3000, 'a number of stocks');
  if (stocks) e.stocks = stocks;
  const minDays = WARMUP_DAYS + 252;
  const days = intInRange(v.days, minDays, 6000, 'a number of days');
  if (days)
    e.days = days.startsWith('Use between')
      ? `Use at least ${minDays} (${WARMUP_DAYS} warm-up days plus one year) and at most 6,000.`
      : days;
  const seed = intInRange(v.seed, 0, 2 ** 31 - 1, 'a seed');
  if (seed) e.seed = seed;
  if (v.phase === 2 && !v.ml) e.ml = 'Phase 2 tests the ML sleeve: turn it on, or choose phase 1.';
  for (const o of overrides) {
    const raw = v.overrides[o.path];
    if (raw === undefined || raw === '' || typeof raw === 'boolean') continue;
    const n = Number(raw);
    const key: FieldName = `override:${o.path}`;
    if (!Number.isFinite(n)) e[key] = `${o.title} must be a number.`;
    else if (o.type === 'integer' && !Number.isInteger(n))
      e[key] = `${o.title} must be a whole number.`;
    else if (o.minimum !== undefined && n < o.minimum)
      e[key] = `${o.title} must be at least ${o.minimum}.`;
    else if (o.maximum !== undefined && n > o.maximum)
      e[key] = `${o.title} must be at most ${o.maximum}.`;
  }
  return e;
}

export function toRequest(v: LaunchFormValues): LaunchRunRequest {
  const config_overrides: Record<string, number | boolean | string> = {};
  for (const [k, raw] of Object.entries(v.overrides)) {
    if (raw === '') continue;
    config_overrides[k] = typeof raw === 'boolean' ? raw : Number(raw);
  }
  return {
    kind: v.kind,
    ml: v.ml,
    phase: v.phase,
    stocks: Number(v.stocks),
    days: Number(v.days),
    seed: Number(v.seed),
    signal_lag_days: v.signal_lag_days === '0' ? 0 : 1,
    config_overrides,
  };
}
