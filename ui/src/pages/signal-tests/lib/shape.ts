import type { IcStats, SignalFamilyNow, SignalTestResult } from '@/shared/api';
import { num } from '@/shared/lib/format';

/** One row of the IC-by-horizon table: the signal's IC and the incremental IC. */
export interface HorizonRow {
  horizon: number;
  primary: boolean;
  ic: IcStats | undefined;
  incremental: IcStats | undefined;
}

const byNumber = (a: string, b: string) => Number(a) - Number(b);

export function horizonRows(r: SignalTestResult): HorizonRow[] {
  return Object.keys(r.ic_by_horizon)
    .sort(byNumber)
    .map((h) => ({
      horizon: Number(h),
      primary: Number(h) === r.primary_horizon,
      ic: r.ic_by_horizon[h],
      incremental: r.incremental.ic_by_horizon[h],
    }));
}

/** Bars of a `{year: {mean_ic}}` map in year order (years without a value dropped). */
export function yearBars(r: SignalTestResult): { label: string; value: number }[] {
  return Object.entries(r.ic_by_year)
    .sort(([a], [b]) => byNumber(a, b))
    .flatMap(([y, v]) => (v.mean_ic === null ? [] : [{ label: y, value: v.mean_ic }]));
}

export function foldBars(r: SignalTestResult): { label: string; value: number }[] {
  return r.ic_by_fold.flatMap((f) =>
    f.mean_ic === null
      ? []
      : [{ label: `${f.start.slice(0, 7)}–${f.end.slice(0, 7)}`, value: f.mean_ic }],
  );
}

export function coverageBars(r: SignalTestResult): { label: string; value: number }[] {
  return Object.entries(r.coverage_by_year)
    .sort(([a], [b]) => byNumber(a, b))
    .flatMap(([y, v]) => (v === null ? [] : [{ label: y, value: v }]));
}

/** The decay curve: IC of the signal lagged k days against the primary label. */
export function decayPoints(r: SignalTestResult): { labels: string[]; values: (number | null)[] } {
  const ks = Object.keys(r.decay.lag_ic).sort(byNumber);
  return { labels: ks, values: ks.map((k) => r.decay.lag_ic[k] ?? null) };
}

/** The strongest correlations with the baseline's features (by |rho|), the composite first. */
export function topCorrelations(r: SignalTestResult, n = 8): { name: string; rho: number }[] {
  return Object.entries(r.correlations)
    .flatMap(([name, rho]) => (rho === null ? [] : [{ name, rho }]))
    .sort((a, b) => Math.abs(b.rho) - Math.abs(a.rho))
    .slice(0, n);
}

/** Where the primary t stands against the hurdle, in words that never omit the family size. */
export function verdict(
  r: SignalTestResult,
  now: SignalFamilyNow,
): { tone: 'pass' | 'fail' | 'caution'; text: string } {
  const t = r.trial.t;
  const n = now.n_tests;
  const tried = `${n} signal${n === 1 ? '' : 's'} tested in this family`;
  if (t === null)
    return {
      tone: 'caution',
      text: `The primary t could not be computed (too few dates). ${tried}.`,
    };
  const holm = now.p_holm ?? r.trial.p_holm;
  const passT = Math.abs(t) >= r.trial.hurdle;
  const passHolm = holm !== null && holm < 0.05;
  if (passT && passHolm)
    return {
      tone: 'pass',
      text: `Clears the |t| ≥ ${num(r.trial.hurdle, { digits: 1 })} hurdle (t ${num(t, { signed: true })}) and survives Holm over ${tried} (p ${num(holm, { digits: 3 })}).`,
    };
  if (passT)
    return {
      tone: 'caution',
      text: `t ${num(t, { signed: true })} clears the hurdle, but after Holm over ${tried} the p is ${num(holm, { digits: 3 })}: not significant once the other tests are counted.`,
    };
  return {
    tone: 'fail',
    text: `t ${num(t, { signed: true })} is below the |t| ≥ ${num(r.trial.hurdle, { digits: 1 })} hurdle; ${tried}, Holm p ${num(holm, { digits: 3 })}. Not evidence of a new signal.`,
  };
}

/** Parse "5, 20,60" into sorted distinct horizons; an error message when invalid. */
export function parseHorizons(
  text: string,
): { ok: true; value: number[] } | { ok: false; error: string } {
  const parts = text.split(/[,\s]+/).filter(Boolean);
  const out = parts.map(Number);
  if (!parts.length) return { ok: false, error: 'Give at least one horizon, e.g. 5,20,60.' };
  if (out.some((h) => !Number.isInteger(h) || h < 1 || h > 252))
    return { ok: false, error: 'Horizons are whole numbers of trading days from 1 to 252.' };
  if (new Set(out).size !== out.length) return { ok: false, error: 'A horizon is listed twice.' };
  if (out.length > 8) return { ok: false, error: 'At most 8 horizons.' };
  return { ok: true, value: [...out].sort((a, b) => a - b) };
}
