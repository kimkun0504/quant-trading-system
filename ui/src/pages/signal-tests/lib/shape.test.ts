import { describe, expect, it } from 'vitest';
import type { SignalTestDetail } from '@/shared/api';
import detailJson from '@/mocks/fixtures/research/signal_test.json';
import {
  decayPoints,
  foldBars,
  horizonRows,
  parseHorizons,
  topCorrelations,
  verdict,
  yearBars,
} from './shape';

const D = detailJson as unknown as SignalTestDetail;

describe('signal test shaping', () => {
  it('orders horizons numerically and marks the primary', () => {
    const rows = horizonRows(D.result);
    expect(rows.map((r) => r.horizon)).toEqual(
      [...rows.map((r) => r.horizon)].sort((a, b) => a - b),
    );
    expect(rows.filter((r) => r.primary)).toHaveLength(1);
    expect(rows.find((r) => r.primary)?.horizon).toBe(D.result.primary_horizon);
  });

  it('years, folds, decay and correlations', () => {
    expect(yearBars(D.result).map((b) => b.label)).toEqual(
      [...yearBars(D.result).map((b) => b.label)].sort(),
    );
    expect(foldBars(D.result).length).toBeGreaterThan(0);
    const d = decayPoints(D.result);
    expect(d.labels.map(Number)).toEqual([...d.labels.map(Number)].sort((a, b) => a - b));
    const c = topCorrelations(D.result, 3);
    expect(c).toHaveLength(3);
    expect(Math.abs(c[0]?.rho ?? 0)).toBeGreaterThanOrEqual(Math.abs(c[2]?.rho ?? 0));
  });

  it('never states a verdict without the family size', () => {
    const v = verdict(D.result, D.family_now);
    expect(v.text).toMatch(new RegExp(`${D.family_now.n_tests} signals? tested in this family`));
    const strong = verdict(
      { ...D.result, trial: { ...D.result.trial, t: 4.2 } },
      { ...D.family_now, p_holm: 0.001 },
    );
    expect(strong.tone).toBe('pass');
    const lucky = verdict(
      { ...D.result, trial: { ...D.result.trial, t: 3.1 } },
      { ...D.family_now, p_holm: 0.2 },
    );
    expect(lucky.tone).toBe('caution');
    expect(lucky.text).toMatch(/not significant once the other tests are counted/);
  });

  it('parses horizons', () => {
    expect(parseHorizons('60, 5,20')).toEqual({ ok: true, value: [5, 20, 60] });
    expect(parseHorizons('5,5')).toMatchObject({ ok: false });
    expect(parseHorizons('0')).toMatchObject({ ok: false });
    expect(parseHorizons('')).toMatchObject({ ok: false });
  });
});
