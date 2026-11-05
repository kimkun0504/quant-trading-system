import { describe, expect, it } from 'vitest';
import type { RunCosts } from '@/shared/api';
import {
  costShares,
  dailyCostBars,
  monthlyCostBars,
  monthlyReturns,
  rollingSharpe,
  runDescriptor,
} from './selectors';

describe('run selectors', () => {
  it('compounds daily returns into months and years', () => {
    const m = monthlyReturns(
      ['2020-01-30', '2020-01-31', '2020-02-03', '2021-01-04'],
      [0.1, 0.1, -0.05, null],
    );
    expect(m.years).toEqual([2020]);
    expect(m.cells.find((c) => c.month === 0)?.ret).toBeCloseTo(0.21);
    expect(m.annual.get(2020)).toBeCloseTo(1.21 * 0.95 - 1);
  });

  it('rolling Sharpe is null until the window fills', () => {
    const r = rollingSharpe([0.01, 0.02, 0.0, 0.01], 3);
    expect(r[0]).toBeNull();
    expect(r[1]).toBeNull();
    expect(r[2]).toBeCloseTo((0.01 / 0.01) * Math.sqrt(252));
    expect(r[3]).not.toBeNull();
  });

  const costs: RunCosts = {
    daily: {
      dates: ['2020-01-30', '2020-01-31', '2020-02-03'],
      series: {
        spread: [100, 100, 50],
        impact: [50, 50, 50],
        fees: [10, 10, 10],
        borrow: [40, 40, 40],
      },
    },
    nav: [1_000_000, 1_000_000, 1_000_000],
    cost_sensitivity: [],
  };

  it('converts costs to bp of NAV and sums by month', () => {
    expect(dailyCostBars(costs)[0]?.values.spread).toBeCloseTo(1);
    const m = monthlyCostBars(costs);
    expect(m).toHaveLength(2);
    expect(m[0]?.values.impact).toBeCloseTo(1);
    expect(m[1]?.label).toBe('2020-02');
  });

  it('cost shares sum to one', () => {
    const s = costShares(costs);
    expect(s.spread + s.impact + s.fees + s.borrow).toBeCloseTo(1);
  });

  it('describes runs in words', () => {
    expect(runDescriptor({ phase: 2, use_ml: true, kind: 'store' })).toBe(
      'Phase 2, composite + ML, data store',
    );
  });
});
