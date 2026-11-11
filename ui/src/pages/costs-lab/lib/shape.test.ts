import { describe, expect, it } from 'vitest';
import type { RunConstraints, RunCosts } from '@/shared/api';
import {
  bindingGrid,
  costDistribution,
  costPerSideDaily,
  participationUsage,
  periodKeys,
  volReasonShares,
} from './shape';

const constraints: RunConstraints = {
  dates: ['2020-01-02', '2020-01-03', '2020-02-03', '2020-02-04'],
  names: ['gross', 'participation'],
  binding: { gross: [1, 0, 1, 1], participation: [3, 0, 0, 2] },
  status: ['optimal', 'optimal', 'no_alpha', 'optimal'],
  vol_target_binding: ['target', 'caps', 'no_alpha', 'solve_failed:infeasible'],
  transfer_coefficient: [0.6, 0.7, null, 0.5],
  n_traded: [10, 10, 0, 4],
  n_solves: [3, 3, 0, 2],
  share_binding: { gross: 0.75, participation: 0.5 },
  vol_target_reasons: ['caps', 'no_alpha', 'solve_failed:infeasible', 'target'],
  n_rebalances: 4,
};

describe('bindingGrid', () => {
  it('bins the share of solved rebalances where each constraint bound, by month', () => {
    const g = bindingGrid(constraints);
    expect(g.grain).toBe('month');
    expect(g.cols).toEqual(['2020-01', '2020-02']);
    expect(g.rowKeys).toEqual(['gross', 'participation']); // by whole-run share
    expect(g.rows).toEqual(['Gross leverage', 'ADV participation']);
    expect(g.values[0]).toEqual([0.5, 1]); // Feb: the no_alpha day is not a solve
    expect(g.values[1]).toEqual([0.5, 1]);
  });

  it('switches to quarters for long runs', () => {
    const dates = Array.from(
      { length: 130 },
      (_, i) => `${2000 + Math.floor(i / 12)}-${String((i % 12) + 1).padStart(2, '0')}-15`,
    );
    expect(periodKeys(dates).grain).toBe('quarter');
    expect(periodKeys(dates).keys[0]).toBe('2000 Q1');
    expect(periodKeys(dates).keys[4]).toBe('2000 Q2');
  });
});

describe('volReasonShares', () => {
  it('gives period shares that sum to one, folding solve_failed:* together', () => {
    const v = volReasonShares(constraints);
    expect(v.reasons).toEqual(['target', 'caps', 'no_alpha', 'solve_failed']);
    expect(v.dates).toEqual(['2020-01-02', '2020-02-03']);
    expect(v.series.target).toEqual([0.5, 0]);
    expect(v.series.caps).toEqual([0.5, 0]);
    expect(v.series.no_alpha).toEqual([0, 0.5]);
    expect(v.series.solve_failed).toEqual([0, 0.5]);
  });
});

describe('participationUsage', () => {
  it('is the mean share of traded names at the cap per period', () => {
    const u = participationUsage(constraints);
    expect(u.share[0]).toBeCloseTo((0.3 + 0) / 2);
    expect(u.share[1]).toBeCloseTo(0.5); // the empty day has no traded names and is skipped
  });
  it('is empty without a participation constraint', () => {
    expect(participationUsage({ ...constraints, binding: { gross: [1, 0, 1, 1] } }).dates).toEqual(
      [],
    );
  });
});

describe('cost per side', () => {
  const costs: RunCosts = {
    daily: {
      dates: ['2020-01-02', '2020-01-03'],
      series: { spread: [100, 50], impact: [50, 50], fees: [10, 0], borrow: [999, 999] },
    },
    nav: [1_000_000, 1_000_000],
    cost_sensitivity: [],
  };
  it('divides trading cost by dollars traded, leaving borrow out', () => {
    const ps = costPerSideDaily(costs, [0.04, 0], ['2020-01-02', '2020-01-03']);
    expect(ps[0]).toBeCloseTo((160 / 40_000) * 1e4);
    expect(ps[1]).toBeNull();
  });
  it('bins the distribution with its quantiles', () => {
    const d = costDistribution([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, null], 5);
    expect(d.n).toBe(10);
    expect(d.median).toBeCloseTo(5.5);
    expect(d.bins).toHaveLength(5);
    expect(d.bins.reduce((a, b) => a + b.count, 0)).toBe(10);
  });
});
