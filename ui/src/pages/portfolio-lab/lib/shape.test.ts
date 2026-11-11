import { describe, expect, it } from 'vitest';
import type { Holding, RunHoldings } from '@/shared/api';
import {
  holdingPeriodSeries,
  industryStack,
  realisedVsExante,
  snapshotTotals,
  stackDates,
  topPositions,
} from './shape';

const h = (ticker: string, weight: number, industry: string | null): Holding => ({
  ticker,
  weight,
  alpha: null,
  specific_vol: null,
  industry,
});

describe('industryStack', () => {
  it('stacks long weight up and short weight down per industry', () => {
    const snaps: RunHoldings[] = [
      {
        date: '2020-01-31',
        available_dates: [],
        positions: [h('A', 0.02, 'Banks'), h('B', -0.01, 'Banks'), h('C', 0.03, 'Software')],
      },
      {
        date: '2020-02-28',
        available_dates: [],
        positions: [h('A', 0.01, 'Banks'), h('D', -0.02, 'Software'), h('E', 0.01, null)],
      },
    ];
    const s = industryStack(snaps);
    expect(s.dates).toEqual(['2020-01-31', '2020-02-28']);
    expect(s.industries).toEqual(['Software', 'Banks', 'Other']);
    expect(s.long.Banks).toEqual([0.02, 0.01]);
    expect(s.short.Banks).toEqual([-0.01, 0]);
    expect(s.short.Software).toEqual([0, -0.02]);
    expect(s.long.Other).toEqual([0, 0.01]);
  });

  it('folds the tail beyond the largest seven industries into Other', () => {
    const positions = Array.from({ length: 9 }, (_, i) => h(`T${i}`, 0.01 * (9 - i), `Ind${i}`));
    const s = industryStack([{ date: '2020-01-31', available_dates: [], positions }]);
    expect(s.industries).toHaveLength(8);
    expect(s.industries[7]).toBe('Other');
    expect(s.long.Other?.[0]).toBeCloseTo(0.02 + 0.01);
    const total = s.industries.reduce((a, k) => a + (s.long[k]?.[0] ?? 0), 0);
    expect(total).toBeCloseTo(0.45);
  });
});

describe('stackDates', () => {
  it('thins to at most the cap and keeps the last date', () => {
    const dates = Array.from({ length: 60 }, (_, i) => `d${i}`);
    const out = stackDates(dates, 10);
    expect(out).toHaveLength(10);
    expect(out[0]).toBe('d0');
    expect(out[9]).toBe('d59');
    expect(stackDates(['a', 'b'], 10)).toEqual(['a', 'b']);
  });
});

describe('realisedVsExante', () => {
  it('annualises the trailing sd of net returns and passes the forecast through', () => {
    const net = Array.from({ length: 100 }, (_, i) => (i % 2 ? 0.01 : -0.01));
    const r = realisedVsExante(
      net,
      net.map(() => 0.1),
      63,
    );
    expect(r.realised.slice(0, 49).every((v) => v === null)).toBe(true);
    const last = r.realised[99];
    expect(last).not.toBeNull();
    expect(last ?? 0).toBeCloseTo(0.01 * Math.sqrt(63 / 62) * Math.sqrt(252), 3);
    expect(r.exante[0]).toBe(0.1);
  });
});

describe('holdingPeriodSeries', () => {
  it('is gross over one-sided turnover and null while the book is empty', () => {
    const turnover = Array.from({ length: 80 }, () => 0.04);
    const gross = Array.from({ length: 80 }, (_, i) => (i < 20 ? 0 : 2));
    const hp = holdingPeriodSeries(turnover, gross, 20);
    expect(hp[9]).toBeNull();
    expect(hp[79]).toBeCloseTo(2 / 0.02);
  });
});

describe('topPositions and snapshotTotals', () => {
  const positions = [
    h('A', 0.03, null),
    h('B', -0.04, null),
    h('C', 0.01, null),
    h('D', -0.01, null),
  ];
  it('ranks longs by weight and shorts by most negative', () => {
    const t = topPositions(positions, 1);
    expect(t.longs.map((p) => p.ticker)).toEqual(['A']);
    expect(t.shorts.map((p) => p.ticker)).toEqual(['B']);
  });
  it('sums gross and net', () => {
    const s = snapshotTotals(positions);
    expect(s.n).toBe(4);
    expect(s.nLong).toBe(2);
    expect(s.nShort).toBe(2);
    expect(s.gross).toBeCloseTo(0.09);
    expect(s.net).toBeCloseTo(-0.01);
    expect(s.maxAbs).toBe(0.04);
  });
});
