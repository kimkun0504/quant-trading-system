import { describe, expect, it } from 'vitest';
import type { AlphaIcByYear, AlphaRefits, RunIc } from '@/shared/api';
import {
  icByYearGrid,
  icWithRollingMean,
  normaliseColumns,
  refitGrid,
  sleeveLabel,
  storyLine,
} from './shape';

const byYear: AlphaIcByYear = {
  years: [2019, 2020, 2021, 2022],
  rows: [
    {
      sleeve: 'combined',
      cells: [
        { year: 2019, mean_ic: 0.04, t_stat: 2.1, n: 250 },
        { year: 2020, mean_ic: 0.02, t_stat: 1.0, n: 252 },
        { year: 2021, mean_ic: -0.01, t_stat: -0.5, n: 251 },
        { year: 2022, mean_ic: 0.03, t_stat: 1.4, n: 120 },
      ],
    },
    {
      sleeve: 'theme_value',
      cells: [
        { year: 2019, mean_ic: 0.05, t_stat: 2.5, n: 250 },
        { year: 2020, mean_ic: 0.04, t_stat: 2.0, n: 252 },
        { year: 2021, mean_ic: 0.0, t_stat: 0, n: 251 },
        { year: 2022, mean_ic: -0.01, t_stat: -0.4, n: 120 },
      ],
    },
    {
      sleeve: 'theme_quality',
      cells: [
        { year: 2019, mean_ic: 0.0, t_stat: 0, n: 250 },
        { year: 2020, mean_ic: 0.01, t_stat: 0.5, n: 252 },
        { year: 2021, mean_ic: 0.03, t_stat: 1.5, n: 251 },
        { year: 2022, mean_ic: 0.04, t_stat: 1.9, n: 120 },
      ],
    },
  ],
  source: 'daily_sleeves',
  overlap_days: 21,
};

describe('icByYearGrid', () => {
  it('bins sleeves × years with t-stats alongside and a shared |max|', () => {
    const g = icByYearGrid(byYear);
    expect(g.rows).toEqual(['Combined', 'Value', 'Quality']);
    expect(g.cols).toEqual(['2019', '2020', '2021', '2022']);
    expect(g.values[1]).toEqual([0.05, 0.04, 0, -0.01]);
    expect(g.t?.[0]?.[2]).toBe(-0.5);
    expect(g.maxAbs).toBeCloseTo(0.05);
  });

  it('leaves a missing year null', () => {
    const g = icByYearGrid({ ...byYear, years: [2018, 2019] });
    expect(g.values[0]).toEqual([null, 0.04]);
  });

  it('tells the story: value faded, quality rose', () => {
    const s = storyLine(byYear);
    expect(s).toContain('value moved most against');
    expect(s).toContain('quality most in its favour');
  });
});

describe('normaliseColumns', () => {
  it('scales every refit to sum 1 and keeps nulls', () => {
    const out = normaliseColumns([
      [2, 1, null],
      [2, 3, null],
      [null, 0, null],
    ]);
    expect(out[0]).toEqual([0.5, 0.25, null]);
    expect(out[1]).toEqual([0.5, 0.75, null]);
    expect(out[2]).toEqual([null, 0, null]);
  });

  it('clamps negative importance to zero before normalising', () => {
    expect(normaliseColumns([[-1], [1]])).toEqual([[0], [1]]);
  });

  it('gives zeros, not NaN, for an all-zero column', () => {
    expect(normaliseColumns([[0], [0]])).toEqual([[0], [0]]);
  });
});

describe('refitGrid', () => {
  const refits: AlphaRefits = {
    refits: [
      {
        refit_date: '2020-01-02',
        train_start: '2017-01-02',
        train_end: '2019-12-02',
        valid_start: '2019-01-02',
        pred_end: '2020-12-31',
        n_train: 1,
        n_valid: 1,
        blend_weights: {},
      },
      {
        refit_date: '2021-01-04',
        train_start: '2017-01-02',
        train_end: '2020-12-02',
        valid_start: '2020-01-02',
        pred_end: '2021-12-31',
        n_train: 1,
        n_valid: 1,
        blend_weights: {},
      },
    ],
    sleeves: ['lgbm', 'ridge'],
    features: ['a', 'b', 'c'],
    importance: {
      lgbm: { a: [1, 3], b: [3, 1], c: [0, 0] },
      ridge: { a: [0.1, 0.1], b: [0.3, 0.1] },
    },
    feature_ic: [],
    embargo_days: 21,
    refit_frequency_days: 252,
    notes: [],
    recorded: true,
  };

  it('normalises per refit and orders features by mean share', () => {
    const g = refitGrid(refits, 'lgbm');
    expect(g).not.toBeNull();
    expect(g?.cols).toEqual(['2020-01-02', '2021-01-04']);
    // a and b both average 0.5; c is 0 and goes last
    expect(g?.rows.slice(2)).toEqual(['c']);
    const a = g?.values[g.rows.indexOf('a')];
    expect(a).toEqual([0.25, 0.75]);
    expect(g?.values.map((r) => r[0]).reduce((x, y) => (x ?? 0) + (y ?? 0), 0)).toBeCloseTo(1);
  });

  it('only lists the features the sleeve recorded', () => {
    const g = refitGrid(refits, 'ridge');
    expect(g?.rows).toEqual(['b', 'a']);
    const b = g?.values[g.rows.indexOf('b')] ?? [];
    expect(b[0]).toBeCloseTo(0.75);
    expect(b[1]).toBeCloseTo(0.5);
  });

  it('is null for an unknown sleeve or without refits', () => {
    expect(refitGrid(refits, 'mlp')).toBeNull();
    expect(refitGrid({ ...refits, refits: [] }, 'lgbm')).toBeNull();
  });
});

describe('icWithRollingMean', () => {
  it('uses a one-year window in sampled points', () => {
    const dates = Array.from(
      { length: 30 },
      (_, i) => `2020-${String(1 + (i % 12)).padStart(2, '0')}-01`,
    );
    const ic: RunIc = {
      ic: { dates, series: { rank_ic: dates.map((_, i) => (i % 2 ? 0.1 : -0.1)) } },
      decay: [],
      horizon_days: 20,
    };
    const r = icWithRollingMean(ic);
    expect(r.window).toBe(13); // 252 / 20 ≈ 12.6
    expect(r.rolling.slice(0, 6).every((v) => v === null)).toBe(true);
    expect(r.rolling[29]).toBeCloseTo(0.1 / 13, 6);
  });
});

describe('sleeveLabel', () => {
  it('names sleeves for people', () => {
    expect(sleeveLabel('theme_low_risk')).toBe('Low risk');
    expect(sleeveLabel('lgbm')).toBe('LightGBM (gain)');
    expect(sleeveLabel('combined')).toBe('Combined');
  });
});
