import { describe, expect, it } from 'vitest';
import type { Field } from '@/features/run-builder/lib/fields';
import type { SweepManifest } from '@/shared/api';
import {
  checkGrid,
  formatMetric,
  metricLines,
  overridesText,
  parseValues,
  sweepableFields,
} from './grid';

const f = (p: Partial<Field> & Pick<Field, 'path' | 'type'>): Field => ({
  name: p.path.split('.').pop() ?? p.path,
  title: p.path,
  section: 'portfolio',
  sectionTitle: 'Portfolio',
  nullable: false,
  ...p,
});

const FIELDS = [
  f({ path: 'portfolio.trade_aversion', type: 'number', minimum: 0, maximum: 10 }),
  f({ path: 'alpha.horizon_days', type: 'integer', minimum: 1, maximum: 252 }),
  f({ path: 'portfolio.book', type: 'string', enum: ['long_short', 'long_only'] }),
  f({ path: 'alpha.neutralize_score', type: 'boolean' }),
  f({ path: 'universe.refresh_anchor', type: 'string' }),
  f({ path: 'alpha.disabled_features', type: 'array' }),
];

describe('sweep grid', () => {
  it('offers numbers, enums and switches only', () => {
    expect(sweepableFields(FIELDS).map((x) => x.path)).toEqual([
      'portfolio.trade_aversion',
      'alpha.horizon_days',
      'portfolio.book',
      'alpha.neutralize_score',
    ]);
  });

  it('parses values by type and bounds', () => {
    expect(parseValues(FIELDS[0], '0.5, 1, 2')).toEqual({ ok: true, values: [0.5, 1, 2] });
    expect(parseValues(FIELDS[0], '11')).toMatchObject({
      ok: false,
      error: /above the maximum 10/,
    });
    expect(parseValues(FIELDS[1], '20, 2.5')).toMatchObject({
      ok: false,
      error: /not a whole number/,
    });
    expect(parseValues(FIELDS[2], 'long_only, short')).toMatchObject({
      ok: false,
      error: /not one of/,
    });
    expect(parseValues(FIELDS[3], 'true,false')).toEqual({ ok: true, values: [true, false] });
    expect(parseValues(FIELDS[0], '1, 1')).toMatchObject({ ok: false, error: /once/ });
    expect(parseValues(undefined, '1')).toMatchObject({ ok: false });
  });

  it('counts members: grid multiplies, zip pairs, at most 12', () => {
    const g = checkGrid(
      [
        { key: 'portfolio.trade_aversion', text: '0.5,1,2' },
        { key: 'alpha.horizon_days', text: '20,60' },
      ],
      FIELDS,
      'grid',
    );
    expect(g).toMatchObject({ ok: true, members: 6 });
    expect(checkGrid([{ key: 'portfolio.trade_aversion', text: '1,2' }, { key: 'alpha.horizon_days', text: '20' }], FIELDS, 'zip')).toMatchObject({ ok: false, error: /same number/ }); // prettier-ignore
    const big = checkGrid(
      [
        { key: 'portfolio.trade_aversion', text: '0.5,1,2,3' },
        { key: 'alpha.horizon_days', text: '5,20,60,120' },
      ],
      FIELDS,
      'grid',
    );
    expect(big).toMatchObject({ ok: false, members: 16, error: /at most 12/ });
    expect(checkGrid([{ key: '', text: '' }], FIELDS, 'grid')).toMatchObject({ ok: false });
    expect(checkGrid([{ key: 'portfolio.book', text: 'long_only' }, { key: 'portfolio.book', text: 'long_short' }], FIELDS, 'grid')).toMatchObject({ ok: false, error: /twice/ }); // prettier-ignore
  });

  it('lays the metric out by the first parameter, a line per value of the second', () => {
    const m = (i: number, a: number, b: number, s: number | null) => ({
      index: i,
      overrides: { 'p.a': a, 'p.b': b },
      config_hash: String(i),
      status: s === null ? 'pending' : 'done',
      run_id: null,
      run_dir: null,
      reference: false,
      error: null,
      metrics: s === null ? {} : { sharpe_net: s },
      started_at: null,
      finished_at: null,
    });
    const s = {
      mode: 'grid',
      params: [
        { key: 'p.a', values: [1, 2] },
        { key: 'p.b', values: [10, 20] },
      ],
      members: [m(0, 1, 10, 0.5), m(1, 1, 20, 0.6), m(2, 2, 10, 0.7), m(3, 2, 20, null)],
    } as unknown as SweepManifest;
    const out = metricLines(s, 'sharpe_net');
    expect(out.labels).toEqual(['1', '2']);
    expect(out.series).toEqual([
      { label: 'b = 10', values: [0.5, 0.7] },
      { label: 'b = 20', values: [0.6, null] },
    ]);
    expect(overridesText({ 'p.a': 1, 'p.b': true })).toBe('a = 1, b = true');
    expect(formatMetric('avg_cost_bps_per_side', 10.37)).toBe('10.4 bp');
  });
});
