import { describe, expect, it } from 'vitest';
import { mockSeal } from '@/mocks/fixtures/generate';
import type { RunSummary } from '@/shared/api';
import {
  clusterFamilies,
  commonLabel,
  configDiff,
  diffKeys,
  formatDiffRow,
  formatLeaf,
  ledgerFamilyOf,
  trialsFor,
} from './families';

const BASE = {
  'alpha.horizon_days': 20,
  'portfolio.trade_aversion': 1,
  'portfolio.book': 'long_short',
  'risk.n_factors': 12,
  'alpha.use_ml': false,
};

function run(
  id: string,
  label: string,
  created: string,
  leaves: Record<string, number | string | boolean | null>,
  extra: Partial<RunSummary> = {},
): RunSummary {
  return {
    id,
    created_at: created,
    label,
    ledger_family: null,
    kind: 'synthetic',
    phase: 1,
    use_ml: false,
    config_hash: id,
    data_hash: 'abcdef1234567890',
    git_sha: 'x',
    status: 'done',
    headline: null,
    config_leaves: { ...BASE, ...leaves },
    seal: mockSeal({}),
    ...extra,
  };
}

describe('config diff', () => {
  it('lists keys that differ, missing keys included', () => {
    expect(diffKeys(BASE, { ...BASE, 'portfolio.trade_aversion': 0 })).toEqual([
      'portfolio.trade_aversion',
    ]);
    expect(diffKeys(BASE, { ...BASE, extra: 1 })).toEqual(['extra']);
    expect(diffKeys(BASE, { ...BASE, 'risk.n_factors': 12 + 1e-15 })).toEqual([]);
  });

  it('builds rows with one value per run and formats them compactly', () => {
    const rows = configDiff([
      BASE,
      { ...BASE, 'portfolio.trade_aversion': 0 },
      { ...BASE, 'alpha.horizon_days': 10, 'portfolio.trade_aversion': 0.5 },
    ]);
    expect(rows.map((r) => r.key)).toEqual(['alpha.horizon_days', 'portfolio.trade_aversion']);
    expect(formatDiffRow(rows[1] as (typeof rows)[number])).toBe(
      'portfolio.trade_aversion: 1.0 | 0.0 | 0.5',
    );
    expect(formatDiffRow(rows[0] as (typeof rows)[number])).toBe(
      'alpha.horizon_days: 20 | 20 | 10',
    );
    expect(formatLeaf(undefined)).toBe('—');
    expect(formatLeaf(null)).toBe('null');
    expect(formatLeaf(true)).toBe('true');
  });
});

describe('families', () => {
  const a = run('a', 'Composite baseline', '2026-11-04T10:00:00Z', {});
  const b = run('b', 'Composite baseline, trade aversion 0', '2026-11-05T10:00:00Z', {
    'portfolio.trade_aversion': 0,
  });
  const c = run('c', 'Composite baseline, horizon 10', '2026-11-06T10:00:00Z', {
    'alpha.horizon_days': 10,
  });
  const ml = run(
    'd',
    'Composite + ML sleeve',
    '2026-11-06T12:00:00Z',
    {
      'alpha.use_ml': true,
      'alpha.lgbm.n_estimators': 400,
      'alpha.lgbm.learning_rate': 0.03,
      'alpha.lgbm.num_leaves': 15,
    },
    { use_ml: true, phase: 2 },
  );
  const running = run(
    'e',
    'Composite, 600 names',
    '2026-11-07T10:00:00Z',
    {},
    {
      status: 'running',
      config_leaves: {},
    },
  );

  it('clusters runs differing in ≤ 3 keys, within their ledger family', () => {
    const fams = clusterFamilies([a, b, c, ml, running]);
    expect(fams.map((f) => f.runs.map((r) => r.id))).toEqual([['e'], ['d'], ['c', 'b', 'a']]);
    const base = fams[2];
    expect(base?.label).toBe('Composite baseline');
    expect(base?.diffKeys).toEqual(['alpha.horizon_days', 'portfolio.trade_aversion']);
    expect(base?.ledgerFamily).toBe('pipeline_phase1:abcdef123456');
    expect(fams[1]?.ledgerFamily).toBe('pipeline_phase2:abcdef123456');
    expect(fams[0]?.label).toBe('Composite, 600 names');
  });

  it('chains through single linkage but splits on a fourth differing key', () => {
    const far = run('f', 'Composite baseline, retuned', '2026-11-08T10:00:00Z', {
      'alpha.horizon_days': 5,
      'portfolio.trade_aversion': 3,
      'risk.n_factors': 8,
      'portfolio.book': 'long_only',
    });
    const fams = clusterFamilies([a, b, c, far]);
    expect(fams.map((f) => f.runs.length)).toEqual([1, 3]);
  });

  it('derives labels and ledger families', () => {
    expect(commonLabel(['Composite baseline', 'Composite baseline, horizon 10'])).toBe(
      'Composite baseline',
    );
    expect(commonLabel(['Alpha run', 'Beta run'])).toBe('');
    expect(commonLabel(['Only one'])).toBe('Only one');
    expect(ledgerFamilyOf({ phase: 1, data_hash: '' })).toBe('legacy');
    // the manifest's own family (keyed by the market hash) wins over the data-hash form
    expect(
      ledgerFamilyOf({
        phase: 1,
        data_hash: 'abcdef0123456789',
        ledger_family: 'pipeline_phase1:mkt',
      }),
    ).toBe('pipeline_phase1:mkt');
    expect(
      trialsFor(a, [
        {
          family: 'pipeline_phase1:abcdef123456',
          n_trials: 4,
          effective_trials: 3,
          var_sr_annual: 0.2,
          n_not_counted: 0,
        },
      ])?.n_trials,
    ).toBe(4);
  });
});
