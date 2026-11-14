import { describe, expect, it } from 'vitest';
import { CONFIG_SCHEMA } from '@/mocks/fixtures/configSchema';
import { cliCommand, cliText } from './cli';
import { DEFAULT_RUN, defaultBase, diffRows, runErrors, toRequest } from './draft';
import type { Base, Draft } from './draft';
import { schemaFields } from './fields';

const FIELDS = schemaFields(CONFIG_SCHEMA);
const storeBase = (): Base => ({ ...defaultBase(), run: { ...DEFAULT_RUN, kind: 'store' } });
const draft = (run: Draft['run']): Draft => ({ id: 'a', values: {}, run });

describe('store prefilters in the run builder (qts run --max-rank / --allow-partial-coverage)', () => {
  it('sends the screened-rank prefilter and the partial-coverage flag, and prints them', () => {
    const base = storeBase();
    const req = toRequest(
      draft({ start: '2000-01-01', maxRank: '1500', allowPartialCoverage: true }),
      base,
      FIELDS,
    );
    expect(req).toMatchObject({
      kind: 'store',
      start: '2000-01-01',
      max_rank: 1500,
      allow_partial_coverage: true,
    });
    expect(cliText(cliCommand(req, '/data/store'))).toBe(
      'qts run --store /data/store --start 2000-01-01 --max-rank 1500 --allow-partial-coverage',
    );
    const rows = diffRows(draft({ maxRank: '2000' }), base, FIELDS);
    expect(rows.map((r) => r.key)).toEqual(['data.max_rank']);
  });

  it('refuses the rank with the peak-cap prefilters; synthetic runs carry neither', () => {
    const store = { ...DEFAULT_RUN, kind: 'store' as const };
    expect(runErrors({ ...store, maxRank: '1500', maxNames: '3000' }).maxRank).toMatch(
      /cannot be combined/,
    );
    expect(runErrors({ ...store, maxRank: '5' }).maxRank).toMatch(/between 10/);
    expect(runErrors({ ...store, maxRank: '1500' })).toEqual({});
    const syn = toRequest(
      draft({ maxRank: '1500', allowPartialCoverage: true }),
      defaultBase(),
      FIELDS,
    );
    expect(syn.max_rank).toBeUndefined();
    expect(syn.allow_partial_coverage).toBeUndefined();
  });
});
