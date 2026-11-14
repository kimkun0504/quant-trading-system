import { describe, expect, it } from 'vitest';
import { MIDCAP_SLICE, REFERENCE_SLICE, parseNumberList, sliceErrors, toSlice } from './store';

describe('store slice drafts', () => {
  it('the reference load flags are valid and convert to the request', () => {
    expect(sliceErrors(REFERENCE_SLICE)).toEqual({});
    expect(toSlice(REFERENCE_SLICE)).toEqual({
      start: '2000-01-01',
      end: null,
      max_names: null,
      min_mcap_ever: null,
      max_rank: 1500,
    });
    expect(toSlice(MIDCAP_SLICE).max_rank).toBe(2750);
  });

  it('flags bad dates, bounds and the rank prefilter combined with the others', () => {
    const e = sliceErrors({
      start: '2020-1-1',
      end: '',
      maxNames: '5',
      minMcapEver: '-1',
      maxRank: '',
    });
    expect(e.start).toMatch(/YYYY-MM-DD/);
    expect(e.maxNames).toMatch(/10 to 20,000/);
    expect(e.minMcapEver).toMatch(/0 or more/);
    expect(sliceErrors({ ...REFERENCE_SLICE, start: '2020-01-02', end: '2020-01-01' }).end).toMatch(
      /before the start/,
    );
    expect(sliceErrors({ ...REFERENCE_SLICE, maxNames: '3000' }).maxRank).toMatch(
      /cannot be combined/,
    );
  });

  it('parses number lists', () => {
    expect(parseNumberList('5e6, 5e7,2.5e8 1e9')).toEqual([5e6, 5e7, 2.5e8, 1e9]);
    expect(parseNumberList('1, x')).toBeNull();
    expect(parseNumberList('  ')).toBeNull();
  });
});
