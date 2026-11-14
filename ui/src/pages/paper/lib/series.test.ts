import { describe, expect, it } from 'vitest';
import { isLevel, liveSpans, modeCounts } from './series';

describe('paper series helpers', () => {
  it('finds the runs of live days', () => {
    expect(liveSpans(['backfill', 'backfill', 'live', 'live', 'backfill', 'live'])).toEqual([
      { from: 2, to: 3 },
      { from: 5, to: 5 },
    ]);
    expect(liveSpans(['backfill', null])).toEqual([]);
  });

  it('counts live and backfill days', () => {
    expect(modeCounts(['live', 'backfill', 'live', null])).toEqual({ live: 2, backfill: 1 });
  });

  it('knows the three levels', () => {
    expect(['OK', 'WATCH', 'STOP', 'ok', null].map(isLevel)).toEqual([
      true,
      true,
      true,
      false,
      false,
    ]);
  });
});
