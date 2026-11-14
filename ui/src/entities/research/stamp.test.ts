import { describe, expect, it } from 'vitest';
import { stampIso } from './stamp';

describe('stampIso', () => {
  it('turns the library stamps into ISO', () => {
    expect(stampIso('20261114T042559244087Z')).toBe('2026-11-14T04:25:59Z');
    expect(stampIso('20261114T042605395445Z_feafa843')).toBe('2026-11-14T04:26:05Z');
    expect(stampIso('2026-11-14T04:26:42+00:00')).toBe('2026-11-14T04:26:42+00:00');
    expect(stampIso(null)).toBeNull();
  });
});
