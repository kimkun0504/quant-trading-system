import { describe, expect, it } from 'vitest';
import { bps, currency, date, dateTime, MINUS, num, pct, sharpe, shortHash, times } from './format';

describe('format', () => {
  it('uses the true minus sign and never prints -0', () => {
    expect(num(-0.0001, { digits: 2 })).toBe('0.00');
    expect(num(-1.5, { digits: 1 })).toBe(`${MINUS}1.5`);
  });

  it('signs gains when asked', () => {
    expect(pct(0.0476, { digits: 2, signed: true })).toBe('+4.76%');
    expect(pct(-0.063, { signed: true })).toBe(`${MINUS}6.3%`);
    expect(pct(0, { signed: true })).toBe('0.0%');
    expect(sharpe(0.742, { signed: true })).toBe('+0.74');
  });

  it('formats basis points, multiples and currency', () => {
    expect(bps(16.337)).toBe('16.3 bp');
    expect(times(1.04)).toBe('1.04×');
    expect(currency(5_212_345)).toBe('$5.21M');
    expect(currency(701_000)).toBe('$701.0k');
    expect(currency(-2_500_000_000)).toBe(`${MINUS}$2.50B`);
    expect(currency(1234567, { compact: false })).toBe('$1,234,567');
  });

  it('returns n/a for missing values', () => {
    expect(num(Number.NaN)).toBe('n/a');
    expect(pct(null)).toBe('n/a');
    expect(bps(undefined)).toBe('n/a');
  });

  it('formats dates in day-month-year order', () => {
    expect(date('2023-10-17')).toBe('17 Oct 2023');
    expect(dateTime('2026-11-05T09:14:02Z')).toBe('5 Nov 2026, 09:14 UTC');
    expect(shortHash('944dd5b41832277e')).toBe('944dd5b4');
  });
});
