import { describe, expect, it } from 'vitest';
import { DEFAULT_VALUES, toRequest, validateLaunch } from './validation';

describe('launch form validation', () => {
  it('defaults are valid', () => {
    expect(validateLaunch(DEFAULT_VALUES)).toEqual({});
  });

  it('names each problem in the field’s own terms', () => {
    const e = validateLaunch({
      ...DEFAULT_VALUES,
      stocks: '10',
      days: '500',
      seed: '1.5',
      phase: 2,
      ml: false,
    });
    expect(e.stocks).toBe('Use between 50 and 3,000.');
    expect(e.days).toMatch(/at least 764 \(512 warm-up days plus one year\)/);
    expect(e.seed).toBe('A seed must be a whole number.');
    expect(e.ml).toMatch(/Phase 2 tests the ML sleeve/);
  });

  it('checks schema bounds of overrides', () => {
    const e = validateLaunch(
      { ...DEFAULT_VALUES, overrides: { 'portfolio.trade_aversion': '-1' } },
      [
        {
          path: 'portfolio.trade_aversion',
          title: 'Trade aversion',
          type: 'number',
          minimum: 0,
          maximum: 10,
        },
      ],
    );
    expect(e['override:portfolio.trade_aversion']).toBe('Trade aversion must be at least 0.');
  });

  it('builds the request', () => {
    const r = toRequest({
      ...DEFAULT_VALUES,
      signal_lag_days: '0',
      overrides: { 'portfolio.trade_aversion': '2', 'x.y': '' },
    });
    expect(r).toMatchObject({
      stocks: 300,
      days: 1512,
      seed: 0,
      signal_lag_days: 0,
      config_overrides: { 'portfolio.trade_aversion': 2 },
    });
    expect(r.config_overrides).not.toHaveProperty('x.y');
  });
});
