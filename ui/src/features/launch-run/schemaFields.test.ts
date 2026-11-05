import { describe, expect, it } from 'vitest';
import type { ConfigSchema } from '@/shared/api';
import { groupFields, overrideFields } from './schemaFields';
import { DEFAULT_VALUES, toRequest, validateLaunch } from './validation';

const SCHEMA: ConfigSchema = {
  type: 'object',
  title: 'SystemConfig',
  properties: {
    seed: { type: 'integer', default: 0 },
    portfolio: {
      type: 'object',
      title: 'Portfolio construction',
      properties: {
        book: {
          type: 'string',
          title: 'Book',
          enum: ['long_short', 'long_only'],
          default: 'long_short',
        },
        trade_aversion: { type: 'number', title: 'Trade aversion', minimum: 0, maximum: 10 },
      },
    },
    execution: {
      type: 'object',
      title: 'Execution',
      properties: { signal_lag_days: { type: 'integer', title: 'Signal lag days' } },
    },
    alpha: {
      type: 'object',
      title: 'Alpha model',
      properties: {
        horizon_days: { type: 'integer', title: 'Horizon days' },
        lgbm: {
          type: 'object',
          title: 'Alpha model: LightGBM',
          properties: { num_leaves: { type: 'integer', title: 'Num leaves', minimum: 2 } },
        },
      },
    },
  },
};

describe('override fields from the config schema', () => {
  it('recurses into nested sections and skips fields the form owns', () => {
    const f = overrideFields(SCHEMA);
    expect(f.map((x) => x.path)).toEqual([
      'portfolio.book',
      'portfolio.trade_aversion',
      'alpha.horizon_days',
      'alpha.lgbm.num_leaves',
    ]);
    expect(groupFields(f).map(([g, xs]) => [g, xs.length])).toEqual([
      ['Portfolio construction', 2],
      ['Alpha model', 1],
      ['Alpha model: LightGBM', 1],
    ]);
  });

  it('keeps string overrides as strings and validates only numbers', () => {
    const fields = overrideFields(SCHEMA);
    const v = {
      ...DEFAULT_VALUES,
      overrides: { 'portfolio.book': 'long_only', 'alpha.lgbm.num_leaves': '1' },
    };
    expect(validateLaunch(v, fields)).toEqual({
      'override:alpha.lgbm.num_leaves': 'Num leaves must be at least 2.',
    });
    const ok = { ...v, overrides: { ...v.overrides, 'alpha.lgbm.num_leaves': '15' } };
    expect(toRequest(ok, fields).config_overrides).toEqual({
      'portfolio.book': 'long_only',
      'alpha.lgbm.num_leaves': 15,
    });
  });
});
