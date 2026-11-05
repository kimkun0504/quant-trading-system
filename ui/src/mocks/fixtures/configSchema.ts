import type { ConfigSchema } from '@/shared/api';

/** JSON Schema of the SystemConfig sections the launch form exposes as overrides. */
export const CONFIG_SCHEMA: ConfigSchema = {
  type: 'object',
  title: 'SystemConfig',
  properties: {
    portfolio: {
      type: 'object',
      title: 'Portfolio construction',
      properties: {
        trade_aversion: {
          type: 'number',
          title: 'Trade aversion',
          description:
            'Multiplier on the transaction-cost term. Raise it when effective holding is below target.',
          default: 1.0,
          minimum: 0,
          maximum: 10,
        },
        risk_aversion_mode: {
          type: 'string',
          title: 'Risk aversion mode',
          description:
            '"target_vol": gamma is searched per solve so the book\u2019s ex-ante vol hits target_vol_annual (capped books accepted; vol_target_mode unused). "fixed": gamma = risk_aversion, target enforced by vol_target_mode.',
          default: 'target_vol',
          enum: ['target_vol', 'fixed'],
        },
        risk_aversion: {
          type: 'number',
          title: 'Risk aversion',
          description:
            'Annual risk aversion gamma on portfolio variance. Used as-is when risk_aversion_mode is "fixed" or no vol target is set; in "target_vol" mode gamma is searched per solve and this value is only the fallback.',
          default: 2.0,
          minimum: 0,
          maximum: 100,
        },
        target_vol_annual: {
          type: 'number',
          title: 'Target volatility',
          description: 'Annualised volatility target, as a fraction (0.10 = 10%).',
          default: 0.1,
          minimum: 0.01,
          maximum: 0.4,
        },
        max_position_weight: {
          type: 'number',
          title: 'Max position weight',
          description: 'Cap on |weight| per name, as a fraction of NAV.',
          default: 0.02,
          minimum: 0.001,
          maximum: 0.1,
        },
        beta_neutral: {
          type: 'boolean',
          title: 'Beta neutral',
          description: 'Constrain the book’s beta exposure to the band around zero.',
          default: true,
        },
      },
    },
    validation: {
      type: 'object',
      title: 'Validation',
      properties: {
        n_splits: {
          type: 'integer',
          title: 'Folds',
          description: 'Purged walk-forward test blocks for the fold-Sharpe gate.',
          default: 6,
          minimum: 2,
          maximum: 12,
        },
        embargo_days: {
          type: 'integer',
          title: 'Embargo (days)',
          description: 'Days dropped after each test block; at least the label horizon.',
          default: 20,
          minimum: 0,
          maximum: 120,
        },
      },
    },
  },
};
