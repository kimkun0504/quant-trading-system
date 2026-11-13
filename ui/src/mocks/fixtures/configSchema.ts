import type { ConfigSchema } from '@/shared/api';

/**
 * JSON Schema of the mock's SystemConfig, in the shape `/api/config/schema` generates from
 * the live dataclasses: one object per section, one leaf per scalar with its default,
 * bounds, enum, `nullable` and description. It covers the scalar leaves of the mock config
 * tree (`fixtures/generate.ts`); `seed`, `execution.signal_lag_days` and the `cache`
 * section are there because the real schema has them (the builder sets those itself).
 */
export const CONFIG_SCHEMA: ConfigSchema = {
  type: 'object',
  title: 'SystemConfig',
  properties: {
    seed: {
      type: 'integer',
      title: 'Seed',
      description: 'Seed of every random stream of the run (set by the launch request).',
      default: 0,
      minimum: 0,
      maximum: 2147483647,
    },
    universe: {
      type: 'object',
      title: 'Universe',
      properties: {
        top_n_by_mcap: {
          type: 'integer',
          title: 'Top n by market cap',
          description: 'Names kept by point-in-time market cap at each universe refresh.',
          default: 1000,
          minimum: 10,
          maximum: 10000,
        },
        min_price: {
          type: 'number',
          title: 'Min price',
          description: 'Minimum share price (USD) for universe membership.',
          default: 5.0,
          minimum: 0,
        },
        min_adv_usd: {
          type: 'number',
          title: 'Min ADV USD',
          description: 'Minimum average daily dollar volume (USD) for universe membership.',
          default: 2000000,
          minimum: 0,
        },
      },
    },
    alpha: {
      type: 'object',
      title: 'Alpha model',
      properties: {
        horizon_days: {
          type: 'integer',
          title: 'Horizon days',
          description:
            'Label horizon in trading days: the forward return the alpha is scored and trained against. Shorter horizons trade faster and cost more.',
          default: 20,
          minimum: 1,
          maximum: 252,
        },
        ic_window: {
          type: 'integer',
          title: 'IC window',
          description: 'Trailing days of realised IC behind the dynamic theme weights.',
          default: 756,
          minimum: 20,
          maximum: 2520,
        },
        score_smoothing_halflife_days: {
          type: 'number',
          title: 'Score smoothing halflife days',
          description:
            'Exponential smoothing of the combined score over time, as a half-life in days; 0 turns it off.',
          default: 10.0,
          minimum: 0,
          maximum: 252,
        },
        sample_weight_halflife_years: {
          type: 'number',
          title: 'Sample weight halflife years',
          description:
            'Exponential recency weighting of the training rows, as a half-life in years; 0 weights all rows equally.',
          default: 0.0,
          minimum: 0,
          maximum: 50,
        },
        embargo_days: {
          type: 'integer',
          title: 'Embargo days',
          description: 'Days purged between training and prediction; defaults to horizon_days.',
          nullable: true,
          minimum: 0,
          maximum: 252,
        },
        lgbm: {
          type: 'object',
          title: 'Alpha model: LightGBM',
          description: 'LightGBM settings of the ML sleeve.',
          properties: {
            n_estimators: {
              type: 'integer',
              title: 'N estimators',
              default: 400,
              minimum: 1,
              maximum: 10000,
            },
            learning_rate: {
              type: 'number',
              title: 'Learning rate',
              default: 0.03,
              minimum: 0.000001,
              maximum: 1,
            },
            num_leaves: {
              type: 'integer',
              title: 'Num leaves',
              default: 15,
              minimum: 2,
              maximum: 4096,
            },
          },
        },
      },
    },
    risk: {
      type: 'object',
      title: 'Risk model',
      properties: {
        n_factors: { type: 'integer', title: 'N factors', default: 12 },
        halflife_days: {
          type: 'integer',
          title: 'Halflife days',
          description: 'Half-life of the factor covariance estimate.',
          default: 90,
        },
        shrinkage: {
          type: 'string',
          title: 'Shrinkage',
          default: 'ledoit_wolf',
          enum: ['ledoit_wolf', 'none'],
        },
      },
    },
    cost: {
      type: 'object',
      title: 'Transaction costs',
      properties: {
        impact_coefficient: {
          type: 'number',
          title: 'Impact coefficient',
          description: 'Square-root market impact coefficient.',
          default: 0.3,
          exclusiveMinimum: 0,
          maximum: 5,
        },
        spread_model: { type: 'string', title: 'Spread model', default: 'quoted' },
        borrow_bps: {
          type: 'number',
          title: 'Borrow bps',
          description: 'Annual general-collateral borrow cost in basis points.',
          default: 50,
          minimum: 0,
        },
        delisting_short_gain_haircut: {
          type: 'number',
          title: 'Delisting short gain haircut',
          description:
            'Share of a short position’s gain that is kept when the name delists (1 = all of it).',
          default: 1.0,
          minimum: 0,
          maximum: 1,
        },
      },
    },
    portfolio: {
      type: 'object',
      title: 'Portfolio construction',
      properties: {
        book: {
          type: 'string',
          title: 'Book',
          description: 'Dollar-neutral long/short, long-only, or 130/30.',
          default: 'long_short',
          enum: ['long_short', 'long_only', '130_30'],
        },
        gross_leverage_max: {
          type: 'number',
          title: 'Gross leverage max',
          description: 'Dollar-neutral long/short is about 100/100 at gross 2.0.',
          default: 2.0,
          minimum: 0,
          maximum: 10,
        },
        risk_aversion_mode: {
          type: 'string',
          title: 'Risk aversion mode',
          description:
            '"target_vol": gamma is searched per solve so the book’s ex-ante vol hits target_vol_annual (capped books accepted; vol_target_mode unused). "fixed": gamma = risk_aversion, target enforced by vol_target_mode.',
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
          title: 'Target vol annual',
          description: 'Annualised volatility target, as a fraction (0.10 = 10%).',
          nullable: true,
          default: 0.1,
          minimum: 0.01,
          maximum: 0.4,
        },
        trade_aversion: {
          type: 'number',
          title: 'Trade aversion',
          description:
            'Multiplier on the transaction-cost term. Raise it when effective holding is below target.',
          default: 1.0,
          minimum: 0,
          maximum: 10,
        },
        max_position_weight: {
          type: 'number',
          title: 'Max position weight',
          description: 'Cap on |weight| per name, as a fraction of NAV.',
          default: 0.02,
          minimum: 0.001,
          maximum: 0.1,
        },
        rebalance_every_days: {
          type: 'integer',
          title: 'Rebalance every days',
          description: 'Solve every k-th trading day; the book drifts in between.',
          default: 5,
          minimum: 1,
          maximum: 252,
        },
        beta_neutral: {
          type: 'boolean',
          title: 'Beta neutral',
          description: 'Constrain the book’s beta exposure to the band around zero.',
          default: true,
        },
      },
    },
    execution: {
      type: 'object',
      title: 'Execution',
      properties: {
        signal_lag_days: {
          type: 'integer',
          title: 'Signal lag days',
          description: 'Days between the signal and the trade (design decision D4).',
          default: 1,
          minimum: 0,
          maximum: 5,
        },
      },
    },
    validation: {
      type: 'object',
      title: 'Validation',
      properties: {
        n_splits: {
          type: 'integer',
          title: 'N splits',
          description: 'Purged walk-forward test blocks for the fold-Sharpe gate.',
          default: 6,
          minimum: 2,
          maximum: 12,
        },
        dsr_threshold: {
          type: 'number',
          title: 'DSR threshold',
          description: 'Deflated Sharpe ratio a run must reach to pass the DSR gate.',
          default: 0.95,
          minimum: 0,
          maximum: 1,
        },
        pbo_threshold: {
          type: 'number',
          title: 'PBO threshold',
          description: 'Probability of backtest overfitting above which the PBO gate fails.',
          default: 0.5,
          minimum: 0,
          maximum: 1,
        },
      },
    },
    cache: {
      type: 'object',
      title: 'Cache',
      properties: {
        enabled: {
          type: 'boolean',
          title: 'Enabled',
          description:
            'Reuse fitted risk and alpha models from the stage cache. Changes no result and no config hash.',
          default: false,
        },
        root: { type: 'string', title: 'Root', default: '' },
      },
    },
  },
};
