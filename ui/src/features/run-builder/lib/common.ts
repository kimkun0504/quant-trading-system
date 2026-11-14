import type { Field } from './fields';

/**
 * The handful of config settings that matter day to day, with plain-language labels and
 * help. A path the schema does not have (an older or newer SystemConfig) is skipped, so
 * the panel follows the live config.
 */
export interface CommonSetting {
  path: string;
  label: string;
  help: string;
}

export const COMMON_SETTINGS: readonly CommonSetting[] = [
  {
    path: 'alpha.horizon_days',
    label: 'Prediction horizon (days)',
    help: 'The forward return the alpha is scored against. Shorter horizons trade faster and pay more cost.',
  },
  {
    path: 'alpha.sample_weight_halflife_years',
    label: 'Training recency half-life (years), 0 = equal weights',
    help: 'How fast older training rows lose weight when the models are fitted. 0 weights every year the same; 8 halves a row’s weight every eight years.',
  },
  {
    path: 'alpha.score_smoothing_halflife_days',
    label: 'Score smoothing half-life (days), 0 = off',
    help: 'Exponential smoothing of the daily alpha score. More smoothing means a steadier book and less turnover, at the price of a slower signal.',
  },
  {
    path: 'portfolio.rebalance_every_days',
    label: 'Rebalance every (days)',
    help: 'The optimiser solves every k-th trading day; in between the book drifts.',
  },
  {
    path: 'portfolio.gross_leverage_max',
    label: 'Gross leverage cap',
    help: 'Longs plus shorts as a multiple of NAV. 2.0 is a 100/100 dollar-neutral book.',
  },
  {
    path: 'portfolio.max_position_weight',
    label: 'Max position weight',
    help: 'Cap on any one name, long or short, as a fraction of NAV (0.02 = 2%).',
  },
  {
    path: 'portfolio.target_vol_annual',
    label: 'Target volatility (annual)',
    help: 'The book’s ex-ante volatility target as a fraction (0.10 = 10%). “No value” runs with the fixed risk aversion instead.',
  },
  {
    path: 'portfolio.trade_aversion',
    label: 'Trade aversion',
    help: 'Multiplier on the transaction-cost term of the objective. 0 ignores costs when trading; raise it when the holding period is too short.',
  },
  {
    path: 'cost.delisting_short_gain_haircut',
    label: 'Delisting short-gain haircut',
    help: 'Share of a short’s gain that is kept when the name delists. 1 keeps it all; lower values assume the gain could not be fully realised.',
  },
  {
    path: 'alpha.feature_set',
    label: 'Feature set',
    help: 'v1: the 18 Tier-1 features. v2 adds earnings reaction, abnormal volume, accruals, net issuance and the 52-week high (needs the earnings tables in the store). The feature switches below follow it.',
  },
  {
    path: 'alpha.disabled_features',
    label: 'Features the model uses',
    help: 'Switch features off to see what the model does without them. Each setting is one logged trial.',
  },
];

export interface CommonField extends CommonSetting {
  field: Field;
}

/** The curated settings the schema actually has, in the curated order. */
export function commonFields(fields: readonly Field[]): CommonField[] {
  const byPath = new Map(fields.map((f) => [f.path, f]));
  const out: CommonField[] = [];
  for (const c of COMMON_SETTINGS) {
    const field = byPath.get(c.path);
    if (field) out.push({ ...c, field });
  }
  return out;
}
