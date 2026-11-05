import type { Phase } from '@/shared/api';
import { bps, days, num, pct, sharpe, times } from '@/shared/lib/format';
import { isFiniteNumber } from '@/shared/lib/format';
import type { HandbookRef } from '@/shared/lib/handbook';

/**
 * Every metric the run page shows, in the handbook's vocabulary.
 *
 * `interpret(value, ctx)` returns the margin note for *this* run: one or two sentences,
 * led by the value, saying what the number means here. It must stay honest: weak or
 * suspicious values are named as such.
 */

export type MetricKey =
  | 'sharpe_net'
  | 'sharpe_gross'
  | 'newey_west_t_net'
  | 'cagr_net'
  | 'cagr_gross'
  | 'ann_vol_net'
  | 'max_drawdown_net'
  | 'calmar_net'
  | 'sortino_net'
  | 'hit_rate_net'
  | 'avg_daily_turnover'
  | 'effective_holding_days'
  | 'avg_cost_bps_per_side'
  | 'breakeven_cost_bps'
  | 'transfer_coefficient_mean'
  | 'realized_vs_exante_vol_ratio'
  | 'mean_ic'
  | 'icir'
  | 'ic_t_stat'
  | 'decile_t'
  | 'dsr'
  | 'pbo';

/** Facts about the run that interpretations may refer to. All optional: notes degrade. */
export interface MetricContext {
  phase?: Phase;
  nDays?: number;
  sharpeNet?: number;
  sharpeGross?: number;
  cagrGross?: number;
  annVolNet?: number;
  costBps?: number;
  breakevenBps?: number;
  icHorizonDays?: number;
  icN?: number;
  icPeriodsPerYear?: number;
  icir?: number;
  icTStat?: number;
  icHitRate?: number;
  nTrials?: number;
  holdingDaysTarget?: number;
  volTarget?: number;
}

export interface MetricDef {
  key: MetricKey;
  /** Short label, sentence case, handbook vocabulary. */
  label: string;
  /** One-sentence definition shown on hover/focus. */
  definition: string;
  /** Plain-text formula. */
  formula: string;
  handbook: HandbookRef;
  format: (v: number | null | undefined) => string;
  interpret: (v: number | null | undefined, ctx: MetricContext) => string;
}

export const SUSPECT_SHARPE = 2;
export const T_STAT_BAR = 3;
export const DSR_BAR = 0.95;
export const PBO_BAR = 0.5;
export const BREAKEVEN_MULTIPLE = 2;
export const DEFAULT_HOLDING_TARGET = 50;
export const DEFAULT_VOL_TARGET = 0.1;

export function phaseSharpeBar(phase: Phase | undefined): number {
  return phase === 2 ? 0.5 : 0.3;
}

const na = (label: string) => `${label} is not available for this run.`;

/** t-statistic implied by an annual Sharpe over n daily observations (i.i.d. approximation). */
export function impliedT(sharpeAnnual: number, nDays: number): number {
  return sharpeAnnual * Math.sqrt(nDays / 252);
}

const defs: MetricDef[] = [
  {
    key: 'sharpe_net',
    label: 'Net Sharpe',
    definition:
      'Mean daily return after all costs and borrow, divided by its standard deviation, annualised by √252. What you keep, per unit of risk.',
    formula: 'SR = mean(r) / sd(r) × √252, r = daily net return',
    handbook: { page: 'backtesting', section: 'Metrics › Sharpe ratio (gross and net)' },
    format: (v) => sharpe(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Net Sharpe');
      const f = sharpe(v);
      if (v > SUSPECT_SHARPE)
        return `Net Sharpe ${f} — above 2, which the plan treats as a defect until leakage, survivorship, costs and selection are ruled out.`;
      if (v <= 0) return `Net Sharpe ${f} — after costs the strategy loses money per unit of risk.`;
      const bar = phaseSharpeBar(ctx.phase);
      const head =
        v >= bar
          ? `Net Sharpe ${f} — clears the phase-${ctx.phase ?? 1} bar of ${sharpe(bar)}`
          : `Net Sharpe ${f} — positive but below the phase-${ctx.phase ?? 1} bar of ${sharpe(bar)}`;
      if (!isFiniteNumber(ctx.nDays)) return `${head}.`;
      const years = ctx.nDays / 252;
      const t = impliedT(v, ctx.nDays);
      const tail =
        t < T_STAT_BAR
          ? `${num(years, { digits: 1 })} years give t ≈ ${num(t, { digits: 1 })}, too short to tell it from zero at t > 3.`
          : `${num(years, { digits: 1 })} years give t ≈ ${num(t, { digits: 1 })}.`;
      return `${head}; ${tail}`;
    },
  },
  {
    key: 'sharpe_gross',
    label: 'Gross Sharpe',
    definition:
      'Sharpe ratio of returns before trading costs and borrow: the signal’s quality as it comes out of the optimiser.',
    formula: 'SR = mean(g) / sd(g) × √252, g = daily gross return',
    handbook: { page: 'backtesting', section: 'Metrics › Sharpe ratio (gross and net)' },
    format: (v) => sharpe(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Gross Sharpe');
      const f = sharpe(v);
      if (!isFiniteNumber(ctx.sharpeNet) || v <= 0)
        return `Gross Sharpe ${f} — the edge before any trading cost.`;
      const share = 1 - ctx.sharpeNet / v;
      const lead = `Gross Sharpe ${f} vs net ${sharpe(ctx.sharpeNet)} — costs take ${pct(share, { digits: 0 })} of the Sharpe`;
      if (share > 0.5) return `${lead}; more than half the edge pays the market.`;
      if (share < 0.25) return `${lead}; a light cost drag.`;
      return `${lead}, typical for a daily-rebalanced book.`;
    },
  },
  {
    key: 'newey_west_t_net',
    label: 'Newey–West t',
    definition:
      'Tests whether the mean daily net return differs from zero, allowing for autocorrelation and heteroskedasticity.',
    formula: 't = mean(r) / √(Ŝ/T), Ŝ = Bartlett-weighted long-run variance, L = ⌊4(T/100)^(2/9)⌋',
    handbook: { page: 'backtesting', section: 'Metrics › Newey–West t-statistic' },
    format: (v) => num(v),
    interpret(v) {
      if (!isFiniteNumber(v)) return na('Newey–West t');
      const f = num(v);
      if (v >= T_STAT_BAR) return `t = ${f} — clears the plan’s t > 3 bar.`;
      if (v >= 2)
        return `t = ${f} — significant at conventional levels, but below the plan’s t > 3 bar for a factor zoo.`;
      return `t = ${f} — the mean return cannot be told apart from zero on this history.`;
    },
  },
  {
    key: 'cagr_net',
    label: 'Net CAGR',
    definition:
      'The constant annual return that turns the starting NAV into the ending NAV, after all costs and borrow.',
    formula: 'CAGR = (∏(1 + rₜ))^(252/T) − 1',
    handbook: { page: 'backtesting', section: 'Metrics › CAGR' },
    format: (v) => pct(v, { digits: 2, signed: true }),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Net CAGR');
      const f = pct(v, { digits: 2, signed: true });
      if (isFiniteNumber(ctx.cagrGross)) {
        const gap = ctx.cagrGross - v;
        return `Net CAGR ${f} — gross ${pct(ctx.cagrGross, { digits: 2, signed: true })}; the ${num(gap * 100, { digits: 1 })}-point gap is the annual cost drag (trading + borrow).`;
      }
      return `Net CAGR ${f} a year after all costs.`;
    },
  },
  {
    key: 'cagr_gross',
    label: 'Gross CAGR',
    definition: 'Compound annual growth rate before trading costs and borrow.',
    formula: 'CAGR = (∏(1 + gₜ))^(252/T) − 1',
    handbook: { page: 'backtesting', section: 'Metrics › CAGR' },
    format: (v) => pct(v, { digits: 2, signed: true }),
    interpret(v) {
      if (!isFiniteNumber(v)) return na('Gross CAGR');
      return `Gross CAGR ${pct(v, { digits: 2, signed: true })} — what the signal would earn if trading were free.`;
    },
  },
  {
    key: 'ann_vol_net',
    label: 'Annualised volatility',
    definition:
      'Standard deviation of daily net returns × √252. Assumes uncorrelated daily returns.',
    formula: 'σ = sd(r) × √252',
    handbook: { page: 'backtesting', section: 'Metrics › Annualised volatility' },
    format: (v) => pct(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Volatility');
      const target = ctx.volTarget ?? DEFAULT_VOL_TARGET;
      const f = pct(v);
      if (v < target * 0.8)
        return `Volatility ${f} — below the ${pct(target, { digits: 0 })} target: the gross-leverage cap binds, so vol targeting cannot lever up.`;
      if (v > target * 1.2)
        return `Volatility ${f} — above the ${pct(target, { digits: 0 })} target; check the vol-target projection.`;
      return `Volatility ${f} — close to the ${pct(target, { digits: 0 })} target.`;
    },
  },
  {
    key: 'max_drawdown_net',
    label: 'Max drawdown',
    definition:
      'The worst peak-to-trough loss of the compounded net NAV. A single-path statistic: noisy, and it grows with history length.',
    formula: 'min over t of NAVₜ / max(NAV₀…NAVₜ) − 1',
    handbook: { page: 'backtesting', section: 'Metrics › Maximum drawdown and Calmar' },
    format: (v) => pct(v, { signed: true }),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Max drawdown');
      const f = pct(v, { signed: true });
      if (isFiniteNumber(ctx.annVolNet) && ctx.annVolNet > 0) {
        const r = Math.abs(v) / ctx.annVolNet;
        if (r < 1)
          return `Max drawdown ${f} — less than one year of volatility; benign for this path. Compare only across equal-length windows.`;
        if (r < 2)
          return `Max drawdown ${f} — ${num(r, { digits: 1 })}× annual volatility, normal for a multi-year history.`;
        return `Max drawdown ${f} — ${num(r, { digits: 1 })}× annual volatility: a deep hole for this risk level.`;
      }
      return `Max drawdown ${f}.`;
    },
  },
  {
    key: 'calmar_net',
    label: 'Calmar',
    definition: 'Net CAGR divided by the absolute maximum drawdown.',
    formula: 'Calmar = CAGR / |max drawdown|',
    handbook: { page: 'backtesting', section: 'Metrics › Maximum drawdown and Calmar' },
    format: (v) => num(v),
    interpret(v) {
      if (!isFiniteNumber(v)) return na('Calmar');
      if (v <= 0) return `Calmar ${num(v)} — the strategy has not earned back its worst loss.`;
      return `Calmar ${num(v)} — about ${num(1 / v, { digits: 1 })} years of average return to earn back the worst drawdown.`;
    },
  },
  {
    key: 'sortino_net',
    label: 'Sortino',
    definition:
      'Like Sharpe, but only downside deviations count as risk. For symmetric returns it is about √2 × Sharpe.',
    formula: 'Sortino = mean(r) / √(mean(min(r, 0)²)) × √252',
    handbook: { page: 'backtesting', section: 'Metrics › Sortino ratio' },
    format: (v) => num(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Sortino');
      if (!isFiniteNumber(ctx.sharpeNet) || ctx.sharpeNet <= 0)
        return `Sortino ${num(v)} — return per unit of downside risk.`;
      const r = v / ctx.sharpeNet;
      if (r < 1.3)
        return `Sortino ${num(v)} — under √2 × Sharpe: losing days are larger than winning days (negative skew).`;
      if (r > 1.55)
        return `Sortino ${num(v)} — over √2 × Sharpe: the downside is lighter than the upside.`;
      return `Sortino ${num(v)} — about √2 × Sharpe, as for roughly symmetric returns.`;
    },
  },
  {
    key: 'hit_rate_net',
    label: 'Hit rate',
    definition: 'Share of non-zero daily net returns that are positive.',
    formula: 'hit = #(r > 0) / #(r ≠ 0)',
    handbook: { page: 'backtesting', section: 'Metrics › Hit rate' },
    format: (v) => pct(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Hit rate');
      const f = pct(v);
      if (v > 0.56)
        return `Hit rate ${f} — unusually high for stock selection (51–54% is typical); check for leakage or stale prices.`;
      if (v < 0.5 && isFiniteNumber(ctx.sharpeNet) && ctx.sharpeNet > 0)
        return `Hit rate ${f} — fewer winning days than losing ones; the edge comes from larger wins.`;
      return `Hit rate ${f} — in the 51–54% band typical of stock-selection books.`;
    },
  },
  {
    key: 'avg_daily_turnover',
    label: 'Turnover per day',
    definition:
      'Two-sided daily turnover: the sum of absolute trades as a fraction of NAV. Buying 1% and selling 1% is 2%.',
    formula: 'TOₜ = Σᵢ |xᵢₜ| / NAVₜ',
    handbook: { page: 'backtesting', section: 'Metrics › Turnover (as defined here)' },
    format: (v) => pct(v, { digits: 2 }),
    interpret(v) {
      if (!isFiniteNumber(v)) return na('Turnover');
      return `Turnover ${pct(v, { digits: 2 })} a day, two-sided — about ${num(v * 252, { digits: 1 })}× NAV traded a year.`;
    },
  },
  {
    key: 'effective_holding_days',
    label: 'Effective holding',
    definition:
      'Average gross exposure divided by one-sided daily turnover: roughly how long a dollar of position stays on.',
    formula: 'EHD = mean gross exposure / (mean two-sided turnover / 2)',
    handbook: { page: 'backtesting', section: 'Metrics › Effective holding days' },
    format: (v) => days(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Effective holding');
      const target = ctx.holdingDaysTarget ?? DEFAULT_HOLDING_TARGET;
      if (v >= target)
        return `Holding ${days(v)} — above the ${target}-day monitored target; it should be of the order of the signal’s half-life.`;
      return `Holding ${days(v)} — below the ${target}-day target: the book trades faster than the signal decays; raise trade_aversion.`;
    },
  },
  {
    key: 'avg_cost_bps_per_side',
    label: 'Cost per side',
    definition:
      'Trading cost (spread + impact + fees) per dollar traded, in basis points. Borrow is a holding cost and is excluded.',
    formula: 'Σ(spread + impact + fees) / Σ(TOₜ × NAVₜ) × 10⁴',
    handbook: { page: 'backtesting', section: 'Metrics › Cost per side' },
    format: (v) => bps(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Cost per side');
      const f = bps(v);
      if (isFiniteNumber(ctx.breakevenBps) && v > 0)
        return `Cost ${f} per side — breakeven is ${bps(ctx.breakevenBps)}, so costs could rise ${times(ctx.breakevenBps / v, 1)} before the gross edge is gone.`;
      return `Cost ${f} per dollar traded, all-in.`;
    },
  },
  {
    key: 'breakeven_cost_bps',
    label: 'Breakeven cost',
    definition: 'The per-side cost at which the mean gross return is exactly used up by trading.',
    formula: 'c* = mean(gₜ) / mean(TOₜ) × 10⁴ bp',
    handbook: { page: 'backtesting', section: 'Metrics › Breakeven cost: derivation' },
    format: (v) => bps(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Breakeven cost');
      const f = bps(v);
      if (v <= 0) return `Breakeven ${f} — there is no gross edge to spend on costs.`;
      if (!isFiniteNumber(ctx.costBps) || ctx.costBps <= 0) return `Breakeven ${f} per side.`;
      const m = v / ctx.costBps;
      if (m >= BREAKEVEN_MULTIPLE)
        return `Breakeven ${f} — ${times(m, 1)} the modelled ${bps(ctx.costBps)}, inside the gate’s 2× safety margin.`;
      return `Breakeven ${f} — only ${times(m, 1)} the modelled ${bps(ctx.costBps)}; the gate wants 2× room for a mis-specified cost model.`;
    },
  },
  {
    key: 'transfer_coefficient_mean',
    label: 'Transfer coefficient',
    definition:
      'Correlation between the unconstrained ideal weights and the weights actually held. Constraints, costs and caps lower it.',
    formula: 'TC = corr(w_ideal, w_held); IR ≈ TC × IC × √breadth',
    handbook: { page: 'portfolio', section: 'Diagnostics › Transfer coefficient' },
    format: (v) => num(v),
    interpret(v) {
      if (!isFiniteNumber(v)) return na('Transfer coefficient');
      if (v < 0.5)
        return `TC ${num(v)} — constraints and cost penalties leak more than half of the forecast into the held book.`;
      if (v < 0.75)
        return `TC ${num(v)} — the book keeps most of the forecast; typical for a constrained long/short.`;
      return `TC ${num(v)} — nearly unconstrained; check that cost penalties are active.`;
    },
  },
  {
    key: 'realized_vs_exante_vol_ratio',
    label: 'Realised / ex-ante vol',
    definition:
      'Realised gross volatility divided by the mean ex-ante volatility the risk model forecast the day before.',
    formula: 'ratio = σ_realised / mean(σ_ex-ante, t−1)',
    handbook: { page: 'risk', section: 'Evaluation › Bias statistic' },
    format: (v) => times(v),
    interpret(v) {
      if (!isFiniteNumber(v)) return na('Realised / ex-ante vol');
      if (v > 1.1)
        return `Ratio ${times(v)} — realised risk runs ${pct(v - 1, { digits: 0 })} above forecast; the risk model under-predicts.`;
      if (v < 0.9)
        return `Ratio ${times(v)} — realised risk is ${pct(1 - v, { digits: 0 })} below forecast; the risk model is conservative.`;
      return `Ratio ${times(v)} — the risk model forecasts this book’s volatility well.`;
    },
  },
  {
    key: 'mean_ic',
    label: 'Mean rank IC',
    definition:
      'Average Spearman correlation between the score and the forward residualised return, one value per non-overlapping date.',
    formula: 'IC̄ = mean over t of corr(rank sₜ, rank yₜ)',
    handbook: { page: 'validation', section: '5. The information coefficient (IC)' },
    format: (v) => num(v, { digits: 4 }),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Mean IC');
      const f = num(v, { digits: 4 });
      const over =
        isFiniteNumber(ctx.icHorizonDays) && isFiniteNumber(ctx.icN)
          ? ` over ${ctx.icN} dates at a ${ctx.icHorizonDays}-day horizon`
          : '';
      if (v > 0.08)
        return `Mean IC ${f}${over} — very high for stock selection; suspect leakage first.`;
      if (v > 0.02) return `Mean IC ${f}${over} — respectable for a cross-sectional equity signal.`;
      if (v > 0) return `Mean IC ${f}${over} — weak ranking skill.`;
      return `Mean IC ${f}${over} — no ranking skill.`;
    },
  },
  {
    key: 'icir',
    label: 'ICIR',
    definition:
      'Mean IC divided by its standard deviation, annualised at the sampling frequency of the IC series (not always 252). The Sharpe ratio of the IC.',
    formula: 'ICIR = IC̄ / sd(IC) × √p, p = IC observations per year',
    handbook: { page: 'validation', section: '5. The information coefficient (IC)' },
    format: (v) => num(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('ICIR');
      const f = num(v);
      if (v < 0) return `ICIR ${f} — negative: the signal ranks backwards on average.`;
      if (v < 0.5) return `ICIR ${f} — barely distinguishable from noise.`;
      if (v < 2) {
        const hit = isFiniteNumber(ctx.icHitRate)
          ? ` (the IC was positive on ${pct(ctx.icHitRate, { digits: 0 })} of dates)`
          : '';
        return `ICIR ${f} — positive but noisy; below ~2 the signal’s sign flips in many periods${hit}.`;
      }
      return `ICIR ${f} — consistent: the IC keeps its sign in most periods.`;
    },
  },
  {
    key: 'ic_t_stat',
    label: 'IC t-stat',
    definition: 'Per-period ICIR × √n: tests whether the mean IC differs from zero.',
    formula: 't = (IC̄ / sd(IC)) × √n',
    handbook: { page: 'validation', section: '5. The information coefficient (IC)' },
    format: (v) => num(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('IC t-stat');
      const f = num(v);
      let check = '';
      if (
        isFiniteNumber(ctx.icir) &&
        isFiniteNumber(ctx.icPeriodsPerYear) &&
        isFiniteNumber(ctx.icN) &&
        ctx.icPeriodsPerYear > 0 &&
        v !== 0
      ) {
        const expected = (ctx.icir / Math.sqrt(ctx.icPeriodsPerYear)) * Math.sqrt(ctx.icN);
        if (Math.abs(expected - v) / Math.abs(v) > 0.25)
          check = ' ICIR and t disagree: check the annualisation frequency (the ×4.5 bug).';
      }
      if (v >= T_STAT_BAR) return `IC t ${f} — the ranking skill is statistically real.${check}`;
      return `IC t ${f} — below 3; the mean IC is not yet distinguishable from zero.${check}`;
    },
  },
  {
    key: 'decile_t',
    label: 'Decile L/S t',
    definition:
      'Newey–West t of the value-weighted top-minus-bottom decile of raw forward returns, on non-overlapping formation dates. Input to the t-stat gate.',
    formula: 't_NW of (top decile − bottom decile) returns, every H days',
    handbook: { page: 'validation', section: '6. Decile portfolios and long–short spreads' },
    format: (v) => num(v),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Decile t');
      const f = num(v);
      const contrast =
        isFiniteNumber(ctx.icTStat) && ctx.icTStat > v + 1
          ? ' The residualised IC is much stronger: the decile portfolio carries unhedged factor noise.'
          : '';
      if (v > T_STAT_BAR) return `Decile t ${f} — passes the t > 3 gate.${contrast}`;
      return `Decile t ${f} — fails the t > 3 gate.${contrast}`;
    },
  },
  {
    key: 'dsr',
    label: 'Deflated Sharpe',
    definition:
      'Probability that the true Sharpe exceeds the best Sharpe you would expect from luck alone, given how many trials were run and how much their Sharpes varied.',
    formula: 'DSR = Φ((SR − E[max SR | N trials]) / σ(SR))',
    handbook: {
      page: 'validation',
      section: '9. Probabilistic Sharpe, expected maximum Sharpe, Deflated Sharpe',
    },
    format: (v) => num(v, { digits: 3 }),
    interpret(v, ctx) {
      if (!isFiniteNumber(v)) return na('Deflated Sharpe');
      const f = num(v, { digits: 3 });
      const n = isFiniteNumber(ctx.nTrials)
        ? num(ctx.nTrials, { digits: ctx.nTrials % 1 ? 1 : 0 })
        : null;
      const trials = n ? `${n} trial${ctx.nTrials === 1 ? '' : 's'}` : 'the trials run so far';
      if (v > DSR_BAR) return `DSR ${f} — clears 0.95 even after counting ${trials}.`;
      return `DSR ${f} — below 0.95: with ${trials} this Sharpe is not yet distinguishable from luck.`;
    },
  },
  {
    key: 'pbo',
    label: 'PBO',
    definition:
      'Probability of backtest overfitting (CSCV): how often the in-sample best configuration ranks below the median out of sample.',
    formula: 'PBO = share of CSCV splits where the IS winner’s OOS rank < median',
    handbook: { page: 'validation', section: '10. Probability of Backtest Overfitting (CSCV)' },
    format: (v) => (isFiniteNumber(v) ? pct(v, { digits: 0 }) : 'skipped'),
    interpret(v) {
      if (!isFiniteNumber(v))
        return 'PBO skipped — no configuration grid was run; set pipeline_pbo_configs to compute it.';
      const f = pct(v, { digits: 0 });
      if (v < PBO_BAR) return `PBO ${f} — the selection procedure beats a coin flip out of sample.`;
      return `PBO ${f} — the in-sample winner usually loses out of sample; the configuration choice is overfit.`;
    },
  },
];

export const METRICS: Readonly<Record<MetricKey, MetricDef>> = Object.fromEntries(
  defs.map((d) => [d.key, d]),
) as Record<MetricKey, MetricDef>;

export function metric(key: MetricKey): MetricDef {
  return METRICS[key];
}
