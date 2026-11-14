import type { HandbookRef } from '@/shared/lib/handbook';

/**
 * One plain-language line per statistic the research pages show: what it is and what a
 * good value looks like (the owner is an engineer, not a quant). The longer reading is in
 * the handbook section the `About` lines link to. Wording follows handbook 10 §15.
 */
export interface StatDef {
  label: string;
  /** What it is and what a good value looks like, in one or two short sentences. */
  help: string;
  formula?: string;
}

export const STATS = {
  // ---- trials and families
  family: {
    label: 'Trial family',
    help: 'The group of tries a result is judged against: the same kind of test on the same data. The more tries in a family, the higher the bar for any one of them.',
  },
  trials: {
    label: 'Trials',
    help: 'How many distinct configurations were tried in the family. Each new one makes a lucky best result more likely, so the deflated Sharpe of the best falls.',
  },
  nEff: {
    label: 'Effective trials',
    help: 'The trial count corrected for trials that behave alike (Nyholt): ten near-copies count as fewer than ten independent tries.',
  },
  expMaxSr: {
    label: 'Expected best Sharpe by luck',
    help: 'The Sharpe ratio the best of this many trials would reach with no skill at all. A result must clear it by a margin to mean anything.',
    formula: 'E[max SR] ≈ √V · ((1−γ) Φ⁻¹(1−1/N) + γ Φ⁻¹(1−1/(N·e)))',
  },
  dsr: {
    label: 'Deflated Sharpe (DSR)',
    help: 'The probability that the true Sharpe is above zero after accounting for the number of trials and fat tails. Above 0.95 passes; 0.5 is a coin flip.',
  },
  dsrHaircut: {
    label: 'DSR haircut',
    help: 'How much the current best trial’s deflated Sharpe falls because of the new trials. A large haircut means the search itself is eroding the evidence.',
  },
  // ---- signal tests
  ic: {
    label: 'Rank IC',
    help: 'Correlation between today’s signal ranks and the next h days’ residual return ranks. Real equity signals are small: 0.02 to 0.05 is good, above 0.1 is suspicious.',
  },
  tNw: {
    label: 'Newey–West t',
    help: 'How many standard errors the mean IC is from zero, corrected for overlapping labels. |t| ≥ 3 is the bar for a new signal; 2 is not enough when many were tried.',
  },
  hurdle: {
    label: 'Harvey–Liu–Zhu hurdle',
    help: 'A new factor needs |t| ≥ 3.0, not 1.96, because hundreds of factors have already been tried in the literature (Harvey, Liu & Zhu 2016).',
  },
  pHolm: {
    label: 'Holm-adjusted p',
    help: 'The p-value after correcting for every signal tested in this family so far (Holm 1979). Below 0.05 means the signal survives the multiple testing; it rises as more tests are logged.',
    formula: 'p_adj(k) = max_{j≤k} min(1, (m − j + 1) · p(j))',
  },
  hit: {
    label: 'Hit rate',
    help: 'Share of dates on which the IC was positive. A stable signal is above 55%; 50% is a coin.',
  },
  halfLife: {
    label: 'Half-life',
    help: 'Days until the signal’s ranks are only half as correlated with today’s. A short half-life trades a lot and costs a lot.',
  },
  turnover: {
    label: 'Turnover',
    help: 'Share of the book traded per day. More turnover means more cost; below 10% a day is moderate for a weekly-rebalanced book.',
  },
  breakeven: {
    label: 'Breakeven cost',
    help: 'The cost per side (in basis points) at which the book’s return would be zero. Comfortably above the modelled cost (2× or more) is safe.',
  },
  incremental: {
    label: 'Incremental IC',
    help: 'The IC of what is left of the signal after removing what the baseline already knows (its traded score and the risk exposures). Near zero means nothing new.',
  },
  residShare: {
    label: 'New information',
    help: 'Share of the signal’s variation that the baseline and the risk exposures do not explain: 1 is all new, 0 is a copy.',
  },
  deltaSharpe: {
    label: 'Blend Δ Sharpe',
    help: 'Net Sharpe of the baseline + 0.25 × signal paper book minus the baseline book alone. The weight 0.25 is fixed: tuning it would be another search.',
  },
  paperSharpe: {
    label: 'Paper-book Sharpe',
    help: 'Sharpe of a simple neutral book on the signal alone (no optimizer, no caps). A scale-free read of tradability, not a backtest.',
  },
  coverage: {
    label: 'Coverage',
    help: 'Share of eligible names with a value. Below 50% the IC describes the covered names only.',
  },
  // ---- sweeps
  pbo: {
    label: 'PBO',
    help: 'Probability of backtest overfitting: how often the member that looked best in one half of the history ranks below the median in the other half. Above 0.5 says the grid’s winner is more likely noise than skill.',
  },
  probOosLoss: {
    label: 'P(best loses out of sample)',
    help: 'Share of splits in which the in-sample best member has a negative Sharpe out of sample.',
  },
  degradation: {
    label: 'Degradation slope',
    help: 'How out-of-sample Sharpe moves with in-sample Sharpe across splits. Negative: the better a member looks in sample, the worse it does after.',
  },
  // ---- capacity
  netSharpe: {
    label: 'Net Sharpe',
    help: 'Annualised return over volatility after trading costs and borrow. Above 0.5 is a good long/short equity book; above 2 is suspicious.',
  },
  costPerSide: {
    label: 'Cost per side',
    help: 'Trading cost per dollar traded, in basis points, split into spread, market impact and fees. Impact grows with the size of the book.',
  },
  advCap: {
    label: 'At the ADV cap',
    help: 'Share of traded names whose trade was cut by the participation cap (a fraction of average daily volume). High values mean the book is too big for its names.',
  },
  halfSharpeNav: {
    label: 'Half-Sharpe NAV',
    help: 'The book size at which the net Sharpe falls to half its value at the smallest NAV: a practical capacity limit.',
  },
  // ---- risk calibration
  volRatio: {
    label: 'Realised / ex-ante vol',
    help: 'Volatility the book actually had divided by what the risk model predicted. 1.0 is a calibrated model; above 1 the model underestimates risk.',
  },
  biasStat: {
    label: 'Bias statistic',
    help: 'Standard deviation of returns divided by their forecast volatility. 1 is unbiased; the band 0.8–1.2 is normal sampling noise over a year.',
  },
  nu: {
    label: 'ν (alignment scale)',
    help: 'How much extra risk the model assigns to the part of the alpha its factors do not explain. Larger ν makes the optimizer more cautious with alpha-heavy books.',
  },
  // ---- paper
  cone: {
    label: 'Expectation cone',
    help: 'Where the backtest says the cumulative return should be after this many days: the middle 95% of bootstrap paths of the reference run.',
  },
  boundary: {
    label: 'Stop boundary',
    help: 'The drawdown (or Sharpe) a book that behaves like the backtest crosses in its first year with only a 2.5% (STOP) or 10% (WATCH) chance.',
  },
  liveMode: {
    label: 'Live / backfill',
    help: 'Live: the orders were generated before the 15:50 ET cutoff and could have been sent. Backfill: generated after the fact; only live days count as paper-trading evidence.',
  },
} satisfies Record<string, StatDef>;

export type StatKey = keyof typeof STATS;

export const HANDBOOK_SEAL: HandbookRef = {
  page: 'validation',
  section: '15. The holdout seal, research families and the research tools',
};
export const HANDBOOK_SIGNALS: HandbookRef = {
  page: 'validation',
  section: '15. Signal tests: the statistics',
};
export const HANDBOOK_SWEEPS: HandbookRef = { page: 'validation', section: '15. Sweeps and PBO' };
export const HANDBOOK_MIDCAP: HandbookRef = {
  page: 'validation',
  section: '15. The mid-cap confirmation',
};
export const HANDBOOK_PAPER: HandbookRef = { page: 'backtesting', section: 'Paper trading' };
export const HANDBOOK_COSTS: HandbookRef = { page: 'tcost', section: 'Capacity' };
export const HANDBOOK_RISK: HandbookRef = { page: 'risk', section: 'Recalibrating ν and b' };
