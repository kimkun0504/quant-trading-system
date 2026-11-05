import type { GateName, GateResult } from '@/shared/api';
import type { HandbookRef } from '@/shared/lib/handbook';

/**
 * The plan's hurdle gates (qts/validation/hurdles.py), in the order the pipeline
 * evaluates them, with the words the verdict line uses for each.
 */
export interface GateDef {
  name: GateName;
  /** Label in the gate strip and tables. */
  label: string;
  /** Noun phrase used inside the verdict sentence: "the Sharpe gate". */
  verdictNoun: string;
  /** What the gate tests, one sentence. */
  test: string;
  /** Threshold as the handbook states it. */
  threshold: string;
  /** Why the gate exists. */
  rationale: string;
  handbook: HandbookRef;
}

const GATE_SECTION: HandbookRef = { page: 'validation', section: '14. The hurdle gates' };

export const GATE_ORDER: readonly GateName[] = [
  't_stat',
  'ic_sign',
  'incremental_ic',
  'net_sharpe_delta',
  'breakeven_cost',
  'dsr',
  'pbo',
  'phase_sharpe',
  'suspect',
  'failed_solves',
];

export const GATES: Readonly<Record<GateName, GateDef>> = {
  t_stat: {
    name: 't_stat',
    label: 'Decile t-stat',
    verdictNoun: 'decile t-stat',
    test: 'Newey–West t of the value-weighted top-minus-bottom decile spread.',
    threshold: 't > 3.0',
    rationale: 'Harvey–Liu–Zhu multiple-testing bar for a new factor; standalone signals only.',
    handbook: GATE_SECTION,
  },
  ic_sign: {
    name: 'ic_sign',
    label: 'IC sign',
    verdictNoun: 'IC-sign',
    test: 'Share of folds whose mean IC is positive.',
    threshold: '≥ 2/3 of folds',
    rationale: 'The signal must rank the right way round most of the time, not on average.',
    handbook: GATE_SECTION,
  },
  incremental_ic: {
    name: 'incremental_ic',
    label: 'Incremental IC',
    verdictNoun: 'incremental-IC',
    test: 'Rank IC after regressing on the existing composite, by fold.',
    threshold: '> 0.005 in ≥ 2/3 of folds',
    rationale: 'Redundancy against the factor zoo; skipped for standalone runs.',
    handbook: GATE_SECTION,
  },
  net_sharpe_delta: {
    name: 'net_sharpe_delta',
    label: 'Fold Sharpe',
    verdictNoun: 'fold-Sharpe',
    test: 'Net Sharpe improvement by fold (standalone: against a zero baseline).',
    threshold: '≥ +0.05 in ≥ 2/3 of folds',
    rationale: 'Plateau-and-majority rule: the edge must hold across time, not in one lucky year.',
    handbook: GATE_SECTION,
  },
  breakeven_cost: {
    name: 'breakeven_cost',
    label: 'Breakeven cost',
    verdictNoun: 'cost',
    test: 'Breakeven cost per side against the modelled cost per side.',
    threshold: 'breakeven ≥ 2 × modelled',
    rationale: 'Room for the cost model to be wrong by a factor of two.',
    handbook: GATE_SECTION,
  },
  dsr: {
    name: 'dsr',
    label: 'Deflated Sharpe',
    verdictNoun: 'deflated-Sharpe',
    test: 'Deflated Sharpe ratio given the trial ledger’s count of trials.',
    threshold: 'DSR > 0.95',
    rationale: 'Corrects for selection: the more you try, the higher a lucky Sharpe gets.',
    handbook: GATE_SECTION,
  },
  pbo: {
    name: 'pbo',
    label: 'PBO',
    verdictNoun: 'overfitting (PBO)',
    test: 'Probability of backtest overfitting over a configuration grid (CSCV).',
    threshold: 'PBO < 0.5',
    rationale: 'The selection procedure must beat a coin flip out of sample.',
    handbook: GATE_SECTION,
  },
  phase_sharpe: {
    name: 'phase_sharpe',
    label: 'Phase Sharpe',
    verdictNoun: 'Sharpe',
    test: 'Net annual Sharpe of the full pipeline.',
    threshold: 'phase 1 ≥ 0.3; phase 2 ≥ 0.5',
    rationale: 'Realistic net targets; the ML phase must earn its complexity.',
    handbook: GATE_SECTION,
  },
  suspect: {
    name: 'suspect',
    label: 'Plausibility',
    verdictNoun: 'plausibility',
    test: 'Flags implausibly high net Sharpe.',
    threshold: 'net Sharpe ≤ 2',
    rationale:
      'A net Sharpe above 2 is a defect until leakage, survivorship, cost and selection are ruled out. DSR cannot detect leakage.',
    handbook: GATE_SECTION,
  },
  failed_solves: {
    name: 'failed_solves',
    label: 'Solver health',
    verdictNoun: 'solver-health',
    test: 'Share of rebalance solves that failed, so the book was held instead of re-optimised.',
    threshold: '≤ 2% of rebalances',
    rationale:
      'A broken optimizer holds the current book every day; the result looks like a strategy but is a stale portfolio.',
    handbook: GATE_SECTION,
  },
};

export function gateDef(name: GateName): GateDef {
  return GATES[name];
}

/** Gates in pipeline order; unknown gates (future backend additions) keep their position at the end. */
export function orderGates(gates: readonly GateResult[]): GateResult[] {
  const rank = (g: GateResult) => {
    const i = GATE_ORDER.indexOf(g.name);
    return i === -1 ? GATE_ORDER.length : i;
  };
  return [...gates].sort((a, b) => rank(a) - rank(b));
}

export const STATUS_WORD: Record<GateResult['status'], string> = {
  PASS: 'passed',
  FAIL: 'failed',
  skipped: 'skipped',
};
