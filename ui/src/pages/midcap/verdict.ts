import type { BadgeTone } from '@/shared/ui';

/** The four outcomes of the pre-registered check (`qts.research.midcap.Verdict.outcome`). */
export function outcomeTone(outcome: string | null | undefined): BadgeTone {
  switch (outcome) {
    case 'PASS':
      return 'pass';
    case 'PASS, cost-fragile':
      return 'caution';
    case 'FAIL':
      return 'fail';
    default:
      return 'skipped';
  }
}

export function outcomeText(outcome: string | null | undefined): string {
  switch (outcome) {
    case 'PASS':
      return 'The strategy held up on names it was never tuned on: C1–C3 pass, the test was valid, and it survives doubled costs.';
    case 'PASS, cost-fragile':
      return 'C1–C3 pass and the test was valid, but at twice the modelled cost the net Sharpe is zero or below: the edge depends on the cost model being right.';
    case 'FAIL':
      return 'The test was valid and at least one of C1–C3 failed: the large-cap result did not carry over to mid-caps.';
    case 'INCONCLUSIVE':
      return 'A validity condition (V1–V3) failed or a criterion could not be measured, so the test says nothing either way about the strategy.';
    default:
      return 'No outcome recorded.';
  }
}
