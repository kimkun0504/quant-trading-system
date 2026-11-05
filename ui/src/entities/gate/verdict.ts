import type { GateName, GateResult, Hurdles, RunKind, RunStatus } from '@/shared/api';
import { num, sharpe } from '@/shared/lib/format';
import { isFiniteNumber } from '@/shared/lib/format';
import { gateDef, orderGates } from './definitions';

/**
 * The verdict line: one honest sentence, generated from the gates, that answers
 * "is this run real, and why?". Failures are named plainly; a suspicious result
 * overrides everything else.
 */

export type VerdictTone = 'pass' | 'fail' | 'suspect' | 'pending';

export interface Verdict {
  tone: VerdictTone;
  /** The main sentence (the page's hero). */
  sentence: string;
  /** Optional follow-up sentence (skipped required gates, synthetic-data caveat). */
  detail: string | null;
}

export interface VerdictInput {
  status: RunStatus;
  hurdles: Hurdles | null;
  sharpeNet?: number | undefined;
  nTrials?: number | undefined;
  kind?: RunKind | undefined;
}

/** "a", "a and b", "a, b and c". */
export function joinWords(words: readonly string[]): string {
  if (words.length === 0) return '';
  if (words.length === 1) return words[0] ?? '';
  return `${words.slice(0, -1).join(', ')} and ${words[words.length - 1] ?? ''}`;
}

function gateList(names: readonly GateName[], nTrials: number | undefined): string {
  const nouns = names.map((n) => {
    if (n === 'dsr' && isFiniteNumber(nTrials)) {
      const t = num(nTrials, { digits: nTrials % 1 ? 1 : 0 });
      return `deflated Sharpe with ${t} trial${nTrials === 1 ? '' : 's'}`;
    }
    return gateDef(n).verdictNoun;
  });
  // "the Sharpe and cost gates" / "the deflated Sharpe with 4 trials"
  const plain = names.filter((n) => !(n === 'dsr' && isFiniteNumber(nTrials)));
  const hasDsrPhrase = plain.length !== names.length;
  const plainNouns = plain.map((n) => gateDef(n).verdictNoun);
  if (!hasDsrPhrase) return `the ${joinWords(plainNouns)} gate${plain.length > 1 ? 's' : ''}`;
  if (plain.length === 0) return nouns[0] ?? '';
  const dsrPhrase = nouns[names.indexOf('dsr')] ?? '';
  return `the ${joinWords(plainNouns)} gate${plain.length > 1 ? 's' : ''} and ${dsrPhrase}`;
}

function conclusion(failed: readonly GateName[], phase: number): string {
  if (failed.includes('dsr') || failed.includes('t_stat'))
    return 'not yet distinguishable from luck';
  if (failed.includes('phase_sharpe'))
    return phase >= 2
      ? 'not enough to justify the ML layer’s extra overfitting risk'
      : 'too weak to clear the phase-1 bar';
  if (failed.includes('breakeven_cost'))
    return 'costs would eat the edge if the model is off by 2×';
  if (failed.includes('pbo')) return 'the configuration choice looks overfit';
  if (failed.includes('ic_sign') || failed.includes('net_sharpe_delta'))
    return 'the edge does not hold across time';
  return 'not admissible yet';
}

export function buildVerdict(input: VerdictInput): Verdict {
  const { status, hurdles, sharpeNet, nTrials, kind } = input;
  if (status === 'running')
    return {
      tone: 'pending',
      sentence: 'Still running — the gates are evaluated when the backtest finishes.',
      detail: null,
    };
  if (status === 'failed' || !hurdles)
    return {
      tone: 'pending',
      sentence: 'The run stopped before evaluation, so no gate was tested.',
      detail: 'Open the run log for the error, fix the cause and launch it again.',
    };

  const gates = orderGates(hurdles.gates);
  const byName = (n: GateName): GateResult | undefined => gates.find((g) => g.name === n);
  const suspect = byName('suspect');
  const synthetic =
    kind === 'synthetic'
      ? 'Synthetic market with a planted signal: this validates plumbing, not alpha.'
      : null;

  if (suspect?.status === 'FAIL') {
    const s = isFiniteNumber(sharpeNet) ? `Net Sharpe ${sharpe(sharpeNet)}` : 'Net Sharpe above 2';
    return {
      tone: 'suspect',
      sentence: `${s} is implausibly high — treat this run as a defect until leakage, survivorship, costs and selection are ruled out.`,
      detail:
        'The deflated Sharpe cannot detect leakage; check the signal lag and the PIT shift test first.',
    };
  }

  // "Passes" lists informative gates only: the plausibility check passing is not news.
  const passed = gates
    .filter((g) => g.status === 'PASS' && g.name !== 'suspect')
    .map((g) => g.name);
  const failed = gates.filter((g) => g.status === 'FAIL').map((g) => g.name);
  const requiredSkipped = gates
    .filter((g) => g.status === 'FAIL' && g.required && /skipped|not supplied/i.test(g.reason))
    .map((g) => gateDef(g.name).label);

  let sentence: string;
  if (failed.length === 0) {
    sentence =
      passed.length > 0
        ? `Passes every applied gate — ${gateList(passed, undefined).replace(/^the /, '')} — and is admissible for phase ${hurdles.phase}.`
        : 'No gate was applied, so nothing has been tested yet.';
  } else if (passed.length === 0) {
    sentence = `Fails ${gateList(failed, nTrials)} — ${conclusion(failed, hurdles.phase)}.`;
  } else {
    sentence = `Passes ${gateList(passed, undefined)} but fails ${gateList(failed, nTrials)} — ${conclusion(failed, hurdles.phase)}.`;
  }
  sentence = sentence.charAt(0).toUpperCase() + sentence.slice(1);

  const detailParts: string[] = [];
  if (requiredSkipped.length)
    detailParts.push(
      `${joinWords(requiredSkipped)} could not be computed and counts as a failure.`,
    );
  if (synthetic) detailParts.push(synthetic);

  return {
    tone: failed.length === 0 && passed.length > 0 ? 'pass' : 'fail',
    sentence,
    detail: detailParts.length ? detailParts.join(' ') : null,
  };
}

/** Whether to show the suspicious-result banner (net Sharpe > 2), independent of gates. */
export function isSuspicious(sharpeNet: number | null | undefined): boolean {
  return isFiniteNumber(sharpeNet) && sharpeNet > 2;
}
