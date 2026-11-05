import { describe, expect, it } from 'vitest';
import type { GateName, GateResult, GateStatus, Hurdles } from '@/shared/api';
import { buildVerdict, isSuspicious, joinWords } from './verdict';

function hurdles(statuses: Partial<Record<GateName, GateStatus>>, phase: 1 | 2 = 1): Hurdles {
  const gates: GateResult[] = (Object.entries(statuses) as [GateName, GateStatus][]).map(
    ([name, status]) => ({
      name,
      status,
      value: null,
      threshold: null,
      reason: '',
      required: ['phase_sharpe', 'suspect', 'dsr'].includes(name),
    }),
  );
  const overall = gates.some((g) => g.status === 'FAIL') ? 'FAIL' : 'PASS';
  return { overall, phase, required: ['phase_sharpe', 'suspect', 'dsr'], gates };
}

const DEMO = hurdles({
  t_stat: 'FAIL',
  ic_sign: 'PASS',
  incremental_ic: 'skipped',
  net_sharpe_delta: 'PASS',
  breakeven_cost: 'PASS',
  dsr: 'FAIL',
  pbo: 'skipped',
  phase_sharpe: 'PASS',
  suspect: 'PASS',
});

describe('buildVerdict', () => {
  it('states passes and failures plainly for the demo run', () => {
    const v = buildVerdict({
      status: 'done',
      hurdles: DEMO,
      sharpeNet: 0.74,
      nTrials: 4,
      kind: 'synthetic',
    });
    expect(v.tone).toBe('fail');
    expect(v.sentence).toBe(
      'Passes the IC-sign, fold-Sharpe, cost and Sharpe gates but fails the decile t-stat gate and deflated Sharpe with 4 trials — not yet distinguishable from luck.',
    );
    expect(v.detail).toMatch(/validates plumbing, not alpha/);
  });

  it('the brief’s example: Sharpe and cost pass, DSR fails', () => {
    const v = buildVerdict({
      status: 'done',
      hurdles: hurdles({
        phase_sharpe: 'PASS',
        breakeven_cost: 'PASS',
        dsr: 'FAIL',
        suspect: 'PASS',
      }),
      nTrials: 4,
    });
    expect(v.sentence).toBe(
      'Passes the cost and Sharpe gates but fails deflated Sharpe with 4 trials — not yet distinguishable from luck.',
    );
  });

  it('a suspicious Sharpe overrides everything', () => {
    const v = buildVerdict({
      status: 'done',
      hurdles: hurdles({ dsr: 'PASS', phase_sharpe: 'PASS', suspect: 'FAIL' }),
      sharpeNet: 2.71,
    });
    expect(v.tone).toBe('suspect');
    expect(v.sentence).toMatch(/^Net Sharpe 2.71 is implausibly high/);
  });

  it('phase 2 below its bar names the ML layer', () => {
    const v = buildVerdict({
      status: 'done',
      hurdles: hurdles({ phase_sharpe: 'FAIL', breakeven_cost: 'PASS', suspect: 'PASS' }, 2),
    });
    expect(v.sentence).toMatch(/fails the Sharpe gate — not enough to justify the ML layer/);
  });

  it('all applied gates passing is admissible', () => {
    const v = buildVerdict({
      status: 'done',
      hurdles: hurdles({ t_stat: 'PASS', dsr: 'PASS', pbo: 'skipped', suspect: 'PASS' }),
      kind: 'store',
    });
    expect(v.tone).toBe('pass');
    expect(v.sentence).toBe(
      'Passes every applied gate — decile t-stat and deflated-Sharpe gates — and is admissible for phase 1.',
    );
    expect(v.detail).toBeNull();
  });

  it('only failures', () => {
    const v = buildVerdict({
      status: 'done',
      hurdles: hurdles({ breakeven_cost: 'FAIL', suspect: 'PASS' }),
    });
    expect(v.sentence).toBe(
      'Fails the cost gate — costs would eat the edge if the model is off by 2×.',
    );
  });

  it('pending and failed runs', () => {
    expect(buildVerdict({ status: 'running', hurdles: null }).tone).toBe('pending');
    expect(buildVerdict({ status: 'failed', hurdles: null }).sentence).toMatch(
      /stopped before evaluation/,
    );
  });

  it('isSuspicious and joinWords', () => {
    expect(isSuspicious(2.01)).toBe(true);
    expect(isSuspicious(2)).toBe(false);
    expect(isSuspicious(null)).toBe(false);
    expect(joinWords(['a', 'b', 'c'])).toBe('a, b and c');
  });
});
