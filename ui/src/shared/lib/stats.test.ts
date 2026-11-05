import { describe, expect, it } from 'vitest';
import {
  annualizedSharpe,
  cagr,
  deflatedSharpe,
  drawdownPath,
  hitRate,
  maxDrawdown,
  normCdf,
  normInv,
} from './stats';

describe('stats', () => {
  it('reproduces the handbook DSR worked example (0.666)', () => {
    const dsr = deflatedSharpe({
      sharpeAnnual: 0.742,
      nObs: 1005,
      nTrials: 4,
      varTrialsAnnual: 0.251,
      skew: -0.029,
      kurtosis: 2.934,
    });
    expect(dsr).toBeCloseTo(0.666, 2);
  });

  it('a single trial gives the PSR against zero (~0.93 in the handbook)', () => {
    const psr = deflatedSharpe({
      sharpeAnnual: 0.742,
      nObs: 1005,
      nTrials: 1,
      varTrialsAnnual: 0.251,
      skew: -0.029,
      kurtosis: 2.934,
    });
    expect(psr).toBeGreaterThan(0.9);
    expect(psr).toBeLessThan(0.95);
  });

  it('normal CDF and inverse agree', () => {
    for (const p of [0.01, 0.25, 0.5, 0.75, 0.99]) expect(normCdf(normInv(p))).toBeCloseTo(p, 4);
  });

  it('computes path statistics', () => {
    const r = [0.1, -0.2, 0.05];
    expect(drawdownPath(r)[1]).toBeCloseTo(-0.2);
    expect(maxDrawdown(r)).toBeCloseTo(-0.2);
    expect(hitRate([0.1, 0, -0.1, 0.2])).toBeCloseTo(2 / 3);
    expect(cagr(Array.from({ length: 252 }, () => 0.0001))).toBeCloseTo(1.0001 ** 252 - 1);
    expect(annualizedSharpe([0.01, 0.01])).toBeNaN();
  });
});
