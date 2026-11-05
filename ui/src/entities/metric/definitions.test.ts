import { describe, expect, it } from 'vitest';
import { METRICS, metric } from './definitions';
import type { MetricContext, MetricKey } from './definitions';

const demo: MetricContext = {
  phase: 1,
  nDays: 1005,
  sharpeNet: 0.74,
  sharpeGross: 1.24,
  cagrGross: 0.082,
  annVolNet: 0.066,
  costBps: 16.3,
  breakevenBps: 46.6,
  icHorizonDays: 20,
  icN: 50,
  icPeriodsPerYear: 12.6,
  icir: 1.51,
  icTStat: 3.02,
  icHitRate: 0.66,
  nTrials: 4,
};

describe('metric definitions', () => {
  it('every metric has a label, definition, formula and handbook link', () => {
    for (const def of Object.values(METRICS)) {
      expect(def.label).not.toBe('');
      expect(def.definition.length).toBeGreaterThan(20);
      expect(def.formula).not.toBe('');
      expect(def.handbook.section).not.toBe('');
    }
  });

  it('every interpret handles missing values', () => {
    for (const key of Object.keys(METRICS) as MetricKey[]) {
      const note = metric(key).interpret(Number.NaN, {});
      expect(note.length).toBeGreaterThan(10);
    }
  });

  it('ICIR 1.51 reads as positive but noisy', () => {
    expect(metric('icir').interpret(1.51, demo)).toBe(
      'ICIR 1.51 — positive but noisy; below ~2 the signal’s sign flips in many periods (the IC was positive on 66% of dates).',
    );
    expect(metric('icir').interpret(2.4, demo)).toMatch(/consistent/);
    expect(metric('icir').interpret(-0.3, demo)).toMatch(/backwards/);
  });

  it('net Sharpe: clears the phase bar but cannot be told from zero in four years', () => {
    const note = metric('sharpe_net').interpret(0.74, demo);
    expect(note).toMatch(/clears the phase-1 bar of 0.30/);
    expect(note).toMatch(/t ≈ 1.5/);
    expect(note).toMatch(/too short/);
  });

  it('net Sharpe: phase 2 bar is 0.5', () => {
    expect(metric('sharpe_net').interpret(0.46, { ...demo, phase: 2 })).toMatch(
      /below the phase-2 bar of 0.50/,
    );
  });

  it('net Sharpe above 2 is called a defect', () => {
    expect(metric('sharpe_net').interpret(2.71, demo)).toMatch(/defect/);
  });

  it('gross Sharpe reports the cost share', () => {
    expect(metric('sharpe_gross').interpret(1.24, demo)).toMatch(/costs take 40% of the Sharpe/);
  });

  it('DSR names the trial count and the 0.95 bar', () => {
    expect(metric('dsr').interpret(0.666, demo)).toBe(
      'DSR 0.666 — below 0.95: with 4 trials this Sharpe is not yet distinguishable from luck.',
    );
    expect(metric('dsr').interpret(0.97, { ...demo, nTrials: 1 })).toMatch(/1 trial\./);
  });

  it('breakeven compares against the 2× margin', () => {
    expect(metric('breakeven_cost_bps').interpret(46.6, demo)).toMatch(
      /2.9× the modelled 16.3 bp, inside/,
    );
    expect(metric('breakeven_cost_bps').interpret(20, demo)).toMatch(/only 1.2×/);
  });

  it('holding below the 50-day target suggests raising trade_aversion', () => {
    expect(metric('effective_holding_days').interpret(30, demo)).toMatch(/raise trade_aversion/);
    expect(metric('effective_holding_days').interpret(58, demo)).toMatch(/above the 50-day/);
  });

  it('realised/ex-ante ratio reads the risk model', () => {
    expect(metric('realized_vs_exante_vol_ratio').interpret(1.04, demo)).toMatch(
      /forecasts this book’s volatility well/,
    );
    expect(metric('realized_vs_exante_vol_ratio').interpret(1.3, demo)).toMatch(/under-predicts/);
  });

  it('IC t-stat flags the ×4.5 annualisation bug', () => {
    // demo_phase2: ICIR 3.51 annualised with 252 instead of 12.6, t = 1.33, n ≈ 36
    expect(
      metric('ic_t_stat').interpret(1.33, { icir: 3.51, icPeriodsPerYear: 12.6, icN: 36 }),
    ).toMatch(/×4.5 bug/);
    expect(metric('ic_t_stat').interpret(3.02, demo)).not.toMatch(/bug/);
  });

  it('decile t contrasts with the residualised IC', () => {
    expect(metric('decile_t').interpret(0.74, demo)).toMatch(
      /fails the t > 3 gate.*unhedged factor noise/,
    );
  });

  it('PBO skipped explains how to compute it', () => {
    expect(metric('pbo').interpret(null, demo)).toMatch(/pipeline_pbo_configs/);
    expect(metric('pbo').format(null)).toBe('skipped');
    expect(metric('pbo').interpret(0.28, demo)).toMatch(/beats a coin flip/);
  });

  it('hit rate above 56% is suspicious', () => {
    expect(metric('hit_rate_net').interpret(0.57, demo)).toMatch(/leakage/);
  });

  it('volatility below target explains the leverage cap', () => {
    expect(metric('ann_vol_net').interpret(0.066, demo)).toMatch(/gross-leverage cap binds/);
  });
});
