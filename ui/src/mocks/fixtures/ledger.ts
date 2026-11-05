import type { Ledger, LedgerTrial } from '@/shared/api';
import { hashString, mulberry32 } from './random';

function hex(seed: string, n: number): string {
  const r = mulberry32(hashString(seed));
  let s = '';
  for (let i = 0; i < n; i++) s += Math.floor(r() * 16).toString(16);
  return s;
}

const rows: [string, string, string, number, number, number | null, number][] = [
  // ts, name, family, sharpe, n_obs, mean_ic, n_reruns
  ['2026-11-04T03:47:32Z', 'pipeline_phase1', 'pipeline_phase1', -0.006, 1005, 0.006, 0],
  ['2026-11-04T03:53:17Z', 'pipeline_phase1', 'pipeline_phase1', -0.466, 1005, 0.0203, 0],
  ['2026-11-04T04:32:27Z', 'pipeline_phase1', 'pipeline_phase1', 0.135, 1005, 0.0331, 1],
  ['2026-11-04T04:50:44Z', 'pipeline_phase2', 'pipeline_phase2', 0.216, 734, 0.0223, 0],
  ['2026-11-04T17:51:59Z', 'pipeline_phase1', 'pipeline_phase1', 0.742, 1005, 0.0369, 2],
  ['2026-11-03T21:45:10Z', 'pipeline_phase1', 'pipeline_phase1', 1.31, 2008, 0.041, 0],
  ['2026-11-05T09:14:02Z', 'pipeline_phase2', 'pipeline_phase2', 0.46, 1000, 0.028, 0],
  ['2026-11-05T10:22:33Z', 'pipeline_phase1', 'pipeline_phase1', 2.71, 1000, 0.091, 0],
];

export function seedLedger(): Ledger {
  const trials: LedgerTrial[] = rows.map(([ts, name, family, sharpe, nObs, ic, reruns], i) => ({
    id: `trial-${i + 1}`,
    ts,
    name,
    family,
    config_hash: hex(`cfg${i}`, 16),
    data_hash: i < 3 ? '' : hex(`data${family}${nObs}`, 16),
    sharpe_annual: sharpe,
    n_obs: nObs,
    mean_ic: ic,
    n_reruns: reruns,
  }));
  return {
    trials,
    families: [
      { family: 'pipeline_phase1', n_trials: 6, effective_trials: 4.0, var_sr_annual: 0.251 },
      { family: 'pipeline_phase2', n_trials: 2, effective_trials: 2.0, var_sr_annual: 0.18 },
    ],
  };
}
