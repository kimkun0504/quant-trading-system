import type { Ledger, LedgerFamily, LedgerTrial } from '@/shared/api';
import { ledgerFamily, mockDataHash } from './generate';
import { hashString, mulberry32 } from './random';

function hex(seed: string, n: number): string {
  const r = mulberry32(hashString(seed));
  let s = '';
  for (let i = 0; i < n; i++) s += Math.floor(r() * 16).toString(16);
  return s;
}

// The synthetic demo market (300 x 1512, seed 0) and the 2010-2017 store slice.
const DEMO = mockDataHash({ kind: 'synthetic', seed: 0, stocks: 300, days: 1512 });
const STORE = mockDataHash({ kind: 'store', seed: 0, stocks: 1000, days: 2520 });

const rows: [string, 1 | 2, string, number, number, number | null, number][] = [
  // ts, phase, data_hash ('' = written before data hashes: family "legacy"), sharpe, n_obs, mean_ic, n_reruns
  ['2026-11-04T03:47:32Z', 1, '', -0.006, 1005, 0.006, 0],
  ['2026-11-04T03:53:17Z', 1, '', -0.466, 1005, 0.0203, 0],
  ['2026-11-04T04:32:27Z', 1, '', 0.135, 1005, 0.0331, 0],
  ['2026-11-04T04:50:44Z', 2, '', 0.216, 734, 0.0223, 0],
  ['2026-11-04T17:51:59Z', 1, DEMO, 0.742, 1006, 0.0369, 2],
  ['2026-11-03T21:45:10Z', 1, STORE, 1.31, 2008, 0.041, 0],
  ['2026-11-05T09:14:02Z', 2, DEMO, 0.46, 683, 0.028, 0],
  ['2026-11-05T10:22:33Z', 1, DEMO, 2.71, 1007, 0.091, 0],
];

/** Per-family DSR inputs as the API reports them (illustrative values). */
const FAMILY_STATS: Record<string, Omit<LedgerFamily, 'family' | 'n_trials'>> = {
  legacy: { effective_trials: 3.4, var_sr_annual: 0.087 },
  [ledgerFamily(1, DEMO)]: { effective_trials: 1.6, var_sr_annual: 1.97 },
  [ledgerFamily(2, DEMO)]: { effective_trials: 1.0, var_sr_annual: 0 },
  [ledgerFamily(1, STORE)]: { effective_trials: 1.0, var_sr_annual: 0 },
};

export function seedLedger(): Ledger {
  const trials: LedgerTrial[] = rows.map(([ts, phase, dataHash, sharpe, nObs, ic, reruns], i) => {
    const family = ledgerFamily(phase, dataHash);
    return {
      id: `trial-${i + 1}`,
      ts,
      name: family,
      family,
      config_hash: hex(`cfg${i}`, 16),
      data_hash: dataHash,
      sharpe_annual: sharpe,
      n_obs: nObs,
      mean_ic: ic,
      n_reruns: reruns,
    };
  });
  const counts = new Map<string, number>();
  for (const t of trials) counts.set(t.family, (counts.get(t.family) ?? 0) + 1);
  return {
    trials,
    families: [...counts.keys()].sort().map((family) => ({
      family,
      n_trials: counts.get(family) ?? 0,
      ...(FAMILY_STATS[family] ?? { effective_trials: 1, var_sr_annual: 0 }),
    })),
  };
}
