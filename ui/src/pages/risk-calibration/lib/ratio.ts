import type { RiskCalibrationResult } from '@/shared/api';
import { num } from '@/shared/lib/format';

export const nuText = (v: number | null | undefined) => num(v, { digits: 2 });

/** Years in order across every ν (a year a ν lacks leaves a gap). */
export function ratioLines(r: Pick<RiskCalibrationResult, 'results'>) {
  const years = [...new Set(r.results.flatMap((x) => Object.keys(x.by_year)))].sort();
  return {
    years,
    series: r.results.map((x, i) => ({
      label: `ν ${nuText(x.nu_scale)}`,
      paletteIndex: i,
      values: years.map((y) => x.by_year[y]?.ratio ?? null),
    })),
  };
}
