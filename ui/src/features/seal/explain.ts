import type { RunSeal } from '@/shared/api';
import { date } from '@/shared/lib/format';

/** What a run's seal state means, in one sentence (the badge's tooltip). */
export function sealExplanation(s: RunSeal): string {
  if (s.holdout_already_used)
    return 'Holdout already used: an earlier unseal of this configuration looked at the sealed period, so this result is not out-of-sample.';
  switch (s.state) {
    case 'sealed':
      return `Sealed: data through ${date(s.sealed_end)}; the holdout from ${date(s.holdout_start)} was not loaded.`;
    case 'unsealed':
      return `Unsealed: the holdout from ${date(s.holdout_start)} was loaded (reason: ${s.unseal_reason ?? 'n/a'}). This configuration's holdout is used.`;
    case 'exempt':
      return `Exempt from the seal by design (${s.seal_exempt ?? 'exempt'}): the full history was loaded.`;
    case 'none':
      return 'No seal was in force when this run was made.';
  }
}

/** Why a run whose load missed universe names is flagged (`load_coverage.ok` false). */
export function coverageExplanation(s: RunSeal): string {
  const c = s.load_coverage;
  const share =
    c?.share_dates_full != null
      ? ` the screened universe was fully loaded on only ${(100 * c.share_dates_full).toFixed(1)}% of dates;`
      : '';
  return `Partial load: the store prefilter (${c?.prefilter ?? 'unknown'}) missed names the universe picks;${share} the run was loaded anyway (--allow-partial-coverage), so its universe partly depends on how big names became later (look-ahead). Re-run with --max-rank 1500.`;
}
