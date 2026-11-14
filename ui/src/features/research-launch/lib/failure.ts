import type { Job } from '@/shared/api';

/** Plain words for a failed task's kind (stale data is not an error: re-run later). */
export function failureHint(j: Job): string | null {
  if (j.status !== 'failed') return null;
  switch (j.error_kind) {
    case 'stale_data':
      return 'Data not yet published (exit code 3): the vendor has not published the session yet. Nothing changed; run again later and the job catches up day by day.';
    case 'book_locked':
      return 'The book is locked by another job (exit code 4). Wait for it to finish, then run again.';
    case 'coverage':
      return 'Refused by the load-coverage guard: the store prefilter misses names the universe picks on some dates (look-ahead). Use --max-rank 1500 (the default), or allow partial coverage explicitly; that run is flagged.';
    case 'refused':
      return 'Refused by a once-only or seal rule: read the message for what to change.';
    case 'seal':
      return 'Refused by the holdout seal.';
    default:
      return null;
  }
}
