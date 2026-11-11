import { useMemo } from 'react';
import type { RunSummary } from '@/shared/api';
import { useRuns } from '@/entities/run';
import { useSelection } from '@/features/selection';

export interface LabRun {
  id: string;
  label: string;
  /** Position in the selection: the series palette index (0 = series 1). */
  index: number;
  summary: RunSummary;
}

/**
 * The runs a lab works on: the selection tray's ids, in order, as finished runs. Ids
 * that are still running, failed or unknown are listed in `missing` so the page can say
 * so rather than show a blank lane.
 */
export function useLabRuns() {
  const ids = useSelection();
  const runs = useRuns();
  return useMemo(() => {
    const byId = new Map((runs.data ?? []).map((r) => [r.id, r]));
    const labRuns: LabRun[] = [];
    const missing: { id: string; status: RunSummary['status'] | 'unknown' }[] = [];
    ids.forEach((id, index) => {
      const s = byId.get(id);
      if (s && s.status === 'done') labRuns.push({ id, label: s.label, index, summary: s });
      else if (runs.data) missing.push({ id, status: s?.status ?? 'unknown' });
    });
    return {
      ids,
      runs: labRuns,
      missing,
      isLoading: runs.isLoading,
      error: runs.error,
      query: runs,
    };
  }, [ids, runs]);
}
