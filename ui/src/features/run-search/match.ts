import type { RunSummary } from '@/shared/api';

export function matchRuns(runs: readonly RunSummary[], query: string, limit = 8): RunSummary[] {
  const q = query.trim().toLowerCase();
  if (!q) return runs.slice(0, limit);
  const terms = q.split(/\s+/);
  return runs
    .filter((r) => {
      const hay = `${r.label} ${r.id} ${r.config_hash} ${r.kind} phase ${r.phase}`.toLowerCase();
      return terms.every((t) => hay.includes(t));
    })
    .slice(0, limit);
}
