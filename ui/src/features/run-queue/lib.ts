import type { Job } from '@/shared/api';

/** Optimistic edits of the queue listing, applied before the API confirms them. */
export type QueueAction =
  | { type: 'cancel'; jobId: string; now?: string }
  | { type: 'move'; jobId: string; position: number }
  | { type: 'upsert'; job: Job };

const ORDER: Record<Job['status'], number> = {
  running: 0,
  queued: 1,
  done: 2,
  failed: 2,
  cancelled: 2,
};

/** The API's listing order: running, the queue by position, then finished jobs newest first. */
export function sortQueue(jobs: readonly Job[]): Job[] {
  return [...jobs].sort((a, b) => {
    const d = ORDER[a.status] - ORDER[b.status];
    if (d !== 0) return d;
    if (a.status === 'queued') return (a.position ?? 0) - (b.position ?? 0);
    if (a.status === 'running') return (a.started_at ?? '').localeCompare(b.started_at ?? '');
    return (b.finished_at ?? '').localeCompare(a.finished_at ?? '');
  });
}

function renumber(jobs: readonly Job[]): Job[] {
  let i = 0;
  return jobs.map((j) =>
    j.status === 'queued'
      ? { ...j, position: i++, message: i === 1 ? 'Queued' : `Queued (${i - 1} ahead)` }
      : j.position === null
        ? j
        : { ...j, position: null },
  );
}

/**
 * Pure queue transitions: cancelling takes a job out of the queue (or stops the running
 * one), moving re-orders the waiting jobs, upsert merges a job the API just returned.
 * Finished jobs never change.
 */
export function queueReducer(jobs: readonly Job[], action: QueueAction): Job[] {
  switch (action.type) {
    case 'cancel': {
      const target = jobs.find((j) => j.job_id === action.jobId);
      if (!target || (target.status !== 'queued' && target.status !== 'running')) return [...jobs];
      const next = jobs.map((j): Job => {
        if (j.job_id !== action.jobId || (j.status !== 'queued' && j.status !== 'running'))
          return j;
        return {
          ...j,
          status: 'cancelled',
          stage: '',
          eta_seconds: null,
          message: j.status === 'queued' ? 'Cancelled before it started' : 'Cancelled',
          finished_at: action.now ?? j.finished_at ?? new Date().toISOString(),
        };
      });
      return renumber(sortQueue(next));
    }
    case 'move': {
      const queued = sortQueue(jobs.filter((j) => j.status === 'queued'));
      const at = queued.findIndex((j) => j.job_id === action.jobId);
      if (at < 0) return [...jobs];
      const [job] = queued.splice(at, 1);
      if (!job) return [...jobs];
      queued.splice(Math.max(0, Math.min(action.position, queued.length)), 0, job);
      const placed = new Map(queued.map((j, i) => [j.job_id, i]));
      return renumber(
        sortQueue(
          jobs.map((j) =>
            placed.has(j.job_id) ? { ...j, position: placed.get(j.job_id) ?? 0 } : j,
          ),
        ),
      );
    }
    case 'upsert': {
      const has = jobs.some((j) => j.job_id === action.job.job_id);
      const next = has
        ? jobs.map((j) => (j.job_id === action.job.job_id ? action.job : j))
        : [...jobs, action.job];
      return sortQueue(next);
    }
  }
}

export const isActive = (j: Pick<Job, 'status'>): boolean =>
  j.status === 'queued' || j.status === 'running';

export interface QueueCounts {
  running: number;
  queued: number;
  finished: number;
}

export function queueCounts(jobs: readonly Job[]): QueueCounts {
  const c: QueueCounts = { running: 0, queued: 0, finished: 0 };
  for (const j of jobs) {
    if (j.status === 'running') c.running++;
    else if (j.status === 'queued') c.queued++;
    else c.finished++;
  }
  return c;
}

/** "1 running, 2 queued", "2 queued", "Queue is empty". */
export function queueSummary(jobs: readonly Job[]): string {
  const c = queueCounts(jobs);
  const parts = [c.running ? `${c.running} running` : '', c.queued ? `${c.queued} queued` : ''];
  return parts.filter(Boolean).join(', ') || 'Queue is empty';
}

/** "about 40 s", "about 3 min", "about 1 h 20 min"; '' when unknown. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0)
    return '';
  if (seconds < 55) return `about ${Math.max(5, Math.round(seconds / 5) * 5)} s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `about ${minutes} min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m ? `about ${h} h ${m} min` : `about ${h} h`;
}

const STAGES: Record<string, string> = {
  start: 'Starting',
  data: 'Market data',
  universe: 'Universe',
  risk: 'Risk model',
  alpha: 'Alpha model',
  backtest: 'Backtest',
  evaluate: 'Gates',
  write: 'Writing',
};

/** The pipeline stages in order, for the stage rail of a running job. */
export const STAGE_ORDER: readonly string[] = [
  'data',
  'universe',
  'risk',
  'alpha',
  'backtest',
  'evaluate',
  'write',
];

export function stageLabel(stage: string): string {
  return STAGES[stage] ?? (stage ? stage : '');
}

/** The log line without its `HH:MM:SS logger:` prefix. */
export function logLineText(line: string | null | undefined): string {
  if (!line) return '';
  return line.replace(/^\d{2}:\d{2}:\d{2}\s+[\w.]+:\s*/, '');
}

/** One line under a job's name: what it is doing, or how it ended. */
export function jobStatusLine(j: Job): string {
  if (j.status === 'running') {
    const eta = formatDuration(j.eta_seconds);
    return [j.message, eta ? `${eta} left` : ''].filter(Boolean).join(', ');
  }
  if (j.status === 'queued') {
    const est = formatDuration(j.estimated_seconds);
    const place = j.position === 0 ? 'Next to run' : `${j.position ?? 0} ahead`;
    return est ? `${place}, takes ${est}` : place;
  }
  return j.message;
}
