import type { Job } from '@/shared/api';

/** Where a finished task's result lives in the UI (null: nothing to open). */
export function taskHref(j: Pick<Job, 'task' | 'subject' | 'result' | 'run_id'>): string | null {
  const id = j.result?.result_id;
  const rid = typeof id === 'string' ? encodeURIComponent(id) : null;
  switch (j.task) {
    case 'pipeline':
      return j.run_id ? `/runs/${encodeURIComponent(j.run_id)}` : null;
    case 'signal_test':
      return rid ? `/signals/${rid}` : '/signals';
    case 'sweep_member':
    case 'sweep_finalize':
      return j.subject ? `/sweeps/${encodeURIComponent(j.subject)}` : '/sweeps';
    case 'capacity':
      return rid ? `/capacity/${rid}` : '/capacity';
    case 'risk_calibration':
      return rid ? `/risk/${rid}` : '/risk';
    case 'midcap_check':
      return rid ? `/checks/midcap/${rid}` : '/checks/midcap';
    case 'paper_run':
    case 'paper_reconcile':
      return j.subject ? `/paper/${encodeURIComponent(j.subject)}` : '/paper';
    case 'market_hash':
      return null;
  }
}

/** A short name for a task (the queue panel's meta line). */
export const TASK_NAMES: Record<Job['task'], string> = {
  pipeline: 'Run',
  signal_test: 'Signal test',
  sweep_member: 'Sweep member',
  sweep_finalize: 'Sweep PBO',
  capacity: 'Capacity curve',
  risk_calibration: 'Risk calibration',
  midcap_check: 'Mid-cap check',
  paper_run: 'Paper book run',
  paper_reconcile: 'Paper book reconcile',
  market_hash: 'Family hash',
};
