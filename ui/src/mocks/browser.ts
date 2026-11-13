import { setupWorker } from 'msw/browser';
import { db } from './db';
import { handlers } from './handlers';

// The screenshot script slows the mock queue down (`localStorage['qts-mock-job-ms']`) so a
// running job can be captured mid-way; unset, a mock run takes 4 s.
try {
  const ms = Number(localStorage.getItem('qts-mock-job-ms'));
  if (Number.isFinite(ms) && ms > 0) db.jobDurationMs = ms;
} catch {
  /* storage unavailable: keep the default */
}

export const worker = setupWorker(...handlers);
