import { useState } from 'react';
import type { Job, JobTask } from '@/shared/api';
import { useCancelJob, useJobLog, useQueue } from '@/entities/run';
import { queueReducer } from '@/features/run-queue';
import { dateTime, pct } from '@/shared/lib/format';
import { Badge, Button, Skeleton } from '@/shared/ui';
import type { BadgeTone } from '@/shared/ui';
import { failureHint } from './lib/failure';
import styles from './research.module.css';

const TONE: Record<Job['status'], BadgeTone> = {
  queued: 'neutral',
  running: 'accent',
  done: 'pass',
  failed: 'fail',
  cancelled: 'skipped',
};
const WORD: Record<Job['status'], string> = {
  queued: 'Queued',
  running: 'Running',
  done: 'Finished',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

function Log({ job }: { job: Job }) {
  const log = useJobLog(job.job_id, { enabled: true, live: job.status === 'running', tail: 80 });
  if (!log.data) return <Skeleton height="4rem" />;
  return (
    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- a scrollable region must be keyboard reachable
    <pre className={styles.log} tabIndex={0} aria-label={`Log of ${job.label}`}>
      {log.data.lines.join('\n') || 'No log lines yet.'}
    </pre>
  );
}

function Row({ job, onCancel }: { job: Job; onCancel: (id: string) => void }) {
  const [open, setOpen] = useState(false);
  const live = job.status === 'running' || job.status === 'queued';
  const hint = failureHint(job);
  return (
    <li className={styles.job} data-job={job.job_id} data-status={job.status}>
      <div className={styles.jobHead}>
        <Badge tone={TONE[job.status]}>{WORD[job.status]}</Badge>
        <span className={styles.jobLabel}>{job.label}</span>
        {live ? (
          <Button
            size="sm"
            variant="quiet"
            onClick={() => onCancel(job.job_id)}
            aria-label={`Cancel ${job.label}`}
          >
            Cancel
          </Button>
        ) : null}
      </div>
      {job.status === 'running' ? (
        <div className={styles.barRow}>
          <progress
            max={1}
            value={job.progress}
            aria-label={`Progress of ${job.label}`}
            className={styles.progress}
          />
          <span>{pct(job.progress, { digits: 0 })}</span>
        </div>
      ) : null}
      <p className={styles.jobMessage} role={job.status === 'failed' ? 'alert' : undefined}>
        {job.message}
      </p>
      {hint ? <p className={styles.jobHint}>{hint}</p> : null}
      <p className={styles.jobMeta}>
        {job.finished_at
          ? dateTime(job.finished_at)
          : job.started_at
            ? `started ${dateTime(job.started_at)}`
            : 'waiting for the worker'}
        {job.status !== 'queued' ? (
          <>
            {' '}
            <button
              type="button"
              className={styles.linkButton}
              aria-expanded={open}
              onClick={() => setOpen((v) => !v)}
            >
              {open ? 'Hide log' : 'Show log'}
            </button>
          </>
        ) : null}
      </p>
      {open && job.status !== 'queued' ? <Log job={job} /> : null}
    </li>
  );
}

/**
 * The background jobs of one research tool (from the shared queue, polled), newest first:
 * progress, message, why a job failed in plain words, the log on demand, cancel.
 */
export function TaskJobs({
  tasks,
  subject,
  limit = 5,
  title = 'Jobs',
}: {
  tasks: readonly JobTask[];
  subject?: string | null;
  limit?: number;
  title?: string;
}) {
  const queue = useQueue();
  const cancel = useCancelJob((jobs, id) => queueReducer(jobs, { type: 'cancel', jobId: id }));
  const jobs = (queue.data?.jobs ?? [])
    .filter((j) => tasks.includes(j.task) && (subject === undefined || j.subject === subject))
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .slice(0, limit);
  if (!jobs.length) return null;
  return (
    <section className={styles.jobs} aria-label={title}>
      <h3 className={styles.jobsTitle}>{title}</h3>
      <ul className={styles.jobList}>
        {jobs.map((j) => (
          <Row key={j.job_id} job={j} onCancel={(id) => cancel.mutate(id)} />
        ))}
      </ul>
    </section>
  );
}
