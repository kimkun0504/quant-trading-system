import { useState } from 'react';
import { Link } from 'react-router';
import type { Job } from '@/shared/api';
import { useCancelJob, useJobLog, useMoveJob, useQueue } from '@/entities/run';
import { MAX_SELECTED, toggleSelected, useSelection } from '@/features/selection';
import { dateTime, pct } from '@/shared/lib/format';
import { cx } from '@/shared/lib/cx';
import { Badge, Button, ErrorState, IconButton, Panel, Skeleton } from '@/shared/ui';
import type { BadgeTone } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import {
  STAGE_ORDER,
  formatDuration,
  jobStatusLine,
  logLineText,
  queueReducer,
  queueSummary,
  stageLabel,
} from './lib';
import styles from './QueuePanel.module.css';

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

function JobLogView({ job }: { job: Job }) {
  const log = useJobLog(job.job_id, { enabled: true, live: job.status === 'running', tail: 60 });
  if (log.isError) return <p className={styles.logEmpty}>The log could not be read.</p>;
  if (!log.data) return <Skeleton height="4rem" />;
  if (log.data.lines.length === 0) return <p className={styles.logEmpty}>No log lines yet.</p>;
  return (
    <pre
      className={styles.log}
      // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- a scrollable region must be keyboard reachable
      tabIndex={0}
      aria-label={`Log of ${job.label}${log.data.truncated ? `, last ${log.data.lines.length} of ${log.data.total_lines} lines` : ''}`}
    >
      {log.data.lines.join('\n')}
    </pre>
  );
}

function StageRail({ stage }: { stage: string }) {
  const at = STAGE_ORDER.indexOf(stage);
  return (
    <ol className={styles.rail} aria-label="Pipeline stages">
      {STAGE_ORDER.map((s, i) => (
        <li
          key={s}
          className={cx(i < at && styles.past, i === at && styles.now)}
          aria-current={i === at ? 'step' : undefined}
        >
          {stageLabel(s)}
        </li>
      ))}
    </ol>
  );
}

function JobItem({
  job,
  queued,
  onCancel,
  onMove,
}: {
  job: Job;
  /** Number of queued jobs (for the move buttons). */
  queued: number;
  onCancel: (id: string) => void;
  onMove: (id: string, position: number) => void;
}) {
  const [showLog, setShowLog] = useState(false);
  const selection = useSelection();
  const inTray = selection.includes(job.run_id);
  const running = job.status === 'running';
  const waiting = job.status === 'queued';
  const pos = job.position ?? 0;
  return (
    <li
      className={cx(styles.job, styles[job.status])}
      data-job={job.job_id}
      data-status={job.status}
    >
      <div className={styles.jobHead}>
        <Badge tone={TONE[job.status]}>{WORD[job.status]}</Badge>
        {job.status === 'done' ? (
          <Link to={`/runs/${encodeURIComponent(job.run_id)}`} className={styles.jobLabel}>
            {job.label}
          </Link>
        ) : (
          <span className={styles.jobLabel}>{job.label}</span>
        )}
        <span className={styles.jobActions}>
          {waiting ? (
            <>
              <IconButton
                size="sm"
                label={`Move ${job.label} up`}
                disabled={pos === 0}
                onClick={() => onMove(job.job_id, pos - 1)}
              >
                <Icons.ArrowUp />
              </IconButton>
              <IconButton
                size="sm"
                label={`Move ${job.label} down`}
                disabled={pos >= queued - 1}
                onClick={() => onMove(job.job_id, pos + 1)}
              >
                <Icons.ArrowDown />
              </IconButton>
            </>
          ) : null}
          {running || waiting ? (
            <Button
              size="sm"
              variant="quiet"
              onClick={() => onCancel(job.job_id)}
              aria-label={`Cancel ${job.label}`}
            >
              Cancel
            </Button>
          ) : null}
          {job.status === 'done' ? (
            <Button
              size="sm"
              variant="quiet"
              aria-pressed={inTray}
              disabled={!inTray && selection.length >= MAX_SELECTED}
              onClick={() => toggleSelected(job.run_id)}
              aria-label={
                inTray ? `Remove ${job.label} from the tray` : `Add ${job.label} to the tray`
              }
            >
              {inTray ? 'In tray' : 'Add to tray'}
            </Button>
          ) : null}
        </span>
      </div>
      {running ? (
        <>
          <div className={styles.barRow}>
            <progress
              className={styles.progress}
              max={1}
              value={job.progress}
              aria-label={`Progress of ${job.label}`}
            />
            <span className={styles.pct}>{pct(job.progress, { digits: 0 })}</span>
          </div>
          <StageRail stage={job.stage} />
        </>
      ) : null}
      <p className={styles.statusLine} aria-live={running ? 'off' : undefined}>
        {jobStatusLine(job)}
      </p>
      {running && job.last_log && !job.message.startsWith(logLineText(job.last_log)) ? (
        <p className={styles.lastLog} title="Last line of the job's log">
          {logLineText(job.last_log)}
        </p>
      ) : null}
      <p className={styles.jobMeta}>
        {job.ml ? 'Composite + ML' : 'Composite'},{' '}
        {job.kind === 'synthetic' ? 'synthetic market' : 'data store'}
        {job.n_overrides
          ? `, ${job.n_overrides} ${job.n_overrides === 1 ? 'override' : 'overrides'}`
          : ''}
        {job.finished_at ? `, ${dateTime(job.finished_at)}` : ''}
        {job.status === 'done' && job.started_at && job.finished_at
          ? `, took ${formatDuration((Date.parse(job.finished_at) - Date.parse(job.started_at)) / 1000).replace('about ', '')}`
          : ''}
        {!waiting ? (
          <>
            {' '}
            <button
              type="button"
              className={styles.logToggle}
              aria-expanded={showLog}
              onClick={() => setShowLog((v) => !v)}
            >
              {showLog ? 'Hide log' : 'Show log'}
            </button>
          </>
        ) : null}
      </p>
      {showLog && !waiting ? <JobLogView job={job} /> : null}
    </li>
  );
}

/**
 * The run queue: the job that is running (stage, progress, last log line, time left),
 * the jobs waiting in order (move, cancel) and the most recent finished ones (open the
 * dossier, add to the tray). One job runs at a time.
 */
export function QueuePanel({ recent = 6 }: { recent?: number }) {
  const queue = useQueue();
  const cancel = useCancelJob((jobs, jobId) => queueReducer(jobs, { type: 'cancel', jobId }));
  const move = useMoveJob((jobs, jobId, position) =>
    queueReducer(jobs, { type: 'move', jobId, position }),
  );
  const jobs = queue.data?.jobs ?? [];
  const active = jobs.filter((j) => j.status === 'running' || j.status === 'queued');
  const finished = jobs.filter((j) => j.status !== 'running' && j.status !== 'queued');
  const nQueued = jobs.filter((j) => j.status === 'queued').length;
  const [all, setAll] = useState(false);
  const shown = all ? finished : finished.slice(0, recent);

  return (
    <Panel
      id="queue"
      title="Queue"
      subtitle={queue.data ? queueSummary(jobs).toLowerCase() : undefined}
      flush
      aria-label="Run queue"
      className={styles.panel}
    >
      {queue.isError ? (
        <div className={styles.pad}>
          <ErrorState error={queue.error} onRetry={() => void queue.refetch()} />
        </div>
      ) : !queue.data ? (
        <div className={styles.pad}>
          <Skeleton height="3rem" />
        </div>
      ) : jobs.length === 0 ? (
        <p className={styles.empty}>
          Nothing is queued. Runs you add wait here and run one at a time, in order.
        </p>
      ) : (
        <>
          {active.length ? (
            <ol className={styles.list} aria-label="Running and queued">
              {active.map((j) => (
                <JobItem
                  key={j.job_id}
                  job={j}
                  queued={nQueued}
                  onCancel={(id) => cancel.mutate(id)}
                  onMove={(jobId, position) => move.mutate({ jobId, position })}
                />
              ))}
            </ol>
          ) : (
            <p className={styles.empty}>Nothing is running or waiting.</p>
          )}
          {shown.length ? (
            <>
              <h3 className={styles.sub}>Recently finished</h3>
              <ul className={styles.list} aria-label="Recently finished">
                {shown.map((j) => (
                  <JobItem
                    key={j.job_id}
                    job={j}
                    queued={nQueued}
                    onCancel={(id) => cancel.mutate(id)}
                    onMove={(jobId, position) => move.mutate({ jobId, position })}
                  />
                ))}
              </ul>
              {finished.length > recent ? (
                <div className={styles.more}>
                  <Button size="sm" variant="quiet" onClick={() => setAll((v) => !v)}>
                    {all ? 'Show fewer' : `Show all ${finished.length}`}
                  </Button>
                </div>
              ) : null}
            </>
          ) : null}
        </>
      )}
    </Panel>
  );
}
