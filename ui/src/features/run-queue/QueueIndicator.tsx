import { Link } from 'react-router';
import { useQueue } from '@/entities/run';
import { pct } from '@/shared/lib/format';
import { cx } from '@/shared/lib/cx';
import { Icons } from '@/shared/ui/icons';
import { queueCounts, queueSummary } from './lib';
import styles from './QueueIndicator.module.css';

/**
 * The queue at a glance, in the top bar of every page: how many runs are running and
 * waiting, and how far the running one is. Opens the queue on the Run builder.
 */
export function QueueIndicator() {
  const queue = useQueue();
  const jobs = queue.data?.jobs ?? [];
  const c = queueCounts(jobs);
  const running = jobs.find((j) => j.status === 'running');
  const busy = c.running + c.queued > 0;
  const summary = queue.data ? queueSummary(jobs) : 'Queue';
  return (
    <Link
      to="/builder#queue"
      className={cx(styles.indicator, busy && styles.busy)}
      aria-label={`Run queue: ${summary.toLowerCase()}${running ? `, ${pct(running.progress, { digits: 0 })} done` : ''}`}
      data-testid="queue-indicator"
    >
      <Icons.Queue />
      <span className={styles.text}>{busy ? summary : 'Queue'}</span>
      {running ? (
        <span className={styles.meter} aria-hidden="true">
          <span style={{ width: `${Math.round(running.progress * 100)}%` }} />
        </span>
      ) : null}
    </Link>
  );
}
