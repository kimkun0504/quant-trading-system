import { useJob } from '@/entities/run';
import { pct } from '@/shared/lib/format';
import { ErrorState, LinkButton } from '@/shared/ui';
import styles from './JobProgress.module.css';

/** Polls /api/jobs/{id} until the run finishes, then offers the run page. */
export function JobProgress({ jobId, runId }: { jobId: string; runId: string }) {
  const job = useJob(jobId);
  if (job.isError) return <ErrorState error={job.error} onRetry={() => void job.refetch()} />;
  const data = job.data;
  const progress = data?.progress ?? 0;
  const done = data?.status === 'done';
  const failed = data?.status === 'failed';
  return (
    <section className={styles.job} aria-labelledby="job-title">
      <h2 id="job-title" className={styles.title}>
        {done ? 'Run finished' : failed ? 'Run failed' : 'Running the pipeline'}
      </h2>
      <div className={styles.barRow}>
        <progress
          className={styles.progress}
          max={1}
          value={progress}
          aria-labelledby="job-title"
        />
        <span className={styles.pct}>{pct(progress, { digits: 0 })}</span>
      </div>
      <p className={styles.message} aria-live="polite">
        {data?.message ?? 'Queued'}
      </p>
      <p className={styles.ids}>
        Job {jobId}, run {runId}
      </p>
      {done ? (
        <div>
          <LinkButton to={`/runs/${encodeURIComponent(runId)}`}>Open the run</LinkButton>
        </div>
      ) : null}
    </section>
  );
}
