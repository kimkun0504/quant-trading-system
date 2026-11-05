import { useSearchParams } from 'react-router';
import { JobProgress, LaunchRunForm } from '@/features/launch-run';
import { Annotated, MarginNote, Page } from '@/shared/ui';
import styles from './NewRunPage.module.css';

export function NewRunPage() {
  const [params, setParams] = useSearchParams();
  const jobId = params.get('job');
  const runId = params.get('run');
  return (
    <Page docTitle="New run" title="Launch a run">
      <Annotated
        note={
          <MarginNote handbook={{ page: 'validation', section: '13. The trial ledger' }}>
            Every launch with a new configuration or new data adds a trial to the ledger, which
            raises the bar for the deflated Sharpe of every run in its family. Re-running an
            identical config on the same data, with the same settings and clean code, does not.
          </MarginNote>
        }
      >
        <p className={styles.lede}>
          Runs the full pipeline: universe, risk model, alpha model, backtest with costs, then the
          hurdle gates.
        </p>
        {jobId && runId ? (
          <JobProgress jobId={jobId} runId={runId} />
        ) : (
          <LaunchRunForm onLaunched={(job, run) => setParams({ job, run })} />
        )}
      </Annotated>
    </Page>
  );
}
