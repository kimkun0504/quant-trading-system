import { useMemo } from 'react';
import { useSearchParams } from 'react-router';
import { useConfigSchema, useOptionalRun, useRuns } from '@/entities/run';
import { RunBuilder, baseFromRun, defaultBase } from '@/features/run-builder';
import { QueuePanel } from '@/features/run-queue';
import { About, Button, ErrorState, Page, Skeleton } from '@/shared/ui';
import styles from './RunBuilderPage.module.css';

/**
 * Run builder: start from the defaults or a run (`?from=<run_id>`), edit the config,
 * read the diff, and queue one or several variants. `?set=key=value` (repeatable) seeds
 * edits, so a link can carry a proposed change.
 */
export function RunBuilderPage() {
  const [params, setParams] = useSearchParams();
  const from = params.get('from');
  const seeds = useMemo(() => params.getAll('set'), [params]);
  const schema = useConfigSchema();
  const runs = useRuns();
  const baseRun = useOptionalRun(from);

  const setFrom = (runId: string | null) =>
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        if (runId) next.set('from', runId);
        else next.delete('from');
        next.delete('set'); // seeded edits belonged to the old base
        return next;
      },
      { replace: true },
    );

  const summary = from ? runs.data?.find((r) => r.id === from) : undefined;
  const base = useMemo(
    () => (from ? (baseRun.data ? baseFromRun(baseRun.data, summary) : null) : defaultBase()),
    [from, baseRun.data, summary],
  );
  const waiting =
    !schema.data || (from !== null && (!baseRun.data || !runs.data) && !baseRun.isError);

  const ready = !schema.isError && !(from !== null && baseRun.isError) && !waiting && base !== null;

  return (
    <Page docTitle="Run builder" title="Run builder">
      <About
        label="About the run builder"
        handbook={{ page: 'validation', section: '13. The trial ledger' }}
      >
        Start from a run you trust, change one thing, and queue it: the diff beside the editor is
        what the comparison will be about. Every new configuration or data set is a trial in the
        ledger and raises the bar for the deflated Sharpe of its whole family; re-running an
        identical configuration on the same data with clean code is not.
      </About>
      {schema.isError ? (
        <ErrorState error={schema.error} onRetry={() => void schema.refetch()} />
      ) : from !== null && baseRun.isError ? (
        <div className={styles.missing}>
          <ErrorState error={baseRun.error} onRetry={() => void baseRun.refetch()} />
          <p>
            Only a finished run can be a base. Start from the defaults, or pick another run there.
          </p>
          <Button onClick={() => setFrom(null)}>Start from the defaults</Button>
        </div>
      ) : waiting || !base ? (
        <div aria-busy="true" aria-label="Loading the run builder" className={styles.loading}>
          <Skeleton height="6rem" />
          <Skeleton height="24rem" />
        </div>
      ) : (
        <RunBuilder
          key={base.runId ?? 'defaults'}
          base={base}
          schema={schema.data}
          seeds={seeds}
          onBaseChange={setFrom}
          aside={<QueuePanel />}
        />
      )}
      {ready ? null : <QueuePanel />}
    </Page>
  );
}
