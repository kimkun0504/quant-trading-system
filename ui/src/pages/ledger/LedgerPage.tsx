import type { LedgerTrial } from '@/shared/api';
import { useLedger } from '@/entities/run';
import { dateTime, int, num, sharpe, shortHash } from '@/shared/lib/format';
import {
  Annotated,
  ErrorState,
  MarginNote,
  Page,
  Section,
  Skeleton,
  SortableTable,
  Wide,
} from '@/shared/ui';
import type { Column } from '@/shared/ui';
import styles from './LedgerPage.module.css';

const COLUMNS: Column<LedgerTrial>[] = [
  {
    key: 'name',
    header: 'Trial',
    headerText: 'Trial',
    rowHeader: true,
    sortValue: (t) => t.name,
    render: (t) => t.name,
  },
  {
    key: 'config',
    header: 'Config hash',
    headerText: 'Config hash',
    sortValue: (t) => t.config_hash,
    render: (t) => shortHash(t.config_hash, 12),
  },
  {
    key: 'data',
    header: 'Data hash',
    headerText: 'Data hash',
    sortValue: (t) => t.data_hash,
    render: (t) =>
      t.data_hash ? shortHash(t.data_hash, 12) : <span className={styles.muted}>not recorded</span>,
  },
  {
    key: 'sharpe',
    header: 'Net Sharpe',
    headerText: 'Net Sharpe',
    align: 'end',
    sortValue: (t) => t.sharpe_annual,
    render: (t) => (
      <span className={t.sharpe_annual >= 0 ? styles.gain : styles.loss}>
        {sharpe(t.sharpe_annual, { signed: true })}
      </span>
    ),
  },
  {
    key: 'ic',
    header: 'Mean IC',
    headerText: 'Mean IC',
    align: 'end',
    sortValue: (t) => t.mean_ic,
    render: (t) => num(t.mean_ic, { digits: 4 }),
  },
  {
    key: 'obs',
    header: 'Days',
    headerText: 'Days',
    align: 'end',
    sortValue: (t) => t.n_obs,
    render: (t) => int(t.n_obs),
  },
  {
    key: 'reruns',
    header: 'Re-runs',
    headerText: 'Re-runs',
    align: 'end',
    sortValue: (t) => t.n_reruns,
    render: (t) => int(t.n_reruns),
  },
  {
    key: 'ts',
    header: 'Created',
    headerText: 'Created',
    sortValue: (t) => t.ts,
    render: (t) => dateTime(t.ts),
  },
];

export function LedgerPage() {
  const ledger = useLedger();
  return (
    <Page docTitle="Trial ledger" title="Trial ledger">
      <Annotated
        note={
          <MarginNote handbook={{ page: 'validation', section: '13. The trial ledger' }}>
            The ledger is append-only and must never be reset: deleting trials would make every
            later Sharpe look more significant than it is.
          </MarginNote>
        }
      >
        <div className={styles.prose}>
          <p>
            Every distinct configuration the pipeline has evaluated, on every dataset. The deflated
            Sharpe ratio reads this ledger: the more trials in a family, and the more their Sharpes
            vary, the higher the Sharpe you would expect from the best of them by luck alone, and
            the higher a run must score to count.
          </p>
          <p>
            Re-running the same config hash on the same data hash updates its record (the re-runs
            column) instead of adding a trial, so reproducing a result never penalises it.
          </p>
        </div>
      </Annotated>
      {ledger.isPending ? (
        <Skeleton height="20rem" />
      ) : ledger.isError ? (
        <ErrorState error={ledger.error} onRetry={() => void ledger.refetch()} />
      ) : (
        <>
          <Section id="families" title="Families">
            <dl className={styles.families}>
              {ledger.data.families.map((f) => (
                <div key={f.family}>
                  <dt>{f.family}</dt>
                  <dd>
                    {f.n_trials} trials, {num(f.effective_trials, { digits: 1 })} effective, Sharpe
                    variance {num(f.var_sr_annual, { digits: 3 })}
                  </dd>
                </div>
              ))}
            </dl>
          </Section>
          <Wide>
            <SortableTable
              caption={`${ledger.data.trials.length} trials, newest first`}
              columns={COLUMNS}
              rows={ledger.data.trials}
              rowKey={(t) => t.id}
              initialSort={{ key: 'ts', dir: 'descending' }}
            />
          </Wide>
        </>
      )}
    </Page>
  );
}
