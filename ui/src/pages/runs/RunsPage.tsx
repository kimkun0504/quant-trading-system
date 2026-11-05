import { Link } from 'react-router';
import type { RunSummary } from '@/shared/api';
import { runDescriptor, useRuns } from '@/entities/run';
import { bps, dateTime, num, pct, sharpe } from '@/shared/lib/format';
import { cx } from '@/shared/lib/cx';
import {
  Annotated,
  EmptyState,
  ErrorState,
  LinkButton,
  MarginNote,
  Page,
  Skeleton,
  SortableTable,
  Wide,
} from '@/shared/ui';
import type { Column } from '@/shared/ui';
import styles from './RunsPage.module.css';

const signClass = (v: number | undefined) =>
  v === undefined ? undefined : v > 0 ? styles.gain : v < 0 ? styles.loss : undefined;

const h = (r: RunSummary) => r.headline;

const COLUMNS: Column<RunSummary>[] = [
  {
    key: 'created',
    header: 'Run',
    headerText: 'Run (sorted by creation time)',
    rowHeader: true,
    sortValue: (r) => r.created_at,
    render: (r) => (
      <span className={styles.run}>
        <Link to={`/runs/${encodeURIComponent(r.id)}`} className={styles.runLink}>
          {r.label}
        </Link>
        <span className={styles.descriptor}>
          {runDescriptor(r)}. {dateTime(r.created_at)}
        </span>
      </span>
    ),
  },
  {
    key: 'status',
    header: 'Gates',
    headerText: 'Gates',
    sortValue: (r) => (r.status !== 'done' ? r.status : (h(r)?.overall_gate ?? '')),
    render: (r) => <StatusCell run={r} />,
  },
  {
    key: 'sharpe_net',
    header: 'Net Sharpe',
    headerText: 'Net Sharpe',
    align: 'end',
    sortValue: (r) => h(r)?.sharpe_net ?? null,
    render: (r) => (
      <span className={signClass(h(r)?.sharpe_net)}>
        {h(r) ? sharpe(h(r)?.sharpe_net, { signed: true }) : ''}
        {h(r) && (h(r)?.sharpe_net ?? 0) > 2 ? <span className={styles.flag}> suspect</span> : null}
      </span>
    ),
  },
  {
    key: 'sharpe_gross',
    header: 'Gross Sharpe',
    headerText: 'Gross Sharpe',
    align: 'end',
    sortValue: (r) => h(r)?.sharpe_gross ?? null,
    render: (r) => (h(r) ? sharpe(h(r)?.sharpe_gross, { signed: true }) : ''),
  },
  {
    key: 'cagr_net',
    header: 'Net CAGR',
    headerText: 'Net CAGR',
    align: 'end',
    sortValue: (r) => h(r)?.cagr_net ?? null,
    render: (r) => (
      <span className={signClass(h(r)?.cagr_net)}>
        {h(r) ? pct(h(r)?.cagr_net, { signed: true }) : ''}
      </span>
    ),
  },
  {
    key: 'max_drawdown',
    header: 'Max drawdown',
    headerText: 'Max drawdown',
    align: 'end',
    sortValue: (r) => h(r)?.max_drawdown ?? null,
    render: (r) => (h(r) ? pct(h(r)?.max_drawdown, { signed: true }) : ''),
  },
  {
    key: 'cost',
    header: 'Cost per side',
    headerText: 'Cost per side',
    align: 'end',
    sortValue: (r) => h(r)?.avg_cost_bps_per_side ?? null,
    render: (r) => (h(r) ? bps(h(r)?.avg_cost_bps_per_side) : ''),
  },
  {
    key: 'dsr',
    header: 'DSR',
    headerText: 'Deflated Sharpe',
    align: 'end',
    sortValue: (r) => h(r)?.dsr ?? null,
    render: (r) => (h(r) ? num(h(r)?.dsr, { digits: 3 }) : ''),
  },
];

function StatusCell({ run }: { run: RunSummary }) {
  if (run.status === 'running') return <span className={styles.muted}>Running</span>;
  if (run.status === 'failed') return <span className={styles.muted}>Run failed</span>;
  const pass = run.headline?.overall_gate === 'PASS';
  return (
    <span className={cx(styles.gate, pass ? styles.pass : styles.fail)}>
      <svg width="10" height="10" aria-hidden="true">
        <rect width="10" height="10" rx="2" />
      </svg>
      {pass ? 'Pass' : 'Fail'}
    </span>
  );
}

export function RunsPage() {
  const runs = useRuns();
  return (
    <Page docTitle="Runs" title="Runs">
      <Annotated
        note={
          <MarginNote handbook={{ page: 'validation', section: '9. Deflated Sharpe' }}>
            Sort by deflated Sharpe rather than net Sharpe: it discounts each result by how many
            trials were run to find it.
          </MarginNote>
        }
      >
        <p className={styles.lede}>
          Every backtest in the run registry, newest first. Net figures are after trading costs and
          borrow; the gates column is the plan’s overall verdict.
        </p>
      </Annotated>
      {runs.isPending ? (
        <Wide>
          <div aria-busy="true" aria-label="Loading runs" className={styles.skeleton}>
            {Array.from({ length: 6 }, (_, i) => (
              <Skeleton key={i} height="2.75rem" />
            ))}
          </div>
        </Wide>
      ) : runs.isError ? (
        <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
      ) : runs.data.length === 0 ? (
        <EmptyState
          title="No runs yet"
          action={<LinkButton to="/runs/new">Launch a run</LinkButton>}
        >
          <p>
            Start with a phase-1 composite on the synthetic market: it takes a few minutes and
            checks that every layer of the pipeline is wired.
          </p>
        </EmptyState>
      ) : (
        <Wide>
          <SortableTable
            caption={`${runs.data.length} runs. Select a column heading to sort.`}
            columns={COLUMNS}
            rows={runs.data}
            rowKey={(r) => r.id}
            initialSort={{ key: 'created', dir: 'descending' }}
          />
        </Wide>
      )}
    </Page>
  );
}
