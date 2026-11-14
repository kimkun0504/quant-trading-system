import { Link } from 'react-router';
import type { PaperBookSummary } from '@/shared/api';
import { HANDBOOK_PAPER, usePaperBooks } from '@/entities/research';
import { Stat, layout } from '@/features/research-launch';
import { currency, date, int } from '@/shared/lib/format';
import {
  About,
  Badge,
  EmptyState,
  ErrorState,
  Page,
  Panel,
  Skeleton,
  SortableTable,
} from '@/shared/ui';
import type { Column } from '@/shared/ui';
import { LEVEL_TONE, isLevel } from './lib/series';

const COLUMNS: Column<PaperBookSummary>[] = [
  {
    key: 'name',
    header: 'Book',
    headerText: 'Book',
    rowHeader: true,
    sortValue: (b) => b.name,
    render: (b) => (
      <Link to={`/paper/${encodeURIComponent(b.name)}`} className={layout.rowLink}>
        {b.name}
      </Link>
    ),
  },
  {
    key: 'level',
    header: 'Status',
    headerText: 'Status',
    sortValue: (b) => b.level,
    render: (b) => (
      <>
        {isLevel(b.level) ? (
          <Badge tone={LEVEL_TONE[b.level]}>{b.level}</Badge>
        ) : (
          <Badge tone="neutral">no day yet</Badge>
        )}
        {b.trading_halted ? <Badge tone="fail">halted</Badge> : null}
      </>
    ),
  },
  {
    key: 'mode',
    header: <Stat k="liveMode">Last day</Stat>,
    headerText: 'Last day mode',
    render: (b) =>
      b.last_mode ? (
        <Badge tone={b.last_mode === 'live' ? 'accent' : 'skipped'}>{b.last_mode}</Badge>
      ) : (
        'n/a'
      ),
  },
  {
    key: 'last',
    header: 'Through',
    headerText: 'Last date',
    sortValue: (b) => b.last_date ?? '',
    render: (b) => date(b.last_date),
  },
  {
    key: 'days',
    header: 'Days',
    headerText: 'Days',
    align: 'end',
    sortValue: (b) => b.n_days,
    render: (b) => int(b.n_days),
  },
  {
    key: 'nav',
    header: 'NAV',
    headerText: 'NAV',
    align: 'end',
    sortValue: (b) => b.nav ?? 0,
    render: (b) => currency(b.nav),
  },
  { key: 'sizing', header: 'Sizing', headerText: 'Sizing', render: (b) => b.sizing },
  { key: 'start', header: 'Started', headerText: 'Start', render: (b) => date(b.start) },
];

export function PaperPage() {
  const q = usePaperBooks();
  return (
    <Page docTitle="Paper trading" title="Paper trading">
      <About label="About paper books" handbook={HANDBOOK_PAPER}>
        A paper book runs the frozen strategy one day at a time on real data, as it would trade: it
        sizes orders before the close, fills them at the close, and compares what happens with what
        the backtest said should happen. Status checks flag WATCH or STOP when the book drifts
        outside the backtest&apos;s range; only live days count as evidence.
      </About>
      <Panel title="Books" subtitle={q.data ? `under ${q.data.paper_root}` : undefined} flush>
        {q.isPending ? (
          <Skeleton height="8rem" />
        ) : q.isError ? (
          <ErrorState error={q.error} onRetry={() => void q.refetch()} />
        ) : q.data.books.length === 0 ? (
          <EmptyState title="No paper book yet">
            Create one on the command line:{' '}
            <code>
              qts paper init --name main --config configs/baseline.yaml --start YYYY-MM-DD
              --max-rank 1500 --max-rank 1500
            </code>
            , then run it here.
          </EmptyState>
        ) : (
          <SortableTable
            caption={`${q.data.books.length} paper books`}
            columns={COLUMNS}
            rows={q.data.books}
            rowKey={(b) => b.name}
          />
        )}
      </Panel>
    </Page>
  );
}
