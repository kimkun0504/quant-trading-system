import { useNavigate, useParams } from 'react-router';
import type { PaperHistoryRow, PaperStatus } from '@/shared/api';
import {
  HANDBOOK_PAPER,
  usePaperBooks,
  usePaperHistory,
  usePaperHoldings,
  usePaperOrders,
  usePaperStatus,
} from '@/entities/research';
import { Facts, SelectField, Stat, layout } from '@/features/research-launch';
import { bps, currency, date, int, num, pct, sharpe, shortHash } from '@/shared/lib/format';
import {
  About,
  Badge,
  EmptyState,
  ErrorState,
  Page,
  Panel,
  Skeleton,
  SortableTable,
  Table,
  Td,
  Th,
} from '@/shared/ui';
import type { Column } from '@/shared/ui';
import { AckPanel, PaperRunPanel } from './PaperActions';
import { PaperCharts } from './PaperCharts';
import { PaperSchedule } from './PaperSchedule';
import { LEVEL_TEXT, LEVEL_TONE, isLevel, modeCounts } from './lib/series';
import styles from './Paper.module.css';

export function ModeBadge({ mode }: { mode: string | null | undefined }) {
  if (mode === 'live') return <Badge tone="accent">live</Badge>;
  if (mode === 'backfill') return <Badge tone="skipped">backfill</Badge>;
  return <Badge tone="neutral">{mode ?? 'n/a'}</Badge>;
}

const RULE_NAMES: Record<string, string> = {
  drawdown: 'Drawdown',
  sharpe: 'Sharpe',
  cumulative_return: 'Cumulative return',
  vol_ratio: 'Realised / ex-ante vol',
  turnover: 'Turnover',
  cost_per_side: 'Cost per side',
  gross: 'Gross exposure',
  optimizer: 'Optimizer',
  data_quality: 'Data quality',
  realised_ic: 'Realised IC',
};

type DataSpec = PaperStatus['header']['data'];

function prefilterText(d: DataSpec): string {
  if (!d) return 'not recorded';
  if (d.max_rank != null) return `--max-rank ${d.max_rank}`;
  const parts = [
    d.max_names != null ? `--max-names ${d.max_names}` : '',
    d.min_mcap_ever != null ? `--min-mcap-ever ${d.min_mcap_ever}` : '',
  ].filter(Boolean);
  return parts.join(' ') || 'none (every id)';
}

function prefilterNote(d: DataSpec): string | undefined {
  if (!d) return undefined;
  if (d.max_rank == null && (d.max_names != null || d.min_mcap_ever != null))
    return 'peak-cap prefilter: misses universe names, so the load-coverage guard refuses it; re-create the book with --max-rank 1500';
  return 'covers the universe by construction';
}

export function PaperBookPage() {
  const { book = '' } = useParams();
  const q = usePaperStatus(book);
  const books = usePaperBooks();
  const navigate = useNavigate();
  const rootFlag = books.data?.paper_root_flag ?? null;
  const picker =
    books.data && books.data.books.length > 1 ? (
      <div className={styles.picker}>
        <SelectField
          name="book"
          label="Book"
          value={book}
          options={books.data.books.map((b) => ({
            value: b.name,
            label: `${b.name} (${b.level})`,
          }))}
          onChange={(v) => void navigate(`/paper/${encodeURIComponent(v)}`)}
        />
      </div>
    ) : null;
  return (
    <Page docTitle={`Paper book ${book}`} title={`Paper book: ${book}`} actions={picker}>
      {q.isPending ? (
        <Skeleton height="24rem" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : (
        <Body s={q.data} rootFlag={rootFlag} />
      )}
    </Page>
  );
}

function Body({ s, rootFlag }: { s: PaperStatus; rootFlag: string | null }) {
  const st = s.status;
  const level = isLevel(st.level) ? st.level : null;
  const latest = s.latest;
  const m = st.metrics;
  const counts = modeCounts(s.series.mode);
  return (
    <>
      <section className={layout.context} aria-label="Book status" data-testid="paper-status">
        <span className={styles.status}>
          <span className={layout.contextLabel}>Status</span>
          <span className={styles.level}>
            {level ? (
              <Badge tone={LEVEL_TONE[level]}>{level}</Badge>
            ) : (
              <Badge tone="neutral">no day yet</Badge>
            )}
            {st.acknowledged ? <Badge tone="neutral">acknowledged</Badge> : null}
            {st.trading_halted ? <Badge tone="fail">trading halted</Badge> : null}
          </span>
        </span>
        <span className={layout.lede}>
          {level ? LEVEL_TEXT[level] : 'The book has no processed day yet.'}
          {st.reasons.length ? (
            <ul className={styles.reasons}>
              {st.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          ) : null}
        </span>
        <span>
          <span className={layout.contextLabel}>
            <Stat k="liveMode">Latest day</Stat>
          </span>
          <br />
          {latest ? (
            <>
              {date(latest.date)} <ModeBadge mode={latest.mode} />
              <br />
              <span className={layout.small}>data as of {date(latest.data_asof)}</span>
            </>
          ) : (
            'none'
          )}
        </span>
        <span>
          <span className={layout.contextLabel}>Days</span>
          <br />
          <span className={layout.contextBig}>{int(s.n_days)}</span>
          <br />
          <span className={layout.small}>
            {counts.live} live, {counts.backfill} backfill
          </span>
        </span>
      </section>
      {counts.live === 0 && s.n_days > 0 ? (
        <div className={layout.notice} role="note">
          <span>
            <Badge tone="skipped">Backfill only</Badge> Every day of this book was generated after
            the fact. It checks the pipeline, but only live days (orders made before the cutoff)
            count as paper-trading evidence.
          </span>
        </div>
      ) : null}
      <AckPanel s={s} rootFlag={rootFlag} />
      <Facts
        label="Book headline"
        facts={[
          {
            key: 'nav',
            label: 'NAV',
            value: currency(latest?.nav ?? null),
            note: `started at ${currency(s.header.nav0)} on ${date(s.header.first_execution)}`,
          },
          {
            key: 'ret',
            label: 'Cumulative return',
            value: pct(m.cum_return, { digits: 2, signed: true }),
          },
          { key: 'dd', label: 'Max drawdown', value: pct(m.max_drawdown, { digits: 2 }) },
          {
            key: 'sharpe',
            label: 'Sharpe so far',
            value: sharpe(m.sharpe, { signed: true }),
            note: 'noisy before 63 days',
          },
          {
            key: 'vol',
            label: <Stat k="volRatio" />,
            value: num(m.vol_ratio, { digits: 2 }),
            note: `backtest ${num(m.vol_ratio_backtest, { digits: 2 })}`,
          },
          {
            key: 'cost',
            label: <Stat k="costPerSide" />,
            value: bps(m.cost_bps_per_side),
            note: '21-day mean',
          },
        ]}
      />
      <Panel title="Checks" subtitle="each rule's value against its threshold" flush>
        <Table label="Status checks">
          <thead>
            <tr>
              <Th>Rule</Th>
              <Th>Level</Th>
              <Th numeric>Value</Th>
              <Th numeric>Threshold</Th>
              <Th>What it says</Th>
            </tr>
          </thead>
          <tbody>
            {st.checks.map((c) => (
              <tr key={c.rule} className={styles.checkRow} data-level={c.level}>
                <Td>{RULE_NAMES[c.rule] ?? c.rule}</Td>
                <Td>
                  {isLevel(c.level) ? <Badge tone={LEVEL_TONE[c.level]}>{c.level}</Badge> : c.level}
                </Td>
                <Td numeric>{num(c.value, { digits: 4 })}</Td>
                <Td numeric>{num(c.threshold, { digits: 4 })}</Td>
                <Td className={layout.wrapCell}>{c.message}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Panel>
      {s.series.date.length > 1 ? <PaperCharts s={s} /> : null}
      <div className={layout.split}>
        <div className={layout.stack}>
          <OrdersPanel book={s.book} />
          <HoldingsPanel book={s.book} />
          <HistoryPanel book={s.book} />
        </div>
        <div className={layout.stack}>
          <PaperRunPanel s={s} rootFlag={rootFlag} />
          <Panel title="Frozen at creation">
            <Facts
              label="Book definition"
              facts={[
                {
                  key: 'cfg',
                  label: 'Config',
                  value: <code>{shortHash(s.header.config_hash)}</code>,
                  note: s.header.config_source,
                },
                {
                  key: 'sizing',
                  label: 'Sizing',
                  value: s.header.sizing,
                  note:
                    s.header.sizing === 'replica'
                      ? 'batch-identical pipeline check'
                      : 'market-on-close, live-tradable',
                },
                {
                  key: 'ml',
                  label: 'Model',
                  value: s.header.use_ml ? 'with the ML sleeve' : 'composite only',
                },
                {
                  key: 'ref',
                  label: 'Expectations from',
                  value: <code>{s.header.reference_run ?? 'none'}</code>,
                },
                {
                  key: 'data',
                  label: 'Store prefilter',
                  value: <code>{prefilterText(s.header.data)}</code>,
                  note: prefilterNote(s.header.data),
                },
              ]}
            />
          </Panel>
        </div>
      </div>
      <PaperSchedule book={s.book} />
      <About label="How the status is decided" handbook={HANDBOOK_PAPER}>
        Each rule compares the book with what the backtest says a book like it does over the same
        number of days. WATCH boundaries are crossed by chance about one time in ten, STOP
        boundaries about one in forty, so a STOP is rare for a healthy book. After a STOP the book
        records days but sends no orders until you acknowledge it.
      </About>
    </>
  );
}

function OrdersPanel({ book }: { book: string }) {
  const q = usePaperOrders(book, undefined);
  const o = q.data;
  return (
    <Panel
      title="Today's orders"
      subtitle={
        o
          ? `${date(o.date)}${o.orders_for ? `, for the close of ${date(o.orders_for)}` : ''}`
          : undefined
      }
      flush
    >
      {q.isPending ? (
        <Skeleton height="6rem" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : (
        <>
          <p className={styles.orderSummary}>
            <ModeBadge mode={o?.mode} /> {int(o?.n_orders ?? 0)} orders: buy{' '}
            {currency(o?.buy_usd ?? 0)}, sell {currency(o?.sell_usd ?? 0)}, estimated cost{' '}
            {currency(o?.est_cost_usd ?? 0)}.
            {o?.cancelled_by_stop ? (
              <>
                {' '}
                <Badge tone="fail">{o.cancelled_by_stop} cancelled by the STOP</Badge>
              </>
            ) : null}
          </p>
          {o && o.orders.length ? (
            <Table
              label="Orders"
              caption="Market-on-close orders sized at the cutoff; estimated cost from the cost model."
            >
              <thead>
                <tr>
                  <Th>Ticker</Th>
                  <Th>Side</Th>
                  <Th numeric>Shares</Th>
                  <Th numeric>Price</Th>
                  <Th numeric>Value</Th>
                  <Th numeric>Weight now → target</Th>
                  <Th numeric>Est. cost</Th>
                </tr>
              </thead>
              <tbody>
                {o.orders.map((x, i) => (
                  <tr key={`${x.ticker}${i}`}>
                    <Td>
                      <code>{x.ticker}</code>
                      {x.close_out ? <Badge tone="neutral">close out</Badge> : null}
                    </Td>
                    <Td>{x.side}</Td>
                    <Td numeric>{int(x.shares)}</Td>
                    <Td numeric>{currency(x.px, { compact: false })}</Td>
                    <Td numeric>{currency(x.usd)}</Td>
                    <Td numeric>
                      {pct(x.current_w, { digits: 2 })} → {pct(x.target_w, { digits: 2 })}
                    </Td>
                    <Td numeric>
                      {currency(x.est_cost_usd, { compact: false })} ({bps(x.est_cost_bps)})
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          ) : (
            <EmptyState title="No orders for this day">
              {o?.status === 'STOP'
                ? 'The book is stopped: no orders until the STOP is acknowledged.'
                : 'Not a rebalance day, or nothing to trade.'}
            </EmptyState>
          )}
        </>
      )}
    </Panel>
  );
}

function HoldingsPanel({ book }: { book: string }) {
  const q = usePaperHoldings(book, undefined);
  return (
    <Panel
      title="Holdings"
      subtitle={q.data ? `${date(q.data.date)}, NAV ${currency(q.data.nav)}` : undefined}
      flush
    >
      {q.isPending ? (
        <Skeleton height="6rem" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : q.data.holdings.length === 0 ? (
        <EmptyState title="No positions">The book is flat.</EmptyState>
      ) : (
        <SortableTable
          caption={`${q.data.holdings.length} positions`}
          rows={q.data.holdings}
          rowKey={(h) => h.ticker}
          limit={12}
          columns={[
            {
              key: 't',
              header: 'Ticker',
              headerText: 'Ticker',
              rowHeader: true,
              sortValue: (h) => h.ticker,
              render: (h) => <code>{h.ticker}</code>,
            },
            {
              key: 'w',
              header: 'Weight',
              headerText: 'Weight',
              align: 'end',
              sortValue: (h) => h.weight ?? 0,
              render: (h) => pct(h.weight, { digits: 2, signed: true }),
            },
            {
              key: 'v',
              header: 'Value',
              headerText: 'Value',
              align: 'end',
              sortValue: (h) => h.usd ?? 0,
              render: (h) => currency(h.usd),
            },
            {
              key: 's',
              header: 'Shares',
              headerText: 'Shares',
              align: 'end',
              sortValue: (h) => h.shares,
              render: (h) => int(h.shares),
            },
            {
              key: 'p',
              header: 'Price',
              headerText: 'Price',
              align: 'end',
              render: (h) => currency(h.px, { compact: false }),
            },
          ]}
        />
      )}
    </Panel>
  );
}

const HISTORY: Column<PaperHistoryRow>[] = [
  {
    key: 'd',
    header: 'Date',
    headerText: 'Date',
    rowHeader: true,
    sortValue: (r) => r.date,
    render: (r) => date(r.date),
  },
  {
    key: 'm',
    header: <Stat k="liveMode">Mode</Stat>,
    headerText: 'Mode',
    render: (r) => <ModeBadge mode={r.mode} />,
  },
  {
    key: 's',
    header: 'Status',
    headerText: 'Status',
    render: (r) =>
      isLevel(r.status) ? (
        <Badge tone={LEVEL_TONE[r.status]}>{r.status}</Badge>
      ) : (
        (r.status ?? 'n/a')
      ),
  },
  {
    key: 'nav',
    header: 'NAV',
    headerText: 'NAV',
    align: 'end',
    sortValue: (r) => r.nav ?? 0,
    render: (r) => currency(r.nav),
  },
  {
    key: 'ret',
    header: 'Return',
    headerText: 'Daily return',
    align: 'end',
    sortValue: (r) => r.ret ?? 0,
    render: (r) => pct(r.ret, { digits: 2, signed: true }),
  },
  {
    key: 'g',
    header: 'Gross',
    headerText: 'Gross',
    align: 'end',
    render: (r) => pct(r.gross, { digits: 0 }),
  },
  {
    key: 'to',
    header: 'Turnover',
    headerText: 'Turnover',
    align: 'end',
    render: (r) => pct(r.turnover, { digits: 2 }),
  },
  {
    key: 'o',
    header: 'Orders',
    headerText: 'Orders',
    align: 'end',
    render: (r) => int(r.n_orders),
  },
  {
    key: 'c',
    header: 'Cost',
    headerText: 'Trading cost',
    align: 'end',
    render: (r) => currency(r.trading_cost, { compact: false }),
  },
  {
    key: 'w',
    header: 'Warnings',
    headerText: 'Warnings',
    align: 'end',
    render: (r) => (r.n_warnings ? <Badge tone="caution">{r.n_warnings}</Badge> : '0'),
  },
];

function HistoryPanel({ book }: { book: string }) {
  const q = usePaperHistory(book);
  const rows = q.data ? [...q.data.rows].reverse() : [];
  return (
    <Panel title="History" subtitle="newest first; one record per processed day" flush>
      {q.isPending ? (
        <Skeleton height="6rem" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : (
        <SortableTable
          caption={`${rows.length} days`}
          columns={HISTORY}
          rows={rows}
          rowKey={(r) => r.date}
          limit={15}
        />
      )}
    </Panel>
  );
}
