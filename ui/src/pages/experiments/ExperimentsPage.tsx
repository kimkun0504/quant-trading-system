import { useMemo, useState } from 'react';
import { Link } from 'react-router';
import type { LedgerFamily, RunSummary } from '@/shared/api';
import { clusterFamilies, runDescriptor, trialsFor, useLedger, useRuns } from '@/entities/run';
import type { RunFamily } from '@/entities/run';
import { ComparePanel } from '@/features/compare';
import { MAX_SELECTED, toggleSelected, useSelection } from '@/features/selection';
import { bps, dateTime, int, num, pct, sharpe } from '@/shared/lib/format';
import {
  About,
  Badge,
  EmptyState,
  ErrorState,
  Kbd,
  LinkButton,
  Page,
  Panel,
  Skeleton,
  Table,
  Td,
  Th,
  Tooltip,
  VisuallyHidden,
  signClass,
} from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import { RunSealBadge } from '@/features/seal';
import styles from './ExperimentsPage.module.css';

type SortKey =
  | 'created'
  | 'gates'
  | 'sharpe_net'
  | 'sharpe_gross'
  | 'dsr'
  | 'cagr_net'
  | 'max_drawdown'
  | 'cost'
  | 'invested';
type Dir = 'ascending' | 'descending';

const h = (r: RunSummary) => r.headline;

const SORT_VALUE: Record<SortKey, (r: RunSummary) => number | string | null> = {
  created: (r) => r.created_at,
  gates: (r) => (r.status !== 'done' ? r.status : (h(r)?.overall_gate ?? '')),
  sharpe_net: (r) => h(r)?.sharpe_net ?? null,
  sharpe_gross: (r) => h(r)?.sharpe_gross ?? null,
  dsr: (r) => h(r)?.dsr ?? null,
  cagr_net: (r) => h(r)?.cagr_net ?? null,
  max_drawdown: (r) => h(r)?.max_drawdown ?? null,
  cost: (r) => h(r)?.avg_cost_bps_per_side ?? null,
  invested: (r) => h(r)?.invested_day_frac ?? null,
};

const COLUMNS: { key: SortKey; label: string; numeric: boolean; title?: string }[] = [
  { key: 'created', label: 'Run', numeric: false },
  { key: 'gates', label: 'Gates', numeric: false },
  { key: 'sharpe_net', label: 'Net Sharpe', numeric: true },
  { key: 'sharpe_gross', label: 'Gross Sharpe', numeric: true },
  { key: 'dsr', label: 'DSR', numeric: true, title: 'Deflated Sharpe ratio' },
  { key: 'cagr_net', label: 'Net CAGR', numeric: true },
  { key: 'max_drawdown', label: 'Max DD', numeric: true, title: 'Maximum drawdown (net)' },
  { key: 'cost', label: 'Cost/side', numeric: true, title: 'Average cost per side' },
  { key: 'invested', label: 'Invested days', numeric: true, title: 'Share of days with a book' },
];

function sortRuns(runs: readonly RunSummary[], key: SortKey, dir: Dir): RunSummary[] {
  const get = SORT_VALUE[key];
  const sign = dir === 'ascending' ? 1 : -1;
  return [...runs].sort((a, b) => {
    const va = get(a);
    const vb = get(b);
    if (va === null && vb === null) return 0;
    if (va === null) return 1;
    if (vb === null) return -1;
    if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * sign;
    return String(va).localeCompare(String(vb)) * sign;
  });
}

function GateCell({ run }: { run: RunSummary }) {
  if (run.status === 'running') return <Badge tone="caution">Running</Badge>;
  if (run.status === 'failed') return <Badge tone="neutral">Failed</Badge>;
  const pass = run.headline?.overall_gate === 'PASS';
  return <Badge tone={pass ? 'pass' : 'fail'}>{pass ? 'Pass' : 'Fail'}</Badge>;
}

function DsrCell({ run, family }: { run: RunSummary; family: LedgerFamily | undefined }) {
  const head = h(run);
  if (!head) return null;
  const tip = family
    ? `Deflated against ${int(family.n_trials)} trials (${num(family.effective_trials, { digits: 1 })} effective) in ${family.family}.`
    : 'Trial count not in the ledger for this run.';
  return (
    <Tooltip content={tip} align="end">
      <button type="button" className={styles.dsrTerm}>
        {num(head.dsr, { digits: 3 })}
      </button>
    </Tooltip>
  );
}

function RunRow({
  run,
  selected,
  disabled,
  families,
}: {
  run: RunSummary;
  selected: boolean;
  disabled: boolean;
  families: LedgerFamily[] | undefined;
}) {
  const head = h(run);
  const suspect = (head?.sharpe_net ?? 0) > 2;
  return (
    <tr aria-selected={selected} data-run-row={run.id}>
      <Td className={styles.checkCell}>
        <label className={styles.check}>
          <input
            type="checkbox"
            checked={selected}
            disabled={disabled}
            onChange={() => toggleSelected(run.id)}
            aria-label={`Select ${run.label} for comparison`}
          />
        </label>
      </Td>
      <Th scope="row">
        <span className={styles.name}>
          <span>
            <Link to={`/runs/${encodeURIComponent(run.id)}`} className={styles.nameLink}>
              {run.label}
            </Link>
            {suspect ? <span className={styles.flag}>suspect</span> : null}
            <RunSealBadge seal={run.seal} compact />
          </span>
          <span className={styles.descriptor}>
            {runDescriptor(run)}. {dateTime(run.created_at)}
            {run.status === 'done' ? (
              <>
                {'. '}
                <Link
                  to={`/builder?from=${encodeURIComponent(run.id)}`}
                  className={styles.clone}
                  aria-label={`Clone and edit ${run.label}`}
                >
                  Clone &amp; edit
                </Link>
              </>
            ) : null}
          </span>
        </span>
      </Th>
      <Td>
        <GateCell run={run} />
      </Td>
      <Td numeric className={signClass(head?.sharpe_net)}>
        {head ? sharpe(head.sharpe_net, { signed: true }) : ''}
      </Td>
      <Td numeric>{head ? sharpe(head.sharpe_gross, { signed: true }) : ''}</Td>
      <Td numeric>
        <DsrCell run={run} family={trialsFor(run, families)} />
      </Td>
      <Td numeric className={signClass(head?.cagr_net)}>
        {head ? pct(head.cagr_net, { signed: true }) : ''}
      </Td>
      <Td numeric>{head ? pct(head.max_drawdown, { signed: true }) : ''}</Td>
      <Td numeric>{head ? bps(head.avg_cost_bps_per_side) : ''}</Td>
      <Td numeric>{head ? pct(head.invested_day_frac, { digits: 0 }) : ''}</Td>
    </tr>
  );
}

function FamilyGroup({
  family,
  sort,
  selection,
  families,
}: {
  family: RunFamily;
  sort: { key: SortKey; dir: Dir };
  selection: readonly string[];
  families: LedgerFamily[] | undefined;
}) {
  const [open, setOpen] = useState(true);
  const rows = useMemo(() => sortRuns(family.runs, sort.key, sort.dir), [family.runs, sort]);
  const full = selection.length >= MAX_SELECTED;
  return (
    <tbody data-family={family.key}>
      <tr className={styles.familyRow}>
        <th scope="colgroup" colSpan={COLUMNS.length + 1}>
          <button
            type="button"
            className={styles.familyToggle}
            aria-expanded={open}
            onClick={() => setOpen((o) => !o)}
          >
            <Icons.Chevron className={styles.familyChevron} />
            {family.label}
            <span className={styles.familyMeta}>
              {family.runs.length === 1 ? '1 run' : `${family.runs.length} runs`}
            </span>
            {family.diffKeys.length ? (
              <span className={styles.familyKnobs} title={family.diffKeys.join(', ')}>
                varies: {family.diffKeys.join(', ')}
              </span>
            ) : null}
          </button>
        </th>
      </tr>
      {open
        ? rows.map((r) => {
            const selected = selection.includes(r.id);
            return (
              <RunRow
                key={r.id}
                run={r}
                selected={selected}
                disabled={(!selected && full) || r.status !== 'done'}
                families={families}
              />
            );
          })
        : null}
    </tbody>
  );
}

/** Home: runs grouped into families, with the side-by-side panel for the selection. */
export function ExperimentsPage() {
  const runs = useRuns();
  const ledger = useLedger();
  const selection = useSelection();
  const [sort, setSort] = useState<{ key: SortKey; dir: Dir }>({
    key: 'created',
    dir: 'descending',
  });
  const families = useMemo(() => clusterFamilies(runs.data ?? []), [runs.data]);
  const trials = ledger.data?.families.reduce((a, f) => a + f.n_trials, 0);
  const compared = selection.filter((id) => runs.data?.some((r) => r.id === id));

  const toggleSort = (key: SortKey) =>
    setSort((s) =>
      s.key === key
        ? { key, dir: s.dir === 'ascending' ? 'descending' : 'ascending' }
        : { key, dir: 'descending' },
    );

  return (
    <Page
      docTitle="Experiments"
      title="Experiments"
      actions={
        <div className={styles.summary}>
          {runs.data ? (
            <span>
              {runs.data.length} runs in {families.length} families
            </span>
          ) : null}
          <span className={styles.trials} aria-label="Trials in the ledger">
            <span>trials:</span>
            <span className={styles.trialsN}>{trials === undefined ? '…' : int(trials)}</span>
          </span>
          <LinkButton to="/builder" variant="secondary">
            New run
          </LinkButton>
        </div>
      }
    >
      <About
        label="About this table"
        handbook={{ page: 'validation', section: '9. Deflated Sharpe' }}
      >
        Runs are grouped into families: the same strategy on the same data, differing in at most
        three config keys. Net figures are after trading costs and borrow. Sort by DSR rather than
        net Sharpe: it discounts each result by how many trials were run to find it (the count in
        the header). Tick two or more runs to compare them below, or press <Kbd>c</Kbd> for the full
        page.
      </About>

      {compared.length >= 2 ? (
        <section className={styles.compare} aria-labelledby="compare-title">
          <div className={styles.compareHead}>
            <h2 id="compare-title" className={styles.compareTitle}>
              Comparing {compared.length} runs
            </h2>
            <span className={styles.compareHint}>
              <Link to="/compare">Open as a page</Link>
            </span>
          </div>
          <ComparePanel ids={compared} />
        </section>
      ) : null}

      {runs.isPending ? (
        <div aria-busy="true" aria-label="Loading runs" className={styles.skeleton}>
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} height="2rem" />
          ))}
        </div>
      ) : runs.isError ? (
        <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
      ) : runs.data.length === 0 ? (
        <EmptyState
          title="No runs yet"
          action={<LinkButton to="/builder">Launch a run</LinkButton>}
        >
          <p>
            Start with a phase-1 composite on the synthetic market: it takes a few minutes and
            checks that every layer of the pipeline is wired.
          </p>
        </EmptyState>
      ) : (
        <Panel flush>
          <Table
            label="Runs by family"
            caption={`${runs.data.length} runs. Select a column heading to sort within each family.`}
          >
            <thead>
              <tr>
                <Th scope="col" className={styles.checkCell}>
                  <VisuallyHidden>Select</VisuallyHidden>
                </Th>
                {COLUMNS.map((c) => {
                  const active = sort.key === c.key;
                  return (
                    <Th
                      key={c.key}
                      scope="col"
                      numeric={c.numeric}
                      aria-sort={active ? sort.dir : undefined}
                      title={c.title}
                    >
                      <button
                        type="button"
                        className={styles.sortButton}
                        onClick={() => toggleSort(c.key)}
                      >
                        {c.label}
                        <span aria-hidden="true">
                          {' '}
                          {active ? (sort.dir === 'ascending' ? '▲' : '▼') : ''}
                        </span>
                      </button>
                    </Th>
                  );
                })}
              </tr>
            </thead>
            {families.map((f) => (
              <FamilyGroup
                key={f.key}
                family={f}
                sort={sort}
                selection={selection}
                families={ledger.data?.families}
              />
            ))}
          </Table>
        </Panel>
      )}
    </Page>
  );
}
