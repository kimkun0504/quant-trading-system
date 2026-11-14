import { useMemo, useState } from 'react';
import { Link } from 'react-router';
import type { CapacityRequest, CapacitySummary } from '@/shared/api';
import { ApiError } from '@/shared/api';
import { useHealth } from '@/entities/run';
import { HANDBOOK_COSTS, useCapacityList, useLaunchCapacity } from '@/entities/research';
import {
  CommandLine,
  ConfirmDialog,
  DEFAULT_BASELINE,
  REFERENCE_SLICE,
  Stat,
  StoreSliceFields,
  SwitchField,
  TaskJobs,
  TextField,
  capacityCommand,
  layout,
  parseNumberList,
  sliceErrors,
  toSlice,
} from '@/features/research-launch';
import type { SliceDraft } from '@/features/research-launch';
import form from '@/features/research-launch/research.module.css';
import { currency, date, int, sharpe } from '@/shared/lib/format';
import {
  About,
  Badge,
  Button,
  EmptyState,
  ErrorState,
  Page,
  Panel,
  Skeleton,
  SortableTable,
} from '@/shared/ui';
import type { Column } from '@/shared/ui';

export const DEFAULT_NAVS = '5e6, 5e7, 2.5e8, 1e9';

const COLUMNS: Column<CapacitySummary>[] = [
  {
    key: 'id',
    header: 'Curve',
    headerText: 'Capacity curve',
    rowHeader: true,
    sortValue: (c) => c.id,
    render: (c) => (
      <Link to={`/capacity/${encodeURIComponent(c.id)}`} className={layout.rowLink}>
        <code>{c.id}</code>
      </Link>
    ),
  },
  {
    key: 'navs',
    header: 'NAVs',
    headerText: 'NAVs',
    render: (c) => c.navs.map((n) => currency(n)).join(', '),
  },
  {
    key: 'ref',
    header: 'Net Sharpe at the smallest',
    headerText: 'Net Sharpe at the smallest NAV',
    align: 'end',
    sortValue: (c) => c.reference_sharpe_net ?? -Infinity,
    render: (c) => sharpe(c.reference_sharpe_net),
  },
  {
    key: 'half',
    header: <Stat k="halfSharpeNav" />,
    headerText: 'Half-Sharpe NAV',
    align: 'end',
    render: (c) => (c.half_sharpe_nav === null ? 'beyond the grid' : currency(c.half_sharpe_nav)),
  },
  {
    key: 'end',
    header: 'Data through',
    headerText: 'Data through',
    render: (c) => date(c.sealed_end),
  },
  {
    key: 'config',
    header: 'Config',
    headerText: 'Config hash',
    render: (c) => <code>{c.config_hash.slice(0, 8)}</code>,
  },
];

export function CapacityPage() {
  const list = useCapacityList();
  return (
    <Page docTitle="Capacity" title="Capacity curve">
      <About label="About the capacity curve" handbook={HANDBOOK_COSTS}>
        The same strategy is backtested at several book sizes. Bigger books pay more market impact
        and hit the participation cap more often, so the net Sharpe falls with NAV; where it halves
        is a practical limit on how much the strategy can run. This is a measurement of one
        configuration, not a trial: it does not count against any family&apos;s deflated Sharpe.
      </About>
      <div className={layout.split}>
        <Panel title="Curves" subtitle="newest first" flush>
          {list.isPending ? (
            <Skeleton height="10rem" />
          ) : list.isError ? (
            <ErrorState error={list.error} onRetry={() => void list.refetch()} />
          ) : list.data.results.length === 0 ? (
            <EmptyState title="No capacity curve yet">
              Run one on the right with the default NAVs.
            </EmptyState>
          ) : (
            <SortableTable
              caption={`${list.data.results.length} capacity curves`}
              columns={COLUMNS}
              rows={list.data.results}
              rowKey={(c) => c.id}
            />
          )}
        </Panel>
        <CapacityLaunch />
      </div>
    </Page>
  );
}

function CapacityLaunch() {
  const health = useHealth();
  const storePath = health.data?.store_path ?? null;
  const [navs, setNavs] = useState(DEFAULT_NAVS);
  const [baseline, setBaseline] = useState(DEFAULT_BASELINE);
  const [maxDays, setMaxDays] = useState('');
  const [ml, setMl] = useState(true);
  const [slice, setSlice] = useState<SliceDraft>(REFERENCE_SLICE);
  const [confirm, setConfirm] = useState(false);
  const [message, setMessage] = useState('');
  const launch = useLaunchCapacity();

  const list = parseNumberList(navs);
  const navErr = !list
    ? 'NAVs in USD, comma separated (5e6 = $5M).'
    : list.some((n) => n <= 0)
      ? 'Every NAV must be above 0.'
      : list.length > 8
        ? 'At most 8 NAVs.'
        : new Set(list).size !== list.length
          ? 'Each NAV once.'
          : undefined;
  const daysN = maxDays.trim() === '' ? null : Number(maxDays);
  const daysErr =
    daysN === null || (Number.isInteger(daysN) && daysN >= 20)
      ? undefined
      : 'Whole days, 20 or more, or empty.';
  const cfgErr = /^[A-Za-z0-9_./-]+\.ya?ml$/.test(baseline)
    ? undefined
    : 'A YAML path such as configs/baseline.yaml.';
  const valid = !navErr && !daysErr && !cfgErr && Object.keys(sliceErrors(slice)).length === 0;
  const request = useMemo<CapacityRequest | undefined>(
    () =>
      valid && list
        ? { navs: list, baseline, max_days: daysN, use_ml: ml, store: toSlice(slice) }
        : undefined,
    // eslint-disable-next-line react-hooks/exhaustive-deps -- list is derived from navs
    [valid, navs, baseline, daysN, ml, slice],
  );
  const command = request ? capacityCommand(request, storePath) : '';

  const run = async () => {
    if (!request) return;
    try {
      const out = await launch.mutateAsync(request);
      setMessage(`Queued: ${out.label}.`);
    } catch (e) {
      setMessage(e instanceof ApiError ? e.detail : 'The capacity curve could not be queued.');
    }
    setConfirm(false);
  };

  return (
    <Panel title="Measure capacity" subtitle="a background job; not a trial">
      <div className={form.form}>
        <TextField
          name="navs"
          label="NAVs (USD)"
          value={navs}
          onChange={setNavs}
          error={navErr}
          help={`One backtest per NAV, comma separated. Default ${DEFAULT_NAVS} ($5M to $1B).`}
          wide
        />
        <div className={form.grid}>
          <TextField
            name="baseline"
            label="Config"
            value={baseline}
            onChange={setBaseline}
            error={cfgErr}
            help="The configuration whose capacity is measured."
          />
          <TextField
            name="max_days"
            label="Max days"
            numeric
            value={maxDays}
            onChange={setMaxDays}
            error={daysErr}
            placeholder="all"
            help="Only the last N backtest days (faster). Empty: the full sealed history."
          />
        </div>
        <SwitchField
          name="ml"
          label="With the ML sleeve"
          checked={ml}
          onChange={setMl}
          help="Off: the phase-1 composite (--no-ml)."
        />
        <StoreSliceFields value={slice} onChange={setSlice} storePath={storePath} />
        <div className={form.footer}>
          {command ? <CommandLine command={command} /> : null}
          <p className={form.help}>
            <Badge tone="skipped">Not a trial</Badge> Logged in the ledger as a capacity record so
            the history is complete; it adds nothing to any family&apos;s trial count or DSR.
          </p>
          <div className={form.actions}>
            <Button
              variant="primary"
              disabled={!request || !storePath}
              onClick={() => setConfirm(true)}
            >
              Run capacity curve…
            </Button>
            {!storePath ? (
              <span className={form.help}>The API has no data store (QTS_STORE_PATH).</span>
            ) : null}
          </div>
          {message ? (
            <p className={form.help} role="status">
              {message}
            </p>
          ) : null}
          <TaskJobs tasks={['capacity']} title="Capacity jobs" />
        </div>
      </div>
      <ConfirmDialog
        open={confirm}
        title="Run the capacity curve?"
        confirmLabel="Run the curve"
        busy={launch.isPending}
        onCancel={() => setConfirm(false)}
        onConfirm={() => void run()}
      >
        <p>
          {int(list?.length ?? 0)} backtests ({(list ?? []).map((n) => currency(n)).join(', ')})
          with the stage cache: the fits are shared, so each NAV costs about one
          portfolio-and-execution pass.
        </p>
        <p>
          <Badge tone="skipped">Not a trial</Badge> No family&apos;s trial count changes.
        </p>
      </ConfirmDialog>
    </Panel>
  );
}
