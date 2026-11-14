import { useMemo, useState } from 'react';
import { Link } from 'react-router';
import type { RiskCalibrationRequest, RiskCalibrationSummary } from '@/shared/api';
import { ApiError } from '@/shared/api';
import { useHealth } from '@/entities/run';
import { HANDBOOK_RISK, useLaunchRisk, useRiskList } from '@/entities/research';
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
  layout,
  parseNumberList,
  riskCommand,
  sliceErrors,
  toSlice,
} from '@/features/research-launch';
import type { SliceDraft } from '@/features/research-launch';
import form from '@/features/research-launch/research.module.css';
import { date, int, num } from '@/shared/lib/format';
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

/** `qts.research.risk_calibration.DEFAULT_NU_GRID`. */
export const DEFAULT_NU = '0, 0.05, 0.25, 1';

const COLUMNS: Column<RiskCalibrationSummary>[] = [
  {
    key: 'id',
    header: 'Report',
    headerText: 'Calibration report',
    rowHeader: true,
    sortValue: (r) => r.id,
    render: (r) => (
      <Link to={`/risk/${encodeURIComponent(r.id)}`} className={layout.rowLink}>
        <code>{r.id}</code>
      </Link>
    ),
  },
  {
    key: 'nus',
    header: <Stat k="nu">ν tried</Stat>,
    headerText: 'ν tried',
    render: (r) => r.nus.map((v) => num(v, { digits: 2 })).join(', '),
  },
  {
    key: 'cur',
    header: 'Current ν',
    headerText: 'Current ν',
    align: 'end',
    render: (r) => num(r.current_nu_scale, { digits: 2 }),
  },
  {
    key: 'rec',
    header: 'Recommended',
    headerText: 'Recommended ν',
    align: 'end',
    render: (r) =>
      r.recommended_nu_scale === null ? 'none' : num(r.recommended_nu_scale, { digits: 2 }),
  },
  {
    key: 'end',
    header: 'Data through',
    headerText: 'Data through',
    render: (r) => date(r.sealed_end),
  },
];

export function RiskPage() {
  const list = useRiskList();
  return (
    <Page docTitle="Risk calibration" title="Risk calibration">
      <About label="About the calibration" handbook={HANDBOOK_RISK}>
        The risk model is right when the book&apos;s realised volatility matches what it predicted.
        The alignment scale ν adds risk for the part of the alpha the factors do not explain; this
        report backtests a few values of ν and compares realised with ex-ante volatility year by
        year. It only reports: nothing changes until you launch a run with the new ν, and that run
        is one logged trial.
      </About>
      <div className={layout.split}>
        <Panel title="Reports" subtitle="newest first" flush>
          {list.isPending ? (
            <Skeleton height="10rem" />
          ) : list.isError ? (
            <ErrorState error={list.error} onRetry={() => void list.refetch()} />
          ) : list.data.results.length === 0 ? (
            <EmptyState title="No calibration report yet">
              Run one on the right with the default ν grid.
            </EmptyState>
          ) : (
            <SortableTable
              caption={`${list.data.results.length} calibration reports`}
              columns={COLUMNS}
              rows={list.data.results}
              rowKey={(r) => r.id}
            />
          )}
        </Panel>
        <RiskLaunch />
      </div>
    </Page>
  );
}

function RiskLaunch() {
  const health = useHealth();
  const storePath = health.data?.store_path ?? null;
  const [nus, setNus] = useState(DEFAULT_NU);
  const [nav, setNav] = useState('5e6');
  const [baseline, setBaseline] = useState(DEFAULT_BASELINE);
  const [maxDays, setMaxDays] = useState('');
  const [ml, setMl] = useState(true);
  const [slice, setSlice] = useState<SliceDraft>(REFERENCE_SLICE);
  const [confirm, setConfirm] = useState(false);
  const [message, setMessage] = useState('');
  const launch = useLaunchRisk();

  const list = parseNumberList(nus);
  const nuErr = !list
    ? 'ν values, comma separated.'
    : list.some((n) => n < 0)
      ? 'ν is 0 or more.'
      : list.length > 8
        ? 'At most 8 values.'
        : new Set(list).size !== list.length
          ? 'Each value once.'
          : undefined;
  const navN = Number(nav);
  const navErr = Number.isFinite(navN) && navN > 0 ? undefined : 'A NAV in USD above 0.';
  const daysN = maxDays.trim() === '' ? null : Number(maxDays);
  const daysErr =
    daysN === null || (Number.isInteger(daysN) && daysN >= 20)
      ? undefined
      : 'Whole days, 20 or more, or empty.';
  const cfgErr = /^[A-Za-z0-9_./-]+\.ya?ml$/.test(baseline)
    ? undefined
    : 'A YAML path such as configs/baseline.yaml.';
  const valid =
    !nuErr && !navErr && !daysErr && !cfgErr && Object.keys(sliceErrors(slice)).length === 0;
  const request = useMemo<RiskCalibrationRequest | undefined>(
    () =>
      valid && list
        ? { nu: list, nav: navN, baseline, max_days: daysN, use_ml: ml, store: toSlice(slice) }
        : undefined,
    // eslint-disable-next-line react-hooks/exhaustive-deps -- list is derived from nus
    [valid, nus, navN, baseline, daysN, ml, slice],
  );
  const command = request ? riskCommand(request, storePath) : '';

  const run = async () => {
    if (!request) return;
    try {
      const out = await launch.mutateAsync(request);
      setMessage(`Queued: ${out.label}.`);
    } catch (e) {
      setMessage(e instanceof ApiError ? e.detail : 'The calibration could not be queued.');
    }
    setConfirm(false);
  };

  return (
    <Panel title="Calibrate ν" subtitle="a background job; a report, not a trial">
      <div className={form.form}>
        <TextField
          name="nu"
          label="ν values (alignment scale)"
          value={nus}
          onChange={setNus}
          error={nuErr}
          help={`risk.alignment_nu_scale values, comma separated. Default ${DEFAULT_NU}.`}
          wide
        />
        <div className={form.grid}>
          <TextField
            name="nav"
            label="NAV (USD)"
            numeric
            value={nav}
            onChange={setNav}
            error={navErr}
            help="Default 5e6 ($5M)."
          />
          <TextField
            name="baseline"
            label="Config"
            value={baseline}
            onChange={setBaseline}
            error={cfgErr}
            help="The configuration whose ν is varied."
          />
          <TextField
            name="max_days"
            label="Max days"
            numeric
            value={maxDays}
            onChange={setMaxDays}
            error={daysErr}
            placeholder="all"
            help="Only the last N days. Empty: the whole sealed history."
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
            <Badge tone="skipped">Report only</Badge> Logged as a calibration record; no
            family&apos;s trial count changes. Adopting the recommended ν later is a new run, and
            that run is one trial.
          </p>
          <div className={form.actions}>
            <Button
              variant="primary"
              disabled={!request || !storePath}
              onClick={() => setConfirm(true)}
            >
              Run calibration…
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
          <TaskJobs tasks={['risk_calibration']} title="Calibration jobs" />
        </div>
      </div>
      <ConfirmDialog
        open={confirm}
        title="Run the risk calibration?"
        confirmLabel="Run the calibration"
        busy={launch.isPending}
        onCancel={() => setConfirm(false)}
        onConfirm={() => void run()}
      >
        <p>
          {int(list?.length ?? 0)} backtests, one per ν, sharing the cached fits.{' '}
          <Badge tone="skipped">Not a trial</Badge>
        </p>
      </ConfirmDialog>
    </Panel>
  );
}
