import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router';
import type { SweepRequest, SweepSummary } from '@/shared/api';
import { ApiError } from '@/shared/api';
import { useConfigSchema, useHealth } from '@/entities/run';
import {
  HANDBOOK_SWEEPS,
  stampIso,
  useCreateSweep,
  useResolveFamily,
  useSweepPlan,
  useSweeps,
} from '@/entities/research';
import {
  CommandLine,
  ConfirmDialog,
  Presets,
  REFERENCE_SLICE,
  SelectField,
  Stat,
  StoreSliceFields,
  SwitchField,
  TaskJobs,
  TextField,
  TrialCostSlip,
  layout,
  sliceErrors,
  sweepCommand,
  toSlice,
} from '@/features/research-launch';
import type { Preset, SliceDraft } from '@/features/research-launch';
import form from '@/features/research-launch/research.module.css';
import { schemaFields } from '@/features/run-builder/lib/fields';
import type { Field } from '@/features/run-builder/lib/fields';
import { dateTime, int, num, sharpe } from '@/shared/lib/format';
import { useDebounced } from '@/shared/lib/useDebounced';
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
  Table,
  Td,
  Th,
} from '@/shared/ui';
import type { Column } from '@/shared/ui';
import {
  MAX_MEMBERS,
  MAX_PARAMS,
  SWEEP_TONE,
  checkGrid,
  overridesText,
  sweepableFields,
} from './lib/grid';
import type { ParamDraft } from './lib/grid';
import styles from './Sweeps.module.css';

const paramsText = (s: Pick<SweepSummary, 'params'>) =>
  s.params.map((p) => `${p.key} = ${p.values.map(String).join(', ')}`).join('; ');

const COLUMNS: Column<SweepSummary>[] = [
  {
    key: 'id',
    header: 'Sweep',
    headerText: 'Sweep',
    rowHeader: true,
    sortValue: (s) => s.id,
    render: (s) => (
      <Link to={`/sweeps/${encodeURIComponent(s.id)}`} className={layout.rowLink}>
        <code>{s.id}</code>
      </Link>
    ),
  },
  { key: 'params', header: 'Varies', headerText: 'Parameters', render: (s) => paramsText(s) },
  {
    key: 'members',
    header: 'Members',
    headerText: 'Members done',
    align: 'end',
    sortValue: (s) => s.n_members,
    render: (s) => `${s.n_done} / ${s.n_members}`,
  },
  {
    key: 'status',
    header: 'Status',
    headerText: 'Status',
    sortValue: (s) => s.status,
    render: (s) => <Badge tone={SWEEP_TONE[s.status] ?? 'neutral'}>{s.status}</Badge>,
  },
  {
    key: 'best',
    header: 'Best net Sharpe',
    headerText: 'Best net Sharpe',
    align: 'end',
    sortValue: (s) => s.best_sharpe_net ?? -Infinity,
    render: (s) => sharpe(s.best_sharpe_net),
  },
  {
    key: 'pbo',
    header: <Stat k="pbo" />,
    headerText: 'PBO',
    align: 'end',
    render: (s) =>
      s.pbo === null ? 'not yet' : `${num(s.pbo, { digits: 2 })}${s.partial ? ' (partial)' : ''}`,
  },
  { key: 'family', header: 'Family', headerText: 'Family', render: (s) => <code>{s.family}</code> },
  {
    key: 'created',
    header: 'Created',
    headerText: 'Created',
    sortValue: (s) => s.created_at,
    render: (s) => dateTime(stampIso(s.created_at)),
  },
];

interface GridPreset {
  params: ParamDraft[];
  mode: 'grid' | 'zip';
}

const PRESETS: readonly Preset<GridPreset>[] = [
  {
    label: 'Trade aversion',
    title: 'portfolio.trade_aversion = 0.5, 1, 2 (3 members)',
    value: { mode: 'grid', params: [{ key: 'portfolio.trade_aversion', text: '0.5, 1, 2' }] },
  },
  {
    label: 'Alignment ν',
    title: 'risk.alignment_nu_scale = 0.02, 0.05, 0.1 (3 members)',
    value: { mode: 'grid', params: [{ key: 'risk.alignment_nu_scale', text: '0.02, 0.05, 0.1' }] },
  },
  {
    label: 'IC shrink × trade aversion',
    title: 'alpha.ic_shrink = 0.25, 0.5 × portfolio.trade_aversion = 0.5, 1, 2 (6 members)',
    value: {
      mode: 'grid',
      params: [
        { key: 'alpha.ic_shrink', text: '0.25, 0.5' },
        { key: 'portfolio.trade_aversion', text: '0.5, 1, 2' },
      ],
    },
  },
];

export function SweepsPage() {
  const sweeps = useSweeps();
  const list = sweeps.data?.sweeps ?? [];
  const trials = list.reduce((a, s) => a + (s.n_new_trials ?? 0), 0);
  return (
    <Page docTitle="Sweeps" title="Sweeps">
      <About label="About sweeps" handbook={HANDBOOK_SWEEPS}>
        A sweep runs the pipeline once per point of a small grid of settings (at most {MAX_MEMBERS}
        ), each member a trial in the family. It reports every member, not just the best, and the
        probability that the grid&apos;s winner is overfit (PBO). Members run one at a time in the
        background with the stage cache, and a stopped sweep resumes where it left off.
      </About>
      <section
        className={layout.context}
        aria-label="Multiple testing"
        data-testid="testing-context"
      >
        <span>
          <span className={layout.contextLabel}>Sweeps</span>
          <br />
          <span className={layout.contextBig}>{sweeps.data ? int(list.length) : '…'}</span>
        </span>
        <span>
          <span className={layout.contextLabel}>
            <Stat k="trials">Trials they added</Stat>
          </span>
          <br />
          <span className={layout.contextBig}>{sweeps.data ? int(trials) : '…'}</span>
        </span>
        <span className={layout.lede}>
          Every member counts against the family&apos;s deflated Sharpe: the preview shows the cut
          before you confirm.
        </span>
      </section>
      <div className={layout.split}>
        <Panel title="Sweeps" subtitle="newest first" flush>
          {sweeps.isPending ? (
            <Skeleton height="10rem" />
          ) : sweeps.isError ? (
            <ErrorState error={sweeps.error} onRetry={() => void sweeps.refetch()} />
          ) : list.length === 0 ? (
            <EmptyState title="No sweep yet">
              Plan one on the right: pick up to four settings and their values.
            </EmptyState>
          ) : (
            <SortableTable
              caption={`${list.length} sweeps`}
              columns={COLUMNS}
              rows={list}
              rowKey={(s) => s.id}
            />
          )}
        </Panel>
        <SweepLaunch />
      </div>
    </Page>
  );
}

function ParamRow({
  index,
  draft,
  fields,
  error,
  onChange,
  onRemove,
}: {
  index: number;
  draft: ParamDraft;
  fields: readonly Field[];
  error: string | undefined;
  onChange: (d: ParamDraft) => void;
  onRemove: (() => void) | null;
}) {
  const f = fields.find((x) => x.path === draft.key);
  const allowed =
    f?.type === 'boolean'
      ? 'true, false'
      : f?.enum
        ? f.enum.join(', ')
        : f
          ? [
              f.minimum !== undefined
                ? `≥ ${String(f.minimum)}`
                : f.exclusiveMinimum !== undefined
                  ? `> ${String(f.exclusiveMinimum)}`
                  : null,
              f.maximum !== undefined
                ? `≤ ${String(f.maximum)}`
                : f.exclusiveMaximum !== undefined
                  ? `< ${String(f.exclusiveMaximum)}`
                  : null,
              f.type === 'integer' ? 'whole numbers' : null,
            ]
              .filter(Boolean)
              .join(', ') || 'any number'
          : '';
  return (
    <fieldset className={styles.paramRow} data-param={index}>
      <legend className={styles.paramLegend}>Setting {index + 1}</legend>
      <SelectField
        name={`param-${index}-key`}
        label="Setting"
        value={draft.key}
        options={[
          { value: '', label: 'Pick a setting…' },
          ...fields.map((x) => ({ value: x.path, label: x.path })),
        ]}
        onChange={(key) => onChange({ key, text: draft.text })}
        help={f ? `${f.description ?? f.title} Default ${String(f.default ?? 'none')}.` : undefined}
      />
      <TextField
        name={`param-${index}-values`}
        label="Values"
        value={draft.text}
        onChange={(text) => onChange({ key: draft.key, text })}
        placeholder="0.5, 1, 2"
        error={error}
        help={allowed ? `Comma separated: ${allowed}.` : undefined}
      />
      {onRemove ? (
        <Button
          size="sm"
          variant="quiet"
          onClick={onRemove}
          aria-label={`Remove setting ${index + 1}`}
        >
          Remove
        </Button>
      ) : null}
    </fieldset>
  );
}

function SweepLaunch() {
  const navigate = useNavigate();
  const schema = useConfigSchema();
  const health = useHealth();
  const storePath = health.data?.store_path ?? null;
  const fields = useMemo(() => sweepableFields(schemaFields(schema.data)), [schema.data]);
  const [params, setParams] = useState<ParamDraft[]>([
    { key: 'portfolio.trade_aversion', text: '0.5, 1, 2' },
  ]);
  const [mode, setMode] = useState<'grid' | 'zip'>('grid');
  const [baseConfig, setBaseConfig] = useState('configs/baseline.yaml');
  const [nav, setNav] = useState('5e6');
  const [maxDays, setMaxDays] = useState('');
  const [ml, setMl] = useState(true);
  const [slice, setSlice] = useState<SliceDraft>(REFERENCE_SLICE);
  const [confirm, setConfirm] = useState(false);
  const [message, setMessage] = useState('');
  const create = useCreateSweep();
  const resolve = useResolveFamily();

  const grid = useMemo(() => checkGrid(params, fields, mode), [params, fields, mode]);
  const navN = Number(nav);
  const navErr =
    Number.isFinite(navN) && navN > 0 ? undefined : 'A NAV in USD above 0 (5e6 = $5M).';
  const daysN = maxDays.trim() === '' ? null : Number(maxDays);
  const daysErr =
    daysN === null || (Number.isInteger(daysN) && daysN >= 20)
      ? undefined
      : 'Whole days, 20 or more, or empty for the whole history.';
  const cfgErr = /^[A-Za-z0-9_./-]+\.ya?ml$/.test(baseConfig)
    ? undefined
    : 'A YAML path such as configs/baseline.yaml.';
  const valid =
    grid.ok && !navErr && !daysErr && !cfgErr && Object.keys(sliceErrors(slice)).length === 0;
  const request = useMemo<SweepRequest | undefined>(
    () =>
      valid
        ? {
            params: grid.params,
            mode,
            base_config: baseConfig,
            nav: navN,
            max_days: daysN,
            use_ml: ml,
            store: toSlice(slice),
          }
        : undefined,
    [valid, grid, mode, baseConfig, navN, daysN, ml, slice],
  );
  const plan = useSweepPlan(useDebounced(request, 300));
  const p = plan.data;
  const apiErr = plan.error instanceof ApiError ? plan.error : null;
  const familyUnknown = apiErr?.status === 409 ? apiErr.detail : null;
  const planError = apiErr && apiErr.status !== 409 ? apiErr.detail : null;
  const refs = p ? p.plan.members.filter((m) => m.reference).length : 0;
  const adds = p
    ? p.n_pending - p.plan.members.filter((m) => m.reference && m.status !== 'done').length
    : 0;
  const command = request ? sweepCommand(request, storePath, p?.plan.id) : '';

  const run = async () => {
    if (!request || !p) return;
    try {
      const out = await create.mutateAsync({ ...request, expected_id: p.plan.id });
      setConfirm(false);
      void navigate(`/sweeps/${encodeURIComponent(out.sweep_id)}`);
    } catch (e) {
      setMessage(e instanceof ApiError ? e.detail : 'The sweep could not be created.');
      setConfirm(false);
    }
  };

  return (
    <Panel title="Plan a sweep" subtitle={`up to ${MAX_PARAMS} settings, ${MAX_MEMBERS} members`}>
      <div className={form.form}>
        <Presets
          presets={PRESETS}
          onPick={(v) => {
            setParams(v.params.map((x) => ({ ...x })));
            setMode(v.mode);
          }}
        />
        {schema.isPending ? <Skeleton height="6rem" /> : null}
        {params.map((d, i) => (
          <ParamRow
            key={i}
            index={i}
            draft={d}
            fields={fields}
            error={
              !grid.ok && grid.error.startsWith(`${d.key}:`)
                ? grid.error.slice(d.key.length + 2)
                : undefined
            }
            onChange={(nd) => setParams(params.map((x, j) => (j === i ? nd : x)))}
            onRemove={params.length > 1 ? () => setParams(params.filter((_, j) => j !== i)) : null}
          />
        ))}
        <div className={form.actions}>
          <Button
            size="sm"
            disabled={params.length >= MAX_PARAMS}
            onClick={() => setParams([...params, { key: '', text: '' }])}
          >
            Add a setting
          </Button>
          <SelectField
            name="mode"
            label="Combine"
            value={mode}
            options={[
              { value: 'grid', label: 'Every combination (grid)' },
              { value: 'zip', label: 'Pair the lists (zip)' },
            ]}
            onChange={setMode}
          />
        </div>
        <p className={styles.count} role="status" data-testid="member-count">
          {grid.members !== null ? (
            <>
              <strong>{grid.members}</strong> {grid.members === 1 ? 'member' : 'members'} of at most{' '}
              {MAX_MEMBERS}
            </>
          ) : null}
          {!grid.ok ? <span className={form.error}> {grid.error}</span> : null}
        </p>
        <div className={form.grid}>
          <TextField
            name="base_config"
            label="Base config"
            value={baseConfig}
            onChange={setBaseConfig}
            error={cfgErr}
            help="The YAML every member starts from; the sweep changes only the listed settings."
            wide
          />
          <TextField
            name="nav"
            label="NAV (USD)"
            numeric
            value={nav}
            onChange={setNav}
            error={navErr}
            help="Book size for costs and capacity. Default 5e6 ($5M)."
          />
          <TextField
            name="max_days"
            label="Max days"
            numeric
            value={maxDays}
            onChange={setMaxDays}
            error={daysErr}
            placeholder="all"
            help="Only the last N backtest days (a quick look). Empty: the full history."
          />
        </div>
        <SwitchField
          name="ml"
          label="With the ML sleeve (phase 2)"
          checked={ml}
          onChange={setMl}
          help="Off: the phase-1 composite (--no-ml), a different family."
        />
        <StoreSliceFields value={slice} onChange={setSlice} storePath={storePath} />
        <div className={form.footer}>
          {command ? <CommandLine command={command} /> : null}
          {p ? (
            <TrialCostSlip
              impact={p.plan.impact ?? emptyImpact(p.plan.family)}
              adds={Math.max(0, adds)}
              references={refs}
              title="What this sweep costs"
            >
              {p.exists ? (
                <p className={form.help}>
                  <Badge tone="accent">Resumes</Badge> This sweep exists:{' '}
                  {p.plan.members.length - p.n_pending} of {p.plan.members.length} members are done
                  and only the rest run.
                </p>
              ) : null}
            </TrialCostSlip>
          ) : familyUnknown ? (
            <div className={form.slip}>
              <p className={form.slipLead}>The trial family is not known yet.</p>
              <p className={form.help}>{familyUnknown}</p>
              <Button
                size="sm"
                onClick={() => resolve.mutate(toSlice(slice))}
                disabled={resolve.isPending}
              >
                Resolve the family (a short job)
              </Button>
            </div>
          ) : planError ? (
            <p className={form.error} role="alert">
              {planError}
            </p>
          ) : request ? (
            <Skeleton height="5rem" />
          ) : null}
          <div className={form.actions}>
            <Button
              variant="primary"
              disabled={!request || !p || !storePath}
              onClick={() => setConfirm(true)}
            >
              Review and run…
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
          <TaskJobs tasks={['sweep_member', 'sweep_finalize', 'market_hash']} title="Sweep jobs" />
        </div>
      </div>
      <ConfirmDialog
        open={confirm && Boolean(p)}
        title={`Run sweep ${p?.plan.id ?? ''}?`}
        confirmLabel={`Run ${p?.n_pending ?? 0} ${p?.n_pending === 1 ? 'member' : 'members'}`}
        busy={create.isPending}
        onCancel={() => setConfirm(false)}
        onConfirm={() => void run()}
      >
        {p ? (
          <>
            <TrialCostSlip
              impact={p.plan.impact ?? emptyImpact(p.plan.family)}
              adds={Math.max(0, adds)}
              references={refs}
            />
            <Table label="Members" caption="Each member is one pipeline run with these overrides.">
              <thead>
                <tr>
                  <Th numeric>#</Th>
                  <Th>Overrides</Th>
                  <Th>Status</Th>
                </tr>
              </thead>
              <tbody>
                {p.plan.members.map((m) => (
                  <tr key={m.index}>
                    <Td numeric>{m.index + 1}</Td>
                    <Td>{overridesText(m.overrides)}</Td>
                    <Td>{m.reference ? <Badge tone="skipped">reference</Badge> : m.status}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <p>
              Members run one at a time in the background queue, then a last job computes PBO.
              Runtime is about one cached pipeline run per member.
            </p>
          </>
        ) : null}
      </ConfirmDialog>
    </Panel>
  );
}

function emptyImpact(family: string) {
  return {
    family,
    n_trials_raw: 0,
    n_eff_before: 0,
    n_new: 0,
    n_reference: 0,
    n_eff_after: 0,
    var_sr_annual: 0,
    exp_max_sr_before: null,
    exp_max_sr_after: null,
    best_sharpe: null,
    best_n_obs: null,
    dsr_best_before: null,
    dsr_best_after: null,
  };
}
