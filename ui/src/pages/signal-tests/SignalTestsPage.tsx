import { useMemo, useState } from 'react';
import { Link } from 'react-router';
import type { SignalTestRequest, SignalTestSummary } from '@/shared/api';
import { ApiError } from '@/shared/api';
import { useHealth } from '@/entities/run';
import {
  HANDBOOK_SIGNALS,
  useLaunchSignalTest,
  useResolveFamily,
  useSignalPreview,
  useSignalTests,
  useSignals,
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
  layout,
  signalTestCommand,
  sliceErrors,
  toSlice,
} from '@/features/research-launch';
import type { Preset, SliceDraft } from '@/features/research-launch';
import form from '@/features/research-launch/research.module.css';
import { date, int, num } from '@/shared/lib/format';
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
} from '@/shared/ui';
import type { Column } from '@/shared/ui';
import { parseHorizons } from './lib/shape';

const COLUMNS: Column<SignalTestSummary>[] = [
  {
    key: 'signal',
    header: 'Signal',
    headerText: 'Signal',
    rowHeader: true,
    sortValue: (t) => t.signal,
    render: (t) => (
      <Link to={`/signals/${encodeURIComponent(t.id)}`} className={layout.rowLink}>
        {t.signal}
      </Link>
    ),
  },
  {
    key: 'theme',
    header: 'Theme',
    headerText: 'Theme',
    sortValue: (t) => t.theme,
    render: (t) => t.theme,
  },
  {
    key: 'h',
    header: 'Horizon',
    headerText: 'Primary horizon',
    align: 'end',
    sortValue: (t) => t.primary_horizon,
    render: (t) => `${t.primary_horizon} d`,
  },
  {
    key: 'ic',
    header: <Stat k="ic">Mean IC</Stat>,
    headerText: 'Mean IC',
    align: 'end',
    render: (t) => num(t.mean_ic_primary, { digits: 4, signed: true }),
  },
  {
    key: 't',
    header: <Stat k="tNw">t</Stat>,
    headerText: 'Newey-West t',
    align: 'end',
    render: (t) => num(t.t, { signed: true }),
  },
  {
    key: 'holm',
    header: <Stat k="pHolm">Holm p now</Stat>,
    headerText: 'Holm-adjusted p now',
    align: 'end',
    render: (t) => `${num(t.p_holm_now, { digits: 3 })} (of ${t.n_tests_now})`,
  },
  {
    key: 'hurdle',
    header: <Stat k="hurdle">|t| ≥ 3</Stat>,
    headerText: 'Hurdle',
    render: (t) =>
      t.passes_hurdle ? <Badge tone="pass">Clears</Badge> : <Badge tone="fail">Below</Badge>,
  },
  {
    key: 'logged',
    header: 'Counted',
    headerText: 'Counted',
    render: (t) => (t.logged ? 'yes' : <Badge tone="caution">not logged</Badge>),
  },
  {
    key: 'window',
    header: 'Window',
    headerText: 'Window',
    sortValue: (t) => t.window[0] ?? '',
    render: (t) => `${date(t.window[0])} – ${date(t.window[1])}`,
  },
];

const PRESETS: readonly Preset<{ horizons: string; step: string; ml: boolean }>[] = [
  {
    label: 'Reference test',
    title: 'Horizons 5, 20, 60, 120; step 5; ML baseline; the reference load flags',
    value: { horizons: '5,20,60,120', step: '5', ml: true },
  },
  {
    label: 'Short horizons',
    title: 'Horizons 5, 10, 20 (fast signals)',
    value: { horizons: '5,10,20', step: '5', ml: true },
  },
  {
    label: 'Composite baseline',
    title: 'The baseline without the ML sleeve (phase 1 fit)',
    value: { horizons: '5,20,60,120', step: '5', ml: false },
  },
];

export function SignalTestsPage() {
  const tests = useSignalTests();
  const signals = useSignals();
  const families = new Set((tests.data?.tests ?? []).map((t) => t.family));
  const fam = tests.data?.tests[0]?.family ?? null;
  const n = tests.data?.tests[0]?.n_tests_now ?? 0;
  return (
    <Page docTitle="Signal tests" title="Signal tests">
      <About label="About signal tests" handbook={HANDBOOK_SIGNALS}>
        A test measures one signal on the baseline&apos;s out-of-sample dates under the seal: rank
        IC by horizon, year and fold, its decay, a neutral paper book, and what it adds to the
        baseline. Every test is a trial in its family, so a good-looking t is read against the |t| ≥
        3 hurdle and the Holm-adjusted p over every signal tried.
      </About>
      <section
        className={layout.context}
        aria-label="Multiple testing"
        data-testid="testing-context"
      >
        <span>
          <span className={layout.contextLabel}>Signals tested</span>
          <br />
          <span className={layout.contextBig}>
            {tests.data ? int(tests.data.tests.length) : '…'}
          </span>
        </span>
        <span>
          <span className={layout.contextLabel}>
            <Stat k="family">Family</Stat>
          </span>
          <br />
          <code>{fam ?? 'none yet'}</code>
          {families.size > 1 ? (
            <span className={layout.small}> and {families.size - 1} more</span>
          ) : null}
        </span>
        <span className={layout.lede}>
          Holm adjusts every test&apos;s p over the {int(n)} logged in its family; the bar for a new
          signal is |t| ≥ 3.
        </span>
      </section>
      <div className={layout.split}>
        <div className={layout.stack}>
          <Panel title="Tests" subtitle="newest first" flush>
            {tests.isPending ? (
              <Skeleton height="10rem" />
            ) : tests.isError ? (
              <ErrorState error={tests.error} onRetry={() => void tests.refetch()} />
            ) : tests.data.tests.length === 0 ? (
              <EmptyState title="No signal test yet">
                Pick a signal on the right and run a test.
              </EmptyState>
            ) : (
              <SortableTable
                caption={`${tests.data.tests.length} signal tests`}
                columns={COLUMNS}
                rows={tests.data.tests}
                rowKey={(t) => t.id}
              />
            )}
          </Panel>
          <Panel
            title="Signals you can test"
            subtitle="the v1 and v2 features and registered custom signals"
            flush
          >
            {signals.data ? (
              <SortableTable
                caption={`${signals.data.length} signals`}
                columns={[
                  {
                    key: 'name',
                    header: 'Signal',
                    headerText: 'Signal',
                    rowHeader: true,
                    sortValue: (s) => s.name,
                    render: (s) => <code>{s.name}</code>,
                  },
                  {
                    key: 'theme',
                    header: 'Theme',
                    headerText: 'Theme',
                    sortValue: (s) => s.theme,
                    render: (s) => s.theme,
                  },
                  {
                    key: 'set',
                    header: 'Set',
                    headerText: 'Feature set',
                    sortValue: (s) => s.feature_set,
                    render: (s) => s.feature_set,
                  },
                  {
                    key: 'desc',
                    header: 'What it measures',
                    headerText: 'Description',
                    render: (s) => s.description,
                  },
                ]}
                rows={signals.data}
                rowKey={(s) => s.name}
                limit={10}
              />
            ) : (
              <Skeleton height="8rem" />
            )}
          </Panel>
        </div>
        <SignalLaunch />
      </div>
    </Page>
  );
}

function SignalLaunch() {
  const signals = useSignals();
  const health = useHealth();
  const storePath = health.data?.store_path ?? null;
  const [feature, setFeature] = useState('high_52w');
  const [horizons, setHorizons] = useState('5,20,60,120');
  const [step, setStep] = useState('5');
  const [cost, setCost] = useState('');
  const [ml, setMl] = useState(true);
  const [cache, setCache] = useState(true);
  const [logTrial, setLogTrial] = useState(true);
  const [slice, setSlice] = useState<SliceDraft>(REFERENCE_SLICE);
  const [confirm, setConfirm] = useState(false);
  const [message, setMessage] = useState('');
  const launch = useLaunchSignalTest();
  const resolve = useResolveFamily();

  const hz = parseHorizons(horizons);
  const stepN = Number(step);
  const stepErr =
    Number.isInteger(stepN) && stepN >= 1 && stepN <= 60
      ? undefined
      : 'A whole number of days from 1 to 60.';
  const costN = cost.trim() === '' ? null : Number(cost);
  const costErr =
    costN === null || (Number.isFinite(costN) && costN >= 0 && costN <= 200)
      ? undefined
      : 'Basis points from 0 to 200, or empty.';
  const valid = hz.ok && !stepErr && !costErr && Object.keys(sliceErrors(slice)).length === 0;
  const request = useMemo<SignalTestRequest | undefined>(
    () =>
      valid
        ? {
            feature,
            horizons: hz.value,
            step: stepN,
            cost_bps: costN,
            use_ml: ml,
            cache,
            log_trial: logTrial,
            store: toSlice(slice),
          }
        : undefined,
    // eslint-disable-next-line react-hooks/exhaustive-deps -- hz is derived from horizons
    [valid, feature, horizons, stepN, costN, ml, cache, logTrial, slice],
  );
  const preview = useSignalPreview(useDebounced(request, 300));
  const def = signals.data?.find((s) => s.name === feature);
  const command = request ? signalTestCommand(request, storePath) : '';
  const p = preview.data;

  const run = async () => {
    if (!request) return;
    try {
      const out = await launch.mutateAsync(request);
      setMessage(`Queued: ${out.label}.`);
      setConfirm(false);
    } catch (e) {
      setMessage(e instanceof ApiError ? e.detail : 'The test could not be queued.');
      setConfirm(false);
    }
  };

  return (
    <Panel title="Test a signal" subtitle="a background job, under the seal">
      <div className={form.form}>
        <Presets
          presets={PRESETS}
          onPick={(v) => {
            setHorizons(v.horizons);
            setStep(v.step);
            setMl(v.ml);
            setSlice(REFERENCE_SLICE);
          }}
        />
        <SelectField
          name="feature"
          label="Signal"
          value={feature}
          options={(signals.data ?? []).map((s) => ({
            value: s.name,
            label: `${s.name} (${s.theme})`,
          }))}
          onChange={setFeature}
          help={
            def
              ? `${def.description} Sign ${def.sign > 0 ? '+' : '−'}, ${def.feature_set}.`
              : undefined
          }
        />
        <div className={form.grid}>
          <TextField
            name="horizons"
            label="Horizons (trading days)"
            value={horizons}
            onChange={setHorizons}
            error={hz.ok ? undefined : hz.error}
            help="Label horizons, comma separated. The primary (the statistic) is the baseline's 60 days when listed; the others are reported, never selected from."
          />
          <TextField
            name="step"
            label="Step (days)"
            numeric
            value={step}
            onChange={setStep}
            error={stepErr}
            help="Sampling step and paper-book holding period. Default 5: the baseline's rebalance cadence."
          />
          <TextField
            name="cost_bps"
            label="Cost per side (bp)"
            numeric
            value={cost}
            onChange={setCost}
            placeholder="the baseline run's"
            error={costErr}
            help="Charged on the paper book's turnover. Empty: the newest baseline run's cost per side."
          />
        </div>
        <SwitchField
          name="ml"
          label="Baseline with the ML sleeve"
          checked={ml}
          onChange={setMl}
          help="Off: the phase-1 composite baseline (--no-ml)."
        />
        <SwitchField
          name="cache"
          label="Baseline fit from the stage cache"
          checked={cache}
          onChange={setCache}
          help="Reuse the sealed baseline run's fits (minutes instead of an hour). Off: --no-cache."
        />
        <SwitchField
          name="log"
          label="Count this test (log it in the ledger)"
          checked={logTrial}
          onChange={setLogTrial}
          help={
            logTrial
              ? 'The test is a trial in its family.'
              : 'Not counted: use only for debugging. An uncounted test hides a try from the multiple-testing correction.'
          }
        />
        <StoreSliceFields value={slice} onChange={setSlice} storePath={storePath} />
        <div className={form.footer}>
          {command ? <CommandLine command={command} /> : null}
          <div className={form.actions}>
            <Button
              variant="primary"
              disabled={!request || !storePath}
              onClick={() => setConfirm(true)}
            >
              Run test…
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
          <TaskJobs tasks={['signal_test', 'market_hash']} title="Signal test jobs" />
        </div>
      </div>
      <ConfirmDialog
        open={confirm}
        title={`Run a signal test of ${feature}?`}
        confirmLabel={logTrial ? 'Run and count the test' : 'Run without counting'}
        busy={launch.isPending}
        onCancel={() => setConfirm(false)}
        onConfirm={() => void run()}
      >
        {p?.family ? (
          <section className={form.slip} data-testid="trial-cost">
            <h3 className={form.slipTitle}>Before you confirm</h3>
            <p className={form.slipLead}>
              {p.adds ? (
                <>
                  Adds <strong>1</strong> test to <code>{p.family}</code>: {p.n_tests} →{' '}
                  {p.n_tests_after} tests.
                </>
              ) : (
                <>
                  <Badge tone="caution">Not counted</Badge> The test is not logged: the family stays
                  at {p.n_tests} tests.
                </>
              )}
            </p>
            <p className={form.help}>
              Holm adjusts every test&apos;s p over all {p.n_tests_after} tests, so the adjusted p
              of the earlier tests rises too. A new signal needs |t| ≥{' '}
              {num(p.hurdle, { digits: 1 })}. (Signal tests are judged by Holm, not by a deflated
              Sharpe: no pipeline family&apos;s DSR changes.)
            </p>
          </section>
        ) : p ? (
          <div className={form.slip}>
            <p className={form.slipLead}>The test&apos;s family is not known yet.</p>
            <p className={form.help}>{p.resolution.reason}</p>
            <Button
              size="sm"
              onClick={() => resolve.mutate(toSlice(slice))}
              disabled={resolve.isPending}
            >
              Resolve the family (a short job)
            </Button>
          </div>
        ) : (
          <Skeleton height="5rem" />
        )}
        <p>
          About 5–10 minutes on the reference load with the baseline fit cached; it runs in the
          background queue.
        </p>
      </ConfirmDialog>
    </Panel>
  );
}
