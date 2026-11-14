import { delay, http, HttpResponse } from 'msw';
import type {
  ApiErrorBody,
  CapacityList,
  CapacityResult,
  FeatureList,
  MidcapCheckResult,
  MidcapPreregistration,
  MidcapResultDetail,
  MidcapStatus,
  PaperAck,
  PaperBooks,
  PaperHistory,
  PaperHoldings,
  PaperOrders,
  PaperReconcile,
  PaperSchedule,
  PaperStatus,
  RiskCalibrationList,
  RiskCalibrationResult,
  SealStatus,
  SignalDef,
  SignalTestDetail,
  SignalTestPreview,
  SignalTestRequest,
  SignalTestSummary,
  StoreSlice,
  SweepDetail,
  SweepManifest,
  SweepPlanView,
  SweepRequest,
  SweepSummary,
  TaskLaunchResponse,
  TrialImpact,
  UnsealCheck,
} from '@/shared/api';
import {
  capacityCommand,
  midcapCommand,
  paperReconcileCommand,
  paperRunCommand,
  riskCommand,
  signalTestCommand,
  storeFlags,
  sweepCommand,
} from '@/features/research-launch/lib/cli';
import { db, launchTask, listJobs } from './db';
import { hashString } from './fixtures/random';
import { MOCK_STORE_PATH } from './fixtures/generate';
import capacityJson from './fixtures/research/capacity.json';
import featuresV1 from './fixtures/research/features_v1.json';
import featuresV2 from './fixtures/research/features_v2.json';
import midcapJson from './fixtures/research/midcap.json';
import midcapDryRun from './fixtures/research/midcap_dry_run.json';
import midcapPrereg from './fixtures/research/midcap_preregistration.json';
import midcapResultJson from './fixtures/research/midcap_result.json';
import paperBooksJson from './fixtures/research/paper_books.json';
import paperHistoryJson from './fixtures/research/paper_history.json';
import paperHoldingsJson from './fixtures/research/paper_holdings.json';
import paperOrdersJson from './fixtures/research/paper_orders.json';
import paperReconcileJson from './fixtures/research/paper_reconcile.json';
import paperScheduleJson from './fixtures/research/paper_schedule.json';
import paperStatusJson from './fixtures/research/paper_status.json';
import riskJson from './fixtures/research/risk.json';
import sealJson from './fixtures/research/seal.json';
import signalTestJson from './fixtures/research/signal_test.json';
import signalTestsJson from './fixtures/research/signal_tests.json';
import signalsJson from './fixtures/research/signals.json';
import sweepJson from './fixtures/research/sweep.json';

/**
 * The research tools and the paper books in the mock API. The fixtures under
 * `fixtures/research/` are real API payloads: `python -m qts_api.research_fixture` runs
 * every research tool and a paper book on a small synthetic store and exports what the
 * API serves (`api/qts_api/research_fixture.py`), so every shape is exact; the numbers
 * are a 60-name synthetic market's (illustrative). On top of them this module keeps a
 * little state: a launch queues a mock task in the shared queue (`db.launchTask`), and
 * when it finishes its result appears (a signal test in the list, a sweep member done,
 * a capacity curve, an acknowledgement on a book). A second book, `baseline` (live MOC
 * sizing, stopped by its drawdown rule), is derived from the fixture book `check`.
 */

const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T;
// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-parameters -- the type argument checks a fixture against its API type at the call site
const json = <T>(body: T, status = 200) => HttpResponse.json(body as object, { status });
const problem = (status: number, detail: string, type = 'error') =>
  HttpResponse.json<ApiErrorBody>({ detail, type, status }, { status });

const SIGNAL_DETAIL = signalTestJson as unknown as SignalTestDetail;
const SWEEP_DETAIL = sweepJson as unknown as SweepDetail;
const CAPACITY = capacityJson as unknown as CapacityResult;
const RISK = riskJson as unknown as RiskCalibrationResult;
const MIDCAP_RESULT = midcapResultJson as unknown as MidcapResultDetail;
const PAPER_STATUS = paperStatusJson as unknown as PaperStatus;
/** The reference research load (`--start 2000-01-01 --max-rank 1500`, handbook 12). */
const REFERENCE_SLICE: StoreSlice = {
  start: '2000-01-01',
  end: null,
  max_names: null,
  min_mcap_ever: null,
  max_rank: 1500,
};
/** The peak-cap load of the runs before 2026-11-14 (its family hash is already cached). */
const LEGACY_SLICE: StoreSlice = { start: '2000-01-01', max_names: 3000, min_mcap_ever: 3e8 };
export const MOCK_FAMILY_HASH = 'c1da477b63d2';
export const MIDCAP_PHRASE = 'not out-of-sample';

function sliceKey(s: StoreSlice | undefined): string {
  const v = s ?? {};
  return JSON.stringify([
    v.start ?? null,
    v.end ?? null,
    v.max_names ?? null,
    v.min_mcap_ever ?? null,
    v.max_rank ?? null,
  ]);
}

/** The holdout seal of research/seal.yaml, in the fixture's shape. */
function sealState(): SealStatus {
  const s = clone(sealJson as unknown as SealStatus);
  const fam = `pipeline_phase2:${MOCK_FAMILY_HASH}`;
  return {
    ...s,
    seal: s.seal
      ? {
          ...s.seal,
          holdout_start: '2023-11-09',
          load_end: '2023-11-08',
          store_end: '2026-11-09',
          sealed_at: '2026-11-13',
          sealed_by: 'repository owner (decision of 2026-11-13)',
          reason:
            'Every design choice of the baseline (configs/baseline.yaml, 9 real-data trials) was made after seeing 2003-06..2026-11, so there is no clean time holdout. The last three years of the store are sealed for all new research from 2026-11-13 on.',
          inherit_family: [fam],
          path: 'research/seal.yaml',
        }
      : null,
    data_through: '2023-11-08',
    inherited_families: { [fam]: 9 },
    by_family: {
      [fam]: { trial: 2, reference: 1 },
      [`capacity:${MOCK_FAMILY_HASH}`]: { capacity: 4 },
      [`calibration:${MOCK_FAMILY_HASH}`]: { calibration: 4 },
    },
    runs_under_seal: 11,
    unseals: (s.unseals.length ? s.unseals : [{}]).map((u) => ({
      ...u,
      ts: '2026-11-13T18:40:12+00:00',
      family: fam,
      phase: 2,
      unseal_reason: 'final check of the ML candidate before paper trading',
      forced: false,
      n_previous_unseals: 0,
    })),
    n_forced: 0,
    ledger_path: 'runs/trial_ledger.jsonl',
  };
}

interface ResearchState {
  seal: SealStatus;
  knownSlices: Set<string>;
  signalTests: SignalTestSummary[];
  signalDetails: Map<string, SignalTestDetail>;
  sweeps: Map<string, SweepManifest>;
  capacity: CapacityResult[];
  risk: RiskCalibrationResult[];
  midcap: MidcapStatus;
  midcapDry: MidcapCheckResult | null;
  books: Map<string, PaperStatus>;
  reconcile: Map<string, PaperReconcile>;
}

function baselineBook(): PaperStatus {
  const b = clone(PAPER_STATUS);
  const n = b.series.date.length;
  b.book = 'baseline';
  b.root = 'paper/baseline';
  b.header = {
    ...b.header,
    name: 'baseline',
    sizing: 'moc',
    reference_run: 'runs/20261113T023313333450Z_e4667d90',
  };
  // backfilled from its start, then run live each day (the usual life of a book)
  b.series.mode = b.series.date.map((_, i) => (i < 12 ? 'backfill' : 'live'));
  // the stop rule fired: the drawdown below the calibrated STOP boundary on the last day
  const dd = b.series.drawdown;
  const stop = b.series.boundaries?.dd_stop ?? [];
  const last = n - 1;
  const boundary = stop[last] ?? -0.02;
  dd[last] = boundary - 0.004;
  if (last > 0) dd[last - 1] = boundary - 0.001;
  const drawdown = b.status.checks.find((c) => c.rule === 'drawdown');
  b.status = {
    ...b.status,
    level: 'STOP',
    stop_rules: ['drawdown'],
    acknowledged: false,
    trading_halted: true,
    reasons: [
      `drawdown: max drawdown ${(100 * (boundary - 0.004)).toFixed(2)}% after ${n} days; below the STOP boundary ${(100 * boundary).toFixed(2)}% (a 2.5% chance in year 1 for a book behaving like the backtest)`,
      ...b.status.reasons,
    ],
    checks: b.status.checks.map((c) =>
      c === drawdown
        ? {
            ...c,
            level: 'STOP',
            value: boundary - 0.004,
            threshold: boundary,
            message: `max drawdown ${(100 * (boundary - 0.004)).toFixed(2)}% after ${n} days; WATCH / STOP boundaries ${(100 * (b.series.boundaries?.dd_watch[last] ?? 0)).toFixed(2)}% / ${(100 * boundary).toFixed(2)}%`,
          }
        : c,
    ),
  };
  if (b.latest)
    b.latest = {
      ...b.latest,
      mode: 'live',
      warnings: ['STOP status: no orders generated (acknowledge with `qts paper ack`)'],
    };
  return b;
}

function midcapState(): MidcapStatus {
  const m = clone(midcapJson as unknown as MidcapStatus);
  // the owner's real check has run (on his Mac): this configuration is once-only now
  return {
    ...m,
    previous_runs: [
      {
        ts: '2026-11-13T21:12:40+00:00',
        family: 'confirm:midcap:7f3a91c20e5b',
        config_hash: m.config_hash,
        sharpe_annual: m.results[0] ? 0.41 : null,
        forced: false,
      },
    ],
    can_launch: false,
    needs_force: true,
    blocked_reason:
      'The check of this configuration already ran: it is once-only, so a second run is no longer out-of-sample. Forcing it is possible but flags the record.',
    store_configured: true,
    last_dry_run: null,
    jobs: [],
  };
}

function create(): ResearchState {
  const tests = clone((signalTestsJson as unknown as { tests: SignalTestSummary[] }).tests);
  const details = new Map<string, SignalTestDetail>();
  for (const t of tests) {
    const d = clone(SIGNAL_DETAIL);
    d.result.id = t.id;
    details.set(t.id, d);
  }
  const sweep = clone(SWEEP_DETAIL.sweep);
  const books = new Map<string, PaperStatus>([
    ['baseline', baselineBook()],
    ['check', clone(PAPER_STATUS)],
  ]);
  return {
    seal: sealState(),
    knownSlices: new Set([
      sliceKey(REFERENCE_SLICE),
      sliceKey(LEGACY_SLICE),
      sliceKey({ max_names: 50 }),
    ]),
    signalTests: tests,
    signalDetails: details,
    sweeps: new Map([[sweep.id, sweep]]),
    capacity: [clone(CAPACITY)],
    risk: [clone(RISK)],
    midcap: midcapState(),
    midcapDry: null,
    books,
    reconcile: new Map(),
  };
}

export let research: ResearchState = create();

export function resetResearch(): void {
  research = create();
}

const storePath = () => MOCK_STORE_PATH;
const respondLaunch = (job: { job_id: string; label: string; command: string | null }) =>
  json<TaskLaunchResponse>(
    { job_id: job.job_id, label: job.label, command: job.command ?? '' },
    202,
  );

function familyOf(store: StoreSlice | undefined): string | null {
  return research.knownSlices.has(sliceKey(store)) ? MOCK_FAMILY_HASH : null;
}

function impactFor(family: string, nNew: number, nRef = 0): TrialImpact {
  // the fixture's numbers with the family's count: illustrative, never computed in the UI
  const base = SWEEP_DETAIL.sweep.impact as TrialImpact;
  const fam = db.ledger.families.find((f) => f.family === family);
  const raw = fam?.n_trials ?? (family.startsWith('pipeline_phase2:') ? 9 : 0);
  const neff = fam?.effective_trials ?? Math.max(1, raw * 0.7);
  const after = neff + nNew;
  const best = raw ? (base.best_sharpe ?? 0.9) : null;
  const dsrBefore = raw ? 0.62 : null;
  const dsrAfter =
    raw && dsrBefore !== null ? dsrBefore * Math.pow(neff / Math.max(after, 1), 0.35) : null;
  return {
    ...base,
    family,
    n_trials_raw: raw,
    n_eff_before: raw ? neff : 0,
    n_new: nNew,
    n_reference: nRef,
    n_eff_after: after,
    exp_max_sr_before: raw ? 0.9 * Math.sqrt(Math.log(Math.max(neff, 1.5))) : 0,
    exp_max_sr_after: 0.9 * Math.sqrt(Math.log(Math.max(after, 1.5))),
    best_sharpe: best,
    dsr_best_before: dsrBefore,
    dsr_best_after: dsrAfter,
  };
}

// ----------------------------------------------------------------------------- sweeps
function expand(body: SweepRequest): Record<string, boolean | number | string>[] {
  const ps = body.params;
  if (body.mode === 'zip') {
    const n = ps[0]?.values.length ?? 0;
    return Array.from({ length: n }, (_, i) =>
      Object.fromEntries(ps.map((p) => [p.key, p.values[i] ?? 0])),
    );
  }
  let out: Record<string, boolean | number | string>[] = [{}];
  for (const p of ps) out = out.flatMap((o) => p.values.map((v) => ({ ...o, [p.key]: v })));
  return out;
}

function planFor(body: SweepRequest): SweepPlanView | { error: Response } {
  const fam = familyOf(body.store);
  if (!fam)
    return {
      error: problem(
        409,
        'The trial family of these load flags is set by a hash of the full-history close panel, which has not been computed for this store snapshot yet. Resolving it reads the close panel once (about a minute on the full store) and is cached afterwards.',
        'family-unknown',
      ),
    };
  const members = expand(body);
  if (members.length > (body.max_runs ?? 12))
    return {
      error: problem(
        422,
        `the sweep has ${members.length} runs > --max-runs ${body.max_runs ?? 12}: narrow it or raise --max-runs`,
      ),
    };
  const id = `sw_${hashString(JSON.stringify(body)).toString(16).padStart(12, '0').slice(0, 12)}`;
  const existing = research.sweeps.get(id);
  const family = `pipeline_phase${body.use_ml === false ? 1 : 2}:${fam}`;
  const plan: SweepManifest = existing ?? {
    ...clone(SWEEP_DETAIL.sweep),
    id,
    base_config: body.base_config ?? 'configs/baseline.yaml',
    params: body.params.map((p) => ({ key: p.key, values: p.values })),
    mode: body.mode ?? 'grid',
    use_ml: body.use_ml ?? true,
    nav: body.nav ?? 5e6,
    max_days: body.max_days ?? null,
    family,
    created_at: new Date().toISOString(),
    pbo: null,
    status: 'pending',
    n_members: members.length,
    members: members.map((ov, i) => ({
      index: i,
      overrides: ov,
      config_hash: hashString(JSON.stringify(ov)).toString(16).padStart(16, '0').slice(0, 16),
      status: 'pending',
      run_id: null,
      run_dir: null,
      reference: false,
      error: null,
      metrics: {},
      started_at: null,
      finished_at: null,
    })),
    impact: impactFor(family, members.length),
  };
  return {
    plan,
    exists: Boolean(existing),
    n_pending: plan.members.filter((m) => m.status !== 'done').length,
    command: sweepCommand(body, storePath(), id),
  };
}

function sweepSummary(s: SweepManifest): SweepSummary {
  const sharpes = s.members
    .map((m) => m.metrics.sharpe_net)
    .filter((x): x is number => typeof x === 'number');
  return {
    id: s.id,
    created_at: s.created_at,
    base_config: s.base_config,
    params: s.params,
    mode: s.mode,
    n_members: s.members.length,
    n_done: s.members.filter((m) => m.status === 'done').length,
    status: s.status,
    family: s.family,
    n_new_trials: s.impact?.n_new ?? null,
    pbo: s.pbo?.pbo ?? null,
    partial: s.pbo?.partial ?? null,
    best_sharpe_net: sharpes.length ? Math.max(...sharpes) : null,
  };
}

function sweepStatus(s: SweepManifest): string {
  const st = new Set(s.members.map((m) => m.status));
  if (st.size === 1 && st.has('done')) return 'done';
  if (st.has('running')) return 'running';
  if (st.has('failed')) return 'failed';
  return st.size === 1 && st.has('pending') ? 'pending' : 'partial';
}

function queueSweep(s: SweepManifest): string[] {
  const jobs = listJobs().jobs.filter(
    (j) => j.subject === s.id && (j.status === 'queued' || j.status === 'running'),
  );
  const busy = new Set(
    jobs
      .filter((j) => j.task === 'sweep_member')
      .map((j) => Number(j.label.match(/member (\d+)/)?.[1] ?? 0) - 1),
  );
  const ids: string[] = [];
  const template = SWEEP_DETAIL.sweep.members[0]?.metrics ?? {};
  for (const m of s.members.filter((x) => x.status !== 'done' && !busy.has(x.index))) {
    const job = launchTask(
      {
        task: 'sweep_member',
        subject: s.id,
        command: sweepCommand(
          {
            params: s.params.map((p) => ({
              key: p.key,
              values: p.values.filter((v): v is boolean | number | string => v !== null),
            })),
            mode: s.mode === 'zip' ? 'zip' : 'grid',
            base_config: s.base_config,
            nav: s.nav,
            max_days: s.max_days,
            use_ml: s.use_ml,
            store: REFERENCE_SLICE,
          },
          storePath(),
          s.id,
        ),
        label: `Sweep ${s.id}: member ${m.index + 1} of ${s.members.length} ${JSON.stringify(m.overrides)}`,
        ml: s.use_ml,
      },
      {
        messages: [
          `Sweep ${s.id}: member ${m.index + 1} of ${s.members.length}`,
          'Fitting the risk model (or reading it from the stage cache)',
          'Running the backtest',
          `Finished: sweep member ${m.index + 1}`,
        ],
        result: { result_id: s.id, index: m.index, status: 'done' },
        onDone: () => {
          const shift = ((hashString(JSON.stringify(m.overrides)) % 100) / 100 - 0.5) * 0.4;
          m.status = 'done';
          m.run_id = `20261114T1200${String(m.index).padStart(2, '0')}000000Z_${m.config_hash.slice(0, 8)}`;
          m.finished_at = new Date().toISOString();
          m.metrics = {
            ...template,
            sharpe_net: (template.sharpe_net ?? 0.5) + shift,
            dsr: Math.max(0, (template.dsr ?? 0.1) + shift / 4),
          };
          s.status = sweepStatus(s);
        },
      },
    );
    m.status = 'pending';
    ids.push(job.job_id);
  }
  return ids;
}

function finalizeJob(s: SweepManifest): string {
  return launchTask(
    {
      task: 'sweep_finalize',
      subject: s.id,
      command: null,
      label: `Sweep ${s.id}: PBO and the metric-vs-parameter table`,
      ml: s.use_ml,
    },
    {
      messages: [
        'Computing PBO (CSCV) over the members’ daily net returns',
        'Finished: PBO computed',
      ],
      result: { result_id: s.id },
      onDone: () => {
        s.pbo = {
          ...(SWEEP_DETAIL.sweep.pbo ?? {}),
          partial: s.members.some((m) => m.status !== 'done'),
          n_configs: s.members.length,
        };
      },
    },
  ).job_id;
}

// ----------------------------------------------------------------------------- handlers
export const researchHandlers = [
  http.get('*/api/seal', async () => {
    await delay(40);
    return json(research.seal);
  }),
  http.get('*/api/seal/unseals', ({ request }) => {
    const q = new URL(request.url).searchParams;
    const hash = q.get('config_hash') ?? '';
    const unsealedRun = [...db.runs.values()].find((r) => r.summary.seal.state === 'unsealed');
    const used = unsealedRun?.summary.config_hash === hash;
    const prev = used ? research.seal.unseals.map((u) => ({ ...u, config_hash: hash })) : [];
    return json<UnsealCheck>({
      config_hash: hash,
      phase: q.get('phase') === '1' ? 1 : 2,
      sealed: true,
      previous: prev,
      requires_force: used,
    });
  }),
  http.get('*/api/ledger/impact', ({ request }) => {
    const q = new URL(request.url).searchParams;
    return json(
      impactFor(
        q.get('family') ?? '',
        Number(q.get('n_new') ?? '1'),
        Number(q.get('n_reference') ?? '0'),
      ),
    );
  }),
  http.post('*/api/families/resolve', async ({ request }) => {
    const body = (await request.json()) as StoreSlice;
    const fam = familyOf(body);
    return json(
      fam
        ? { known: true, market_hash_full: `${fam}ffff`, reason: '' }
        : {
            known: false,
            market_hash_full: null,
            reason:
              'The trial family of these load flags is set by a hash of the full-history close panel, which has not been computed for this store snapshot yet. Resolving it reads the close panel once (about a minute on the full store) and is cached afterwards.',
          },
    );
  }),
  http.post('*/api/families/resolve/jobs', async ({ request }) => {
    const body = (await request.json()) as StoreSlice;
    const job = launchTask(
      {
        task: 'market_hash',
        subject: null,
        command: `qts sweep --param portfolio.trade_aversion=1.0 ${storeFlags(body, storePath()).join(' ')} --dry-run`,
        label: 'Resolve the trial family of a store slice',
        ml: true,
      },
      {
        messages: [
          'Hashing the full-history close panel of these load flags',
          `Finished: family hash ${MOCK_FAMILY_HASH}`,
        ],
        result: { result_id: MOCK_FAMILY_HASH },
        onDone: () => research.knownSlices.add(sliceKey(body)),
      },
    );
    return respondLaunch(job);
  }),
  http.get('*/api/features', ({ request }) => {
    const set = new URL(request.url).searchParams.get('feature_set') ?? 'v1';
    if (set !== 'v1' && set !== 'v2')
      return problem(422, `unknown feature_set '${set}'; known: ['v1', 'v2']`);
    return json<FeatureList>(set === 'v2' ? featuresV2 : featuresV1);
  }),

  // ---- signals
  http.get('*/api/signals', () => json(signalsJson as unknown as SignalDef[])),
  http.get('*/api/signals/tests', async () => {
    await delay(40);
    const n = research.signalTests.length;
    return json({ tests: research.signalTests.map((t) => ({ ...t, n_tests_now: n })) });
  }),
  http.get<{ id: string }>('*/api/signals/tests/:id', ({ params }) => {
    const d = research.signalDetails.get(params.id);
    if (!d) return problem(404, `No signal test with id ${params.id}`, 'result-not-found');
    const tests = research.signalTests.map((t) => ({
      signal: t.signal,
      t: t.t,
      ts: null,
      id: t.id,
      p: t.p,
      p_holm: t.p_holm_now,
    }));
    return json<SignalTestDetail>({
      ...d,
      family_now: {
        ...d.family_now,
        n_tests: tests.length,
        tests,
        p_holm: research.signalTests.find((t) => t.id === params.id)?.p_holm_now ?? null,
      },
    });
  }),
  http.post('*/api/signals/tests/preview', async ({ request }) => {
    const body = (await request.json()) as SignalTestRequest;
    const fam = familyOf(body.store);
    const n = research.signalTests.length;
    const adds = body.log_trial === false ? 0 : 1;
    return json<SignalTestPreview>({
      family: fam ? `signal:${fam}` : null,
      resolution: fam
        ? { known: true, market_hash_full: `${fam}ffff`, reason: '' }
        : {
            known: false,
            market_hash_full: null,
            reason:
              'The trial family of these load flags is not known yet (the close panel has not been hashed).',
          },
      n_tests: fam ? n : 0,
      n_tests_after: (fam ? n : 0) + adds,
      adds,
      hurdle: 3,
      command: signalTestCommand(body, storePath()),
    });
  }),
  http.post('*/api/signals/tests', async ({ request }) => {
    const body = (await request.json()) as SignalTestRequest;
    const def = (signalsJson as unknown as SignalDef[]).find((s) => s.name === body.feature);
    if (!def) return problem(422, `unknown signal '${body.feature}'`);
    const id = `20261114T${String(100000 + research.signalTests.length).slice(1)}000000Z_${body.feature}`;
    const job = launchTask(
      {
        task: 'signal_test',
        subject: body.feature,
        command: signalTestCommand(body, storePath()),
        label: `Signal test: ${body.feature} (h ${[...body.horizons].sort((a, b) => a - b).join(',')})`,
        ml: body.use_ml ?? true,
      },
      {
        messages: [
          'Loading the store under the seal and the baseline fit (stage cache)',
          'Computing labels at every horizon',
          'Rank IC by horizon, year and fold',
          'Paper books and the incremental test',
          `Finished: signal test ${id}`,
        ],
        result: { result_id: id, n_tests: research.signalTests.length + 1 },
        onDone: () => {
          const d = clone(SIGNAL_DETAIL);
          d.result.id = id;
          d.result.signal = { ...d.result.signal, ...def };
          research.signalDetails.set(id, d);
          const base = research.signalTests[0];
          if (base)
            research.signalTests.unshift({
              ...base,
              id,
              signal: def.name,
              theme: def.theme,
              source: def.source,
              logged: body.log_trial !== false,
            });
        },
      },
    );
    return respondLaunch(job);
  }),

  // ---- sweeps
  http.post('*/api/sweeps/plan', async ({ request }) => {
    const out = planFor((await request.json()) as SweepRequest);
    return 'error' in out ? out.error : json(out);
  }),
  http.post('*/api/sweeps', async ({ request }) => {
    const { expected_id, ...req } = (await request.json()) as SweepRequest & {
      expected_id: string;
    };
    const body = { ...req, expected_id };
    const out = planFor(req);
    if ('error' in out) return out.error;
    if (out.plan.id !== body.expected_id)
      return problem(
        409,
        `the sweep changed since it was confirmed (${body.expected_id} -> ${out.plan.id}); review the new plan`,
        'plan-changed',
      );
    research.sweeps.set(out.plan.id, out.plan);
    const ids = queueSweep(out.plan);
    const fin = finalizeJob(out.plan);
    return json(
      {
        sweep_id: out.plan.id,
        job_ids: ids,
        finalize_job_id: fin,
        n_queued: ids.length + 1,
        command: out.command,
      },
      202,
    );
  }),
  http.get('*/api/sweeps', () =>
    json({
      sweeps: [...research.sweeps.values()]
        .map(sweepSummary)
        .sort((a, b) => b.created_at.localeCompare(a.created_at)),
    }),
  ),
  http.get<{ id: string }>('*/api/sweeps/:id', ({ params }) => {
    const s = research.sweeps.get(params.id);
    if (!s) return problem(404, `No sweep with id ${params.id}`, 'result-not-found');
    const jobs = listJobs().jobs.filter((j) => j.subject === s.id);
    const activeJobs = jobs.filter((j) => j.status === 'queued' || j.status === 'running');
    return json<SweepDetail>({
      sweep: s,
      jobs,
      active: activeJobs.length,
      resumable: activeJobs.length ? 0 : s.members.filter((m) => m.status !== 'done').length,
      command: sweepCommand(
        {
          params: s.params.map((p) => ({
            key: p.key,
            values: p.values.filter((v): v is boolean | number | string => v !== null),
          })),
          mode: s.mode === 'zip' ? 'zip' : 'grid',
          base_config: s.base_config,
          nav: s.nav,
          max_days: s.max_days,
          use_ml: s.use_ml,
          store: REFERENCE_SLICE,
        },
        storePath(),
        s.id,
      ),
    });
  }),
  http.post<{ id: string }>('*/api/sweeps/:id/resume', ({ params }) => {
    const s = research.sweeps.get(params.id);
    if (!s) return problem(404, `No sweep with id ${params.id}`, 'result-not-found');
    const ids = queueSweep(s);
    if (!ids.length)
      return problem(
        409,
        'nothing to resume: every member is done or already queued',
        'nothing-to-resume',
      );
    const fin = finalizeJob(s);
    return json(
      { sweep_id: s.id, job_ids: ids, finalize_job_id: fin, n_queued: ids.length + 1, command: '' },
      202,
    );
  }),

  // ---- capacity / risk
  http.get('*/api/capacity', () =>
    json<CapacityList>({
      results: research.capacity.map((c) => ({
        id: c.id,
        config_hash: c.config_hash,
        family: c.family,
        navs: c.points.map((p) => p.nav),
        half_sharpe_nav: c.half_sharpe_nav,
        reference_sharpe_net: c.reference_sharpe_net,
        sealed_end: c.sealed_end,
      })),
    }),
  ),
  http.get<{ id: string }>('*/api/capacity/:id', ({ params }) => {
    const c = research.capacity.find((x) => x.id === params.id);
    return c ? json(c) : problem(404, `No capacity curve with id ${params.id}`, 'result-not-found');
  }),
  http.post('*/api/capacity', async ({ request }) => {
    const body = (await request.json()) as {
      navs: number[];
      store?: StoreSlice;
      max_days?: number | null;
      baseline?: string;
      use_ml?: boolean;
    };
    if (!body.navs.length || new Set(body.navs).size !== body.navs.length)
      return problem(422, 'navs must be distinct amounts between 1e4 and 1e12 USD');
    const id = `20261114T${String(200000 + research.capacity.length).slice(1)}000000Z_feafa843`;
    const job = launchTask(
      {
        task: 'capacity',
        subject: null,
        command: capacityCommand({ ...body, navs: body.navs }, storePath()),
        label: `Capacity curve at ${body.navs.length} NAVs`,
        ml: body.use_ml ?? true,
      },
      {
        messages: [
          'Fitting the baseline once (stage cache)',
          ...body.navs.map((n) => `NAV $${n / 1e6}M: backtest`),
          `Finished: capacity curve ${id}`,
        ],
        result: { result_id: id },
        onDone: () => research.capacity.unshift({ ...clone(CAPACITY), id }),
      },
    );
    return respondLaunch(job);
  }),
  http.get('*/api/risk-calibrations', () =>
    json<RiskCalibrationList>({
      results: research.risk.map((r) => ({
        id: r.id,
        config_hash: r.config_hash,
        nus: r.results.map((x) => x.nu_scale),
        current_nu_scale: r.current_nu_scale,
        recommended_nu_scale: r.recommended_nu_scale,
        sealed_end: r.sealed_end,
      })),
    }),
  ),
  http.get<{ id: string }>('*/api/risk-calibrations/:id', ({ params }) => {
    const r = research.risk.find((x) => x.id === params.id);
    return r
      ? json(r)
      : problem(404, `No risk calibration report with id ${params.id}`, 'result-not-found');
  }),
  http.post('*/api/risk-calibrations', async ({ request }) => {
    const body = (await request.json()) as {
      nu: number[];
      store?: StoreSlice;
      nav?: number;
      max_days?: number | null;
      use_ml?: boolean;
    };
    const id = `20261114T${String(300000 + research.risk.length).slice(1)}000000Z_feafa843`;
    const job = launchTask(
      {
        task: 'risk_calibration',
        subject: null,
        command: riskCommand({ ...body, nu: body.nu }, storePath()),
        label: `Risk calibration over nu = ${body.nu.join(', ')}`,
        ml: body.use_ml ?? true,
      },
      {
        messages: [
          'Fitting the baseline once (stage cache)',
          ...body.nu.map((n) => `nu ${n}: backtest`),
          `Finished: risk calibration report ${id}`,
        ],
        result: { result_id: id },
        onDone: () => research.risk.unshift({ ...clone(RISK), id }),
      },
    );
    return respondLaunch(job);
  }),

  // ---- mid-cap
  http.get('*/api/checks/midcap', () => {
    const jobs = listJobs().jobs.filter((j) => j.task === 'midcap_check');
    const busy = jobs.some((j) => j.status === 'queued' || j.status === 'running');
    const m = research.midcap;
    return json<MidcapStatus>({
      ...m,
      jobs,
      last_dry_run: research.midcapDry,
      can_launch: !busy && m.can_launch,
      blocked_reason: busy ? 'A mid-cap check job is already queued or running.' : m.blocked_reason,
    });
  }),
  http.get('*/api/checks/midcap/preregistration', () => json<MidcapPreregistration>(midcapPrereg)),
  http.get<{ id: string }>('*/api/checks/midcap/results/:id', ({ params }) => {
    if (params.id !== MIDCAP_RESULT.result.verdict?.run_id)
      return problem(404, `No mid-cap check result with id ${params.id}`, 'result-not-found');
    return json(MIDCAP_RESULT);
  }),
  http.post('*/api/checks/midcap', async ({ request }) => {
    const body = (await request.json()) as {
      dry_run?: boolean;
      force?: boolean;
      confirm?: string | null;
      store?: StoreSlice;
      cache?: boolean;
    };
    const m = research.midcap;
    if (
      listJobs().jobs.some(
        (j) => j.task === 'midcap_check' && (j.status === 'queued' || j.status === 'running'),
      )
    )
      return problem(409, 'A mid-cap check job is already queued or running.', 'midcap-busy');
    if (!body.dry_run) {
      if (m.needs_force && !body.force)
        return problem(409, m.blocked_reason ?? 'already ran', 'midcap-already-ran');
      if (body.force && (body.confirm ?? '').trim() !== MIDCAP_PHRASE)
        return problem(422, `to force a repeat, type "${MIDCAP_PHRASE}"`);
    }
    const job = launchTask(
      {
        task: 'midcap_check',
        subject: 'midcap',
        command: midcapCommand(body, storePath()),
        label: body.dry_run
          ? 'Mid-cap check: coverage dry run'
          : body.force
            ? 'Mid-cap check (forced repeat)'
            : 'Mid-cap check',
        ml: true,
      },
      {
        messages: [
          'measuring load coverage (DuckDB over the store)',
          body.dry_run ? 'Finished: coverage checked (dry run)' : 'Finished: verdict',
        ],
        result: { dry_run: Boolean(body.dry_run) },
        onDone: () => {
          if (body.dry_run)
            research.midcapDry = clone(midcapDryRun as unknown as MidcapCheckResult);
        },
      },
    );
    return respondLaunch(job);
  }),

  // ---- paper
  http.get('*/api/paper/books', async () => {
    await delay(40);
    const fixture = paperBooksJson as unknown as PaperBooks;
    const check = fixture.books[0];
    const books = [...research.books.values()].map((b) => ({
      ...(check ?? ({} as PaperBooks['books'][number])),
      name: b.book,
      sizing: b.header.sizing,
      level: b.status.level,
      trading_halted: b.status.trading_halted,
      last_mode: b.latest?.mode ?? null,
      reference_run: b.header.reference_run,
    }));
    return json<PaperBooks>({ paper_root: 'paper', paper_root_flag: null, books });
  }),
  http.get<{ name: string }>('*/api/paper/books/:name/status', async ({ params }) => {
    await delay(40);
    const b = research.books.get(params.name);
    if (!b) return problem(404, `No paper book '${params.name}' under paper`, 'book-not-found');
    const jobs = listJobs().jobs.filter(
      (j) => j.subject === params.name && (j.task === 'paper_run' || j.task === 'paper_reconcile'),
    );
    return json<PaperStatus>({
      ...b,
      jobs,
      last_reconcile: research.reconcile.get(params.name) ?? null,
    });
  }),
  http.get<{ name: string }>('*/api/paper/books/:name/history', ({ params }) => {
    const b = research.books.get(params.name);
    if (!b) return problem(404, `No paper book '${params.name}'`, 'book-not-found');
    const h = clone(paperHistoryJson as unknown as PaperHistory);
    const at = new Map(b.series.date.map((d, i) => [d, i]));
    return json<PaperHistory>({
      book: params.name,
      rows: h.rows.map((r) => {
        const i = at.get(r.date);
        return i === undefined
          ? r
          : { ...r, mode: b.series.mode[i] ?? r.mode, status: b.series.status[i] ?? r.status };
      }),
    });
  }),
  http.get<{ name: string }>('*/api/paper/books/:name/orders', ({ params, request }) => {
    if (!research.books.has(params.name))
      return problem(404, `No paper book '${params.name}'`, 'book-not-found');
    const date = new URL(request.url).searchParams.get('date');
    const o = clone(paperOrdersJson as unknown as PaperOrders);
    const holdings = (paperHoldingsJson as unknown as PaperHoldings).holdings;
    // the fixture's last day traded nothing; a rebalance day shows the shape of the orders
    const orders = holdings.slice(0, 8).map((h, i) => ({
      ticker: h.ticker,
      side: i % 2 ? 'sell' : 'buy',
      shares: (i % 2 ? -1 : 1) * Math.round(Math.abs(h.shares) * 0.12),
      px: h.px,
      usd: (i % 2 ? -1 : 1) * Math.abs((h.usd ?? 0) * 0.12),
      target_w: (h.weight ?? 0) * (i % 2 ? 0.88 : 1.12),
      current_w: h.weight,
      kind: 'rebalance',
      close_out: false,
      exec_date: o.orders_for,
      est_cost_usd: Math.abs((h.usd ?? 0) * 0.12) * 0.0006,
      est_cost_bps: 6,
    }));
    return json<PaperOrders>({
      ...o,
      date: date ?? o.date,
      orders: params.name === 'baseline' ? [] : orders,
      n_orders: params.name === 'baseline' ? 0 : orders.length,
      buy_usd: orders.filter((x) => x.usd > 0).reduce((a, x) => a + x.usd, 0),
      sell_usd: -orders.filter((x) => x.usd < 0).reduce((a, x) => a + x.usd, 0),
      est_cost_usd: orders.reduce((a, x) => a + x.est_cost_usd, 0),
      cancelled_by_stop: params.name === 'baseline' ? 6 : 0,
    });
  }),
  http.get<{ name: string }>('*/api/paper/books/:name/holdings', ({ params }) =>
    research.books.has(params.name)
      ? json(paperHoldingsJson as unknown as PaperHoldings)
      : problem(404, `No paper book '${params.name}'`, 'book-not-found'),
  ),
  http.get<{ name: string }>('*/api/paper/books/:name/schedule', ({ params }) => {
    if (!research.books.has(params.name))
      return problem(404, `No paper book '${params.name}'`, 'book-not-found');
    const s = clone<PaperSchedule>(paperScheduleJson);
    const swap = (t: string) =>
      t
        .replace(/com\.qts\.paper\.check/g, `com.qts.paper.${params.name}`)
        .replace(/--name check/g, `--name ${params.name}`)
        .replace(/paper\/check/g, `paper/${params.name}`);
    return json<PaperSchedule>({
      label: swap(s.label),
      plist: swap(s.plist),
      crontab: swap(s.crontab),
      log: swap(s.log),
      install: swap(s.install),
    });
  }),
  http.post<{ name: string }>('*/api/paper/books/:name/ack', async ({ params, request }) => {
    const b = research.books.get(params.name);
    if (!b) return problem(404, `No paper book '${params.name}'`, 'book-not-found');
    const { note } = (await request.json()) as { note: string };
    if (!note || note.trim().length < 3)
      return problem(422, 'note: at least 3 characters: why it is safe to continue');
    const a: PaperAck = {
      date: b.last_date ?? '',
      ts: new Date().toISOString(),
      level: b.status.level,
      rules: b.status.stop_rules,
      reasons: b.status.reasons,
      note: note.trim(),
      user: 'me',
    };
    b.acks = [...b.acks, a];
    b.status = {
      ...b.status,
      acknowledged: true,
      trading_halted: false,
      level: b.status.level === 'STOP' ? 'WATCH' : b.status.level,
      checks: b.status.checks.map((c) =>
        c.level === 'STOP'
          ? { ...c, level: 'WATCH', message: `${c.message} [STOP acknowledged]` }
          : c,
      ),
    };
    return json(a);
  }),
  http.post<{ name: string }>('*/api/paper/books/:name/run', async ({ params, request }) => {
    const b = research.books.get(params.name);
    if (!b) return problem(404, `No paper book '${params.name}'`, 'book-not-found');
    if (
      listJobs().jobs.some(
        (j) => j.subject === params.name && (j.status === 'queued' || j.status === 'running'),
      )
    )
      return problem(409, 'A job of this book is already queued or running.', 'book-busy');
    const body = (await request.json()) as {
      date?: string | null;
      until?: string | null;
      no_sync?: boolean;
      dry_run?: boolean;
      redo_last?: boolean;
      signal_mode?: 'daily' | 'single';
    };
    const today = !body.until && !body.date;
    const job = launchTask(
      {
        task: 'paper_run',
        subject: params.name,
        command: paperRunCommand(params.name, body, null),
        label: `Paper book ${params.name}: ${body.dry_run ? 'dry run' : body.until ? 'backfill' : 'run today'}`,
        ml: true,
      },
      {
        messages: [
          'sync: incremental Sharadar sync',
          'freshness: checking the store holds the expected session',
          'Finished: nothing to do',
        ],
        result: { result_id: params.name, n_days: 0 },
        // the mock store ends before today's session: "run today" meets the vendor's delay
        ...(today && !body.dry_run
          ? {
              failure: {
                message:
                  'Failed: Data not yet published: the store does not hold the session the vendor should have published by now. Nothing was changed; run it again later (the next run catches up day by day).',
                error_kind: 'stale_data' as const,
                exit_code: 3,
              },
            }
          : {}),
      },
    );
    return respondLaunch(job);
  }),
  http.post<{ name: string }>('*/api/paper/books/:name/reconcile', ({ params }) => {
    if (!research.books.has(params.name))
      return problem(404, `No paper book '${params.name}'`, 'book-not-found');
    if (
      listJobs().jobs.some(
        (j) => j.subject === params.name && (j.status === 'queued' || j.status === 'running'),
      )
    )
      return problem(409, 'A job of this book is already queued or running.', 'book-busy');
    const job = launchTask(
      {
        task: 'paper_reconcile',
        subject: params.name,
        command: paperReconcileCommand(params.name, null),
        label: `Paper book ${params.name}: reconcile with the batch simulator`,
        ml: true,
      },
      {
        messages: [
          "Re-running the batch simulator over the book's dates (one fit)",
          'Finished: reconciliation against the batch simulator',
        ],
        result: { result_id: params.name },
        onDone: () =>
          research.reconcile.set(
            params.name,
            clone(paperReconcileJson as unknown as PaperReconcile),
          ),
      },
    );
    return respondLaunch(job);
  }),
];
