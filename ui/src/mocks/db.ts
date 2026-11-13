import type {
  ConfigLeaf,
  ConfigValidation,
  Job,
  JobLog,
  JobQueue,
  LaunchRunRequest,
  Ledger,
  RunSummary,
  ValidationIssue,
} from '@/shared/api';
import { CONFIG_SCHEMA } from './fixtures/configSchema';
import {
  MOCK_STORE_PATH,
  generateRun,
  ledgerFamily,
  mockConfigHash,
  mockDataHash,
  mockWarmup,
} from './fixtures/generate';
import type { GeneratedRun, RunSpec } from './fixtures/generate';
import { seedLedger } from './fixtures/ledger';
import { FAILED_SPEC, RUNNING_SPEC, SEED_SPECS } from './fixtures/runs';
import { hashString } from './fixtures/random';

/**
 * In-memory registry and job queue behind the MSW handlers. `resetDb()` restores the
 * seed state. The queue behaves like the API's: FIFO, one job at a time, each taking
 * `jobDurationMs`; time only moves when someone asks (`advanceQueue(now)`), so tests
 * and screenshots are deterministic.
 */
interface JobRecord {
  job: Job;
  spec: RunSpec;
  request: LaunchRunRequest;
  /** ms since epoch when the job started running (set when it leaves the queue). */
  startedAt: number | null;
}

export interface MockDb {
  runs: Map<string, GeneratedRun>;
  pending: Map<string, RunSummary>;
  jobs: Map<string, JobRecord>;
  /** Job ids waiting, in order. */
  queue: string[];
  ledger: Ledger;
  jobDurationMs: number;
  /** Median duration of "finished jobs of similar size" (null: no history, no ETA). */
  historySeconds: number | null;
  seq: number;
}

function placeholder(spec: RunSpec): RunSummary {
  return { ...generateRun(spec).summary, headline: null, status: spec.status };
}

function create(): MockDb {
  const runs = new Map<string, GeneratedRun>();
  for (const spec of SEED_SPECS) runs.set(spec.id, generateRun(spec));
  const pending = new Map<string, RunSummary>();
  pending.set(RUNNING_SPEC.id, placeholder(RUNNING_SPEC));
  pending.set(FAILED_SPEC.id, placeholder(FAILED_SPEC));
  return {
    runs,
    pending,
    jobs: new Map(),
    queue: [],
    ledger: seedLedger(),
    jobDurationMs: 4000,
    historySeconds: 240,
    seq: 0,
  };
}

export let db: MockDb = create();

export function resetDb(
  overrides: Partial<Pick<MockDb, 'jobDurationMs' | 'historySeconds'>> = {},
): void {
  db = { ...create(), ...overrides };
}

export function listSummaries(): RunSummary[] {
  const all = [...[...db.runs.values()].map((r) => r.summary), ...db.pending.values()];
  return all.sort((a, b) => b.created_at.localeCompare(a.created_at));
}

// ----------------------------------------------------------------------------- validate

type Leaf = {
  type: string;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  enum?: string[];
  nullable?: boolean;
  title?: string;
};

function schemaLeaves(): Map<string, Leaf> {
  const out = new Map<string, Leaf>();
  const walk = (node: { properties: Record<string, unknown> }, prefix: string) => {
    for (const [k, v] of Object.entries(node.properties)) {
      const n = v as Leaf & { properties?: Record<string, unknown> };
      if (n.type === 'object' && n.properties) walk({ properties: n.properties }, `${prefix}${k}.`);
      else out.set(`${prefix}${k}`, n);
    }
  };
  walk(CONFIG_SCHEMA, '');
  return out;
}

const RESERVED = new Set(['seed', 'execution.signal_lag_days', 'cache.enabled', 'cache.root']);

function overrideIssues(overrides: Record<string, ConfigLeaf>): ValidationIssue[] {
  const leaves = schemaLeaves();
  const issues: ValidationIssue[] = [];
  for (const [key, v] of Object.entries(overrides)) {
    const loc = ['body', 'config_overrides', key];
    const leaf = leaves.get(key);
    const bad = (msg: string, type = 'value_error.invalid') => issues.push({ loc, msg, type });
    if (RESERVED.has(key)) bad(`${key} cannot be overridden here`, 'value_error.reserved');
    else if (!leaf) bad(`unknown config field '${key}'`, 'value_error.unknown_key');
    else if (v === null) {
      if (!leaf.nullable) bad(`${leaf.title ?? key} cannot be empty (null)`);
    } else if (leaf.type === 'boolean') {
      if (typeof v !== 'boolean') bad(`${leaf.title ?? key} must be true or false`);
    } else if (leaf.type === 'string') {
      if (typeof v !== 'string') bad(`${leaf.title ?? key} must be a string`);
      else if (leaf.enum && !leaf.enum.includes(v))
        bad(`${leaf.title ?? key} must be one of ${leaf.enum.join(', ')}`);
    } else if (typeof v !== 'number') bad(`${leaf.title ?? key} must be a number`);
    else if (leaf.type === 'integer' && !Number.isInteger(v))
      bad(`${leaf.title ?? key} must be a whole number`);
    else if (leaf.minimum !== undefined && v < leaf.minimum)
      bad(`${leaf.title ?? key} must be at least ${leaf.minimum}`);
    else if (leaf.maximum !== undefined && v > leaf.maximum)
      bad(`${leaf.title ?? key} must be at most ${leaf.maximum}`);
    else if (leaf.exclusiveMinimum !== undefined && v <= leaf.exclusiveMinimum)
      bad(`${leaf.title ?? key} must be greater than ${leaf.exclusiveMinimum}`);
  }
  return issues;
}

const DEFAULT_TARGETS = {
  sharpeGross: 1.1,
  sharpeNet: 0.6,
  meanIc: 0.032,
  icirAnnual: 1.4,
  decileT: 1.0,
  varTrials: 0.25,
  pbo: null,
} as const;

/** How one override moves the mock's net Sharpe (a story, so variants differ visibly). */
function sharpeShift(overrides: Record<string, ConfigLeaf>): number {
  let d = 0;
  for (const [k, v] of Object.entries(overrides)) {
    const u = ((hashString(`${k}=${String(v)}`) % 1000) / 1000 - 0.5) * 0.3;
    d += k === 'alpha.sample_weight_halflife_years' && typeof v === 'number' && v > 0 ? 0.12 : u;
  }
  return d;
}

/** The spec a request would produce (its config tree gives the hash before launch). */
function specFor(req: LaunchRunRequest, id: string, createdAt: string): RunSpec | null {
  const base = req.base_run_id ? db.runs.get(req.base_run_id)?.spec : undefined;
  if (req.base_run_id && !base) return null;
  const synthetic = req.kind === 'synthetic';
  const sameSlice =
    base !== undefined &&
    base.kind === req.kind &&
    (synthetic
      ? base.stocks === req.stocks && base.days === req.days
      : (base.startDate ?? null) === (req.start ?? null) &&
        (base.source?.end ?? null) === (req.end ?? null) &&
        base.stocks === (req.max_names ?? base.stocks));
  const spanDays =
    req.start && req.end
      ? Math.round(((Date.parse(req.end) - Date.parse(req.start)) / 86_400_000) * (5 / 7))
      : 2520;
  const lagZero = req.signal_lag_days === 0;
  const shift = sharpeShift(req.config_overrides) - (req.ml && !base?.use_ml ? 0.05 : 0);
  const t = base ?? { ...DEFAULT_TARGETS, sharpeNet: 0.35 + ((req.seed * 37) % 60) / 100 };
  const sharpeNet = lagZero ? 2.45 : Math.round((t.sharpeNet + shift) * 100) / 100;
  const spec: RunSpec = {
    id,
    label: req.label ?? `${req.ml ? 'Composite + ML' : 'Composite'}, seed ${req.seed}`,
    created_at: createdAt,
    kind: req.kind,
    phase: req.phase,
    use_ml: req.ml,
    stocks: synthetic ? req.stocks : sameSlice ? base.stocks : (req.max_names ?? 1000),
    days: synthetic ? req.days : sameSlice ? base.days : Math.max(800, spanDays),
    seed: req.seed,
    signal_lag_days: req.signal_lag_days,
    status: 'running',
    sharpeGross: sharpeNet + (t.sharpeGross - t.sharpeNet),
    sharpeNet,
    meanIc: lagZero ? 0.085 : t.meanIc,
    icirAnnual: lagZero ? 4 : t.icirAnnual,
    decileT: lagZero ? 5.8 : t.decileT,
    nTrials: 1,
    varTrials: t.varTrials,
    pbo: t.pbo,
    overrides: { ...base?.overrides, ...req.config_overrides },
    source: {
      origin: 'api',
      max_days: req.max_days ?? null,
      cache: req.cache ?? false,
      ...(synthetic
        ? {}
        : {
            store_path: MOCK_STORE_PATH,
            start: req.start ?? null,
            end: req.end ?? null,
            max_names: req.max_names ?? null,
            min_mcap_ever: req.min_mcap_ever ?? null,
          }),
    },
  };
  if (!synthetic) {
    const startDate = req.start ?? base?.startDate;
    if (startDate) spec.startDate = startDate;
  } else if (base?.startDate && sameSlice) spec.startDate = base.startDate;
  if (base?.investedDayFrac !== undefined) spec.investedDayFrac = base.investedDayFrac;
  if (base?.icLevel !== undefined) spec.icLevel = base.icLevel;
  return spec;
}

/** `POST /api/config/validate` (and the launch's own check). */
export function validate(req: LaunchRunRequest): ConfigValidation {
  const errors: ValidationIssue[] = [];
  const warnings: ValidationIssue[] = [];
  const issue = (loc: string[], msg: string, type = 'value_error') => ({ loc, msg, type });
  if (req.phase === 2 && !req.ml)
    errors.push(issue(['body', 'ml'], 'phase 2 tests the ML sleeve: set ml=true or phase=1'));
  if (req.kind === 'store' && req.start && req.end && req.start > req.end)
    errors.push(issue(['body', 'end'], `end (${req.end}) is before start (${req.start})`));
  const base = req.base_run_id ? db.runs.get(req.base_run_id) : undefined;
  if (req.base_run_id && !base)
    errors.push(
      issue(['body', 'base_run_id'], `no finished run with id ${req.base_run_id} to start from`),
    );
  errors.push(...overrideIssues(req.config_overrides));
  const wu = mockWarmup(req.ml, req.signal_lag_days);
  if (req.kind === 'synthetic' && req.days < wu.min_days)
    errors.push(
      issue(
        ['body', 'days'],
        `days must be at least ${wu.min_days} (${wu.warmup_days} warm-up days for this configuration plus one year)`,
      ),
    );
  const baseHash = base ? base.summary.config_hash : mockConfigHash(defaultSpec());
  const spec = errors.length ? null : specFor(req, 'preview', '2026-01-01T00:00:00Z');
  const hash = spec ? mockConfigHash(spec) : null;
  if (hash) {
    if (req.signal_lag_days === 0)
      warnings.push(
        issue(
          ['body', 'signal_lag_days'],
          'same-close execution (lag 0) trades at the price that produced the signal: optimistic, and the run will be flagged suspect',
          'warning.lag0',
        ),
      );
    const mlKeys = Object.keys(req.config_overrides).filter((k) => k.startsWith('alpha.lgbm.'));
    if (!req.ml && mlKeys.length)
      warnings.push(
        issue(
          ['body', 'config_overrides', mlKeys[0] ?? ''],
          `${mlKeys.join(', ')} only affects the ML sleeve, which is off: the result will not change`,
          'warning.ml_off',
        ),
      );
    const same = [...db.runs.values()].find((r) => r.summary.config_hash === hash);
    if (same)
      warnings.push(
        issue(
          ['body', 'config_overrides'],
          `same configuration as run ${same.spec.id} (${same.spec.label}): on the same data and clean code this is a re-run, not a new trial`,
          'warning.same_config',
        ),
      );
  }
  return {
    valid: errors.length === 0,
    errors,
    warnings,
    config_hash: hash,
    base_config_hash: baseHash,
    warmup_days: wu.warmup_days,
    min_days: wu.min_days,
    overrides_applied: Object.keys(req.config_overrides).length,
  };
}

function defaultSpec(): RunSpec {
  return {
    id: 'defaults',
    label: 'defaults',
    created_at: '2026-01-01T00:00:00Z',
    kind: 'synthetic',
    phase: 1,
    use_ml: false,
    stocks: 1000,
    days: 1512,
    seed: 0,
    signal_lag_days: 1,
    status: 'done',
    nTrials: 1,
    ...DEFAULT_TARGETS,
  };
}

// ----------------------------------------------------------------------------- queue

const STAGES: [number, string, string][] = [
  [0, 'data', 'Building the market data'],
  [0.05, 'universe', 'Building the point-in-time universe'],
  [0.08, 'risk', 'Fitting the risk model'],
  [0.2, 'alpha', 'Fitting the alpha model'],
  [0.35, 'backtest', 'Running the backtest'],
  [0.91, 'evaluate', 'Evaluating gates and writing the trial ledger'],
  [0.96, 'write', 'Writing the run directory'],
];

function stageAt(progress: number): [string, string] {
  let cur: [string, string] = ['data', STAGES[0]?.[2] ?? ''];
  for (const [at, stage, msg] of STAGES) if (progress >= at) cur = [stage, msg];
  return cur;
}

function backtestDays(spec: RunSpec): number {
  return Math.max(120, spec.days - mockWarmup(spec.use_ml, spec.signal_lag_days).warmup_days);
}

function clock(ms: number): string {
  return new Date(ms).toISOString().slice(11, 19);
}

/** The log a job has written by `progress` (stage lines, then the simulator's day lines). */
function logLines(rec: JobRecord, progress: number): string[] {
  if (rec.startedAt === null) return [];
  const t0 = rec.startedAt;
  const at = (p: number) => clock(t0 + p * db.jobDurationMs);
  const lines = [`${at(0)} qts_api.job: job ${rec.job.job_id} -> run ${rec.job.run_id}`];
  const n = backtestDays(rec.spec);
  for (const [p, stage, msg] of STAGES) {
    if (progress < p) break;
    lines.push(`${at(p)} qts_api.job: ${msg}${stage === 'backtest' ? `: day 0 of ${n}` : ''}`);
    if (stage !== 'backtest') continue;
    const upTo = Math.min(progress, 0.9);
    const day = Math.floor(((upTo - 0.35) / 0.55) * n);
    for (let d = 50; d <= day; d += 50) {
      const nav = 5 + (d / n) * 0.9 * (rec.spec.sharpeNet / 2);
      lines.push(
        `${at(0.35 + (d / n) * 0.55)} qts.backtest.simulator: day ${d} nav=${nav.toFixed(3)}M gross=1.98 names=${Math.min(rec.spec.stocks, 412)}`,
      );
    }
  }
  if (progress >= 1)
    lines.push(`${at(1)} qts_api.job: finished: net Sharpe ${rec.spec.sharpeNet.toFixed(2)}`);
  return lines;
}

function finish(rec: JobRecord, finishedAt: number): void {
  const done = generateRun({
    ...rec.spec,
    status: 'done',
    nTrials: trialsInFamily(rec.spec) + 1,
  });
  db.runs.set(rec.spec.id, done);
  db.pending.delete(rec.spec.id);
  const s = done.detail.summary;
  const family = ledgerFamily(rec.spec.phase, done.summary.data_hash);
  db.ledger.trials.push({
    id: `trial-${db.ledger.trials.length + 1}`,
    ts: rec.spec.created_at,
    name: family,
    family,
    config_hash: done.summary.config_hash,
    data_hash: done.summary.data_hash,
    sharpe_annual: s.sharpe_net,
    n_obs: s.n_days,
    mean_ic: done.detail.ic_report.mean_ic,
    n_reruns: 0,
  });
  const fam = db.ledger.families.find((f) => f.family === family);
  if (fam) fam.n_trials += 1;
  else {
    db.ledger.families.push({ family, n_trials: 1, effective_trials: 1, var_sr_annual: 0 });
    db.ledger.families.sort((a, b) => a.family.localeCompare(b.family));
  }
  rec.job = {
    ...rec.job,
    status: 'done',
    progress: 1,
    stage: '',
    message: `Finished: gates evaluated (net Sharpe ${s.sharpe_net.toFixed(2)}, overall ${done.detail.hurdles.overall})`,
    finished_at: new Date(finishedAt).toISOString(),
    eta_seconds: null,
    last_log: logLines(rec, 1).at(-1) ?? null,
  };
}

function trialsInFamily(spec: RunSpec): number {
  const family = ledgerFamily(spec.phase, mockDataHash(spec));
  return db.ledger.families.find((f) => f.family === family)?.n_trials ?? 0;
}

/**
 * Move the queue to `now`: the running job progresses and finishes after `jobDurationMs`;
 * the next queued job starts the moment the previous one ended (FIFO, one at a time).
 */
export function advanceQueue(now = Date.now()): void {
  for (;;) {
    let running = [...db.jobs.values()].find((r) => r.job.status === 'running');
    if (!running) {
      const nextId = db.queue.shift();
      const next = nextId ? db.jobs.get(nextId) : undefined;
      if (!next) break;
      const prevEnd = Math.max(
        0,
        ...[...db.jobs.values()].map((r) =>
          r.job.finished_at && r.job.status === 'done' ? Date.parse(r.job.finished_at) : 0,
        ),
      );
      next.startedAt = Math.min(now, Math.max(Date.parse(next.job.created_at), prevEnd));
      next.job = {
        ...next.job,
        status: 'running',
        position: null,
        started_at: new Date(next.startedAt).toISOString(),
      };
      running = next;
    }
    const t0 = running.startedAt ?? now;
    const progress = Math.min(1, (now - t0) / db.jobDurationMs);
    if (progress >= 1) {
      finish(running, t0 + db.jobDurationMs);
      continue;
    }
    const [stage, msg] = stageAt(progress);
    const n = backtestDays(running.spec);
    const day = Math.round(((Math.min(progress, 0.9) - 0.35) / 0.55) * n);
    const est = db.historySeconds;
    running.job = {
      ...running.job,
      progress,
      stage,
      message: stage === 'backtest' ? `${msg}: day ${day} of ${n}` : msg,
      last_log: logLines(running, progress).at(-1) ?? null,
      estimated_seconds: est,
      eta_seconds: est === null ? null : Math.max(0, est * (1 - progress)),
    };
    break;
  }
  db.queue.forEach((id, i) => {
    const rec = db.jobs.get(id);
    if (rec)
      rec.job = {
        ...rec.job,
        position: i,
        message: i ? `Queued (${i} job${i === 1 ? '' : 's'} ahead)` : 'Queued',
        estimated_seconds: db.historySeconds,
      };
  });
}

export function launch(req: LaunchRunRequest, now = Date.now()): Job {
  db.seq += 1;
  const stamp = new Date(now + db.seq) // distinct, sortable ids for variants queued together
    .toISOString()
    .replace(/[-:.]/g, '')
    .replace(/(\d{3})Z$/, '$1000Z');
  const createdAt = new Date(now + db.seq).toISOString();
  const spec = specFor(req, 'pending', createdAt);
  if (!spec) throw new Error('unknown base run');
  const hash = mockConfigHash(spec);
  spec.id = `${stamp}_${hash.slice(0, 8)}`;
  const jobId = `job_${hashString(`${spec.id}${db.seq}`).toString(16).padStart(12, '0').slice(0, 12)}`;
  const job: Job = {
    job_id: jobId,
    run_id: spec.id,
    status: 'queued',
    progress: 0,
    message: 'Queued',
    label: spec.label,
    kind: spec.kind,
    ml: spec.use_ml,
    phase: spec.phase,
    config_hash: hash,
    base_run_id: req.base_run_id ?? null,
    n_overrides: Object.keys(req.config_overrides).length,
    created_at: createdAt,
    started_at: null,
    finished_at: null,
    position: db.queue.length,
    stage: '',
    last_log: null,
    eta_seconds: null,
    estimated_seconds: db.historySeconds,
  };
  db.jobs.set(jobId, { job, spec, request: req, startedAt: null });
  db.queue.push(jobId);
  db.pending.set(spec.id, placeholder(spec));
  advanceQueue(now);
  return db.jobs.get(jobId)?.job ?? job;
}

export function pollJob(jobId: string, now = Date.now()): Job | undefined {
  advanceQueue(now);
  return db.jobs.get(jobId)?.job;
}

export function listJobs(now = Date.now()): JobQueue {
  advanceQueue(now);
  const jobs = [...db.jobs.values()].map((r) => r.job);
  const rank = (j: Job) => (j.status === 'running' ? 0 : j.status === 'queued' ? 1 : 2);
  jobs.sort(
    (a, b) =>
      rank(a) - rank(b) ||
      (a.status === 'queued'
        ? (a.position ?? 0) - (b.position ?? 0)
        : (b.finished_at ?? '').localeCompare(a.finished_at ?? '')),
  );
  const counts: Record<string, number> = {};
  for (const j of jobs) counts[j.status] = (counts[j.status] ?? 0) + 1;
  return { jobs, max_workers: 1, max_queued: 20, counts };
}

export function cancelJob(jobId: string, now = Date.now()): Job | undefined {
  advanceQueue(now);
  const rec = db.jobs.get(jobId);
  if (!rec) return undefined;
  if (rec.job.status !== 'queued' && rec.job.status !== 'running') return rec.job;
  const wasQueued = rec.job.status === 'queued';
  db.queue = db.queue.filter((id) => id !== jobId);
  rec.job = {
    ...rec.job,
    status: 'cancelled',
    stage: '',
    position: null,
    eta_seconds: null,
    message: wasQueued ? 'Cancelled before it started' : 'Cancelled',
    finished_at: new Date(now).toISOString(),
  };
  db.pending.delete(rec.spec.id); // a cancelled job is not a run
  advanceQueue(now);
  return rec.job;
}

/** `POST /api/jobs/{id}/move`: 'missing' (404), 'not-queued' (409) or the moved job. */
export function moveJob(
  jobId: string,
  position: number,
  now = Date.now(),
): Job | 'missing' | 'not-queued' {
  advanceQueue(now);
  const rec = db.jobs.get(jobId);
  if (!rec) return 'missing';
  if (!db.queue.includes(jobId)) return 'not-queued';
  db.queue = db.queue.filter((id) => id !== jobId);
  db.queue.splice(Math.max(0, Math.min(position, db.queue.length)), 0, jobId);
  advanceQueue(now);
  return db.jobs.get(jobId)?.job ?? rec.job;
}

export function jobLog(jobId: string, tail: number, now = Date.now()): JobLog | undefined {
  advanceQueue(now);
  const rec = db.jobs.get(jobId);
  if (!rec) return undefined;
  const all = logLines(rec, rec.job.status === 'done' ? 1 : rec.job.progress);
  const lines = all.slice(-tail);
  return { job_id: jobId, lines, total_lines: all.length, truncated: all.length > lines.length };
}
