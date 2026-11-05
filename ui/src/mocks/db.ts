import type { Job, LaunchRunRequest, Ledger, RunSummary } from '@/shared/api';
import { generateRun } from './fixtures/generate';
import type { GeneratedRun, RunSpec } from './fixtures/generate';
import { seedLedger } from './fixtures/ledger';
import { FAILED_SPEC, RUNNING_SPEC, SEED_SPECS } from './fixtures/runs';
import { hashString } from './fixtures/random';

/** In-memory registry behind the MSW handlers. `resetDb()` restores the seed state. */
interface JobRecord {
  job: Job;
  spec: RunSpec;
  startedAt: number;
}

export interface MockDb {
  runs: Map<string, GeneratedRun>;
  pending: Map<string, RunSummary>;
  jobs: Map<string, JobRecord>;
  ledger: Ledger;
  jobDurationMs: number;
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
  return { runs, pending, jobs: new Map(), ledger: seedLedger(), jobDurationMs: 4000 };
}

export let db: MockDb = create();

export function resetDb(overrides: Partial<Pick<MockDb, 'jobDurationMs'>> = {}): void {
  db = { ...create(), ...overrides };
}

export function listSummaries(): RunSummary[] {
  const all = [...[...db.runs.values()].map((r) => r.summary), ...db.pending.values()];
  return all.sort((a, b) => b.created_at.localeCompare(a.created_at));
}

const STAGES: [number, string][] = [
  [0, 'Building the point-in-time universe'],
  [0.08, 'Fitting the risk model'],
  [0.2, 'Fitting the alpha model'],
  [0.35, 'Running the backtest'],
  [0.9, 'Evaluating gates and writing the trial ledger'],
];

function stageMessage(progress: number, spec: RunSpec): string {
  let msg = STAGES[0]?.[1] ?? '';
  for (const [at, m] of STAGES) if (progress >= at) msg = m;
  if (msg === 'Running the backtest') {
    const n = spec.days - 512;
    const day = Math.round(((progress - 0.35) / 0.55) * n);
    return `${msg}: day ${day} of ${n}`;
  }
  return msg;
}

export function launch(req: LaunchRunRequest, now = Date.now()): Job {
  const stamp = new Date(now)
    .toISOString()
    .replace(/[-:]/g, '')
    .replace(/\.\d+Z$/, 'Z');
  const h = hashString(JSON.stringify(req)).toString(16).padStart(8, '0').slice(0, 8);
  const id = `${stamp}_${h}`;
  const lagZero = req.signal_lag_days === 0;
  const base = 0.35 + ((req.seed * 37) % 60) / 100 - (req.ml ? 0.05 : 0);
  const sharpeNet = lagZero ? 2.45 : base;
  const family = `pipeline_phase${req.phase}`;
  const nTrials = db.ledger.trials.filter((t) => t.family === family).length + 1;
  const spec: RunSpec = {
    id,
    label: `${req.ml ? 'Composite + ML' : 'Composite'}, seed ${req.seed}`,
    created_at: new Date(now).toISOString().replace(/\.\d+Z$/, 'Z'),
    kind: req.kind,
    phase: req.phase,
    use_ml: req.ml,
    stocks: req.stocks,
    days: req.days,
    seed: req.seed,
    signal_lag_days: req.signal_lag_days,
    status: 'running',
    sharpeGross: sharpeNet + 0.5,
    sharpeNet,
    meanIc: lagZero ? 0.085 : 0.02 + sharpeNet * 0.02,
    icirAnnual: lagZero ? 4 : 0.8 + sharpeNet,
    decileT: lagZero ? 5.8 : 0.6 + sharpeNet * 1.4,
    nTrials,
    varTrials: 0.25,
    pbo: null,
  };
  const job: Job = {
    job_id: `job_${h}`,
    run_id: id,
    status: 'queued',
    progress: 0,
    message: 'Queued',
  };
  db.jobs.set(job.job_id, { job, spec, startedAt: now });
  db.pending.set(id, placeholder(spec));
  return job;
}

export function pollJob(jobId: string, now = Date.now()): Job | undefined {
  const rec = db.jobs.get(jobId);
  if (!rec) return undefined;
  if (rec.job.status === 'done' || rec.job.status === 'failed') return rec.job;
  const progress = Math.min(1, (now - rec.startedAt) / db.jobDurationMs);
  if (progress >= 1) {
    const done = generateRun({ ...rec.spec, status: 'done' });
    db.runs.set(rec.spec.id, done);
    db.pending.delete(rec.spec.id);
    const s = done.detail.summary;
    db.ledger.trials.push({
      id: `trial-${db.ledger.trials.length + 1}`,
      ts: rec.spec.created_at,
      name: `pipeline_phase${rec.spec.phase}`,
      family: `pipeline_phase${rec.spec.phase}`,
      config_hash: done.summary.config_hash,
      data_hash: done.summary.data_hash,
      sharpe_annual: s.sharpe_net,
      n_obs: s.n_days,
      mean_ic: done.detail.ic_report.mean_ic,
      n_reruns: 0,
    });
    rec.job = { ...rec.job, status: 'done', progress: 1, message: 'Finished: gates evaluated' };
    return rec.job;
  }
  rec.job = {
    ...rec.job,
    status: 'running',
    progress,
    message: stageMessage(progress, rec.spec),
  };
  return rec.job;
}

export function cancelJob(jobId: string): Job | undefined {
  const rec = db.jobs.get(jobId);
  if (!rec) return undefined;
  if (rec.job.status === 'done' || rec.job.status === 'failed') return rec.job;
  rec.job = { ...rec.job, status: 'failed', message: 'Cancelled' };
  const pending = db.pending.get(rec.spec.id);
  if (pending) db.pending.set(rec.spec.id, { ...pending, status: 'failed' });
  return rec.job;
}
