import { beforeEach, describe, expect, it } from 'vitest';
import type { Job, LaunchRunRequest } from '@/shared/api';
import {
  advanceQueue,
  cancelJob,
  db,
  jobLog,
  launch,
  listJobs,
  moveJob,
  resetDb,
  validate,
} from '@/mocks/db';
import {
  formatDuration,
  jobStatusLine,
  logLineText,
  queueCounts,
  queueReducer,
  queueSummary,
  sortQueue,
  stageLabel,
} from './lib';

const job = (id: string, status: Job['status'], extra: Partial<Job> = {}): Job => ({
  task: 'pipeline',
  subject: null,
  command: null,
  result: null,
  error_kind: null,
  exit_code: null,
  job_id: id,
  run_id: `run_${id}`,
  status,
  progress: status === 'done' ? 1 : 0,
  message: '',
  label: `Run ${id}`,
  kind: 'synthetic',
  ml: false,
  phase: 1,
  config_hash: 'abc',
  base_run_id: null,
  n_overrides: 0,
  created_at: '2026-01-01T00:00:00Z',
  started_at: null,
  finished_at: null,
  position: null,
  stage: '',
  last_log: null,
  eta_seconds: null,
  estimated_seconds: null,
  ...extra,
});

const ids = (jobs: readonly Job[]) =>
  jobs.map((j) => `${j.job_id}:${j.status}:${j.position ?? '-'}`);

describe('queue reducer', () => {
  const state = [
    job('r', 'running', { started_at: '2026-01-01T00:01:00Z' }),
    job('a', 'queued', { position: 0 }),
    job('b', 'queued', { position: 1 }),
    job('c', 'queued', { position: 2 }),
    job('d', 'done', { finished_at: '2026-01-01T00:00:30Z' }),
  ];

  it('cancels a queued job and closes the gap', () => {
    const next = queueReducer(state, { type: 'cancel', jobId: 'b', now: '2026-01-01T00:02:00Z' });
    expect(ids(next)).toEqual([
      'r:running:-',
      'a:queued:0',
      'c:queued:1',
      'b:cancelled:-',
      'd:done:-',
    ]);
    expect(next.find((j) => j.job_id === 'b')?.message).toBe('Cancelled before it started');
    expect(next.find((j) => j.job_id === 'c')?.message).toBe('Queued (1 ahead)');
    expect(state[2]?.status).toBe('queued'); // the input is not modified
  });

  it('cancels the running job, and leaves finished jobs alone', () => {
    const next = queueReducer(state, { type: 'cancel', jobId: 'r', now: '2026-01-01T00:02:00Z' });
    expect(next.find((j) => j.job_id === 'r')).toMatchObject({
      status: 'cancelled',
      message: 'Cancelled',
      stage: '',
    });
    expect(queueReducer(state, { type: 'cancel', jobId: 'd' })).toEqual(state);
    expect(queueReducer(state, { type: 'cancel', jobId: 'nope' })).toEqual(state);
  });

  it('moves a queued job, clamped to the queue', () => {
    expect(ids(queueReducer(state, { type: 'move', jobId: 'c', position: 0 })).slice(1, 4)).toEqual(
      ['c:queued:0', 'a:queued:1', 'b:queued:2'],
    );
    expect(
      ids(queueReducer(state, { type: 'move', jobId: 'a', position: 99 })).slice(1, 4),
    ).toEqual(['b:queued:0', 'c:queued:1', 'a:queued:2']);
    expect(queueReducer(state, { type: 'move', jobId: 'r', position: 0 })).toEqual(state); // not queued
  });

  it('upserts a job the API returned and keeps the listing order', () => {
    const started = job('a', 'running', { started_at: '2026-01-01T00:03:00Z' });
    const next = queueReducer(state, { type: 'upsert', job: started });
    expect(next.slice(0, 2).map((j) => j.job_id)).toEqual(['r', 'a']);
    const added = queueReducer(state, { type: 'upsert', job: job('e', 'queued', { position: 3 }) });
    expect(added.map((j) => j.job_id)).toEqual(['r', 'a', 'b', 'c', 'e', 'd']);
    expect(sortQueue([...state].reverse()).map((j) => j.job_id)).toEqual(['r', 'a', 'b', 'c', 'd']);
  });

  it('summarises and words the queue', () => {
    expect(queueCounts(state)).toEqual({ running: 1, queued: 3, finished: 1 });
    expect(queueSummary(state)).toBe('1 running, 3 queued');
    expect(queueSummary([job('d', 'done')])).toBe('Queue is empty');
    expect([
      formatDuration(12),
      formatDuration(200),
      formatDuration(4800),
      formatDuration(null),
    ]).toEqual(['about 10 s', 'about 3 min', 'about 1 h 20 min', '']);
    expect(stageLabel('backtest')).toBe('Backtest');
    expect(logLineText('12:00:01 qts.backtest.simulator: day 50 nav=5.1M')).toBe('day 50 nav=5.1M');
    expect(
      jobStatusLine(
        job('r', 'running', { message: 'Running the backtest: day 300 of 700', eta_seconds: 130 }),
      ),
    ).toBe('Running the backtest: day 300 of 700, about 2 min left');
    expect(jobStatusLine(job('a', 'queued', { position: 0, estimated_seconds: 240 }))).toBe(
      'Next to run, takes about 4 min',
    );
    expect(jobStatusLine(job('b', 'queued', { position: 2 }))).toBe('2 ahead');
  });
});

describe('mock queue (the MSW backend)', () => {
  const T0 = Date.parse('2026-11-12T10:00:00Z');
  const req = (overrides: LaunchRunRequest['config_overrides'] = {}): LaunchRunRequest => ({
    kind: 'synthetic',
    ml: false,
    phase: 1,
    stocks: 300,
    days: 1512,
    seed: 0,
    signal_lag_days: 1,
    config_overrides: overrides,
    base_run_id: '20261104T175159Z_944dd5b4',
  });
  beforeEach(() => resetDb({ jobDurationMs: 1000 }));

  it('runs one job at a time, in order, and progresses with time', () => {
    const a = launch(req({ 'alpha.sample_weight_halflife_years': 8 }), T0);
    const b = launch(req({ 'alpha.sample_weight_halflife_years': 4 }), T0);
    const status = (now: number) =>
      listJobs(now).jobs.map((j) => `${j.status}:${j.position ?? '-'}`);
    expect(status(T0)).toEqual(['running:-', 'queued:0']);
    const mid = listJobs(T0 + 500).jobs[0];
    expect(mid).toMatchObject({ job_id: a.job_id, stage: 'backtest', eta_seconds: 120 });
    expect(mid?.message).toMatch(/^Running the backtest: day \d+ of \d+$/);
    expect(mid?.last_log).toMatch(/qts\.backtest\.simulator: day \d+ nav=/);
    // a finishes at T0 + 1 s; b starts then and finishes one duration later
    const later = listJobs(T0 + 1500).jobs;
    expect(later.map((j) => [j.job_id, j.status])).toEqual([
      [b.job_id, 'running'],
      [a.job_id, 'done'],
    ]);
    expect(later[0]?.progress).toBeCloseTo(0.5, 1);
    expect(status(T0 + 2500)).toEqual(['done:-', 'done:-']);
    // the finished runs are in the registry, with their overrides and a new ledger trial each
    const run = db.runs.get(a.run_id);
    expect(run?.summary.config_leaves['alpha.sample_weight_halflife_years']).toBe(8);
    expect(run?.detail.config.source).toMatchObject({
      origin: 'api',
      kind: 'synthetic',
      stocks: 300,
    });
    expect(run?.summary.config_hash).toBe(a.config_hash);
    expect(db.pending.has(a.run_id)).toBe(false);
  });

  it('cancels and reorders queued jobs', () => {
    const a = launch(req({ 'alpha.horizon_days': 5 }), T0);
    const b = launch(req({ 'alpha.horizon_days': 10 }), T0);
    const c = launch(req({ 'alpha.horizon_days': 15 }), T0);
    expect(moveJob(c.job_id, 0, T0)).toMatchObject({ position: 0 });
    expect(moveJob(a.job_id, 0, T0)).toBe('not-queued'); // running
    expect(moveJob('job_nope', 0, T0)).toBe('missing');
    expect(cancelJob(b.job_id, T0)).toMatchObject({
      status: 'cancelled',
      message: 'Cancelled before it started',
    });
    expect(db.pending.has(b.run_id)).toBe(false);
    advanceQueue(T0 + 1100);
    expect(db.jobs.get(c.job_id)?.job.status).toBe('running');
    expect(cancelJob(c.job_id, T0 + 1200)).toMatchObject({
      status: 'cancelled',
      message: 'Cancelled',
    });
    expect(listJobs(T0 + 5000).counts).toEqual({ done: 1, cancelled: 2 });
  });

  it('serves the log tail of a job', () => {
    const a = launch(req(), T0);
    expect(jobLog(a.job_id, 200, T0)?.lines).toHaveLength(2);
    const log = jobLog(a.job_id, 3, T0 + 2000);
    expect(log).toMatchObject({ truncated: true });
    expect(log?.lines).toHaveLength(3);
    expect(log?.lines.at(-1)).toMatch(/finished: net Sharpe/);
    expect(jobLog('job_nope', 10)).toBeUndefined();
  });

  it('validates like the API: errors per key, warnings, the hash of the run', () => {
    const ok = validate(req({ 'alpha.sample_weight_halflife_years': 8 }));
    expect(ok).toMatchObject({
      valid: true,
      errors: [],
      warnings: [],
      warmup_days: 506,
      min_days: 758,
    });
    expect(ok.config_hash).not.toBe(ok.base_config_hash);
    const same = validate(req());
    expect(same.config_hash).toBe(same.base_config_hash);
    expect(same.warnings.map((w) => w.type)).toEqual(['warning.same_config']);
    const bad = validate(
      req({ 'portfolio.trade_aversion': 50, 'nope.x': 1, 'alpha.horizon_days': null }),
    );
    expect(bad.valid).toBe(false);
    expect(bad.config_hash).toBeNull();
    expect(bad.errors.map((e) => e.loc[2])).toEqual([
      'portfolio.trade_aversion',
      'nope.x',
      'alpha.horizon_days',
    ]);
    expect(validate({ ...req(), days: 600 }).errors[0]?.loc).toEqual(['body', 'days']);
    expect(validate({ ...req(), base_run_id: 'missing' }).errors[0]?.loc).toEqual([
      'body',
      'base_run_id',
    ]);
    expect(validate({ ...req(), signal_lag_days: 0 }).warnings.map((w) => w.type)).toContain(
      'warning.lag0',
    );
    expect(validate(req({ 'portfolio.target_vol_annual': null })).valid).toBe(true);
  });
});
