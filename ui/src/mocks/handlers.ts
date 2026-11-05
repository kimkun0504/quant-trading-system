import { delay, http, HttpResponse } from 'msw';
import type {
  ApiErrorBody,
  Health,
  LaunchRunRequest,
  LaunchRunResponse,
  RunSeriesName,
} from '@/shared/api';
import { CONFIG_SCHEMA } from './fixtures/configSchema';
import { cancelJob, db, launch, listSummaries, pollJob } from './db';
import { mockWarmup } from './fixtures/generate';

/**
 * MSW handlers implementing the API contract in shared/api/schema.ts.
 * Paths start with '*' so they match whatever VITE_API_URL points at.
 */

const SERIES_NAMES: readonly RunSeriesName[] = [
  'nav',
  'gross',
  'net',
  'drawdown',
  'turnover',
  'exante_vol',
];

function notFound(detail: string) {
  return HttpResponse.json<ApiErrorBody>({ detail }, { status: 404 });
}

function findRun(id: string) {
  const run = db.runs.get(id);
  if (run) return { run } as const;
  const pending = db.pending.get(id);
  if (pending)
    return {
      error: HttpResponse.json<ApiErrorBody>(
        {
          detail:
            pending.status === 'running'
              ? 'This run is still running; its results appear when the backtest finishes.'
              : 'This run failed before evaluation; no results were written.',
        },
        { status: 409 },
      ),
    } as const;
  return { error: notFound(`No run with id ${id}`) } as const;
}

export const handlers = [
  http.get('*/api/runs', async () => {
    await delay();
    return HttpResponse.json(listSummaries());
  }),

  http.get<{ id: string }>('*/api/runs/:id', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    return 'error' in r ? r.error : HttpResponse.json(r.run.detail);
  }),

  http.get<{ id: string }>('*/api/runs/:id/series', async ({ params, request }) => {
    await delay();
    const r = findRun(params.id);
    if ('error' in r) return r.error;
    const requested = (new URL(request.url).searchParams.get('names') ?? '')
      .split(',')
      .filter((n): n is RunSeriesName => (SERIES_NAMES as readonly string[]).includes(n));
    const names = requested.length ? requested : SERIES_NAMES;
    const series = Object.fromEntries(names.map((n) => [n, r.run.series[n]]));
    return HttpResponse.json({ dates: r.run.dates, series });
  }),

  http.get<{ id: string }>('*/api/runs/:id/costs', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    return 'error' in r ? r.error : HttpResponse.json(r.run.costs);
  }),

  http.get<{ id: string }>('*/api/runs/:id/exposures', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    return 'error' in r ? r.error : HttpResponse.json(r.run.exposures);
  }),

  http.get<{ id: string }>('*/api/runs/:id/holdings', async ({ params, request }) => {
    await delay();
    const r = findRun(params.id);
    if ('error' in r) return r.error;
    const date = new URL(request.url).searchParams.get('date') ?? undefined;
    return HttpResponse.json(r.run.holdingsFor(date));
  }),

  http.get<{ id: string }>('*/api/runs/:id/ic', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    return 'error' in r ? r.error : HttpResponse.json(r.run.ic);
  }),

  http.post('*/api/runs', async ({ request }) => {
    await delay();
    const body = (await request.json()) as LaunchRunRequest;
    const wu = mockWarmup(body.ml, body.signal_lag_days);
    if (body.stocks < 50 || body.days < wu.min_days)
      return HttpResponse.json<ApiErrorBody | LaunchRunResponse>(
        {
          detail: `stocks must be ≥ 50 and days ≥ ${wu.min_days} (${wu.warmup_days} warm-up days + 1 year)`,
        },
        { status: 422 },
      );
    const job = launch(body);
    return HttpResponse.json<ApiErrorBody | LaunchRunResponse>(
      { job_id: job.job_id, run_id: job.run_id },
      { status: 202 },
    );
  }),

  http.get<{ jobId: string }>('*/api/jobs/:jobId', async ({ params }) => {
    await delay(80);
    const job = pollJob(params.jobId);
    return job ? HttpResponse.json(job) : notFound(`No job with id ${params.jobId}`);
  }),

  http.delete<{ jobId: string }>('*/api/jobs/:jobId', async ({ params }) => {
    await delay(80);
    const job = cancelJob(params.jobId);
    return job ? HttpResponse.json(job) : notFound(`No job with id ${params.jobId}`);
  }),

  http.get('*/api/config/warmup', ({ request }) => {
    const q = new URL(request.url).searchParams;
    return HttpResponse.json(
      mockWarmup(q.get('ml') === 'true', q.get('signal_lag_days') === '0' ? 0 : 1),
    );
  }),

  http.get('*/api/health', () =>
    HttpResponse.json<Health>({
      status: 'ok',
      version: 'mock',
      qts_version: 'mock',
      runs_root: '(in-browser mock)',
      runs_root_exists: true,
      n_runs: db.runs.size,
      jobs: {
        queued: 0,
        running: [...db.jobs.values()].filter((j) => j.job.status === 'running').length,
      },
    }),
  ),

  http.get('*/api/ledger', async () => {
    await delay();
    return HttpResponse.json(db.ledger);
  }),

  http.get('*/api/config/schema', async () => {
    await delay();
    return HttpResponse.json(CONFIG_SCHEMA);
  }),
];
