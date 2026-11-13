import { delay, http, HttpResponse } from 'msw';
import type {
  ApiErrorBody,
  ConfigValidation,
  Health,
  Job,
  JobLog,
  JobQueue,
  LaunchRunRequest,
  LaunchRunResponse,
  RunCompare,
  RunFiles,
  RunSeriesName,
} from '@/shared/api';
import { CONFIG_SCHEMA } from './fixtures/configSchema';
import {
  cancelJob,
  db,
  jobLog,
  launch,
  listJobs,
  listSummaries,
  moveJob,
  pollJob,
  validate,
} from './db';
import { MOCK_STORE_PATH, mockWarmup, runFileBody, runFiles } from './fixtures/generate';

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
  'gross_exposure',
  'net_exposure',
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

  http.get('*/api/runs/compare', async ({ request }) => {
    await delay();
    const ids = [
      ...new Set(
        (new URL(request.url).searchParams.get('ids') ?? '')
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
      ),
    ];
    if (ids.length === 0 || ids.length > 8)
      return HttpResponse.json<ApiErrorBody>(
        { detail: 'ids: give 1 to 8 run ids, comma-separated' },
        { status: 422 },
      );
    const found = [];
    for (const id of ids) {
      const r = findRun(id);
      if ('error' in r) return r.error;
      found.push(r.run);
    }
    const dates = [...new Set(found.flatMap((r) => r.dates))].sort();
    const body: RunCompare = {
      dates,
      nav: Object.fromEntries(
        found.map((r) => {
          const byDate = new Map(r.dates.map((d, i) => [d, r.series.nav[i] ?? null]));
          return [r.spec.id, dates.map((d) => byDate.get(d) ?? null)];
        }),
      ),
      runs: found.map((r) => ({
        id: r.spec.id,
        label: r.spec.label,
        created_at: r.spec.created_at,
        phase: r.spec.phase,
        start: r.detail.summary.start,
        end: r.detail.summary.end,
        headline: r.summary.headline ?? {
          sharpe_net: 0,
          sharpe_gross: 0,
          cagr_net: 0,
          max_drawdown: 0,
          avg_cost_bps_per_side: 0,
          dsr: 0,
          invested_day_frac: 0,
          overall_gate: 'FAIL',
        },
        n_trials: r.detail.dsr_inputs.n_trials,
        fold_sharpes: r.detail.fold_sharpes,
        hurdles: r.detail.hurdles,
        config_leaves: r.summary.config_leaves,
      })),
      downsampled: false,
    };
    return HttpResponse.json(body);
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

  http.get<{ id: string }>('*/api/runs/:id/alpha/refits', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    return 'error' in r ? r.error : HttpResponse.json(r.run.alphaRefits);
  }),

  http.get<{ id: string }>('*/api/runs/:id/alpha/theme-weights', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    return 'error' in r ? r.error : HttpResponse.json(r.run.themeWeights);
  }),

  http.get<{ id: string }>('*/api/runs/:id/alpha/ic-by-year', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    return 'error' in r ? r.error : HttpResponse.json(r.run.icByYear);
  }),

  http.get<{ id: string }>('*/api/runs/:id/constraints', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    return 'error' in r ? r.error : HttpResponse.json(r.run.constraints);
  }),

  http.get<{ id: string }>('*/api/runs/:id/files', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    if ('error' in r) return r.error;
    return HttpResponse.json<RunFiles>({ run_id: params.id, files: runFiles(r.run) });
  }),

  http.get<{ id: string; name: string }>('*/api/runs/:id/files/:name', async ({ params }) => {
    await delay();
    const r = findRun(params.id);
    if ('error' in r) return r.error;
    const body = runFileBody(r.run, params.name);
    if (body === null) return notFound(`No file '${params.name}' in run ${params.id}`);
    return new HttpResponse(body.text, {
      headers: {
        'Content-Type': body.type,
        'Content-Disposition': `attachment; filename="${params.id}_${params.name}"`,
      },
    });
  }),

  http.post('*/api/runs', async ({ request }) => {
    await delay();
    const body = (await request.json()) as LaunchRunRequest;
    const v = validate(body);
    if (!v.valid)
      return HttpResponse.json<ApiErrorBody | LaunchRunResponse>(
        {
          title: 'Invalid run request',
          status: 422,
          detail: v.errors.map((e) => `${e.loc.slice(1).join('.')}: ${e.msg}`).join('; '),
          errors: v.errors,
        },
        { status: 422 },
      );
    const job = launch(body);
    return HttpResponse.json<ApiErrorBody | LaunchRunResponse>(
      { job_id: job.job_id, run_id: job.run_id },
      { status: 202 },
    );
  }),

  http.post('*/api/config/validate', async ({ request }) => {
    await delay(60);
    const body = (await request.json()) as LaunchRunRequest;
    return HttpResponse.json<ConfigValidation>(validate(body));
  }),

  http.get('*/api/jobs', async () => {
    await delay(60);
    return HttpResponse.json<JobQueue>(listJobs());
  }),

  http.get<{ jobId: string }>('*/api/jobs/:jobId', async ({ params }) => {
    await delay(60);
    const job = pollJob(params.jobId);
    return job ? HttpResponse.json(job) : notFound(`No job with id ${params.jobId}`);
  }),

  http.get<{ jobId: string }>('*/api/jobs/:jobId/log', async ({ params, request }) => {
    await delay(60);
    const tail = Number(new URL(request.url).searchParams.get('tail') ?? '200');
    const log = jobLog(params.jobId, Number.isFinite(tail) && tail > 0 ? tail : 200);
    return log ? HttpResponse.json<JobLog>(log) : notFound(`No job with id ${params.jobId}`);
  }),

  http.post<{ jobId: string }>('*/api/jobs/:jobId/move', async ({ params, request }) => {
    await delay(60);
    const { position } = (await request.json()) as { position: number };
    const out = moveJob(params.jobId, position);
    if (out === 'missing') return notFound(`No job with id ${params.jobId}`);
    if (out === 'not-queued')
      return HttpResponse.json<ApiErrorBody | Job>(
        { detail: 'Only a queued job can be moved; this one has started or finished.' },
        { status: 409 },
      );
    return HttpResponse.json<ApiErrorBody | Job>(out);
  }),

  http.delete<{ jobId: string }>('*/api/jobs/:jobId', async ({ params }) => {
    await delay(60);
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
      store_path: MOCK_STORE_PATH,
      status: 'ok',
      version: 'mock',
      qts_version: 'mock',
      runs_root: '(in-browser mock)',
      runs_root_exists: true,
      n_runs: db.runs.size,
      jobs: {
        queued: db.queue.length,
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
