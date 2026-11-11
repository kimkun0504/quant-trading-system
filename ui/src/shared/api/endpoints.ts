import { apiDelete, apiGet, apiPost } from './client';
import type {
  AlphaIcByYear,
  AlphaRefits,
  AlphaThemeWeights,
  ConfigSchema,
  Health,
  Job,
  LaunchRunRequest,
  LaunchRunResponse,
  Ledger,
  ColumnarSeries,
  RunCompare,
  RunConstraints,
  RunCosts,
  RunDetail,
  RunExposures,
  RunFiles,
  RunHoldings,
  RunIc,
  RunId,
  RunSeriesName,
  RunSummary,
  Warmup,
} from './schema';

/** One function per endpoint; the only place that knows URL shapes (relative to API_BASE_URL). */
export const api = {
  listRuns: (signal?: AbortSignal) => apiGet<RunSummary[]>('/runs', undefined, signal),
  getRun: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunDetail>(`/runs/${encodeURIComponent(id)}`, undefined, signal),
  getRunSeries: <N extends RunSeriesName>(id: RunId, names: readonly N[], signal?: AbortSignal) =>
    apiGet<ColumnarSeries<N>>(
      `/runs/${encodeURIComponent(id)}/series`,
      { names: names.join(',') },
      signal,
    ),
  compareRuns: (ids: readonly RunId[], signal?: AbortSignal) =>
    apiGet<RunCompare>('/runs/compare', { ids: ids.join(',') }, signal),
  getRunCosts: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunCosts>(`/runs/${encodeURIComponent(id)}/costs`, undefined, signal),
  getRunExposures: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunExposures>(`/runs/${encodeURIComponent(id)}/exposures`, undefined, signal),
  getRunHoldings: (id: RunId, date: string | undefined, signal?: AbortSignal) =>
    apiGet<RunHoldings>(`/runs/${encodeURIComponent(id)}/holdings`, { date }, signal),
  getRunIc: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunIc>(`/runs/${encodeURIComponent(id)}/ic`, undefined, signal),
  getRunFiles: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunFiles>(`/runs/${encodeURIComponent(id)}/files`, undefined, signal),
  getAlphaRefits: (id: RunId, signal?: AbortSignal) =>
    apiGet<AlphaRefits>(`/runs/${encodeURIComponent(id)}/alpha/refits`, undefined, signal),
  getThemeWeights: (id: RunId, signal?: AbortSignal) =>
    apiGet<AlphaThemeWeights>(
      `/runs/${encodeURIComponent(id)}/alpha/theme-weights`,
      undefined,
      signal,
    ),
  getIcByYear: (id: RunId, signal?: AbortSignal) =>
    apiGet<AlphaIcByYear>(`/runs/${encodeURIComponent(id)}/alpha/ic-by-year`, undefined, signal),
  getRunConstraints: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunConstraints>(`/runs/${encodeURIComponent(id)}/constraints`, undefined, signal),
  launchRun: (body: LaunchRunRequest) => apiPost<LaunchRunResponse>('/runs', body),
  getJob: (jobId: string, signal?: AbortSignal) =>
    apiGet<Job>(`/jobs/${encodeURIComponent(jobId)}`, undefined, signal),
  getLedger: (signal?: AbortSignal) => apiGet<Ledger>('/ledger', undefined, signal),
  getConfigSchema: (signal?: AbortSignal) =>
    apiGet<ConfigSchema>('/config/schema', undefined, signal),
  getWarmup: (ml: boolean, signalLagDays: 0 | 1, signal?: AbortSignal) =>
    apiGet<Warmup>(
      '/config/warmup',
      { ml: String(ml), signal_lag_days: String(signalLagDays) },
      signal,
    ),
  cancelJob: (jobId: string) => apiDelete<Job>(`/jobs/${encodeURIComponent(jobId)}`),
  getHealth: (signal?: AbortSignal) => apiGet<Health>('/health', undefined, signal),
};

export type Api = typeof api;
