import { apiGet, apiPost } from './client';
import type {
  ConfigSchema,
  Job,
  LaunchRunRequest,
  LaunchRunResponse,
  Ledger,
  ColumnarSeries,
  RunCosts,
  RunDetail,
  RunExposures,
  RunHoldings,
  RunIc,
  RunId,
  RunSeriesName,
  RunSummary,
} from './schema';

/** One function per endpoint; the only place that knows URL shapes. */
export const api = {
  listRuns: (signal?: AbortSignal) => apiGet<RunSummary[]>('/api/runs', undefined, signal),
  getRun: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunDetail>(`/api/runs/${encodeURIComponent(id)}`, undefined, signal),
  getRunSeries: <N extends RunSeriesName>(id: RunId, names: readonly N[], signal?: AbortSignal) =>
    apiGet<ColumnarSeries<N>>(
      `/api/runs/${encodeURIComponent(id)}/series`,
      { names: names.join(',') },
      signal,
    ),
  getRunCosts: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunCosts>(`/api/runs/${encodeURIComponent(id)}/costs`, undefined, signal),
  getRunExposures: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunExposures>(`/api/runs/${encodeURIComponent(id)}/exposures`, undefined, signal),
  getRunHoldings: (id: RunId, date: string | undefined, signal?: AbortSignal) =>
    apiGet<RunHoldings>(`/api/runs/${encodeURIComponent(id)}/holdings`, { date }, signal),
  getRunIc: (id: RunId, signal?: AbortSignal) =>
    apiGet<RunIc>(`/api/runs/${encodeURIComponent(id)}/ic`, undefined, signal),
  launchRun: (body: LaunchRunRequest) => apiPost<LaunchRunResponse>('/api/runs', body),
  getJob: (jobId: string, signal?: AbortSignal) =>
    apiGet<Job>(`/api/jobs/${encodeURIComponent(jobId)}`, undefined, signal),
  getLedger: (signal?: AbortSignal) => apiGet<Ledger>('/api/ledger', undefined, signal),
  getConfigSchema: (signal?: AbortSignal) =>
    apiGet<ConfigSchema>('/api/config/schema', undefined, signal),
};

export type Api = typeof api;
