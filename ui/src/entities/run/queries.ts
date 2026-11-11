import {
  keepPreviousData,
  queryOptions,
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { api } from '@/shared/api';
import type { LaunchRunRequest, RunId, RunSeriesName } from '@/shared/api';

/**
 * Query keys are hierarchical so one invalidation clears a run's dependants:
 * ['runs'] ⊃ ['runs', 'detail', id] ⊃ ['runs', 'detail', id, 'series', 'nav,net'].
 */
export const runKeys = {
  all: ['runs'] as const,
  list: () => [...runKeys.all, 'list'] as const,
  detail: (id: RunId) => [...runKeys.all, 'detail', id] as const,
  series: (id: RunId, names: readonly RunSeriesName[]) =>
    [...runKeys.detail(id), 'series', names.join(',')] as const,
  costs: (id: RunId) => [...runKeys.detail(id), 'costs'] as const,
  exposures: (id: RunId) => [...runKeys.detail(id), 'exposures'] as const,
  holdings: (id: RunId, date: string | undefined) =>
    [...runKeys.detail(id), 'holdings', date ?? 'latest'] as const,
  ic: (id: RunId) => [...runKeys.detail(id), 'ic'] as const,
  files: (id: RunId) => [...runKeys.detail(id), 'files'] as const,
  alphaRefits: (id: RunId) => [...runKeys.detail(id), 'alpha', 'refits'] as const,
  themeWeights: (id: RunId) => [...runKeys.detail(id), 'alpha', 'theme-weights'] as const,
  icByYear: (id: RunId) => [...runKeys.detail(id), 'alpha', 'ic-by-year'] as const,
  constraints: (id: RunId) => [...runKeys.detail(id), 'constraints'] as const,
  compare: (ids: readonly RunId[]) => [...runKeys.all, 'compare', ids.join(',')] as const,
};

export const jobKeys = {
  detail: (jobId: string) => ['jobs', jobId] as const,
};

export const runsQuery = () =>
  queryOptions({ queryKey: runKeys.list(), queryFn: ({ signal }) => api.listRuns(signal) });

export const runQuery = (id: RunId) =>
  queryOptions({ queryKey: runKeys.detail(id), queryFn: ({ signal }) => api.getRun(id, signal) });

export function useRuns() {
  return useQuery(runsQuery());
}

export function useRun(id: RunId) {
  return useQuery(runQuery(id));
}

export function useRunSeries<N extends RunSeriesName>(id: RunId, names: readonly N[]) {
  return useQuery({
    queryKey: runKeys.series(id, names),
    queryFn: ({ signal }) => api.getRunSeries(id, names, signal),
    staleTime: Infinity, // a finished run's series never change
  });
}

export function useRunCosts(id: RunId) {
  return useQuery({
    queryKey: runKeys.costs(id),
    queryFn: ({ signal }) => api.getRunCosts(id, signal),
    staleTime: Infinity,
  });
}

export function useRunExposures(id: RunId) {
  return useQuery({
    queryKey: runKeys.exposures(id),
    queryFn: ({ signal }) => api.getRunExposures(id, signal),
    staleTime: Infinity,
  });
}

export function useRunHoldings(id: RunId, date: string | undefined) {
  return useQuery({
    queryKey: runKeys.holdings(id, date),
    queryFn: ({ signal }) => api.getRunHoldings(id, date, signal),
    staleTime: Infinity,
    placeholderData: keepPreviousData,
  });
}

export function useRunIc(id: RunId) {
  return useQuery({
    queryKey: runKeys.ic(id),
    queryFn: ({ signal }) => api.getRunIc(id, signal),
    staleTime: Infinity,
  });
}

/** `GET /api/runs/{id}/files`: what is in the run directory, with download links. */
export function useRunFiles(id: RunId) {
  return useQuery({
    queryKey: runKeys.files(id),
    queryFn: ({ signal }) => api.getRunFiles(id, signal),
    staleTime: Infinity,
  });
}

/** `GET /api/runs/{id}/alpha/refits`: per-refit feature importance, windows, feature IC. */
export function useAlphaRefits(id: RunId) {
  return useQuery({
    queryKey: runKeys.alphaRefits(id),
    queryFn: ({ signal }) => api.getAlphaRefits(id, signal),
    staleTime: Infinity,
  });
}

/** `GET /api/runs/{id}/alpha/theme-weights`: the composite's theme weights by date. */
export function useThemeWeights(id: RunId) {
  return useQuery({
    queryKey: runKeys.themeWeights(id),
    queryFn: ({ signal }) => api.getThemeWeights(id, signal),
    staleTime: Infinity,
  });
}

/** `GET /api/runs/{id}/alpha/ic-by-year`: mean rank IC and t per sleeve per year. */
export function useIcByYear(id: RunId) {
  return useQuery({
    queryKey: runKeys.icByYear(id),
    queryFn: ({ signal }) => api.getIcByYear(id, signal),
    staleTime: Infinity,
  });
}

/** `GET /api/runs/{id}/constraints`: the per-rebalance constraint log. */
export function useRunConstraints(id: RunId) {
  return useQuery({
    queryKey: runKeys.constraints(id),
    queryFn: ({ signal }) => api.getRunConstraints(id, signal),
    staleTime: Infinity,
  });
}

/** The same per-run query for several runs at once (labs with several lanes), in order. */
export function useManyRunIc(ids: readonly RunId[]) {
  return useQueries({
    queries: ids.map((id) => ({
      queryKey: runKeys.ic(id),
      queryFn: ({ signal }: { signal: AbortSignal }) => api.getRunIc(id, signal),
      staleTime: Infinity,
    })),
  });
}

export function useManyRunSeries<N extends RunSeriesName>(
  ids: readonly RunId[],
  names: readonly N[],
) {
  return useQueries({
    queries: ids.map((id) => ({
      queryKey: runKeys.series(id, names),
      queryFn: ({ signal }: { signal: AbortSignal }) => api.getRunSeries(id, names, signal),
      staleTime: Infinity,
    })),
  });
}

/** Holdings of one run on several dates (the industry-over-time stack). */
export function useManyRunHoldings(id: RunId, dates: readonly string[]) {
  return useQueries({
    queries: dates.map((d) => ({
      queryKey: runKeys.holdings(id, d),
      queryFn: ({ signal }: { signal: AbortSignal }) => api.getRunHoldings(id, d, signal),
      staleTime: Infinity,
    })),
  });
}

export function useManyRunConstraints(ids: readonly RunId[]) {
  return useQueries({
    queries: ids.map((id) => ({
      queryKey: runKeys.constraints(id),
      queryFn: ({ signal }: { signal: AbortSignal }) => api.getRunConstraints(id, signal),
      staleTime: Infinity,
    })),
  });
}

/** `GET /api/runs/compare?ids=`: aligned NAV, folds, gates and flattened configs. */
export function useCompareRuns(ids: readonly RunId[]) {
  return useQuery({
    queryKey: runKeys.compare(ids),
    queryFn: ({ signal }) => api.compareRuns(ids, signal),
    enabled: ids.length > 0,
    staleTime: Infinity,
    placeholderData: keepPreviousData,
  });
}

export function useLaunchRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: LaunchRunRequest) => api.launchRun(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: runKeys.list() }),
  });
}

export const JOB_POLL_MS = 750;

export function useJob(jobId: string | undefined) {
  const qc = useQueryClient();
  return useQuery({
    queryKey: jobKeys.detail(jobId ?? 'none'),
    queryFn: async ({ signal }) => {
      const job = await api.getJob(jobId ?? '', signal);
      if (job.status === 'done' || job.status === 'failed') {
        void qc.invalidateQueries({ queryKey: runKeys.all });
      }
      return job;
    },
    enabled: Boolean(jobId),
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === 'done' || s === 'failed' ? false : JOB_POLL_MS;
    },
  });
}

export function useLedger() {
  return useQuery({ queryKey: ['ledger'], queryFn: ({ signal }) => api.getLedger(signal) });
}

export function useConfigSchema() {
  return useQuery({
    queryKey: ['config-schema'],
    queryFn: ({ signal }) => api.getConfigSchema(signal),
    staleTime: Infinity,
  });
}

/** Effective warm-up for the default config (`GET /api/config/warmup`); the launch form's `days` floor. */
export function useWarmup(ml: boolean, signalLagDays: 0 | 1) {
  return useQuery({
    queryKey: ['config-warmup', ml, signalLagDays],
    queryFn: ({ signal }) => api.getWarmup(ml, signalLagDays, signal),
    staleTime: Infinity,
  });
}
