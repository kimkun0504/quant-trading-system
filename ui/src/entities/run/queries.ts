import {
  keepPreviousData,
  queryOptions,
  useMutation,
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
