import {
  keepPreviousData,
  queryOptions,
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { useRef } from 'react';
import { api } from '@/shared/api';
import type { Job, JobQueue, LaunchRunRequest, RunId, RunSeriesName } from '@/shared/api';

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

/** `useRun` for an id that may not be chosen yet (the builder's "start from a run"). */
export function useOptionalRun(id: RunId | null) {
  return useQuery({ ...runQuery(id ?? ''), enabled: id !== null });
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

export const queueKeys = {
  all: ['jobs'] as const,
  list: () => ['jobs', 'list'] as const,
  log: (jobId: string) => ['jobs', 'log', jobId] as const,
};

const isActiveJob = (j: Pick<Job, 'status'>) => j.status === 'queued' || j.status === 'running';

/** Queue polling: fast while something runs or waits, slow when idle. */
export const QUEUE_POLL_ACTIVE_MS = 1000;
export const QUEUE_POLL_IDLE_MS = 15_000;

/**
 * `GET /api/jobs`, polled. When a job reaches a terminal state the runs list and the
 * ledger are refreshed (a finished run appears in Experiments, the trial count moves).
 */
export function useQueue() {
  const qc = useQueryClient();
  const seen = useRef<Map<string, Job['status']>>(new Map());
  return useQuery({
    queryKey: queueKeys.list(),
    queryFn: async ({ signal }) => {
      const q = await api.listJobs(signal);
      let settled = false;
      for (const j of q.jobs) {
        const before = seen.current.get(j.job_id);
        if (before !== undefined && before !== j.status) settled = true;
        seen.current.set(j.job_id, j.status);
      }
      if (settled) {
        void qc.invalidateQueries({ queryKey: runKeys.all });
        void qc.invalidateQueries({ queryKey: ['ledger'] });
      }
      return q;
    },
    refetchInterval: (q) =>
      q.state.data?.jobs.some(isActiveJob) ? QUEUE_POLL_ACTIVE_MS : QUEUE_POLL_IDLE_MS,
  });
}

/** `GET /api/jobs/{id}/log`: the tail of a job's log, refreshed while the job runs. */
export function useJobLog(jobId: string, opts: { enabled: boolean; live: boolean; tail?: number }) {
  return useQuery({
    queryKey: queueKeys.log(jobId),
    queryFn: ({ signal }) => api.getJobLog(jobId, opts.tail ?? 200, signal),
    enabled: opts.enabled,
    refetchInterval: opts.enabled && opts.live ? 2000 : false,
  });
}

/** Adds a run to the queue (`POST /api/runs`). */
export function useLaunchRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: LaunchRunRequest) => api.launchRun(body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queueKeys.list() });
      return qc.invalidateQueries({ queryKey: runKeys.list() });
    },
  });
}

/** Cancel a job / move a queued job; `apply` edits the cached listing before the API answers. */
export function useCancelJob(apply: (jobs: Job[], jobId: string) => Job[]) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.cancelJob(jobId),
    onMutate: (jobId) => {
      qc.setQueryData<JobQueue>(queueKeys.list(), (q) =>
        q ? { ...q, jobs: apply(q.jobs, jobId) } : q,
      );
    },
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: queueKeys.list() });
      void qc.invalidateQueries({ queryKey: runKeys.list() });
    },
  });
}

export function useMoveJob(apply: (jobs: Job[], jobId: string, position: number) => Job[]) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, position }: { jobId: string; position: number }) =>
      api.moveJob(jobId, position),
    onMutate: ({ jobId, position }) => {
      qc.setQueryData<JobQueue>(queueKeys.list(), (q) =>
        q ? { ...q, jobs: apply(q.jobs, jobId, position) } : q,
      );
    },
    onSettled: () => void qc.invalidateQueries({ queryKey: queueKeys.list() }),
  });
}

/**
 * `POST /api/config/validate` for a launch body: errors, warnings, the resulting config
 * hash and warm-up. Keyed by the body, so each distinct edit is validated once.
 */
export function useValidateConfig(body: LaunchRunRequest | undefined) {
  return useQuery({
    queryKey: ['config-validate', body] as const,
    queryFn: ({ signal }) => api.validateConfig(body as LaunchRunRequest, signal),
    enabled: body !== undefined,
    staleTime: 30_000,
    placeholderData: keepPreviousData,
    retry: false,
  });
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => api.getHealth(signal),
    staleTime: 60_000,
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
