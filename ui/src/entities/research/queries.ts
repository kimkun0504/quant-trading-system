import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/shared/api';
import type {
  CapacityRequest,
  Job,
  MidcapRequest,
  PaperRunRequest,
  RiskCalibrationRequest,
  SignalTestRequest,
  StoreSlice,
  SweepCreate,
  SweepRequest,
} from '@/shared/api';
import { queueKeys } from '@/entities/run';

/**
 * Query keys of the research tools and the paper books. Results written by the library
 * never change once written (a sweep manifest does while its members run, so the sweep
 * and the paper status poll while a job of theirs is active).
 */
export const researchKeys = {
  all: ['research'] as const,
  seal: () => ['research', 'seal'] as const,
  unseals: (hash: string, phase: 1 | 2) => ['research', 'unseals', hash, phase] as const,
  impact: (family: string, n: number, ref: number) =>
    ['research', 'impact', family, n, ref] as const,
  family: (store: StoreSlice) => ['research', 'family', store] as const,
  features: (set: string) => ['research', 'features', set] as const,
  signals: () => ['research', 'signals'] as const,
  signalTests: () => ['research', 'signal-tests'] as const,
  signalTest: (id: string) => ['research', 'signal-tests', id] as const,
  signalPreview: (body: SignalTestRequest) => ['research', 'signal-preview', body] as const,
  sweeps: () => ['research', 'sweeps'] as const,
  sweep: (id: string) => ['research', 'sweeps', id] as const,
  sweepPlan: (body: SweepRequest) => ['research', 'sweep-plan', body] as const,
  capacity: () => ['research', 'capacity'] as const,
  capacityOne: (id: string) => ['research', 'capacity', id] as const,
  risk: () => ['research', 'risk'] as const,
  riskOne: (id: string) => ['research', 'risk', id] as const,
  midcap: () => ['research', 'midcap'] as const,
  midcapResult: (id: string) => ['research', 'midcap', 'result', id] as const,
  preregistration: () => ['research', 'midcap', 'preregistration'] as const,
};

export const paperKeys = {
  all: ['paper'] as const,
  books: () => ['paper', 'books'] as const,
  status: (b: string) => ['paper', b, 'status'] as const,
  history: (b: string) => ['paper', b, 'history'] as const,
  orders: (b: string, d: string | undefined) => ['paper', b, 'orders', d ?? 'latest'] as const,
  holdings: (b: string, d: string | undefined) => ['paper', b, 'holdings', d ?? 'latest'] as const,
  schedule: (b: string) => ['paper', b, 'schedule'] as const,
};

const active = (jobs: readonly Pick<Job, 'status'>[] | undefined) =>
  (jobs ?? []).some((j) => j.status === 'queued' || j.status === 'running');

export const POLL_MS = 2000;

// ----------------------------------------------------------------------------- seal
export function useSeal() {
  return useQuery({
    queryKey: researchKeys.seal(),
    queryFn: ({ signal }) => api.getSeal(signal),
    staleTime: 30_000,
  });
}

export function useUnsealCheck(configHash: string | null, phase: 1 | 2) {
  return useQuery({
    queryKey: researchKeys.unseals(configHash ?? '', phase),
    queryFn: ({ signal }) => api.getUnsealCheck(configHash ?? '', phase, signal),
    enabled: Boolean(configHash),
  });
}

export function useTrialImpact(family: string | null, nNew: number, nReference = 0) {
  return useQuery({
    queryKey: researchKeys.impact(family ?? '', nNew, nReference),
    queryFn: ({ signal }) => api.getTrialImpact(family ?? '', nNew, nReference, signal),
    enabled: Boolean(family),
    placeholderData: keepPreviousData,
  });
}

export function useFeatures(featureSet: string | null) {
  return useQuery({
    queryKey: researchKeys.features(featureSet ?? 'v1'),
    queryFn: ({ signal }) => api.getFeatures(featureSet ?? 'v1', signal),
    staleTime: Infinity,
    enabled: featureSet !== null,
  });
}

/** Queue the job that resolves a store slice's trial family (about a minute, once). */
export function useResolveFamily() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (store: StoreSlice) => api.launchResolveFamily(store),
    onSuccess: () => void qc.invalidateQueries({ queryKey: queueKeys.list() }),
  });
}

// ----------------------------------------------------------------------------- signals
export function useSignals() {
  return useQuery({
    queryKey: researchKeys.signals(),
    queryFn: ({ signal }) => api.listSignals(signal),
    staleTime: Infinity,
  });
}

export function useSignalTests() {
  return useQuery({
    queryKey: researchKeys.signalTests(),
    queryFn: ({ signal }) => api.listSignalTests(signal),
  });
}

export function useSignalTest(id: string) {
  return useQuery({
    queryKey: researchKeys.signalTest(id),
    queryFn: ({ signal }) => api.getSignalTest(id, signal),
  });
}

/** The trial preview of a signal test, keyed by the request (debounced by the caller). */
export function useSignalPreview(body: SignalTestRequest | undefined) {
  return useQuery({
    queryKey: researchKeys.signalPreview(body as SignalTestRequest),
    queryFn: ({ signal }) => api.previewSignalTest(body as SignalTestRequest, signal),
    enabled: body !== undefined,
    placeholderData: keepPreviousData,
    retry: false,
  });
}

function useLaunch<B, R>(fn: (b: B) => Promise<R>, keys: readonly (readonly unknown[])[]) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: queueKeys.list() });
      for (const k of keys) void qc.invalidateQueries({ queryKey: k });
    },
  });
}

export function useLaunchSignalTest() {
  return useLaunch((b: SignalTestRequest) => api.launchSignalTest(b), [researchKeys.signalTests()]);
}

// ----------------------------------------------------------------------------- sweeps
export function useSweeps() {
  return useQuery({
    queryKey: researchKeys.sweeps(),
    queryFn: ({ signal }) => api.listSweeps(signal),
  });
}

export function useSweep(id: string) {
  return useQuery({
    queryKey: researchKeys.sweep(id),
    queryFn: ({ signal }) => api.getSweep(id, signal),
    refetchInterval: (q) => (active(q.state.data?.jobs) ? POLL_MS : false),
  });
}

export function useSweepPlan(body: SweepRequest | undefined) {
  return useQuery({
    queryKey: researchKeys.sweepPlan(body as SweepRequest),
    queryFn: ({ signal }) => api.planSweep(body as SweepRequest, signal),
    enabled: body !== undefined,
    retry: false,
    placeholderData: keepPreviousData,
  });
}

export function useCreateSweep() {
  return useLaunch((b: SweepCreate) => api.createSweep(b), [researchKeys.sweeps()]);
}

export function useResumeSweep(id: string) {
  return useLaunch(() => api.resumeSweep(id), [researchKeys.sweep(id)]);
}

// ----------------------------------------------------------------------------- capacity / risk
export function useCapacityList() {
  return useQuery({
    queryKey: researchKeys.capacity(),
    queryFn: ({ signal }) => api.listCapacity(signal),
  });
}

export function useCapacity(id: string) {
  return useQuery({
    queryKey: researchKeys.capacityOne(id),
    queryFn: ({ signal }) => api.getCapacity(id, signal),
    staleTime: Infinity,
  });
}

export function useLaunchCapacity() {
  return useLaunch((b: CapacityRequest) => api.launchCapacity(b), [researchKeys.capacity()]);
}

export function useRiskList() {
  return useQuery({ queryKey: researchKeys.risk(), queryFn: ({ signal }) => api.listRisk(signal) });
}

export function useRisk(id: string) {
  return useQuery({
    queryKey: researchKeys.riskOne(id),
    queryFn: ({ signal }) => api.getRisk(id, signal),
    staleTime: Infinity,
  });
}

export function useLaunchRisk() {
  return useLaunch((b: RiskCalibrationRequest) => api.launchRisk(b), [researchKeys.risk()]);
}

// ----------------------------------------------------------------------------- mid-cap
export function useMidcap() {
  return useQuery({
    queryKey: researchKeys.midcap(),
    queryFn: ({ signal }) => api.getMidcap(signal),
    refetchInterval: (q) => (active(q.state.data?.jobs) ? POLL_MS : false),
  });
}

export function useMidcapResult(runId: string) {
  return useQuery({
    queryKey: researchKeys.midcapResult(runId),
    queryFn: ({ signal }) => api.getMidcapResult(runId, signal),
    staleTime: Infinity,
  });
}

export function usePreregistration(enabled = true) {
  return useQuery({
    queryKey: researchKeys.preregistration(),
    queryFn: ({ signal }) => api.getPreregistration(signal),
    staleTime: Infinity,
    enabled,
  });
}

export function useLaunchMidcap() {
  return useLaunch((b: MidcapRequest) => api.launchMidcap(b), [researchKeys.midcap()]);
}

// ----------------------------------------------------------------------------- paper
export function usePaperBooks() {
  return useQuery({ queryKey: paperKeys.books(), queryFn: ({ signal }) => api.listBooks(signal) });
}

export function usePaperStatus(book: string) {
  return useQuery({
    queryKey: paperKeys.status(book),
    queryFn: ({ signal }) => api.getPaperStatus(book, signal),
    refetchInterval: (q) => (active(q.state.data?.jobs) ? POLL_MS : false),
  });
}

export function usePaperHistory(book: string) {
  return useQuery({
    queryKey: paperKeys.history(book),
    queryFn: ({ signal }) => api.getPaperHistory(book, signal),
  });
}

export function usePaperOrders(book: string, date: string | undefined) {
  return useQuery({
    queryKey: paperKeys.orders(book, date),
    queryFn: ({ signal }) => api.getPaperOrders(book, date, signal),
    placeholderData: keepPreviousData,
  });
}

export function usePaperHoldings(book: string, date: string | undefined) {
  return useQuery({
    queryKey: paperKeys.holdings(book, date),
    queryFn: ({ signal }) => api.getPaperHoldings(book, date, signal),
    placeholderData: keepPreviousData,
  });
}

export function usePaperSchedule(book: string) {
  return useQuery({
    queryKey: paperKeys.schedule(book),
    queryFn: ({ signal }) => api.getPaperSchedule(book, signal),
    staleTime: Infinity,
  });
}

export function usePaperAck(book: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (note: string) => api.paperAck(book, note),
    onSuccess: () => void qc.invalidateQueries({ queryKey: paperKeys.all }),
  });
}

export function usePaperRun(book: string) {
  return useLaunch((b: PaperRunRequest) => api.paperRun(book, b), [paperKeys.status(book)]);
}

export function usePaperReconcile(book: string) {
  return useLaunch(() => api.paperReconcile(book), [paperKeys.status(book)]);
}
