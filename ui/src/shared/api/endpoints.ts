import { apiDelete, apiGet, apiPost } from './client';
import type {
  AlphaIcByYear,
  AlphaRefits,
  AlphaThemeWeights,
  ConfigSchema,
  ConfigValidation,
  Health,
  Job,
  JobLog,
  JobQueue,
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
import type {
  CapacityList,
  CapacityRequest,
  CapacityResult,
  FamilyResolution,
  FeatureList,
  MidcapPreregistration,
  MidcapRequest,
  MidcapResultDetail,
  MidcapStatus,
  PaperAck,
  PaperBooks,
  PaperHistory,
  PaperHoldings,
  PaperOrders,
  PaperRunRequest,
  PaperSchedule,
  PaperStatus,
  RiskCalibrationList,
  RiskCalibrationRequest,
  RiskCalibrationResult,
  SealStatus,
  SignalDef,
  SignalTestDetail,
  SignalTestList,
  SignalTestPreview,
  SignalTestRequest,
  StoreSlice,
  SweepCreate,
  SweepDetail,
  SweepLaunchResponse,
  SweepList,
  SweepPlanView,
  SweepRequest,
  TaskLaunchResponse,
  TrialImpact,
  UnsealCheck,
} from './research';

const enc = encodeURIComponent;

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
  /** Dry run of a launch: errors, warnings, the resulting config hash and warm-up. */
  validateConfig: (body: LaunchRunRequest, signal?: AbortSignal) =>
    apiPost<ConfigValidation>('/config/validate', body, signal),
  listJobs: (signal?: AbortSignal) => apiGet<JobQueue>('/jobs', undefined, signal),
  getJobLog: (jobId: string, tail: number, signal?: AbortSignal) =>
    apiGet<JobLog>(`/jobs/${encodeURIComponent(jobId)}/log`, { tail: String(tail) }, signal),
  moveJob: (jobId: string, position: number) =>
    apiPost<Job>(`/jobs/${encodeURIComponent(jobId)}/move`, { position }),
  getHealth: (signal?: AbortSignal) => apiGet<Health>('/health', undefined, signal),

  // ---- the holdout seal, trial families
  getSeal: (signal?: AbortSignal) => apiGet<SealStatus>('/seal', undefined, signal),
  getUnsealCheck: (configHash: string, phase: 1 | 2, signal?: AbortSignal) =>
    apiGet<UnsealCheck>('/seal/unseals', { config_hash: configHash, phase }, signal),
  getTrialImpact: (family: string, nNew: number, nReference = 0, signal?: AbortSignal) =>
    apiGet<TrialImpact>('/ledger/impact', { family, n_new: nNew, n_reference: nReference }, signal),
  resolveFamily: (body: StoreSlice, signal?: AbortSignal) =>
    apiPost<FamilyResolution>('/families/resolve', body, signal),
  launchResolveFamily: (body: StoreSlice) =>
    apiPost<TaskLaunchResponse>('/families/resolve/jobs', body),
  getFeatures: (featureSet: string, signal?: AbortSignal) =>
    apiGet<FeatureList>('/features', { feature_set: featureSet }, signal),

  // ---- signal tests
  listSignals: (signal?: AbortSignal) => apiGet<SignalDef[]>('/signals', undefined, signal),
  listSignalTests: (signal?: AbortSignal) =>
    apiGet<SignalTestList>('/signals/tests', undefined, signal),
  getSignalTest: (id: string, signal?: AbortSignal) =>
    apiGet<SignalTestDetail>(`/signals/tests/${enc(id)}`, undefined, signal),
  previewSignalTest: (body: SignalTestRequest, signal?: AbortSignal) =>
    apiPost<SignalTestPreview>('/signals/tests/preview', body, signal),
  launchSignalTest: (body: SignalTestRequest) =>
    apiPost<TaskLaunchResponse>('/signals/tests', body),

  // ---- sweeps
  planSweep: (body: SweepRequest, signal?: AbortSignal) =>
    apiPost<SweepPlanView>('/sweeps/plan', body, signal),
  createSweep: (body: SweepCreate) => apiPost<SweepLaunchResponse>('/sweeps', body),
  listSweeps: (signal?: AbortSignal) => apiGet<SweepList>('/sweeps', undefined, signal),
  getSweep: (id: string, signal?: AbortSignal) =>
    apiGet<SweepDetail>(`/sweeps/${enc(id)}`, undefined, signal),
  resumeSweep: (id: string) => apiPost<SweepLaunchResponse>(`/sweeps/${enc(id)}/resume`, {}),

  // ---- capacity, risk calibration
  listCapacity: (signal?: AbortSignal) => apiGet<CapacityList>('/capacity', undefined, signal),
  getCapacity: (id: string, signal?: AbortSignal) =>
    apiGet<CapacityResult>(`/capacity/${enc(id)}`, undefined, signal),
  launchCapacity: (body: CapacityRequest) => apiPost<TaskLaunchResponse>('/capacity', body),
  listRisk: (signal?: AbortSignal) =>
    apiGet<RiskCalibrationList>('/risk-calibrations', undefined, signal),
  getRisk: (id: string, signal?: AbortSignal) =>
    apiGet<RiskCalibrationResult>(`/risk-calibrations/${enc(id)}`, undefined, signal),
  launchRisk: (body: RiskCalibrationRequest) =>
    apiPost<TaskLaunchResponse>('/risk-calibrations', body),

  // ---- the mid-cap check
  getMidcap: (signal?: AbortSignal) => apiGet<MidcapStatus>('/checks/midcap', undefined, signal),
  getMidcapResult: (runId: string, signal?: AbortSignal) =>
    apiGet<MidcapResultDetail>(`/checks/midcap/results/${enc(runId)}`, undefined, signal),
  getPreregistration: (signal?: AbortSignal) =>
    apiGet<MidcapPreregistration>('/checks/midcap/preregistration', undefined, signal),
  launchMidcap: (body: MidcapRequest) => apiPost<TaskLaunchResponse>('/checks/midcap', body),

  // ---- paper books
  listBooks: (signal?: AbortSignal) => apiGet<PaperBooks>('/paper/books', undefined, signal),
  getPaperStatus: (name: string, signal?: AbortSignal) =>
    apiGet<PaperStatus>(`/paper/books/${enc(name)}/status`, undefined, signal),
  getPaperHistory: (name: string, signal?: AbortSignal) =>
    apiGet<PaperHistory>(`/paper/books/${enc(name)}/history`, undefined, signal),
  getPaperOrders: (name: string, date: string | undefined, signal?: AbortSignal) =>
    apiGet<PaperOrders>(`/paper/books/${enc(name)}/orders`, { date }, signal),
  getPaperHoldings: (name: string, date: string | undefined, signal?: AbortSignal) =>
    apiGet<PaperHoldings>(`/paper/books/${enc(name)}/holdings`, { date }, signal),
  getPaperSchedule: (name: string, signal?: AbortSignal) =>
    apiGet<PaperSchedule>(`/paper/books/${enc(name)}/schedule`, undefined, signal),
  paperAck: (name: string, note: string) =>
    apiPost<PaperAck>(`/paper/books/${enc(name)}/ack`, { note }),
  paperRun: (name: string, body: PaperRunRequest) =>
    apiPost<TaskLaunchResponse>(`/paper/books/${enc(name)}/run`, body),
  paperReconcile: (name: string) =>
    apiPost<TaskLaunchResponse>(`/paper/books/${enc(name)}/reconcile`, {}),
};

export type Api = typeof api;
