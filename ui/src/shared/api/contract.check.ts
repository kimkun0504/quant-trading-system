/**
 * Compile-time contract test: the hand-written types in `schema.ts` against the types
 * `openapi-typescript` generated from the API's OpenAPI document (`generated.ts`,
 * regenerate with `npm run api:types`). Nothing here runs; `npm run typecheck` fails when
 * the two drift apart.
 *
 * - `Same<A, B>`: mutually assignable (the shapes agree exactly).
 * - `Sub<A, B>`: A is assignable to B. Used where the UI narrows a generic shape
 *   (`ColumnarSeries<N>`, `ConfigSchema`'s recursive union), and for the request body,
 *   where what the UI sends must satisfy what the API accepts.
 * - The endpoint section ties each `api.*` call's declared result type to the response
 *   type of the matching OpenAPI operation, so a wrong URL-to-type pairing fails too.
 */
import type { components, paths } from './generated';
import type * as S from './schema';
import type * as R from './research';

type G = components['schemas'];
type IsNever<A> = [A] extends [never] ? true : false;
type Sub<A, B> = IsNever<A> extends true ? false : [A] extends [B] ? true : false;
type Same<A, B> = Sub<A, B> extends true ? Sub<B, A> : false;
type Assert<T extends true> = T;

type Op<P extends keyof paths, M extends keyof paths[P]> = paths[P][M];

/** JSON body of the 200 (or 202) response of an operation. */
type Json<P extends keyof paths, M extends keyof paths[P]> =
  Op<P, M> extends { responses: { 200: { content: { 'application/json': infer B } } } }
    ? B
    : Op<P, M> extends { responses: { 202: { content: { 'application/json': infer B } } } }
      ? B
      : never;

/** JSON request body of an operation. */
type Body<P extends keyof paths, M extends keyof paths[P]> =
  Op<P, M> extends { requestBody: { content: { 'application/json': infer B } } } ? B : never;

/** problem+json body of an error response `C` of an operation. */
type ErrorBody<P extends keyof paths, M extends keyof paths[P], C extends number> =
  Op<P, M> extends { responses: Record<C, { content: { 'application/problem+json': infer B } }> }
    ? B
    : never;

/** Schemas: response shapes are identical on both sides. */
export type SchemaContract = [
  Assert<Same<G['RunHeadline'], S.RunHeadline>>,
  Assert<Same<G['RunSummary'], S.RunSummary>>,
  Assert<Same<G['BacktestSummary'], S.BacktestSummary>>,
  Assert<Same<G['RunManifest'], S.RunManifest>>,
  Assert<Same<G['GateResult'], S.GateResult>>,
  Assert<Same<G['Hurdles'], S.Hurdles>>,
  Assert<Same<G['IcByYear'], S.IcByYear>>,
  Assert<Same<G['IcReport'], S.IcReport>>,
  Assert<Same<G['DecileSpread'], S.DecileSpread>>,
  Assert<Same<G['DsrInputs'], S.DsrInputs>>,
  Assert<Same<G['FoldSharpe'], S.FoldSharpe>>,
  Assert<Same<G['RunSource'], S.RunSource>>,
  Assert<Same<G['RunConfig'], S.RunConfig>>,
  Assert<Same<G['RunDetail'], S.RunDetail>>,
  Assert<Same<G['RunCompareRun'], S.RunCompareRun>>,
  Assert<Same<G['RunCompare'], S.RunCompare>>,
  Assert<Same<G['ColumnarSeries'], S.ColumnarSeries>>,
  Assert<Same<G['CostSensitivityRow'], S.CostSensitivityRow>>,
  Assert<Same<G['RunCosts'], S.RunCosts>>,
  Assert<Same<G['ExposureSnapshotRow'], S.ExposureSnapshotRow>>,
  Assert<Same<G['ExposureSnapshot'], S.ExposureSnapshot>>,
  Assert<Same<G['RunExposures'], S.RunExposures>>,
  Assert<Same<G['Holding'], S.Holding>>,
  Assert<Same<G['RunHoldings'], S.RunHoldings>>,
  Assert<Same<G['IcDecayPoint'], S.IcDecayPoint>>,
  Assert<Same<G['RunIc'], S.RunIc>>,
  Assert<Same<G['LaunchRunResponse'], S.LaunchRunResponse>>,
  Assert<Same<G['Job'], S.Job>>,
  Assert<Same<G['JobQueue'], S.JobQueue>>,
  Assert<Same<G['JobLog'], S.JobLog>>,
  Assert<Same<G['JobMove'], S.JobMove>>,
  Assert<Same<G['ConfigValidation'], S.ConfigValidation>>,
  Assert<Same<G['LedgerTrial'], S.LedgerTrial>>,
  Assert<Same<G['LedgerFamily'], S.LedgerFamily>>,
  Assert<Same<G['Ledger'], S.Ledger>>,
  Assert<Same<G['JsonSchemaNumber'], S.JsonSchemaNumber>>,
  Assert<Same<G['JsonSchemaBoolean'], S.JsonSchemaBoolean>>,
  Assert<Same<G['JsonSchemaString'], S.JsonSchemaString>>,
  Assert<Same<G['JsonSchemaArray'], S.JsonSchemaArray>>,
  Assert<Same<G['RunSeal'], S.RunSeal>>,
  Assert<Sub<G['JsonSchemaObject'], S.ConfigSchema>>,
  Assert<Same<G['ValidationIssue'], S.ValidationIssue>>,
  Assert<Same<G['ApiErrorBody'], S.ApiErrorBody>>,
  Assert<Same<G['Health'], S.Health>>,
  Assert<Same<G['RunFile'], S.RunFile>>,
  Assert<Same<G['RunFiles'], S.RunFiles>>,
  Assert<Same<G['Warmup'], S.Warmup>>,
  Assert<Same<G['AlphaRefitWindow'], S.AlphaRefitWindow>>,
  Assert<Same<G['AlphaFeatureIc'], S.AlphaFeatureIc>>,
  Assert<Same<G['AlphaRefits'], S.AlphaRefits>>,
  Assert<Same<G['AlphaThemeWeights'], S.AlphaThemeWeights>>,
  Assert<Same<G['IcByYearCell'], S.IcByYearCell>>,
  Assert<Same<G['IcByYearRow'], S.IcByYearRow>>,
  Assert<Same<G['AlphaIcByYear'], S.AlphaIcByYear>>,
  Assert<Same<G['RunConstraints'], S.RunConstraints>>,
  // the request body: everything the form sends is accepted, and nothing more is required
  Assert<Sub<S.LaunchRunRequest, G['LaunchRunRequest']>>,
  Assert<Sub<G['LaunchRunRequest'], S.LaunchRunRequest>>,
];

/** Endpoints: each `api.*` result type is what the matching operation returns. */
export type EndpointContract = [
  Assert<Same<Json<'/api/runs', 'get'>, S.RunSummary[]>>,
  Assert<Same<Json<'/api/runs/{run_id}', 'get'>, S.RunDetail>>,
  Assert<Same<Json<'/api/runs/compare', 'get'>, S.RunCompare>>,
  Assert<Same<ErrorBody<'/api/runs/compare', 'get', 422>, S.ApiErrorBody>>,
  Assert<Sub<S.ColumnarSeries<S.RunSeriesName>, Json<'/api/runs/{run_id}/series', 'get'>>>,
  Assert<Same<Json<'/api/runs/{run_id}/costs', 'get'>, S.RunCosts>>,
  Assert<Same<Json<'/api/runs/{run_id}/exposures', 'get'>, S.RunExposures>>,
  Assert<Same<Json<'/api/runs/{run_id}/holdings', 'get'>, S.RunHoldings>>,
  Assert<Same<Json<'/api/runs/{run_id}/ic', 'get'>, S.RunIc>>,
  Assert<Same<Json<'/api/runs/{run_id}/files', 'get'>, S.RunFiles>>,
  Assert<Same<Json<'/api/runs/{run_id}/alpha/refits', 'get'>, S.AlphaRefits>>,
  Assert<Same<Json<'/api/runs/{run_id}/alpha/theme-weights', 'get'>, S.AlphaThemeWeights>>,
  Assert<Same<Json<'/api/runs/{run_id}/alpha/ic-by-year', 'get'>, S.AlphaIcByYear>>,
  Assert<Same<Json<'/api/runs/{run_id}/constraints', 'get'>, S.RunConstraints>>,
  Assert<Same<Json<'/api/runs', 'post'>, S.LaunchRunResponse>>,
  Assert<Sub<S.LaunchRunRequest, Body<'/api/runs', 'post'>>>,
  Assert<Same<Json<'/api/jobs/{job_id}', 'get'>, S.Job>>,
  Assert<Same<Json<'/api/jobs/{job_id}', 'delete'>, S.Job>>,
  Assert<Same<Json<'/api/jobs', 'get'>, S.JobQueue>>,
  Assert<Same<Json<'/api/jobs/{job_id}/log', 'get'>, S.JobLog>>,
  Assert<Same<Json<'/api/jobs/{job_id}/move', 'post'>, S.Job>>,
  Assert<Sub<S.JobMove, Body<'/api/jobs/{job_id}/move', 'post'>>>,
  Assert<Same<Json<'/api/config/validate', 'post'>, S.ConfigValidation>>,
  Assert<Sub<S.LaunchRunRequest, Body<'/api/config/validate', 'post'>>>,
  Assert<Same<Json<'/api/ledger', 'get'>, S.Ledger>>,
  Assert<Sub<Json<'/api/config/schema', 'get'>, S.ConfigSchema>>,
  Assert<Same<Json<'/api/health', 'get'>, S.Health>>,
  Assert<Same<Json<'/api/config/warmup', 'get'>, S.Warmup>>,
  // research tools and paper books (the types are the generated shapes; these tie each
  // api.* call to its operation, so a wrong URL-to-type pairing fails)
  Assert<Same<Json<'/api/seal', 'get'>, R.SealStatus>>,
  Assert<Same<Json<'/api/seal/unseals', 'get'>, R.UnsealCheck>>,
  Assert<Same<Json<'/api/ledger/impact', 'get'>, R.TrialImpact>>,
  Assert<Same<Json<'/api/families/resolve', 'post'>, R.FamilyResolution>>,
  Assert<Same<Json<'/api/families/resolve/jobs', 'post'>, R.TaskLaunchResponse>>,
  Assert<Same<Json<'/api/features', 'get'>, R.FeatureList>>,
  Assert<Same<Json<'/api/signals', 'get'>, R.SignalDef[]>>,
  Assert<Same<Json<'/api/signals/tests', 'get'>, R.SignalTestList>>,
  Assert<Same<Json<'/api/signals/tests', 'post'>, R.TaskLaunchResponse>>,
  // what the API declares is accepted by the UI's request types (defaulted fields optional)
  Assert<Sub<Body<'/api/signals/tests', 'post'>, R.SignalTestRequest>>,
  Assert<Same<Json<'/api/signals/tests/{test_id}', 'get'>, R.SignalTestDetail>>,
  Assert<Same<Json<'/api/signals/tests/preview', 'post'>, R.SignalTestPreview>>,
  Assert<Same<Json<'/api/sweeps/plan', 'post'>, R.SweepPlanView>>,
  Assert<Same<Json<'/api/sweeps', 'post'>, R.SweepLaunchResponse>>,
  Assert<Sub<Body<'/api/sweeps', 'post'>, R.SweepCreate>>,
  Assert<Sub<Body<'/api/capacity', 'post'>, R.CapacityRequest>>,
  Assert<Sub<Body<'/api/risk-calibrations', 'post'>, R.RiskCalibrationRequest>>,
  Assert<Sub<Body<'/api/checks/midcap', 'post'>, R.MidcapRequest>>,
  Assert<Sub<Body<'/api/paper/books/{name}/run', 'post'>, R.PaperRunRequest>>,
  Assert<Same<Json<'/api/sweeps', 'get'>, R.SweepList>>,
  Assert<Same<Json<'/api/sweeps/{sweep_id}', 'get'>, R.SweepDetail>>,
  Assert<Same<Json<'/api/sweeps/{sweep_id}/resume', 'post'>, R.SweepLaunchResponse>>,
  Assert<Same<Json<'/api/capacity', 'get'>, R.CapacityList>>,
  Assert<Same<Json<'/api/capacity', 'post'>, R.TaskLaunchResponse>>,
  Assert<Same<Json<'/api/capacity/{result_id}', 'get'>, R.CapacityResult>>,
  Assert<Same<Json<'/api/risk-calibrations', 'get'>, R.RiskCalibrationList>>,
  Assert<Same<Json<'/api/risk-calibrations', 'post'>, R.TaskLaunchResponse>>,
  Assert<Same<Json<'/api/risk-calibrations/{result_id}', 'get'>, R.RiskCalibrationResult>>,
  Assert<Same<Json<'/api/checks/midcap', 'get'>, R.MidcapStatus>>,
  Assert<Same<Json<'/api/checks/midcap', 'post'>, R.TaskLaunchResponse>>,
  Assert<Same<Json<'/api/checks/midcap/results/{run_id}', 'get'>, R.MidcapResultDetail>>,
  Assert<Same<Json<'/api/checks/midcap/preregistration', 'get'>, R.MidcapPreregistration>>,
  Assert<Same<Json<'/api/paper/books', 'get'>, R.PaperBooks>>,
  Assert<Same<Json<'/api/paper/books/{name}/status', 'get'>, R.PaperStatus>>,
  Assert<Same<Json<'/api/paper/books/{name}/history', 'get'>, R.PaperHistory>>,
  Assert<Same<Json<'/api/paper/books/{name}/orders', 'get'>, R.PaperOrders>>,
  Assert<Same<Json<'/api/paper/books/{name}/holdings', 'get'>, R.PaperHoldings>>,
  Assert<Same<Json<'/api/paper/books/{name}/schedule', 'get'>, R.PaperSchedule>>,
  Assert<Same<Json<'/api/paper/books/{name}/ack', 'post'>, R.PaperAck>>,
  Assert<Same<Json<'/api/paper/books/{name}/run', 'post'>, R.TaskLaunchResponse>>,
  Assert<Same<Json<'/api/paper/books/{name}/reconcile', 'post'>, R.TaskLaunchResponse>>,
  // a launch's job is a Job: the research jobs carry task, subject, command and result
  Assert<Same<G['Job'], S.Job>>,
  // error responses are problem documents
  Assert<Same<ErrorBody<'/api/runs/{run_id}', 'get', 404>, S.ApiErrorBody>>,
  Assert<Same<ErrorBody<'/api/runs/{run_id}', 'get', 409>, S.ApiErrorBody>>,
  Assert<Same<ErrorBody<'/api/runs', 'post', 422>, S.ApiErrorBody>>,
];
