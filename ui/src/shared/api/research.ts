/**
 * Research tools and paper books: the API serves the library's own JSON (`to_dict()` of the
 * research dataclasses, `qts.paper.status()` and friends) through permissive Pydantic
 * models, so these types are the generated ones (`generated.ts`, `npm run api:types`)
 * under friendly names rather than hand-written mirrors: the shapes are the library's, and
 * a change on the Python side changes them here at the next regeneration. Numbers that are
 * not finite in Python arrive as `null`, and the types say so (`number | null`).
 */
import type { components } from './generated';

type G = components['schemas'];

/**
 * A request body with its defaulted fields optional: the API fills them (Pydantic
 * defaults), but openapi-typescript types every field that has a default as required.
 */
type Loose<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

export type TaskLaunchResponse = G['TaskLaunchResponse'];
export type StoreSlice = Partial<G['StoreSlice']>;
export type StoreSpecView = G['StoreSpecView'];

export type SealInfo = G['SealInfo'];
export type SealUnseal = G['SealUnseal'];
export type SealStatus = G['SealStatus'];
export type UnsealCheck = G['UnsealCheck'];
export type TrialImpact = G['TrialImpact'];
export type FamilyResolution = G['FamilyResolution'];

export type FeatureInfo = G['FeatureInfo'];
export type FeatureTheme = G['FeatureTheme'];
export type FeatureList = G['FeatureList'];

export type SignalDef = G['SignalDef'];
export type IcStats = G['IcStats'];
export type PaperBookStats = G['PaperBookStats'];
export type SignalFamilyTest = G['SignalFamilyTest'];
export type SignalTestResult = G['SignalTestResult'];
export type SignalFamilyNow = G['SignalFamilyNow'];
export type SignalTestDetail = G['SignalTestDetail'];
export type SignalTestSummary = G['SignalTestSummary'];
export type SignalTestList = G['SignalTestList'];
export type SignalTestRequest = Loose<
  G['SignalTestRequest'],
  'store' | 'step' | 'cost_bps' | 'baseline' | 'use_ml' | 'cache' | 'log_trial'
>;
export type SignalTestPreview = G['SignalTestPreview'];

export type SweepParam = G['SweepParam'];
export type SweepMember = G['SweepMember'];
export type SweepPbo = G['SweepPbo'];
export type SweepManifest = G['SweepManifest'];
export type SweepRequest = Loose<
  G['SweepRequest'],
  'mode' | 'base_config' | 'store' | 'use_ml' | 'nav' | 'max_days' | 'max_runs'
>;
export type SweepCreate = SweepRequest & { expected_id: string };
export type SweepPlanView = G['SweepPlanView'];
export type SweepLaunchResponse = G['SweepLaunchResponse'];
export type SweepSummary = G['SweepSummary'];
export type SweepList = G['SweepList'];
export type SweepDetail = G['SweepDetail'];

export type CapacityPoint = G['CapacityPoint'];
export type CapacityResult = G['CapacityResult'];
export type CapacitySummary = G['CapacitySummary'];
export type CapacityList = G['CapacityList'];
export type CapacityRequest = Loose<
  G['CapacityRequest'],
  'store' | 'baseline' | 'max_days' | 'use_ml'
>;

export type VolStats = G['VolStats'];
export type NuResult = G['NuResult'];
export type RiskCalibrationResult = G['RiskCalibrationResult'];
export type RiskCalibrationSummary = G['RiskCalibrationSummary'];
export type RiskCalibrationList = G['RiskCalibrationList'];
export type RiskCalibrationRequest = Loose<
  G['RiskCalibrationRequest'],
  'store' | 'nav' | 'baseline' | 'max_days' | 'use_ml'
>;

export type MidcapCriterion = G['MidcapCriterion'];
export type MidcapVerdict = G['MidcapVerdict'];
export type MidcapCoverage = G['MidcapCoverage'];
export type MidcapCheckResult = G['MidcapCheckResult'];
export type MidcapResultSummary = G['MidcapResultSummary'];
export type MidcapStatus = G['MidcapStatus'];
export type MidcapResultDetail = G['MidcapResultDetail'];
export type MidcapPreregistration = G['MidcapPreregistration'];
export type MidcapRequest = Loose<
  G['MidcapRequest'],
  'store' | 'dry_run' | 'force' | 'confirm' | 'cache' | 'nav'
>;

export type PaperLevel = PaperCheck['level'];
export type PaperBookSummary = G['PaperBookSummary'];
export type PaperBooks = G['PaperBooks'];
export type PaperCheck = G['PaperCheck'];
export type PaperAck = G['PaperAck'];
export type PaperStatus = G['PaperStatus'];
export type PaperSeries = G['PaperSeries'];
export type PaperHistoryRow = G['PaperHistoryRow'];
export type PaperHistory = G['PaperHistory'];
export type PaperOrder = G['PaperOrder'];
export type PaperOrders = G['PaperOrders'];
export type PaperHolding = G['PaperHolding'];
export type PaperHoldings = G['PaperHoldings'];
export type PaperSchedule = G['PaperSchedule'];
export type PaperReconcile = G['PaperReconcile'];
export type PaperAckRequest = G['PaperAckRequest'];
export type PaperRunRequest = Loose<
  G['PaperRunRequest'],
  'date' | 'until' | 'no_sync' | 'dry_run' | 'redo_last' | 'signal_mode'
>;
