import type {
  CapacityRequest,
  MidcapRequest,
  PaperRunRequest,
  RiskCalibrationRequest,
  SignalTestRequest,
  StoreSlice,
  SweepRequest,
} from '@/shared/api';
import { shellQuote } from '@/features/run-builder/lib/cli';

/**
 * The `qts` command lines of the research launch forms, before launch. They mirror
 * `qts_api.services.tasks.task_command` (which records the same text on the job), so
 * "Copy as CLI command" reproduces exactly what the job does; `cli.test.ts` checks the
 * strings the API tests check.
 */

export const DEFAULT_BASELINE = 'configs/baseline.yaml';
export const MIDCAP_DEFAULT_MAX_RANK = 2750;

/** Python's `%g` (6 significant digits; exponent form below 1e-4 and from 1e6): 5e+06, 2.5e+08, 0.05. */
export function g(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return '0';
  const [mant = '0', ex = '0'] = v.toExponential(5).split('e');
  const exp = Number(ex);
  if (exp < -4 || exp >= 6)
    return `${String(Number(mant))}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  return String(Number(v.toPrecision(6)));
}

/** A sweep value as the API prints it (a whole number stays whole: 1, not 1.0). */
function value(v: boolean | number | string): string {
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  return String(v);
}

export function storeFlags(store: StoreSlice | undefined, storePath: string | null): string[] {
  const s = store ?? {};
  const parts = ['--store', storePath ?? '<store>'];
  if (s.start) parts.push('--start', s.start);
  if (s.end) parts.push('--end', s.end);
  if (s.max_names !== null && s.max_names !== undefined)
    parts.push('--max-names', String(s.max_names));
  if (s.min_mcap_ever !== null && s.min_mcap_ever !== undefined)
    parts.push('--min-mcap-ever', g(s.min_mcap_ever));
  if (s.max_rank !== null && s.max_rank !== undefined) parts.push('--max-rank', String(s.max_rank));
  return parts;
}

const join = (parts: readonly string[]) => parts.map(shellQuote).join(' ');

export function signalTestCommand(r: SignalTestRequest, storePath: string | null): string {
  const p = ['qts', 'signal', 'test', r.feature, ...storeFlags(r.store, storePath)];
  p.push('--horizons', [...r.horizons].sort((a, b) => a - b).join(','));
  if ((r.step ?? 5) !== 5) p.push('--step', String(r.step));
  if (r.cost_bps !== null && r.cost_bps !== undefined) p.push('--cost-bps', g(r.cost_bps));
  if ((r.baseline ?? DEFAULT_BASELINE) !== DEFAULT_BASELINE) p.push('--baseline', r.baseline ?? '');
  if (r.use_ml === false) p.push('--no-ml');
  if (r.cache === false) p.push('--no-cache');
  if (r.log_trial === false) p.push('--no-log');
  return join(p);
}

export function sweepCommand(
  r: SweepRequest,
  storePath: string | null,
  sweepId?: string | null,
): string {
  const p = ['qts', 'sweep', '--config', r.base_config ?? DEFAULT_BASELINE];
  for (const prm of r.params) p.push('--param', `${prm.key}=${prm.values.map(value).join(',')}`);
  if (r.mode === 'zip') p.push('--zip');
  p.push(...storeFlags(r.store, storePath));
  if ((r.nav ?? 5e6) !== 5e6) p.push('--nav', g(r.nav ?? 5e6));
  if (r.max_days !== null && r.max_days !== undefined) p.push('--max-days', String(r.max_days));
  if (r.use_ml === false) p.push('--no-ml');
  if (sweepId) p.push('--id', sweepId);
  return join(p);
}

export function capacityCommand(r: CapacityRequest, storePath: string | null): string {
  const p = ['qts', 'capacity', ...storeFlags(r.store, storePath)];
  p.push(
    '--navs',
    [...r.navs]
      .sort((a, b) => a - b)
      .map(g)
      .join(','),
  );
  if ((r.baseline ?? DEFAULT_BASELINE) !== DEFAULT_BASELINE) p.push('--config', r.baseline ?? '');
  if (r.max_days !== null && r.max_days !== undefined) p.push('--max-days', String(r.max_days));
  if (r.use_ml === false) p.push('--no-ml');
  return join(p);
}

export function riskCommand(r: RiskCalibrationRequest, storePath: string | null): string {
  const p = ['qts', 'risk', 'calibrate', ...storeFlags(r.store, storePath)];
  p.push(
    '--nu',
    [...r.nu]
      .sort((a, b) => a - b)
      .map(g)
      .join(','),
  );
  if ((r.nav ?? 5e6) !== 5e6) p.push('--nav', g(r.nav ?? 5e6));
  if ((r.baseline ?? DEFAULT_BASELINE) !== DEFAULT_BASELINE) p.push('--config', r.baseline ?? '');
  if (r.max_days !== null && r.max_days !== undefined) p.push('--max-days', String(r.max_days));
  if (r.use_ml === false) p.push('--no-ml');
  return join(p);
}

/** The mid-cap check: without a prefilter the check's default is `--max-rank 2750`. */
export function midcapCommand(r: MidcapRequest, storePath: string | null): string {
  const s = { ...(r.store ?? {}) };
  if (s.max_names == null && s.min_mcap_ever == null && s.max_rank == null)
    s.max_rank = MIDCAP_DEFAULT_MAX_RANK;
  const p = ['qts', 'check', 'midcap', ...storeFlags(s, storePath)];
  if ((r.nav ?? 5e6) !== 5e6) p.push('--nav', g(r.nav ?? 5e6));
  if (r.cache ?? true) p.push('--cache');
  if (r.dry_run) p.push('--dry-run');
  if (r.force && !r.dry_run) p.push('--force');
  return join(p);
}

export function paperRunCommand(
  book: string,
  r: PaperRunRequest,
  paperRootFlag: string | null,
): string {
  const p = ['qts', 'paper', 'run', '--name', book];
  if (paperRootFlag) p.push('--paper-root', paperRootFlag);
  if (r.date) p.push('--date', r.date);
  if (r.until) p.push('--until', r.until);
  if (r.no_sync) p.push('--no-sync');
  if (r.dry_run) p.push('--dry-run');
  if (r.redo_last) p.push('--redo-last');
  if ((r.signal_mode ?? 'daily') !== 'daily') p.push('--signal-mode', r.signal_mode ?? 'daily');
  return join(p);
}

export function paperReconcileCommand(book: string, paperRootFlag: string | null): string {
  return join([
    'qts',
    'paper',
    'reconcile',
    '--name',
    book,
    ...(paperRootFlag ? ['--paper-root', paperRootFlag] : []),
  ]);
}

export function paperAckCommand(book: string, note: string, paperRootFlag: string | null): string {
  return join([
    'qts', 'paper', 'ack', '--name', book,
    ...(paperRootFlag ? ['--paper-root', paperRootFlag] : []),
    '--note', note || '<why it is safe to continue>',
  ]); // prettier-ignore
}

/** `qts run --store ... --unseal REASON [--force]` lines are the run builder's (lib/cli.ts). */
