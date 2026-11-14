import type {
  ConfigLeaf,
  ConfigLeaves,
  LaunchRunRequest,
  RunDetail,
  RunKind,
  RunSummary,
} from '@/shared/api';
import type { Field } from './fields';
import { listText } from './fields';

/** A field's edited input: text for numbers and strings, a boolean, or null ("no value"). */
export type Raw = string | boolean | null;

/** Launch settings that are not SystemConfig fields: the model switch, seed, lag and data. */
export interface RunSettings {
  label: string;
  ml: boolean;
  seed: string;
  lag: '0' | '1';
  kind: RunKind;
  /** Synthetic market. */
  stocks: string;
  days: string;
  /** Store slice (YYYY-MM-DD, '' = the whole store) and prefilter ('' = off). */
  start: string;
  end: string;
  maxNames: string;
  minMcapEver: string;
  /** Screened market-cap rank prefilter (`--max-rank`; '' = off). Not with the two above. */
  maxRank: string;
  /** Load although the prefilter misses universe names (`--allow-partial-coverage`, flagged). */
  allowPartialCoverage: boolean;
  /** Backtest only the last N out-of-sample days ('' = all). */
  maxDays: string;
  cache: boolean;
  /** Store runs: load the sealed holdout (`--unseal REASON`; '' = sealed). */
  unsealReason: string;
  /** With an unseal: a configuration whose holdout was already used (`--force`, flagged). */
  force: boolean;
}

export type RunSettingKey = keyof RunSettings;

/** What the edit starts from: the defaults, or one finished run. */
export interface Base {
  runId: string | null;
  label: string;
  configHash: string | null;
  dataHash: string | null;
  /** The trial family the base run was counted in, when its manifest says. */
  ledgerFamily: string | null;
  /** Config values of the base by dotted path (schema defaults fill what it lacks). */
  leaves: ConfigLeaves;
  run: RunSettings;
  /** False when the run did not record where its data came from. */
  sourceRecorded: boolean;
  /** The store path the base run read, when it says. */
  storePath: string | null;
}

/** One variant being edited: only what differs from the base is stored. */
export interface Draft {
  id: string;
  /** Edited config inputs by path; a path that is absent keeps the base value. */
  values: Record<string, Raw>;
  /** Edited launch settings; absent keys keep the base's. */
  run: Partial<RunSettings>;
}

export const DEFAULT_RUN: RunSettings = {
  label: '',
  ml: false,
  seed: '0',
  lag: '1',
  kind: 'synthetic',
  stocks: '300',
  days: '1512',
  start: '',
  end: '',
  maxNames: '',
  minMcapEver: '',
  // the store prefilter of every store load (`qts run --max-rank 1500`): covers the
  // baseline's 1,000-name universe by construction (handbook 12; a synthetic run ignores it)
  maxRank: '1500',
  allowPartialCoverage: false,
  maxDays: '',
  cache: false,
  unsealReason: '',
  force: false,
};

export function defaultBase(): Base {
  return {
    runId: null,
    label: 'SystemConfig defaults',
    configHash: null,
    dataHash: null,
    ledgerFamily: null,
    leaves: {},
    run: DEFAULT_RUN,
    sourceRecorded: true,
    storePath: null,
  };
}

const text = (v: number | string | null | undefined): string =>
  v === null || v === undefined ? '' : String(v);

/** The base for "start from this run": its config leaves and its recorded data source. */
export function baseFromRun(detail: RunDetail, summary: RunSummary | undefined): Base {
  const c = detail.config;
  const src = c.source;
  return {
    runId: detail.id,
    label: detail.label,
    configHash: detail.manifest.config_hash,
    dataHash: detail.manifest.data_hash || null,
    ledgerFamily: detail.manifest.ledger_family,
    leaves: summary?.config_leaves ?? {},
    run: {
      label: '',
      ml: c.use_ml,
      seed: String(c.seed),
      lag: c.signal_lag_days === 0 ? '0' : '1',
      kind: src.kind,
      stocks: text(src.stocks ?? c.stocks) || DEFAULT_RUN.stocks,
      days: text(src.days ?? c.days) || DEFAULT_RUN.days,
      start: text(src.start),
      end: text(src.end),
      maxNames: text(src.max_names),
      minMcapEver: text(src.min_mcap_ever),
      // a store run keeps its own prefilter (none recorded: no prefilter); a synthetic
      // base switched to the store starts from the default rank prefilter
      maxRank: src.kind === 'store' ? text(src.max_rank) : DEFAULT_RUN.maxRank,
      // a partial load is a decision for this launch, never inherited from the base run
      allowPartialCoverage: false,
      maxDays: text(src.max_days),
      cache: src.cache,
      // an unseal is a decision for this launch, never inherited from the base run
      unsealReason: '',
      force: false,
    },
    sourceRecorded: src.recorded,
    storePath: src.store_path,
  };
}

// ----------------------------------------------------------------------------- values

/** The base's value of a field: the run's own, else the schema default, else null. */
export function baseValue(base: Base, f: Field): ConfigLeaf {
  const own = base.leaves[f.path];
  // a list arrives as compact JSON in config_leaves ('["mom_change"]'): compare it as names
  if (f.type === 'array')
    return listText(typeof own === 'string' ? own : typeof f.default === 'string' ? f.default : '');
  if (own !== undefined) return own;
  return f.default ?? null;
}

export function toRaw(v: ConfigLeaf): Raw {
  return typeof v === 'number' ? String(v) : v;
}

/** What the input shows: the edit when there is one, else the base value. */
export function rawValue(draft: Draft, base: Base, f: Field): Raw {
  const edited = draft.values[f.path];
  return edited !== undefined ? edited : toRaw(baseValue(base, f));
}

export type Parsed = { ok: true; value: ConfigLeaf } | { ok: false; error: string };

/** Type and bounds check of one input, in the words the API uses for the same failures. */
export function parseRaw(f: Field, raw: Raw): Parsed {
  if (raw === null)
    return f.nullable
      ? { ok: true, value: null }
      : { ok: false, error: `${f.title} needs a value.` };
  if (f.type === 'boolean')
    return typeof raw === 'boolean'
      ? { ok: true, value: raw }
      : { ok: false, error: `${f.title} must be on or off.` };
  if (typeof raw === 'boolean') return { ok: false, error: `${f.title} must be a value.` };
  if (f.type === 'array') {
    const names = listText(raw);
    if (f.choices) {
      const known = new Set(Object.values(f.choices).flat());
      const unknown = names ? names.split(',').filter((n) => !known.has(n)) : [];
      if (unknown.length)
        return { ok: false, error: `${f.title}: unknown name ${unknown.join(', ')}.` };
    }
    return { ok: true, value: names };
  }
  if (f.type === 'string') {
    if (f.enum && !f.enum.includes(raw))
      return { ok: false, error: `${f.title} must be one of ${f.enum.join(', ')}.` };
    return { ok: true, value: raw };
  }
  const s = raw.trim();
  if (s === '')
    return {
      ok: false,
      error: f.nullable
        ? `Enter a number, or choose “No value”.`
        : `Enter a number${f.default !== undefined ? ` (default ${String(f.default)})` : ''}.`,
    };
  const n = Number(s);
  if (!Number.isFinite(n)) return { ok: false, error: `${f.title} must be a number.` };
  if (f.type === 'integer' && !Number.isInteger(n))
    return { ok: false, error: `${f.title} must be a whole number.` };
  if (f.minimum !== undefined && n < f.minimum)
    return { ok: false, error: `${f.title} must be at least ${f.minimum}.` };
  if (f.maximum !== undefined && n > f.maximum)
    return { ok: false, error: `${f.title} must be at most ${f.maximum}.` };
  if (f.exclusiveMinimum !== undefined && n <= f.exclusiveMinimum)
    return { ok: false, error: `${f.title} must be greater than ${f.exclusiveMinimum}.` };
  if (f.exclusiveMaximum !== undefined && n >= f.exclusiveMaximum)
    return { ok: false, error: `${f.title} must be less than ${f.exclusiveMaximum}.` };
  return { ok: true, value: n };
}

export function sameLeaf(x: ConfigLeaf | undefined, y: ConfigLeaf | undefined): boolean {
  if (x === y) return true;
  if (typeof x === 'number' && typeof y === 'number') return Math.abs(x - y) <= 1e-12;
  return false;
}

/** True when the field's edit differs from the base (an unparseable edit counts). */
export function isChanged(draft: Draft, base: Base, f: Field): boolean {
  const edited = draft.values[f.path];
  if (edited === undefined) return false;
  const p = parseRaw(f, edited);
  return !p.ok || !sameLeaf(p.value, baseValue(base, f));
}

/** The override map of a draft: parsed values of the fields that differ from the base. */
export function draftOverrides(
  draft: Draft,
  base: Base,
  fields: readonly Field[],
): Record<string, ConfigLeaf> {
  const out: Record<string, ConfigLeaf> = {};
  for (const f of fields) {
    const edited = draft.values[f.path];
    if (edited === undefined) continue;
    const p = parseRaw(f, edited);
    if (p.ok && !sameLeaf(p.value, baseValue(base, f))) out[f.path] = p.value;
  }
  return out;
}

/** Client-side errors of a draft's config edits, by path. */
export function fieldErrors(draft: Draft, fields: readonly Field[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const f of fields) {
    const edited = draft.values[f.path];
    if (edited === undefined) continue;
    const p = parseRaw(f, edited);
    if (!p.ok) out[f.path] = p.error;
  }
  return out;
}

// ----------------------------------------------------------------------------- run settings

export function runSettings(draft: Draft, base: Base): RunSettings {
  return { ...base.run, ...draft.run };
}

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;

function intError(raw: string, min: number, max: number, what: string): string | undefined {
  const n = Number(raw);
  if (raw.trim() === '' || !Number.isInteger(n)) return `${what} must be a whole number.`;
  if (n < min || n > max)
    return `${what} must be between ${min.toLocaleString('en-US')} and ${max.toLocaleString('en-US')}.`;
  return undefined;
}

/** Client-side errors of the launch settings (`minDays`: the warm-up floor, when known). */
export function runErrors(
  r: RunSettings,
  minDays?: number | null,
): Partial<Record<RunSettingKey, string>> {
  const e: Partial<Record<RunSettingKey, string>> = {};
  const set = (k: RunSettingKey, msg: string | undefined) => {
    if (msg) e[k] = msg;
  };
  set('seed', intError(r.seed, 0, 2 ** 31 - 1, 'Seed'));
  if (r.label.length > 120) e.label = 'Use at most 120 characters.';
  if (r.maxDays.trim() !== '') set('maxDays', intError(r.maxDays, 20, 100_000, 'Backtest days'));
  if (r.kind === 'synthetic') {
    set('stocks', intError(r.stocks, 50, 3000, 'Stocks'));
    const days = intError(r.days, 300, 6000, 'Days');
    if (days) e.days = days;
    else if (minDays !== undefined && minDays !== null && Number(r.days) < minDays)
      e.days = `Use at least ${minDays.toLocaleString('en-US')} days: the warm-up of this configuration plus one year.`;
  } else {
    if (r.start && !ISO_DATE.test(r.start)) e.start = 'Use a date as YYYY-MM-DD.';
    if (r.end && !ISO_DATE.test(r.end)) e.end = 'Use a date as YYYY-MM-DD.';
    if (!e.start && !e.end && r.start && r.end && r.start > r.end)
      e.end = 'The end date is before the start date.';
    if (r.maxNames.trim() !== '') set('maxNames', intError(r.maxNames, 10, 20_000, 'Max names'));
    if (r.minMcapEver.trim() !== '') {
      const n = Number(r.minMcapEver);
      if (!Number.isFinite(n) || n < 0) e.minMcapEver = 'Enter a market cap in USD, 0 or more.';
    }
    if (r.maxRank.trim() !== '') {
      set('maxRank', intError(r.maxRank, 10, 20_000, 'Max rank'));
      if (!e.maxRank && (r.maxNames.trim() !== '' || r.minMcapEver.trim() !== ''))
        e.maxRank = 'The rank prefilter cannot be combined with max names / min peak market cap.';
    }
  }
  return e;
}

// ----------------------------------------------------------------------------- diff

export interface DiffRow {
  /** Config path, or the launch setting's name (`ml`, `seed`, `data.start`). */
  key: string;
  group: 'config' | 'run' | 'data';
  from: ConfigLeaf;
  to: ConfigLeaf;
  /** Set when the edit does not parse; `to` is then the raw text. */
  error?: string;
}

const DATA_KEYS: readonly [RunSettingKey, string][] = [
  ['kind', 'data.kind'],
  ['stocks', 'data.stocks'],
  ['days', 'data.days'],
  ['start', 'data.start'],
  ['end', 'data.end'],
  ['maxNames', 'data.max_names'],
  ['minMcapEver', 'data.min_mcap_ever'],
  ['maxRank', 'data.max_rank'],
  ['allowPartialCoverage', 'data.allow_partial_coverage'],
];
const RUN_KEYS: readonly [RunSettingKey, string][] = [
  ['ml', 'ml'],
  ['lag', 'signal_lag_days'],
  ['seed', 'seed'],
  ['maxDays', 'max_days'],
];

const blank = (v: string | boolean): ConfigLeaf => (v === '' ? null : v);

/**
 * What differs from the base, config keys first (sorted), then the model / execution
 * switches, then the data source. The stage cache and the label change no result and
 * are left out.
 */
export function diffRows(draft: Draft, base: Base, fields: readonly Field[]): DiffRow[] {
  const rows: DiffRow[] = [];
  for (const f of fields) {
    const edited = draft.values[f.path];
    if (edited === undefined) continue;
    const from = baseValue(base, f);
    const p = parseRaw(f, edited);
    if (!p.ok) rows.push({ key: f.path, group: 'config', from, to: edited, error: p.error });
    else if (!sameLeaf(p.value, from))
      rows.push({ key: f.path, group: 'config', from, to: p.value });
  }
  rows.sort((a, b) => a.key.localeCompare(b.key));
  const now = runSettings(draft, base);
  const relevant = (k: RunSettingKey) =>
    now.kind === 'synthetic'
      ? !['start', 'end', 'maxNames', 'minMcapEver', 'maxRank', 'allowPartialCoverage'].includes(k)
      : !['stocks', 'days'].includes(k);
  for (const [k, key] of RUN_KEYS)
    if (now[k] !== base.run[k])
      rows.push({ key, group: 'run', from: blank(base.run[k]), to: blank(now[k]) });
  for (const [k, key] of DATA_KEYS)
    if (now[k] !== base.run[k] && (k === 'kind' || relevant(k)))
      rows.push({ key, group: 'data', from: blank(base.run[k]), to: blank(now[k]) });
  return rows;
}

/** Keys that change what the strategy does: config, the ML switch and the signal lag. */
export function strategyDiffCount(rows: readonly DiffRow[]): number {
  return rows.filter((r) => r.group === 'config' || r.key === 'ml' || r.key === 'signal_lag_days')
    .length;
}

/** More than this many strategy keys: a new strategy, not a variant of the base. */
export const VARIANT_MAX_KEYS = 3;

export function dataChanged(rows: readonly DiffRow[]): boolean {
  return rows.some((r) => r.group === 'data');
}

/** `1.0 → 0.0` style text of one side of a diff row (like the compare view's config diff). */
export function formatDiffValue(v: ConfigLeaf, other: ConfigLeaf): string {
  if (v === null) return 'none';
  if (typeof v === 'number') {
    const decimal = !Number.isInteger(v) || (typeof other === 'number' && !Number.isInteger(other));
    return decimal && Number.isInteger(v) ? v.toFixed(1) : String(v);
  }
  if (typeof v === 'boolean') return v ? 'on' : 'off';
  return v;
}

/** A label for the variant: the base's label plus what changed (at most two keys). */
export function autoLabel(base: Base, rows: readonly DiffRow[]): string {
  const stem = base.runId
    ? base.label
    : rows.some((r) => r.key === 'ml' && r.to === true)
      ? 'Composite + ML'
      : 'Composite';
  const parts = rows
    .filter((r) => !r.error && r.group !== 'data')
    .map(
      (r) =>
        `${r.key.split('.').pop()?.replace(/_/g, ' ') ?? r.key} ${formatDiffValue(r.to, r.from)}`,
    );
  if (parts.length === 0) return base.runId ? `${stem}, re-run` : stem;
  const text = parts.length <= 2 ? parts.join(', ') : `${parts.length} changes`;
  return `${stem}, ${text}`.slice(0, 120);
}

// ----------------------------------------------------------------------------- request

const optInt = (s: string): number | null => (s.trim() === '' ? null : Number(s));

/** The launch (and validate) body of a draft. Phase 2 is the ML sleeve's phase. */
export function toRequest(draft: Draft, base: Base, fields: readonly Field[]): LaunchRunRequest {
  const r = runSettings(draft, base);
  const rows = diffRows(draft, base, fields);
  const synthetic = r.kind === 'synthetic';
  const req: LaunchRunRequest = {
    kind: r.kind,
    ml: r.ml,
    phase: r.ml ? 2 : 1,
    stocks: synthetic ? Number(r.stocks) : Number(DEFAULT_RUN.stocks),
    days: synthetic ? Number(r.days) : Number(DEFAULT_RUN.days),
    seed: Number(r.seed),
    signal_lag_days: r.lag === '0' ? 0 : 1,
    config_overrides: draftOverrides(draft, base, fields),
    label: r.label.trim() || autoLabel(base, rows),
    base_run_id: base.runId,
    max_days: optInt(r.maxDays),
    cache: r.cache,
  };
  if (!synthetic) {
    req.start = r.start || null;
    req.end = r.end || null;
    req.max_names = optInt(r.maxNames);
    req.min_mcap_ever = r.minMcapEver.trim() === '' ? null : Number(r.minMcapEver);
    req.max_rank = optInt(r.maxRank);
    if (r.allowPartialCoverage) req.allow_partial_coverage = true;
    if (r.unsealReason.trim()) {
      req.unseal_reason = r.unsealReason.trim();
      if (r.force) req.force = true;
    }
  }
  return req;
}
