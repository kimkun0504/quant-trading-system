import type { ConfigLeaf, ConfigLeaves, LedgerFamily, RunSummary } from '@/shared/api';

/**
 * Experiment families: runs of the same strategy that differ in a knob or two.
 *
 * Two runs are *close* when their flattened configs differ in at most
 * `MAX_DIFF_KEYS` leaves (a key missing on one side counts as a difference). Runs are
 * clustered by single linkage over that relation inside their ledger family (the
 * pipeline's `pipeline_phase{p}:{data_hash[:12]}`, i.e. one phase on one data set),
 * so a chain baseline → trade_aversion 0 → horizon 10 is one family even when the
 * ends differ in two keys. Runs still running or failed (no config) each form their own
 * single-run family.
 */
export const MAX_DIFF_KEYS = 3;

export interface ConfigDiffRow {
  key: string;
  /** One value per run, in the order the runs were given. */
  values: (ConfigLeaf | undefined)[];
}

export interface RunFamily {
  /** Stable key: ledger family + cluster index. */
  key: string;
  /** Common label prefix of the members, else the shortest member label (the base run). */
  label: string;
  /** The pipeline's ledger family (phase + data hash) the members share. */
  ledgerFamily: string;
  /** Members newest first. */
  runs: RunSummary[];
  /** Keys that differ between members (empty for a family of one). */
  diffKeys: string[];
}

type FamilyKeyed = Pick<RunSummary, 'phase' | 'data_hash'> &
  Partial<Pick<RunSummary, 'ledger_family'>>;

/**
 * The trial family the pipeline logs under: the one the run's manifest records (keyed by the
 * market hash today), else `pipeline_phase{p}:{data_hash[:12]}` for runs written before the
 * manifest carried it; records without a data hash are `legacy`.
 */
export function ledgerFamilyOf(run: FamilyKeyed): string {
  if (run.ledger_family) return run.ledger_family;
  return run.data_hash ? `pipeline_phase${run.phase}:${run.data_hash.slice(0, 12)}` : 'legacy';
}

/** Keys whose values differ between two flattened configs (missing counts as different). */
export function diffKeys(a: ConfigLeaves, b: ConfigLeaves): string[] {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  const out: string[] = [];
  for (const k of keys) if (!sameLeaf(a[k], b[k])) out.push(k);
  return out.sort();
}

function sameLeaf(x: ConfigLeaf | undefined, y: ConfigLeaf | undefined): boolean {
  if (x === y) return true;
  if (typeof x === 'number' && typeof y === 'number') return Math.abs(x - y) <= 1e-12;
  return false;
}

/** Rows for every key on which at least two of the configs disagree, sorted by key. */
export function configDiff(configs: readonly ConfigLeaves[]): ConfigDiffRow[] {
  const keys = new Set<string>();
  for (const c of configs) for (const k of Object.keys(c)) keys.add(k);
  const rows: ConfigDiffRow[] = [];
  for (const key of [...keys].sort()) {
    const values = configs.map((c) => c[key]);
    const first = values[0];
    if (values.some((v) => !sameLeaf(v, first))) rows.push({ key, values });
  }
  return rows;
}

/** `portfolio.trade_aversion: 1.0 | 0.0`, the compact one-line form of a diff row. */
export function formatDiffRow(row: ConfigDiffRow): string {
  const decimal = row.values.some((v) => typeof v === 'number' && !Number.isInteger(v));
  return `${row.key}: ${row.values.map((v) => formatLeaf(v, decimal)).join(' | ')}`;
}

/**
 * A config value as text. `decimal` prints whole numbers with one decimal so a row
 * that mixes 1 and 0.5 reads `1.0 | 0.5`; a missing key is an em dash.
 */
export function formatLeaf(v: ConfigLeaf | undefined, decimal = false): string {
  if (v === undefined) return '—';
  if (v === null) return 'null';
  if (typeof v === 'number') return decimal && Number.isInteger(v) ? v.toFixed(1) : String(v);
  return String(v);
}

/** Longest common prefix of the labels on word boundaries, trimmed of separators. */
export function commonLabel(labels: readonly string[]): string {
  if (labels.length === 0) return '';
  const words = labels.map((l) => l.split(/(?=[\s,;:–—-])/));
  const first = words[0] ?? [];
  let n = 0;
  while (n < first.length && words.every((w) => w[n] === first[n])) n++;
  return first
    .slice(0, n)
    .join('')
    .replace(/[\s,;:–—-]+$/u, '')
    .trim();
}

function union(parent: number[], a: number): number {
  let r = a;
  while (parent[r] !== r) r = parent[r] ?? r;
  let x = a;
  while (parent[x] !== r) {
    const next = parent[x] ?? r;
    parent[x] = r;
    x = next;
  }
  return r;
}

/** Group runs into families (see module doc). Families are ordered by their newest run. */
export function clusterFamilies(runs: readonly RunSummary[]): RunFamily[] {
  const byLedger = new Map<string, RunSummary[]>();
  for (const r of runs) {
    const key = ledgerFamilyOf(r);
    byLedger.set(key, [...(byLedger.get(key) ?? []), r]);
  }
  const families: RunFamily[] = [];
  for (const [ledgerFamily, members] of byLedger) {
    const configured = members.filter((r) => Object.keys(r.config_leaves).length > 0);
    const bare = members.filter((r) => Object.keys(r.config_leaves).length === 0);
    const parent = configured.map((_, i) => i);
    for (let i = 0; i < configured.length; i++) {
      for (let j = i + 1; j < configured.length; j++) {
        const a = configured[i];
        const b = configured[j];
        if (!a || !b) continue;
        if (diffKeys(a.config_leaves, b.config_leaves).length <= MAX_DIFF_KEYS) {
          const ra = union(parent, i);
          const rb = union(parent, j);
          if (ra !== rb) parent[rb] = ra;
        }
      }
    }
    const clusters = new Map<number, RunSummary[]>();
    configured.forEach((r, i) => {
      const root = union(parent, i);
      clusters.set(root, [...(clusters.get(root) ?? []), r]);
    });
    let idx = 0;
    const groups = [...clusters.values(), ...bare.map((r) => [r])];
    for (const group of groups) {
      const sorted = [...group].sort((a, b) => b.created_at.localeCompare(a.created_at));
      const label =
        commonLabel(sorted.map((r) => r.label)) ||
        [...sorted].sort((a, b) => a.label.length - b.label.length)[0]?.label ||
        '';
      families.push({
        key: `${ledgerFamily}#${idx++}`,
        label,
        ledgerFamily,
        runs: sorted,
        diffKeys: configDiff(sorted.map((r) => r.config_leaves)).map((d) => d.key),
      });
    }
  }
  return families.sort((a, b) =>
    (b.runs[0]?.created_at ?? '').localeCompare(a.runs[0]?.created_at ?? ''),
  );
}

/** The ledger's trial count for a run's family, when the ledger knows it. */
export function trialsFor(
  run: FamilyKeyed,
  families: readonly LedgerFamily[] | undefined,
): LedgerFamily | undefined {
  const key = ledgerFamilyOf(run);
  return families?.find((f) => f.family === key);
}
