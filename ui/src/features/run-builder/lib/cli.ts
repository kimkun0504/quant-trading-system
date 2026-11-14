import type { LaunchRunRequest } from '@/shared/api';

/** POSIX shell quoting, only where needed (like Python's shlex.quote). */
export function shellQuote(s: string): string {
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(s) ? s : `'${s.replace(/'/g, `'"'"'`)}'`;
}

function num(v: number): string {
  // Python's %g for the sizes that occur here: 2e+09 for a market cap, plain otherwise
  if (Number.isInteger(v) && Math.abs(v) >= 1e6) {
    const [m, e] = v.toExponential().split('e');
    const exp = Number(e);
    return `${m}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  }
  return String(v);
}

function yaml(v: number | boolean | string | null): string {
  return v === null ? 'null' : typeof v === 'boolean' ? (v ? 'true' : 'false') : String(v);
}

/**
 * The `qts` command line equivalent to a launch request: `qts demo` for a synthetic
 * market, `qts run --store` for the data store, `--config runs/<base>/config.yaml` when
 * the run starts from another run, then one `--set key=value` per changed key. Mirrors
 * `qts_api.services.jobs.cli_command` (which records the same text in launch.json).
 */
export function cliCommand(req: LaunchRunRequest, storePath: string | null): string[] {
  const parts: string[] = [];
  if (req.kind === 'synthetic') {
    parts.push('qts', 'demo', '--stocks', String(req.stocks), '--days', String(req.days));
    parts.push('--seed', String(req.seed));
  } else {
    parts.push('qts', 'run', '--store', storePath ?? '<store>');
    if (req.start) parts.push('--start', req.start);
    if (req.end) parts.push('--end', req.end);
    if (req.max_names !== null && req.max_names !== undefined)
      parts.push('--max-names', String(req.max_names));
    if (req.min_mcap_ever !== null && req.min_mcap_ever !== undefined)
      parts.push('--min-mcap-ever', num(req.min_mcap_ever));
    if (req.max_rank !== null && req.max_rank !== undefined)
      parts.push('--max-rank', String(req.max_rank));
  }
  if (req.base_run_id) parts.push('--config', `runs/${req.base_run_id}/config.yaml`);
  if (req.ml) parts.push('--ml');
  if (req.signal_lag_days !== 1) parts.push('--signal-lag', String(req.signal_lag_days));
  if (req.max_days !== null && req.max_days !== undefined)
    parts.push('--max-days', String(req.max_days));
  if (req.cache) parts.push('--cache');
  if (req.kind === 'store' && req.unseal_reason) {
    parts.push('--unseal', req.unseal_reason);
    if (req.force) parts.push('--force');
  }
  if (req.kind === 'store' && req.allow_partial_coverage) parts.push('--allow-partial-coverage');
  const overrides: [string, number | boolean | string | null][] = Object.entries(
    req.config_overrides,
  );
  // `qts run` has no --seed flag: there the seed is a config key
  if (req.kind === 'store' && (req.base_run_id || req.seed !== 0))
    overrides.unshift(['seed', req.seed]);
  for (const [k, v] of overrides) parts.push('--set', `${k}=${yaml(v)}`);
  return parts.map(shellQuote);
}

/** The command as one line, and wrapped one flag per line for display. */
export function cliText(parts: readonly string[]): string {
  return parts.join(' ');
}

export function cliLines(parts: readonly string[]): string[] {
  const lines: string[] = [];
  let cur = '';
  for (const p of parts) {
    if (p.startsWith('--') && cur) {
      lines.push(cur);
      cur = p;
    } else cur = cur ? `${cur} ${p}` : p;
  }
  if (cur) lines.push(cur);
  return lines;
}
