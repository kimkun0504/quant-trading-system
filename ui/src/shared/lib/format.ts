/**
 * Number and date formatting. Negative numbers use the true minus sign (U+2212) so
 * columns of tabular figures line up; signed formats always show "+" for gains.
 */

export const MINUS = '−';
export const MISSING = 'n/a';

export function isFiniteNumber(x: unknown): x is number {
  return typeof x === 'number' && Number.isFinite(x);
}

interface NumOpts {
  digits?: number;
  signed?: boolean;
}

function fixed(x: number, digits: number): string {
  // Avoid "-0.00": round first, then format the magnitude.
  const factor = 10 ** digits;
  const r = Math.round(x * factor) / factor;
  const body = Math.abs(r).toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  if (r < 0) return `${MINUS}${body}`;
  return body;
}

function withSign(s: string, signed: boolean): string {
  if (!signed) return s;
  if (s.startsWith(MINUS)) return s;
  return Number(s.replace(/,/g, '')) === 0 ? s : `+${s}`;
}

export function num(x: number | null | undefined, { digits = 2, signed = false }: NumOpts = {}) {
  if (!isFiniteNumber(x)) return MISSING;
  return withSign(fixed(x, digits), signed);
}

/** A fraction rendered as a percentage: 0.0476 → "4.76%". */
export function pct(x: number | null | undefined, { digits = 1, signed = false }: NumOpts = {}) {
  if (!isFiniteNumber(x)) return MISSING;
  return `${withSign(fixed(x * 100, digits), signed)}%`;
}

/** Basis points: 16.337 → "16.3 bp". */
export function bps(x: number | null | undefined, { digits = 1, signed = false }: NumOpts = {}) {
  if (!isFiniteNumber(x)) return MISSING;
  return `${withSign(fixed(x, digits), signed)} bp`;
}

/** Sharpe-like ratios: two decimals. */
export function sharpe(
  x: number | null | undefined,
  { signed = false }: { signed?: boolean } = {},
) {
  return num(x, { digits: 2, signed });
}

/** Multiplicative ratio: 1.04 → "1.04×". */
export function times(x: number | null | undefined, digits = 2) {
  if (!isFiniteNumber(x)) return MISSING;
  return `${fixed(x, digits)}×`;
}

export function days(x: number | null | undefined) {
  if (!isFiniteNumber(x)) return MISSING;
  return `${fixed(x, 0)} d`;
}

export function int(x: number | null | undefined) {
  if (!isFiniteNumber(x)) return MISSING;
  return fixed(x, 0);
}

/** USD with compact suffix: 5_212_345 → "$5.21M". */
export function currency(
  x: number | null | undefined,
  { compact = true }: { compact?: boolean } = {},
) {
  if (!isFiniteNumber(x)) return MISSING;
  const sign = x < 0 ? MINUS : '';
  const a = Math.abs(x);
  if (!compact || a < 10_000) return `${sign}$${fixed(a, a < 100 ? 2 : 0)}`;
  if (a < 1_000_000) return `${sign}$${fixed(a / 1_000, 1)}k`;
  if (a < 1_000_000_000) return `${sign}$${fixed(a / 1_000_000, 2)}M`;
  return `${sign}$${fixed(a / 1_000_000_000, 2)}B`;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export const MONTH_LABELS: readonly string[] = MONTHS;

/** "2023-10-17" → "17 Oct 2023". Accepts full ISO timestamps too (date part is used). */
export function date(iso: string | null | undefined): string {
  if (!iso) return MISSING;
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!m) return iso;
  const [, y, mo, d] = m;
  return `${Number(d)} ${MONTHS[Number(mo) - 1] ?? mo} ${y}`;
}

/** "2026-11-05T09:14:00Z" → "5 Nov 2026, 09:14 UTC". */
export function dateTime(iso: string | null | undefined): string {
  if (!iso) return MISSING;
  const t = /T(\d{2}):(\d{2})/.exec(iso);
  return t ? `${date(iso)}, ${t[1]}:${t[2]} UTC` : date(iso);
}

/** Short hash for display: first 8 characters. */
export function shortHash(h: string | null | undefined, n = 8): string {
  if (!h) return MISSING;
  return h.slice(0, n);
}

/** Sign of a value as a word, for text alternatives: "positive" | "negative" | "zero". */
export function signWord(x: number): 'positive' | 'negative' | 'zero' {
  if (x > 0) return 'positive';
  if (x < 0) return 'negative';
  return 'zero';
}
