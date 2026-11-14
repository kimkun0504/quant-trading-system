import type { StoreSlice } from '@/shared/api';

/** The store slice as the form edits it: text inputs ('' = not set). */
export interface SliceDraft {
  start: string;
  end: string;
  maxNames: string;
  minMcapEver: string;
  maxRank: string;
}

/**
 * The reference load flags of every research command (handbook 12 "Research tools"):
 * `--start 2000-01-01 --max-rank 1500` (the screened-rank prefilter covers the universe by
 * construction; the peak-cap `--max-names 3000 --min-mcap-ever 3e8` of the runs before
 * 2026-11-14 misses names and is refused by the load-coverage guard). The stage cache is
 * keyed on the data, so these must match the sealed baseline run for the fits to be reused.
 */
export const REFERENCE_SLICE: SliceDraft = {
  start: '2000-01-01',
  end: '',
  maxNames: '',
  minMcapEver: '',
  maxRank: '1500',
};

/** The mid-cap check's default: the screened-rank prefilter `--max-rank 2750`. */
export const MIDCAP_SLICE: SliceDraft = {
  start: '2000-01-01',
  end: '',
  maxNames: '',
  minMcapEver: '',
  maxRank: '2750',
};

const ISO = /^\d{4}-\d{2}-\d{2}$/;

export function sliceErrors(d: SliceDraft): Partial<Record<keyof SliceDraft, string>> {
  const e: Partial<Record<keyof SliceDraft, string>> = {};
  if (d.start && !ISO.test(d.start)) e.start = 'Use a date as YYYY-MM-DD.';
  if (d.end && !ISO.test(d.end)) e.end = 'Use a date as YYYY-MM-DD.';
  if (!e.start && !e.end && d.start && d.end && d.start > d.end)
    e.end = 'The end is before the start.';
  const int = (k: 'maxNames' | 'maxRank', what: string) => {
    const v = d[k].trim();
    if (!v) return;
    const n = Number(v);
    if (!Number.isInteger(n) || n < 10 || n > 20_000)
      e[k] = `${what}: a whole number from 10 to 20,000.`;
  };
  int('maxNames', 'Max names');
  int('maxRank', 'Max rank');
  if (d.minMcapEver.trim()) {
    const n = Number(d.minMcapEver);
    if (!Number.isFinite(n) || n < 0)
      e.minMcapEver = 'A market cap in USD, 0 or more (3e8 = $300M).';
  }
  if (d.maxRank.trim() && (d.maxNames.trim() || d.minMcapEver.trim()))
    e.maxRank = 'The rank prefilter cannot be combined with max names / min market cap.';
  return e;
}

export function toSlice(d: SliceDraft): StoreSlice {
  const num = (s: string) => (s.trim() === '' ? null : Number(s));
  return {
    start: d.start || null,
    end: d.end || null,
    max_names: num(d.maxNames),
    min_mcap_ever: num(d.minMcapEver),
    max_rank: num(d.maxRank),
  };
}

/** Parse "5e6, 5e7, 2.5e8" into numbers; null when any item is not a number. */
export function parseNumberList(text: string): number[] | null {
  const items = text
    .split(/[,\s]+/)
    .map((x) => x.trim())
    .filter(Boolean);
  if (!items.length) return null;
  const out = items.map(Number);
  return out.every((x) => Number.isFinite(x)) ? out : null;
}
