/**
 * Small pure helpers for shaping time series before charting: rolling statistics,
 * calendar grouping, histogram bins and even thinning. Nulls are gaps: a rolling window
 * needs `minObs` finite values or it gives null.
 */

export type Maybe = number | null | undefined;

export function finite(v: Maybe): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** Trailing mean over `window` points (null until `minObs` finite values are in the window). */
export function rollingMean(
  values: readonly Maybe[],
  window: number,
  minObs = Math.ceil(window / 2),
): (number | null)[] {
  const out: (number | null)[] = new Array<number | null>(values.length).fill(null);
  let sum = 0;
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (finite(v)) {
      sum += v;
      n++;
    }
    const drop = values[i - window];
    if (i >= window && finite(drop)) {
      sum -= drop;
      n--;
    }
    out[i] = n >= minObs ? sum / n : null;
  }
  return out;
}

/** Trailing sample standard deviation over `window` points. */
export function rollingSd(
  values: readonly Maybe[],
  window: number,
  minObs = Math.ceil(window / 2),
): (number | null)[] {
  const out: (number | null)[] = new Array<number | null>(values.length).fill(null);
  for (let i = 0; i < values.length; i++) {
    const w = values.slice(Math.max(0, i - window + 1), i + 1).filter(finite);
    if (w.length < Math.max(2, minObs)) continue;
    const m = w.reduce((a, b) => a + b, 0) / w.length;
    out[i] = Math.sqrt(w.reduce((a, b) => a + (b - m) ** 2, 0) / (w.length - 1));
  }
  return out;
}

/** `YYYY-MM` of an ISO date. */
export function monthKey(iso: string): string {
  return iso.slice(0, 7);
}

/** Indices grouped by a key, in first-seen order. */
export function groupIndices(keys: readonly string[]): { key: string; indices: number[] }[] {
  const map = new Map<string, number[]>();
  keys.forEach((k, i) => {
    const arr = map.get(k);
    if (arr) arr.push(i);
    else map.set(k, [i]);
  });
  return [...map.entries()].map(([key, indices]) => ({ key, indices }));
}

/** Mean of the finite values at `indices` (null when none). */
export function meanAt(values: readonly Maybe[], indices: readonly number[]): number | null {
  let s = 0;
  let n = 0;
  for (const i of indices) {
    const v = values[i];
    if (finite(v)) {
      s += v;
      n++;
    }
  }
  return n ? s / n : null;
}

/** Positions of at most `max` evenly spaced items of `n`, always the first and the last. */
export function thinIndices(n: number, max: number): number[] {
  if (n <= max) return Array.from({ length: n }, (_, i) => i);
  const step = (n - 1) / (max - 1);
  const set = new Set<number>();
  for (let i = 0; i < max; i++) set.add(Math.round(i * step));
  set.add(0);
  set.add(n - 1);
  return [...set].sort((a, b) => a - b).filter((i) => i >= 0 && i < n);
}

export interface Bin {
  /** Inclusive lower edge. */
  from: number;
  /** Exclusive upper edge (inclusive for the last bin). */
  to: number;
  count: number;
}

/** Equal-width histogram bins over the finite values. */
export function histogram(values: readonly Maybe[], nBins = 20, range?: [number, number]): Bin[] {
  const xs = values.filter(finite);
  if (!xs.length) return [];
  const lo = range ? range[0] : Math.min(...xs);
  const hi = range ? range[1] : Math.max(...xs);
  const width = hi > lo ? (hi - lo) / nBins : 1;
  const bins: Bin[] = Array.from({ length: nBins }, (_, i) => ({
    from: lo + i * width,
    to: lo + (i + 1) * width,
    count: 0,
  }));
  for (const x of xs) {
    const k = Math.min(nBins - 1, Math.max(0, Math.floor((x - lo) / width)));
    const b = bins[k];
    if (b) b.count++;
  }
  return bins;
}

/** Quantile of the finite values (linear interpolation). */
export function quantile(values: readonly Maybe[], q: number): number | null {
  const xs = values.filter(finite).sort((a, b) => a - b);
  if (!xs.length) return null;
  const pos = (xs.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  const a = xs[lo] ?? 0;
  const b = xs[hi] ?? a;
  return a + (b - a) * (pos - lo);
}

/** Share of each category per group (rows sum to 1 unless the group is empty). */
export function shareByGroup(
  groups: readonly { key: string; indices: readonly number[] }[],
  category: readonly string[],
  categories: readonly string[],
): { key: string; shares: Record<string, number> }[] {
  return groups.map((g) => {
    const counts: Record<string, number> = Object.fromEntries(categories.map((c) => [c, 0]));
    let n = 0;
    for (const i of g.indices) {
      const c = category[i];
      if (c !== undefined && c in counts) {
        counts[c] = (counts[c] ?? 0) + 1;
        n++;
      }
    }
    const shares: Record<string, number> = {};
    for (const c of categories) shares[c] = n ? (counts[c] ?? 0) / n : 0;
    return { key: g.key, shares };
  });
}
