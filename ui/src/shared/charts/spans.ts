import type { Span } from './TimeSeriesChart';

/** Contiguous runs of indices where `empty(i)` holds, as inclusive spans. */
export function spansWhere(n: number, empty: (i: number) => boolean): Span[] {
  const out: Span[] = [];
  let start: number | null = null;
  for (let i = 0; i <= n; i++) {
    const hit = i < n && empty(i);
    if (hit && start === null) start = i;
    if (!hit && start !== null) {
      out.push({ from: start, to: i - 1 });
      start = null;
    }
  }
  return out;
}
