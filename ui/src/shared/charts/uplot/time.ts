/** "2023-10-17" → unix seconds at 00:00 UTC (uPlot's time unit). */
export function toUnix(d: string): number {
  return Date.parse(`${d}T00:00:00Z`) / 1000;
}

export function toUnixAll(dates: readonly string[]): number[] {
  return dates.map(toUnix);
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/** Axis labels: years at year boundaries, otherwise "Mon" (UTC). */
export function timeTickLabels(_u: unknown, splits: number[]): string[] {
  const span = (splits[splits.length - 1] ?? 0) - (splits[0] ?? 0);
  return splits.map((s) => {
    const d = new Date(s * 1000);
    if (span > 3 * 365 * 86400 || d.getUTCMonth() === 0) return String(d.getUTCFullYear());
    return `${MONTHS[d.getUTCMonth()] ?? ''} ${String(d.getUTCFullYear()).slice(2)}`;
  });
}
