/** "2023-10-17" → unix seconds at 00:00 UTC (uPlot's time unit). */
export function toUnix(d: string): number {
  return Date.parse(`${d}T00:00:00Z`) / 1000;
}

export function toUnixAll(dates: readonly string[]): number[] {
  return dates.map(toUnix);
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/**
 * Axis labels (UTC): the year the first time it appears, "Mon" inside a year that is
 * already labelled; on spans under three years "Mon YY" everywhere but January.
 */
export function timeTickLabels(_u: unknown, splits: number[]): string[] {
  const span = (splits[splits.length - 1] ?? 0) - (splits[0] ?? 0);
  const long = span > 3 * 365 * 86400;
  let lastYear: number | null = null;
  return splits.map((s) => {
    const d = new Date(s * 1000);
    const year = d.getUTCFullYear();
    const month = MONTHS[d.getUTCMonth()] ?? '';
    if (long) {
      if (year !== lastYear) {
        lastYear = year;
        return String(year);
      }
      return month;
    }
    if (d.getUTCMonth() === 0) return String(year);
    return `${month} ${String(year).slice(2)}`;
  });
}
