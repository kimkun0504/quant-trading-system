import type { PaperLevel, PaperSeries } from '@/shared/api';
import type { BadgeTone } from '@/shared/ui';

export const LEVEL_TONE: Record<PaperLevel, BadgeTone> = {
  OK: 'pass',
  WATCH: 'caution',
  STOP: 'fail',
};

export const LEVEL_TEXT: Record<PaperLevel, string> = {
  OK: 'The book behaves like its backtest: every check is inside its expected range.',
  WATCH: 'Something is outside its usual range. Trading continues; look at the checks below.',
  STOP: 'A stop rule fired. No new orders are generated until you acknowledge the status with a reason.',
};

export function isLevel(x: unknown): x is PaperLevel {
  return x === 'OK' || x === 'WATCH' || x === 'STOP';
}

/** Index ranges of consecutive live days (for shading on the charts). */
export function liveSpans(modes: readonly (string | null)[]): { from: number; to: number }[] {
  const out: { from: number; to: number }[] = [];
  let start = -1;
  modes.forEach((m, i) => {
    if (m === 'live' && start < 0) start = i;
    if (m !== 'live' && start >= 0) {
      out.push({ from: start, to: i - 1 });
      start = -1;
    }
  });
  if (start >= 0) out.push({ from: start, to: modes.length - 1 });
  return out;
}

/** Counts of live and backfill days. */
export function modeCounts(modes: readonly (string | null)[]): { live: number; backfill: number } {
  let live = 0;
  let backfill = 0;
  for (const m of modes) {
    if (m === 'live') live++;
    else if (m === 'backfill') backfill++;
  }
  return { live, backfill };
}

const nums = (v: readonly (number | null)[] | undefined) => v ?? [];

/** The series the book page draws, with the cones and boundaries aligned to the dates. */
export function chartSeries(s: PaperSeries) {
  const n = s.date.length;
  const pad = (v: readonly (number | null)[] | undefined) => {
    const a = [...nums(v)];
    while (a.length < n) a.push(null);
    return a.slice(0, n);
  };
  return {
    dates: s.date,
    cum: pad(s.cum_log),
    coneLo: pad(s.cone_analytic?.lower),
    coneHi: pad(s.cone_analytic?.upper),
    coneMid: pad(s.cone_analytic?.expected),
    bootLo: pad(s.cone_bootstrap?.lower),
    bootHi: pad(s.cone_bootstrap?.upper),
    dd: pad(s.drawdown),
    ddWatch: pad(s.boundaries?.dd_watch),
    ddStop: pad(s.boundaries?.dd_stop),
    gross: pad(s.gross),
    net: pad(s.net),
    turnover: pad(s.turnover),
    exante: pad(s.exante_vol),
    realised: pad(s.realised_vol_21d),
  };
}
