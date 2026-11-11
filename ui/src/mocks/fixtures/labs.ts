import type {
  AlphaIcByYear,
  AlphaRefits,
  AlphaThemeWeights,
  IcByYearRow,
  RunConstraints,
} from '@/shared/api';
import { hashString, mulberry32, normal } from './random';
import type { Rng } from './random';

/**
 * Lab fixtures (`/alpha/refits`, `/alpha/theme-weights`, `/alpha/ic-by-year`,
 * `/constraints`). They tell one 24-year story keyed on the calendar year, so a six-year
 * run and a 24-year run agree on the years they share: value fades after 2010, quality
 * rises from 2012, momentum crashes in 2009 and wobbles in 2020, low risk is steady, and
 * the ML sleeve's importance follows the same drift (gain moves from the value features to
 * quality and investment).
 */

/** Processed features in the order `FeatureBuilder.process` emits them, with the composite's sign. */
export const FEATURES: readonly { name: string; sign: 1 | -1; theme: string | null }[] = [
  { name: 'value_bp', sign: 1, theme: 'value' },
  { name: 'value_ep', sign: 1, theme: 'value' },
  { name: 'mom_12_1', sign: 1, theme: 'momentum' },
  { name: 'mom_6_1', sign: 1, theme: 'momentum' },
  { name: 'industry_momentum', sign: 1, theme: 'momentum' },
  { name: 'quality_gpa', sign: 1, theme: 'quality' },
  { name: 'investment_ag', sign: 1, theme: 'investment' },
  { name: 'beta_252', sign: -1, theme: 'low_risk' },
  { name: 'idio_vol_252', sign: -1, theme: 'low_risk' },
  { name: 'max_ret_21', sign: -1, theme: 'low_risk' },
  { name: 'short_interest', sign: -1, theme: 'short_interest' },
  { name: 'borrow_fee', sign: -1, theme: 'short_interest' },
  { name: 'mom_change', sign: 1, theme: null },
  { name: 'turnover_63', sign: 1, theme: null },
  { name: 'amihud_63', sign: 1, theme: null },
  { name: 'strev_21', sign: 1, theme: null },
];

export const THEMES = [
  'value',
  'momentum',
  'quality',
  'investment',
  'low_risk',
  'short_interest',
] as const;

const STATIC_WEIGHTS: Record<string, number> = {
  value: 0.2,
  momentum: 0.25,
  quality: 0.2,
  investment: 0.1,
  low_risk: 0.15,
  short_interest: 0.1,
};

/** Smooth "true" mean IC of each theme in a calendar year (the story). */
export function themeIcForYear(theme: string, year: number): number {
  const t = year - 2001;
  switch (theme) {
    case 'value':
      // strong through the 2000s, fading after 2010, slightly negative 2018–2020, a bounce in 2022
      return (
        0.045 * Math.exp(-Math.max(0, t - 9) / 6) -
        0.012 * (year >= 2017 && year <= 2020 ? 1 : 0) +
        (year === 2022 ? 0.02 : 0)
      );
    case 'quality':
      // weak early, rising from 2012
      return 0.006 + 0.034 / (1 + Math.exp(-(t - 13) / 2.2));
    case 'momentum':
      // steady, a crash in 2009, a wobble in 2020
      return (
        0.034 +
        (year === 2009 ? -0.07 : 0) +
        (year === 2020 ? -0.025 : 0) +
        (year === 2003 ? -0.02 : 0)
      );
    case 'investment':
      return 0.018 - 0.004 * Math.sin(t / 3);
    case 'low_risk':
      return (
        0.02 +
        (year === 2008 ? 0.03 : 0) +
        (year === 2009 ? -0.03 : 0) +
        (year === 2020 ? 0.015 : 0)
      );
    case 'short_interest':
      return 0.015 + 0.006 * Math.cos(t / 2.5) - (year === 2021 ? 0.03 : 0);
    default:
      return 0;
  }
}

function yearsBetween(dates: readonly string[]): number[] {
  const ys = new Set(dates.map((d) => Number(d.slice(0, 4))));
  return [...ys].sort((a, b) => a - b);
}

function countByYear(dates: readonly string[]): Map<number, number> {
  const m = new Map<number, number>();
  for (const d of dates) {
    const y = Number(d.slice(0, 4));
    m.set(y, (m.get(y) ?? 0) + 1);
  }
  return m;
}

/** `/alpha/ic-by-year`: rows combined, composite, ml (ML runs), theme_* with per-year mean and t. */
export function mockIcByYear(args: {
  id: string;
  dates: readonly string[];
  useMl: boolean;
  /** Scales every theme's story (a weak run scores lower everywhere). */
  level?: number;
}): AlphaIcByYear {
  const years = yearsBetween(args.dates);
  const n = countByYear(args.dates);
  const rng = mulberry32(hashString(`${args.id}:ic-by-year`));
  const level = args.level ?? 1;
  const dailySd = 0.11;
  const overlap = 21;
  const cell = (y: number, mean: number) => {
    const count = n.get(y) ?? 0;
    const noise = (dailySd / Math.sqrt(Math.max(1, count))) * normal(rng) * 0.6;
    const m = mean * level + noise;
    // Newey–West t of overlapping daily labels ≈ plain t over effective samples
    const t = (m / dailySd) * Math.sqrt(Math.max(1, count) / overlap);
    return { year: y, mean_ic: m, t_stat: t, n: count };
  };
  const rows: IcByYearRow[] = [];
  const themeMeans = (y: number) =>
    Object.fromEntries(THEMES.map((th) => [th, themeIcForYear(th, y)])) as Record<string, number>;
  const composite = (y: number) => {
    const tm = themeMeans(y);
    const num = THEMES.reduce((a, th) => a + (STATIC_WEIGHTS[th] ?? 0) * (tm[th] ?? 0), 0);
    // a diversified blend of correlated sleeves: a little better than the weighted mean
    return num * 1.35;
  };
  const ml = (y: number) => composite(y) * 1.12 + 0.004;
  const combined = (y: number) =>
    args.useMl ? 0.6 * composite(y) + 0.4 * ml(y) + 0.003 : composite(y);
  rows.push({ sleeve: 'combined', cells: years.map((y) => cell(y, combined(y))) });
  rows.push({ sleeve: 'composite', cells: years.map((y) => cell(y, composite(y))) });
  if (args.useMl) rows.push({ sleeve: 'ml', cells: years.map((y) => cell(y, ml(y))) });
  for (const th of THEMES)
    rows.push({ sleeve: `theme_${th}`, cells: years.map((y) => cell(y, themeIcForYear(th, y))) });
  return { years, rows, source: 'daily_sleeves', overlap_days: overlap };
}

/** Raw gain of a feature at a refit (the UI normalises per refit). */
function featureGain(feature: string, year: number, rng: Rng): number {
  const f = FEATURES.find((x) => x.name === feature);
  const themeIc = f?.theme ? themeIcForYear(f.theme, year) : 0.012;
  // trees see |IC|: gain tracks the squared signal plus a liquidity-ish baseline
  const base = feature === 'turnover_63' || feature === 'amihud_63' ? 0.9 : 0.35;
  const perMember = f?.theme === 'momentum' ? 0.6 : f?.theme === 'value' ? 0.8 : 1;
  return (base + 420 * themeIc * themeIc * perMember) * (0.85 + 0.3 * rng());
}

/** `/alpha/refits`: one refit a year from the first backtest date (ML runs); composite runs record only the feature IC. */
export function mockAlphaRefits(args: {
  id: string;
  dates: readonly string[];
  useMl: boolean;
  level?: number;
}): AlphaRefits {
  const rng = mulberry32(hashString(`${args.id}:refits`));
  const level = args.level ?? 1;
  const feature_ic = FEATURES.map((f) => {
    const years = yearsBetween(args.dates);
    const mean =
      (f.theme
        ? years.reduce((a, y) => a + themeIcForYear(f.theme ?? '', y), 0) /
          Math.max(1, years.length)
        : 0.008) *
        level *
        (0.7 + 0.5 * rng()) +
      0.003 * normal(rng);
    const n = args.dates.length;
    return {
      feature: f.name,
      sign: f.sign,
      theme: f.theme,
      mean_ic: mean,
      t_stat: (mean / 0.11) * Math.sqrt(n / 21),
      n,
    };
  });
  if (!args.useMl) {
    return {
      refits: [],
      sleeves: [],
      features: [],
      importance: {},
      feature_ic,
      embargo_days: 21,
      refit_frequency_days: 252,
      notes: [],
      recorded: true,
    };
  }
  const refitIdx: number[] = [];
  for (let i = 0; i < args.dates.length; i += 252) refitIdx.push(i);
  const refits = refitIdx.map((i, k) => {
    const d = args.dates[i] ?? '';
    const end = args.dates[Math.min(args.dates.length - 1, i + 251)] ?? d;
    const trainEnd = args.dates[Math.max(0, i - 21)] ?? d;
    return {
      refit_date: d,
      train_start: shiftYears(d, -(3 + k)),
      train_end: i >= 21 ? trainEnd : shiftDays(d, -30),
      valid_start: shiftYears(d, -1),
      pred_end: end,
      n_train: 180_000 + 62_000 * k,
      n_valid: 62_000,
      blend_weights: { lgbm: 0.6, mlp: 0.4 },
    };
  });
  const features = FEATURES.map((f) => f.name);
  const importance: AlphaRefits['importance'] = { lgbm: {}, ridge: {}, ensemble: {} };
  for (const f of features) {
    const lgbm = refits.map((r) => featureGain(f, Number(r.refit_date.slice(0, 4)), rng));
    const total = lgbm.reduce((a, b) => a + b, 0) / Math.max(1, lgbm.length);
    const ridge = refits.map((r) => {
      const fd = FEATURES.find((x) => x.name === f);
      const ic = fd?.theme ? themeIcForYear(fd.theme, Number(r.refit_date.slice(0, 4))) : 0.006;
      return Math.abs(ic * 8 * (0.8 + 0.4 * rng()));
    });
    importance.lgbm = { ...importance.lgbm, [f]: lgbm.map((v) => v / Math.max(1e-9, total)) };
    importance.ridge = { ...importance.ridge, [f]: ridge };
    importance.ensemble = {
      ...importance.ensemble,
      [f]: lgbm.map((v, i) => (v / Math.max(1e-9, total) + (ridge[i] ?? 0)) / 2),
    };
  }
  return {
    refits,
    sleeves: ['lgbm', 'ridge', 'ensemble'],
    features,
    importance,
    feature_ic,
    embargo_days: 21,
    refit_frequency_days: 252,
    notes: ['mlp: no feature importance recorded (permutation importance is not computed)'],
    recorded: true,
  };
}

function shiftDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function shiftYears(iso: string, years: number): string {
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCFullYear(d.getUTCFullYear() + years);
  return d.toISOString().slice(0, 10);
}

/** `/alpha/theme-weights`: static weights × a factor-momentum tilt (±25 %), momentum also crash- and vol-scaled. */
export function mockThemeWeights(args: {
  id: string;
  dates: readonly string[];
  dynamic?: boolean;
}): AlphaThemeWeights {
  const rng = mulberry32(hashString(`${args.id}:theme-weights`));
  const dynamic = args.dynamic ?? true;
  const series: Record<string, number[]> = {};
  for (const th of THEMES) {
    const base = STATIC_WEIGHTS[th] ?? 0;
    let tilt = 0;
    series[th] = args.dates.map((d, i) => {
      if (!dynamic) return base;
      // AR(1) tilt inside ±0.25, nudged by the year's story
      const y = Number(d.slice(0, 4));
      const story = Math.max(-0.25, Math.min(0.25, themeIcForYear(th, y) * 4 - 0.1));
      tilt = 0.97 * tilt + 0.03 * story + 0.01 * normal(rng);
      tilt = Math.max(-0.25, Math.min(0.25, tilt));
      let w = base * (1 + tilt);
      if (th === 'momentum') {
        const crash =
          (y === 2009 && i % 252 < 160) || (y === 2020 && i % 252 > 50 && i % 252 < 140);
        const volScalar = 0.5 + 1.0 * (0.5 + 0.5 * Math.sin(i / 90));
        w *= (crash ? 0.25 : 1) * Math.max(0.5, Math.min(1.5, volScalar));
      }
      return w;
    });
  }
  return {
    dates: [...args.dates],
    series,
    themes: [...THEMES],
    static: { ...STATIC_WEIGHTS },
    dynamic,
    downsampled: false,
  };
}

const CONSTRAINT_NAMES = ['gross', 'position', 'participation', 'style_bands', 'industry'] as const;

/** `/constraints`: one row per decision date; caps bind more in the crash years and when trade aversion is off. */
export function mockConstraints(args: {
  id: string;
  dates: readonly string[];
  turnover: readonly number[];
  emptyDays: ReadonlySet<number>;
  tradeAversion: number;
  volTargetMode: 'target' | 'none';
}): RunConstraints {
  const rng = mulberry32(hashString(`${args.id}:constraints`));
  const n = args.dates.length;
  const names = [...CONSTRAINT_NAMES];
  const binding: Record<string, number[]> = Object.fromEntries(names.map((k) => [k, []]));
  const status: string[] = [];
  const vtb: string[] = [];
  const tc: (number | null)[] = [];
  const nTraded: number[] = [];
  const nSolves: number[] = [];
  let tcLevel = 0.62;
  for (let i = 0; i < n; i++) {
    const d = args.dates[i] ?? '';
    const y = Number(d.slice(0, 4));
    const stress = y === 2008 || y === 2009 || y === 2020 ? 1.6 : 1;
    const to = args.turnover[i] ?? 0.05;
    const empty = args.emptyDays.has(i);
    const failed = !empty && rng() < 0.004;
    status.push(empty ? 'no_alpha' : failed ? 'infeasible' : 'optimal');
    binding.gross?.push(empty ? 0 : rng() < 0.55 * stress ? 1 : 0);
    binding.position?.push(
      empty ? 0 : Math.round(rng() * 14 * stress + (args.tradeAversion === 0 ? 6 : 0)),
    );
    binding.participation?.push(
      empty ? 0 : Math.round(Math.max(0, (to - 0.04) * 900 * stress + rng() * 6)),
    );
    binding.style_bands?.push(empty ? 0 : rng() < 0.3 ? 1 + Math.floor(rng() * 3) : 0);
    binding.industry?.push(empty ? 0 : rng() < 0.18 ? 1 + Math.floor(rng() * 2) : 0);
    const r = rng();
    vtb.push(
      args.volTargetMode === 'none'
        ? ''
        : empty
          ? 'no_alpha'
          : failed
            ? 'solve_failed:infeasible'
            : r < (stress > 1 ? 0.45 : 0.14)
              ? 'caps'
              : r < 0.97
                ? 'target'
                : 'bracket',
    );
    tcLevel =
      0.985 * tcLevel + 0.015 * (args.tradeAversion === 0 ? 0.74 : 0.6) + 0.012 * normal(rng);
    tc.push(empty ? null : Math.max(0.2, Math.min(0.95, tcLevel - 0.08 * (stress - 1))));
    nTraded.push(empty ? 0 : Math.round(120 + 90 * Math.min(1, to / 0.08) + rng() * 20));
    nSolves.push(empty ? 0 : args.volTargetMode === 'none' ? 1 : 3 + Math.floor(rng() * 5));
  }
  const share_binding = Object.fromEntries(
    names.map((k) => [k, (binding[k] ?? []).filter((v) => v > 0).length / Math.max(1, n)]),
  );
  return {
    dates: [...args.dates],
    names,
    binding,
    status,
    vol_target_binding: vtb,
    transfer_coefficient: tc,
    n_traded: nTraded,
    n_solves: nSolves,
    share_binding,
    vol_target_reasons: [...new Set(vtb.filter(Boolean))].sort(),
    n_rebalances: n,
  };
}
