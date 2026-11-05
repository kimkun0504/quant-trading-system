/** Small numeric helpers shared by selectors and fixtures. Inputs are plain arrays. */

export function mean(xs: readonly number[]): number {
  if (xs.length === 0) return Number.NaN;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/** Sample standard deviation (ddof = 1), as in pandas. */
export function sd(xs: readonly number[]): number {
  const n = xs.length;
  if (n < 2) return Number.NaN;
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) ** 2;
  return Math.sqrt(s / (n - 1));
}

/** Annualised Sharpe of periodic returns (rf = 0, as in qts.backtest.metrics). */
export function annualizedSharpe(rets: readonly number[], periodsPerYear = 252): number {
  const s = sd(rets);
  if (!Number.isFinite(s) || s === 0) return Number.NaN;
  return (mean(rets) / s) * Math.sqrt(periodsPerYear);
}

export function annualizedVol(rets: readonly number[], periodsPerYear = 252): number {
  return sd(rets) * Math.sqrt(periodsPerYear);
}

export function cagr(rets: readonly number[], periodsPerYear = 252): number {
  if (rets.length === 0) return Number.NaN;
  let g = 1;
  for (const r of rets) g *= 1 + r;
  if (g <= 0) return -1;
  return g ** (periodsPerYear / rets.length) - 1;
}

/** Drawdown path of compounded returns: NAV / running max − 1 (≤ 0). */
export function drawdownPath(rets: readonly number[]): number[] {
  const out: number[] = [];
  let nav = 1;
  let peak = 1;
  for (const r of rets) {
    nav *= 1 + r;
    peak = Math.max(peak, nav);
    out.push(nav / peak - 1);
  }
  return out;
}

export function maxDrawdown(rets: readonly number[]): number {
  const dd = drawdownPath(rets);
  return dd.length ? Math.min(0, ...dd) : Number.NaN;
}

export function sortino(rets: readonly number[], periodsPerYear = 252): number {
  if (rets.length === 0) return Number.NaN;
  let s = 0;
  for (const r of rets) s += Math.min(r, 0) ** 2;
  const dd = Math.sqrt(s / rets.length);
  return dd === 0 ? Number.NaN : (mean(rets) / dd) * Math.sqrt(periodsPerYear);
}

export function hitRate(rets: readonly number[]): number {
  const nz = rets.filter((r) => r !== 0);
  return nz.length ? nz.filter((r) => r > 0).length / nz.length : Number.NaN;
}

/** Standard normal CDF (Abramowitz & Stegun 7.1.26 via erf). */
export function normCdf(x: number): number {
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-z * z);
  return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

/** Inverse standard normal CDF (Acklam's rational approximation). */
export function normInv(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [
    -39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472,
    2.50662827745924,
  ];
  const b = [
    -54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857,
  ];
  const c = [
    -0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373,
    4.37466414146497, 2.93816398269878,
  ];
  const d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742];
  const at = (arr: number[], i: number) => arr[i] ?? 0;
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((at(c, 0) * q + at(c, 1)) * q + at(c, 2)) * q + at(c, 3)) * q + at(c, 4)) * q +
        at(c, 5)) /
      ((((at(d, 0) * q + at(d, 1)) * q + at(d, 2)) * q + at(d, 3)) * q + 1)
    );
  }
  if (p > 1 - pl) return -normInv(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((at(a, 0) * r + at(a, 1)) * r + at(a, 2)) * r + at(a, 3)) * r + at(a, 4)) * r + at(a, 5)) *
      q) /
    (((((at(b, 0) * r + at(b, 1)) * r + at(b, 2)) * r + at(b, 3)) * r + at(b, 4)) * r + 1)
  );
}

/**
 * Deflated Sharpe ratio (Bailey & López de Prado), as in the handbook's worked example:
 * expected max SR of N trials with cross-trial variance V, then PSR against it.
 */
export function deflatedSharpe(args: {
  sharpeAnnual: number;
  nObs: number;
  nTrials: number;
  varTrialsAnnual: number;
  skew: number;
  kurtosis: number;
}): number {
  const { sharpeAnnual, nObs, nTrials, varTrialsAnnual, skew, kurtosis } = args;
  const gamma = 0.5772156649;
  const srMax =
    nTrials <= 1
      ? 0
      : Math.sqrt(varTrialsAnnual) *
        ((1 - gamma) * normInv(1 - 1 / nTrials) + gamma * normInv(1 - 1 / (nTrials * Math.E)));
  const srd = sharpeAnnual / Math.sqrt(252);
  const seD = Math.sqrt((1 - skew * srd + ((kurtosis - 1) / 4) * srd * srd) / (nObs - 1));
  const z = (sharpeAnnual - srMax) / (seD * Math.sqrt(252));
  return normCdf(z);
}
