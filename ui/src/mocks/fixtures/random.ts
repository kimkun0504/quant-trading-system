/** Seeded randomness and normal-distribution helpers for deterministic fixtures. */

export type Rng = () => number;

/** mulberry32: small, fast, deterministic PRNG in [0, 1). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export function normal(rng: Rng): number {
  let u = 0;
  while (u === 0) u = rng();
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Standardise to sample mean 0 and sample sd 1 exactly. */
export function standardise(xs: number[]): number[] {
  const n = xs.length;
  const m = xs.reduce((a, b) => a + b, 0) / n;
  const s = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (n - 1));
  return xs.map((x) => (x - m) / s);
}

export { normCdf, normInv, deflatedSharpe } from '@/shared/lib/stats';
