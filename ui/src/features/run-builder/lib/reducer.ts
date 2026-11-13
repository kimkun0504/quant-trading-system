import type { Base, Draft, Raw, RunSettingKey, RunSettings } from './draft';
import { baseValue, defaultBase, parseRaw, sameLeaf } from './draft';
import type { Field } from './fields';

/** At most this many variants are lined up at once (the tray's own limit). */
export const MAX_VARIANTS = 8;

export interface BuilderState {
  base: Base;
  variants: Draft[];
  /** Id of the variant being edited. */
  active: string;
  /** Counter behind the variant ids (stable across removals). */
  nextId: number;
}

export type BuilderAction =
  | { type: 'loadBase'; base: Base }
  | { type: 'setValue'; field: Field; raw: Raw }
  | { type: 'resetField'; path: string }
  | { type: 'setRun'; key: RunSettingKey; value: RunSettings[RunSettingKey] }
  | { type: 'resetRun'; key: RunSettingKey }
  | { type: 'resetVariant' }
  | { type: 'addVariant' }
  | { type: 'removeVariant'; id: string }
  | { type: 'selectVariant'; id: string }
  | { type: 'launched'; ids: readonly string[] };

const fresh = (n: number): Draft => ({ id: `v${n}`, values: {}, run: {} });

export function initialState(base: Base = defaultBase()): BuilderState {
  return { base, variants: [fresh(1)], active: 'v1', nextId: 2 };
}

function editActive(s: BuilderState, edit: (d: Draft) => Draft): BuilderState {
  return { ...s, variants: s.variants.map((d) => (d.id === s.active ? edit(d) : d)) };
}

function omit<T extends object>(obj: T, key: string): T {
  return Object.fromEntries(Object.entries(obj).filter(([k]) => k !== key)) as T;
}

/**
 * The builder's state: one base and the variants edited against it. A value equal to
 * the base's is dropped from the draft, so "changed" always means "differs from base".
 */
export function builderReducer(s: BuilderState, a: BuilderAction): BuilderState {
  switch (a.type) {
    case 'loadBase':
      // a new base invalidates every edit: they were differences from the old one
      return initialState(a.base);
    case 'setValue':
      return editActive(s, (d) => {
        const p = parseRaw(a.field, a.raw);
        const same = p.ok && sameLeaf(p.value, baseValue(s.base, a.field));
        return {
          ...d,
          values: same ? omit(d.values, a.field.path) : { ...d.values, [a.field.path]: a.raw },
        };
      });
    case 'resetField':
      return editActive(s, (d) => ({ ...d, values: omit(d.values, a.path) }));
    case 'setRun':
      return editActive(s, (d) => ({
        ...d,
        run: s.base.run[a.key] === a.value ? omit(d.run, a.key) : { ...d.run, [a.key]: a.value },
      }));
    case 'resetRun':
      return editActive(s, (d) => ({
        ...d,
        run: omit(d.run, a.key),
      }));
    case 'resetVariant':
      return editActive(s, (d) => ({ ...d, values: {}, run: {} }));
    case 'addVariant': {
      if (s.variants.length >= MAX_VARIANTS) return s;
      const src = s.variants.find((d) => d.id === s.active) ?? fresh(0);
      // the copy gets its own (automatic) label
      const copy: Draft = {
        id: `v${s.nextId}`,
        values: { ...src.values },
        run: omit(src.run, 'label'),
      };
      const at = s.variants.findIndex((d) => d.id === s.active);
      const variants = [...s.variants.slice(0, at + 1), copy, ...s.variants.slice(at + 1)];
      return { ...s, variants, active: copy.id, nextId: s.nextId + 1 };
    }
    case 'removeVariant': {
      if (s.variants.length <= 1) return s;
      const at = s.variants.findIndex((d) => d.id === a.id);
      if (at < 0) return s;
      const variants = s.variants.filter((d) => d.id !== a.id);
      const active =
        s.active === a.id
          ? (variants[Math.min(at, variants.length - 1)]?.id ?? s.active)
          : s.active;
      return { ...s, variants, active };
    }
    case 'selectVariant':
      return s.variants.some((d) => d.id === a.id) ? { ...s, active: a.id } : s;
    case 'launched': {
      // queued variants leave the bench; the last one stays as the next starting point
      const left = s.variants.filter((d) => !a.ids.includes(d.id));
      if (left.length === 0) {
        const keep = s.variants[s.variants.length - 1] ?? fresh(s.nextId);
        return { ...s, variants: [keep], active: keep.id };
      }
      const active = left.some((d) => d.id === s.active) ? s.active : (left[0]?.id ?? s.active);
      return { ...s, variants: left, active };
    }
  }
}

/** "A", "B", ... by position. */
export function variantLetter(index: number): string {
  return String.fromCharCode(65 + (index % 26));
}
