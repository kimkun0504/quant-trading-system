import type { ConfigLeaves } from '@/shared/api';
import { formatLeaf } from '@/entities/run';

export interface ConfigGroup {
  section: string;
  rows: { key: string; leaf: string; value: string }[];
}

/** `{ "alpha.horizon_days": 20 }` → groups by top-level section, keys sorted inside. */
export function groupLeaves(leaves: ConfigLeaves, query: string): ConfigGroup[] {
  const q = query.trim().toLowerCase();
  const groups = new Map<string, ConfigGroup>();
  for (const key of Object.keys(leaves).sort()) {
    const v = leaves[key];
    const value = formatLeaf(v, typeof v === 'number' && !Number.isInteger(v));
    if (q && !key.toLowerCase().includes(q) && !value.toLowerCase().includes(q)) continue;
    const dot = key.indexOf('.');
    const section = dot < 0 ? 'top level' : key.slice(0, dot);
    const leaf = dot < 0 ? key : key.slice(dot + 1);
    const g: ConfigGroup = groups.get(section) ?? { section, rows: [] };
    g.rows.push({ key, leaf, value });
    groups.set(section, g);
  }
  return [...groups.values()];
}
