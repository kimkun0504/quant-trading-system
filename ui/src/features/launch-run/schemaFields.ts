import type { ConfigSchema, JsonSchemaLeaf, JsonSchemaObject } from '@/shared/api';
import type { OverrideBounds } from './validation';

export type OverrideField = OverrideBounds & { leaf: JsonSchemaLeaf; group: string };

/** Paths the form sets through its own fields (the API rejects them as overrides). */
const RESERVED = new Set(['seed', 'execution.signal_lag_days']);

/**
 * Flatten the schema's leaf properties into dotted override paths ("portfolio.trade_aversion",
 * "alpha.lgbm.num_leaves"), in schema order. Each leaf's group is the title of the object
 * that holds it. Top-level leaves of SystemConfig are left out (the form owns seed).
 */
export function overrideFields(schema: ConfigSchema | undefined): OverrideField[] {
  if (!schema) return [];
  const out: OverrideField[] = [];
  const walk = (obj: JsonSchemaObject, prefix: string, group: string) => {
    for (const [key, node] of Object.entries(obj.properties)) {
      const path = `${prefix}${key}`;
      if (node.type === 'object') {
        walk(node, `${path}.`, node.title ?? path);
        continue;
      }
      if (!prefix || RESERVED.has(path)) continue;
      out.push({
        path,
        title: node.title ?? key,
        type: node.type,
        ...(node.type !== 'boolean' && node.type !== 'string' && node.minimum !== undefined
          ? { minimum: node.minimum }
          : {}),
        ...(node.type !== 'boolean' && node.type !== 'string' && node.maximum !== undefined
          ? { maximum: node.maximum }
          : {}),
        leaf: node,
        group,
      });
    }
  };
  walk(schema, '', schema.title ?? 'SystemConfig');
  return out;
}

/** Fields grouped by section, in schema order. */
export function groupFields(fields: readonly OverrideField[]): [string, OverrideField[]][] {
  const groups = new Map<string, OverrideField[]>();
  for (const f of fields) groups.set(f.group, [...(groups.get(f.group) ?? []), f]);
  return [...groups.entries()];
}
