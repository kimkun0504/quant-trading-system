import type { ConfigSchema, JsonSchemaLeaf, JsonSchemaObject } from '@/shared/api';
import type { OverrideBounds } from './validation';

export type OverrideField = OverrideBounds & { leaf: JsonSchemaLeaf; group: string };

/** Flatten the schema's leaf properties into dotted override paths ("portfolio.trade_aversion"). */
export function overrideFields(schema: ConfigSchema | undefined): OverrideField[] {
  if (!schema) return [];
  const out: OverrideField[] = [];
  for (const [section, node] of Object.entries(schema.properties)) {
    if (node.type !== 'object') continue;
    const obj: JsonSchemaObject = node;
    for (const [key, leaf] of Object.entries(obj.properties)) {
      if (leaf.type === 'object') continue;
      out.push({
        path: `${section}.${key}`,
        title: leaf.title ?? key,
        type: leaf.type,
        ...(leaf.type !== 'boolean' && leaf.type !== 'string' && leaf.minimum !== undefined
          ? { minimum: leaf.minimum }
          : {}),
        ...(leaf.type !== 'boolean' && leaf.type !== 'string' && leaf.maximum !== undefined
          ? { maximum: leaf.maximum }
          : {}),
        leaf,
        group: obj.title ?? section,
      });
    }
  }
  return out;
}
