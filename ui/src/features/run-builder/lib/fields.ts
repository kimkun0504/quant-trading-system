import type { ConfigLeaf, ConfigSchema, JsonSchemaLeaf, JsonSchemaObject } from '@/shared/api';

/** One editable scalar of SystemConfig, flattened from the API's config schema. */
export interface Field {
  /** Dotted path, the override key: `portfolio.trade_aversion`. */
  path: string;
  /** Last path segment. */
  name: string;
  title: string;
  type: JsonSchemaLeaf['type'];
  /** Dotted path of the section that holds it ('' for top-level leaves). */
  section: string;
  sectionTitle: string;
  description?: string;
  /** The schema default; undefined when the default is null (or there is none). */
  default?: ConfigLeaf;
  nullable: boolean;
  enum?: readonly string[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
}

export interface FieldSection {
  path: string;
  title: string;
  description?: string;
  fields: Field[];
}

/**
 * Paths the builder sets through its own controls (the API rejects them as overrides) and
 * the operational `cache` section, which is a launch flag, not part of the config hash.
 */
export const RESERVED_PATHS: ReadonlySet<string> = new Set(['seed', 'execution.signal_lag_days']);
export const HIDDEN_SECTIONS: ReadonlySet<string> = new Set(['cache']);

const GENERAL = 'General';

/**
 * Every scalar leaf of the schema as a `Field`, in schema order: whatever `SystemConfig`
 * has today, so a field added to the config appears here with its type, default, bounds
 * and description without a UI change.
 */
export function schemaFields(schema: ConfigSchema | undefined): Field[] {
  if (!schema) return [];
  const out: Field[] = [];
  const walk = (obj: JsonSchemaObject, prefix: string, sectionTitle: string) => {
    for (const [key, node] of Object.entries(obj.properties)) {
      const path = `${prefix}${key}`;
      if (node.type === 'object') {
        if (!HIDDEN_SECTIONS.has(path)) walk(node, `${path}.`, node.title ?? path);
        continue;
      }
      if (RESERVED_PATHS.has(path)) continue;
      const f: Field = {
        path,
        name: key,
        title: node.title ?? key,
        type: node.type,
        section: prefix.replace(/\.$/, ''),
        sectionTitle,
        nullable: node.nullable === true,
      };
      if (node.description) f.description = node.description;
      if (node.default !== undefined) f.default = node.default;
      if (node.type === 'string' && node.enum) f.enum = node.enum;
      if (node.type === 'number' || node.type === 'integer') {
        if (node.minimum !== undefined) f.minimum = node.minimum;
        if (node.maximum !== undefined) f.maximum = node.maximum;
        if (node.exclusiveMinimum !== undefined) f.exclusiveMinimum = node.exclusiveMinimum;
        if (node.exclusiveMaximum !== undefined) f.exclusiveMaximum = node.exclusiveMaximum;
      }
      out.push(f);
    }
  };
  walk(schema, '', GENERAL);
  return out;
}

/** Fields grouped by section in schema order; top-level leaves go last, under "General". */
export function fieldSections(fields: readonly Field[], schema?: ConfigSchema): FieldSection[] {
  const map = new Map<string, FieldSection>();
  for (const f of fields) {
    let s = map.get(f.section);
    if (!s) {
      s = { path: f.section, title: f.sectionTitle, fields: [] };
      const d = sectionDescription(schema, f.section);
      if (d) s.description = d;
      map.set(f.section, s);
    }
    s.fields.push(f);
  }
  const all = [...map.values()];
  return [...all.filter((s) => s.path !== ''), ...all.filter((s) => s.path === '')];
}

function sectionDescription(schema: ConfigSchema | undefined, path: string): string | undefined {
  if (!schema || !path) return undefined;
  let node: JsonSchemaObject = schema;
  for (const part of path.split('.')) {
    const next = node.properties[part];
    if (!next || next.type !== 'object') return undefined;
    node = next;
  }
  return node.description;
}

/** "0 to 10", "at least 1", "greater than 0 and at most 1"; '' without bounds. */
export function boundsText(f: Field): string {
  const lo =
    f.exclusiveMinimum !== undefined
      ? `greater than ${f.exclusiveMinimum}`
      : f.minimum !== undefined
        ? `at least ${f.minimum}`
        : '';
  const hi =
    f.exclusiveMaximum !== undefined
      ? `less than ${f.exclusiveMaximum}`
      : f.maximum !== undefined
        ? `at most ${f.maximum}`
        : '';
  if (f.minimum !== undefined && f.maximum !== undefined) return `${f.minimum} to ${f.maximum}`;
  return [lo, hi].filter(Boolean).join(' and ');
}

/** Case-insensitive match of every term against path, title and description. */
export function matchesField(f: Field, query: string): boolean {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return true;
  const hay = `${f.path} ${f.title} ${f.sectionTitle} ${f.description ?? ''}`.toLowerCase();
  return terms.every((t) => hay.includes(t));
}
