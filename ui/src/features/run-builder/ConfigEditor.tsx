import { useMemo, useState } from 'react';
import { Panel } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import { FieldControl } from './FieldControl';
import { baseValue, isChanged, rawValue } from './lib/draft';
import type { Base, Draft, Raw } from './lib/draft';
import { commonFields } from './lib/common';
import { fieldSections, matchesField } from './lib/fields';
import type { Field } from './lib/fields';
import styles from './ConfigEditor.module.css';

/** The draft's value of the field a multi-select follows (`alpha.feature_set`). */
function choiceKeyValue(
  fields: readonly Field[],
  draft: Draft,
  base: Base,
  f: Field,
): string | null {
  if (!f.choicesKey) return null;
  const key = fields.find((x) => x.path === f.choicesKey);
  if (!key) return null;
  const v = rawValue(draft, base, key);
  return typeof v === 'string' ? v : null;
}

interface EditorProps {
  fields: readonly Field[];
  draft: Draft;
  base: Base;
  /** Client and API errors by path. */
  errors: Readonly<Record<string, string>>;
  onChange: (field: Field, raw: Raw) => void;
  onReset: (path: string) => void;
}

/** The handful of settings that matter day to day, with plain-language labels. */
export function CommonSettings({ fields, draft, base, errors, onChange, onReset }: EditorProps) {
  const common = useMemo(() => commonFields(fields), [fields]);
  if (common.length === 0) return null;
  return (
    <Panel title="Common settings" subtitle="the knobs that change most often">
      <div className={styles.common}>
        {common.map((c) => (
          <FieldControl
            key={c.path}
            field={c.field}
            layout="stack"
            label={c.label}
            help={c.help}
            raw={rawValue(draft, base, c.field)}
            base={baseValue(base, c.field)}
            changed={isChanged(draft, base, c.field)}
            error={errors[c.path]}
            choiceKey={choiceKeyValue(fields, draft, base, c.field)}
            onChange={(raw) => onChange(c.field, raw)}
            onReset={() => onReset(c.path)}
          />
        ))}
      </div>
    </Panel>
  );
}

/**
 * Every scalar of SystemConfig, generated from the API's schema: one collapsible group
 * per section, a search over names and descriptions, and a "changed only" filter.
 * Searching or filtering opens the groups that match.
 */
export function ConfigEditor({
  fields,
  draft,
  base,
  errors,
  onChange,
  onReset,
  schemaSections,
}: EditorProps & { schemaSections?: Parameters<typeof fieldSections>[1] }) {
  const [query, setQuery] = useState('');
  const [changedOnly, setChangedOnly] = useState(false);
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set());
  const sections = useMemo(() => fieldSections(fields, schemaSections), [fields, schemaSections]);
  const changed = useMemo(
    () => new Set(fields.filter((f) => isChanged(draft, base, f)).map((f) => f.path)),
    [fields, draft, base],
  );
  const filtering = query.trim() !== '' || changedOnly;
  const visible = sections
    .map((s) => ({
      ...s,
      shown: s.fields.filter(
        (f) => matchesField(f, query) && (!changedOnly || changed.has(f.path)),
      ),
      nChanged: s.fields.filter((f) => changed.has(f.path)).length,
      nErrors: s.fields.filter((f) => errors[f.path]).length,
    }))
    .filter((s) => !filtering || s.shown.length > 0);
  const nShown = visible.reduce((a, s) => a + s.shown.length, 0);

  return (
    <Panel
      title="All settings"
      subtitle={`every scalar of SystemConfig, ${fields.length} in ${sections.length} sections`}
      flush
    >
      <div className={styles.toolbar}>
        <label className={styles.search}>
          <Icons.Search />
          <input
            type="search"
            aria-label="Search settings"
            placeholder="Search by name, path or description"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            autoComplete="off"
          />
        </label>
        <label className={styles.filter}>
          <input
            type="checkbox"
            checked={changedOnly}
            onChange={(e) => setChangedOnly(e.target.checked)}
          />
          Changed only
        </label>
        <span className={styles.count} role="status">
          {filtering ? `${nShown} of ${fields.length} settings` : `${changed.size} changed`}
        </span>
      </div>
      {visible.length === 0 ? (
        <p className={styles.none}>
          {changedOnly && changed.size === 0
            ? 'Nothing is changed yet: this variant is the base.'
            : `No setting matches “${query}”.`}
        </p>
      ) : null}
      {visible.map((s) => {
        const isOpen = filtering || open.has(s.path);
        return (
          <details
            key={s.path}
            className={styles.section}
            open={isOpen}
            onToggle={(e) => {
              if (filtering) return;
              const now = e.currentTarget.open;
              setOpen((prev) => {
                if (prev.has(s.path) === now) return prev;
                const next = new Set(prev);
                if (now) next.add(s.path);
                else next.delete(s.path);
                return next;
              });
            }}
          >
            <summary className={styles.summary}>
              <Icons.Chevron className={styles.chevron} />
              <span className={styles.sectionTitle}>{s.title}</span>
              {s.path ? <code className={styles.sectionPath}>{s.path}</code> : null}
              <span className={styles.sectionMeta}>
                {filtering
                  ? `${s.shown.length} of ${s.fields.length}`
                  : `${s.fields.length} settings`}
                {s.nChanged ? (
                  <span className={styles.changedCount}>{s.nChanged} changed</span>
                ) : null}
                {s.nErrors ? <span className={styles.errorCount}>{s.nErrors} to fix</span> : null}
              </span>
            </summary>
            {isOpen ? (
              <div>
                {s.description ? (
                  <p className={styles.sectionDescription}>{s.description}</p>
                ) : null}
                {s.shown.map((f) => (
                  <FieldControl
                    key={f.path}
                    field={f}
                    raw={rawValue(draft, base, f)}
                    base={baseValue(base, f)}
                    changed={changed.has(f.path)}
                    error={errors[f.path]}
                    choiceKey={choiceKeyValue(fields, draft, base, f)}
                    onChange={(raw) => onChange(f, raw)}
                    onReset={() => onReset(f.path)}
                  />
                ))}
              </div>
            ) : null}
          </details>
        );
      })}
    </Panel>
  );
}
