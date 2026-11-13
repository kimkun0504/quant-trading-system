import { useId } from 'react';
import type { ConfigLeaf } from '@/shared/api';
import { formatLeaf } from '@/entities/run';
import { cx } from '@/shared/lib/cx';
import { IconButton } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import type { Raw } from './lib/draft';
import { boundsText } from './lib/fields';
import type { Field } from './lib/fields';
import styles from './FieldControl.module.css';

export interface FieldControlProps {
  field: Field;
  /** What the input shows: the edit, or the base value. */
  raw: Raw;
  /** The base's value, shown beside a changed field and restored by Reset. */
  base: ConfigLeaf;
  changed: boolean;
  /** Client-side or API error for this field. */
  error?: string | undefined;
  /** Plain-language label and help (the Common settings panel); default: the schema's. */
  label?: string;
  help?: string;
  /** `row`: label left, input right (the full editor). `stack`: label above the input. */
  layout?: 'row' | 'stack';
  onChange: (raw: Raw) => void;
  onReset: () => void;
}

/**
 * One config field, drawn by its schema type: a switch for booleans, a select for
 * enums, a text box for strings, a numeric box with its bounds for numbers, and a
 * "No value" switch beside any field that accepts null. A changed field is marked,
 * shows the base value and has its own reset.
 */
export function FieldControl({
  field: f,
  raw,
  base,
  changed,
  error,
  label,
  help,
  layout = 'row',
  onChange,
  onReset,
}: FieldControlProps) {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [hintId, error ? errorId : ''].filter(Boolean).join(' ');
  const isNull = raw === null;
  const bounds = boundsText(f);
  const facts = [
    f.default !== undefined
      ? `Default ${formatLeaf(f.default)}`
      : f.nullable
        ? 'Default: no value'
        : '',
    bounds
      ? f.type === 'integer'
        ? `whole number, ${bounds}`
        : bounds
      : f.type === 'integer'
        ? 'whole number'
        : '',
  ].filter(Boolean);
  const description = help ?? f.description;

  let input;
  if (f.type === 'boolean') {
    input = (
      <label className={styles.switch}>
        <input
          id={id}
          type="checkbox"
          role="switch"
          checked={raw === true}
          disabled={isNull}
          onChange={(e) => onChange(e.target.checked)}
          aria-describedby={describedBy}
          aria-invalid={Boolean(error)}
        />
        <span className={styles.track} aria-hidden="true" />
        <span className={styles.switchWord}>{raw === true ? 'On' : 'Off'}</span>
      </label>
    );
  } else if (f.type === 'string' && f.enum) {
    input = (
      <select
        id={id}
        className={styles.input}
        value={typeof raw === 'string' ? raw : ''}
        disabled={isNull}
        onChange={(e) => onChange(e.target.value)}
        aria-describedby={describedBy}
        aria-invalid={Boolean(error)}
      >
        {isNull ? <option value="">No value</option> : null}
        {f.enum.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  } else {
    const numeric = f.type !== 'string';
    input = (
      <input
        id={id}
        className={cx(styles.input, numeric && styles.numeric)}
        type="text"
        inputMode={f.type === 'integer' ? 'numeric' : numeric ? 'decimal' : 'text'}
        value={typeof raw === 'string' ? raw : ''}
        placeholder={isNull ? 'No value' : ''}
        disabled={isNull}
        onChange={(e) => onChange(e.target.value)}
        aria-describedby={describedBy}
        aria-invalid={Boolean(error)}
        autoComplete="off"
        spellCheck={false}
      />
    );
  }

  return (
    <div
      className={cx(styles.field, styles[layout], changed && styles.changed)}
      data-field={f.path}
      data-changed={changed || undefined}
    >
      <div className={styles.head}>
        <label htmlFor={id} className={styles.label}>
          {label ?? f.title}
        </label>
        <code className={styles.path}>{f.path}</code>
      </div>
      <div className={styles.control}>
        <div className={styles.inputRow}>
          {input}
          {f.nullable ? (
            <label className={styles.none}>
              <input
                type="checkbox"
                checked={isNull}
                onChange={(e) =>
                  onChange(
                    e.target.checked
                      ? null
                      : f.type === 'boolean'
                        ? false
                        : f.default !== undefined && f.default !== null
                          ? String(f.default)
                          : '',
                  )
                }
                aria-label={`${label ?? f.title}: no value`}
              />
              No value
            </label>
          ) : null}
          {changed ? (
            <IconButton
              size="sm"
              label={`Reset ${label ?? f.title} to ${formatLeaf(base)}`}
              onClick={onReset}
            >
              <Icons.Reset />
            </IconButton>
          ) : null}
        </div>
        {error ? (
          <p id={errorId} className={styles.error}>
            {error}
          </p>
        ) : null}
        <p id={hintId} className={styles.hint}>
          {changed ? (
            <span className={styles.was}>
              Base <code>{formatLeaf(base)}</code>.{' '}
            </span>
          ) : null}
          {facts.join(', ')}
          {facts.length ? '.' : ''}
        </p>
      </div>
      {description ? <p className={styles.description}>{description}</p> : null}
    </div>
  );
}
