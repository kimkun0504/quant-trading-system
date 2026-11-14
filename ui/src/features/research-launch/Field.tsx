import { useId } from 'react';
import type { ReactNode } from 'react';
import styles from './research.module.css';

/** A labelled text input with its default, help and error (the research launch forms). */
export function TextField({
  label,
  value,
  onChange,
  help,
  error,
  placeholder,
  numeric = false,
  wide = false,
  name,
}: {
  label: ReactNode;
  value: string;
  onChange: (v: string) => void;
  help?: ReactNode;
  error?: string | undefined;
  placeholder?: string;
  numeric?: boolean;
  wide?: boolean;
  name?: string;
}) {
  const id = useId();
  const helpId = `${id}-help`;
  const errId = `${id}-err`;
  return (
    <div className={styles.field} data-field={name}>
      <label htmlFor={id} className={styles.label}>
        {label}
      </label>
      <input
        id={id}
        name={name}
        className={`${styles.input} ${numeric ? styles.numeric : ''} ${wide ? styles.wide : ''}`}
        value={value}
        placeholder={placeholder}
        inputMode={numeric ? 'decimal' : undefined}
        autoComplete="off"
        spellCheck={false}
        aria-invalid={Boolean(error)}
        aria-describedby={
          [help ? helpId : '', error ? errId : ''].filter(Boolean).join(' ') || undefined
        }
        onChange={(e) => onChange(e.target.value)}
      />
      {error ? (
        <p id={errId} className={styles.error}>
          {error}
        </p>
      ) : null}
      {help ? (
        <p id={helpId} className={styles.help}>
          {help}
        </p>
      ) : null}
    </div>
  );
}

/** A labelled switch (checkbox). */
export function SwitchField({
  label,
  checked,
  onChange,
  help,
  name,
}: {
  label: ReactNode;
  checked: boolean;
  onChange: (v: boolean) => void;
  help?: ReactNode;
  name?: string;
}) {
  const id = useId();
  return (
    <div className={styles.field} data-field={name}>
      <label className={styles.check} htmlFor={id}>
        <input
          id={id}
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          aria-describedby={help ? `${id}-help` : undefined}
        />
        <span>{label}</span>
      </label>
      {help ? (
        <p id={`${id}-help`} className={styles.help}>
          {help}
        </p>
      ) : null}
    </div>
  );
}

/** A labelled select. */
export function SelectField<V extends string>({
  label,
  value,
  options,
  onChange,
  help,
  name,
}: {
  label: ReactNode;
  value: V;
  options: readonly { value: V; label: string }[];
  onChange: (v: V) => void;
  help?: ReactNode;
  name?: string;
}) {
  const id = useId();
  return (
    <div className={styles.field} data-field={name}>
      <label htmlFor={id} className={styles.label}>
        {label}
      </label>
      <select
        id={id}
        className={styles.input}
        value={value}
        aria-describedby={help ? `${id}-help` : undefined}
        onChange={(e) => {
          const next = options.find((o) => o.value === e.target.value);
          if (next) onChange(next.value);
        }}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      {help ? (
        <p id={`${id}-help`} className={styles.help}>
          {help}
        </p>
      ) : null}
    </div>
  );
}
