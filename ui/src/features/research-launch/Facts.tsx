import type { ReactNode } from 'react';
import styles from './layout.module.css';

export interface Fact {
  label: ReactNode;
  value: ReactNode;
  note?: ReactNode;
  key: string;
}

/** A row of headline facts: label (often a self-explaining Stat), value, short note. */
export function Facts({ facts, label }: { facts: readonly Fact[]; label: string }) {
  return (
    <dl className={styles.facts} aria-label={label}>
      {facts.map((f) => (
        <div key={f.key} data-fact={f.key}>
          <dt>{f.label}</dt>
          <dd>
            {f.value}
            {f.note ? <span className={styles.factNote}>{f.note}</span> : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}
