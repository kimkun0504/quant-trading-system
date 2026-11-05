import type { CSSProperties } from 'react';
import type { GateResult } from '@/shared/api';
import { STATUS_WORD, gateDef, orderGates } from '@/entities/gate';
import { cx } from '@/shared/lib/cx';
import styles from './GateStrip.module.css';

/**
 * The hurdle gates as an ordered list. Each item states its name and status in text; the
 * status word is the signal. The short bar before it repeats the status as a data mark
 * (solid: passed, hatched: failed, dashed outline: skipped), never as a control: no box,
 * no tick, nothing that reads as a checkbox.
 */
export function GateStrip({
  gates,
  animate = false,
}: {
  gates: readonly GateResult[];
  animate?: boolean;
}) {
  const ordered = orderGates(gates);
  return (
    <ol className={cx(styles.strip, animate && styles.animate)} aria-label="Hurdle gates">
      {ordered.map((g, i) => {
        const def = gateDef(g.name);
        return (
          <li
            key={g.name}
            className={styles.gate}
            data-status={g.status}
            style={{ '--i': i } as CSSProperties}
            title={g.reason}
          >
            <span className={styles.name}>{def.label}</span>
            <span className={styles.status}>
              <span className={styles.marker} aria-hidden="true" />
              <span className={styles.word}>{STATUS_WORD[g.status]}</span>
              {g.required ? <span className={styles.required}> (required)</span> : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
