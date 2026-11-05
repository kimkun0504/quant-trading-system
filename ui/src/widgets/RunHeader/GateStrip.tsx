import type { CSSProperties } from 'react';
import type { GateResult } from '@/shared/api';
import { STATUS_WORD, gateDef, orderGates } from '@/entities/gate';
import { cx } from '@/shared/lib/cx';
import styles from './GateStrip.module.css';

/**
 * The hurdle gates as an ordered list. Each item states its name and status in text;
 * the marker's colour and shape repeat that, they never carry it alone.
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
            <Marker status={g.status} />
            <span className={styles.name}>{def.label}</span>
            <span className={styles.status}>
              {STATUS_WORD[g.status]}
              {g.required ? <span className={styles.required}> (required)</span> : null}
            </span>
          </li>
        );
      })}
    </ol>
  );
}

function Marker({ status }: { status: GateResult['status'] }) {
  return (
    <svg className={styles.marker} width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      {status === 'PASS' ? (
        <>
          <rect className={styles.fillPass} x="1" y="1" width="12" height="12" rx="2" />
          <path d="M3.8 7.2 6 9.3 10.3 4.8" className={styles.tick} />
        </>
      ) : status === 'FAIL' ? (
        <>
          <rect className={styles.fillFail} x="1" y="1" width="12" height="12" rx="2" />
          <path d="M4.3 4.3 9.7 9.7M9.7 4.3 4.3 9.7" className={styles.tick} />
        </>
      ) : (
        <rect className={styles.skip} x="1.5" y="1.5" width="11" height="11" rx="2" />
      )}
    </svg>
  );
}
