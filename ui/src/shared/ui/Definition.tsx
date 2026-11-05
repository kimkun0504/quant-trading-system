import { useId, useState } from 'react';
import type { KeyboardEvent, ReactNode } from 'react';
import styles from './Definition.module.css';

/**
 * A term that explains itself: hover or focus shows the definition (and formula),
 * Escape dismisses it (WCAG 1.4.13). The definition is also the button's description,
 * so screen readers hear it without opening anything.
 */
export function Definition({
  children,
  definition,
  formula,
}: {
  children: ReactNode;
  definition: string;
  formula?: string;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape') setOpen(false);
  };
  return (
    <span className={styles.wrap}>
      <button
        type="button"
        className={styles.term}
        aria-describedby={id}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={onKey}
      >
        {children}
      </button>
      <span role="tooltip" id={id} className={styles.tip} data-open={open}>
        {definition}
        {formula ? <span className={styles.formula}>{formula}</span> : null}
      </span>
    </span>
  );
}
