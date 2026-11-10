import type { ReactNode } from 'react';
import { Tooltip } from './Tooltip';
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
  return (
    <Tooltip content={definition} formula={formula}>
      <button type="button" className={styles.term}>
        {children}
      </button>
    </Tooltip>
  );
}
