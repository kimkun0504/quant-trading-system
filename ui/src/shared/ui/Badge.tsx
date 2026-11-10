import type { HTMLAttributes, ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import styles from './Badge.module.css';

export type BadgeTone = 'neutral' | 'pass' | 'fail' | 'skipped' | 'caution' | 'accent';

/**
 * A small status chip. The word is the signal; the dot repeats it as a mark. Gate
 * statuses map to pass / fail / skipped.
 */
export function Badge({
  tone = 'neutral',
  dot = true,
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone; dot?: boolean; children: ReactNode }) {
  return (
    <span className={cx(styles.badge, styles[tone], className)} {...rest}>
      {dot ? <span className={styles.dot} aria-hidden="true" /> : null}
      {children}
    </span>
  );
}
