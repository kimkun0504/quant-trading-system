import type { ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import { handbookLabel, handbookUrl } from '@/shared/lib/handbook';
import type { HandbookRef } from '@/shared/lib/handbook';
import { VisuallyHidden } from './VisuallyHidden';
import styles from './MarginNote.module.css';

/**
 * The run page's explanatory voice: a serif note in the margin column, beside the
 * figure or row it explains. Collapses under the figure below 1100 px.
 */
export function MarginNote({
  children,
  handbook,
  tone = 'plain',
  className,
}: {
  children: ReactNode;
  handbook?: HandbookRef;
  tone?: 'plain' | 'caution';
  className?: string;
}) {
  return (
    <div className={cx(styles.note, tone === 'caution' && styles.caution, className)}>
      <p>{children}</p>
      {handbook ? <HandbookLink refTo={handbook} /> : null}
    </div>
  );
}

export function HandbookLink({ refTo, children }: { refTo: HandbookRef; children?: ReactNode }) {
  return (
    <a className={styles.link} href={handbookUrl(refTo)} target="_blank" rel="noreferrer">
      {children ?? handbookLabel(refTo)}
      <VisuallyHidden> (handbook, opens in a new tab)</VisuallyHidden>
    </a>
  );
}
