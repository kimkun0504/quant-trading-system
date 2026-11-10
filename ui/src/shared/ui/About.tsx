import type { ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import { handbookLabel, handbookUrl } from '@/shared/lib/handbook';
import type { HandbookRef } from '@/shared/lib/handbook';
import { VisuallyHidden } from './VisuallyHidden';
import styles from './About.module.css';

/**
 * The explanatory voice, folded away: a one-line "About …" disclosure under a chart or
 * table that opens into the handbook's explanation and a link to the section. Replaces
 * the first design's margin notes.
 */
export function About({
  children,
  handbook,
  label = 'About this',
  tone = 'plain',
  className,
  defaultOpen = false,
}: {
  children: ReactNode;
  handbook?: HandbookRef;
  label?: string;
  tone?: 'plain' | 'caution';
  className?: string;
  defaultOpen?: boolean;
}) {
  return (
    <details
      className={cx(styles.about, tone === 'caution' && styles.caution, className)}
      open={defaultOpen || undefined}
    >
      <summary className={styles.summary}>
        <svg viewBox="0 0 12 12" aria-hidden="true" className={styles.chevron}>
          <path d="M4 2l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
        {label}
      </summary>
      <div className={styles.body}>
        <div>{children}</div>
        {handbook ? <HandbookLink refTo={handbook} /> : null}
      </div>
    </details>
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
