import type { ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import { useDocumentTitle } from '@/shared/lib/useDocumentTitle';
import styles from './Page.module.css';

/**
 * The reading column + margin column grid. Direct children sit in the reading column;
 * <Annotated> and `wide` children span both.
 */
export function Page({
  title,
  children,
  docTitle,
}: {
  title?: ReactNode;
  children: ReactNode;
  docTitle: string;
}) {
  useDocumentTitle(docTitle);
  return (
    <div className={styles.page}>
      {title ? <h1 className={styles.h1}>{title}</h1> : null}
      {children}
    </div>
  );
}

/** A grid-transparent group, for components that return several page rows. */
export function Flow({ children }: { children: ReactNode }) {
  return <div className={styles.flow}>{children}</div>;
}

/** Content in the reading column with a note in the margin column beside it. */
export function Annotated({
  children,
  note,
  className,
}: {
  children: ReactNode;
  note?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cx(styles.annotated, className)}>
      <div className={styles.body}>{children}</div>
      {note ? <div className={styles.margin}>{note}</div> : null}
    </div>
  );
}

/** A titled section of the reading column; `note` goes to the margin beside the heading. */
export function Section({
  id,
  title,
  lede,
  note,
  children,
}: {
  id: string;
  title: string;
  lede?: ReactNode;
  note?: ReactNode;
  children: ReactNode;
}) {
  const headingId = `${id}-title`;
  return (
    <section id={id} aria-labelledby={headingId} className={styles.section}>
      <div className={styles.sectionHead}>
        <h2 id={headingId} className={styles.h2}>
          {title}
        </h2>
        {lede ? <div className={styles.lede}>{lede}</div> : null}
      </div>
      {note ? <div className={styles.sectionNote}>{note}</div> : null}
      {children}
    </section>
  );
}

export function Wide({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx(styles.wide, className)}>{children}</div>;
}
