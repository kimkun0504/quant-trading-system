import type { ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import { useDocumentTitle } from '@/shared/lib/useDocumentTitle';
import styles from './Page.module.css';

/** One column of page rows. */
export function Page({
  title,
  children,
  docTitle,
  actions,
}: {
  title?: ReactNode;
  children: ReactNode;
  docTitle: string;
  actions?: ReactNode;
}) {
  useDocumentTitle(docTitle);
  return (
    <div className={styles.page}>
      {title ? (
        <div className={styles.titleRow}>
          <h1 className={styles.h1}>{title}</h1>
          {actions}
        </div>
      ) : null}
      {children}
    </div>
  );
}

/** A grid-transparent group, for components that return several page rows. */
export function Flow({ children }: { children: ReactNode }) {
  return <div className={styles.flow}>{children}</div>;
}

/** Content with its explanation (an About line) under it. */
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
      {note ? <div>{note}</div> : null}
    </div>
  );
}

/** A titled section; `note` (an About line) sits under the heading. */
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
        {note ? <div>{note}</div> : null}
      </div>
      {children}
    </section>
  );
}

export function Wide({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx(styles.wide, className)}>{children}</div>;
}
