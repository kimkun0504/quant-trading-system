import type { ReactNode } from 'react';
import styles from './States.module.css';

export function EmptyState({
  title,
  children,
  action,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className={styles.state}>
      <h2 className={styles.title}>{title}</h2>
      {children ? <div className={styles.body}>{children}</div> : null}
      {action ? <div className={styles.action}>{action}</div> : null}
    </div>
  );
}
