import type { ReactNode } from 'react';
import styles from './VisuallyHidden.module.css';

export function VisuallyHidden({ children }: { children: ReactNode }) {
  return <span className={styles.vh}>{children}</span>;
}
