import type { HTMLAttributes, ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import styles from './Panel.module.css';

/**
 * A bordered surface. The border is structure (it groups one thing), not decoration;
 * panels never carry shadows. `flush` removes the body padding for tables and charts.
 */
export function Panel({
  title,
  subtitle,
  actions,
  flush = false,
  className,
  children,
  as: Tag = 'section',
  ...rest
}: HTMLAttributes<HTMLElement> & {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  flush?: boolean;
  children: ReactNode;
  as?: 'section' | 'div' | 'aside';
}) {
  return (
    <Tag className={cx(styles.panel, className)} {...rest}>
      {title || actions ? (
        <div className={styles.head}>
          {title ? (
            <h2 className={styles.title}>
              {title}
              {subtitle ? <span className={styles.subtitle}>{subtitle}</span> : null}
            </h2>
          ) : (
            <span />
          )}
          {actions ? <div className={styles.actions}>{actions}</div> : null}
        </div>
      ) : null}
      <div className={cx(styles.body, flush && styles.flush)}>{children}</div>
    </Tag>
  );
}
