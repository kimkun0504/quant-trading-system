import type { HTMLAttributes, ReactNode, TdHTMLAttributes, ThHTMLAttributes } from 'react';
import { cx } from '@/shared/lib/cx';
import styles from './Table.module.css';

/**
 * Plain dense table primitives for hand-built tables (the sortable list is
 * `SortableTable`). Numeric cells pass `numeric` to right-align tabular figures.
 */
export function Table({
  caption,
  className,
  children,
  label,
  ...rest
}: HTMLAttributes<HTMLTableElement> & { caption?: ReactNode; label?: string }) {
  return (
    // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
    <div className={styles.scroll} tabIndex={0} role="region" aria-label={label ?? 'Table'}>
      <table className={cx(styles.table, className)} {...rest}>
        {caption ? <caption className={styles.caption}>{caption}</caption> : null}
        {children}
      </table>
    </div>
  );
}

export function Th({
  numeric,
  className,
  ...rest
}: ThHTMLAttributes<HTMLTableCellElement> & { numeric?: boolean }) {
  return <th className={cx(numeric && styles.end, className)} {...rest} />;
}

export function Td({
  numeric,
  className,
  ...rest
}: TdHTMLAttributes<HTMLTableCellElement> & { numeric?: boolean }) {
  return <td className={cx(numeric && styles.end, className)} {...rest} />;
}
