import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { cx } from '@/shared/lib/cx';
import { Button } from './Button';
import styles from './Table.module.css';

export interface Column<Row> {
  key: string;
  header: ReactNode;
  /** Plain-text header for the sort button's accessible name. */
  headerText: string;
  render: (row: Row) => ReactNode;
  sortValue?: (row: Row) => number | string | null;
  align?: 'start' | 'end';
  /** Rendered as <th scope="row">. */
  rowHeader?: boolean;
}

type Dir = 'ascending' | 'descending';

export function SortableTable<Row>({
  caption,
  columns,
  rows,
  rowKey,
  initialSort,
  rowClassName,
  limit,
}: {
  caption: ReactNode;
  columns: readonly Column<Row>[];
  rows: readonly Row[];
  rowKey: (row: Row) => string;
  initialSort?: { key: string; dir: Dir };
  rowClassName?: (row: Row) => string | undefined;
  /** Show only the first `limit` rows (after sorting) until expanded. */
  limit?: number;
}) {
  const [sort, setSort] = useState(initialSort ?? null);
  const scrollLabel = typeof caption === 'string' ? caption : 'Table';
  const [expanded, setExpanded] = useState(false);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const col = columns.find((c) => c.key === sort.key);
    if (!col?.sortValue) return rows;
    const get = col.sortValue;
    const sign = sort.dir === 'ascending' ? 1 : -1;
    return [...rows].sort((a, b) => {
      const va = get(a);
      const vb = get(b);
      if (va === null && vb === null) return 0;
      if (va === null) return 1; // missing values last in either direction
      if (vb === null) return -1;
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * sign;
      return String(va).localeCompare(String(vb)) * sign;
    });
  }, [rows, columns, sort]);

  const toggle = (key: string) =>
    setSort((s) =>
      s?.key === key
        ? { key, dir: s.dir === 'ascending' ? 'descending' : 'ascending' }
        : { key, dir: 'descending' },
    );

  return (
    <div>
      {/* Focusable so keyboard users can scroll a wide table (WCAG 2.1.1). */}
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */}
      <div className={styles.scroll} tabIndex={0} role="region" aria-label={scrollLabel}>
        <table className={styles.table}>
          <caption className={styles.caption}>{caption}</caption>
          <thead>
            <tr>
              {columns.map((c) => {
                const active = sort?.key === c.key;
                return (
                  <th
                    key={c.key}
                    scope="col"
                    aria-sort={active ? sort.dir : undefined}
                    className={cx(c.align === 'end' && styles.end)}
                  >
                    {c.sortValue ? (
                      <button type="button" className={styles.sort} onClick={() => toggle(c.key)}>
                        {c.align === 'end' ? (
                          <span aria-hidden="true" className={styles.arrow}>
                            {active ? (sort.dir === 'ascending' ? '▲' : '▼') : ''}
                          </span>
                        ) : null}
                        {c.header}
                        {c.align === 'end' ? null : (
                          <span aria-hidden="true" className={styles.arrow}>
                            {active ? (sort.dir === 'ascending' ? '▲' : '▼') : ''}
                          </span>
                        )}
                      </button>
                    ) : (
                      c.header
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {(limit && !expanded ? sorted.slice(0, limit) : sorted).map((row) => (
              <tr key={rowKey(row)} className={rowClassName?.(row)}>
                {columns.map((c) =>
                  c.rowHeader ? (
                    <th
                      key={c.key}
                      scope="row"
                      className={cx(styles.rowHeader, c.align === 'end' && styles.end)}
                    >
                      {c.render(row)}
                    </th>
                  ) : (
                    <td key={c.key} className={cx(c.align === 'end' && styles.end)}>
                      {c.render(row)}
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {limit && rows.length > limit ? (
        <div className={styles.more}>
          <Button variant="quiet" aria-expanded={expanded} onClick={() => setExpanded((e) => !e)}>
            {expanded ? `Show the first ${limit} rows` : `Show all ${rows.length} rows`}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
