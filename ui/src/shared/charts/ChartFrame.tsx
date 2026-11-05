import { useId, useState } from 'react';
import type { ReactNode } from 'react';
import { Button } from '@/shared/ui/Button';
import styles from './ChartFrame.module.css';

export interface DataTableSpec {
  caption: string;
  columns: readonly string[];
  rows: readonly (readonly (string | number)[])[];
}

/**
 * Wraps a chart in a <figure> with a caption, a legend slot and a "Show data table"
 * toggle: the text alternative for every chart.
 */
export function ChartFrame({
  title,
  description,
  legend,
  children,
  table,
  controls,
}: {
  title: string;
  description?: ReactNode;
  legend?: ReactNode;
  children: ReactNode;
  table: DataTableSpec | (() => DataTableSpec);
  controls?: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  const tableId = useId();
  const spec = showTable ? (typeof table === 'function' ? table() : table) : null;
  return (
    <figure className={styles.figure}>
      <figcaption className={styles.caption}>
        <span className={styles.title}>{title}</span>
        {description ? <span className={styles.description}>{description}</span> : null}
      </figcaption>
      {controls || legend ? (
        <div className={styles.toolbar}>
          {legend ? <div className={styles.legend}>{legend}</div> : <span />}
          {controls}
        </div>
      ) : null}
      <div className={styles.plot}>{children}</div>
      <div className={styles.footer}>
        <Button
          variant="quiet"
          aria-expanded={showTable}
          aria-controls={tableId}
          onClick={() => setShowTable((s) => !s)}
        >
          {showTable ? 'Hide data table' : 'Show data table'}
        </Button>
      </div>
      <div id={tableId} hidden={!showTable}>
        {spec ? (
          // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex
          <div className={styles.tableScroll} tabIndex={0} role="region" aria-label={spec.caption}>
            <table className={styles.table}>
              <caption className={styles.tableCaption}>{spec.caption}</caption>
              <thead>
                <tr>
                  {spec.columns.map((c, i) => (
                    <th key={c} scope="col" className={i > 0 ? styles.num : undefined}>
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {spec.rows.map((r, i) => (
                  <tr key={i}>
                    {r.map((v, j) =>
                      j === 0 ? (
                        <th key={j} scope="row">
                          {v}
                        </th>
                      ) : (
                        <td key={j} className={styles.num}>
                          {v}
                        </td>
                      ),
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </figure>
  );
}

export function LegendItem({ swatch, label }: { swatch: ReactNode; label: string }) {
  return (
    <span className={styles.legendItem}>
      {swatch}
      {label}
    </span>
  );
}

export function LineSwatch({ color, dashed = false }: { color: string; dashed?: boolean }) {
  return (
    <svg width="18" height="10" aria-hidden="true" className={styles.swatch}>
      <line
        x1="0"
        x2="18"
        y1="5"
        y2="5"
        stroke={color}
        strokeWidth="2"
        strokeDasharray={dashed ? '4 3' : undefined}
      />
    </svg>
  );
}

export function BoxSwatch({ color, opacity = 1 }: { color: string; opacity?: number }) {
  return (
    <svg width="12" height="12" aria-hidden="true" className={styles.swatch}>
      <rect width="12" height="12" rx="2" fill={color} fillOpacity={opacity} />
    </svg>
  );
}
