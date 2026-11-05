import { metric } from '@/entities/metric';
import type { MetricContext, MetricKey } from '@/entities/metric';
import { handbookLabel, handbookUrl } from '@/shared/lib/handbook';
import { Definition, VisuallyHidden } from '@/shared/ui';
import styles from './MetricTable.module.css';

export interface MetricRow {
  key: MetricKey;
  value: number | null | undefined;
}

/**
 * Key metrics with a margin note per row. Rows are subgrids of the page grid, so each
 * note lines up with its metric in the margin column (and drops under it below 1100 px).
 * ARIA table roles keep it a table for assistive technology.
 */
export function MetricTable({
  label,
  rows,
  ctx,
}: {
  label: string;
  rows: readonly MetricRow[];
  ctx: MetricContext;
}) {
  return (
    <div role="table" aria-label={label} className={styles.table}>
      <div role="rowgroup" className={styles.group}>
        <div role="row" className={`${styles.row} ${styles.head}`}>
          <span role="columnheader" className={styles.label}>
            Metric
          </span>
          <span role="columnheader" className={styles.value}>
            Value
          </span>
          <span role="columnheader" className={styles.note}>
            What it means for this run
          </span>
        </div>
      </div>
      <div role="rowgroup" className={styles.group}>
        {rows.map(({ key, value }) => {
          const def = metric(key);
          return (
            <div role="row" key={key} className={styles.row} data-metric={key}>
              <span role="rowheader" className={styles.label}>
                <Definition definition={def.definition} formula={def.formula}>
                  {def.label}
                </Definition>
              </span>
              <span role="cell" className={styles.value}>
                {def.format(value)}
              </span>
              <span role="cell" className={styles.note}>
                {def.interpret(value, ctx)}{' '}
                <a
                  href={handbookUrl(def.handbook)}
                  target="_blank"
                  rel="noreferrer"
                  className={styles.link}
                >
                  Handbook
                  <VisuallyHidden>
                    : {handbookLabel(def.handbook)} (opens in a new tab)
                  </VisuallyHidden>
                </a>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
