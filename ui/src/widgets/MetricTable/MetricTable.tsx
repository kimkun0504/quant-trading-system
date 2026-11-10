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
 * Key metrics as a dense table: the label explains itself on hover (definition and
 * formula), the value is right-aligned, and the reading column says what the value
 * means for this run with a link to the handbook section.
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
    <table aria-label={label} className={styles.table}>
      <thead>
        <tr>
          <th scope="col" className={styles.label}>
            Metric
          </th>
          <th scope="col" className={styles.value}>
            Value
          </th>
          <th scope="col" className={styles.note}>
            Reading for this run
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ key, value }) => {
          const def = metric(key);
          return (
            <tr key={key} data-metric={key}>
              <th scope="row" className={styles.label}>
                <Definition definition={def.definition} formula={def.formula}>
                  {def.label}
                </Definition>
              </th>
              <td className={styles.value}>{def.format(value)}</td>
              <td className={styles.note}>
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
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
