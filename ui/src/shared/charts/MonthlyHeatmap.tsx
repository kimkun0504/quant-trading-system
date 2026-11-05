import { MONTH_LABELS, pct } from '@/shared/lib/format';
import styles from './MonthlyHeatmap.module.css';

export interface HeatCell {
  year: number;
  month: number;
  ret: number;
}

/**
 * Year × month returns as a real table (it is its own text alternative). Cell tint is
 * a diverging gain/loss wash whose strength follows |return|; the signed value is
 * always printed, so colour is secondary. Tint is capped so ink text keeps AA contrast.
 */
export function MonthlyHeatmap({
  years,
  cells,
  annual,
  caption,
}: {
  years: readonly number[];
  cells: readonly HeatCell[];
  annual: ReadonlyMap<number, number>;
  caption: string;
}) {
  const maxAbs = Math.max(1e-9, ...cells.map((c) => Math.abs(c.ret)));
  const lookup = new Map(cells.map((c) => [`${c.year}-${c.month}`, c.ret]));
  const tint = (r: number) => {
    const strength = Math.round(6 + 44 * Math.min(1, Math.abs(r) / maxAbs));
    const token = r >= 0 ? 'var(--color-gain)' : 'var(--color-loss)';
    return `color-mix(in srgb, ${token} ${strength}%, var(--color-sheet))`;
  };
  return (
    <div>
      <p className={styles.caption} aria-hidden="true">
        {caption}
      </p>
      {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex */}
      <div className={styles.scroll} tabIndex={0} role="region" aria-label="Monthly returns">
        <table className={styles.table}>
          <caption className={styles.srOnly}>{caption}</caption>
          <thead>
            <tr>
              <th scope="col">Year</th>
              {MONTH_LABELS.map((m) => (
                <th key={m} scope="col" className={styles.num}>
                  {m}
                </th>
              ))}
              <th scope="col" className={`${styles.num} ${styles.year}`}>
                Full year
              </th>
            </tr>
          </thead>
          <tbody>
            {years.map((y) => (
              <tr key={y}>
                <th scope="row">{y}</th>
                {MONTH_LABELS.map((m, i) => {
                  const r = lookup.get(`${y}-${i}`);
                  return (
                    <td
                      key={m}
                      className={styles.num}
                      style={r === undefined ? undefined : { background: tint(r) }}
                    >
                      {r === undefined ? '' : pct(r, { digits: 1, signed: true })}
                    </td>
                  );
                })}
                <td className={`${styles.num} ${styles.year}`}>
                  {pct(annual.get(y), { digits: 1, signed: true })}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
