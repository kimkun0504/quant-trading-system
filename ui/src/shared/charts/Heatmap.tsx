import { useState } from 'react';
import { scaleBand } from 'd3-scale';
import { useMeasure } from '@/shared/lib/useMeasure';
import styles from './Svg.module.css';
import readout from './Readout.module.css';

export interface HeatCellInfo {
  row: number;
  col: number;
  value: number | null;
}

/**
 * Rows × columns of values as tinted cells. `diverging` centres a gain/loss wash at
 * zero (symmetric about the largest |value| unless `max` is given); `sequential` runs
 * one hue from the surface to full strength. Colour never carries the value alone: the
 * hover readout prints it and the frame's data table lists every cell.
 */
export function Heatmap({
  rows,
  cols,
  values,
  scale = 'diverging',
  max,
  format,
  readoutFor,
  ariaLabel,
  rowLabelWidth = 112,
  cellHeight = 22,
  emphasisRows = [],
  colLabel,
}: {
  rows: readonly string[];
  cols: readonly string[];
  /** `values[row][col]`; null draws an empty cell. */
  values: readonly (readonly (number | null)[])[];
  scale?: 'diverging' | 'sequential';
  /** Value at full tint; defaults to the largest |value|. */
  max?: number;
  format: (v: number) => string;
  /** Extra readout lines for the hovered cell (e.g. the t-stat). */
  readoutFor?: (cell: HeatCellInfo) => readonly { label: string; value: string }[];
  ariaLabel: string;
  rowLabelWidth?: number;
  cellHeight?: number;
  /** Row indices drawn in a stronger label weight. */
  emphasisRows?: readonly number[];
  /** Column header text (null hides it); defaults to the column key, thinned to fit. */
  colLabel?: (col: string, index: number) => string | null;
}) {
  const [ref, width] = useMeasure();
  const [hover, setHover] = useState<HeatCellInfo | null>(null);
  const m = { top: 20, right: 8, bottom: 4, left: rowLabelWidth };
  const w = Math.max(0, width - m.left - m.right);
  const h = rows.length * cellHeight;
  const x = scaleBand<number>()
    .domain(cols.map((_, i) => i))
    .range([0, w])
    .paddingInner(0.08);
  const y = scaleBand<number>()
    .domain(rows.map((_, i) => i))
    .range([0, h])
    .paddingInner(0.12);
  const all = values.flat().filter((v): v is number => v !== null && Number.isFinite(v));
  const top = max ?? Math.max(1e-9, ...all.map((v) => Math.abs(v)));
  const tint = (v: number) => {
    const strength = Math.min(1, Math.abs(v) / top);
    if (scale === 'sequential')
      return { fill: 'var(--color-gain)', opacity: 0.08 + 0.92 * strength };
    return {
      fill: v >= 0 ? 'var(--color-gain)' : 'var(--color-loss)',
      opacity: 0.1 + 0.9 * strength,
    };
  };
  const every = cols.length > 0 && w > 0 ? Math.max(1, Math.ceil((cols.length * 34) / w)) : 1;
  const active = hover ? values[hover.row]?.[hover.col] : undefined;

  return (
    <div className={readout.wrap}>
      <dl className={readout.readout} aria-hidden="true">
        {hover ? (
          <>
            <div>
              <dt>{rows[hover.row]}</dt>
              <dd>{cols[hover.col]}</dd>
            </div>
            <div>
              <dt>Value</dt>
              <dd>{active === null || active === undefined ? 'n/a' : format(active)}</dd>
            </div>
            {readoutFor?.(hover).map((r) => (
              <div key={r.label}>
                <dt>{r.label}</dt>
                <dd>{r.value}</dd>
              </div>
            ))}
          </>
        ) : (
          <div>
            <dt>Hover a cell for its value</dt>
            <dd />
          </div>
        )}
      </dl>
      <div ref={ref} className={styles.host}>
        {width > 0 ? (
          <svg
            width={width}
            height={h + m.top + m.bottom}
            role="img"
            aria-label={ariaLabel}
            className={styles.svg}
            onMouseLeave={() => setHover(null)}
          >
            <g transform={`translate(${m.left},${m.top})`}>
              {cols.map((c, j) =>
                (colLabel ? colLabel(c, j) !== null : j % every === 0) ? (
                  <text
                    key={j}
                    x={(x(j) ?? 0) + x.bandwidth() / 2}
                    y={-7}
                    textAnchor="middle"
                    className={styles.tickSmall}
                  >
                    {colLabel ? colLabel(c, j) : c}
                  </text>
                ) : null,
              )}
              {rows.map((r, i) => (
                <text
                  key={i}
                  x={-8}
                  y={(y(i) ?? 0) + y.bandwidth() / 2}
                  dy="0.32em"
                  textAnchor="end"
                  className={emphasisRows.includes(i) ? styles.heatRowStrong : styles.heatRow}
                >
                  {r}
                </text>
              ))}
              {rows.map((_, i) =>
                cols.map((_, j) => {
                  const v = values[i]?.[j] ?? null;
                  const isHover = hover?.row === i && hover.col === j;
                  const t = v === null ? null : tint(v);
                  return (
                    <rect
                      key={`${i}-${j}`}
                      x={x(j) ?? 0}
                      y={y(i) ?? 0}
                      width={Math.max(1, x.bandwidth())}
                      height={Math.max(1, y.bandwidth())}
                      rx={2}
                      fill={t ? t.fill : 'var(--color-surface-2)'}
                      fillOpacity={t ? t.opacity : 1}
                      stroke={isHover ? 'var(--color-ink)' : 'none'}
                      strokeWidth={isHover ? 1.5 : 0}
                      onMouseEnter={() => setHover({ row: i, col: j, value: v })}
                    />
                  );
                }),
              )}
            </g>
          </svg>
        ) : (
          <div style={{ height: h + m.top + m.bottom }} />
        )}
      </div>
    </div>
  );
}

/** Legend for a heatmap's tint scale, as a small gradient bar with end labels. */
export function HeatLegend({
  scale,
  max,
  format,
}: {
  scale: 'diverging' | 'sequential';
  max: number;
  format: (v: number) => string;
}) {
  const stops =
    scale === 'diverging'
      ? 'linear-gradient(to right, var(--color-loss), var(--color-sheet) 50%, var(--color-gain))'
      : 'linear-gradient(to right, var(--color-sheet), var(--color-gain))';
  return (
    <span className={styles.heatLegend} aria-hidden="true">
      <span>{scale === 'diverging' ? format(-max) : format(0)}</span>
      <span className={styles.heatLegendBar} style={{ background: stops }} />
      <span>{format(max)}</span>
    </span>
  );
}
