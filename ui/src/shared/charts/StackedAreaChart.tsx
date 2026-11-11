import { useMemo, useState } from 'react';
import { scaleLinear } from 'd3-scale';
import { area, stack, stackOffsetDiverging, stackOffsetNone, stackOrderNone } from 'd3-shape';
import { useMeasure } from '@/shared/lib/useMeasure';
import { thinIndices } from '@/shared/lib/series';
import { date as fmtDate } from '@/shared/lib/format';
import styles from './Svg.module.css';
import readout from './Readout.module.css';

export interface AreaKey<K extends string> {
  key: K;
  label: string;
  /** CSS colour expression from tokens, e.g. var(--color-series-1). */
  color: string;
  opacity?: number;
}

/**
 * Stacked areas over dates (SVG, d3-shape). Positive values stack up from zero and
 * negative values down (`stackOffsetDiverging`), so a long/short book by industry reads
 * as two stacks around the zero line. Long series are thinned to `maxPoints` for drawing;
 * the readout and the data table use the thinned points too.
 */
export function StackedAreaChart<K extends string>({
  dates,
  series,
  keys,
  format,
  height = 220,
  ariaLabel,
  maxPoints = 500,
  diverging = true,
  yDomain,
}: {
  dates: readonly string[];
  /** `series[key][i]` belongs to `dates[i]`; null is 0 for stacking. */
  series: Readonly<Record<K, readonly (number | null)[]>>;
  keys: readonly AreaKey<K>[];
  format: (v: number) => string;
  height?: number;
  ariaLabel: string;
  maxPoints?: number;
  diverging?: boolean;
  /** Fixed y-domain (e.g. [0, 1] for shares). */
  yDomain?: [number, number];
}) {
  const [ref, width] = useMeasure();
  const [hover, setHover] = useState<number | null>(null);
  const m = { top: 8, right: 8, bottom: 24, left: 52 };
  const w = Math.max(0, width - m.left - m.right);
  const h = height - m.top - m.bottom;

  const thin = useMemo(() => thinIndices(dates.length, maxPoints), [dates.length, maxPoints]);
  const rows = useMemo(
    () =>
      thin.map((i) => {
        const r: Record<string, number> = {};
        for (const k of keys) r[k.key] = series[k.key][i] ?? 0;
        return r;
      }),
    [thin, keys, series],
  );
  const stacked = useMemo(
    () =>
      stack<Record<string, number>>()
        .keys(keys.map((k) => k.key))
        .order(stackOrderNone)
        .offset(diverging ? stackOffsetDiverging : stackOffsetNone)(rows),
    [rows, keys, diverging],
  );
  const lo = yDomain ? yDomain[0] : Math.min(0, ...stacked.flatMap((s) => s.map((d) => d[0])));
  const hi = yDomain ? yDomain[1] : Math.max(1e-9, ...stacked.flatMap((s) => s.map((d) => d[1])));
  const y = scaleLinear().domain([lo, hi]).nice(4).range([h, 0]);
  const x = scaleLinear()
    .domain([0, Math.max(1, rows.length - 1)])
    .range([0, w]);
  const path = area()
    .x((_, i) => x(i))
    .y0((d) => y(d[0]))
    .y1((d) => y(d[1]));
  const yearTicks = thin
    .map((di, i) => ({ di, i }))
    .filter(
      ({ di, i }) => i === 0 || dates[di]?.slice(0, 4) !== dates[thin[i - 1] ?? 0]?.slice(0, 4),
    )
    .filter((t, k, arr) => {
      const next = arr[k + 1];
      return !next || x(next.i) - x(t.i) >= 40;
    });
  const hoverRow = hover === null ? null : rows[hover];
  const hoverDate = hover === null ? null : dates[thin[hover] ?? 0];

  return (
    <div className={readout.wrap}>
      <dl className={readout.readout} aria-hidden="true">
        {hoverRow && hoverDate ? (
          <>
            <div>
              <dt>Date</dt>
              <dd>{fmtDate(hoverDate)}</dd>
            </div>
            {keys.map((k) => (
              <div key={k.key}>
                <dt>{k.label}</dt>
                <dd>{format(hoverRow[k.key] ?? 0)}</dd>
              </div>
            ))}
          </>
        ) : (
          <div>
            <dt>Hover the chart for a date&apos;s breakdown</dt>
            <dd />
          </div>
        )}
      </dl>
      <div ref={ref} className={styles.host}>
        {width > 0 ? (
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={ariaLabel}
            className={styles.svg}
            onMouseMove={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const px = e.clientX - rect.left - m.left;
              const i = Math.round(x.invert(Math.max(0, Math.min(w, px))));
              setHover(Math.max(0, Math.min(rows.length - 1, i)));
            }}
            onMouseLeave={() => setHover(null)}
          >
            <g transform={`translate(${m.left},${m.top})`}>
              {y.ticks(4).map((tk) => (
                <g key={tk} transform={`translate(0,${y(tk)})`}>
                  <line x2={w} className={tk === 0 ? styles.zero : styles.grid} />
                  <text x={-8} dy="0.32em" textAnchor="end" className={styles.tick}>
                    {format(tk)}
                  </text>
                </g>
              ))}
              {stacked.map((s, si) => {
                const k = keys[si];
                if (!k) return null;
                const d = path(s.map((p) => [p[0], p[1]] as [number, number])) ?? '';
                return (
                  <path
                    key={k.key}
                    d={d}
                    fill={k.color}
                    fillOpacity={k.opacity ?? 0.85}
                    className={styles.area}
                  />
                );
              })}
              {hover !== null ? (
                <line x1={x(hover)} x2={x(hover)} y1={0} y2={h} className={styles.cursor} />
              ) : null}
              {yearTicks.map(({ di, i }) => (
                <text key={`y-${di}`} x={x(i)} y={h + 16} className={styles.tick}>
                  {dates[di]?.slice(0, 4)}
                </text>
              ))}
            </g>
          </svg>
        ) : (
          <div style={{ height }} />
        )}
      </div>
    </div>
  );
}
