import { useState } from 'react';
import { scaleLinear, scalePoint } from 'd3-scale';
import { line } from 'd3-shape';
import { useMeasure } from '@/shared/lib/useMeasure';
import styles from './Svg.module.css';
import readout from './Readout.module.css';

export interface PointSeries {
  label: string;
  /** Index into the categorical series palette (0 = series 1). */
  paletteIndex: number;
  values: readonly (number | null)[];
}

/**
 * Several short categorical lines on one y-scale (e.g. IC decay per run), coloured by
 * the series palette in selection order, with a hover readout per x position.
 */
export function MultiPointLineChart({
  labels,
  series,
  format,
  height = 200,
  ariaLabel,
  xLabel,
}: {
  labels: readonly string[];
  series: readonly PointSeries[];
  format: (v: number) => string;
  height?: number;
  ariaLabel: string;
  xLabel: string;
}) {
  const [ref, width] = useMeasure();
  const [hover, setHover] = useState<number | null>(null);
  const m = { top: 12, right: 24, bottom: 40, left: 56 };
  const w = Math.max(0, width - m.left - m.right);
  const h = height - m.top - m.bottom;
  const vals = series.flatMap((s) => s.values.filter((v): v is number => v !== null));
  const y = scaleLinear()
    .domain([Math.min(0, ...vals), Math.max(1e-9, ...vals)])
    .nice(4)
    .range([h, 0]);
  const x = scalePoint()
    .domain([...labels])
    .range([0, w])
    .padding(0.4);
  const colour = (i: number) => `var(--color-series-${(i % 8) + 1})`;

  return (
    <div className={readout.wrap}>
      <dl className={readout.readout} aria-hidden="true">
        {hover !== null ? (
          <>
            <div>
              <dt>{xLabel}</dt>
              <dd>{labels[hover]}</dd>
            </div>
            {series.map((s) => {
              const v = s.values[hover];
              return (
                <div key={s.label}>
                  <dt style={{ color: colour(s.paletteIndex) }}>{s.label}</dt>
                  <dd>{v === null || v === undefined ? 'n/a' : format(v)}</dd>
                </div>
              );
            })}
          </>
        ) : (
          <div>
            <dt>Hover a horizon for every run&apos;s value</dt>
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
              {labels.map((l, i) => (
                <g key={l}>
                  <rect
                    x={(x(l) ?? 0) - (x.step() / 2 || 10)}
                    y={0}
                    width={x.step() || 20}
                    height={h}
                    className={styles.hit}
                    onMouseEnter={() => setHover(i)}
                  />
                  <text x={x(l) ?? 0} y={h + 16} textAnchor="middle" className={styles.tick}>
                    {l}
                  </text>
                </g>
              ))}
              <text x={w / 2} y={h + 34} textAnchor="middle" className={styles.axisLabel}>
                {xLabel}
              </text>
              {series.map((s) => {
                const d =
                  line<number | null>()
                    .defined((v) => v !== null)
                    .x((_, i) => x(labels[i] ?? '') ?? 0)
                    .y((v) => y(v ?? 0))([...s.values]) ?? '';
                return (
                  <g key={s.label} style={{ pointerEvents: 'none' }}>
                    <path
                      d={d}
                      className={styles.areaLine}
                      stroke={colour(s.paletteIndex)}
                      strokeWidth={2}
                    />
                    {s.values.map((v, i) =>
                      v === null ? null : (
                        <circle
                          key={i}
                          cx={x(labels[i] ?? '') ?? 0}
                          cy={y(v)}
                          r={hover === i ? 5 : 3.5}
                          fill={colour(s.paletteIndex)}
                          stroke="var(--color-sheet)"
                          strokeWidth={1.5}
                        />
                      ),
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
        ) : (
          <div style={{ height }} />
        )}
      </div>
    </div>
  );
}
