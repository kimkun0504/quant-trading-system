import { useState } from 'react';
import { scaleLinear, scaleLog } from 'd3-scale';
import { line } from 'd3-shape';
import { useMeasure } from '@/shared/lib/useMeasure';
import styles from './Svg.module.css';
import readout from './Readout.module.css';

export interface LogSeries {
  label: string;
  /** Index into the categorical series palette (0 = series 1). */
  paletteIndex: number;
  /** One value per x (null: no value there). */
  values: readonly (number | null)[];
  dashed?: boolean;
}

/**
 * A few lines over a numeric x on a log scale (NAV from $5M to $1B), one y-scale, with a
 * hover readout of every series at the nearest x, optional horizontal reference lines and
 * one optional vertical marker (e.g. the NAV where net Sharpe halves).
 */
export function LogLineChart({
  xs,
  series,
  format,
  formatX,
  xLabel,
  ariaLabel,
  height = 220,
  refLines = [],
  marker,
  yDomain,
}: {
  xs: readonly number[];
  series: readonly LogSeries[];
  format: (v: number) => string;
  formatX: (v: number) => string;
  xLabel: string;
  ariaLabel: string;
  height?: number;
  refLines?: readonly { value: number; label: string }[];
  marker?: { x: number; label: string } | null;
  yDomain?: [number, number];
}) {
  const [ref, width] = useMeasure();
  const [hover, setHover] = useState<number | null>(null);
  const m = { top: 12, right: 24, bottom: 40, left: 56 };
  const w = Math.max(0, width - m.left - m.right);
  const h = height - m.top - m.bottom;
  const pos = xs.filter((x) => x > 0);
  const lo = Math.min(...pos, marker?.x ?? Infinity);
  const hi = Math.max(...pos, marker?.x ?? -Infinity);
  const x = scaleLog()
    .domain([lo / 1.4, hi * 1.4])
    .range([0, w]);
  const vals = [
    ...series.flatMap((s) => s.values.filter((v): v is number => v !== null)),
    ...refLines.map((r) => r.value),
  ];
  const y = scaleLinear()
    .domain(yDomain ?? [Math.min(0, ...vals), Math.max(1e-9, ...vals)])
    .nice(4)
    .range([h, 0]);
  const colour = (i: number) => `var(--color-series-${(i % 8) + 1})`;
  const cell = (i: number) => {
    const xi = xs[i] ?? lo;
    const prev = xs[i - 1];
    const next = xs[i + 1];
    const a = prev === undefined ? 0 : (x(prev) + x(xi)) / 2;
    const b = next === undefined ? w : (x(next) + x(xi)) / 2;
    return [a, b] as const;
  };

  return (
    <div className={readout.wrap}>
      <dl className={readout.readout} aria-hidden="true">
        {hover !== null ? (
          <>
            <div>
              <dt>{xLabel}</dt>
              <dd>{formatX(xs[hover] ?? 0)}</dd>
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
            <dt>Hover a point for every series&apos; value</dt>
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
              {xs.map((xv, i) => {
                const [a, b] = cell(i);
                return (
                  <g key={xv}>
                    <rect
                      x={a}
                      y={0}
                      width={Math.max(0, b - a)}
                      height={h}
                      className={styles.hit}
                      onMouseEnter={() => setHover(i)}
                    />
                    <text x={x(xv)} y={h + 16} textAnchor="middle" className={styles.tick}>
                      {formatX(xv)}
                    </text>
                  </g>
                );
              })}
              <text x={w / 2} y={h + 34} textAnchor="middle" className={styles.axisLabel}>
                {xLabel} (log scale)
              </text>
              {refLines.map((r) => (
                <g key={r.label} style={{ pointerEvents: 'none' }}>
                  <line
                    x1={0}
                    x2={w}
                    y1={y(r.value)}
                    y2={y(r.value)}
                    className={styles.threshold}
                  />
                  <text x={w} y={y(r.value) - 4} textAnchor="end" className={styles.thresholdLabel}>
                    {r.label}
                  </text>
                </g>
              ))}
              {marker ? (
                <g style={{ pointerEvents: 'none' }}>
                  <line
                    x1={x(marker.x)}
                    x2={x(marker.x)}
                    y1={0}
                    y2={h}
                    className={styles.threshold}
                  />
                  <text x={x(marker.x) + 4} y={10} className={styles.thresholdLabel}>
                    {marker.label}
                  </text>
                </g>
              ) : null}
              {hover !== null ? (
                <line
                  x1={x(xs[hover] ?? lo)}
                  x2={x(xs[hover] ?? lo)}
                  y1={0}
                  y2={h}
                  className={styles.cursor}
                  style={{ pointerEvents: 'none' }}
                />
              ) : null}
              {series.map((s) => {
                const d =
                  line<number | null>()
                    .defined((v) => v !== null)
                    .x((_, i) => x(xs[i] ?? lo))
                    .y((v) => y(v ?? 0))([...s.values]) ?? '';
                return (
                  <g key={s.label} style={{ pointerEvents: 'none' }}>
                    <path
                      d={d}
                      className={styles.areaLine}
                      stroke={colour(s.paletteIndex)}
                      strokeWidth={2}
                      strokeDasharray={s.dashed ? '5 4' : undefined}
                    />
                    {s.values.map((v, i) =>
                      v === null ? null : (
                        <circle
                          key={i}
                          cx={x(xs[i] ?? lo)}
                          cy={y(v)}
                          r={hover === i ? 5 : 4}
                          fill={colour(s.paletteIndex)}
                          stroke="var(--color-surface)"
                          strokeWidth={2}
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
