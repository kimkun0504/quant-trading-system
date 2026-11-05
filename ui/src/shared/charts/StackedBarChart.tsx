import { useState } from 'react';
import { scaleBand, scaleLinear } from 'd3-scale';
import { useMeasure } from '@/shared/lib/useMeasure';
import styles from './Svg.module.css';
import readout from './Readout.module.css';

export interface StackKey<K extends string> {
  key: K;
  label: string;
  /** CSS colour expression from tokens, e.g. var(--color-gain). */
  color: string;
  opacity?: number;
}

/** Positive stacked bars (costs by component). Thin bars, a 1px surface gap between segments. */
export function StackedBarChart<K extends string>({
  bars,
  keys,
  format,
  height = 200,
  ariaLabel,
  labelFor,
}: {
  bars: readonly { label: string; values: Record<K, number> }[];
  keys: readonly StackKey<K>[];
  format: (v: number) => string;
  height?: number;
  ariaLabel: string;
  labelFor: (label: string) => string;
}) {
  const [ref, width] = useMeasure();
  const [hover, setHover] = useState<number | null>(null);
  const m = { top: 8, right: 8, bottom: 24, left: 52 };
  const w = Math.max(0, width - m.left - m.right);
  const h = height - m.top - m.bottom;
  const totals = bars.map((b) => keys.reduce((a, k) => a + Math.max(0, b.values[k.key]), 0));
  const y = scaleLinear()
    .domain([0, Math.max(...totals, 1e-9)])
    .nice(4)
    .range([h, 0]);
  const x = scaleBand()
    .domain(bars.map((b) => b.label))
    .range([0, w])
    .paddingInner(bars.length > 200 ? 0 : 0.2)
    .paddingOuter(0.05);
  const bw = x.bandwidth();
  const gap = bw > 4 ? 1 : 0;
  const ticks = y.ticks(4);
  const yearTicks = bars
    .map((b, i) => ({ b, i }))
    .filter(({ b, i }) => i === 0 || b.label.slice(0, 4) !== bars[i - 1]?.label.slice(0, 4))
    .filter(({ b }, k, arr) => {
      const next = arr[k + 1];
      // drop a partial-year label that would collide with the next one
      return !next || (x(next.b.label) ?? 0) - (x(b.label) ?? 0) >= 36;
    });
  const active = hover === null ? null : bars[hover];

  return (
    <div className={readout.wrap}>
      <dl className={readout.readout} aria-hidden="true">
        {active ? (
          <>
            <div>
              <dt>Period</dt>
              <dd>{labelFor(active.label)}</dd>
            </div>
            {keys.map((k) => (
              <div key={k.key}>
                <dt>{k.label}</dt>
                <dd>{format(active.values[k.key])}</dd>
              </div>
            ))}
          </>
        ) : (
          <div>
            <dt>Hover a bar for its breakdown</dt>
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
          >
            <g transform={`translate(${m.left},${m.top})`}>
              {ticks.map((tk) => (
                <g key={tk} transform={`translate(0,${y(tk)})`}>
                  <line x2={w} className={tk === 0 ? styles.zero : styles.grid} />
                  <text x={-8} dy="0.32em" textAnchor="end" className={styles.tick}>
                    {format(tk)}
                  </text>
                </g>
              ))}
              {bars.map((b, i) => {
                let acc = 0;
                const x0 = x(b.label) ?? 0;
                return (
                  <g
                    key={b.label}
                    onMouseEnter={() => setHover(i)}
                    onMouseLeave={() => setHover(null)}
                    opacity={hover === null || hover === i ? 1 : 0.55}
                  >
                    <rect x={x0} y={0} width={Math.max(bw, 2)} height={h} className={styles.hit} />
                    {keys.map((k) => {
                      const v = Math.max(0, b.values[k.key]);
                      const y1 = y(acc);
                      acc += v;
                      const y2 = y(acc);
                      const segH = Math.max(0, y1 - y2 - gap);
                      return segH > 0 ? (
                        <rect
                          key={k.key}
                          x={x0}
                          y={y2}
                          width={Math.max(bw, 0.6)}
                          height={segH}
                          fill={k.color}
                          fillOpacity={k.opacity ?? 1}
                        />
                      ) : null;
                    })}
                  </g>
                );
              })}
              {yearTicks.map(({ b }) => (
                <text key={`y-${b.label}`} x={x(b.label) ?? 0} y={h + 16} className={styles.tick}>
                  {b.label.slice(0, 4)}
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
