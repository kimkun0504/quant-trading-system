import { useState } from 'react';
import { scaleBand, scaleLinear } from 'd3-scale';
import { useMeasure } from '@/shared/lib/useMeasure';
import styles from './Svg.module.css';
import readout from './Readout.module.css';

export interface Bar {
  label: string;
  value: number;
}

/**
 * Signed vertical bars around a zero baseline. Positive bars use the gain token,
 * negative the loss token; the sign is also in the value text, never colour alone.
 */
export function BarChart({
  bars,
  format,
  height = 180,
  ariaLabel,
  threshold,
  valueLabel = 'Value',
  showValues = false,
  tickFormat = (l: string) => l,
}: {
  bars: readonly Bar[];
  format: (v: number) => string;
  height?: number;
  ariaLabel: string;
  threshold?: { value: number; label: string };
  valueLabel?: string;
  showValues?: boolean;
  tickFormat?: (label: string) => string;
}) {
  const [ref, width] = useMeasure();
  const [hover, setHover] = useState<number | null>(null);
  const m = { top: showValues ? 18 : 8, right: threshold ? 64 : 8, bottom: 24, left: 52 };
  const w = Math.max(0, width - m.left - m.right);
  const h = height - m.top - m.bottom;
  const vals = bars.map((b) => b.value);
  const lo = Math.min(0, ...vals, threshold?.value ?? 0);
  const hi = Math.max(0, ...vals, threshold?.value ?? 0);
  const y = scaleLinear().domain([lo, hi]).nice(4).range([h, 0]);
  const x = scaleBand()
    .domain(bars.map((b) => b.label))
    .range([0, w])
    .paddingInner(bars.length > 30 ? 0.15 : 0.3)
    .paddingOuter(0.1);
  const ticks = y.ticks(4);
  const every = Math.ceil(bars.length / Math.max(1, Math.floor(w / 56)));
  const active = hover === null ? null : bars[hover];

  return (
    <div className={readout.wrap}>
      <dl className={readout.readout} aria-hidden="true">
        {active ? (
          <>
            <div>
              <dt>Period</dt>
              <dd>{tickFormat(active.label)}</dd>
            </div>
            <div>
              <dt>{valueLabel}</dt>
              <dd>{format(active.value)}</dd>
            </div>
          </>
        ) : (
          <div>
            <dt>Hover a bar for its value</dt>
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
              {threshold ? (
                <g transform={`translate(0,${y(threshold.value)})`}>
                  <line x2={w} className={styles.threshold} />
                  <text x={w + 6} dy="0.32em" className={styles.thresholdLabel}>
                    {threshold.label}
                  </text>
                </g>
              ) : null}
              {bars.map((b, i) => {
                const x0 = x(b.label) ?? 0;
                const y0 = y(Math.max(0, b.value));
                const bh = Math.abs(y(b.value) - y(0));
                return (
                  <g
                    key={b.label}
                    onMouseEnter={() => setHover(i)}
                    onMouseLeave={() => setHover(null)}
                  >
                    <rect
                      x={x0 - 2}
                      y={0}
                      width={x.bandwidth() + 4}
                      height={h}
                      className={styles.hit}
                    />
                    <rect
                      x={x0}
                      y={y0}
                      width={x.bandwidth()}
                      height={Math.max(1, bh)}
                      rx={Math.min(2, x.bandwidth() / 4)}
                      className={b.value >= 0 ? styles.pos : styles.neg}
                      data-active={hover === i}
                    />
                    {showValues ? (
                      <text
                        x={x0 + x.bandwidth() / 2}
                        y={b.value >= 0 ? y0 - 4 : y0 + bh + 12}
                        textAnchor="middle"
                        className={styles.value}
                      >
                        {format(b.value)}
                      </text>
                    ) : null}
                  </g>
                );
              })}
              {bars.map((b, i) =>
                i % every === 0 ? (
                  <text
                    key={`l-${b.label}`}
                    x={(x(b.label) ?? 0) + x.bandwidth() / 2}
                    y={h + 16}
                    textAnchor="middle"
                    className={styles.tick}
                  >
                    {tickFormat(b.label)}
                  </text>
                ) : null,
              )}
            </g>
          </svg>
        ) : (
          <div style={{ height }} />
        )}
      </div>
    </div>
  );
}
