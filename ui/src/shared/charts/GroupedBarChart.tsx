import { useState } from 'react';
import { scaleBand, scaleLinear } from 'd3-scale';
import { useMeasure } from '@/shared/lib/useMeasure';
import styles from './Svg.module.css';
import readout from './Readout.module.css';

export interface BarGroup {
  label: string;
  /** One value per series (null when the series has no value in this group). */
  values: readonly (number | null)[];
}

export interface GroupSeries {
  label: string;
  paletteIndex: number;
}

/**
 * Paired bars around a zero baseline: one group per category (a fold), one bar per
 * series (a run), coloured by the categorical palette. Signed values keep their sign
 * in the readout; colour marks the run, not the sign.
 */
export function GroupedBarChart({
  groups,
  series,
  format,
  height = 200,
  ariaLabel,
  threshold,
  tickFormat = (l: string) => l,
}: {
  groups: readonly BarGroup[];
  series: readonly GroupSeries[];
  format: (v: number) => string;
  height?: number;
  ariaLabel: string;
  threshold?: { value: number; label: string };
  tickFormat?: (label: string) => string;
}) {
  const [ref, width] = useMeasure();
  const [hover, setHover] = useState<[number, number] | null>(null);
  const m = { top: 8, right: threshold ? 72 : 8, bottom: 24, left: 52 };
  const w = Math.max(0, width - m.left - m.right);
  const h = height - m.top - m.bottom;
  const vals = groups.flatMap((g) => g.values.filter((v): v is number => v !== null));
  const lo = Math.min(0, ...vals, threshold?.value ?? 0);
  const hi = Math.max(0, ...vals, threshold?.value ?? 0);
  const y = scaleLinear().domain([lo, hi]).nice(4).range([h, 0]);
  const x0 = scaleBand()
    .domain(groups.map((g) => g.label))
    .range([0, w])
    .paddingInner(0.25)
    .paddingOuter(0.1);
  const x1 = scaleBand()
    .domain(series.map((_, i) => String(i)))
    .range([0, x0.bandwidth()])
    .paddingInner(0.12);
  const ticks = y.ticks(4);
  const every = Math.ceil(groups.length / Math.max(1, Math.floor(w / 64)));
  const active = hover ? groups[hover[0]] : null;
  const activeSeries = hover ? series[hover[1]] : null;
  const activeValue = hover ? (active?.values[hover[1]] ?? null) : null;

  return (
    <div className={readout.wrap}>
      <dl className={readout.readout} aria-hidden="true">
        {active && activeSeries ? (
          <>
            <div>
              <dt>Fold</dt>
              <dd>{tickFormat(active.label)}</dd>
            </div>
            <div>
              <dt style={{ color: `var(--color-series-${activeSeries.paletteIndex + 1})` }}>
                {activeSeries.label}
              </dt>
              <dd>{activeValue === null ? 'n/a' : format(activeValue)}</dd>
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
              {groups.map((g, gi) => {
                const gx = x0(g.label) ?? 0;
                return (
                  <g key={g.label} transform={`translate(${gx},0)`}>
                    {g.values.map((v, si) => {
                      if (v === null) return null;
                      const bx = x1(String(si)) ?? 0;
                      const top = y(Math.max(0, v));
                      const bh = Math.abs(y(v) - y(0));
                      const isActive = hover?.[0] === gi && hover[1] === si;
                      return (
                        <rect
                          key={si}
                          x={bx}
                          y={top}
                          width={x1.bandwidth()}
                          height={Math.max(1, bh)}
                          rx={Math.min(2, x1.bandwidth() / 4)}
                          fill={`var(--color-series-${(series[si]?.paletteIndex ?? si) + 1})`}
                          className={styles.seriesBar}
                          data-active={isActive}
                          onMouseEnter={() => setHover([gi, si])}
                          onMouseLeave={() => setHover(null)}
                        />
                      );
                    })}
                  </g>
                );
              })}
              {groups.map((g, i) =>
                i % every === 0 ? (
                  <text
                    key={`l-${g.label}`}
                    x={(x0(g.label) ?? 0) + x0.bandwidth() / 2}
                    y={h + 16}
                    textAnchor="middle"
                    className={styles.tick}
                  >
                    {tickFormat(g.label)}
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
