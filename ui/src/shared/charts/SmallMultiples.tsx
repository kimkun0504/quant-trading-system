import { extent } from 'd3-array';
import { scaleLinear } from 'd3-scale';
import { line } from 'd3-shape';
import { useMeasure } from '@/shared/lib/useMeasure';
import { num } from '@/shared/lib/format';
import styles from './Svg.module.css';

export interface Multiple {
  key: string;
  label: string;
  values: readonly (number | null)[];
  /** Shaded band, e.g. the optimiser's allowed exposure range. */
  band?: [number, number];
  emphasis?: boolean;
}

/**
 * One small line chart per factor, all on a shared y-scale so tilts compare honestly.
 */
export function SmallMultiples({
  items,
  startLabel,
  endLabel,
  ariaLabel,
  height = 92,
}: {
  items: readonly Multiple[];
  startLabel: string;
  endLabel: string;
  ariaLabel: string;
  height?: number;
}) {
  const [ref, width] = useMeasure();
  const cols = width >= 640 ? 3 : width >= 400 ? 2 : 1;
  const gapX = 20;
  const cellW = cols > 0 ? (width - gapX * (cols - 1)) / cols : 0;
  const all = items.flatMap((i) => i.values.filter((v): v is number => v !== null));
  const [lo0, hi0] = extent(all) as [number | undefined, number | undefined];
  const lo = Math.min(lo0 ?? 0, -0.06);
  const hi = Math.max(hi0 ?? 0, 0.06);
  const m = { top: 22, bottom: 4, left: 34, right: 6 };
  const h = height - m.top - m.bottom;
  const y = scaleLinear().domain([lo, hi]).nice(3).range([h, 0]);

  return (
    <div ref={ref} className={styles.host}>
      <div
        role="img"
        aria-label={ariaLabel}
        className={styles.multiples}
        style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
      >
        {width > 0
          ? items.map((it) => {
              const w = cellW - m.left - m.right;
              const n = it.values.length;
              const x = scaleLinear()
                .domain([0, Math.max(1, n - 1)])
                .range([0, w]);
              const d =
                line<number | null>()
                  .defined((v) => v !== null)
                  .x((_, i) => x(i))
                  .y((v) => y(v ?? 0))([...it.values]) ?? '';
              const last = it.values[n - 1] ?? null;
              return (
                <svg key={it.key} width={cellW} height={height} aria-hidden="true">
                  <text x={0} y={13} className={styles.multipleLabel}>
                    {it.label}
                  </text>
                  <text
                    x={cellW - m.right}
                    y={13}
                    textAnchor="end"
                    className={styles.multipleValue}
                  >
                    {num(last, { digits: 2, signed: true })}
                  </text>
                  <g transform={`translate(${m.left},${m.top})`}>
                    {it.band ? (
                      <rect
                        x={0}
                        width={w}
                        y={y(it.band[1])}
                        height={y(it.band[0]) - y(it.band[1])}
                        className={styles.band}
                      />
                    ) : null}
                    {y.ticks(2).map((tk) => (
                      <g key={tk} transform={`translate(0,${y(tk)})`}>
                        <line x2={w} className={tk === 0 ? styles.zero : styles.grid} />
                        <text x={-6} dy="0.32em" textAnchor="end" className={styles.tickSmall}>
                          {num(tk, { digits: 1 })}
                        </text>
                      </g>
                    ))}
                    <path d={d} className={it.emphasis ? styles.line : styles.lineMuted} />
                  </g>
                </svg>
              );
            })
          : null}
      </div>
      <div className={styles.multiplesAxis} aria-hidden="true">
        <span>{startLabel}</span>
        <span>{endLabel}</span>
      </div>
    </div>
  );
}
