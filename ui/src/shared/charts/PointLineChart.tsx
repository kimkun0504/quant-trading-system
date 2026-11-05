import { scaleLinear, scalePoint } from 'd3-scale';
import { line } from 'd3-shape';
import { useMeasure } from '@/shared/lib/useMeasure';
import styles from './Svg.module.css';

/** A short categorical line with labelled points (e.g. IC decay by horizon). */
export function PointLineChart({
  points,
  format,
  height = 200,
  ariaLabel,
  xLabel,
}: {
  points: readonly { label: string; value: number }[];
  format: (v: number) => string;
  height?: number;
  ariaLabel: string;
  xLabel: string;
}) {
  const [ref, width] = useMeasure();
  const m = { top: 24, right: 32, bottom: 40, left: 56 };
  const w = Math.max(0, width - m.left - m.right);
  const h = height - m.top - m.bottom;
  const vals = points.map((p) => p.value);
  const y = scaleLinear()
    .domain([Math.min(0, ...vals), Math.max(0, ...vals)])
    .nice(4)
    .range([h, 0]);
  const x = scalePoint()
    .domain(points.map((p) => p.label))
    .range([0, w])
    .padding(0.3);
  const path =
    line<{ label: string; value: number }>()
      .x((p) => x(p.label) ?? 0)
      .y((p) => y(p.value))(points as { label: string; value: number }[]) ?? '';

  return (
    <div ref={ref} className={styles.host}>
      {width > 0 ? (
        <svg width={width} height={height} role="img" aria-label={ariaLabel} className={styles.svg}>
          <g transform={`translate(${m.left},${m.top})`}>
            {y.ticks(4).map((tk) => (
              <g key={tk} transform={`translate(0,${y(tk)})`}>
                <line x2={w} className={tk === 0 ? styles.zero : styles.grid} />
                <text x={-8} dy="0.32em" textAnchor="end" className={styles.tick}>
                  {format(tk)}
                </text>
              </g>
            ))}
            <path d={path} className={styles.line} />
            {points.map((p) => (
              <g key={p.label} transform={`translate(${x(p.label) ?? 0},${y(p.value)})`}>
                <circle r={4.5} className={styles.dot} />
                <text y={-10} textAnchor="middle" className={styles.value}>
                  {format(p.value)}
                </text>
              </g>
            ))}
            {points.map((p) => (
              <text
                key={`x-${p.label}`}
                x={x(p.label) ?? 0}
                y={h + 18}
                textAnchor="middle"
                className={styles.tick}
              >
                {p.label}
              </text>
            ))}
            <text x={w / 2} y={h + 34} textAnchor="middle" className={styles.axisLabel}>
              {xLabel}
            </text>
          </g>
        </svg>
      ) : (
        <div style={{ height }} />
      )}
    </div>
  );
}
