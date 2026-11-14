import { scaleLinear } from 'd3-scale';
import { num } from '@/shared/lib/format';
import { useMeasure } from '@/shared/lib/useMeasure';
import svg from '@/shared/charts/Svg.module.css';
import styles from './SignalTest.module.css';

/**
 * The primary t on a number line with the Harvey–Liu–Zhu hurdle drawn on both sides
 * (|t| ≥ 3) and the conventional 1.96 for reference: a single t is read against the bar a
 * new factor must clear, not against "significant at 5%".
 */
export function TMeter({ t, hurdle }: { t: number | null; hurdle: number }) {
  const [ref, width] = useMeasure();
  const span = Math.max(5, Math.ceil(Math.abs(t ?? 0) + 1));
  const m = { left: 16, right: 16 };
  const w = Math.max(0, width - m.left - m.right);
  const x = scaleLinear().domain([-span, span]).range([0, w]);
  const ticks = x.ticks(Math.min(10, span * 2));
  const label =
    t === null
      ? 'The primary t could not be computed.'
      : `Primary t ${num(t, { signed: true })} against the hurdle of |t| ≥ ${num(hurdle, { digits: 1 })}: ${Math.abs(t) >= hurdle ? 'clears it' : 'below it'}.`;
  return (
    <div ref={ref} className={styles.meter}>
      {width > 0 ? (
        <svg width={width} height={64} role="img" aria-label={label} className={svg.svg}>
          <g transform={`translate(${m.left},8)`}>
            <rect x={0} y={8} width={x(-hurdle)} height={18} className={styles.hurdleZone} />
            <rect
              x={x(hurdle)}
              y={8}
              width={w - x(hurdle)}
              height={18}
              className={styles.hurdleZone}
            />
            <line x1={0} x2={w} y1={17} y2={17} className={svg.grid} />
            {[-1.96, 1.96].map((v) => (
              <line key={v} x1={x(v)} x2={x(v)} y1={10} y2={24} className={styles.weakMark} />
            ))}
            {ticks.map((tk) => (
              <text key={tk} x={x(tk)} y={42} textAnchor="middle" className={svg.tick}>
                {num(tk, { digits: 0, signed: tk !== 0 })}
              </text>
            ))}
            <text x={x(hurdle) + 4} y={6} className={svg.thresholdLabel}>
              hurdle {num(hurdle, { digits: 1 })}
            </text>
            {t !== null ? (
              <g transform={`translate(${x(Math.max(-span, Math.min(span, t)))},17)`}>
                <circle r={7} className={Math.abs(t) >= hurdle ? styles.tPass : styles.tFail} />
                <text y={-12} textAnchor="middle" className={styles.tLabel}>
                  t {num(t, { signed: true })}
                </text>
              </g>
            ) : null}
          </g>
        </svg>
      ) : (
        <div style={{ height: 64 }} />
      )}
      <p className={styles.meterNote}>
        Shaded: |t| ≥ {num(hurdle, { digits: 1 })}, the bar for a new signal. Short marks: ±1.96,
        the bar for one pre-registered test, not for a search.
      </p>
    </div>
  );
}
