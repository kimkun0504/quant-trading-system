import { useCallback, useMemo, useState } from 'react';
import type uPlot from 'uplot';
import { date as fmtDate } from '@/shared/lib/format';
import { useChartTokens } from './tokens';
import type { ChartTokens } from './tokens';
import { UplotChart } from './uplot/UplotChart';
import { cursorBase, timeAxis, valueAxis } from './uplot/axes';
import { toUnixAll } from './uplot/time';
import styles from './Readout.module.css';

export type TokenColor = 'gain' | 'loss' | 'caution' | 'graphite' | 'ink';

export interface LineSeries {
  label: string;
  values: readonly (number | null)[];
  color: TokenColor;
  dashed?: boolean;
}

export interface RefLine {
  value: number;
  color: TokenColor;
  dashed?: boolean;
}

/** One or more lines on one shared y-scale (never dual-axis), with a crosshair readout. */
export function TimeSeriesChart({
  dates,
  series,
  format,
  height = 200,
  ariaLabel,
  refLines = [],
}: {
  dates: readonly string[];
  series: readonly LineSeries[];
  format: (v: number) => string;
  height?: number;
  ariaLabel: string;
  refLines?: readonly RefLine[];
}) {
  const t = useChartTokens();
  const [idx, setIdx] = useState<number | null>(null);
  const xs = useMemo(() => toUnixAll(dates), [dates]);
  const data = useMemo<uPlot.AlignedData>(
    () => [xs, ...series.map((s) => [...s.values])] as uPlot.AlignedData,
    [xs, series],
  );
  const colorOf = useCallback((c: TokenColor, tk: ChartTokens) => tk[c], []);

  const makeOptions = useCallback(
    (): uPlot.Options => ({
      width: 0,
      height: 0,
      padding: [8, 12, 0, 4],
      cursor: cursorBase(),
      legend: { show: false },
      scales: { x: { time: true } },
      axes: [timeAxis(t), valueAxis(t, format, 56)],
      series: [
        {},
        ...series.map((s) => ({
          stroke: colorOf(s.color, t),
          width: 2,
          dash: s.dashed ? [5, 4] : undefined,
          points: { show: false },
          spanGaps: false,
        })),
      ],
      hooks: {
        draw: [
          (u: uPlot) => {
            const ctx = u.ctx;
            for (const r of refLines) {
              const y = u.valToPos(r.value, 'y', true);
              if (y < u.bbox.top || y > u.bbox.top + u.bbox.height) continue;
              ctx.save();
              ctx.strokeStyle = colorOf(r.color, t);
              ctx.lineWidth = 1 * devicePixelRatio;
              if (r.dashed) ctx.setLineDash([4 * devicePixelRatio, 4 * devicePixelRatio]);
              ctx.beginPath();
              ctx.moveTo(u.bbox.left, y);
              ctx.lineTo(u.bbox.left + u.bbox.width, y);
              ctx.stroke();
              ctx.restore();
            }
          },
        ],
      },
    }),
    [t, series, format, refLines, colorOf],
  );

  let i = idx ?? dates.length - 1;
  // show the last non-null value by default
  if (idx === null) {
    while (i > 0 && series.every((s) => s.values[i] === null || s.values[i] === undefined)) i--;
  }
  return (
    <div className={styles.wrap}>
      <dl className={styles.readout} aria-hidden="true">
        <div>
          <dt>Date</dt>
          <dd>{fmtDate(dates[i])}</dd>
        </div>
        {series.map((s) => {
          const v = s.values[i];
          return (
            <div key={s.label}>
              <dt>{s.label}</dt>
              <dd>{v === null || v === undefined ? 'n/a' : format(v)}</dd>
            </div>
          );
        })}
      </dl>
      <UplotChart
        data={data}
        makeOptions={makeOptions}
        height={height}
        onCursor={setIdx}
        ariaLabel={ariaLabel}
      />
    </div>
  );
}
