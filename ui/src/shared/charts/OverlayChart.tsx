import { useCallback, useMemo, useState } from 'react';
import type uPlot from 'uplot';
import { date as fmtDate } from '@/shared/lib/format';
import { useChartTokens } from './tokens';
import { UplotChart } from './uplot/UplotChart';
import { cursorBase, timeAxis, valueAxis } from './uplot/axes';
import { toUnixAll } from './uplot/time';
import styles from './Readout.module.css';

/** `uPlot.Scale.Distr.Logarithmic`: a const enum in uPlot's typings, so spelled out. */
// eslint-disable-next-line @typescript-eslint/no-unsafe-enum-assignment -- const enum, not importable at runtime
const LOG_DISTR = 3 as uPlot.Scale.Distr;

export interface OverlaySeries {
  label: string;
  values: readonly (number | null)[];
  /** Index into the categorical series palette (0 = series 1). */
  paletteIndex: number;
}

/**
 * Several runs' series overlaid on one y-scale with a crosshair readout, coloured by
 * the categorical palette (same order as the selection tray). `log` switches the
 * y-axis to log10 so equal vertical distances are equal percentage moves.
 */
export function OverlayChart({
  dates,
  series,
  format,
  height = 260,
  ariaLabel,
  log = false,
}: {
  dates: readonly string[];
  series: readonly OverlaySeries[];
  format: (v: number) => string;
  height?: number;
  ariaLabel: string;
  log?: boolean;
}) {
  const t = useChartTokens();
  const [idx, setIdx] = useState<number | null>(null);
  const xs = useMemo(() => toUnixAll(dates), [dates]);
  const data = useMemo<uPlot.AlignedData>(
    () => [xs, ...series.map((s) => [...s.values])] as uPlot.AlignedData,
    [xs, series],
  );
  const colour = useCallback((i: number) => t.series[i % t.series.length] ?? t.gain, [t]);

  const makeOptions = useCallback(
    (): uPlot.Options => ({
      width: 0,
      height: 0,
      padding: [8, 12, 0, 4],
      cursor: cursorBase(),
      legend: { show: false },
      scales: {
        x: { time: true },
        y: log
          ? { distr: LOG_DISTR, log: 10 }
          : { range: (_u, min, max) => [min * 0.995, max * 1.005] },
      },
      axes: [timeAxis(t), valueAxis(t, format, 64)],
      series: [
        {},
        ...series.map((s) => ({
          stroke: colour(s.paletteIndex),
          width: 1.75,
          points: { show: false },
          spanGaps: true,
        })),
      ],
    }),
    [t, series, format, log, colour],
  );

  let i = idx ?? dates.length - 1;
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
              <dt style={{ color: colour(s.paletteIndex) }}>{s.label}</dt>
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
