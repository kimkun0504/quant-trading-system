import { useCallback, useMemo, useState } from 'react';
import type uPlot from 'uplot';
import { currency, date, pct } from '@/shared/lib/format';
import { useChartTokens, withAlpha } from './tokens';
import { UplotChart } from './uplot/UplotChart';
import { cursorBase, timeAxis, valueAxis } from './uplot/axes';
import { toUnixAll } from './uplot/time';
import styles from './Readout.module.css';

export interface NavDrawdownChartProps {
  dates: readonly string[];
  nav: readonly number[];
  drawdown: readonly number[];
  syncKey: string;
  animate?: boolean;
}

/**
 * NAV (net, compounded) above its drawdown: two panes sharing the time axis and a
 * synced crosshair, never two y-scales on one plot. Drawdown is a caution-tinted area.
 */
export function NavDrawdownChart({
  dates,
  nav,
  drawdown,
  syncKey,
  animate = false,
}: NavDrawdownChartProps) {
  const t = useChartTokens();
  const [idx, setIdx] = useState<number | null>(null);
  const xs = useMemo(() => toUnixAll(dates), [dates]);
  const navData = useMemo<uPlot.AlignedData>(() => [xs, [...nav]], [xs, nav]);
  const ddData = useMemo<uPlot.AlignedData>(() => [xs, [...drawdown]], [xs, drawdown]);
  const nav0 = nav[0] ?? 1;
  const ddMin = Math.min(...drawdown);

  const navOpts = useCallback(
    (): uPlot.Options => ({
      width: 0,
      height: 0,
      padding: [8, 12, 0, 4],
      cursor: cursorBase(syncKey),
      legend: { show: false },
      scales: {
        x: { time: true, range: [xs[0] ?? 0, xs[xs.length - 1] ?? 1] },
        y: { range: (_u, min, max) => [min * 0.995, max * 1.005] },
      },
      axes: [{ ...timeAxis(t), show: false }, valueAxis(t, (v) => currency(v), 64)],
      series: [{}, { stroke: t.gain, width: 2, points: { show: false } }],
    }),
    [t, xs, syncKey],
  );

  const ddOpts = useCallback(
    (): uPlot.Options => ({
      width: 0,
      height: 0,
      padding: [12, 12, 0, 4],
      cursor: cursorBase(syncKey),
      legend: { show: false },
      scales: { x: { time: true }, y: { range: [Math.min(ddMin * 1.15, -0.01), 0] } },
      axes: [timeAxis(t), valueAxis(t, (v) => pct(v, { digits: 0 }), 64)],
      series: [
        {},
        {
          stroke: t.caution,
          width: 1,
          fill: withAlpha(t.caution, t.drawdownFillOpacity),
          points: { show: false },
        },
      ],
    }),
    [t, ddMin, syncKey],
  );

  const i = idx ?? dates.length - 1;
  const navI = nav[i];
  return (
    <div className={styles.wrap}>
      <dl className={styles.readout} aria-hidden="true">
        <div>
          <dt>Date</dt>
          <dd>{date(dates[i])}</dd>
        </div>
        <div>
          <dt>NAV</dt>
          <dd>{currency(navI)}</dd>
        </div>
        <div>
          <dt>Since start</dt>
          <dd>{navI === undefined ? '' : pct(navI / nav0 - 1, { digits: 1, signed: true })}</dd>
        </div>
        <div>
          <dt>Drawdown</dt>
          <dd>{pct(drawdown[i], { digits: 1, signed: true })}</dd>
        </div>
      </dl>
      <UplotChart
        data={navData}
        makeOptions={navOpts}
        height={220}
        animate={animate}
        onCursor={setIdx}
        ariaLabel={`Net asset value from ${date(dates[0])} to ${date(dates[dates.length - 1])}, ${currency(nav0)} to ${currency(nav[nav.length - 1])}.`}
      />
      <UplotChart
        data={ddData}
        makeOptions={ddOpts}
        height={110}
        onCursor={setIdx}
        ariaLabel={`Drawdown from the running peak; worst ${pct(ddMin, { digits: 1, signed: true })}.`}
      />
    </div>
  );
}
