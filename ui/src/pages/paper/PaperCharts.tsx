import type { PaperStatus } from '@/shared/api';
import { Stat, layout } from '@/features/research-launch';
import { ChartFrame, LegendItem, LineSwatch } from '@/shared/charts';
import { TimeSeriesChart } from '@/shared/charts/TimeSeriesChart';
import type { LineSeries } from '@/shared/charts/TimeSeriesChart';
import { date, num, pct } from '@/shared/lib/format';
import { Panel } from '@/shared/ui';
import { chartSeries, liveSpans, modeCounts } from './lib/series';

const GAIN = 'var(--color-gain)';
const INK = 'var(--color-ink)';
const GRAPHITE = 'var(--color-graphite)';
const CAUTION = 'var(--color-caution)';
const LOSS = 'var(--color-loss)';

function legend(items: readonly { label: string; color: string; dashed?: boolean }[]) {
  return items.map((i) => (
    <LegendItem
      key={i.label}
      swatch={<LineSwatch color={i.color} dashed={i.dashed} />}
      label={i.label}
    />
  ));
}

function table(
  dates: readonly string[],
  series: readonly LineSeries[],
  fmt: (v: number) => string,
  caption: string,
) {
  return () => ({
    caption,
    columns: ['Date', ...series.map((s) => s.label)],
    rows: dates.map((d, i) => [date(d), ...series.map((s) => { const v = s.values[i]; return v === null || v === undefined ? 'n/a' : fmt(v); })]), // prettier-ignore
  });
}

/** The book's charts: cumulative return in its cones, drawdown against the boundaries, exposures, turnover and risk. */
export function PaperCharts({ s }: { s: PaperStatus }) {
  const c = chartSeries(s.series);
  const counts = modeCounts(s.series.mode);
  const mixed = counts.live > 0 && counts.backfill > 0;
  const spans = mixed
    ? liveSpans(s.series.mode).map((x) => ({ ...x, color: 'graphite' as const }))
    : [];
  const live = mixed
    ? ' Shaded: live days.'
    : counts.live
      ? ' Every day is live.'
      : ' Every day so far is a backfill.';
  const logRet = (v: number) => num(v, { digits: 4, signed: true });
  const p1 = (v: number) => pct(v, { digits: 1, signed: true });
  const cone: LineSeries[] = [
    { label: 'Book', values: c.cum, color: 'gain' },
    { label: 'Expected', values: c.coneMid, color: 'graphite', dashed: true },
    { label: 'Analytic 2.5%', values: c.coneLo, color: 'graphite', dashed: true },
    { label: 'Analytic 97.5%', values: c.coneHi, color: 'graphite', dashed: true },
    { label: 'Bootstrap 2.5%', values: c.bootLo, color: 'ink', dashed: true },
    { label: 'Bootstrap 97.5%', values: c.bootHi, color: 'ink', dashed: true },
  ];
  const dd: LineSeries[] = [
    { label: 'Drawdown', values: c.dd, color: 'gain' },
    { label: 'WATCH boundary', values: c.ddWatch, color: 'caution', dashed: true },
    { label: 'STOP boundary', values: c.ddStop, color: 'loss', dashed: true },
  ];
  const expo: LineSeries[] = [
    { label: 'Gross', values: c.gross, color: 'gain' },
    { label: 'Net', values: c.net, color: 'graphite' },
  ];
  const turn: LineSeries[] = [{ label: 'Turnover', values: c.turnover, color: 'gain' }];
  const vol: LineSeries[] = [
    { label: 'Realised (21 d)', values: c.realised, color: 'gain' },
    { label: 'Ex-ante', values: c.exante, color: 'graphite', dashed: true },
  ];
  return (
    <>
      <Panel
        title="Cumulative return in its expectation cones"
        subtitle="log return since the first execution"
      >
        <ChartFrame
          title="Cumulative log return against the analytic and bootstrap cones"
          description={`The book should stay inside the cones 95% of the time if it behaves like the backtest.${live}`}
          legend={legend([
            { label: 'Book', color: GAIN },
            { label: 'Analytic cone (normal)', color: GRAPHITE, dashed: true },
            { label: 'Bootstrap cone (resampled backtest)', color: INK, dashed: true },
          ])}
          table={table(c.dates, cone, logRet, 'Cumulative log return and cones')}
        >
          <TimeSeriesChart
            ariaLabel="Cumulative log return of the book with its analytic and bootstrap 95% cones"
            dates={c.dates}
            series={cone}
            format={logRet}
            refLines={[{ value: 0, color: 'graphite' }]}
            spans={spans}
            bands={[
              { lower: 2, upper: 3, color: 'graphite', opacity: 0.1 },
              { lower: 4, upper: 5, color: 'ink', opacity: 0.06 },
            ]}
            height={240}
          />
        </ChartFrame>
        <p className={layout.lede}>
          <Stat k="cone">Cones</Stat>: the analytic cone assumes normal returns at the
          backtest&apos;s mean and volatility; the bootstrap cone resamples the backtest&apos;s own
          days, fat tails included.
        </p>
      </Panel>
      <div className={layout.row2}>
        <Panel
          title="Drawdown against the boundaries"
          subtitle="calibrated on bootstrap paths of the backtest"
        >
          <ChartFrame
            title="Drawdown with the WATCH and STOP boundaries"
            description={`Crossing a boundary sets the status.${live}`}
            legend={legend([
              { label: 'Drawdown', color: GAIN },
              { label: 'WATCH', color: CAUTION, dashed: true },
              { label: 'STOP', color: LOSS, dashed: true },
            ])}
            table={table(c.dates, dd, p1, 'Drawdown and boundaries')}
          >
            <TimeSeriesChart
              ariaLabel="Drawdown of the book against its WATCH and STOP boundaries"
              dates={c.dates}
              series={dd}
              format={(v) => pct(v, { digits: 2 })}
              spans={spans}
            />
          </ChartFrame>
          <p className={layout.lede}>
            <Stat k="boundary">Boundaries</Stat> widen with time: a longer book has had more chances
            for a deep drawdown.
          </p>
        </Panel>
        <Panel title="Ex-ante and realised volatility" subtitle="annualised">
          <ChartFrame
            title="Realised 21-day volatility against the risk model's ex-ante volatility"
            legend={legend([
              { label: 'Realised (21 d)', color: GAIN },
              { label: 'Ex-ante', color: GRAPHITE, dashed: true },
            ])}
            table={table(c.dates, vol, (v) => pct(v, { digits: 2 }), 'Volatility')}
          >
            <TimeSeriesChart
              ariaLabel="Realised and ex-ante volatility of the book"
              dates={c.dates}
              series={vol}
              format={(v) => pct(v, { digits: 2 })}
              spans={spans}
            />
          </ChartFrame>
        </Panel>
      </div>
      <div className={layout.row2}>
        <Panel title="Gross and net exposure" subtitle="fraction of NAV">
          <ChartFrame
            title="Gross and net exposure"
            legend={legend([
              { label: 'Gross', color: GAIN },
              { label: 'Net', color: GRAPHITE },
            ])}
            table={table(c.dates, expo, (v) => pct(v, { digits: 1 }), 'Exposure')}
          >
            <TimeSeriesChart
              ariaLabel="Gross and net exposure of the book"
              dates={c.dates}
              series={expo}
              format={(v) => pct(v, { digits: 1 })}
              refLines={[{ value: 0, color: 'graphite' }]}
              spans={spans}
            />
          </ChartFrame>
        </Panel>
        <Panel title="Turnover" subtitle="fraction of NAV traded per day (one side)">
          <ChartFrame
            title="Daily turnover"
            table={table(c.dates, turn, (v) => pct(v, { digits: 2 }), 'Turnover')}
          >
            <TimeSeriesChart
              ariaLabel="Daily turnover of the book"
              dates={c.dates}
              series={turn}
              format={(v) => pct(v, { digits: 2 })}
              spans={spans}
            />
          </ChartFrame>
        </Panel>
      </div>
    </>
  );
}
