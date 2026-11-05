import { useMemo } from 'react';
import type { BacktestSummary } from '@/shared/api';
import { metricContext, monthlyReturns, rollingSharpe, useRunSeries } from '@/entities/run';
import { metric, phaseSharpeBar } from '@/entities/metric';
import type { MetricKey } from '@/entities/metric';
import { ChartFrame, LegendItem, LineSwatch, MonthlyHeatmap } from '@/shared/charts';
import { NavDrawdownChart } from '@/shared/charts/NavDrawdownChart';
import { TimeSeriesChart } from '@/shared/charts/TimeSeriesChart';
import { currency, date, MONTH_LABELS, num, pct, sharpe } from '@/shared/lib/format';
import { ErrorState, Flow, MarginNote, Section, Skeleton } from '@/shared/ui';
import { MetricTable } from '@/widgets/MetricTable';
import { useRunOutlet } from '../context';

const SERIES = ['nav', 'drawdown', 'net'] as const;

const KEY_METRICS: readonly (MetricKey & keyof BacktestSummary)[] = [
  'sharpe_net',
  'sharpe_gross',
  'newey_west_t_net',
  'cagr_net',
  'ann_vol_net',
  'max_drawdown_net',
  'calmar_net',
  'sortino_net',
  'hit_rate_net',
  'avg_daily_turnover',
  'effective_holding_days',
  'avg_cost_bps_per_side',
  'breakeven_cost_bps',
  'transfer_coefficient_mean',
  'realized_vs_exante_vol_ratio',
];

export function PerformanceTab() {
  const { run, animate } = useRunOutlet();
  const series = useRunSeries(run.id, SERIES);
  const ctx = metricContext(run);
  const s = run.summary;

  const derived = useMemo(() => {
    if (!series.data) return null;
    const { dates } = series.data;
    const net = series.data.series.net;
    const monthly = monthlyReturns(dates, net);
    const rolling = rollingSharpe(net, 126);
    const valid = rolling.filter((v): v is number => v !== null);
    const bar = phaseSharpeBar(run.config.phase);
    const above = valid.filter((v) => v >= bar).length / Math.max(1, valid.length);
    const sorted = [...monthly.cells].sort((a, b) => a.ret - b.ret);
    const positive =
      monthly.cells.filter((c) => c.ret > 0).length / Math.max(1, monthly.cells.length);
    return {
      dates,
      nav: series.data.series.nav.map((v) => v ?? Number.NaN),
      drawdown: series.data.series.drawdown.map((v) => v ?? 0),
      monthly,
      rolling,
      above,
      minRolling: Math.min(...valid),
      maxRolling: Math.max(...valid),
      worst: sorted[0],
      best: sorted[sorted.length - 1],
      positive,
      bar,
    };
  }, [series.data, run.config.phase]);

  const keyRows = KEY_METRICS.map((k) => ({ key: k, value: s[k] }));

  return (
    <Flow>
      <Section
        id="nav"
        title="Net asset value and drawdown"
        note={
          <MarginNote handbook={metric('max_drawdown_net').handbook}>
            {metric('max_drawdown_net').interpret(s.max_drawdown_net, ctx)} The NAV ends at{' '}
            {currency(s.final_nav)} from {currency(5_000_000)}.
          </MarginNote>
        }
      >
        {derived ? (
          <ChartFrame
            title="Net NAV, compounded daily, with drawdown from the running peak"
            description={`${date(s.start)} to ${date(s.end)}. Costs and borrow are deducted every day.`}
            legend={
              <>
                <LegendItem swatch={<LineSwatch color="var(--color-gain)" />} label="Net NAV" />
                <LegendItem swatch={<LineSwatch color="var(--color-caution)" />} label="Drawdown" />
              </>
            }
            table={() => ({
              caption: 'Month-end NAV and drawdown',
              columns: ['Date', 'NAV', 'Drawdown'],
              rows: derived.dates
                .map((d, i) => ({ d, i }))
                .filter(({ d, i }) => derived.dates[i + 1]?.slice(0, 7) !== d.slice(0, 7))
                .map(({ d, i }) => [
                  date(d),
                  currency(derived.nav[i], { compact: false }),
                  pct(derived.drawdown[i], { signed: true }),
                ]),
            })}
          >
            <NavDrawdownChart
              dates={derived.dates}
              nav={derived.nav}
              drawdown={derived.drawdown}
              syncKey={`navdd-${run.id}`}
              animate={animate}
            />
          </ChartFrame>
        ) : series.isError ? (
          <ErrorState error={series.error} onRetry={() => void series.refetch()} />
        ) : (
          <Skeleton height="26rem" />
        )}
      </Section>

      <Section
        id="metrics"
        title="Key metrics"
        lede={<p>Hover or focus a metric name for its definition.</p>}
      >
        <MetricTable label="Key metrics" rows={keyRows} ctx={ctx} />
      </Section>

      <Section
        id="monthly"
        title="Monthly returns"
        note={
          derived?.best && derived.worst ? (
            <MarginNote>
              {pct(derived.positive, { digits: 0 })} of months were positive. Best{' '}
              {MONTH_LABELS[derived.best.month]} {derived.best.year} at{' '}
              {pct(derived.best.ret, { signed: true })}; worst {MONTH_LABELS[derived.worst.month]}{' '}
              {derived.worst.year} at {pct(derived.worst.ret, { signed: true })}. Look for clusters:
              a run of bad months in one year says more than any single month.
            </MarginNote>
          ) : null
        }
      >
        {derived ? (
          <MonthlyHeatmap
            years={derived.monthly.years}
            cells={derived.monthly.cells}
            annual={derived.monthly.annual}
            caption="Net return by calendar month, compounded from daily returns. Partial first and last years."
          />
        ) : (
          <Skeleton height="12rem" />
        )}
      </Section>

      <Section
        id="rolling"
        title="Rolling Sharpe, 126 days"
        note={
          derived ? (
            <MarginNote handbook={metric('sharpe_net').handbook}>
              The half-year net Sharpe ranged from {sharpe(derived.minRolling, { signed: true })} to{' '}
              {sharpe(derived.maxRolling, { signed: true })} and was above the phase-
              {run.config.phase} bar ({num(derived.bar, { digits: 1 })}) on{' '}
              {pct(derived.above, { digits: 0 })} of days. Wide swings are normal: six months is too
              short to estimate a Sharpe.
            </MarginNote>
          ) : null
        }
      >
        {derived ? (
          <ChartFrame
            title="Net Sharpe over the trailing 126 trading days"
            description="Annualised. The dashed line is the phase bar used by the Sharpe gate."
            legend={
              <>
                <LegendItem
                  swatch={<LineSwatch color="var(--color-gain)" />}
                  label="Rolling net Sharpe"
                />
                <LegendItem
                  swatch={<LineSwatch color="var(--color-caution)" dashed />}
                  label={`Phase-${run.config.phase} bar (${num(derived.bar, { digits: 1 })})`}
                />
              </>
            }
            table={() => ({
              caption: 'Rolling 126-day net Sharpe at month ends',
              columns: ['Date', 'Rolling Sharpe'],
              rows: derived.dates
                .map((d, i) => ({ d, i }))
                .filter(
                  ({ d, i }) =>
                    derived.rolling[i] !== null &&
                    derived.dates[i + 1]?.slice(0, 7) !== d.slice(0, 7),
                )
                .map(({ d, i }) => [date(d), sharpe(derived.rolling[i], { signed: true })]),
            })}
          >
            <TimeSeriesChart
              dates={derived.dates}
              series={[{ label: 'Rolling Sharpe', values: derived.rolling, color: 'gain' }]}
              refLines={[
                { value: 0, color: 'graphite' },
                { value: derived.bar, color: 'caution', dashed: true },
              ]}
              format={(v) => num(v, { digits: 1 })}
              ariaLabel={`Rolling 126-day net Sharpe, between ${sharpe(derived.minRolling)} and ${sharpe(derived.maxRolling)}.`}
            />
          </ChartFrame>
        ) : (
          <Skeleton height="16rem" />
        )}
      </Section>
    </Flow>
  );
}
