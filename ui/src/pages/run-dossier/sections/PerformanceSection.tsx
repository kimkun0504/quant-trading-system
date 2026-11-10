import { useMemo } from 'react';
import type { BacktestSummary, RunDetail } from '@/shared/api';
import { metric } from '@/entities/metric';
import type { MetricKey } from '@/entities/metric';
import { metricContext, monthlyReturns, useRunSeries } from '@/entities/run';
import { BarChart, ChartFrame, LegendItem, LineSwatch, MonthlyHeatmap } from '@/shared/charts';
import { NavDrawdownChart } from '@/shared/charts/NavDrawdownChart';
import { currency, date, MONTH_LABELS, num, pct, sharpe } from '@/shared/lib/format';
import { About, ErrorState, Panel, Skeleton } from '@/shared/ui';
import { MetricTable } from '@/widgets/MetricTable';
import { DossierSection } from '../DossierSection';
import styles from './Sections.module.css';

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
];

export function PerformanceSection({ run, animate }: { run: RunDetail; animate: boolean }) {
  const series = useRunSeries(run.id, SERIES);
  const ctx = metricContext(run);
  const s = run.summary;

  const derived = useMemo(() => {
    if (!series.data) return null;
    const { dates } = series.data;
    const monthly = monthlyReturns(dates, series.data.series.net);
    const sorted = [...monthly.cells].sort((a, b) => a.ret - b.ret);
    return {
      dates,
      nav: series.data.series.nav.map((v) => v ?? Number.NaN),
      drawdown: series.data.series.drawdown.map((v) => v ?? 0),
      monthly,
      worst: sorted[0],
      best: sorted[sorted.length - 1],
      positive: monthly.cells.filter((c) => c.ret > 0).length / Math.max(1, monthly.cells.length),
    };
  }, [series.data]);

  const foldsOk = run.fold_sharpes.filter((f) => f.sharpe_net >= 0.05).length;

  return (
    <DossierSection
      id="performance"
      lede={`${metric('max_drawdown_net').interpret(s.max_drawdown_net, ctx)} The NAV ends at ${currency(s.final_nav)} from ${currency(5_000_000)}.`}
    >
      <Panel>
        {derived ? (
          <ChartFrame
            title="Net asset value and drawdown"
            description={`${date(s.start)} to ${date(s.end)}. Compounded daily; costs and borrow deducted every day; drawdown from the running peak.`}
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
        <About
          label="About this chart"
          handbook={{ page: 'backtesting', section: 'Reading a NAV curve' }}
        >
          A straight line on this scale is a constant dollar gain, not a constant growth rate; read
          the slope with the drawdown band under it, which says how long and how deep the strategy
          was under water at each point.
        </About>
      </Panel>

      <div className={styles.pair}>
        <Panel>
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
          {derived?.best && derived.worst ? (
            <About label="About this table">
              {pct(derived.positive, { digits: 0 })} of months were positive. Best{' '}
              {MONTH_LABELS[derived.best.month]} {derived.best.year} at{' '}
              {pct(derived.best.ret, { signed: true })}; worst {MONTH_LABELS[derived.worst.month]}{' '}
              {derived.worst.year} at {pct(derived.worst.ret, { signed: true })}. Look for clusters:
              a run of bad months in one year says more than any single month.
            </About>
          ) : null}
        </Panel>
        <Panel flush>
          <MetricTable
            label="Key metrics"
            ctx={ctx}
            rows={KEY_METRICS.map((k) => ({ key: k, value: s[k] }))}
          />
        </Panel>
      </div>

      <Panel>
        <ChartFrame
          title="Net Sharpe by walk-forward fold"
          description="Contiguous test blocks of the purged walk-forward split, labelled by first and last month. Dashed line: the +0.05 fold bar."
          table={{
            caption: 'Net Sharpe by fold',
            columns: ['Fold', 'Net Sharpe'],
            rows: run.fold_sharpes.map((f) => [f.label, sharpe(f.sharpe_net, { signed: true })]),
          }}
        >
          <BarChart
            bars={run.fold_sharpes.map((f) => ({ label: f.label, value: f.sharpe_net }))}
            format={(v) => num(v, { digits: 2 })}
            valueLabel="Net Sharpe"
            showValues
            threshold={{ value: 0.05, label: '+0.05' }}
            ariaLabel={`Net Sharpe by fold: ${run.fold_sharpes.map((f) => `${f.label} ${sharpe(f.sharpe_net)}`).join(', ')}.`}
          />
        </ChartFrame>
        <About
          label="About this chart"
          handbook={{ page: 'validation', section: '14. The hurdle gates' }}
        >
          {foldsOk} of {run.fold_sharpes.length} folds clear +0.05. Consistency across folds matters
          more than the average: one great stretch can carry a mediocre strategy.
        </About>
      </Panel>
    </DossierSection>
  );
}
