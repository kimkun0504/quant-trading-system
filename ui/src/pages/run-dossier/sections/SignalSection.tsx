import type { RunDetail } from '@/shared/api';
import { metric } from '@/entities/metric';
import { metricContext, useRunIc } from '@/entities/run';
import { BarChart, ChartFrame, PointLineChart } from '@/shared/charts';
import { date, num } from '@/shared/lib/format';
import { About, ErrorState, Panel, Skeleton } from '@/shared/ui';
import { MetricTable } from '@/widgets/MetricTable';
import { DossierSection } from '../DossierSection';
import styles from './Sections.module.css';

function decayNote(
  points: readonly { horizon_days: number; mean_ic: number }[],
  horizon: number,
): string {
  if (points.length < 2) return 'Not enough horizons to read a decay curve.';
  const peak = points.reduce((a, b) => (b.mean_ic > a.mean_ic ? b : a));
  const last = points[points.length - 1];
  const first = points[0];
  if (!last || !first) return '';
  const prev = points[points.length - 2];
  if (peak.horizon_days === last.horizon_days && prev && last.mean_ic > prev.mean_ic * 1.1)
    return `The IC is still rising at ${last.horizon_days} days: a slow signal that tolerates slow trading, so the ${horizon}-day target leaves information on the table.`;
  if (peak.horizon_days === last.horizon_days && prev)
    return `The IC rises to ${prev.horizon_days} days and then flattens (${num(prev.mean_ic, { digits: 4 })} to ${num(last.mean_ic, { digits: 4 })} at ${last.horizon_days} days): most of the information arrives within ${prev.horizon_days} days, so holding positions about that long matches the signal’s life.`;
  if (peak.horizon_days === first.horizon_days)
    return `The IC is highest at ${first.horizon_days} day${first.horizon_days === 1 ? '' : 's'} and fades after: a fast signal that costs will eat unless trading is quick and cheap.`;
  return `The IC peaks at ${peak.horizon_days} days (${num(peak.mean_ic, { digits: 4 })}) and flattens after; holding positions about that long matches the signal’s life.`;
}

export function SignalSection({ run }: { run: RunDetail }) {
  const ctx = metricContext(run);
  const ic = useRunIc(run.id);
  const r = run.ic_report;
  const positiveYears = r.by_year.filter((y) => y.mean_ic >= 0).length;

  return (
    <DossierSection id="signal" lede={metric('icir').interpret(r.icir, ctx)}>
      <div className={styles.pair}>
        <Panel flush>
          <MetricTable
            label="Signal statistics"
            ctx={ctx}
            rows={[
              { key: 'mean_ic', value: r.mean_ic },
              { key: 'icir', value: r.icir },
              { key: 'ic_t_stat', value: r.t_stat },
              { key: 'decile_t', value: run.decile?.t_stat },
              { key: 'transfer_coefficient_mean', value: run.summary.transfer_coefficient_mean },
            ]}
          />
        </Panel>
        <Panel>
          {ic.data ? (
            <ChartFrame
              title="IC decay"
              description="Mean rank IC against forward-return horizon. Cumulative, not marginal: each point uses the return over the whole horizon."
              table={{
                caption: 'Mean rank IC by horizon',
                columns: ['Horizon', 'Mean IC'],
                rows: ic.data.decay.map((p) => [
                  `${p.horizon_days} d`,
                  num(p.mean_ic, { digits: 4 }),
                ]),
              }}
            >
              <PointLineChart
                points={ic.data.decay.map((p) => ({
                  label: `${p.horizon_days} d`,
                  value: p.mean_ic,
                }))}
                format={(v) => num(v, { digits: 3 })}
                xLabel="Forward-return horizon (trading days)"
                ariaLabel={`IC decay: ${ic.data.decay.map((p) => `${p.horizon_days} days ${num(p.mean_ic, { digits: 3 })}`).join(', ')}.`}
              />
            </ChartFrame>
          ) : ic.isError ? (
            <ErrorState error={ic.error} onRetry={() => void ic.refetch()} />
          ) : (
            <Skeleton height="14rem" />
          )}
          {ic.data ? (
            <About
              label="About this chart"
              handbook={{ page: 'alpha', section: 'Diagnostics › IC decay and half-life' }}
            >
              {decayNote(ic.data.decay, ic.data.horizon_days)}
            </About>
          ) : null}
        </Panel>
      </div>

      <div className={styles.pairEven}>
        <Panel>
          <ChartFrame
            title="Mean rank IC by calendar year"
            description={`Non-overlapping ${r.horizon_days}-day formation dates; partial first and last years.`}
            table={{
              caption: 'Mean rank IC by year',
              columns: ['Year', 'Mean IC', 'Dates'],
              rows: r.by_year.map((y) => [
                String(y.year),
                num(y.mean_ic, { digits: 4, signed: true }),
                y.n,
              ]),
            }}
          >
            <BarChart
              bars={r.by_year.map((y) => ({ label: String(y.year), value: y.mean_ic }))}
              format={(v) => num(v, { digits: 3 })}
              valueLabel="Mean IC"
              showValues
              ariaLabel={`Mean rank IC by year: ${r.by_year.map((y) => `${y.year} ${num(y.mean_ic, { digits: 3 })}`).join(', ')}.`}
            />
          </ChartFrame>
          <About
            label="About this chart"
            handbook={{ page: 'validation', section: '14. The hurdle gates' }}
          >
            The mean IC was positive in {positiveYears} of {r.by_year.length} calendar years. The
            IC-sign gate is scored on the purged walk-forward blocks (the folds in Performance), not
            on years: it needs a positive mean IC in at least two thirds of them.
          </About>
        </Panel>
        <Panel>
          {ic.data ? (
            <ChartFrame
              title="Rank IC per formation date"
              description={`Spearman correlation of the score with the forward residualised return, one value every ${ic.data.horizon_days} trading days (${r.n} dates).`}
              table={() => ({
                caption: 'Rank IC by formation date',
                columns: ['Date', 'Rank IC'],
                rows: ic.data.ic.dates.map((d, i) => [
                  date(d),
                  num(ic.data.ic.series.rank_ic[i], { digits: 4, signed: true }),
                ]),
              })}
            >
              <BarChart
                bars={ic.data.ic.dates.map((d, i) => ({
                  label: d,
                  value: ic.data.ic.series.rank_ic[i] ?? 0,
                }))}
                format={(v) => num(v, { digits: 2 })}
                valueLabel="Rank IC"
                tickFormat={(d) => date(d).replace(/^\d+ /, '')}
                threshold={{ value: r.mean_ic, label: `mean ${num(r.mean_ic, { digits: 4 })}` }}
                ariaLabel={`Rank IC on ${r.n} dates, mean ${num(r.mean_ic, { digits: 4 })}, positive on ${Math.round(r.hit_rate * 100)} percent of dates.`}
              />
            </ChartFrame>
          ) : (
            <Skeleton height="14rem" />
          )}
        </Panel>
      </div>
    </DossierSection>
  );
}
