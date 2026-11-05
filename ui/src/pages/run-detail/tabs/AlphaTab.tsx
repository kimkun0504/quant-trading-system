import { metric } from '@/entities/metric';
import { metricContext, useRunIc } from '@/entities/run';
import { BarChart, ChartFrame, PointLineChart } from '@/shared/charts';
import { date, num } from '@/shared/lib/format';
import { ErrorState, Flow, MarginNote, Section, Skeleton } from '@/shared/ui';
import { MetricTable } from '@/widgets/MetricTable';
import { useRunOutlet } from '../context';

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

export function AlphaTab() {
  const { run } = useRunOutlet();
  const ctx = metricContext(run);
  const ic = useRunIc(run.id);
  const r = run.ic_report;

  return (
    <Flow>
      <Section
        id="ic-series"
        title="Rank IC per formation date"
        note={
          <MarginNote handbook={metric('icir').handbook}>
            {metric('icir').interpret(r.icir, ctx)}
          </MarginNote>
        }
      >
        {ic.data ? (
          <ChartFrame
            title="Spearman correlation of the score with the forward residualised return"
            description={`One value every ${ic.data.horizon_days} trading days (non-overlapping), ${r.n} dates.`}
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
              threshold={{
                value: r.mean_ic,
                label: `mean ${num(r.mean_ic, { digits: 4 })}`,
              }}
              ariaLabel={`Rank IC on ${r.n} dates, mean ${num(r.mean_ic, { digits: 4 })}, positive on ${Math.round(r.hit_rate * 100)} percent of dates.`}
            />
          </ChartFrame>
        ) : ic.isError ? (
          <ErrorState error={ic.error} onRetry={() => void ic.refetch()} />
        ) : (
          <Skeleton height="16rem" />
        )}
      </Section>

      <Section
        id="decay"
        title="IC decay"
        note={
          ic.data ? (
            <MarginNote
              handbook={{
                page: 'alpha',
                section: 'Diagnostics › IC decay and half-life',
              }}
            >
              {decayNote(ic.data.decay, ic.data.horizon_days)}
            </MarginNote>
          ) : null
        }
      >
        {ic.data ? (
          <ChartFrame
            title="Mean rank IC against forward-return horizon"
            description="Cumulative, not marginal: each point uses the return over the whole horizon."
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
        ) : (
          <Skeleton height="14rem" />
        )}
      </Section>

      <Section id="alpha-metrics" title="Signal statistics">
        <MetricTable
          label="Signal statistics"
          ctx={ctx}
          rows={[
            { key: 'mean_ic', value: r.mean_ic },
            { key: 'icir', value: r.icir },
            { key: 'ic_t_stat', value: r.t_stat },
            { key: 'decile_t', value: run.decile?.t_stat },
            {
              key: 'transfer_coefficient_mean',
              value: run.summary.transfer_coefficient_mean,
            },
          ]}
        />
      </Section>
    </Flow>
  );
}
