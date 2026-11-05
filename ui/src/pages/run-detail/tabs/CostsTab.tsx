import { useMemo, useState } from 'react';
import type { CostComponent } from '@/shared/api';
import {
  costShares,
  dailyCostBars,
  metricContext,
  monthlyCostBars,
  useRunCosts,
} from '@/entities/run';
import { metric } from '@/entities/metric';
import { BoxSwatch, ChartFrame, LegendItem, StackedBarChart } from '@/shared/charts';
import type { StackKey } from '@/shared/charts';
import { bps, currency, date, num, pct, sharpe, times } from '@/shared/lib/format';
import {
  Annotated,
  ErrorState,
  Flow,
  MarginNote,
  Section,
  SegmentedControl,
  Skeleton,
  SortableTable,
} from '@/shared/ui';
import type { Column } from '@/shared/ui';
import tableStyles from '@/shared/ui/Table.module.css';
import { MetricTable } from '@/widgets/MetricTable';
import { useRunOutlet } from '../context';
import styles from './Tabs.module.css';

/**
 * Cost components: trading costs are one hue in decreasing strength (they add up to
 * cost per side); borrow, a holding cost, is the neutral graphite.
 */
const KEYS: readonly StackKey<CostComponent>[] = [
  { key: 'spread', label: 'Spread', color: 'var(--color-gain)', opacity: 1 },
  { key: 'impact', label: 'Impact', color: 'var(--color-gain)', opacity: 0.62 },
  { key: 'fees', label: 'Fees', color: 'var(--color-gain)', opacity: 0.4 },
  { key: 'borrow', label: 'Borrow', color: 'var(--color-graphite)', opacity: 0.75 },
];

type Grain = 'monthly' | 'daily';

interface SensRow {
  multiplier: number;
  cost_bps_per_side: number;
  net_sharpe: number;
  net_cagr: number;
  mean_daily_cost_bps: number;
}

const SENS_COLUMNS: Column<SensRow>[] = [
  {
    key: 'm',
    header: 'Cost multiple',
    headerText: 'Cost multiple',
    rowHeader: true,
    render: (r) =>
      r.multiplier === 1 ? `${times(r.multiplier, 1)} (modelled)` : times(r.multiplier, 1),
  },
  {
    key: 'c',
    header: 'Cost per side',
    headerText: 'Cost per side',
    align: 'end',
    render: (r) => bps(r.cost_bps_per_side),
  },
  {
    key: 's',
    header: 'Net Sharpe',
    headerText: 'Net Sharpe',
    align: 'end',
    render: (r) => sharpe(r.net_sharpe, { signed: true }),
  },
  {
    key: 'g',
    header: 'Net CAGR',
    headerText: 'Net CAGR',
    align: 'end',
    render: (r) => pct(r.net_cagr, { signed: true }),
  },
  {
    key: 'd',
    header: 'Mean daily cost',
    headerText: 'Mean daily cost',
    align: 'end',
    render: (r) => `${num(r.mean_daily_cost_bps, { digits: 2 })} bp of NAV`,
  },
];

export function CostsTab() {
  const { run } = useRunOutlet();
  const costs = useRunCosts(run.id);
  const [grain, setGrain] = useState<Grain>('monthly');
  const s = run.summary;
  const ctx = metricContext(run);

  const bars = useMemo(() => {
    if (!costs.data) return null;
    return grain === 'monthly' ? monthlyCostBars(costs.data) : dailyCostBars(costs.data);
  }, [costs.data, grain]);
  const shares = useMemo(() => (costs.data ? costShares(costs.data) : null), [costs.data]);

  const multiple = s.breakeven_cost_bps / s.avg_cost_bps_per_side;
  const gross = (s.breakeven_cost_bps * s.avg_daily_turnover) / 1e4;
  const oneX = costs.data?.cost_sensitivity.find((r) => r.multiplier === 1);

  return (
    <Flow>
      <Section
        id="cost-breakdown"
        title="Where the costs go"
        note={
          shares ? (
            <MarginNote handbook={{ page: 'tcost', section: 'Cost components' }}>
              Spread {pct(shares.spread, { digits: 0 })}, impact {pct(shares.impact, { digits: 0 })}
              , fees {pct(shares.fees, { digits: 0 })}, borrow {pct(shares.borrow, { digits: 0 })}{' '}
              of {currency(s.total_costs_usd)} in total. Impact is the component a larger book would
              feel first: it grows with the square root of participation.
            </MarginNote>
          ) : null
        }
      >
        {bars ? (
          <ChartFrame
            title={grain === 'monthly' ? 'Costs summed by month' : 'Daily costs'}
            description="In basis points of NAV, stacked by component. Borrow is a holding cost and is not part of cost per side."
            legend={KEYS.map((k) => (
              <LegendItem
                key={k.key}
                swatch={<BoxSwatch color={k.color} opacity={k.opacity} />}
                label={k.label}
              />
            ))}
            controls={
              <SegmentedControl
                legend="Grain"
                value={grain}
                onChange={setGrain}
                options={[
                  { value: 'monthly', label: 'Monthly' },
                  { value: 'daily', label: 'Daily' },
                ]}
              />
            }
            table={() => ({
              caption: `${grain === 'monthly' ? 'Monthly' : 'Daily'} costs by component, bp of NAV`,
              columns: ['Period', ...KEYS.map((k) => k.label), 'Total'],
              rows: bars.map((b) => [
                grain === 'monthly' ? b.label : date(b.label),
                ...KEYS.map((k) => num(b.values[k.key], { digits: 2 })),
                num(
                  KEYS.reduce((a, k) => a + b.values[k.key], 0),
                  { digits: 2 },
                ),
              ]),
            })}
          >
            <StackedBarChart
              bars={bars}
              keys={KEYS}
              format={(v) => `${num(v, { digits: grain === 'daily' ? 1 : 0 })} bp`}
              labelFor={(l) => (grain === 'monthly' ? l : date(l))}
              ariaLabel={`${grain === 'monthly' ? 'Monthly' : 'Daily'} costs in basis points of NAV, stacked by spread, impact, fees and borrow.`}
            />
          </ChartFrame>
        ) : costs.isError ? (
          <ErrorState error={costs.error} onRetry={() => void costs.refetch()} />
        ) : (
          <Skeleton height="18rem" />
        )}
      </Section>

      <Section id="cost-metrics" title="Cost metrics">
        <MetricTable
          label="Cost metrics"
          ctx={ctx}
          rows={[
            { key: 'avg_cost_bps_per_side', value: s.avg_cost_bps_per_side },
            { key: 'breakeven_cost_bps', value: s.breakeven_cost_bps },
            { key: 'avg_daily_turnover', value: s.avg_daily_turnover },
            { key: 'effective_holding_days', value: s.effective_holding_days },
            { key: 'sharpe_gross', value: s.sharpe_gross },
          ]}
        />
      </Section>

      <Section
        id="sensitivity"
        title="Cost sensitivity"
        lede={<p>How fast does the strategy die if costs are worse than modelled?</p>}
        note={
          oneX ? (
            <MarginNote
              handbook={{
                page: 'backtesting',
                section: 'Metrics › Cost sensitivity table',
              }}
            >
              Read the table for its slope, not its level. The 1× row ({sharpe(oneX.net_sharpe)})
              differs from the reported net Sharpe ({sharpe(s.sharpe_net)}) because it charges a
              flat average cost per side and leaves out borrow.
            </MarginNote>
          ) : null
        }
      >
        {costs.data ? (
          <SortableTable
            caption="Net results recomputed as gross − multiple × cost per side × turnover"
            columns={SENS_COLUMNS}
            rows={costs.data.cost_sensitivity}
            rowKey={(r) => String(r.multiplier)}
            rowClassName={(r) => (r.multiplier === 1 ? tableStyles.highlight : undefined)}
          />
        ) : (
          <Skeleton height="10rem" />
        )}
      </Section>

      <Section id="breakeven" title="Breakeven cost">
        <Annotated
          note={
            <MarginNote handbook={metric('breakeven_cost_bps').handbook}>
              {metric('breakeven_cost_bps').interpret(s.breakeven_cost_bps, ctx)}
            </MarginNote>
          }
        >
          <div className={styles.prose}>
            <p>
              Breakeven is the cost per side at which the gross edge is used up exactly. If every
              dollar traded costs <em>c</em>, the mean net return is the mean gross return minus{' '}
              <em>c</em> × turnover; setting it to zero gives <em>c*</em> = mean gross return ÷ mean
              two-sided turnover.
            </p>
            <p>
              Here the book earns about {num(gross * 1e4, { digits: 1 })} bp of NAV a day gross and
              trades {pct(s.avg_daily_turnover, { digits: 2 })} of NAV a day, so <em>c*</em> ={' '}
              {bps(s.breakeven_cost_bps)} per side. The modelled cost is{' '}
              {bps(s.avg_cost_bps_per_side)}, so costs could rise {times(multiple, 1)} before the
              edge is gone. The breakeven gate asks for at least 2×: room for the cost model to be
              wrong by a factor of two.
            </p>
          </div>
        </Annotated>
      </Section>
    </Flow>
  );
}
