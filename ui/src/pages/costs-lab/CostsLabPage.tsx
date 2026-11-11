import { useMemo } from 'react';
import type { CostComponent, RunCosts, RunDetail } from '@/shared/api';
import { metric } from '@/entities/metric';
import {
  metricContext,
  monthlyCostBars,
  useManyRunConstraints,
  useRun,
  useRunConstraints,
  useRunCosts,
  useRunSeries,
} from '@/entities/run';
import { LabPage, RunLanes } from '@/features/lab-runs';
import type { LabRun } from '@/features/lab-runs';
import {
  BarChart,
  BoxSwatch,
  ChartFrame,
  HeatLegend,
  Heatmap,
  LegendItem,
  LineSwatch,
  StackedAreaChart,
  StackedBarChart,
} from '@/shared/charts';
import type { AreaKey, StackKey } from '@/shared/charts';
import { OverlayChart } from '@/shared/charts/OverlayChart';
import { bps, date, int, num, pct, sharpe, times } from '@/shared/lib/format';
import { About, EmptyState, ErrorState, Panel, Skeleton, SortableTable } from '@/shared/ui';
import type { Column } from '@/shared/ui';
import tableStyles from '@/shared/ui/Table.module.css';
import {
  bindingGrid,
  constraintLabel,
  costDistribution,
  costPerSideDaily,
  participationUsage,
  periodMeans,
  volReasonLabel,
  volReasonShares,
} from './lib/shape';
import styles from './CostsLabPage.module.css';

const WHAT =
  'What the optimiser could not do and what trading cost: which constraints bind and when, why the vol target was missed, the transfer coefficient, costs by kind over time, the cost-per-side distribution, sensitivity and breakeven, and how often the ADV cap was hit.';

export function CostsLabPage() {
  return (
    <LabPage title="Cost & constraint lab" docTitle="Costs & constraints" what={WHAT}>
      {(runs) => <CostsLab runs={runs} />}
    </LabPage>
  );
}

function CostsLab({ runs }: { runs: readonly LabRun[] }) {
  return (
    <>
      <RunLanes runs={runs} render={(r, solo) => <BindingPanel run={r} solo={solo} />} />
      <div className={styles.pair}>
        <TransferPanel runs={runs} />
        <ParticipationPanel runs={runs} />
      </div>
      <RunLanes runs={runs} render={(r, solo) => <VolReasonPanel run={r} solo={solo} />} />
      <RunLanes runs={runs} render={(r, solo) => <CostBreakdownPanel run={r} solo={solo} />} />
      <RunLanes runs={runs} render={(r) => <CostDistributionPanel run={r} />} />
      <RunLanes runs={runs} render={(r) => <SensitivityPanel run={r} />} />
    </>
  );
}

function NotRecorded({ title }: { title: string }) {
  return (
    <Panel title={title}>
      <EmptyState title="Not recorded for this run">
        <p>
          The run was written before the simulator kept its per-rebalance constraint log
          (`constraints.parquet`). Re-run it to see this chart.
        </p>
      </EmptyState>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- binding constraints
function BindingPanel({ run, solo }: { run: LabRun; solo: boolean }) {
  const q = useRunConstraints(run.id);
  const grid = useMemo(() => (q.data?.dates.length ? bindingGrid(q.data) : null), [q.data]);
  if (q.isError)
    return (
      <Panel>
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      </Panel>
    );
  if (!q.data)
    return (
      <Panel>
        <Skeleton height="16rem" />
      </Panel>
    );
  if (!grid) return <NotRecorded title="Binding constraints over time" />;
  const c = q.data;
  const fmt = (v: number) => pct(v, { digits: 0 });
  const shares = grid.rowKeys.map((k) => ({
    label: constraintLabel(k),
    value: c.share_binding[k] ?? 0,
  }));
  return (
    <Panel>
      <ChartFrame
        title="Binding constraints over time"
        description={`Share of each ${grid.grain}'s rebalance solves on which the constraint had a non-zero dual (it bound), over ${int(c.n_rebalances)} rebalances. Rows ordered by their whole-run share.`}
        legend={<HeatLegend scale="sequential" max={1} format={fmt} />}
        table={() => ({
          caption: `Share of solves where each constraint bound, by ${grid.grain}`,
          columns: ['Constraint', ...grid.cols],
          rows: grid.rows.map((r, i) => [
            r,
            ...(grid.values[i] ?? []).map((v) => (v === null ? 'n/a' : pct(v, { digits: 0 }))),
          ]),
        })}
      >
        <div className={styles.stack}>
          <Heatmap
            rows={grid.rows}
            cols={grid.cols}
            values={grid.values}
            scale="sequential"
            max={1}
            format={fmt}
            colLabel={(col, j) => {
              const y = col.slice(0, 4);
              return j === 0 || grid.cols[j - 1]?.slice(0, 4) !== y ? y : null;
            }}
            rowLabelWidth={132}
            cellHeight={solo ? 24 : 20}
            ariaLabel={`Share of solves on which each of ${grid.rows.length} constraints bound, by ${grid.grain}, for ${run.label}.`}
          />
          <BarChart
            bars={shares}
            format={fmt}
            valueLabel="Share of solves"
            showValues
            height={140}
            ariaLabel={`Whole-run share of solves where each constraint bound: ${shares.map((s) => `${s.label} ${fmt(s.value)}`).join(', ')}.`}
          />
        </div>
      </ChartFrame>
      <About label="About this chart" handbook={{ page: 'portfolio', section: 'Constraints' }}>
        A constraint that binds on most solves is shaping the book more than the alpha is: the
        position cap binding on many names means the signal wants concentration it is not allowed;
        the ADV participation cap binding is the book trading faster than the market lets it. The
        transfer coefficient below is the price of all of them together.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- transfer coefficient
function TransferPanel({ runs }: { runs: readonly LabRun[] }) {
  const queries = useManyRunConstraints(runs.map((r) => r.id));
  const ready = queries.every((q) => q.data);
  const error = queries.find((q) => q.isError);
  const shaped = useMemo(
    () =>
      queries.map((q) =>
        q.data?.dates.length ? periodMeans(q.data.dates, q.data.transfer_coefficient) : null,
      ),
    [queries],
  );
  const union = useMemo(() => {
    const set = new Set<string>();
    for (const s of shaped) for (const d of s?.dates ?? []) set.add(d);
    return [...set].sort();
  }, [shaped]);
  if (error)
    return (
      <Panel>
        <ErrorState error={error.error} onRetry={() => void error.refetch()} />
      </Panel>
    );
  if (!ready)
    return (
      <Panel>
        <Skeleton height="14rem" />
      </Panel>
    );
  if (!shaped.some(Boolean)) return <NotRecorded title="Transfer coefficient over time" />;
  const series = runs.map((r, k) => {
    const s = shaped[k];
    const byDate = new Map(s?.dates.map((d, i) => [d, s.values[i] ?? null]));
    return {
      label: r.label,
      paletteIndex: r.index,
      values: union.map((d) => byDate.get(d) ?? null),
    };
  });
  return (
    <Panel>
      <ChartFrame
        title="Transfer coefficient over time"
        description="Period mean of the correlation between the solved book and the unconstrained alpha-over-risk book, per rebalance."
        legend={
          runs.length > 1
            ? runs.map((r) => (
                <LegendItem
                  key={r.id}
                  swatch={<LineSwatch color={`var(--color-series-${(r.index % 8) + 1})`} />}
                  label={r.label}
                />
              ))
            : undefined
        }
        table={() => ({
          caption: 'Mean transfer coefficient by period and run',
          columns: ['Period', ...runs.map((r) => r.label)],
          rows: union.map((d, i) => [
            date(d),
            ...series.map((s) => num(s.values[i], { digits: 3 })),
          ]),
        })}
      >
        <OverlayChart
          dates={union}
          series={series}
          format={(v) => num(v, { digits: 2 })}
          height={220}
          ariaLabel={`Transfer coefficient over time of ${runs.length} run${runs.length === 1 ? '' : 's'}.`}
        />
      </ChartFrame>
      <About
        label="About this chart"
        handbook={{ page: 'portfolio', section: 'Transfer coefficient and trade aversion' }}
      >
        One is a book that is the alpha; 0.6 is a book that keeps about 60 % of the signal&apos;s
        information ratio after constraints and trade aversion. A falling TC with a rising binding
        share above is the constraint set tightening around the signal, not the signal fading.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- participation
function ParticipationPanel({ runs }: { runs: readonly LabRun[] }) {
  const queries = useManyRunConstraints(runs.map((r) => r.id));
  const ready = queries.every((q) => q.data);
  const error = queries.find((q) => q.isError);
  const shaped = useMemo(
    () => queries.map((q) => (q.data?.dates.length ? participationUsage(q.data) : null)),
    [queries],
  );
  const union = useMemo(() => {
    const set = new Set<string>();
    for (const s of shaped) for (const d of s?.dates ?? []) set.add(d);
    return [...set].sort();
  }, [shaped]);
  if (error)
    return (
      <Panel>
        <ErrorState error={error.error} onRetry={() => void error.refetch()} />
      </Panel>
    );
  if (!ready)
    return (
      <Panel>
        <Skeleton height="14rem" />
      </Panel>
    );
  if (!shaped.some((s) => s && s.dates.length))
    return (
      <Panel title="ADV participation cap usage">
        <EmptyState title="The participation cap never bound">
          <p>
            No rebalance in the selected runs had a name at its ADV cap (or the log was not
            recorded).
          </p>
        </EmptyState>
      </Panel>
    );
  const series = runs.map((r, k) => {
    const s = shaped[k];
    const byDate = new Map(s?.dates.map((d, i) => [d, s.share[i] ?? null]));
    return {
      label: r.label,
      paletteIndex: r.index,
      values: union.map((d) => byDate.get(d) ?? null),
    };
  });
  return (
    <Panel>
      <ChartFrame
        title="ADV participation cap usage"
        description="Period mean of the share of traded names whose trade sat at the per-name ADV participation cap."
        legend={
          runs.length > 1
            ? runs.map((r) => (
                <LegendItem
                  key={r.id}
                  swatch={<LineSwatch color={`var(--color-series-${(r.index % 8) + 1})`} />}
                  label={r.label}
                />
              ))
            : undefined
        }
        table={() => ({
          caption: 'Share of traded names at the participation cap, by period and run',
          columns: ['Period', ...runs.map((r) => r.label)],
          rows: union.map((d, i) => [
            date(d),
            ...series.map((s) => pct(s.values[i], { digits: 1 })),
          ]),
        })}
      >
        <OverlayChart
          dates={union}
          series={series}
          format={(v) => pct(v, { digits: 0 })}
          height={220}
          ariaLabel={`Share of traded names at the ADV participation cap over time, ${runs.length} run${runs.length === 1 ? '' : 's'}.`}
        />
      </ChartFrame>
      <About
        label="About this chart"
        handbook={{ page: 'tcost', section: 'Participation and impact' }}
      >
        The cap (5 % of ADV by default) keeps impact inside the square-root model&apos;s calibrated
        range. A high share here means the book is bigger or faster than the names are liquid: the
        trades it wanted are spread over more days, which lowers the transfer coefficient and
        lengthens the holding period.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- vol-target reason
const REASON_COLOURS: Record<string, string> = {
  target: 'var(--color-series-1)',
  caps: 'var(--color-series-4)',
  bracket: 'var(--color-series-5)',
  max_solves: 'var(--color-series-6)',
  no_alpha: 'var(--color-series-8)',
  solve_failed: 'var(--color-series-2)',
};

function VolReasonPanel({ run, solo }: { run: LabRun; solo: boolean }) {
  const q = useRunConstraints(run.id);
  const shaped = useMemo(() => (q.data?.dates.length ? volReasonShares(q.data) : null), [q.data]);
  if (q.isError)
    return (
      <Panel>
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      </Panel>
    );
  if (!q.data)
    return (
      <Panel>
        <Skeleton height="12rem" />
      </Panel>
    );
  if (!shaped) return <NotRecorded title="Why the vol target was or was not met" />;
  if (!shaped.reasons.length)
    return (
      <Panel title="Why the vol target was or was not met">
        <EmptyState title="Not vol-targeted">
          <p>This run sized the book with a fixed risk aversion, so there is no target to miss.</p>
        </EmptyState>
      </Panel>
    );
  const keys: AreaKey<string>[] = shaped.reasons.map((r) => ({
    key: r,
    label: volReasonLabel(r),
    color: REASON_COLOURS[r] ?? 'var(--color-graphite)',
  }));
  const fmt = (v: number) => pct(v, { digits: 0 });
  return (
    <Panel>
      <ChartFrame
        title="Why the vol target was or was not met"
        description="Share of each period's rebalances by the outcome of the gamma search: at target, capped below it by gross / position / ADV limits, the search bracket or budget hit, an empty book, or a failed solve."
        legend={keys.map((k) => (
          <LegendItem key={k.key} swatch={<BoxSwatch color={k.color} />} label={k.label} />
        ))}
        table={() => ({
          caption: 'Share of rebalances by vol-target outcome and period',
          columns: ['Period', ...keys.map((k) => k.label)],
          rows: shaped.dates.map((d, i) => [
            date(d),
            ...keys.map((k) => fmt(shaped.series[k.key]?.[i] ?? 0)),
          ]),
        })}
      >
        <StackedAreaChart
          dates={shaped.dates}
          series={shaped.series}
          keys={keys}
          format={fmt}
          diverging={false}
          yDomain={[0, 1]}
          height={solo ? 180 : 150}
          ariaLabel={`Vol-target outcome shares over time for ${run.label}.`}
        />
      </ChartFrame>
      <About label="About this chart" handbook={{ page: 'portfolio', section: 'Vol targeting' }}>
        In target-vol mode the optimiser searches the risk aversion until the ex-ante vol is within
        2 % of the target. &ldquo;Capped below target&rdquo; is the honest failure: the caps would
        not let the book get big enough, and the capped book is accepted. A rising capped share is a
        strategy that has outgrown its liquidity or its position limit.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- cost breakdown
const COST_KEYS: readonly StackKey<CostComponent>[] = [
  { key: 'spread', label: 'Spread', color: 'var(--color-gain)', opacity: 1 },
  { key: 'impact', label: 'Impact', color: 'var(--color-gain)', opacity: 0.62 },
  { key: 'fees', label: 'Fees', color: 'var(--color-gain)', opacity: 0.4 },
  { key: 'borrow', label: 'Borrow', color: 'var(--color-graphite)', opacity: 0.75 },
];

function CostBreakdownPanel({ run, solo }: { run: LabRun; solo: boolean }) {
  const q = useRunCosts(run.id);
  const bars = useMemo(() => (q.data ? monthlyCostBars(q.data) : null), [q.data]);
  if (q.isError)
    return (
      <Panel>
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      </Panel>
    );
  if (!bars)
    return (
      <Panel>
        <Skeleton height="16rem" />
      </Panel>
    );
  return (
    <Panel>
      <ChartFrame
        title="Costs by month and kind"
        description="Basis points of NAV a month, stacked: spread, impact and fees are trading costs; borrow is a holding cost."
        legend={COST_KEYS.map((k) => (
          <LegendItem
            key={k.key}
            swatch={<BoxSwatch color={k.color} opacity={k.opacity} />}
            label={k.label}
          />
        ))}
        table={() => ({
          caption: 'Monthly costs by component, bp of NAV',
          columns: ['Month', ...COST_KEYS.map((k) => k.label), 'Total'],
          rows: bars.map((b) => [
            b.label,
            ...COST_KEYS.map((k) => num(b.values[k.key], { digits: 2 })),
            num(
              COST_KEYS.reduce((a, k) => a + b.values[k.key], 0),
              { digits: 2 },
            ),
          ]),
        })}
      >
        <StackedBarChart
          bars={bars}
          keys={COST_KEYS}
          format={(v) => `${num(v, { digits: 0 })} bp`}
          labelFor={(l) => l}
          height={solo ? 220 : 170}
          ariaLabel={`Monthly costs of ${run.label} in basis points of NAV, stacked by spread, impact, fees and borrow.`}
        />
      </ChartFrame>
      <About label="About this chart" handbook={{ page: 'tcost', section: 'Cost components' }}>
        Impact is the component that scales with size (square root of participation) and the one a
        bigger book feels first; spread and fees scale with turnover alone; borrow with the short
        book and the names in it. Months that spike without a turnover spike are hard-to-borrow or
        illiquid names entering the book.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- cost per side distribution
function CostDistributionPanel({ run }: { run: LabRun }) {
  const costs = useRunCosts(run.id);
  const series = useRunSeries(run.id, ['turnover'] as const);
  const dist = useMemo(() => {
    if (!costs.data || !series.data) return null;
    const ps = costPerSideDaily(costs.data, series.data.series.turnover, series.data.dates);
    return costDistribution(ps);
  }, [costs.data, series.data]);
  const err = costs.isError ? costs : series.isError ? series : null;
  if (err)
    return (
      <Panel>
        <ErrorState error={err.error} onRetry={() => void err.refetch()} />
      </Panel>
    );
  if (!dist)
    return (
      <Panel>
        <Skeleton height="14rem" />
      </Panel>
    );
  const bars = dist.bins.map((b) => ({
    label: `${num(b.from, { digits: 1 })}–${num(b.to, { digits: 1 })}`,
    value: b.count,
  }));
  return (
    <Panel>
      <ChartFrame
        title="Cost per side, day by day"
        description={`Trading cost per dollar traded on each of ${int(dist.n)} trading days (borrow excluded), in bp; the cheapest and dearest 1 % of days fall in the end bins.`}
        table={{
          caption: 'Days by cost-per-side bin',
          columns: ['Cost per side (bp)', 'Days'],
          rows: bars.map((b) => [b.label, b.value]),
        }}
      >
        <BarChart
          bars={bars}
          format={(v) => int(v)}
          valueLabel="Days"
          height={180}
          tickFormat={(l) => l.split('–')[0] ?? l}
          ariaLabel={`Distribution of daily cost per side for ${run.label}: median ${bps(dist.median)}, 90th percentile ${bps(dist.p90)}.`}
        />
      </ChartFrame>
      <About
        label="About this chart"
        handbook={{ page: 'tcost', section: 'Reading the TCA ledger' }}
      >
        Median {bps(dist.median)}, mean {bps(dist.mean)}, 90th percentile {bps(dist.p90)}. The
        headline cost per side is a dollar-weighted average, so a long right tail (a few very
        expensive days) pulls it above the median; those are the days impact dominated.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- sensitivity + breakeven
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
];

function SensitivityPanel({ run }: { run: LabRun }) {
  const detail = useRun(run.id);
  const costs = useRunCosts(run.id);
  if (detail.isError)
    return (
      <Panel>
        <ErrorState error={detail.error} onRetry={() => void detail.refetch()} />
      </Panel>
    );
  if (!detail.data || !costs.data)
    return (
      <Panel>
        <Skeleton height="14rem" />
      </Panel>
    );
  return <Sensitivity run={detail.data} costs={costs.data} />;
}

function Sensitivity({ run, costs }: { run: RunDetail; costs: RunCosts }) {
  const s = run.summary;
  const ctx = metricContext(run);
  const multiple = s.breakeven_cost_bps / s.avg_cost_bps_per_side;
  return (
    <Panel
      title="Cost sensitivity and breakeven"
      subtitle="how fast the edge dies if costs are worse than modelled"
      flush
    >
      <SortableTable
        caption="Net results recomputed as gross − multiple × cost per side × turnover"
        columns={SENS_COLUMNS}
        rows={costs.cost_sensitivity}
        rowKey={(r) => String(r.multiplier)}
        rowClassName={(r) => (r.multiplier === 1 ? tableStyles.highlight : undefined)}
      />
      <p className={styles.breakeven}>
        Breakeven <strong>{bps(s.breakeven_cost_bps)}</strong> per side against{' '}
        {bps(s.avg_cost_bps_per_side)} modelled: a margin of {times(multiple, 1)}; the gate asks for
        2×.
      </p>
      <About
        label="About breakeven"
        handbook={metric('breakeven_cost_bps').handbook}
        className={styles.inset}
      >
        {metric('breakeven_cost_bps').interpret(s.breakeven_cost_bps, ctx)}
      </About>
    </Panel>
  );
}
