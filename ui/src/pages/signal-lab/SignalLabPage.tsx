import { useMemo, useState } from 'react';
import type { AlphaFeatureIc, AlphaRefits, AlphaThemeWeights } from '@/shared/api';
import { useAlphaRefits, useIcByYear, useManyRunIc, useThemeWeights } from '@/entities/run';
import { LabPage, RunLanes } from '@/features/lab-runs';
import type { LabRun } from '@/features/lab-runs';
import {
  BoxSwatch,
  ChartFrame,
  HeatLegend,
  Heatmap,
  LegendItem,
  LineSwatch,
  MultiPointLineChart,
  StackedAreaChart,
} from '@/shared/charts';
import type { AreaKey } from '@/shared/charts';
import { OverlayChart } from '@/shared/charts/OverlayChart';
import { TimeSeriesChart } from '@/shared/charts/TimeSeriesChart';
import { date, int, num, pct } from '@/shared/lib/format';
import {
  About,
  EmptyState,
  ErrorState,
  Panel,
  SegmentedControl,
  Skeleton,
  SortableTable,
} from '@/shared/ui';
import type { Column } from '@/shared/ui';
import {
  featureIcRows,
  icByYearGrid,
  icWithRollingMean,
  refitGrid,
  sleeveLabel,
  storyLine,
  yearLabel,
} from './lib/shape';
import styles from './SignalLabPage.module.css';

const WHAT =
  'Where the signal comes from and how it has moved: IC by theme and year, IC decay, the IC series, each feature’s IC, and what every refit learned.';

/**
 * Signal lab: the alpha model across time. One run is the deep view; several are lanes
 * side by side with the overlays (decay, rolling IC) drawn on one chart.
 */
export function SignalLabPage() {
  return (
    <LabPage title="Signal lab" docTitle="Signal lab" what={WHAT}>
      {(runs) => <SignalLab runs={runs} />}
    </LabPage>
  );
}

function SignalLab({ runs }: { runs: readonly LabRun[] }) {
  return (
    <>
      <RunLanes runs={runs} render={(r, solo) => <IcByYearPanel run={r} solo={solo} />} />
      <div className={styles.pair}>
        <DecayPanel runs={runs} />
        <RollingIcPanel runs={runs} />
      </div>
      <RunLanes runs={runs} render={(r, solo) => <LearnedWeightsPanel run={r} solo={solo} />} />
      <RunLanes runs={runs} render={(r) => <ThemeWeightsPanel run={r} />} />
      <RunLanes runs={runs} render={(r) => <FeatureIcPanel run={r} />} />
    </>
  );
}

// ----------------------------------------------------------------------------- IC by theme by year
function IcByYearPanel({ run, solo }: { run: LabRun; solo: boolean }) {
  const q = useIcByYear(run.id);
  const grid = useMemo(() => (q.data ? icByYearGrid(q.data) : null), [q.data]);
  const story = useMemo(() => (q.data ? storyLine(q.data) : ''), [q.data]);
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  if (!q.data || !grid) return <Skeleton height={solo ? '18rem' : '14rem'} />;
  const fmt = (v: number) => num(v, { digits: 3, signed: true });
  const topRows = grid.rowKeys
    .map((k, i) => (k === 'combined' || k === 'composite' || k === 'ml' ? i : -1))
    .filter((i) => i >= 0);
  return (
    <Panel>
      <ChartFrame
        title="IC by theme by year"
        description={
          q.data.source === 'daily_sleeves'
            ? `Mean daily rank IC of each score against the ${q.data.overlap_days - 1}-day residualised label, by calendar year; hover for the Newey–West t.`
            : 'This run recorded only the sampled combined IC, so the theme rows are missing.'
        }
        legend={<HeatLegend scale="diverging" max={grid.maxAbs} format={fmt} />}
        table={{
          caption: 'Mean rank IC by sleeve and year (t in brackets)',
          columns: ['Sleeve', ...grid.cols],
          rows: grid.rows.map((r, i) => [
            r,
            ...grid.cols.map((_, j) => {
              const v = grid.values[i]?.[j];
              const t = grid.t?.[i]?.[j];
              return v === null || v === undefined
                ? 'n/a'
                : `${fmt(v)} (${t === null || t === undefined ? 'n/a' : num(t, { digits: 1 })})`;
            }),
          ]),
        }}
      >
        <Heatmap
          rows={grid.rows}
          cols={grid.cols}
          values={grid.values}
          scale="diverging"
          format={fmt}
          emphasisRows={topRows}
          cellHeight={solo ? 24 : 20}
          readoutFor={(c) => {
            const t = grid.t?.[c.row]?.[c.col];
            const n = q.data.rows[c.row]?.cells[c.col]?.n ?? 0;
            return [
              { label: 't', value: t === null || t === undefined ? 'n/a' : num(t, { digits: 2 }) },
              { label: 'Dates', value: int(n) },
            ];
          }}
          ariaLabel={`Mean rank IC by sleeve and calendar year for ${run.label}: ${grid.rows.length} sleeves over ${grid.cols[0] ?? ''} to ${grid.cols[grid.cols.length - 1] ?? ''}.`}
        />
      </ChartFrame>
      <About
        label="About this chart"
        handbook={{ page: 'alpha', section: 'Diagnostics › IC by year' }}
      >
        {story ? `${story} ` : ''}
        Colour is centred at zero: blue is a year the sleeve ranked next month’s residual returns
        the right way round, red the wrong way. A |t| above 2 is a year the sign is unlikely to be
        noise; most single years are not, so read rows for their drift rather than their cells.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- decay
function DecayPanel({ runs }: { runs: readonly LabRun[] }) {
  const queries = useManyRunIc(runs.map((r) => r.id));
  const ready = queries.every((q) => q.data);
  const error = queries.find((q) => q.isError);
  const labels = useMemo(() => {
    const seen = new Set<number>();
    for (const q of queries) for (const p of q.data?.decay ?? []) seen.add(p.horizon_days);
    return [...seen].sort((a, b) => a - b);
  }, [queries]);
  const series = runs.map((r, i) => ({
    label: r.label,
    paletteIndex: r.index,
    values: labels.map(
      (h) => queries[i]?.data?.decay.find((p) => p.horizon_days === h)?.mean_ic ?? null,
    ),
  }));
  return (
    <Panel>
      {error ? (
        <ErrorState error={error.error} onRetry={() => void error.refetch()} />
      ) : ready ? (
        <ChartFrame
          title="IC decay by horizon"
          description="Mean rank IC against the forward-return horizon. Cumulative: each point uses the whole horizon’s return."
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
          table={{
            caption: 'Mean rank IC by horizon and run',
            columns: ['Horizon', ...runs.map((r) => r.label)],
            rows: labels.map((h, j) => [
              `${h} d`,
              ...series.map((s) => {
                const v = s.values[j];
                return v === null || v === undefined ? 'n/a' : num(v, { digits: 4 });
              }),
            ]),
          }}
        >
          <MultiPointLineChart
            labels={labels.map((h) => `${h} d`)}
            series={series}
            format={(v) => num(v, { digits: 3 })}
            xLabel="Forward-return horizon (trading days)"
            ariaLabel={`IC decay of ${runs.length} run${runs.length === 1 ? '' : 's'} over horizons ${labels.join(', ')} days.`}
          />
        </ChartFrame>
      ) : (
        <Skeleton height="14rem" />
      )}
      <About
        label="About this chart"
        handbook={{ page: 'alpha', section: 'Diagnostics › IC decay and half-life' }}
      >
        A curve still rising at the longest horizon is a slow signal that tolerates slow, cheap
        trading; one that peaks early and fades is fast and will be eaten by costs unless trading is
        quick. The alpha horizon (20 days by default) should sit near the knee.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- rolling IC
function RollingIcPanel({ runs }: { runs: readonly LabRun[] }) {
  const queries = useManyRunIc(runs.map((r) => r.id));
  const ready = queries.every((q) => q.data);
  const error = queries.find((q) => q.isError);
  const shaped = useMemo(
    () => queries.map((q) => (q.data ? icWithRollingMean(q.data) : null)),
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
  const solo = runs.length === 1 ? shaped[0] : null;
  const window = shaped[0]?.window ?? 0;
  return (
    <Panel>
      {solo ? (
        <ChartFrame
          title="Rank IC with its trailing one-year mean"
          description={`One value every ${queries[0]?.data?.horizon_days ?? 20} trading days; the mean runs over the last ${window} of them.`}
          legend={
            <>
              <LegendItem swatch={<LineSwatch color="var(--color-graphite)" />} label="Rank IC" />
              <LegendItem
                swatch={<LineSwatch color="var(--color-gain)" />}
                label="Trailing 252-day mean"
              />
            </>
          }
          table={() => ({
            caption: 'Rank IC and trailing mean by formation date',
            columns: ['Date', 'Rank IC', 'Trailing mean'],
            rows: solo.dates.map((d, i) => [
              date(d),
              num(solo.raw[i], { digits: 4, signed: true }),
              num(solo.rolling[i], { digits: 4, signed: true }),
            ]),
          })}
        >
          <TimeSeriesChart
            dates={solo.dates}
            series={[
              { label: 'Rank IC', values: solo.raw, color: 'graphite' },
              { label: 'Trailing mean', values: solo.rolling, color: 'gain' },
            ]}
            refLines={[{ value: 0, color: 'ink' }]}
            format={(v) => num(v, { digits: 3 })}
            ariaLabel={`Rank IC over ${solo.dates.length} formation dates with its trailing one-year mean.`}
          />
        </ChartFrame>
      ) : (
        <ChartFrame
          title="Trailing one-year mean rank IC"
          description={`Each run’s IC, sampled every alpha horizon, averaged over its last ${window} values.`}
          legend={runs.map((r) => (
            <LegendItem
              key={r.id}
              swatch={<LineSwatch color={`var(--color-series-${(r.index % 8) + 1})`} />}
              label={r.label}
            />
          ))}
          table={() => ({
            caption: 'Trailing one-year mean rank IC by date and run',
            columns: ['Date', ...runs.map((r) => r.label)],
            rows: union.map((d) => [
              date(d),
              ...shaped.map((s) => {
                const i = s?.dates.indexOf(d) ?? -1;
                return i >= 0 ? num(s?.rolling[i], { digits: 4, signed: true }) : 'n/a';
              }),
            ]),
          })}
        >
          <OverlayChart
            dates={union}
            series={runs.map((r, k) => {
              const s = shaped[k];
              const byDate = new Map(s?.dates.map((d, i) => [d, s.rolling[i] ?? null]));
              return {
                label: r.label,
                paletteIndex: r.index,
                values: union.map((d) => byDate.get(d) ?? null),
              };
            })}
            format={(v) => num(v, { digits: 3 })}
            height={220}
            ariaLabel={`Trailing one-year mean rank IC of ${runs.length} runs.`}
          />
        </ChartFrame>
      )}
      <About
        label="About this chart"
        handbook={{ page: 'validation', section: 'IC, ICIR and t-statistics' }}
      >
        The raw IC is noisy by construction (one cross-section per point); the trailing mean is what
        the Grinold scaling sees, and a stretch below zero is where the IC floor keeps the book from
        emptying. Compare runs on the mean, not the spikes.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- learned weights
function LearnedWeightsPanel({ run, solo }: { run: LabRun; solo: boolean }) {
  const q = useAlphaRefits(run.id);
  const [sleeve, setSleeve] = useState<string | null>(null);
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
  return (
    <LearnedWeights data={q.data} run={run} solo={solo} sleeve={sleeve} setSleeve={setSleeve} />
  );
}

function LearnedWeights({
  data,
  run,
  solo,
  sleeve,
  setSleeve,
}: {
  data: AlphaRefits;
  run: LabRun;
  solo: boolean;
  sleeve: string | null;
  setSleeve: (s: string) => void;
}) {
  const chosen =
    sleeve && data.sleeves.includes(sleeve)
      ? sleeve
      : (data.sleeves.find((s) => s === 'ensemble') ?? data.sleeves[0] ?? null);
  const grid = useMemo(() => (chosen ? refitGrid(data, chosen) : null), [data, chosen]);
  if (!data.recorded)
    return (
      <Panel title="Learned weights over time">
        <EmptyState title="Not recorded for this run">
          <p>
            The run was written before the alpha model recorded what each refit learned. Re-run it
            to see the heatmap.
          </p>
        </EmptyState>
      </Panel>
    );
  if (!data.refits.length || !grid || !chosen)
    return (
      <Panel title="Learned weights over time">
        <EmptyState title="No ML sleeve in this run">
          <p>
            The composite has no fitted weights per feature: its theme weights over time are the
            chart below, and the feature IC table says what each feature contributed.
          </p>
        </EmptyState>
      </Panel>
    );
  const fmt = (v: number) => pct(v, { digits: 1 });
  const first = data.refits[0];
  const last = data.refits[data.refits.length - 1];
  return (
    <Panel>
      <ChartFrame
        title="Learned weights over time"
        description={`Feature importance of the ${sleeveLabel(chosen).toLowerCase()} at each of the ${data.refits.length} walk-forward refits, normalised per refit to sum 1. Rows are ordered by mean share.`}
        legend={<HeatLegend scale="sequential" max={grid.maxAbs} format={fmt} />}
        controls={
          data.sleeves.length > 1 ? (
            <SegmentedControl
              legend="Sleeve"
              value={chosen}
              onChange={setSleeve}
              options={data.sleeves.map((s) => ({
                value: s,
                label: sleeveLabel(s).replace(/ \(.*\)$/, ''),
              }))}
            />
          ) : undefined
        }
        table={() => ({
          caption: `Importance share by feature and refit (${sleeveLabel(chosen)})`,
          columns: ['Feature', ...data.refits.map((r) => r.refit_date)],
          rows: grid.rows.map((f, i) => [
            f,
            ...(grid.values[i] ?? []).map((v) => (v === null ? 'n/a' : pct(v, { digits: 1 }))),
          ]),
        })}
      >
        <Heatmap
          rows={grid.rows}
          cols={grid.cols}
          values={grid.values}
          scale="sequential"
          format={fmt}
          colLabel={yearLabel(grid.cols)}
          cellHeight={solo ? 22 : 18}
          rowLabelWidth={128}
          readoutFor={(c) => {
            const w = data.refits[c.col];
            return w
              ? [
                  { label: 'Refit', value: date(w.refit_date) },
                  { label: 'Trained on', value: `${date(w.train_start)} to ${date(w.train_end)}` },
                  { label: 'Rows', value: int(w.n_train) },
                ]
              : [];
          }}
          ariaLabel={`Feature importance of ${sleeveLabel(chosen)} across ${data.refits.length} refits for ${run.label}, ${grid.rows.length} features.`}
        />
      </ChartFrame>
      <About
        label="About this chart"
        handbook={{ page: 'alpha', section: 'Walk-forward training and refit importance' }}
      >
        Each column is a model trained only on labels that ended {data.embargo_days} trading days
        before its refit date ({first ? date(first.refit_date) : ''} to{' '}
        {last ? date(last.refit_date) : ''}, every {data.refit_frequency_days} days), so a row that
        brightens over time is a feature the data kept rewarding, not hindsight. LightGBM gives gain
        shares, ridge |coefficients|, the ensemble their mean.{' '}
        {data.notes.length ? data.notes.join(' ') + '.' : ''}
        {first && Object.keys(first.blend_weights).length
          ? ` Sleeves are blended ${Object.entries(first.blend_weights)
              .map(([k, v]) => `${sleeveLabel(k).replace(/ \(.*\)$/, '')} ${pct(v, { digits: 0 })}`)
              .join(', ')}.`
          : ''}
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- theme weights
const THEME_KEYS: readonly AreaKey<string>[] = [
  'value',
  'momentum',
  'quality',
  'investment',
  'low_risk',
  'short_interest',
].map((k, i) => ({
  key: k,
  label: sleeveLabel(`theme_${k}`),
  color: `var(--color-series-${i + 1})`,
}));

function ThemeWeightsPanel({ run }: { run: LabRun }) {
  const q = useThemeWeights(run.id);
  if (q.isError)
    return (
      <Panel>
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      </Panel>
    );
  if (!q.data)
    return (
      <Panel>
        <Skeleton height="14rem" />
      </Panel>
    );
  return <ThemeWeights data={q.data} run={run} />;
}

function ThemeWeights({ data, run }: { data: AlphaThemeWeights; run: LabRun }) {
  const keys = useMemo(
    () =>
      THEME_KEYS.filter((k) => data.themes.includes(k.key)).concat(
        data.themes
          .filter((t) => !THEME_KEYS.some((k) => k.key === t))
          .map((t, i) => ({
            key: t,
            label: sleeveLabel(`theme_${t}`),
            color: `var(--color-series-${((THEME_KEYS.length + i) % 8) + 1})`,
          })),
      ),
    [data.themes],
  );
  if (!data.dates.length)
    return (
      <Panel title="Theme weights over time">
        <EmptyState title="Not recorded for this run">
          <p>The run was written before the composite recorded its theme weights.</p>
        </EmptyState>
      </Panel>
    );
  const fmt = (v: number) => num(v, { digits: 2 });
  return (
    <Panel>
      <ChartFrame
        title="Theme weights over time"
        description={
          data.dynamic
            ? 'The composite’s weight on each theme by decision date: the static weights tilted by factor momentum (±25 %), momentum also scaled down in bear, high-vol states and by its trailing vol.'
            : 'Static weights: this run did not use dynamic theme weighting.'
        }
        legend={keys.map((k) => (
          <LegendItem key={k.key} swatch={<BoxSwatch color={k.color} />} label={k.label} />
        ))}
        table={() => ({
          caption: 'Theme weights at month ends',
          columns: ['Date', ...keys.map((k) => k.label)],
          rows: data.dates
            .map((d, i) => ({ d, i }))
            .filter(({ d, i }) => data.dates[i + 1]?.slice(0, 7) !== d.slice(0, 7))
            .map(({ d, i }) => [date(d), ...keys.map((k) => fmt(data.series[k.key]?.[i] ?? 0))]),
        })}
      >
        <StackedAreaChart
          dates={data.dates}
          series={Object.fromEntries(keys.map((k) => [k.key, data.series[k.key] ?? []]))}
          keys={keys}
          format={fmt}
          diverging={false}
          height={200}
          ariaLabel={`Theme weights of the composite over time for ${run.label}, stacked by theme.`}
        />
      </ChartFrame>
      <About
        label="About this chart"
        handbook={{ page: 'alpha', section: 'Dynamic theme weights' }}
      >
        The static weights are{' '}
        {Object.entries(data.static)
          .map(([k, v]) => `${sleeveLabel(`theme_${k}`).toLowerCase()} ${num(v, { digits: 2 })}`)
          .join(', ')}
        . The scoring renormalises by the weights’ sum on each date, so only the proportions matter;
        the total moves when momentum is scaled.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- feature IC
const FEATURE_COLUMNS: Column<AlphaFeatureIc>[] = [
  {
    key: 'f',
    header: 'Feature',
    headerText: 'Feature',
    rowHeader: true,
    render: (r) => r.feature,
    sortValue: (r) => r.feature,
  },
  {
    key: 'theme',
    header: 'Theme',
    headerText: 'Theme',
    render: (r) => (r.theme ? sleeveLabel(`theme_${r.theme}`) : 'ML only'),
    sortValue: (r) => r.theme ?? '',
  },
  {
    key: 'sign',
    header: 'Sign',
    headerText: 'Sign as used',
    align: 'end',
    render: (r) => (r.sign < 0 ? '−1' : '+1'),
    sortValue: (r) => r.sign,
  },
  {
    key: 'ic',
    header: 'Mean IC',
    headerText: 'Mean IC',
    align: 'end',
    render: (r) => num(r.mean_ic, { digits: 4, signed: true }),
    sortValue: (r) => r.mean_ic,
  },
  {
    key: 't',
    header: 't',
    headerText: 'Newey–West t',
    align: 'end',
    render: (r) => num(r.t_stat, { digits: 2, signed: true }),
    sortValue: (r) => r.t_stat,
  },
  {
    key: 'n',
    header: 'Dates',
    headerText: 'Dates',
    align: 'end',
    render: (r) => int(r.n),
    sortValue: (r) => r.n,
  },
];

function FeatureIcPanel({ run }: { run: LabRun }) {
  const q = useAlphaRefits(run.id);
  const rows = useMemo(() => (q.data ? featureIcRows(q.data) : []), [q.data]);
  if (q.isError)
    return (
      <Panel>
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      </Panel>
    );
  if (!q.data)
    return (
      <Panel>
        <Skeleton height="14rem" />
      </Panel>
    );
  const strongest = rows[0];
  return (
    <Panel title="Feature IC" subtitle="full sample, sign as the composite uses it" flush>
      {rows.length ? (
        <SortableTable
          caption={`${rows.length} processed features; daily rank IC against the residualised label`}
          columns={FEATURE_COLUMNS}
          rows={rows}
          rowKey={(r) => r.feature}
          initialSort={{ key: 't', dir: 'descending' }}
        />
      ) : (
        <EmptyState title="Not recorded for this run">
          <p>The run was written before the per-feature IC was recorded.</p>
        </EmptyState>
      )}
      {strongest ? (
        <About
          label="About this table"
          handbook={{ page: 'alpha', section: 'Features' }}
          className={styles.inset}
        >
          The strongest feature here is {strongest.feature} (t{' '}
          {num(strongest.t_stat, { digits: 1 })}). The sign column is the direction the composite
          trades the feature; a negative IC with the sign applied means the feature worked against
          the book over the whole sample. ML-only features enter the trees unsigned.
        </About>
      ) : null}
    </Panel>
  );
}
