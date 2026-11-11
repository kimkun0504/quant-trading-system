import { useMemo, useState } from 'react';
import type { Holding, RunExposures, RunHoldings } from '@/shared/api';
import {
  useManyRunHoldings,
  useManyRunSeries,
  useRunExposures,
  useRunHoldings,
  useRunSeries,
} from '@/entities/run';
import { LabPage, RunLanes } from '@/features/lab-runs';
import type { LabRun } from '@/features/lab-runs';
import {
  BoxSwatch,
  ChartFrame,
  LegendItem,
  LineSwatch,
  SmallMultiples,
  StackedAreaChart,
} from '@/shared/charts';
import type { AreaKey } from '@/shared/charts';
import { OverlayChart } from '@/shared/charts/OverlayChart';
import { TimeSeriesChart } from '@/shared/charts/TimeSeriesChart';
import { spansWhere } from '@/shared/charts/spans';
import { date, days, int, num, pct } from '@/shared/lib/format';
import { cx } from '@/shared/lib/cx';
import {
  About,
  EmptyState,
  ErrorState,
  Panel,
  Skeleton,
  SortableTable,
  signClass,
} from '@/shared/ui';
import type { Column } from '@/shared/ui';
import {
  EMPTY_BOOK,
  holdingPeriodSeries,
  industryStack,
  monthEndIndices,
  orderedFactors,
  realisedVsExante,
  snapshotTotals,
  stackDates,
  topPositions,
} from './lib/shape';
import styles from './PortfolioLabPage.module.css';

const WHAT =
  'What the book is made of and how it moves: exposure by industry and style over time, gross, net and turnover with empty-book gaps, holding period, realised against forecast risk, and the positions on any month end.';
const SERIES = ['gross_exposure', 'net_exposure', 'turnover', 'exante_vol', 'net'] as const;

export function PortfolioLabPage() {
  return (
    <LabPage title="Portfolio anatomy" docTitle="Portfolio" what={WHAT}>
      {(runs) => <PortfolioLab runs={runs} />}
    </LabPage>
  );
}

function PortfolioLab({ runs }: { runs: readonly LabRun[] }) {
  return (
    <>
      <RunLanes runs={runs} render={(r, solo) => <BookPanel run={r} solo={solo} />} />
      <RunLanes runs={runs} render={(r, solo) => <IndustryPanel run={r} solo={solo} />} />
      <RunLanes runs={runs} render={(r, solo) => <StylePanel run={r} solo={solo} />} />
      {runs.length === 1 ? (
        <div className={styles.pair}>
          <HoldingPeriodPanel runs={runs} />
          <RunLanes runs={runs} render={(r) => <VolPanel run={r} />} />
        </div>
      ) : (
        <>
          <HoldingPeriodPanel runs={runs} />
          <RunLanes runs={runs} render={(r) => <VolPanel run={r} />} />
        </>
      )}
      <RunLanes runs={runs} render={(r, solo) => <HoldingsPanel run={r} solo={solo} />} />
    </>
  );
}

const pretty = (f: string) => {
  const s = f.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
};

function useBook(id: string) {
  const q = useRunSeries(id, SERIES);
  const derived = useMemo(() => {
    if (!q.data) return null;
    const { dates, series } = q.data;
    const recorded = series.gross_exposure.some((v) => v !== null);
    const gaps = recorded
      ? spansWhere(dates.length, (i) => {
          const g = series.gross_exposure[i];
          return g !== null && g !== undefined && g <= EMPTY_BOOK;
        })
      : [];
    const emptyDays = gaps.reduce((a, g) => a + (g.to - g.from + 1), 0);
    return { dates, series, recorded, gaps, emptyDays };
  }, [q.data]);
  return { q, derived };
}

// ----------------------------------------------------------------------------- gross / net / turnover
function BookPanel({ run, solo }: { run: LabRun; solo: boolean }) {
  const { q, derived } = useBook(run.id);
  if (q.isError)
    return (
      <Panel>
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      </Panel>
    );
  if (!derived)
    return (
      <Panel>
        <Skeleton height="16rem" />
      </Panel>
    );
  const monthEnds = monthEndIndices(derived.dates);
  const gapLegend = derived.gaps.length ? (
    <LegendItem
      swatch={
        <span
          aria-hidden="true"
          style={{
            display: 'inline-block',
            width: '1.1rem',
            height: '0.7rem',
            background: 'var(--color-caution-wash)',
            border: '1px solid var(--color-caution)',
          }}
        />
      }
      label="No book"
    />
  ) : null;
  return (
    <Panel>
      <ChartFrame
        title="Gross, net and turnover"
        description={
          derived.recorded
            ? `Post-trade book as a share of NAV by decision date. ${
                derived.gaps.length
                  ? `${derived.gaps.length} stretch${derived.gaps.length === 1 ? '' : 'es'} with no book (${derived.emptyDays} days, ${pct(1 - (run.summary.headline?.invested_day_frac ?? 1), { digits: 1 })} of the run) shaded.`
                  : 'Invested on every decision day.'
              }`
            : 'This run did not record its weights.'
        }
        legend={
          <>
            <LegendItem swatch={<LineSwatch color="var(--color-gain)" />} label="Gross" />
            <LegendItem swatch={<LineSwatch color="var(--color-loss)" />} label="Net" />
            {gapLegend}
          </>
        }
        table={() => ({
          caption: 'Gross, net and turnover at month ends',
          columns: ['Date', 'Gross', 'Net', 'Turnover'],
          rows: monthEnds.map((i) => [
            date(derived.dates[i]),
            num(derived.series.gross_exposure[i], { digits: 3 }),
            num(derived.series.net_exposure[i], { digits: 3, signed: true }),
            pct(derived.series.turnover[i], { digits: 2 }),
          ]),
        })}
      >
        <div className={styles.stack}>
          <TimeSeriesChart
            dates={derived.dates}
            series={[
              { label: 'Gross', values: derived.series.gross_exposure, color: 'gain' },
              { label: 'Net', values: derived.series.net_exposure, color: 'loss' },
            ]}
            refLines={[{ value: 0, color: 'graphite' }]}
            spans={derived.gaps}
            format={(v) => num(v, { digits: 2 })}
            height={solo ? 200 : 150}
            ariaLabel={`Gross and net exposure of ${run.label} over the backtest; ${derived.gaps.length} empty-book stretches shaded.`}
          />
          <TimeSeriesChart
            dates={derived.dates}
            series={[{ label: 'Turnover', values: derived.series.turnover, color: 'graphite' }]}
            spans={derived.gaps}
            format={(v) => pct(v, { digits: 1 })}
            height={solo ? 130 : 110}
            ariaLabel={`Daily two-sided turnover of ${run.label} as a share of NAV.`}
          />
        </div>
      </ChartFrame>
      <About
        label="About this chart"
        handbook={{ page: 'portfolio', section: 'Constraints and vol targeting' }}
      >
        Gross is Σ|w|, net Σw, turnover Σ|trade|, all as fractions of NAV. A stretch with no book is
        a day the optimiser found nothing worth its cost or failed to solve; those days earn nothing
        and the invested-day share in the header counts them.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- industry
const INDUSTRY_COLOURS = Array.from({ length: 8 }, (_, i) => `var(--color-series-${i + 1})`);

function IndustryPanel({ run, solo }: { run: LabRun; solo: boolean }) {
  const latest = useRunHoldings(run.id, undefined);
  const dates = useMemo(
    () => (latest.data ? stackDates(latest.data.available_dates) : []),
    [latest.data],
  );
  const snaps = useManyRunHoldings(run.id, dates);
  const ready = latest.data && snaps.length > 0 && snaps.every((s) => s.data);
  const stack = useMemo(
    () =>
      ready ? industryStack(snaps.map((s) => s.data).filter((d): d is RunHoldings => !!d)) : null,
    [ready, snaps],
  );
  if (latest.isError)
    return (
      <Panel>
        <ErrorState error={latest.error} onRetry={() => void latest.refetch()} />
      </Panel>
    );
  if (!stack)
    return (
      <Panel>
        <Skeleton height="16rem" />
      </Panel>
    );
  const known = stack.industries.filter((k) => k !== 'Other');
  if (!known.length)
    return (
      <Panel title="Exposure by industry">
        <EmptyState title="No industries recorded">
          <p>
            The run was written before holdings carried an industry; the book over time is in the
            chart above.
          </p>
        </EmptyState>
      </Panel>
    );
  const keys: AreaKey<string>[] = [];
  stack.industries.forEach((ind, i) => {
    const color =
      ind === 'Other' ? 'var(--color-graphite)' : (INDUSTRY_COLOURS[i % 8] ?? 'var(--color-gain)');
    keys.push({ key: `${ind}|long`, label: ind, color, opacity: 0.9 });
    keys.push({ key: `${ind}|short`, label: `${ind} (short)`, color, opacity: 0.45 });
  });
  const series: Record<string, number[]> = {};
  for (const ind of stack.industries) {
    series[`${ind}|long`] = stack.long[ind] ?? [];
    series[`${ind}|short`] = stack.short[ind] ?? [];
  }
  const fmt = (v: number) => pct(v, { digits: 0, signed: true });
  return (
    <Panel>
      <ChartFrame
        title="Exposure by industry"
        description={`Long weight stacked above zero, short weight below, on ${stack.dates.length} month-end dates (the ${known.length} largest industries by gross weight; the rest are Other).`}
        legend={stack.industries.map((ind, i) => (
          <LegendItem
            key={ind}
            swatch={
              <BoxSwatch
                color={
                  ind === 'Other'
                    ? 'var(--color-graphite)'
                    : (INDUSTRY_COLOURS[i % 8] ?? 'var(--color-gain)')
                }
              />
            }
            label={ind}
          />
        ))}
        table={() => ({
          caption: 'Net weight by industry and month end',
          columns: ['Date', ...stack.industries],
          rows: stack.dates.map((d, i) => [
            date(d),
            ...stack.industries.map((ind) =>
              pct((stack.long[ind]?.[i] ?? 0) + (stack.short[ind]?.[i] ?? 0), {
                digits: 1,
                signed: true,
              }),
            ),
          ]),
        })}
      >
        <StackedAreaChart
          dates={stack.dates}
          series={series}
          keys={keys}
          format={fmt}
          height={solo ? 260 : 200}
          ariaLabel={`Long and short weight by industry over ${stack.dates.length} month ends for ${run.label}.`}
        />
      </ChartFrame>
      <About label="About this chart" handbook={{ page: 'portfolio', section: 'Constraints' }}>
        The optimiser holds each industry’s net weight inside a band, so the two sides of a bar
        should roughly mirror; a long stack without its short is an industry bet the risk model let
        through. Weights are the post-trade book at month ends, the dates the pipeline keeps.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- style
function StylePanel({ run, solo }: { run: LabRun; solo: boolean }) {
  const q = useRunExposures(run.id);
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
  if (!q.data.factors.length)
    return (
      <Panel title="Exposure by style">
        <EmptyState title="No exposure history for this run">
          <p>The run was written before the pipeline stored the book’s factor exposures.</p>
        </EmptyState>
      </Panel>
    );
  return <StyleCharts data={q.data} run={run} solo={solo} />;
}

function StyleCharts({ data, run, solo }: { data: RunExposures; run: LabRun; solo: boolean }) {
  const factors = useMemo(() => orderedFactors(data), [data]);
  const themes = factors.filter((f) => data.roles[f] === 'alpha_theme');
  const keys: AreaKey<string>[] = themes.map((f, i) => ({
    key: f,
    label: pretty(f),
    color: `var(--color-series-${(i % 8) + 1})`,
  }));
  const monthEnds = monthEndIndices(data.history.dates);
  const fmt = (v: number) => num(v, { digits: 2, signed: true });
  return (
    <Panel>
      <ChartFrame
        title="Exposure by style"
        description="Net exposure of the book to each alpha theme (standard deviations of the factor), positive tilts stacked up and negative down; every factor below on one scale, risk-only styles with their ±0.05 band."
        legend={keys.map((k) => (
          <LegendItem key={k.key} swatch={<BoxSwatch color={k.color} />} label={k.label} />
        ))}
        table={() => ({
          caption: 'Factor exposure at month ends',
          columns: ['Date', ...factors.map(pretty)],
          rows: monthEnds.map((i) => [
            date(data.history.dates[i]),
            ...factors.map((f) => fmt(data.history.series[f]?.[i] ?? 0)),
          ]),
        })}
      >
        <div className={styles.stack}>
          <StackedAreaChart
            dates={data.history.dates}
            series={Object.fromEntries(themes.map((f) => [f, data.history.series[f] ?? []]))}
            keys={keys}
            format={fmt}
            height={solo ? 220 : 170}
            ariaLabel={`Net exposure to the ${themes.length} alpha themes over time for ${run.label}, stacked.`}
          />
          <SmallMultiples
            items={factors.map((f) => ({
              key: f,
              label: pretty(f),
              values: data.history.series[f] ?? [],
              band: data.roles[f] === 'risk_only' ? [-0.05, 0.05] : undefined,
              emphasis: data.roles[f] === 'alpha_theme',
            }))}
            startLabel={date(data.history.dates[0])}
            endLabel={date(data.history.dates[data.history.dates.length - 1])}
            ariaLabel={`Exposure to each of ${factors.length} factors over time, one small chart per factor.`}
          />
        </div>
      </ChartFrame>
      <About label="About this chart" handbook={{ page: 'risk', section: 'Style factors' }}>
        Alpha themes are where the book means to take risk, so their tilts should be sizeable and
        move with the theme weights; risk-only styles and the market are held inside ±0.05 by the
        optimiser, and a line that leaves its band marks a day the constraint was relaxed or the
        solve failed.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- holding period
function HoldingPeriodPanel({ runs }: { runs: readonly LabRun[] }) {
  const queries = useManyRunSeries(
    runs.map((r) => r.id),
    ['turnover', 'gross_exposure'] as const,
  );
  const ready = queries.every((q) => q.data);
  const error = queries.find((q) => q.isError);
  const shaped = useMemo(
    () =>
      queries.map((q) =>
        q.data
          ? {
              dates: q.data.dates,
              hp: holdingPeriodSeries(q.data.series.turnover, q.data.series.gross_exposure),
            }
          : null,
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
  const series = runs.map((r, k) => {
    const s = shaped[k];
    const byDate = new Map(s?.dates.map((d, i) => [d, s.hp[i] ?? null]));
    return {
      label: r.label,
      paletteIndex: r.index,
      values: union.map((d) => byDate.get(d) ?? null),
    };
  });
  return (
    <Panel>
      <ChartFrame
        title="Effective holding period"
        description="Trailing 63-day mean gross over mean one-sided turnover, in days; null while the book is empty."
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
          caption: 'Effective holding period at month ends',
          columns: ['Date', ...runs.map((r) => r.label)],
          rows: monthEndIndices(union).map((i) => [
            date(union[i]),
            ...series.map((s) => days(s.values[i])),
          ]),
        })}
      >
        <OverlayChart
          dates={union}
          series={series}
          format={(v) => days(v)}
          height={220}
          ariaLabel={`Effective holding period over time of ${runs.length} run${runs.length === 1 ? '' : 's'}.`}
        />
      </ChartFrame>
      <About
        label="About this chart"
        handbook={{ page: 'portfolio', section: 'Transfer coefficient and trade aversion' }}
      >
        Holding period is the inverse of turnover: a 20-day alpha held for 50 days is traded slowly
        on purpose (trade aversion, Gârleanu–Pedersen shrink). The whole-run figure is in each
        dossier’s Portfolio section; the series shows when it changed, and a run with trade aversion
        off should sit well below one with it on.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- realised vs ex-ante
function VolPanel({ run }: { run: LabRun }) {
  const { q, derived } = useBook(run.id);
  const vol = useMemo(
    () => (derived ? realisedVsExante(derived.series.net, derived.series.exante_vol, 63) : null),
    [derived],
  );
  if (q.isError)
    return (
      <Panel>
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      </Panel>
    );
  if (!derived || !vol)
    return (
      <Panel>
        <Skeleton height="14rem" />
      </Panel>
    );
  const hasExante = vol.exante.some((v) => v !== null);
  const monthEnds = monthEndIndices(derived.dates);
  return (
    <Panel>
      <ChartFrame
        title="Realised against forecast risk"
        description={
          hasExante
            ? 'Trailing 63-day realised vol of the net return against the risk model’s ex-ante vol of the book, annualised.'
            : 'This run did not record its ex-ante vol; only the realised series is drawn.'
        }
        legend={
          <>
            <LegendItem swatch={<LineSwatch color="var(--color-gain)" />} label="Realised (63 d)" />
            {hasExante ? (
              <LegendItem
                swatch={<LineSwatch color="var(--color-loss)" dashed />}
                label="Ex-ante"
              />
            ) : null}
          </>
        }
        table={() => ({
          caption: 'Realised and ex-ante vol at month ends',
          columns: ['Date', 'Realised', 'Ex-ante'],
          rows: monthEnds.map((i) => [
            date(derived.dates[i]),
            pct(vol.realised[i], { digits: 1 }),
            pct(vol.exante[i], { digits: 1 }),
          ]),
        })}
      >
        <TimeSeriesChart
          dates={derived.dates}
          series={[
            { label: 'Realised', values: vol.realised, color: 'gain' },
            ...(hasExante
              ? [{ label: 'Ex-ante', values: vol.exante, color: 'loss' as const, dashed: true }]
              : []),
          ]}
          spans={derived.gaps}
          format={(v) => pct(v, { digits: 1 })}
          height={220}
          ariaLabel={`Realised 63-day volatility against the ex-ante forecast for ${run.label}.`}
        />
      </ChartFrame>
      <About
        label="About this chart"
        handbook={{ page: 'risk', section: 'Volatility regime adjustment' }}
      >
        A ratio near one over the invested days means the covariance model sizes the book correctly;
        realised persistently above forecast is a model that underestimates risk (and a vol target
        that is not being met). The whole-run ratio over invested days is in the dossier’s Portfolio
        metrics.
      </About>
    </Panel>
  );
}

// ----------------------------------------------------------------------------- holdings snapshot
const HOLDING_COLUMNS: Column<Holding>[] = [
  {
    key: 't',
    header: 'Ticker',
    headerText: 'Ticker',
    rowHeader: true,
    render: (r) => r.ticker,
    sortValue: (r) => r.ticker,
  },
  {
    key: 'i',
    header: 'Industry',
    headerText: 'Industry',
    render: (r) => r.industry ?? '—',
    sortValue: (r) => r.industry ?? '',
  },
  {
    key: 'w',
    header: 'Weight',
    headerText: 'Weight',
    align: 'end',
    render: (r) => (
      <span className={signClass(r.weight)}>{pct(r.weight, { digits: 2, signed: true })}</span>
    ),
    sortValue: (r) => r.weight,
  },
  {
    key: 'a',
    header: 'Alpha',
    headerText: 'Expected return over the horizon',
    align: 'end',
    render: (r) => (r.alpha === null ? '—' : pct(r.alpha, { digits: 2, signed: true })),
    sortValue: (r) => r.alpha,
  },
  {
    key: 'v',
    header: 'Specific vol',
    headerText: 'Specific vol',
    align: 'end',
    render: (r) => (r.specific_vol === null ? '—' : pct(r.specific_vol, { digits: 0 })),
    sortValue: (r) => r.specific_vol,
  },
];

function HoldingsPanel({ run, solo }: { run: LabRun; solo: boolean }) {
  const [chosen, setChosen] = useState<string | undefined>(undefined);
  const q = useRunHoldings(run.id, chosen);
  if (q.isError)
    return (
      <Panel>
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      </Panel>
    );
  if (!q.data)
    return (
      <Panel>
        <Skeleton height="20rem" />
      </Panel>
    );
  const data = q.data;
  const idx = Math.max(0, data.available_dates.indexOf(data.date));
  const totals = snapshotTotals(data.positions);
  const top = topPositions(data.positions, solo ? 10 : 5);
  const scrubId = `scrub-${run.id}`;
  return (
    <Panel title="Holdings" subtitle={`on ${date(data.date)}`}>
      <div className={styles.scrubber}>
        <label htmlFor={scrubId}>Month end</label>
        <input
          id={scrubId}
          type="range"
          min={0}
          max={Math.max(0, data.available_dates.length - 1)}
          value={idx}
          onChange={(e) => setChosen(data.available_dates[Number(e.target.value)])}
          aria-valuetext={date(data.date)}
          list={`${scrubId}-ticks`}
        />
        <select
          aria-label="Holdings date"
          value={data.date}
          onChange={(e) => setChosen(e.target.value)}
        >
          {data.available_dates.map((d) => (
            <option key={d} value={d}>
              {date(d)}
            </option>
          ))}
        </select>
        <span className={styles.totals}>
          {int(totals.n)} names ({int(totals.nLong)} long, {int(totals.nShort)} short), gross{' '}
          {num(totals.gross, { digits: 2 })}, net {pct(totals.net, { digits: 1, signed: true })}
        </span>
      </div>
      <div className={cx(styles.topGrid, !solo && styles.topGridNarrow)}>
        <TopList title="Largest longs" rows={top.longs} max={totals.maxAbs} />
        <TopList title="Largest shorts" rows={top.shorts} max={totals.maxAbs} />
      </div>
      <div className={styles.tableWrap}>
        <SortableTable
          caption={`Every position on ${date(data.date)}`}
          columns={HOLDING_COLUMNS}
          rows={data.positions}
          rowKey={(r) => r.ticker}
          initialSort={{ key: 'w', dir: 'descending' }}
          limit={solo ? 25 : 10}
        />
      </div>
      <About label="About this table" handbook={{ page: 'portfolio', section: 'Reading a book' }}>
        Alpha is the expected return the optimiser saw for the name (Grinold-scaled, over the alpha
        horizon) as of the signal date the book was built from; specific vol and industry are as of
        the holding date. A large weight on a small alpha is a position held for its risk
        contribution or to save a trade, not for its signal.
      </About>
    </Panel>
  );
}

function TopList({ title, rows, max }: { title: string; rows: readonly Holding[]; max: number }) {
  return (
    <div>
      <h3 className={styles.topTitle}>{title}</h3>
      {rows.length ? (
        <ol className={styles.topList}>
          {rows.map((p) => {
            const w = (Math.abs(p.weight) / Math.max(1e-9, max)) * 100;
            return (
              <li key={p.ticker} className={styles.topRow}>
                <span className={styles.topTicker}>{p.ticker}</span>
                <span className={styles.topBar} aria-hidden="true">
                  <span
                    className={p.weight >= 0 ? styles.pos : styles.neg}
                    style={{ width: `${w}%` }}
                  />
                </span>
                <span className={cx(styles.topVal, signClass(p.weight))}>
                  {pct(p.weight, { digits: 2, signed: true })}
                </span>
                <span className={styles.topInd}>{p.industry ?? ''}</span>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className={styles.muted}>None.</p>
      )}
    </div>
  );
}
