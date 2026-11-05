import { useMemo, useState } from 'react';
import type { Holding } from '@/shared/api';
import { metric } from '@/entities/metric';
import { metricContext, useRunExposures, useRunHoldings, useRunSeries } from '@/entities/run';
import { ChartFrame, LegendItem, LineSwatch, SmallMultiples } from '@/shared/charts';
import { TimeSeriesChart } from '@/shared/charts/TimeSeriesChart';
import { annualizedVol } from '@/shared/lib/stats';
import { date, num, pct } from '@/shared/lib/format';
import { ErrorState, Flow, MarginNote, Section, Skeleton, SortableTable } from '@/shared/ui';
import type { Column } from '@/shared/ui';
import { useRunOutlet } from '../context';
import styles from './Tabs.module.css';

const VOL_SERIES = ['gross', 'exante_vol'] as const;
const RISK_BAND: [number, number] = [-0.05, 0.05];

const pretty = (f: string) => {
  const s = f.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
};

const HOLDING_COLUMNS: Column<Holding>[] = [
  {
    key: 'ticker',
    header: 'Ticker',
    headerText: 'Ticker',
    rowHeader: true,
    sortValue: (h) => h.ticker,
    render: (h) => h.ticker,
  },
  {
    key: 'weight',
    header: 'Weight',
    headerText: 'Weight',
    align: 'end',
    sortValue: (h) => h.weight,
    render: (h) => (
      <span className={h.weight >= 0 ? styles.gain : styles.loss}>
        {pct(h.weight, { digits: 2, signed: true })}
      </span>
    ),
  },
  {
    key: 'alpha',
    header: 'Expected return',
    headerText: 'Expected return',
    align: 'end',
    sortValue: (h) => h.alpha,
    render: (h) => pct(h.alpha, { digits: 2, signed: true }),
  },
  {
    key: 'svol',
    header: 'Specific vol',
    headerText: 'Specific vol',
    align: 'end',
    sortValue: (h) => h.specific_vol,
    render: (h) => pct(h.specific_vol, { digits: 0 }),
  },
  {
    key: 'industry',
    header: 'Industry',
    headerText: 'Industry',
    sortValue: (h) => h.industry,
    render: (h) => h.industry,
  },
];

export function RiskTab() {
  const { run } = useRunOutlet();
  const ctx = metricContext(run);
  const exposures = useRunExposures(run.id);
  const vol = useRunSeries(run.id, VOL_SERIES);
  const [holdingDate, setHoldingDate] = useState<string | undefined>(undefined);
  const holdings = useRunHoldings(run.id, holdingDate);

  const multiples = useMemo(() => {
    if (!exposures.data) return null;
    const { factors, roles, history } = exposures.data;
    return factors.map((f) => ({
      key: f,
      label: pretty(f),
      values: history.series[f] ?? [],
      emphasis: roles[f] === 'alpha_theme',
      ...(roles[f] === 'risk_only' ? { band: RISK_BAND } : {}),
    }));
  }, [exposures.data]);

  const volLines = useMemo(() => {
    if (!vol.data) return null;
    const g = vol.data.series.gross;
    const window = 63;
    const realised = g.map((_, i) => {
      if (i < window) return null;
      const slice = g.slice(i - window, i).filter((v): v is number => v !== null);
      return annualizedVol(slice);
    });
    return { dates: vol.data.dates, realised, exante: vol.data.series.exante_vol };
  }, [vol.data]);

  const net = holdings.data?.positions.reduce((a, h) => a + h.weight, 0) ?? 0;
  const grossW = holdings.data?.positions.reduce((a, h) => a + Math.abs(h.weight), 0) ?? 0;

  return (
    <Flow>
      <Section
        id="exposures"
        title="Factor exposures over time"
        note={
          <MarginNote handbook={{ page: 'risk', section: 'Style factors' }}>
            Alpha themes (dark lines) are where the book means to take risk. Risk-only styles should
            stay inside the shaded ±0.05 band the optimiser enforces; a line leaving it means a
            constraint was relaxed.
          </MarginNote>
        }
      >
        {multiples && exposures.data ? (
          <ChartFrame
            title="Net book exposure to each style factor"
            description="In standard deviations of the factor, one panel per factor on a shared scale."
            table={() => ({
              caption: `Exposures on ${date(exposures.data.latest.date)}`,
              columns: ['Factor', 'Role', 'Exposure'],
              rows: exposures.data.latest.rows.map((r) => [
                pretty(r.factor),
                r.role === 'alpha_theme' ? 'Alpha theme' : 'Risk only',
                num(r.exposure, { digits: 3, signed: true }),
              ]),
            })}
          >
            <SmallMultiples
              items={multiples}
              startLabel={date(exposures.data.history.dates[0])}
              endLabel={date(exposures.data.latest.date)}
              ariaLabel={`Factor exposures over time. Latest: ${exposures.data.latest.rows
                .map((r) => `${pretty(r.factor)} ${num(r.exposure, { digits: 2 })}`)
                .join(', ')}.`}
            />
          </ChartFrame>
        ) : exposures.isError ? (
          <ErrorState error={exposures.error} onRetry={() => void exposures.refetch()} />
        ) : (
          <Skeleton height="24rem" />
        )}
      </Section>

      <Section
        id="vol"
        title="Realised against forecast volatility"
        note={
          <MarginNote handbook={metric('realized_vs_exante_vol_ratio').handbook}>
            {metric('realized_vs_exante_vol_ratio').interpret(
              run.summary.realized_vs_exante_vol_ratio,
              ctx,
            )}
          </MarginNote>
        }
      >
        {volLines ? (
          <ChartFrame
            title="Realised 63-day volatility of gross returns and the risk model’s ex-ante forecast"
            description="Annualised. The forecast made at the close of t is compared with the return from t to t+1."
            legend={
              <>
                <LegendItem
                  swatch={<LineSwatch color="var(--color-gain)" />}
                  label="Realised (63 days)"
                />
                <LegendItem
                  swatch={<LineSwatch color="var(--color-graphite)" dashed />}
                  label="Ex-ante forecast"
                />
              </>
            }
            table={() => ({
              caption: 'Realised and ex-ante volatility at month ends',
              columns: ['Date', 'Realised', 'Ex-ante'],
              rows: volLines.dates
                .map((d, i) => ({ d, i }))
                .filter(
                  ({ d, i }) =>
                    volLines.realised[i] !== null &&
                    volLines.dates[i + 1]?.slice(0, 7) !== d.slice(0, 7),
                )
                .map(({ d, i }) => [date(d), pct(volLines.realised[i]), pct(volLines.exante[i])]),
            })}
          >
            <TimeSeriesChart
              dates={volLines.dates}
              series={[
                { label: 'Realised', values: volLines.realised, color: 'gain' },
                {
                  label: 'Ex-ante',
                  values: volLines.exante,
                  color: 'graphite',
                  dashed: true,
                },
              ]}
              format={(v) => pct(v, { digits: 1 })}
              ariaLabel="Realised 63-day volatility against the ex-ante forecast over the backtest."
            />
          </ChartFrame>
        ) : (
          <Skeleton height="16rem" />
        )}
      </Section>

      <Section
        id="holdings"
        title="Positions"
        note={
          holdings.data ? (
            <MarginNote>
              {holdings.data.positions.length} names, gross {num(grossW, { digits: 2 })}, net{' '}
              {pct(net, { digits: 1, signed: true })} of NAV. Expected return is the Grinold-scaled
              alpha the optimiser saw that day.
            </MarginNote>
          ) : null
        }
      >
        {holdings.data ? (
          <>
            <label className={styles.select}>
              Positions at the close of
              <select value={holdings.data.date} onChange={(e) => setHoldingDate(e.target.value)}>
                {[...holdings.data.available_dates].reverse().map((d) => (
                  <option key={d} value={d}>
                    {date(d)}
                  </option>
                ))}
              </select>
            </label>
            <SortableTable
              caption={`Post-trade book on ${date(holdings.data.date)}, largest long first. Sort by weight ascending for the largest shorts.`}
              columns={HOLDING_COLUMNS}
              rows={holdings.data.positions}
              rowKey={(h) => `${h.ticker}-${h.weight}`}
              limit={20}
            />
          </>
        ) : holdings.isError ? (
          <ErrorState error={holdings.error} />
        ) : (
          <Skeleton height="20rem" />
        )}
      </Section>
    </Flow>
  );
}
