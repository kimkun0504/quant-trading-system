import { useMemo } from 'react';
import type { RunDetail } from '@/shared/api';
import { metric } from '@/entities/metric';
import { metricContext, useRunExposures, useRunSeries } from '@/entities/run';
import { ChartFrame, LegendItem, LineSwatch } from '@/shared/charts';
import { TimeSeriesChart } from '@/shared/charts/TimeSeriesChart';
import { spansWhere } from '@/shared/charts/spans';
import { date, days, num, pct } from '@/shared/lib/format';
import { cx } from '@/shared/lib/cx';
import { About, EmptyState, ErrorState, Panel, Skeleton } from '@/shared/ui';
import { MetricTable } from '@/widgets/MetricTable';
import { DossierSection } from '../DossierSection';
import styles from './Sections.module.css';

const SERIES = ['gross_exposure', 'net_exposure', 'turnover'] as const;
/** Gross exposure below this share of NAV is an empty book (`INVESTED_GROSS_EPS`). */
const EMPTY_BOOK = 1e-4;

const pretty = (f: string) => {
  const s = f.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
};

function ExposureBars({
  rows,
}: {
  rows: readonly { factor: string; exposure: number; role: string }[];
}) {
  const max = Math.max(0.1, ...rows.map((r) => Math.abs(r.exposure)));
  return (
    <div className={styles.exposure} role="list" aria-label="Exposure snapshot">
      {rows.map((r) => {
        const w = (Math.abs(r.exposure) / max) * 50;
        return (
          <div key={r.factor} role="listitem" style={{ display: 'contents' }}>
            <span className={cx(r.role === 'alpha_theme' && styles.theme)}>{pretty(r.factor)}</span>
            <span className={styles.bar} aria-hidden="true">
              <span className={styles.zero} style={{ left: '50%' }} />
              <span
                className={r.exposure >= 0 ? styles.pos : styles.neg}
                style={{
                  left: r.exposure >= 0 ? '50%' : `${50 - w}%`,
                  width: `${w}%`,
                }}
              />
            </span>
            <span className={styles.val}>{num(r.exposure, { digits: 2, signed: true })}</span>
          </div>
        );
      })}
    </div>
  );
}

export function PortfolioSection({ run }: { run: RunDetail }) {
  const ctx = metricContext(run);
  const s = run.summary;
  const series = useRunSeries(run.id, SERIES);
  const exposures = useRunExposures(run.id);

  const derived = useMemo(() => {
    if (!series.data) return null;
    const { dates } = series.data;
    const gross = series.data.series.gross_exposure;
    const net = series.data.series.net_exposure;
    const turnover = series.data.series.turnover;
    const recorded = gross.some((v) => v !== null);
    const gaps = recorded
      ? spansWhere(dates.length, (i) => {
          const g = gross[i];
          return g !== null && g !== undefined && g <= EMPTY_BOOK;
        })
      : [];
    const emptyDays = gaps.reduce((a, g) => a + (g.to - g.from + 1), 0);
    return { dates, gross, net, turnover, recorded, gaps, emptyDays };
  }, [series.data]);

  const latest = exposures.data?.latest ?? null;

  return (
    <DossierSection
      id="portfolio"
      lede={`Average gross ${num(s.avg_gross_exposure, { digits: 2 })}× NAV, net ${pct(s.avg_net_exposure, { digits: 1, signed: true })}; ${pct(s.avg_daily_turnover, { digits: 2 })} of NAV traded a day, an effective holding period of ${days(s.effective_holding_days)}.`}
    >
      <Panel>
        {derived ? (
          <ChartFrame
            title="Gross and net exposure"
            description={
              derived.recorded
                ? `Post-trade book as a share of NAV, by decision date. ${
                    derived.gaps.length
                      ? `Shaded spans: ${derived.gaps.length} stretch${derived.gaps.length === 1 ? '' : 'es'} with no book (${derived.emptyDays} days, ${pct(1 - s.invested_day_frac, { digits: 1 })} of the run).`
                      : 'The book was invested on every decision day.'
                  }`
                : 'This run did not record its weights, so exposure over time is not available.'
            }
            legend={
              <>
                <LegendItem swatch={<LineSwatch color="var(--color-gain)" />} label="Gross" />
                <LegendItem swatch={<LineSwatch color="var(--color-loss)" />} label="Net" />
                {derived.gaps.length ? (
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
                ) : null}
              </>
            }
            table={() => ({
              caption: 'Gross and net exposure at month ends',
              columns: ['Date', 'Gross', 'Net'],
              rows: derived.dates
                .map((d, i) => ({ d, i }))
                .filter(({ d, i }) => derived.dates[i + 1]?.slice(0, 7) !== d.slice(0, 7))
                .map(({ d, i }) => [
                  date(d),
                  num(derived.gross[i], { digits: 3 }),
                  num(derived.net[i], { digits: 3, signed: true }),
                ]),
            })}
          >
            <TimeSeriesChart
              dates={derived.dates}
              series={[
                { label: 'Gross', values: derived.gross, color: 'gain' },
                { label: 'Net', values: derived.net, color: 'loss' },
              ]}
              refLines={[{ value: 0, color: 'graphite' }]}
              spans={derived.gaps}
              format={(v) => num(v, { digits: 2 })}
              ariaLabel={`Gross and net exposure over the backtest; ${derived.gaps.length} empty-book stretches shaded.`}
            />
          </ChartFrame>
        ) : series.isError ? (
          <ErrorState error={series.error} onRetry={() => void series.refetch()} />
        ) : (
          <Skeleton height="16rem" />
        )}
        <About
          label="About this chart"
          handbook={{ page: 'portfolio', section: 'Constraints and vol targeting' }}
        >
          Gross is Σ|w|, the size of the book in multiples of NAV; net is Σw, the market bet left
          after the dollar-neutral constraint. A stretch with no book means the optimiser found
          nothing worth its cost that day (or failed to solve): those days earn nothing, and the
          invested-day share in the header counts them.
        </About>
      </Panel>

      <div className={styles.pair}>
        <Panel>
          {derived ? (
            <ChartFrame
              title="Daily turnover"
              description="Two-sided, as a share of NAV, by decision date."
              table={() => ({
                caption: 'Turnover at month ends',
                columns: ['Date', 'Turnover'],
                rows: derived.dates
                  .map((d, i) => ({ d, i }))
                  .filter(({ d, i }) => derived.dates[i + 1]?.slice(0, 7) !== d.slice(0, 7))
                  .map(({ d, i }) => [date(d), pct(derived.turnover[i], { digits: 2 })]),
              })}
            >
              <TimeSeriesChart
                dates={derived.dates}
                series={[{ label: 'Turnover', values: derived.turnover, color: 'gain' }]}
                spans={derived.gaps}
                format={(v) => pct(v, { digits: 1 })}
                height={160}
                ariaLabel="Daily two-sided turnover as a share of NAV over the backtest."
              />
            </ChartFrame>
          ) : (
            <Skeleton height="12rem" />
          )}
        </Panel>
        <Panel title="Exposure snapshot" subtitle={latest ? `on ${date(latest.date)}` : undefined}>
          {latest ? (
            <>
              <ExposureBars rows={latest.rows} />
              <About
                label="About this snapshot"
                handbook={{ page: 'risk', section: 'Style factors' }}
              >
                Net book exposure to each style factor, in standard deviations of the factor. Alpha
                themes (bold) are where the book means to take risk; risk-only styles are held
                inside ±0.05 by the optimiser.
              </About>
            </>
          ) : exposures.isError ? (
            <ErrorState error={exposures.error} onRetry={() => void exposures.refetch()} />
          ) : exposures.data ? (
            <EmptyState title="No exposure history for this run">
              <p>The run was written before the pipeline stored the book’s factor exposures.</p>
            </EmptyState>
          ) : (
            <Skeleton height="12rem" />
          )}
        </Panel>
      </div>

      <Panel flush>
        <MetricTable
          label="Portfolio metrics"
          ctx={ctx}
          rows={[
            { key: 'avg_daily_turnover', value: s.avg_daily_turnover },
            { key: 'effective_holding_days', value: s.effective_holding_days },
            { key: 'transfer_coefficient_mean', value: s.transfer_coefficient_mean },
            { key: 'realized_vs_exante_vol_ratio', value: s.realized_vs_exante_vol_ratio },
            { key: 'effective_n_names', value: s.effective_n_names },
            { key: 'specific_var_share', value: s.specific_var_share },
            { key: 'exante_vol_at_gross2', value: s.exante_vol_at_gross2 },
          ]}
        />
      </Panel>
      <p className={styles.muted} style={{ fontSize: 'var(--text-sm)' }}>
        {metric('realized_vs_exante_vol_ratio').interpret(s.realized_vs_exante_vol_ratio, ctx)}
      </p>
    </DossierSection>
  );
}
