import { useParams } from 'react-router';
import { HANDBOOK_SIGNALS, useSignalTest } from '@/entities/research';
import { Facts, Stat, layout } from '@/features/research-launch';
import {
  BarChart,
  BoxSwatch,
  ChartFrame,
  GroupedBarChart,
  LegendItem,
  MultiPointLineChart,
} from '@/shared/charts';
import { bps, date, int, num, pct, sharpe } from '@/shared/lib/format';
import { About, Badge, ErrorState, Page, Panel, Skeleton, Table, Td, Th } from '@/shared/ui';
import type { SignalTestDetail } from '@/shared/api';
import {
  coverageBars,
  decayPoints,
  foldBars,
  horizonRows,
  topCorrelations,
  verdict,
  yearBars,
} from './lib/shape';
import { TMeter } from './TMeter';
import styles from './SignalTest.module.css';

const ic = (v: number) => num(v, { digits: 3, signed: true });

export function SignalTestPage() {
  const { testId = '' } = useParams();
  const q = useSignalTest(testId);
  return (
    <Page
      docTitle={q.data ? `Signal test: ${q.data.result.signal.name}` : 'Signal test'}
      title={q.data ? `Signal test: ${q.data.result.signal.name}` : 'Signal test'}
    >
      {q.isPending ? (
        <Skeleton height="24rem" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : (
        <Body d={q.data} />
      )}
    </Page>
  );
}

function Body({ d }: { d: SignalTestDetail }) {
  const r = d.result;
  const now = d.family_now;
  const v = verdict(r, now);
  const rows = horizonRows(r);
  const decay = decayPoints(r);
  const years = yearBars(r);
  const folds = foldBars(r);
  const cov = coverageBars(r);
  const corr = topCorrelations(r);
  const inc = r.incremental;
  return (
    <>
      <p className={layout.lede}>
        {r.signal.description.replace(/\.?\s*$/, '.')} Theme {r.signal.theme}, sign{' '}
        {r.signal.sign > 0 ? '+' : '−'}, {r.signal.source} {r.signal.feature_set}. Window{' '}
        {date(r.window[0])} – {date(r.window[1])} (the baseline&apos;s out-of-sample dates, every{' '}
        {r.step} days), data through {date(r.baseline.sealed_end)}.
      </p>
      <section
        className={layout.context}
        aria-label="Multiple-testing context"
        data-testid="testing-context"
      >
        <span>
          <span className={layout.contextLabel}>
            <Stat k="tNw">Primary t</Stat> ({r.primary_horizon} d)
          </span>
          <br />
          <span className={layout.contextBig}>{num(r.trial.t, { signed: true })}</span>
        </span>
        <span>
          <span className={layout.contextLabel}>
            <Stat k="pHolm">Holm-adjusted p, now</Stat>
          </span>
          <br />
          <span className={layout.contextBig}>{num(now.p_holm, { digits: 3 })}</span>
          <br />
          <span className={layout.small}>
            at the test: {num(r.trial.p_holm, { digits: 3 })} of {int(r.trial.n_tests)}
          </span>
        </span>
        <span>
          <span className={layout.contextLabel}>Signals tested in the family</span>
          <br />
          <span className={layout.contextBig}>{int(now.n_tests)}</span>
          <br />
          <code className={layout.small}>{now.family}</code>
        </span>
        <span>
          <span className={layout.contextLabel}>
            <Stat k="hurdle">Hurdle</Stat>
          </span>
          <br />
          {r.trial.passes_hurdle ? (
            <Badge tone="pass">Clears |t| ≥ 3</Badge>
          ) : (
            <Badge tone="fail">Below |t| ≥ 3</Badge>
          )}
          {!r.trial.logged ? (
            <>
              {' '}
              <Badge tone="caution">Not logged: not counted</Badge>
            </>
          ) : null}
        </span>
      </section>
      <p className={styles.verdict} data-testid="signal-verdict">
        <Badge tone={v.tone}>
          {v.tone === 'pass' ? 'Evidence' : v.tone === 'caution' ? 'Not yet' : 'No evidence'}
        </Badge>
        <span>{v.text}</span>
      </p>
      <Panel title="The primary t against the hurdle">
        <TMeter t={r.trial.t} hurdle={r.trial.hurdle} />
      </Panel>
      <About label="About these statistics" handbook={HANDBOOK_SIGNALS}>
        The signal is processed as the model processes the feature (winsorised, demeaned within
        industry where the model does, rank-Gaussian) and compared with the residual return h days
        after the execution close. The t uses Newey–West errors with the lag set to the label
        overlap. The primary horizon is the baseline&apos;s, fixed before the test; the other
        horizons are reported, never picked from.
      </About>
      <div className={layout.row2}>
        <Panel title="Rank IC by horizon" subtitle="the signal and what it adds to the baseline">
          <ChartFrame
            title="Mean rank IC by label horizon"
            description="Bars: the signal's mean IC and its incremental IC (after removing what the baseline already knows)."
            legend={
              <>
                <LegendItem
                  swatch={<BoxSwatch color="var(--color-series-1)" />}
                  label="Signal IC"
                />
                <LegendItem
                  swatch={<BoxSwatch color="var(--color-series-3)" />}
                  label="Incremental IC"
                />
              </>
            }
            table={() => ({
              caption: 'Rank IC by horizon',
              columns: [
                'Horizon',
                'Mean IC',
                'NW t',
                'Lags',
                'Hit rate',
                'Dates',
                'Incremental IC',
                'Incremental t',
              ],
              rows: rows.map((x) => [
                `${x.horizon} d${x.primary ? ' (primary)' : ''}`,
                ic(x.ic?.mean_ic ?? NaN),
                num(x.ic?.t_nw),
                int(x.ic?.nw_lags),
                pct(x.ic?.hit, { digits: 0 }),
                int(x.ic?.n),
                ic(x.incremental?.mean_ic ?? NaN),
                num(x.incremental?.t_nw),
              ]),
            })}
          >
            <GroupedBarChart
              ariaLabel={`Mean rank IC by horizon: ${rows.map((x) => `${x.horizon} days ${ic(x.ic?.mean_ic ?? NaN)}`).join(', ')}`}
              groups={rows.map((x) => ({
                label: `${x.horizon} d`,
                values: [x.ic?.mean_ic ?? null, x.incremental?.mean_ic ?? null],
              }))}
              series={[
                { label: 'Signal IC', paletteIndex: 0 },
                { label: 'Incremental IC', paletteIndex: 2 },
              ]}
              format={ic}
            />
          </ChartFrame>
          <Table label="Rank IC by horizon">
            <thead>
              <tr>
                <Th>Horizon</Th>
                <Th numeric>
                  <Stat k="ic">Mean IC</Stat>
                </Th>
                <Th numeric>
                  <Stat k="tNw">NW t</Stat>
                </Th>
                <Th numeric>
                  <Stat k="hit">Hit</Stat>
                </Th>
                <Th numeric>Dates</Th>
                <Th numeric>
                  <Stat k="incremental">Incr. IC</Stat>
                </Th>
                <Th numeric>Incr. t</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((x) => (
                <tr key={x.horizon} data-primary={x.primary || undefined}>
                  <Td>
                    {x.horizon} d{x.primary ? <Badge tone="accent">primary</Badge> : null}
                  </Td>
                  <Td numeric>{ic(x.ic?.mean_ic ?? NaN)}</Td>
                  <Td numeric>{num(x.ic?.t_nw, { signed: true })}</Td>
                  <Td numeric>{pct(x.ic?.hit, { digits: 0 })}</Td>
                  <Td numeric>{int(x.ic?.n)}</Td>
                  <Td numeric>{ic(x.incremental?.mean_ic ?? NaN)}</Td>
                  <Td numeric>{num(x.incremental?.t_nw, { signed: true })}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Panel>
        <Panel
          title="Decay"
          subtitle={`IC of the signal lagged k days against the ${r.primary_horizon}-day label`}
        >
          <ChartFrame
            title="IC decay with signal age"
            description={`How fast the signal goes stale. Rank-autocorrelation half-life ${num(r.decay.half_life_days, { digits: 0 })} days.`}
            table={() => ({
              caption: 'IC by signal lag',
              columns: ['Lag (days)', 'Mean IC'],
              rows: decay.labels.map((l, i) => [l, ic(decay.values[i] ?? NaN)]),
            })}
          >
            <MultiPointLineChart
              ariaLabel={`IC by signal lag: ${decay.labels.map((l, i) => `${l} days ${ic(decay.values[i] ?? NaN)}`).join(', ')}`}
              labels={decay.labels}
              series={[{ label: 'Mean IC', paletteIndex: 0, values: decay.values }]}
              format={ic}
              xLabel="Signal lag (trading days)"
              hint="Hover a lag for its IC"
            />
          </ChartFrame>
          <Facts
            label="Decay and turnover"
            facts={[
              {
                key: 'hl',
                label: <Stat k="halfLife" />,
                value: `${num(r.decay.half_life_days, { digits: 0 })} d`,
              },
              {
                key: 'to',
                label: <Stat k="turnover">Paper-book turnover</Stat>,
                value: `${pct(r.paper.turnover_per_day, { digits: 1 })}/day`,
              },
              {
                key: 'be',
                label: <Stat k="breakeven" />,
                value: bps(r.paper.breakeven_cost_bps),
                note: `charged ${bps(r.paper.cost_bps_per_side)} per side`,
              },
            ]}
          />
        </Panel>
      </div>
      <div className={layout.row2}>
        <Panel title="IC by year" subtitle={`primary horizon, ${r.primary_horizon} d`}>
          <ChartFrame
            title="Mean rank IC by calendar year"
            table={() => ({
              caption: 'IC by year',
              columns: ['Year', 'Mean IC', 'Dates'],
              rows: Object.entries(r.ic_by_year).map(([y, x]) => [
                y,
                ic(x.mean_ic ?? NaN),
                int(x.n),
              ]),
            })}
          >
            <BarChart
              ariaLabel={`IC by year: ${years.map((b) => `${b.label} ${ic(b.value)}`).join(', ')}`}
              bars={years}
              format={ic}
              valueLabel="Mean IC"
            />
          </ChartFrame>
        </Panel>
        <Panel title="IC by fold" subtitle="the purged walk-forward test blocks of the gates">
          <ChartFrame
            title="Mean rank IC by fold"
            table={() => ({
              caption: 'IC by fold',
              columns: ['Fold', 'Mean IC', 'Dates'],
              rows: r.ic_by_fold.map((f) => [
                `${f.start} – ${f.end}`,
                ic(f.mean_ic ?? NaN),
                int(f.n),
              ]),
            })}
          >
            <BarChart
              ariaLabel={`IC by fold: ${folds.map((b) => `${b.label} ${ic(b.value)}`).join(', ')}`}
              bars={folds}
              format={ic}
              valueLabel="Mean IC"
              tickFormat={(l) => l.slice(0, 7)}
            />
          </ChartFrame>
        </Panel>
      </div>
      <div className={layout.row2}>
        <Panel
          title="Value over the baseline"
          subtitle={`baseline + ${num(inc.weight, { digits: 2 })} × signal, weight fixed in advance`}
        >
          <Facts
            label="Incremental value"
            facts={[
              {
                key: 'resid',
                label: <Stat k="residShare" />,
                value: pct(inc.resid_var_share, { digits: 0 }),
                note: 'of the signal’s variance is new',
              },
              {
                key: 'base',
                label: 'Baseline book, net Sharpe',
                value: sharpe(inc.baseline_book.sharpe_net),
              },
              {
                key: 'blend',
                label: 'With the signal, net Sharpe',
                value: sharpe(inc.blend_book.sharpe_net),
              },
              {
                key: 'delta',
                label: <Stat k="deltaSharpe" />,
                value: sharpe(inc.delta_sharpe_net, { signed: true }),
                note: `NW t of the daily difference ${num(inc.delta_t_nw, { signed: true })}`,
              },
            ]}
          />
        </Panel>
        <Panel
          title="The signal's own paper book"
          subtitle="rank-weighted, industry and risk-style neutral, gross 2"
        >
          <Facts
            label="Paper book"
            facts={[
              { key: 'sg', label: 'Gross Sharpe', value: sharpe(r.paper.sharpe_gross) },
              {
                key: 'sn',
                label: <Stat k="paperSharpe">Net Sharpe</Stat>,
                value: sharpe(r.paper.sharpe_net),
              },
              { key: 'vol', label: 'Annual vol', value: pct(r.paper.ann_vol) },
              { key: 'days', label: 'Days', value: int(r.paper.n_days) },
            ]}
          />
          <p className={layout.lede}>
            A scale-free read of tradability, not a backtest: no optimizer, no ADV caps, no impact.
          </p>
        </Panel>
      </div>
      <div className={layout.row2}>
        <Panel
          title="Correlation with the baseline"
          subtitle="mean per-date Spearman; strongest first"
        >
          <Table label="Correlations with the baseline's features">
            <thead>
              <tr>
                <Th>Feature</Th>
                <Th numeric>ρ</Th>
              </tr>
            </thead>
            <tbody>
              {corr.map((c) => (
                <tr key={c.name}>
                  <Td>
                    <code>{c.name.replace(/^_/, '')}</code>
                  </Td>
                  <Td numeric>{num(c.rho, { signed: true })}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Panel>
        <Panel title="Coverage by year">
          <ChartFrame
            title="Share of eligible names with a value"
            description="Below 50% the IC describes the covered names only."
            table={() => ({
              caption: 'Coverage by year',
              columns: ['Year', 'Coverage'],
              rows: cov.map((b) => [b.label, pct(b.value, { digits: 0 })]),
            })}
          >
            <BarChart
              ariaLabel={`Coverage by year: ${cov.map((b) => `${b.label} ${pct(b.value, { digits: 0 })}`).join(', ')}`}
              bars={cov}
              format={(x) => pct(x, { digits: 0 })}
              valueLabel="Coverage"
            />
          </ChartFrame>
        </Panel>
      </div>
      <Panel
        title="Every test in the family"
        subtitle={`${now.n_tests} tests: Holm over all of them`}
      >
        <Table
          label="Tests in the family"
          caption="Holm's step-down adjustment over every logged test, as it is now."
        >
          <thead>
            <tr>
              <Th>Signal</Th>
              <Th numeric>t</Th>
              <Th numeric>p</Th>
              <Th numeric>Holm p</Th>
            </tr>
          </thead>
          <tbody>
            {now.tests.map((x, i) => (
              <tr key={`${x.id ?? ''}${i}`} data-current={x.id === r.id || undefined}>
                <Td>
                  <code>{x.signal}</code>{' '}
                  {x.id === r.id ? <Badge tone="accent">this test</Badge> : null}
                </Td>
                <Td numeric>{num(x.t, { signed: true })}</Td>
                <Td numeric>{num(x.p, { digits: 4 })}</Td>
                <Td numeric>{num(x.p_holm, { digits: 4 })}</Td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Panel>
      {r.notes.length ? (
        <Panel title="Notes">
          <ul className={layout.list}>
            {r.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </Panel>
      ) : null}
    </>
  );
}
