import { Link, useParams } from 'react-router';
import type { NuResult, RiskCalibrationResult } from '@/shared/api';
import { HANDBOOK_RISK, useRisk } from '@/entities/research';
import { Facts, Stat, layout } from '@/features/research-launch';
import { ChartFrame, MultiPointLineChart } from '@/shared/charts';
import { date, int, num, pct, sharpe } from '@/shared/lib/format';
import { nuText, ratioLines } from './lib/ratio';
import {
  About,
  Badge,
  ErrorState,
  LinkButton,
  Page,
  Panel,
  Skeleton,
  Table,
  Td,
  Th,
} from '@/shared/ui';

export function RiskResultPage() {
  const { resultId = '' } = useParams();
  const q = useRisk(resultId);
  return (
    <Page docTitle="Risk calibration" title="Risk calibration">
      {q.isPending ? (
        <Skeleton height="24rem" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : (
        <Body r={q.data} />
      )}
    </Page>
  );
}

function Body({ r }: { r: RiskCalibrationResult }) {
  const { years, series } = ratioLines(r);
  const rec = r.recommended_nu_scale;
  const same = rec !== null && Math.abs(rec - r.current_nu_scale) < 1e-12;
  const byNu = (v: number | null): NuResult | undefined => r.results.find((x) => x.nu_scale === v);
  const recRow = byNu(rec);
  const curRow = byNu(r.current_nu_scale);
  return (
    <>
      <p className={layout.lede}>
        <code>{r.id}</code>: config <code>{r.config_hash.slice(0, 8)}</code>, data through{' '}
        {date(r.sealed_end)}. <Badge tone="skipped">Report only</Badge> logged as{' '}
        <code>{r.family}</code>.
      </p>
      <section className={layout.context} aria-label="Recommendation" data-testid="recommendation">
        <span>
          <span className={layout.contextLabel}>Current ν</span>
          <br />
          <span className={layout.contextBig}>{nuText(r.current_nu_scale)}</span>
          <br />
          <span className={layout.small}>ratio {num(curRow?.overall.ratio, { digits: 2 })}</span>
        </span>
        <span>
          <span className={layout.contextLabel}>Recommended ν</span>
          <br />
          <span className={layout.contextBig}>{rec === null ? 'none' : nuText(rec)}</span>
          <br />
          <span className={layout.small}>ratio {num(recRow?.overall.ratio, { digits: 2 })}</span>
        </span>
        <span className={layout.lede}>
          Rule: {r.rule}.{' '}
          {rec === null ? null : same ? (
            'The current value is already the best calibrated: nothing to change.'
          ) : (
            <>
              Adopting it is a new run, and that run is <strong>one logged trial</strong> in its
              pipeline family.
            </>
          )}
        </span>
        {rec !== null && !same ? (
          <span>
            <LinkButton
              to={`/builder?set=${encodeURIComponent(`risk.alignment_nu_scale=${String(rec)}`)}`}
              variant="secondary"
            >
              Open a run with ν = {nuText(rec)}
            </LinkButton>
          </span>
        ) : null}
      </section>
      <About label="How to read the report" handbook={HANDBOOK_RISK}>
        A ratio of 1 means the model predicted the book&apos;s volatility exactly. Above 1 the book
        was riskier than the model said, so it ran hotter than its target; below 1 it was more
        cautious than needed. A ν that is right in every year is worth more than one that is right
        on average.
      </About>
      <Panel
        title="Realised over ex-ante volatility by year"
        subtitle="one line per ν; 1 is calibrated"
      >
        <ChartFrame
          title="Realised / ex-ante vol by year"
          table={() => ({
            caption: 'Realised / ex-ante vol by year and ν',
            columns: ['Year', ...series.map((s) => s.label)],
            rows: years.map((y, i) => [y, ...series.map((s) => num(s.values[i], { digits: 2 }))]),
          })}
        >
          <MultiPointLineChart
            ariaLabel={`Realised over ex-ante vol by year for ${series.length} values of ν`}
            labels={years}
            series={series}
            format={(v) => num(v, { digits: 2 })}
            xLabel="Year"
            refLine={{ value: 1, label: 'calibrated' }}
            hint="Hover a year for every ν's ratio"
          />
        </ChartFrame>
      </Panel>
      <Panel title="Every ν, whole period">
        <Table label="Calibration by ν" caption="Over the invested days before the seal.">
          <thead>
            <tr>
              <Th numeric>
                <Stat k="nu">ν</Stat>
              </Th>
              <Th numeric>Realised vol</Th>
              <Th numeric>Ex-ante vol</Th>
              <Th numeric>
                <Stat k="volRatio">Ratio</Stat>
              </Th>
              <Th numeric>
                <Stat k="biasStat" />
              </Th>
              <Th numeric>
                <Stat k="netSharpe" />
              </Th>
              <Th>Run</Th>
            </tr>
          </thead>
          <tbody>
            {r.results.map((x) => (
              <tr key={x.nu_scale}>
                <Td numeric>
                  {nuText(x.nu_scale)}
                  {x.nu_scale === rec ? (
                    <>
                      {' '}
                      <Badge tone="accent">recommended</Badge>
                    </>
                  ) : null}
                  {x.nu_scale === r.current_nu_scale ? (
                    <>
                      {' '}
                      <Badge tone="neutral">current</Badge>
                    </>
                  ) : null}
                </Td>
                <Td numeric>{pct(x.overall.realised_vol)}</Td>
                <Td numeric>{pct(x.overall.exante_vol)}</Td>
                <Td numeric>{num(x.overall.ratio, { digits: 2 })}</Td>
                <Td numeric>{num(x.overall.bias_stat, { digits: 2 })}</Td>
                <Td numeric>{sharpe(x.sharpe_net, { signed: true })}</Td>
                <Td>
                  {x.run_id ? (
                    <Link to={`/runs/${encodeURIComponent(x.run_id)}`} className={layout.rowLink}>
                      {x.run_id.slice(0, 16)}
                    </Link>
                  ) : null}
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
        <Facts
          label="Days"
          facts={years.map((y) => ({
            key: y,
            label: y,
            value: `${int(r.results[0]?.by_year[y]?.n_days ?? null)} days`,
          }))}
        />
        {r.notes.length ? (
          <ul className={layout.list}>
            {r.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        ) : null}
      </Panel>
    </>
  );
}
