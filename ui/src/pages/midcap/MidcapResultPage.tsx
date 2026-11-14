import { Link, useParams } from 'react-router';
import type { MidcapCriterion, MidcapResultDetail } from '@/shared/api';
import { HANDBOOK_MIDCAP, useMidcapResult } from '@/entities/research';
import { Facts, Stat, layout } from '@/features/research-launch';
import { RunSealBadge } from '@/features/seal';
import { BarChart, ChartFrame } from '@/shared/charts';
import { bps, int, num, pct, sharpe, shortHash } from '@/shared/lib/format';
import { About, Badge, ErrorState, Page, Panel, Skeleton, Table, Td, Th } from '@/shared/ui';
import { CoverageFacts } from './CoverageFacts';
import { outcomeText, outcomeTone } from './verdict';
import styles from './Midcap.module.css';

export function MidcapResultPage() {
  const { runId = '' } = useParams();
  const q = useMidcapResult(runId);
  return (
    <Page docTitle="Mid-cap check result" title="Mid-cap check result">
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

function Mark({ c }: { c: MidcapCriterion }) {
  if (c.passed === true) return <Badge tone="pass">PASS</Badge>;
  if (c.passed === false) return <Badge tone="fail">FAIL</Badge>;
  return <Badge tone="skipped">not measurable</Badge>;
}

function CriteriaTable({ label, rows }: { label: string; rows: readonly MidcapCriterion[] }) {
  return (
    <Table label={label}>
      <thead>
        <tr>
          <Th>Criterion</Th>
          <Th>What it measures</Th>
          <Th numeric>Value</Th>
          <Th numeric>Bar</Th>
          <Th>Result</Th>
        </tr>
      </thead>
      <tbody>
        {rows.map((c) => (
          <tr key={c.id} data-criterion={c.id}>
            <Td>
              <strong>{c.id}</strong>
            </Td>
            <Td className={layout.wrapCell}>{c.name}</Td>
            <Td numeric>{c.value}</Td>
            <Td numeric>{c.threshold}</Td>
            <Td>
              <Mark c={c} />
            </Td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

function Body({ d }: { d: MidcapResultDetail }) {
  const r = d.result;
  const v = r.verdict;
  const sec = v?.secondary;
  const folds = Object.entries(sec?.sharpe_by_fold ?? {})
    .sort(([a], [b]) => a.localeCompare(b))
    .flatMap(([k, x]) =>
      typeof x === 'number' && Number.isFinite(x) ? [{ label: k, value: x }] : [],
    );
  const c = r.coverage;
  return (
    <>
      <section className={layout.context} aria-label="Verdict" data-testid="midcap-verdict">
        <span>
          <span className={layout.contextLabel}>Verdict</span>
          <br />
          <span className={layout.contextBig}>
            <Badge tone={outcomeTone(v?.outcome)}>
              {v?.outcome ?? (r.dry_run ? 'dry run' : 'none')}
            </Badge>
          </span>
          {r.forced ? (
            <>
              <br />
              <Badge tone="caution">forced repeat: not out-of-sample</Badge>
            </>
          ) : null}
        </span>
        <p className={styles.verdictText}>{outcomeText(v?.outcome)}</p>
        <span>
          <span className={layout.contextLabel}>Pre-registration</span>
          <br />
          {d.sha_matches ? (
            <Badge tone="pass">unchanged since the run</Badge>
          ) : (
            <Badge tone="fail">changed since the run</Badge>
          )}
          <br />
          <Link to="/checks/midcap" className={layout.rowLink}>
            <code>{r.preregistration}</code>
          </Link>
          <br />
          <code className={styles.sha} title={r.preregistration_sha256}>
            sha256 {shortHash(r.preregistration_sha256, 16)}
          </code>
        </span>
      </section>
      <About label="How the verdict is reached" handbook={HANDBOOK_MIDCAP}>
        V1–V3 decide whether the test could say anything (enough names, the data loaded, the
        optimiser behaved). If one fails, or a criterion cannot be measured (too few folds, say),
        the verdict is INCONCLUSIVE whatever the returns. Only then do C1–C3 decide PASS or FAIL. S1
        is reported and only adds the cost-fragile label.
      </About>
      {v ? (
        <>
          <Panel title="Was the test valid? (V1–V3)" subtitle="decided first">
            <CriteriaTable label="Validity conditions" rows={v.validity} />
          </Panel>
          <Panel title="Does the strategy work? (C1–C3)">
            <CriteriaTable label="Strategy criteria" rows={v.criteria} />
          </Panel>
        </>
      ) : null}
      {v ? (
        <Panel
          title="Mid-cap against the large-cap reference"
          subtitle="the same frozen configuration on two universes"
        >
          <Table
            label="Side by side"
            caption="The large-cap reference is the sealed baseline run the configuration was frozen from."
          >
            <thead>
              <tr>
                <Th>Metric</Th>
                <Th numeric>Mid-cap</Th>
                <Th numeric>Large-cap reference</Th>
              </tr>
            </thead>
            <tbody>
              {v.side_by_side.map((row) => (
                <tr key={row.metric}>
                  <Td>{row.metric}</Td>
                  <Td numeric>{row.midcap}</Td>
                  <Td numeric>{row.largecap}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
          <p className={layout.lede}>
            Mid-cap run{' '}
            {d.run ? (
              <>
                <Link to={`/runs/${encodeURIComponent(d.run.id)}`} className={layout.rowLink}>
                  {d.run.id}
                </Link>{' '}
                <RunSealBadge seal={d.run.seal} compact />
              </>
            ) : (
              <code>{v.run_id ?? 'not on this machine'}</code>
            )}
            ; reference{' '}
            {d.reference ? (
              <Link to={`/runs/${encodeURIComponent(d.reference.id)}`} className={layout.rowLink}>
                {d.reference.id}
              </Link>
            ) : (
              <code>{v.reference_run ?? 'not on this machine'}</code>
            )}
            .
          </p>
        </Panel>
      ) : null}
      <div className={layout.row2}>
        {sec ? (
          <Panel title="Secondary" subtitle="reported, not criteria (except S1)">
            <Facts
              label="Secondary results"
              facts={[
                {
                  key: 's1',
                  label: 'S1: net Sharpe at 2× cost',
                  value: sharpe(sec.net_sharpe_cost_x2, { signed: true }),
                  note: (sec.net_sharpe_cost_x2 ?? 0) > 0 ? 'not cost-fragile' : 'cost-fragile',
                },
                {
                  key: 'be',
                  label: <Stat k="breakeven" />,
                  value: bps(sec.breakeven_cost_bps),
                  note: `charged ${bps(sec.cost_bps_per_side)} per side`,
                },
                {
                  key: 'psr',
                  label: 'PSR',
                  value: num(sec.psr, { digits: 3 }),
                  note: 'probability the true Sharpe is above 0',
                },
                {
                  key: 'cap',
                  label: <Stat k="advCap" />,
                  value: pct(sec.traded_at_cap, { digits: 0 }),
                },
                {
                  key: 'univ',
                  label: 'Median universe',
                  value: `${int(sec.median_universe)} names`,
                },
              ]}
            />
          </Panel>
        ) : null}
        {folds.length ? (
          <Panel title="Net Sharpe by fold" subtitle="C3 counts the positive folds">
            <ChartFrame
              title="Net Sharpe by purged walk-forward fold"
              table={() => ({
                caption: 'Net Sharpe by fold',
                columns: ['Fold start', 'Net Sharpe'],
                rows: folds.map((f) => [f.label, sharpe(f.value)]),
              })}
            >
              <BarChart
                ariaLabel={`Net Sharpe by fold: ${folds.map((f) => `${f.label} ${sharpe(f.value)}`).join(', ')}`}
                bars={folds}
                format={(x) => sharpe(x, { signed: true })}
                valueLabel="Net Sharpe"
                threshold={{ value: 0, label: '0' }}
              />
            </ChartFrame>
          </Panel>
        ) : null}
      </div>
      <Panel
        title="Data coverage"
        subtitle={c ? `${c.prefilter}; ${int(c.n_dates)} dates` : 'not recorded'}
      >
        {c ? <CoverageFacts c={c} /> : null}
        {v?.notes.length ? (
          <ul className={layout.list}>
            {v.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        ) : null}
        <p className={layout.lede}>
          Family <code>{r.family}</code>; written to <code>{r.json_path}</code>.
        </p>
      </Panel>
    </>
  );
}
