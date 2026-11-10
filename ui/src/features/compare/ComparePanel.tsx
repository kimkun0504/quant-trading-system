import { useMemo, useState } from 'react';
import { Link } from 'react-router';
import type { GateName, RunCompare, RunCompareRun } from '@/shared/api';
import { GATE_ORDER, STATUS_WORD, gateDef } from '@/entities/gate';
import { configDiff, formatLeaf, useCompareRuns } from '@/entities/run';
import { GroupedBarChart } from '@/shared/charts';
import type { BarGroup } from '@/shared/charts';
import { ChartFrame } from '@/shared/charts/ChartFrame';
import { OverlayChart } from '@/shared/charts/OverlayChart';
import { currency, date, int, num, pct, sharpe } from '@/shared/lib/format';
import { cx } from '@/shared/lib/cx';
import {
  About,
  Badge,
  ErrorState,
  Panel,
  SegmentedControl,
  Skeleton,
  Table,
  Td,
  Th,
  Tooltip,
  signClass,
} from '@/shared/ui';
import type { BadgeTone } from '@/shared/ui';
import styles from './ComparePanel.module.css';

const GATE_TONE: Record<RunCompareRun['hurdles']['gates'][number]['status'], BadgeTone> = {
  PASS: 'pass',
  FAIL: 'fail',
  skipped: 'skipped',
};

function Swatch({ index }: { index: number }) {
  return (
    <span
      className={styles.swatch}
      style={{ background: `var(--color-series-${index + 1})` }}
      aria-hidden="true"
    />
  );
}

function Legend({ runs }: { runs: readonly RunCompareRun[] }) {
  return (
    <div className={styles.legend}>
      {runs.map((r, i) => (
        <span key={r.id} className={styles.legendItem}>
          <Swatch index={i} />
          {r.label}
        </span>
      ))}
    </div>
  );
}

/** Headline numbers of each run, one row per run, in selection order. */
function HeadlineTable({ runs }: { runs: readonly RunCompareRun[] }) {
  return (
    <table className={styles.headline} aria-label="Headline numbers">
      <thead>
        <tr>
          <th scope="col">Run</th>
          <th scope="col">Gates</th>
          <th scope="col" className={styles.num}>
            Net Sharpe
          </th>
          <th scope="col" className={styles.num}>
            DSR
          </th>
          <th scope="col" className={styles.num}>
            Trials
          </th>
          <th scope="col" className={styles.num}>
            Net CAGR
          </th>
          <th scope="col" className={styles.num}>
            Max DD
          </th>
          <th scope="col" className={styles.num}>
            Cost/side
          </th>
          <th scope="col">Period</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((r, i) => (
          <tr key={r.id}>
            <th scope="row" className={styles.rowHead}>
              <Link to={`/runs/${encodeURIComponent(r.id)}`} className={styles.runCell}>
                <Swatch index={i} />
                {r.label}
              </Link>
            </th>
            <td>
              <Badge tone={r.hurdles.overall === 'PASS' ? 'pass' : 'fail'}>
                {r.hurdles.overall === 'PASS' ? 'Pass' : 'Fail'}
              </Badge>
            </td>
            <td className={cx(styles.num, signClass(r.headline.sharpe_net))}>
              {sharpe(r.headline.sharpe_net, { signed: true })}
            </td>
            <td className={styles.num}>{num(r.headline.dsr, { digits: 3 })}</td>
            <td className={styles.num}>{int(r.n_trials)}</td>
            <td className={cx(styles.num, signClass(r.headline.cagr_net))}>
              {pct(r.headline.cagr_net, { signed: true })}
            </td>
            <td className={styles.num}>{pct(r.headline.max_drawdown, { signed: true })}</td>
            <td className={styles.num}>
              {num(r.headline.avg_cost_bps_per_side, { digits: 1 })} bp
            </td>
            <td>
              {date(r.start)} to {date(r.end)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function NavOverlay({ data }: { data: RunCompare }) {
  const [scale, setScale] = useState<'linear' | 'log'>('linear');
  const series = useMemo(
    () =>
      data.runs.map((r, i) => ({
        label: r.label,
        values: data.nav[r.id] ?? [],
        paletteIndex: i,
      })),
    [data],
  );
  return (
    <ChartFrame
      title="Net asset value"
      description={
        data.downsampled
          ? 'Daily, thinned to 2,000 points for drawing; the data table has the same points.'
          : 'Daily, after costs and borrow.'
      }
      legend={<Legend runs={data.runs} />}
      controls={
        <SegmentedControl
          legend="Scale"
          options={[
            { value: 'linear', label: 'Linear' },
            { value: 'log', label: 'Log' },
          ]}
          value={scale}
          onChange={setScale}
        />
      }
      table={() => ({
        caption: 'NAV by date',
        columns: ['Date', ...data.runs.map((r) => r.label)],
        rows: data.dates.map((d, i) => [
          d,
          ...data.runs.map((r) => {
            const v = data.nav[r.id]?.[i];
            return v === null || v === undefined ? 'n/a' : currency(v);
          }),
        ]),
      })}
    >
      <OverlayChart
        dates={data.dates}
        series={series}
        format={(v) => currency(v)}
        log={scale === 'log'}
        ariaLabel={`Net asset value of ${data.runs.length} runs from ${date(data.dates[0])} to ${date(data.dates[data.dates.length - 1])}`}
      />
    </ChartFrame>
  );
}

function FoldBars({ runs }: { runs: readonly RunCompareRun[] }) {
  const labels = useMemo(() => {
    const seen = new Set<string>();
    for (const r of runs) for (const f of r.fold_sharpes) seen.add(f.label);
    return [...seen].sort();
  }, [runs]);
  const groups: BarGroup[] = labels.map((label) => ({
    label,
    values: runs.map((r) => r.fold_sharpes.find((f) => f.label === label)?.sharpe_net ?? null),
  }));
  return (
    <ChartFrame
      title="Net Sharpe by fold"
      description="Purged walk-forward test blocks; the fold-Sharpe gate asks for ≥ +0.05 in two thirds of them."
      legend={<Legend runs={runs} />}
      table={{
        caption: 'Net Sharpe by fold and run',
        columns: ['Fold', ...runs.map((r) => r.label)],
        rows: groups.map((g) => [
          g.label,
          ...g.values.map((v) => (v === null ? 'n/a' : sharpe(v, { signed: true }))),
        ]),
      }}
    >
      <GroupedBarChart
        groups={groups}
        series={runs.map((r, i) => ({ label: r.label, paletteIndex: i }))}
        format={(v) => sharpe(v, { signed: true })}
        threshold={{ value: 0.05, label: '+0.05' }}
        tickFormat={(l) => l.slice(0, 7)}
        ariaLabel={`Net Sharpe of ${runs.length} runs across ${labels.length} folds (labels show the fold's first month)`}
      />
    </ChartFrame>
  );
}

function GatesGrid({ runs }: { runs: readonly RunCompareRun[] }) {
  const names = useMemo(() => {
    const present = new Set<GateName>();
    for (const r of runs) for (const g of r.hurdles.gates) present.add(g.name);
    return GATE_ORDER.filter((n) => present.has(n));
  }, [runs]);
  return (
    <Table
      label="Gates by run"
      caption="Each gate's verdict per run; hover a verdict for the reason."
    >
      <thead>
        <tr>
          <Th scope="col">Gate</Th>
          {runs.map((r, i) => (
            <Th scope="col" key={r.id}>
              <span className={styles.runCell}>
                <Swatch index={i} />
                {r.label}
              </span>
            </Th>
          ))}
        </tr>
      </thead>
      <tbody>
        {names.map((name) => (
          <tr key={name}>
            <Th scope="row" className={styles.rowHead}>
              <Tooltip content={gateDef(name).test} formula={gateDef(name).threshold}>
                <button type="button" className={styles.gateTerm}>
                  {gateDef(name).label}
                </button>
              </Tooltip>
            </Th>
            {runs.map((r) => {
              const g = r.hurdles.gates.find((x) => x.name === name);
              return (
                <Td key={r.id}>
                  {g ? (
                    <Tooltip content={g.reason || 'No reason recorded.'}>
                      <button type="button" className={styles.gateTerm}>
                        <Badge tone={GATE_TONE[g.status]}>{STATUS_WORD[g.status]}</Badge>
                      </button>
                    </Tooltip>
                  ) : (
                    <span className={styles.same}>—</span>
                  )}
                </Td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

function ConfigDiff({ runs }: { runs: readonly RunCompareRun[] }) {
  const rows = useMemo(() => configDiff(runs.map((r) => r.config_leaves)), [runs]);
  if (rows.length === 0)
    return (
      <p className={styles.same}>
        These runs share one configuration; any difference in results is seed, data or code.
      </p>
    );
  return (
    <Table
      label="Config diff"
      caption={`${rows.length} of ${Object.keys(runs[0]?.config_leaves ?? {}).length} keys differ.`}
    >
      <thead>
        <tr>
          <Th scope="col">Key</Th>
          {runs.map((r, i) => (
            <Th scope="col" key={r.id}>
              <span className={styles.runCell}>
                <Swatch index={i} />
                {r.label}
              </span>
            </Th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const decimal = row.values.some((v) => typeof v === 'number' && !Number.isInteger(v));
          return (
            <tr key={row.key}>
              <Th scope="row" className={styles.diffKey}>
                {row.key}
              </Th>
              {row.values.map((v, i) => (
                <Td key={i} className={styles.diffValue}>
                  <code>{formatLeaf(v, decimal)}</code>
                </Td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </Table>
  );
}

/**
 * Side by side: headline numbers, NAV overlaid, fold Sharpe as paired bars, the gates
 * grid and the config diff. Used inline on Experiments and as the Compare page.
 */
export function ComparePanel({ ids }: { ids: readonly string[] }) {
  const q = useCompareRuns(ids);
  if (q.isPending)
    return (
      <div className={styles.panel} aria-busy="true" aria-label="Loading comparison">
        <Skeleton height="6rem" />
        <Skeleton height="20rem" />
      </div>
    );
  if (q.isError) return <ErrorState error={q.error} onRetry={() => void q.refetch()} />;
  const data = q.data;
  return (
    <div className={styles.panel}>
      <Panel flush>
        <HeadlineTable runs={data.runs} />
      </Panel>
      <div className={styles.grid}>
        <Panel className={styles.full}>
          <NavOverlay data={data} />
          <About
            label="About this chart"
            handbook={{ page: 'backtesting', section: 'Reading a NAV curve' }}
          >
            All runs start from the same capital, so the vertical gap between two curves on a date
            is the cumulative difference in net return up to then. On the log scale a constant slope
            is a constant growth rate and the curves are directly comparable at any height.
          </About>
        </Panel>
        <Panel>
          <FoldBars runs={data.runs} />
          <About
            label="About this chart"
            handbook={{ page: 'validation', section: '14. The hurdle gates' }}
          >
            A change that helps in every fold is a change; one that helps in one fold and hurts in
            the rest is noise with a good year.
          </About>
        </Panel>
        <Panel title="Gates" flush className={cx(data.runs.length > 2 && styles.full)}>
          <GatesGrid runs={data.runs} />
        </Panel>
        <Panel
          title="Config diff"
          subtitle="only the keys that differ"
          flush
          className={styles.full}
        >
          <ConfigDiff runs={data.runs} />
        </Panel>
      </div>
    </div>
  );
}
