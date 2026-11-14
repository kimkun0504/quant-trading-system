import { useState } from 'react';
import { Link, useParams } from 'react-router';
import type { SweepDetail, SweepManifest, SweepMember } from '@/shared/api';
import { ApiError } from '@/shared/api';
import { HANDBOOK_SWEEPS, stampIso, useResumeSweep, useSweep } from '@/entities/research';
import {
  CommandLine,
  Facts,
  SelectField,
  Stat,
  TaskJobs,
  layout,
} from '@/features/research-launch';
import form from '@/features/research-launch/research.module.css';
import { ChartFrame, MultiPointLineChart } from '@/shared/charts';
import { currency, dateTime, int, num, pct, sharpe } from '@/shared/lib/format';
import {
  About,
  Badge,
  Button,
  ErrorState,
  Page,
  Panel,
  Skeleton,
  Table,
  Td,
  Th,
} from '@/shared/ui';
import {
  METRICS,
  SWEEP_TONE,
  formatMetric,
  metricLines,
  metricValue,
  overridesText,
} from './lib/grid';
import type { MetricKey } from './lib/grid';
import styles from './Sweeps.module.css';

export function SweepPage() {
  const { sweepId = '' } = useParams();
  const q = useSweep(sweepId);
  return (
    <Page docTitle={`Sweep ${sweepId}`} title={`Sweep ${sweepId}`}>
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

function Body({ d }: { d: SweepDetail }) {
  const s = d.sweep;
  const done = s.members.filter((m) => m.status === 'done');
  const counted = s.members.filter((m) => !m.reference).length;
  const resume = useResumeSweep(s.id);
  const [message, setMessage] = useState('');
  const best = done.reduce<SweepMember | null>(
    (b, m) =>
      (metricValue(m, 'sharpe_net') ?? -Infinity) >
      (b ? (metricValue(b, 'sharpe_net') ?? -Infinity) : -Infinity)
        ? m
        : b,
    null,
  );
  const doResume = async () => {
    try {
      const out = await resume.mutateAsync(undefined);
      setMessage(`Queued ${out.n_queued} ${out.n_queued === 1 ? 'job' : 'jobs'}.`);
    } catch (e) {
      setMessage(e instanceof ApiError ? e.detail : 'The sweep could not be resumed.');
    }
  };
  return (
    <>
      <p className={layout.lede}>
        {s.params
          .map((p) => `${p.key} = ${p.values.map(String).join(', ')}`)
          .join(s.mode === 'zip' ? ', paired with ' : ' × ')}{' '}
        on <code>{s.base_config}</code>; NAV {currency(s.nav)},{' '}
        {s.max_days ? `last ${s.max_days} days` : 'full history'},{' '}
        {s.use_ml ? 'with the ML sleeve' : 'composite only'}. Created{' '}
        {dateTime(stampIso(s.created_at))}.
      </p>
      <section
        className={layout.context}
        aria-label="Multiple testing"
        data-testid="testing-context"
      >
        <span>
          <span className={layout.contextLabel}>Status</span>
          <br />
          <Badge tone={SWEEP_TONE[s.status] ?? 'neutral'}>{s.status}</Badge>{' '}
          <span className={layout.small}>
            {done.length} of {s.n_members} members done
          </span>
        </span>
        <span>
          <span className={layout.contextLabel}>
            <Stat k="trials">Trials this sweep adds</Stat>
          </span>
          <br />
          <span className={layout.contextBig}>{int(counted)}</span>
        </span>
        <span>
          <span className={layout.contextLabel}>
            <Stat k="family" />
          </span>
          <br />
          <code>{s.family}</code>
        </span>
        <span>
          <span className={layout.contextLabel}>
            <Stat k="pbo" />
          </span>
          <br />
          <span className={layout.contextBig}>
            {s.pbo?.pbo === null || !s.pbo ? '…' : num(s.pbo.pbo, { digits: 2 })}
          </span>
        </span>
      </section>
      {d.resumable > 0 || d.active > 0 ? (
        <div className={layout.notice} role="status">
          {d.active > 0 ? (
            <span>
              {d.active} {d.active === 1 ? 'job is' : 'jobs are'} queued or running; this page
              updates as members finish.
            </span>
          ) : (
            <>
              <span>
                {d.resumable} {d.resumable === 1 ? 'member has' : 'members have'} not finished
                (stopped or failed). Resuming runs only those, then recomputes PBO; finished members
                are not re-run.
              </span>
              <Button
                size="sm"
                variant="primary"
                onClick={() => void doResume()}
                disabled={resume.isPending}
              >
                Resume the sweep
              </Button>
            </>
          )}
          {message ? <span>{message}</span> : null}
        </div>
      ) : null}
      <About label="How to read a sweep" handbook={HANDBOOK_SWEEPS}>
        Each member is a full backtest with one combination of the settings. Picking the best member
        is itself a search, so its Sharpe is biased up: the deflated Sharpe (DSR) and PBO say how
        much of the winner&apos;s edge survives that search. A sweep result is a reason to
        pre-register a choice, not the choice itself.
      </About>
      <MetricPanel s={s} />
      <div className={layout.row2}>
        <PboPanel s={s} />
        <Panel
          title="Effect on the family"
          subtitle="the deflated Sharpe of the family's best, before and after"
        >
          {s.impact ? (
            <Facts
              label="Trial impact"
              facts={[
                {
                  key: 'trials',
                  label: <Stat k="trials" />,
                  value: `${int(s.impact.n_trials_raw)} → ${int(s.impact.n_trials_raw + s.impact.n_new)}`,
                },
                {
                  key: 'exp',
                  label: <Stat k="expMaxSr" />,
                  value: `${sharpe(s.impact.exp_max_sr_before)} → ${sharpe(s.impact.exp_max_sr_after)}`,
                },
                {
                  key: 'dsr',
                  label: <Stat k="dsr">DSR of the best before</Stat>,
                  value: `${num(s.impact.dsr_best_before, { digits: 3 })} → ${num(s.impact.dsr_best_after, { digits: 3 })}`,
                },
                {
                  key: 'best',
                  label: 'Best net Sharpe in the sweep',
                  value: best ? sharpe(metricValue(best, 'sharpe_net')) : 'not yet',
                  note: best ? overridesText(best.overrides) : undefined,
                },
              ]}
            />
          ) : (
            <p className={layout.lede}>The plan recorded no trial impact.</p>
          )}
        </Panel>
      </div>
      <Panel title="Members" subtitle="every member, not just the best" flush>
        <MembersTable members={s.members} best={best?.index ?? null} />
      </Panel>
      <Panel title="Reproduce">
        <CommandLine command={d.command} />
        <TaskJobs
          tasks={['sweep_member', 'sweep_finalize']}
          subject={s.id}
          limit={14}
          title="This sweep's jobs"
        />
      </Panel>
    </>
  );
}

function MetricPanel({ s }: { s: SweepManifest }) {
  const [metric, setMetric] = useState<MetricKey>('sharpe_net');
  const lines = metricLines(s, metric);
  const fmt = (v: number) => formatMetric(metric, v);
  return (
    <Panel
      title={`${METRICS[metric].label} by ${lines.xKey || 'parameter'}`}
      subtitle="out-of-sample backtest of each member"
      actions={
        <div className={styles.metricPick}>
          <SelectField
            name="metric"
            label="Metric"
            value={metric}
            options={(Object.keys(METRICS) as MetricKey[]).map((k) => ({
              value: k,
              label: METRICS[k].label,
            }))}
            onChange={setMetric}
          />
        </div>
      }
    >
      <ChartFrame
        title={`${METRICS[metric].label} against ${lines.xKey}`}
        description={
          lines.series.length > 1 ? 'One line per value of the second setting.' : undefined
        }
        table={() => ({
          caption: `${METRICS[metric].label} by member`,
          columns: [lines.xKey, ...lines.series.map((x) => x.label)],
          rows: lines.labels.map((l, i) => [
            l,
            ...lines.series.map((x) => fmt(x.values[i] ?? NaN)),
          ]),
        })}
      >
        <MultiPointLineChart
          ariaLabel={`${METRICS[metric].label} by ${lines.xKey}: ${lines.labels.map((l, i) => `${l} ${lines.series.map((x) => fmt(x.values[i] ?? NaN)).join(' / ')}`).join(', ')}`}
          labels={lines.labels}
          series={lines.series.map((x, i) => ({ ...x, paletteIndex: i }))}
          format={fmt}
          xLabel={lines.xKey}
          refLine={METRICS[metric].kind === 'ratio' ? { value: 0, label: '0' } : undefined}
          hint="Hover a value for every member's result"
        />
      </ChartFrame>
      <p className={layout.lede}>
        Read the shape, not the peak: a smooth plateau says the setting is robust; a lone spike is
        what PBO warns about.
      </p>
    </Panel>
  );
}

function PboPanel({ s }: { s: SweepManifest }) {
  const p = s.pbo;
  const v = p?.pbo ?? null;
  return (
    <Panel
      title="Probability of backtest overfitting"
      subtitle="combinatorially symmetric cross-validation"
    >
      {!p ? (
        <p className={layout.lede}>
          Computed when every member has finished (the last job of the sweep).
        </p>
      ) : v === null ? (
        <p className={layout.lede}>Not computed: {p.reason ?? 'too few members or days'}.</p>
      ) : (
        <>
          <p className={styles.pboValue} data-testid="pbo">
            {num(v, { digits: 2 })}{' '}
            {v > 0.5 ? (
              <Badge tone="fail">Winner likely overfit</Badge>
            ) : v > 0.25 ? (
              <Badge tone="caution">Some overfitting</Badge>
            ) : (
              <Badge tone="pass">Winner holds up</Badge>
            )}
            {p.partial ? <Badge tone="caution">partial: not every member finished</Badge> : null}
          </p>
          <p className={layout.lede}>
            The history is cut into {int(p.n_blocks)} blocks and split in half every way (
            {int(p.n_combinations)} splits). In each split the member with the best Sharpe in one
            half is ranked in the other. PBO is the share of splits where it lands below the median:
            0.5 is what pure luck gives, lower is better.
          </p>
          <Facts
            label="PBO details"
            facts={[
              {
                key: 'loss',
                label: <Stat k="probOosLoss" />,
                value: pct(p.prob_oos_loss, { digits: 0 }),
              },
              {
                key: 'deg',
                label: <Stat k="degradation" />,
                value: num(p.degradation_slope, { signed: true }),
                note: `R² ${num(p.degradation_r2)}`,
              },
              {
                key: 'rank',
                label: 'Mean out-of-sample rank',
                value: pct(p.mean_oos_rank_pct, { digits: 0 }),
                note: '50% = no better than the median',
              },
              {
                key: 'n',
                label: 'Members × days',
                value: `${int(p.n_configs)} × ${int(p.n_days)}`,
              },
            ]}
          />
        </>
      )}
    </Panel>
  );
}

function MembersTable({ members, best }: { members: readonly SweepMember[]; best: number | null }) {
  const keys: MetricKey[] = [
    'sharpe_net',
    'sharpe_gross',
    'dsr',
    'avg_daily_turnover',
    'avg_cost_bps_per_side',
    'max_drawdown_net',
  ];
  return (
    <Table
      label="Sweep members"
      caption="DSR deflates each member's Sharpe by every trial in the family at the time it ran."
    >
      <thead>
        <tr>
          <Th numeric>#</Th>
          <Th>Overrides</Th>
          <Th>Status</Th>
          {keys.map((k) => (
            <Th key={k} numeric>
              {k === 'dsr' ? <Stat k="dsr">DSR</Stat> : METRICS[k].label}
            </Th>
          ))}
          <Th numeric>Trials</Th>
          <Th>Run</Th>
        </tr>
      </thead>
      <tbody>
        {members.map((m) => (
          <tr
            key={m.index}
            data-status={m.status}
            className={m.index === best ? styles.best : undefined}
          >
            <Td numeric>{m.index + 1}</Td>
            <Td>
              {overridesText(m.overrides)}
              {m.reference ? (
                <>
                  {' '}
                  <Badge tone="skipped">reference</Badge>
                </>
              ) : null}
              {m.index === best ? (
                <>
                  {' '}
                  <Badge tone="accent">best net Sharpe</Badge>
                </>
              ) : null}
            </Td>
            <Td>
              <Badge tone={SWEEP_TONE[m.status] ?? 'neutral'}>{m.status}</Badge>
              {m.error ? <span className={form.error}> {m.error}</span> : null}
            </Td>
            {keys.map((k) => (
              <Td key={k} numeric>
                {m.status === 'done' ? formatMetric(k, metricValue(m, k)) : ''}
              </Td>
            ))}
            <Td numeric>
              {m.status === 'done'
                ? int(typeof m.metrics.n_trials === 'number' ? m.metrics.n_trials : null)
                : ''}
            </Td>
            <Td>
              {m.run_id ? (
                <Link to={`/runs/${encodeURIComponent(m.run_id)}`} className={layout.rowLink}>
                  {m.run_id.slice(0, 16)}
                </Link>
              ) : null}
            </Td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
}
