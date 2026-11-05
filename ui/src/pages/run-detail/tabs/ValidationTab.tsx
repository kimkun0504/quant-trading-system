import type { GateResult } from '@/shared/api';
import { gateDef, orderGates, STATUS_WORD } from '@/entities/gate';
import { metric, phaseSharpeBar } from '@/entities/metric';
import { metricContext } from '@/entities/run';
import { BarChart, ChartFrame } from '@/shared/charts';
import { deflatedSharpe } from '@/shared/lib/stats';
import { isFiniteNumber, num, sharpe } from '@/shared/lib/format';
import { Annotated, Flow, MarginNote, Section, SortableTable } from '@/shared/ui';
import type { Column } from '@/shared/ui';
import { MetricTable } from '@/widgets/MetricTable';
import { useRunOutlet } from '../context';
import styles from './Tabs.module.css';

function fmtGateValue(v: GateResult['value']): string {
  if (v === null) return 'n/a';
  if (typeof v === 'string') return v;
  return Math.abs(v) >= 10 ? num(v, { digits: 1 }) : num(v, { digits: 3 });
}

const GATE_COLUMNS: Column<GateResult>[] = [
  {
    key: 'name',
    header: 'Gate',
    headerText: 'Gate',
    rowHeader: true,
    render: (g) => (
      <span className={styles.gateName}>
        <span>
          {gateDef(g.name).label}
          {g.required ? ' (required)' : ''}
        </span>
        <span className={styles.gateTest}>{gateDef(g.name).test}</span>
      </span>
    ),
  },
  {
    key: 'status',
    header: 'Status',
    headerText: 'Status',
    render: (g) => (
      <span className={styles[`status${g.status}`]}>
        <svg width="10" height="10" aria-hidden="true">
          <rect x="0.5" y="0.5" width="9" height="9" rx="2" />
        </svg>
        {STATUS_WORD[g.status]}
      </span>
    ),
  },
  {
    key: 'value',
    header: 'Value',
    headerText: 'Value',
    align: 'end',
    render: (g) => fmtGateValue(g.value),
  },
  {
    key: 'threshold',
    header: 'Threshold',
    headerText: 'Threshold',
    render: (g) => gateDef(g.name).threshold,
  },
  {
    key: 'reason',
    header: 'Reason (from the pipeline)',
    headerText: 'Reason',
    render: (g) => <span className={styles.reason}>{g.reason}</span>,
  },
];

export function ValidationTab() {
  const { run } = useRunOutlet();
  const ctx = metricContext(run);
  const ic = run.ic_report;
  const inp = run.dsr_inputs;
  const positiveYears = ic.by_year.filter((y) => y.mean_ic >= 0).length;
  const psr1 = deflatedSharpe({
    sharpeAnnual: run.summary.sharpe_net,
    nObs: inp.n_obs,
    nTrials: 1,
    varTrialsAnnual: inp.var_trials_sr_annual,
    skew: inp.skew,
    kurtosis: inp.kurtosis,
  });
  const foldsOk = run.fold_sharpes.filter((f) => f.sharpe_net >= 0.05).length;

  return (
    <Flow>
      <Section
        id="gates"
        title="Hurdle gates"
        lede={
          <p>
            A checklist fixed before any result was seen. Each gate passes, fails, or is skipped
            when its input is missing; a skipped gate fails the run only if it is required.
          </p>
        }
        note={
          <MarginNote handbook={{ page: 'validation', section: '14. The hurdle gates' }}>
            Required gates: {run.hurdles.required.map((n) => gateDef(n).label).join(', ')}. Overall:{' '}
            {run.hurdles.overall === 'PASS' ? 'pass' : 'fail'} for phase {run.hurdles.phase}.
          </MarginNote>
        }
      >
        <SortableTable
          caption="Gates in the order the pipeline evaluates them"
          columns={GATE_COLUMNS}
          rows={orderGates(run.hurdles.gates)}
          rowKey={(g) => g.name}
        />
      </Section>

      <Section id="validation-metrics" title="Signal statistics">
        <MetricTable
          label="Signal statistics"
          ctx={ctx}
          rows={[
            { key: 'mean_ic', value: ic.mean_ic },
            { key: 'icir', value: ic.icir },
            { key: 'ic_t_stat', value: ic.t_stat },
            { key: 'decile_t', value: run.decile?.t_stat },
            { key: 'dsr', value: run.dsr },
            { key: 'pbo', value: run.pbo },
          ]}
        />
      </Section>

      <Section
        id="ic-by-year"
        title="IC by year"
        note={
          <MarginNote handbook={{ page: 'validation', section: '14. The hurdle gates' }}>
            The mean IC was positive in {positiveYears} of {ic.by_year.length} years. The IC-sign
            gate needs at least two thirds; with four or five yearly folds the rule is coarse, so
            read the bars, not just the count.
          </MarginNote>
        }
      >
        <ChartFrame
          title="Mean rank IC by calendar year"
          description={`Non-overlapping ${ic.horizon_days}-day formation dates; partial first and last years.`}
          table={{
            caption: 'Mean rank IC by year',
            columns: ['Year', 'Mean IC', 'Dates'],
            rows: ic.by_year.map((y) => [
              String(y.year),
              num(y.mean_ic, { digits: 4, signed: true }),
              y.n,
            ]),
          }}
        >
          <BarChart
            bars={ic.by_year.map((y) => ({ label: String(y.year), value: y.mean_ic }))}
            format={(v) => num(v, { digits: 3 })}
            valueLabel="Mean IC"
            showValues
            ariaLabel={`Mean rank IC by year: ${ic.by_year.map((y) => `${y.year} ${num(y.mean_ic, { digits: 3 })}`).join(', ')}.`}
          />
        </ChartFrame>
      </Section>

      <Section
        id="folds"
        title="Net Sharpe by fold"
        note={
          <MarginNote>
            {foldsOk} of {run.fold_sharpes.length} folds clear +0.05. Consistency across folds
            matters more than the average: one great year can carry a mediocre strategy.
          </MarginNote>
        }
      >
        <ChartFrame
          title="Net Sharpe by calendar-year fold"
          description="Years with at least 60 return days. Dashed line: the +0.05 fold bar."
          table={{
            caption: 'Net Sharpe by fold',
            columns: ['Fold', 'Net Sharpe'],
            rows: run.fold_sharpes.map((f) => [f.label, sharpe(f.sharpe_net, { signed: true })]),
          }}
        >
          <BarChart
            bars={run.fold_sharpes.map((f) => ({
              label: f.label,
              value: f.sharpe_net,
            }))}
            format={(v) => num(v, { digits: 2 })}
            valueLabel="Net Sharpe"
            showValues
            threshold={{ value: 0.05, label: '+0.05' }}
            ariaLabel={`Net Sharpe by fold: ${run.fold_sharpes.map((f) => `${f.label} ${sharpe(f.sharpe_net)}`).join(', ')}.`}
          />
        </ChartFrame>
      </Section>

      <Section id="dsr" title="Deflated Sharpe and the trial count">
        <Annotated
          note={
            <MarginNote handbook={metric('dsr').handbook}>
              {metric('dsr').interpret(run.dsr, ctx)}
            </MarginNote>
          }
        >
          <div className={styles.prose}>
            <p>
              Every configuration you try is another draw from a distribution of Sharpe ratios; the
              best of many draws is high by luck alone. The deflated Sharpe asks how likely it is
              that this run’s true Sharpe beats the best you would expect from{' '}
              {num(inp.n_trials, { digits: inp.n_trials % 1 ? 1 : 0 })} trials whose Sharpes varied
              by {num(inp.var_trials_sr_annual, { digits: 3 })} (annual variance), allowing for this
              run’s skew ({num(inp.skew, { digits: 2, signed: true })}) and kurtosis (
              {num(inp.kurtosis, { digits: 2 })}) over {inp.n_obs} days.
            </p>
            <p>
              With a single trial the same Sharpe of {sharpe(run.summary.sharpe_net)} would score{' '}
              {num(psr1, { digits: 3 })}; counting the trials brings it to{' '}
              {num(run.dsr, { digits: 3 })}. The gate needs more than 0.95.{' '}
              {psr1 <= 0.95
                ? 'Even one trial would not reach it: this history is too short for this Sharpe.'
                : run.dsr <= 0.95
                  ? 'The trials run so far are what holds it back — the price of having looked.'
                  : 'It clears the bar even after counting the trials.'}
            </p>
          </div>
          <dl className={styles.dl} aria-label="Deflated Sharpe inputs">
            <dt>Net Sharpe</dt>
            <dd>{sharpe(run.summary.sharpe_net)}</dd>
            <dt>Effective trials</dt>
            <dd>{num(inp.n_trials, { digits: 1 })}</dd>
            <dt>Cross-trial Sharpe variance</dt>
            <dd>{num(inp.var_trials_sr_annual, { digits: 3 })}</dd>
            <dt>Observations</dt>
            <dd>{inp.n_obs}</dd>
            <dt>Phase bar for net Sharpe</dt>
            <dd>{num(phaseSharpeBar(run.config.phase), { digits: 1 })}</dd>
            <dt>PBO</dt>
            <dd>{isFiniteNumber(run.pbo) ? num(run.pbo, { digits: 2 }) : 'skipped'}</dd>
          </dl>
        </Annotated>
      </Section>

      {run.notes.length ? (
        <Section id="notes" title="Pipeline notes">
          <ul className={styles.list}>
            {run.notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </Section>
      ) : null}
    </Flow>
  );
}
