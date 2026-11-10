import type { GateResult, RunDetail } from '@/shared/api';
import { gateDef, orderGates, STATUS_WORD } from '@/entities/gate';
import { metric, phaseSharpeBar } from '@/entities/metric';
import { metricContext } from '@/entities/run';
import { deflatedSharpe } from '@/shared/lib/stats';
import { isFiniteNumber, num, sharpe } from '@/shared/lib/format';
import { About, Badge, Panel, SortableTable } from '@/shared/ui';
import type { BadgeTone, Column } from '@/shared/ui';
import { MetricTable } from '@/widgets/MetricTable';
import { DossierSection } from '../DossierSection';
import styles from './Sections.module.css';

const TONE: Record<GateResult['status'], BadgeTone> = {
  PASS: 'pass',
  FAIL: 'fail',
  skipped: 'skipped',
};

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
    header: 'Verdict',
    headerText: 'Verdict',
    render: (g) => <Badge tone={TONE[g.status]}>{STATUS_WORD[g.status]}</Badge>,
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
    render: (g) => <span className={styles.reason}>{g.reason || '—'}</span>,
  },
];

export function ValidationSection({ run }: { run: RunDetail }) {
  const ctx = metricContext(run);
  const inp = run.dsr_inputs;
  const psr1 = deflatedSharpe({
    sharpeAnnual: run.summary.sharpe_net,
    nObs: inp.n_obs,
    nTrials: 1,
    varTrialsAnnual: inp.var_trials_sr_annual,
    skew: inp.skew,
    kurtosis: inp.kurtosis,
  });
  const failed = run.hurdles.gates.filter((g) => g.status === 'FAIL');

  return (
    <DossierSection
      id="validation"
      lede={`Overall ${run.hurdles.overall === 'PASS' ? 'pass' : 'fail'} for phase ${run.hurdles.phase}. Required: ${run.hurdles.required.map((n) => gateDef(n).label).join(', ')}. ${
        failed.length
          ? `Failing: ${failed.map((g) => gateDef(g.name).label).join(', ')}.`
          : 'No gate fails.'
      } Skipped gates count only when required.`}
    >
      <Panel flush>
        <SortableTable
          caption="Gates in the order the pipeline evaluates them; a checklist fixed before any result was seen"
          columns={GATE_COLUMNS}
          rows={orderGates(run.hurdles.gates)}
          rowKey={(g) => g.name}
        />
      </Panel>

      <div className={styles.pair}>
        <Panel title="Deflated Sharpe" subtitle="the price of having looked">
          <div className={styles.prose}>
            <p>
              Every configuration tried is another draw from a distribution of Sharpe ratios; the
              best of many draws is high by luck alone. The deflated Sharpe asks how likely it is
              that this run’s true Sharpe beats the best expected from{' '}
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
                  ? 'The trials run so far are what holds it back.'
                  : 'It clears the bar even after counting the trials.'}
            </p>
          </div>
          <dl className={styles.facts} aria-label="Deflated Sharpe inputs">
            <dt>Net Sharpe</dt>
            <dd>{sharpe(run.summary.sharpe_net)}</dd>
            <dt>Effective trials</dt>
            <dd>{num(inp.n_trials, { digits: 1 })}</dd>
            <dt>Cross-trial Sharpe variance</dt>
            <dd>{num(inp.var_trials_sr_annual, { digits: 3 })}</dd>
            <dt>Skew / kurtosis</dt>
            <dd>
              {num(inp.skew, { digits: 2, signed: true })} / {num(inp.kurtosis, { digits: 2 })}
            </dd>
            <dt>Observations</dt>
            <dd>{inp.n_obs}</dd>
            <dt>Phase bar for net Sharpe</dt>
            <dd>{num(phaseSharpeBar(run.config.phase), { digits: 1 })}</dd>
            <dt>PBO</dt>
            <dd>{isFiniteNumber(run.pbo) ? num(run.pbo, { digits: 2 }) : 'skipped (phase 1)'}</dd>
          </dl>
          <About label="About the deflated Sharpe" handbook={metric('dsr').handbook}>
            {metric('dsr').interpret(run.dsr, ctx)}
          </About>
        </Panel>
        <div className={styles.stack}>
          <Panel flush>
            <MetricTable
              label="Validation statistics"
              ctx={ctx}
              rows={[
                { key: 'dsr', value: run.dsr },
                { key: 'pbo', value: run.pbo },
                { key: 'ic_t_stat', value: run.ic_report.t_stat },
                { key: 'decile_t', value: run.decile?.t_stat },
              ]}
            />
          </Panel>
          {run.notes.length ? (
            <Panel title="Pipeline notes">
              <ul className={styles.list}>
                {run.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </Panel>
          ) : null}
        </div>
      </div>
    </DossierSection>
  );
}
