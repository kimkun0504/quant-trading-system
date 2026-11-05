import { Link } from 'react-router';
import type { RunDetail } from '@/shared/api';
import { buildVerdict, isSuspicious } from '@/entities/gate';
import { runDescriptor } from '@/entities/run';
import { date, dateTime, int, sharpe, shortHash } from '@/shared/lib/format';
import { cx } from '@/shared/lib/cx';
import { Annotated, HandbookLink } from '@/shared/ui';
import { GateStrip } from './GateStrip';
import styles from './RunHeader.module.css';

/**
 * The top of a run page: suspicious-result banner (first, when it applies), the run's
 * name, the verdict line in prose, the gate strip; the manifest sits in the margin.
 */
export function RunHeader({ run, animate = false }: { run: RunDetail; animate?: boolean }) {
  const s = run.summary;
  const verdict = buildVerdict({
    status: run.status,
    hurdles: run.hurdles,
    sharpeNet: s.sharpe_net,
    nTrials: run.dsr_inputs.n_trials,
    kind: run.config.kind,
  });
  const suspicious = isSuspicious(s.sharpe_net);

  return (
    <header className={styles.header}>
      {suspicious ? (
        <Annotated>
          <div className={styles.banner} role="note" aria-label="Suspicious result">
            <p className={styles.bannerTitle}>
              Suspicious result: net Sharpe {sharpe(s.sharpe_net)}
            </p>
            <p>
              A net backtest Sharpe above 2 is a defect until leakage, survivorship, costs and
              selection are ruled out.
              {run.config.signal_lag_days === 0
                ? ' This run trades on the same close that produced its signal (signal lag 0).'
                : ''}{' '}
              <HandbookLink
                refTo={{
                  page: 'backtesting',
                  section: 'Backtest biases and how this engine guards each',
                }}
              >
                Checklist for reading a backtest critically
              </HandbookLink>
            </p>
          </div>
        </Annotated>
      ) : null}
      <Annotated
        note={
          <dl className={styles.manifest} aria-label="Run manifest">
            <div>
              <dt>Run</dt>
              <dd>{run.id}</dd>
            </div>
            <div>
              <dt>Created</dt>
              <dd>{dateTime(run.manifest.created_at)}</dd>
            </div>
            <div>
              <dt>Period</dt>
              <dd>
                {date(s.start)} to {date(s.end)}, {int(s.n_days)} days
              </dd>
            </div>
            <div>
              <dt>Config</dt>
              <dd>{shortHash(run.manifest.config_hash, 16)}</dd>
            </div>
            <div>
              <dt>Data</dt>
              <dd>{shortHash(run.manifest.data_hash, 16)}</dd>
            </div>
            <div>
              <dt>Git</dt>
              <dd>
                {shortHash(run.manifest.git_sha, 10)}
                {run.manifest.git_dirty ? ' (dirty)' : ''}
              </dd>
            </div>
          </dl>
        }
      >
        <p className={styles.meta}>
          {runDescriptor({
            phase: run.config.phase,
            use_ml: run.config.use_ml,
            kind: run.config.kind,
          })}
          {run.config.stocks !== null ? `, ${int(run.config.stocks)} stocks` : ''}, seed{' '}
          {run.config.seed}, signal lag {run.config.signal_lag_days}
        </p>
        <h1 className={styles.title}>{run.label}</h1>
        <p className={cx(styles.verdict, styles[verdict.tone])}>{verdict.sentence}</p>
        {verdict.detail ? <p className={styles.detail}>{verdict.detail}</p> : null}
      </Annotated>
      <Annotated
        note={
          <p className={styles.gateNote}>
            {run.hurdles.overall === 'PASS' ? 'Overall: pass' : 'Overall: fail'} (phase{' '}
            {run.hurdles.phase}). Skipped gates count only when required.{' '}
            <Link to={`/runs/${encodeURIComponent(run.id)}/validation`}>
              Thresholds and reasons
            </Link>
          </p>
        }
      >
        <GateStrip gates={run.hurdles.gates} animate={animate} />
      </Annotated>
    </header>
  );
}
