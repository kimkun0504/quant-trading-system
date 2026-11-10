import { useEffect, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';
import type { GateResult, RunDetail } from '@/shared/api';
import { STATUS_WORD, buildVerdict, gateDef, isSuspicious, orderGates } from '@/entities/gate';
import { metric } from '@/entities/metric';
import { runDescriptor } from '@/entities/run';
import { toggleSelected, useSelection } from '@/features/selection';
import { bps, date, dateTime, int, num, pct, sharpe, shortHash } from '@/shared/lib/format';
import { cx } from '@/shared/lib/cx';
import {
  Badge,
  Button,
  Definition,
  HandbookLink,
  IconButton,
  LinkButton,
  Tooltip,
  VisuallyHidden,
  signClass,
} from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import styles from './DossierHeader.module.css';

function CopyHash({ hash }: { hash: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const t = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(t);
  }, [copied]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(hash);
      setCopied(true);
    } catch {
      /* clipboard unavailable (insecure context): the hash is still selectable text */
    }
  };
  return (
    <span className={styles.hash}>
      <code title={hash}>{shortHash(hash, 12)}</code>
      <IconButton
        size="sm"
        label={copied ? 'Copied' : 'Copy config hash'}
        onClick={() => void copy()}
      >
        {copied ? <Icons.Check /> : <Icons.Copy />}
      </IconButton>
      <span role="status" className={styles.status}>
        {copied ? 'Config hash copied' : ''}
      </span>
    </span>
  );
}

/**
 * The gates as ten short marks in pipeline order (solid: passed, hatched: failed,
 * dashed: skipped); each mark's tooltip carries the gate's name, verdict and reason.
 */
function GateRow({ gates, animate }: { gates: readonly GateResult[]; animate: boolean }) {
  return (
    <ol className={cx(styles.gates, animate && styles.animate)} aria-label="Hurdle gates">
      {orderGates(gates).map((g, i) => (
        <li key={g.name} data-status={g.status} style={{ '--i': i } as CSSProperties}>
          <Tooltip
            content={`${gateDef(g.name).label}: ${STATUS_WORD[g.status]}${g.required ? ' (required)' : ''}. ${g.reason || 'No reason recorded.'}`}
          >
            <a href="#validation" className={styles.gate}>
              <span className={styles.mark} aria-hidden="true" />
              <VisuallyHidden>
                {gateDef(g.name).label} {STATUS_WORD[g.status]}
                {g.required ? ' (required)' : ''}
              </VisuallyHidden>
            </a>
          </Tooltip>
        </li>
      ))}
    </ol>
  );
}

function Fact({
  label,
  definition,
  formula,
  children,
  className,
}: {
  label: string;
  definition: string;
  formula?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={styles.fact}>
      <dt>
        <Definition definition={definition} formula={formula}>
          {label}
        </Definition>
      </dt>
      <dd className={cx(styles.factValue, className)}>{children}</dd>
    </div>
  );
}

/**
 * Sticky top of the dossier: the run's name and provenance, the verdict, the headline
 * numbers as one line of facts, the gates as a row of marks, and the tray actions.
 */
export function DossierHeader({ run, animate }: { run: RunDetail; animate: boolean }) {
  const s = run.summary;
  const selection = useSelection();
  const inTray = selection.includes(run.id);
  const verdict = buildVerdict({
    status: run.status,
    hurdles: run.hurdles,
    sharpeNet: s.sharpe_net,
    nTrials: run.dsr_inputs.n_trials,
    kind: run.config.kind,
  });
  const suspicious = isSuspicious(s.sharpe_net);
  const trials = run.dsr_inputs.n_trials;

  return (
    <header className={styles.header} aria-label="Run summary">
      <div className={styles.row}>
        <div className={styles.identity}>
          <h1 className={styles.title}>{run.label}</h1>
          <p className={styles.meta}>
            <span>
              {runDescriptor({
                phase: run.config.phase,
                use_ml: run.config.use_ml,
                kind: run.config.kind,
              })}
              {run.config.stocks !== null ? `, ${int(run.config.stocks)} stocks` : ''}, seed{' '}
              {run.config.seed}, lag {run.config.signal_lag_days}
            </span>
            <span className={styles.sep} aria-hidden="true" />
            <span>{dateTime(run.manifest.created_at)}</span>
            <span className={styles.sep} aria-hidden="true" />
            <span>
              {date(s.start)} to {date(s.end)}, {int(s.n_days)} days
            </span>
            <span className={styles.sep} aria-hidden="true" />
            <span className={styles.hashLabel}>
              config <CopyHash hash={run.manifest.config_hash} />
            </span>
          </p>
        </div>
        <div className={styles.actions}>
          <Button
            variant={inTray ? 'secondary' : 'primary'}
            size="sm"
            aria-pressed={inTray}
            onClick={() => toggleSelected(run.id)}
          >
            {inTray ? 'Remove from tray' : 'Add to tray'}
          </Button>
          <LinkButton
            to="/compare"
            variant="secondary"
            className={cx(styles.compare, selection.length < 2 && styles.disabled)}
            aria-disabled={selection.length < 2}
            tabIndex={selection.length < 2 ? -1 : undefined}
          >
            Compare {selection.length >= 2 ? `(${selection.length})` : ''}
          </LinkButton>
        </div>
      </div>

      <p className={cx(styles.verdict, styles[verdict.tone])}>{verdict.sentence}</p>
      {suspicious ? (
        <p className={styles.banner} role="note" aria-label="Suspicious result">
          Suspicious result: net Sharpe {sharpe(s.sharpe_net)}. A net backtest Sharpe above 2 is a
          defect until leakage, survivorship, costs and selection are ruled out.
          {run.config.signal_lag_days === 0
            ? ' This run trades on the same close that produced its signal (signal lag 0).'
            : ''}{' '}
          <HandbookLink
            refTo={{
              page: 'backtesting',
              section: 'Backtest biases and how this engine guards each',
            }}
          >
            Reading a backtest critically
          </HandbookLink>
        </p>
      ) : null}

      <div className={styles.strip}>
        <div className={styles.gateCell}>
          <Badge tone={run.hurdles.overall === 'PASS' ? 'pass' : 'fail'}>
            {run.hurdles.overall === 'PASS' ? 'Gates pass' : 'Gates fail'}
          </Badge>
          <GateRow gates={run.hurdles.gates} animate={animate} />
        </div>
        <dl className={styles.facts} aria-label="Headline numbers">
          <Fact
            label="Net Sharpe"
            definition={metric('sharpe_net').definition}
            formula={metric('sharpe_net').formula}
            className={signClass(s.sharpe_net)}
          >
            {sharpe(s.sharpe_net, { signed: true })}
          </Fact>
          <Fact
            label="Gross Sharpe"
            definition={metric('sharpe_gross').definition}
            formula={metric('sharpe_gross').formula}
          >
            {sharpe(s.sharpe_gross, { signed: true })}
          </Fact>
          <Fact label="DSR" definition={metric('dsr').definition} formula={metric('dsr').formula}>
            {num(run.dsr, { digits: 3 })}
            <span className={styles.sub}>
              {' '}
              {num(trials, { digits: trials % 1 ? 1 : 0 })} trial{trials === 1 ? '' : 's'}
            </span>
          </Fact>
          <Fact
            label="Net CAGR"
            definition={metric('cagr_net').definition}
            formula={metric('cagr_net').formula}
            className={signClass(s.cagr_net)}
          >
            {pct(s.cagr_net, { signed: true })}
          </Fact>
          <Fact
            label="Max DD"
            definition={metric('max_drawdown_net').definition}
            formula={metric('max_drawdown_net').formula}
          >
            {pct(s.max_drawdown_net, { signed: true })}
          </Fact>
          <Fact
            label="Cost/side"
            definition={metric('avg_cost_bps_per_side').definition}
            formula={metric('avg_cost_bps_per_side').formula}
          >
            {bps(s.avg_cost_bps_per_side)}
          </Fact>
          <Fact
            label="Invested days"
            definition="Share of decision days on which the book held positions (gross exposure above 0.01% of NAV)."
          >
            {pct(s.invested_day_frac, { digits: 0 })}
          </Fact>
        </dl>
      </div>
    </header>
  );
}
