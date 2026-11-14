import type { ReactNode } from 'react';
import type { TrialImpact } from '@/shared/api';
import { int, num, sharpe } from '@/shared/lib/format';
import { Badge } from '@/shared/ui';
import { Stat } from './Stat';
import styles from './research.module.css';

/**
 * What an action costs in trials, shown before it is confirmed: how many trials it adds,
 * to which family, and what that does to the family's current best trial's deflated
 * Sharpe ratio (the numbers come from `qts.research.sweep.trial_impact`, via the API;
 * nothing is computed here).
 */
export function TrialCostSlip({
  impact,
  adds,
  references = 0,
  title = 'Before you confirm',
  children,
}: {
  impact: TrialImpact;
  /** New trials (members that are not reference re-runs). */
  adds: number;
  references?: number;
  title?: string;
  children?: ReactNode;
}) {
  const cut =
    impact.dsr_best_before !== null && impact.dsr_best_after !== null
      ? impact.dsr_best_before - impact.dsr_best_after
      : null;
  return (
    <section className={styles.slip} aria-label="Trial cost" data-testid="trial-cost">
      <h3 className={styles.slipTitle}>{title}</h3>
      <p className={styles.slipLead}>
        Adds <strong>{adds}</strong> {adds === 1 ? 'trial' : 'trials'} to{' '}
        <code>{impact.family}</code>
        {references
          ? `; ${references} ${references === 1 ? 'member re-runs' : 'members re-run'} a counted configuration (reference, not counted)`
          : ''}
        .
      </p>
      <dl className={styles.slipGrid}>
        <div>
          <dt>
            <Stat k="trials" />
          </dt>
          <dd>
            {int(impact.n_trials_raw)} → {int(impact.n_trials_raw + adds)}
          </dd>
        </div>
        <div>
          <dt>
            <Stat k="nEff" />
          </dt>
          <dd>
            {num(impact.n_eff_before, { digits: 1 })} → {num(impact.n_eff_after, { digits: 1 })}
          </dd>
        </div>
        <div>
          <dt>
            <Stat k="expMaxSr" />
          </dt>
          <dd>
            {sharpe(impact.exp_max_sr_before)} → {sharpe(impact.exp_max_sr_after)}
          </dd>
        </div>
        <div>
          <dt>
            <Stat k="dsr">DSR of the current best</Stat>
          </dt>
          <dd>
            {impact.best_sharpe === null ? (
              'no trial yet'
            ) : (
              <>
                {num(impact.dsr_best_before, { digits: 3 })} →{' '}
                {num(impact.dsr_best_after, { digits: 3 })}
                {cut !== null && cut > 0 ? (
                  <Badge tone="caution" className={styles.inlineBadge}>
                    −{num(cut, { digits: 3 })}
                  </Badge>
                ) : null}
              </>
            )}
          </dd>
        </div>
      </dl>
      {impact.best_sharpe !== null ? (
        <p className={styles.help}>
          The current best trial has net Sharpe {sharpe(impact.best_sharpe)}. New trials are counted
          as independent (the cautious direction).
        </p>
      ) : null}
      {children}
    </section>
  );
}
