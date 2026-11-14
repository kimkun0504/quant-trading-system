import type { ReactNode } from 'react';
import type { ConfigValidation, ValidationIssue } from '@/shared/api';
import { formatLeaf } from '@/entities/run';
import { useTrialImpact } from '@/entities/research';
import { num, sharpe, shortHash } from '@/shared/lib/format';
import { cx } from '@/shared/lib/cx';
import { Badge, Button, CopyButton, Panel, Table, Td, Th } from '@/shared/ui';
import { VARIANT_MAX_KEYS, dataChanged, strategyDiffCount } from './lib/draft';
import type { Base, DiffRow } from './lib/draft';
import { cliLines, cliText } from './lib/cli';
import type { TrialForecast } from './lib/trials';
import { variantLetter } from './lib/reducer';
import styles from './LaunchSlip.module.css';

export interface SlipVariant {
  id: string;
  rows: DiffRow[];
  /** Client-side problems (fields and run settings) that block the launch. */
  clientErrors: number;
  validation: ConfigValidation | undefined;
  validating: boolean;
  cli: string[];
  trial: TrialForecast;
  label: string;
  /** Error of the last launch attempt of this variant. */
  launchError?: string | undefined;
}

type State = 'ready' | 'checking' | 'blocked';

/**
 * What the new trial(s) do to the deflated Sharpe of the family's current best trial
 * (`qts.research.sweep.trial_impact` through the API; nothing is computed here).
 */
function Haircut({ family, nNew }: { family: string; nNew: number }) {
  const q = useTrialImpact(family, nNew);
  const i = q.data;
  if (!i || i.best_sharpe === null || i.dsr_best_before === null) return null;
  return (
    <span data-testid="dsr-haircut">
      The family&apos;s current best (net Sharpe {sharpe(i.best_sharpe)}) goes from DSR{' '}
      {num(i.dsr_best_before, { digits: 3 })} to {num(i.dsr_best_after, { digits: 3 })}.
    </span>
  );
}

function variantState(v: SlipVariant): State {
  if (v.clientErrors > 0 || (v.validation && !v.validation.valid)) return 'blocked';
  if (v.validating || !v.validation) return 'checking';
  return 'ready';
}

function issueText(i: ValidationIssue): string {
  const where = i.loc
    .slice(1)
    .filter((p) => p !== 'config_overrides')
    .join('.');
  return where ? `${where}: ${i.msg}` : i.msg;
}

/** Keys that differ in any variant, one column per variant (the compare view's config diff). */
function DiffTable({
  base,
  variants,
  active,
}: {
  base: Base;
  variants: readonly SlipVariant[];
  active: string;
}) {
  const keys: string[] = [];
  for (const v of variants) for (const r of v.rows) if (!keys.includes(r.key)) keys.push(r.key);
  const fromOf = (key: string) =>
    variants.flatMap((v) => v.rows).find((r) => r.key === key)?.from ?? null;
  if (keys.length === 0)
    return (
      <p className={styles.same}>
        Nothing differs from the base yet. Queued as it is, this{' '}
        {base.runId ? 're-runs the base run' : 'runs the defaults'}.
      </p>
    );
  const single = variants.length === 1;
  return (
    <Table
      label="Changes from the base"
      caption={`${keys.length} ${keys.length === 1 ? 'key differs' : 'keys differ'} from the base.`}
      className={styles.diff}
    >
      <thead>
        <tr>
          <Th scope="col">Key</Th>
          <Th scope="col">Base</Th>
          {variants.map((v, i) => (
            <Th
              scope="col"
              key={v.id}
              className={cx(v.id === active && !single && styles.activeCol)}
            >
              {single ? 'This run' : `Variant ${variantLetter(i)}`}
            </Th>
          ))}
        </tr>
      </thead>
      <tbody>
        {keys.map((key) => {
          const from = fromOf(key);
          const cells = variants.map((v) => v.rows.find((r) => r.key === key));
          const decimal = [from, ...cells.map((c) => c?.to ?? null)].some(
            (x) => typeof x === 'number' && !Number.isInteger(x),
          );
          const show = (x: DiffRow['to']) =>
            x === null
              ? 'none'
              : typeof x === 'boolean'
                ? x
                  ? 'on'
                  : 'off'
                : formatLeaf(x, decimal);
          return (
            <tr key={key}>
              <Th scope="row" className={styles.key}>
                {key}
              </Th>
              <Td className={styles.from}>
                <code>{show(from)}</code>
              </Td>
              {cells.map((c, i) => (
                <Td
                  key={variants[i]?.id ?? i}
                  className={cx(c ? styles.to : styles.unchanged, c?.error && styles.bad)}
                >
                  {c ? (
                    <>
                      <code>{show(c.to)}</code>
                      {c.error ? <span className={styles.badWord}> not valid</span> : null}
                    </>
                  ) : (
                    <span aria-label="same as base">=</span>
                  )}
                </Td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </Table>
  );
}

function Line({ tone, children }: { tone: 'caution' | 'error' | 'info'; children: ReactNode }) {
  return (
    <li className={cx(styles.line, styles[tone])}>
      <span className={styles.mark} aria-hidden="true" />
      <span>{children}</span>
    </li>
  );
}

/**
 * The launch slip: what differs from the base, what the API says about it, which trial
 * it will be, and the command line that does the same. Sticky beside the editor, so the
 * diff is in view while a field is being changed.
 */
export function LaunchSlip({
  base,
  variants,
  active,
  launching,
  status,
  onLaunch,
}: {
  base: Base;
  variants: readonly SlipVariant[];
  active: string;
  launching: boolean;
  /** Result of the last launch, announced politely. */
  status: string;
  onLaunch: (ids: readonly string[]) => void;
}) {
  const at = Math.max(
    0,
    variants.findIndex((v) => v.id === active),
  );
  const cur = variants[at];
  if (!cur) return null;
  const letter = variantLetter(at);
  const state = variantState(cur);
  const nKeys = strategyDiffCount(cur.rows);
  const v = cur.validation;
  const serverErrors = (v?.errors ?? []).filter(
    (e) => e.loc[1] !== 'config_overrides' || e.loc.length < 3,
  );
  const allReady = variants.every((x) => variantState(x) === 'ready');
  const t = cur.trial;
  const many = variants.length > 1;

  return (
    <Panel
      as="aside"
      title="Before launch"
      subtitle={many ? `variant ${letter} of ${variants.length}` : undefined}
      aria-label="Before launch"
      flush
      className={styles.slip}
      actions={
        <Badge
          tone={state === 'ready' ? 'pass' : state === 'blocked' ? 'fail' : 'neutral'}
          role="status"
          data-testid="slip-state"
        >
          {state === 'ready' ? 'Valid' : state === 'blocked' ? 'Needs a fix' : 'Checking'}
        </Badge>
      }
    >
      <DiffTable base={base} variants={variants} active={active} />

      <dl className={styles.facts}>
        <div>
          <dt>Config hash</dt>
          <dd>
            {v?.config_hash ? (
              <>
                <code title={v.config_hash} data-testid="config-hash">
                  {shortHash(v.config_hash, 12)}
                </code>
                {v.base_config_hash && v.base_config_hash === v.config_hash ? (
                  <span className={styles.muted}> same as the base</span>
                ) : v.base_config_hash ? (
                  <span className={styles.muted}>
                    {' '}
                    base <code title={v.base_config_hash}>{shortHash(v.base_config_hash, 8)}</code>
                  </span>
                ) : null}
              </>
            ) : (
              <span className={styles.muted}>
                {state === 'blocked' ? 'none until the problems are fixed' : 'checking'}
              </span>
            )}
          </dd>
        </div>
        {v?.warmup_days !== null && v?.warmup_days !== undefined ? (
          <div>
            <dt>Warm-up</dt>
            <dd>{v.warmup_days.toLocaleString('en-US')} trading days before the first trade</dd>
          </div>
        ) : null}
      </dl>

      <ul className={styles.lines} aria-label="Checks">
        {cur.clientErrors > 0 ? (
          <Line tone="error">
            {cur.clientErrors === 1 ? 'One field needs' : `${cur.clientErrors} fields need`} a fix
            before this can be queued.
          </Line>
        ) : null}
        {serverErrors.map((e, i) => (
          <Line key={`e${i}`} tone="error">
            {issueText(e)}
          </Line>
        ))}
        {cur.launchError ? <Line tone="error">{cur.launchError}</Line> : null}
        {nKeys > VARIANT_MAX_KEYS ? (
          <Line tone="caution">
            {nKeys} keys differ: this is a new strategy, not a variant: comparisons with the base
            will not isolate one change.
          </Line>
        ) : null}
        {dataChanged(cur.rows) ? (
          <Line tone="caution">
            The data differs from the base, so the two runs are not comparable key for key and
            belong to different trial families.
          </Line>
        ) : null}
        {(v?.warnings ?? []).map((w, i) => (
          <Line key={`w${i}`} tone="caution">
            {issueText(w)}
          </Line>
        ))}
        <Line tone="info">
          {t.family && t.trialNumber !== null ? (
            <span data-testid="trial-line">
              This will be trial {t.trialNumber} in family <code>{t.family}</code>: its deflated
              Sharpe is judged against {t.trialNumber} {t.trialNumber === 1 ? 'trial' : 'trials'} (
              {t.nTrials} logged so far
              {t.trialNumber - t.nTrials > 1 ? ', plus the variants queued before it' : ''}).{' '}
              <Haircut family={t.family} nNew={t.trialNumber - t.nTrials} />
            </span>
          ) : (
            <span data-testid="trial-line">
              Every new configuration is a new trial. {t.unknownReason}
            </span>
          )}
        </Line>
      </ul>

      <div className={styles.cli}>
        <div className={styles.cliHead}>
          <span>Same run from the command line</span>
          <CopyButton text={cliText(cur.cli)} label="Copy the command" />
        </div>
        {/* eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- a scrollable region must be keyboard reachable */}
        <pre className={styles.pre} aria-label="Equivalent CLI command" tabIndex={0}>
          {cliLines(cur.cli).map((l, i, all) => (
            <span key={i} className={styles.cliLine}>
              {i ? '  ' : ''}
              {l}
              {i < all.length - 1 ? ' \\\n' : ''}
            </span>
          ))}
        </pre>
      </div>

      <div className={styles.actions}>
        {many ? (
          <>
            <Button
              variant="primary"
              disabled={launching || !allReady}
              onClick={() => onLaunch(variants.map((x) => x.id))}
            >
              {launching ? 'Adding to the queue…' : `Queue all ${variants.length} variants`}
            </Button>
            <Button disabled={launching || state !== 'ready'} onClick={() => onLaunch([cur.id])}>
              Queue variant {letter} only
            </Button>
          </>
        ) : (
          <Button
            variant="primary"
            disabled={launching || state !== 'ready'}
            onClick={() => onLaunch([cur.id])}
          >
            {launching ? 'Adding to the queue…' : 'Add to queue'}
          </Button>
        )}
        <span className={styles.status} role="status">
          {status}
        </span>
      </div>
      {many && !allReady ? (
        <p className={styles.blocked}>
          {variants
            .map((x, i) => (variantState(x) === 'blocked' ? `Variant ${variantLetter(i)}` : ''))
            .filter(Boolean)
            .join(', ') || 'A variant'}{' '}
          {variants.filter((x) => variantState(x) === 'blocked').length > 1 ? 'need' : 'needs'} a
          fix before all can be queued together.
        </p>
      ) : null}
    </Panel>
  );
}
