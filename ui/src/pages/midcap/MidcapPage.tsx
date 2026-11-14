import { useState } from 'react';
import { Link } from 'react-router';
import type { MidcapRequest, MidcapStatus } from '@/shared/api';
import { ApiError } from '@/shared/api';
import { useHealth } from '@/entities/run';
import {
  HANDBOOK_MIDCAP,
  useLaunchMidcap,
  useMidcap,
  usePreregistration,
} from '@/entities/research';
import {
  CommandLine,
  ConfirmDialog,
  Facts,
  MIDCAP_SLICE,
  StoreSliceFields,
  SwitchField,
  TaskJobs,
  TextField,
  layout,
  midcapCommand,
  sliceErrors,
  toSlice,
} from '@/features/research-launch';
import type { SliceDraft } from '@/features/research-launch';
import form from '@/features/research-launch/research.module.css';
import { dateTime, shortHash } from '@/shared/lib/format';
import {
  About,
  Badge,
  Button,
  EmptyState,
  ErrorState,
  Page,
  Panel,
  Skeleton,
  Table,
  Td,
  Th,
} from '@/shared/ui';
import { CoverageFacts } from './CoverageFacts';
import { outcomeText, outcomeTone } from './verdict';
import styles from './Midcap.module.css';

export function MidcapPage() {
  const q = useMidcap();
  return (
    <Page docTitle="Mid-cap check" title="Mid-cap check">
      <About label="About the mid-cap check" handbook={HANDBOOK_MIDCAP}>
        The frozen baseline was built and tuned on large caps. This check runs it once, unchanged,
        on the next tier of names (market-cap ranks 1001–2000), which it has never seen, and judges
        it by criteria written down before the run. Because it is out-of-sample only the first time,
        it runs once per configuration.
      </About>
      {q.isPending ? (
        <Skeleton height="20rem" />
      ) : q.isError ? (
        <ErrorState error={q.error} onRetry={() => void q.refetch()} />
      ) : (
        <Body m={q.data} />
      )}
    </Page>
  );
}

function Body({ m }: { m: MidcapStatus }) {
  const latest = m.results[0];
  return (
    <>
      <section className={layout.context} aria-label="Verdict" data-testid="midcap-verdict">
        <span>
          <span className={layout.contextLabel}>Verdict</span>
          <br />
          {latest ? (
            <Badge tone={outcomeTone(latest.outcome)}>{latest.outcome ?? 'none'}</Badge>
          ) : (
            <Badge tone="neutral">Not run yet</Badge>
          )}
          {latest?.forced ? (
            <>
              {' '}
              <Badge tone="caution">forced repeat</Badge>
            </>
          ) : null}
        </span>
        <span className={layout.lede}>
          {latest
            ? outcomeText(latest.outcome)
            : 'The check has not run for this configuration. Do a dry run first: it checks data coverage without running the backtest.'}
        </span>
        {latest ? (
          <span>
            <Link
              to={`/checks/midcap/${encodeURIComponent(latest.run_id)}`}
              className={layout.rowLink}
            >
              Open the result
            </Link>
          </span>
        ) : null}
      </section>
      <div className={layout.split}>
        <div className={layout.stack}>
          <Panel
            title="Pre-registered criteria"
            subtitle={`for ${m.config_path} (config ${shortHash(m.config_hash)})`}
          >
            <Table
              label="Pre-registered criteria"
              caption="C: the strategy works. V: the test was valid. S: reported, decides only the cost-fragile label."
            >
              <thead>
                <tr>
                  <Th>Criterion</Th>
                  <Th>Rule</Th>
                </tr>
              </thead>
              <tbody>
                {m.criteria.map((c) => (
                  <tr key={c.id}>
                    <Td>
                      <strong>{c.id}</strong>
                    </Td>
                    <Td className={layout.wrapCell}>{c.rule}</Td>
                  </tr>
                ))}
              </tbody>
            </Table>
            <p className={layout.lede}>
              PASS: C1–C3 pass and V1–V3 hold. PASS, cost-fragile: as PASS, but S1 (net Sharpe at
              twice the cost) is zero or below. FAIL: valid, and a C fails. INCONCLUSIVE: a V fails
              or a C cannot be measured.
            </p>
          </Panel>
          <Preregistration m={m} />
          <Panel
            title="Results on disk"
            subtitle="runs/_checks/ and each run's midcap_check.json"
            flush
          >
            {m.results.length === 0 ? (
              <EmptyState title="No result yet">
                A finished check writes its verdict here.
              </EmptyState>
            ) : (
              <Table label="Mid-cap results">
                <thead>
                  <tr>
                    <Th>Run</Th>
                    <Th>Outcome</Th>
                    <Th>Pre-registration</Th>
                    <Th>Family</Th>
                  </tr>
                </thead>
                <tbody>
                  {m.results.map((r) => (
                    <tr key={r.run_id}>
                      <Td>
                        <Link
                          to={`/checks/midcap/${encodeURIComponent(r.run_id)}`}
                          className={layout.rowLink}
                        >
                          <code>{r.run_id}</code>
                        </Link>
                      </Td>
                      <Td>
                        <Badge tone={outcomeTone(r.outcome)}>{r.outcome ?? 'none'}</Badge>
                        {r.forced ? <Badge tone="caution">forced</Badge> : null}
                      </Td>
                      <Td>
                        {r.sha_matches ? (
                          <Badge tone="pass">matches</Badge>
                        ) : (
                          <Badge tone="fail">changed since the run</Badge>
                        )}{' '}
                        <code>{shortHash(r.preregistration_sha256, 12)}</code>
                      </Td>
                      <Td>
                        <code>{r.family}</code>
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Panel>
          {m.previous_runs.length ? (
            <Panel title="Ledger records of this check">
              <ul className={layout.list}>
                {m.previous_runs.map((p, i) => (
                  <li key={i}>
                    {dateTime(typeof p.ts === 'string' ? p.ts : null)}: <code>{p.family}</code>
                    {p.forced ? <Badge tone="caution">forced</Badge> : null}
                  </li>
                ))}
              </ul>
            </Panel>
          ) : null}
        </div>
        <MidcapLaunch m={m} />
      </div>
    </>
  );
}

function Preregistration({ m }: { m: MidcapStatus }) {
  const [open, setOpen] = useState(false);
  const pre = usePreregistration(open);
  return (
    <Panel title="Pre-registration note" subtitle="written and hashed before the run">
      <Facts
        label="Pre-registration"
        facts={[
          {
            key: 'path',
            label: 'File',
            value: <code className={styles.path}>{m.preregistration}</code>,
          },
          {
            key: 'sha',
            label: 'sha256 now',
            value: <code className={styles.sha}>{m.preregistration_sha256 || 'missing'}</code>,
            note: 'every result records the hash it was judged against',
          },
        ]}
      />
      <button
        type="button"
        className={form.linkButton}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? 'Hide the note' : 'Read the note'}
      </button>
      {open ? (
        pre.data ? (
          // eslint-disable-next-line jsx-a11y/no-noninteractive-tabindex -- a scrollable region must be keyboard reachable
          <pre className={styles.note} tabIndex={0} aria-label="Pre-registration note">
            {pre.data.text}
          </pre>
        ) : pre.isError ? (
          <ErrorState error={pre.error} onRetry={() => void pre.refetch()} />
        ) : (
          <Skeleton height="8rem" />
        )
      ) : null}
    </Panel>
  );
}

function MidcapLaunch({ m }: { m: MidcapStatus }) {
  const health = useHealth();
  const storePath = health.data?.store_path ?? null;
  const [slice, setSlice] = useState<SliceDraft>(MIDCAP_SLICE);
  const [nav, setNav] = useState('5e6');
  const [cache, setCache] = useState(true);
  const [force, setForce] = useState(false);
  const [phrase, setPhrase] = useState('');
  const [confirm, setConfirm] = useState<'dry' | 'real' | null>(null);
  const [message, setMessage] = useState('');
  const launch = useLaunchMidcap();
  const busy = m.jobs.some((j) => j.status === 'queued' || j.status === 'running');
  const navN = Number(nav);
  const navErr = Number.isFinite(navN) && navN > 0 ? undefined : 'A NAV in USD above 0.';
  const valid = !navErr && Object.keys(sliceErrors(slice)).length === 0;
  const phraseOk = phrase.trim() === m.force_phrase;
  const body = (dry: boolean): MidcapRequest => ({
    store: toSlice(slice),
    nav: navN,
    cache,
    dry_run: dry,
    force: !dry && m.needs_force && force,
    confirm: !dry && m.needs_force && force ? phrase.trim() : null,
  });
  const realAllowed =
    valid && m.store_configured && !busy && (m.can_launch || (m.needs_force && force && phraseOk));
  const run = async (dry: boolean) => {
    try {
      const out = await launch.mutateAsync(body(dry));
      setMessage(`Queued: ${out.label}.`);
    } catch (e) {
      setMessage(e instanceof ApiError ? e.detail : 'The check could not be queued.');
    }
    setConfirm(null);
  };
  return (
    <Panel title="Run the check" subtitle="once per configuration">
      <div className={form.form}>
        {!m.can_launch && m.blocked_reason ? (
          <div className={layout.notice} role="status" data-testid="midcap-blocked">
            <span>
              <Badge tone="caution">Not available</Badge> {m.blocked_reason}
            </span>
          </div>
        ) : null}
        <StoreSliceFields value={slice} onChange={setSlice} storePath={storePath} />
        <div className={form.grid}>
          <TextField
            name="nav"
            label="NAV (USD)"
            numeric
            value={nav}
            onChange={setNav}
            error={navErr}
            help="Default 5e6 ($5M), as pre-registered."
          />
        </div>
        <SwitchField
          name="cache"
          label="Stage cache"
          checked={cache}
          onChange={setCache}
          help="Reuse fits of an identical earlier pass (--cache)."
        />
        {m.last_dry_run?.coverage ? (
          <section className={styles.dry} aria-label="Last dry run">
            <h3 className={form.slipTitle}>Last dry run</h3>
            <CoverageFacts c={m.last_dry_run.coverage} />
          </section>
        ) : null}
        <div className={form.footer}>
          <CommandLine command={midcapCommand(body(true), storePath)} note="The dry run." />
          <div className={form.actions}>
            <Button
              disabled={!valid || !m.store_configured || busy}
              onClick={() => setConfirm('dry')}
            >
              Dry run: check coverage…
            </Button>
          </div>
          <p className={form.help}>
            A dry run loads the data and reports coverage (V2) without the backtest. It is not a
            trial and can be repeated.
          </p>
          {m.needs_force ? (
            <fieldset className={`${form.fieldset} ${styles.force}`}>
              <legend className={form.legend}>Force a repeat</legend>
              <p className={form.help}>
                The check already ran for this configuration. A repeat is no longer out-of-sample:
                its record is flagged as forced and it must not be reported as a confirmation.
              </p>
              <SwitchField
                name="force"
                label="Force a repeat (flagged)"
                checked={force}
                onChange={setForce}
              />
              {force ? (
                <TextField
                  name="confirm"
                  label={`Type “${m.force_phrase}” to confirm`}
                  value={phrase}
                  onChange={setPhrase}
                  error={phrase && !phraseOk ? `Type exactly: ${m.force_phrase}` : undefined}
                />
              ) : null}
            </fieldset>
          ) : null}
          {m.can_launch || (m.needs_force && force) ? (
            <CommandLine command={midcapCommand(body(false), storePath)} />
          ) : null}
          <div className={form.actions}>
            <Button variant="primary" disabled={!realAllowed} onClick={() => setConfirm('real')}>
              {m.needs_force ? 'Run a forced repeat…' : 'Run the check…'}
            </Button>
          </div>
          {message ? (
            <p className={form.help} role="status">
              {message}
            </p>
          ) : null}
          <TaskJobs tasks={['midcap_check']} title="Mid-cap jobs" />
        </div>
      </div>
      <ConfirmDialog
        open={confirm !== null}
        title={
          confirm === 'dry'
            ? 'Run the coverage dry run?'
            : m.needs_force
              ? 'Force a repeat of the mid-cap check?'
              : 'Run the mid-cap check?'
        }
        confirmLabel={
          confirm === 'dry'
            ? 'Run the dry run'
            : m.needs_force
              ? 'Run the flagged repeat'
              : 'Run the check once'
        }
        tone={confirm === 'real' && m.needs_force ? 'danger' : 'plain'}
        busy={launch.isPending}
        onCancel={() => setConfirm(null)}
        onConfirm={() => void run(confirm === 'dry')}
      >
        {confirm === 'dry' ? (
          <p>Loads the mid-cap slice and checks coverage. Not a trial; nothing is logged.</p>
        ) : (
          <>
            <p>
              Adds <strong>1</strong> trial to its own confirmation family (
              <code>confirm:midcap:…</code>), judged once by the pre-registered criteria. It reads
              the full history, past the holdout seal, by design.
            </p>
            <p>
              {m.needs_force
                ? 'This configuration was checked before: the repeat is flagged as forced on its ledger record and cannot count as out-of-sample evidence.'
                : 'Once it runs, a second run of this configuration needs an explicit, flagged force.'}
            </p>
            <p>
              About one full pipeline run on roughly twice the names (plan an hour with a cold
              cache).
            </p>
          </>
        )}
      </ConfirmDialog>
    </Panel>
  );
}
