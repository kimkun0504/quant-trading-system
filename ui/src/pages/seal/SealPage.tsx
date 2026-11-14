import { HANDBOOK_SEAL, useSeal } from '@/entities/research';
import { Stat, layout } from '@/features/research-launch';
import { date, dateTime, int, shortHash } from '@/shared/lib/format';
import {
  About,
  Badge,
  EmptyState,
  ErrorState,
  LinkButton,
  Page,
  Panel,
  Skeleton,
  Table,
  Td,
  Th,
} from '@/shared/ui';

/**
 * The research holdout seal (`qts seal status`): the seal, its inherited trial family,
 * every unseal so far (a forced one flagged "holdout already used") and the research runs
 * made under it, by family and kind.
 */
export function SealPage() {
  const seal = useSeal();
  return (
    <Page
      docTitle="Holdout seal"
      title="Holdout seal"
      actions={
        <LinkButton to="/builder" variant="secondary">
          Unseal from the run builder
        </LinkButton>
      }
    >
      <About label="About the seal" handbook={HANDBOOK_SEAL}>
        Every design choice of the baseline was made after seeing almost all of the store, so the
        last three years are sealed for new research: every research command loads data only through
        the day before the holdout. A configuration can be unsealed once, with a reason, to look at
        the holdout; that look is logged before the run starts and cannot be undone. Unseal only the
        final candidate.
      </About>
      {seal.isPending ? (
        <Skeleton height="16rem" />
      ) : seal.isError ? (
        <ErrorState error={seal.error} onRetry={() => void seal.refetch()} />
      ) : !seal.data.seal ? (
        <EmptyState title="No seal in force">
          research/seal.yaml (or the file at QTS_SEAL_FILE) does not exist: research commands load
          the whole history.
        </EmptyState>
      ) : (
        <SealBody data={seal.data} />
      )}
    </Page>
  );
}

function SealBody({ data }: { data: NonNullable<ReturnType<typeof useSeal>['data']> }) {
  const s = data.seal;
  if (!s) return null;
  const fam = s.inherit_family[0] ?? null;
  const n = fam ? data.inherited_families[fam] : null;
  const families = Object.entries(data.by_family).sort(([a], [b]) => a.localeCompare(b));
  return (
    <>
      <section className={layout.context} aria-label="The seal in force" data-testid="seal-context">
        <span>
          <span className={layout.contextLabel}>Holdout from</span>
          <br />
          <span className={layout.contextBig}>{date(s.holdout_start)}</span>
        </span>
        <span>
          <span className={layout.contextLabel}>Research data through</span>
          <br />
          <span className={layout.contextBig}>{date(data.data_through)}</span>
        </span>
        <span>
          <span className={layout.contextLabel}>
            <Stat k="family">Inherited family</Stat>
          </span>
          <br />
          <span className={layout.contextBig}>
            {n === null || n === undefined ? 'n/a' : `${int(n)} trials`}
          </span>
          <br />
          <code className={layout.small}>{fam ?? 'none'}</code>
        </span>
        <span>
          <span className={layout.contextLabel}>Unseals</span>
          <br />
          <span className={layout.contextBig}>{int(data.unseals.length)}</span>
          {data.n_forced ? (
            <>
              {' '}
              <Badge tone="fail">{data.n_forced} forced</Badge>
            </>
          ) : null}
        </span>
      </section>
      <div className={layout.row2}>
        <Panel title="The seal" subtitle={s.path}>
          <dl className={layout.facts}>
            <div>
              <dt>Sealed by</dt>
              <dd className={layout.small}>{s.sealed_by || 'n/a'}</dd>
            </div>
            <div>
              <dt>Sealed at</dt>
              <dd className={layout.small}>{date(s.sealed_at)}</dd>
            </div>
            <div>
              <dt>Store ends</dt>
              <dd className={layout.small}>{date(s.store_end)}</dd>
            </div>
            <div>
              <dt>Research runs under it</dt>
              <dd>{int(data.runs_under_seal)}</dd>
            </div>
          </dl>
          <p className={layout.lede}>{s.reason}</p>
          <p className={layout.lede}>
            A sealed run logs into the family of the full-history panel of the same load, so it
            counts the trials that already saw this history (
            {fam ? <code>{fam}</code> : 'no inherited family'}
            {n !== null && n !== undefined ? `: ${int(n)} trials` : ''}). A sealed re-run of an
            already counted configuration is a reference, not a new trial.
          </p>
        </Panel>
        <Panel title="Unseals" subtitle="each one used a configuration's holdout">
          {data.unseals.length === 0 ? (
            <p className={layout.lede}>
              No configuration has been unsealed: the holdout is unseen.
            </p>
          ) : (
            <Table
              label="Unseals"
              caption={`${data.unseals.length} unseal${data.unseals.length === 1 ? '' : 's'}, oldest first`}
            >
              <thead>
                <tr>
                  <Th>When</Th>
                  <Th>Configuration</Th>
                  <Th>Reason</Th>
                  <Th>Flag</Th>
                </tr>
              </thead>
              <tbody>
                {data.unseals.map((u, i) => (
                  <tr key={`${u.ts ?? ''}${i}`} data-forced={u.forced ? 'true' : 'false'}>
                    <Td>{dateTime(u.ts)}</Td>
                    <Td>
                      <code>{shortHash(u.config_hash, 12)}</code>
                      <br />
                      <span className={layout.small}>{u.family}</span>
                    </Td>
                    <Td className={layout.wrapCell}>{u.unseal_reason}</Td>
                    <Td>
                      {u.forced ? (
                        <Badge tone="fail">Holdout already used</Badge>
                      ) : (
                        <Badge tone="caution">First look</Badge>
                      )}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          )}
        </Panel>
      </div>
      <Panel title="Research runs under the seal" subtitle="by family and record kind">
        {families.length === 0 ? (
          <p className={layout.lede}>No research run has been made under the seal yet.</p>
        ) : (
          <Table
            label="Runs under the seal by family"
            caption="Only trials count towards a family's deflated Sharpe; references and the other kinds are logged, not counted."
          >
            <thead>
              <tr>
                <Th>Family</Th>
                <Th numeric>Trials</Th>
                <Th numeric>References</Th>
                <Th>Not trials</Th>
              </tr>
            </thead>
            <tbody>
              {families.map(([f, kinds]) => {
                const other = Object.entries(kinds).filter(
                  ([k]) => k !== 'trial' && k !== 'reference',
                );
                return (
                  <tr key={f}>
                    <Td>
                      <code>{f}</code>
                    </Td>
                    <Td numeric>{int(kinds.trial ?? 0)}</Td>
                    <Td numeric>{int(kinds.reference ?? 0)}</Td>
                    <Td>{other.length ? other.map(([k, v]) => `${k} ${v}`).join(', ') : 'none'}</Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </Panel>
      {data.exempt_runs.length ? (
        <Panel
          title="Exempt full-history runs"
          subtitle="by design: the pre-registered mid-cap check"
        >
          <ul className={layout.list}>
            {data.exempt_runs.map((e, i) => (
              <li key={`${e.ts ?? ''}${i}`}>
                {dateTime(e.ts)}: <code>{e.family}</code> ({e.seal_exempt})
              </li>
            ))}
          </ul>
        </Panel>
      ) : null}
    </>
  );
}
