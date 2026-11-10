import { useDeferredValue, useId, useMemo, useState } from 'react';
import type { RunDetail } from '@/shared/api';
import { useCompareRuns, useRuns } from '@/entities/run';
import { ConfigDiff } from '@/features/compare';
import { About, EmptyState, ErrorState, Panel, Skeleton, Table, Td, Th } from '@/shared/ui';
import { DossierSection } from '../DossierSection';
import { groupLeaves } from '../lib/configGroups';
import styles from './Sections.module.css';

function Highlight({ text, query }: { text: string; query: string }) {
  const q = query.trim();
  if (!q) return <>{text}</>;
  const i = text.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, i)}
      <mark>{text.slice(i, i + q.length)}</mark>
      {text.slice(i + q.length)}
    </>
  );
}

export function ConfigSection({ run }: { run: RunDetail }) {
  const [query, setQuery] = useState('');
  const deferred = useDeferredValue(query);
  const [against, setAgainst] = useState('');
  const searchId = useId();
  const pickId = useId();
  const runs = useRuns();
  const ids = useMemo(() => (against ? [run.id, against] : [run.id]), [run.id, against]);
  const cmp = useCompareRuns(ids);

  const leaves = cmp.data?.runs.find((r) => r.id === run.id)?.config_leaves ?? null;
  const groups = useMemo(() => (leaves ? groupLeaves(leaves, deferred) : []), [leaves, deferred]);
  const total = leaves ? Object.keys(leaves).length : 0;
  const shown = groups.reduce((a, g) => a + g.rows.length, 0);
  const others = (runs.data ?? []).filter((r) => r.id !== run.id && r.status === 'done');

  return (
    <DossierSection
      id="config"
      lede={`Config ${run.manifest.config_hash.slice(0, 12)} on data ${run.manifest.data_hash.slice(0, 12)}, git ${run.manifest.git_sha.slice(0, 10)}${run.manifest.git_dirty ? ' (dirty)' : ''}; Python ${run.manifest.python}.`}
    >
      <Panel
        title="Diff against…"
        subtitle="only the keys that differ"
        flush
        actions={
          <label className={styles.search} htmlFor={pickId}>
            <span>Compare with</span>
            <select id={pickId} value={against} onChange={(e) => setAgainst(e.target.value)}>
              <option value="">Choose a run</option>
              {others.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
        }
      >
        {against ? (
          cmp.data && cmp.data.runs.length === 2 ? (
            <ConfigDiff runs={cmp.data.runs} />
          ) : cmp.isError ? (
            <ErrorState error={cmp.error} onRetry={() => void cmp.refetch()} />
          ) : (
            <Skeleton height="6rem" />
          )
        ) : (
          <p className={styles.muted} style={{ padding: 'var(--space-3) var(--space-4)' }}>
            Pick a run to see which keys differ between the two configurations.
          </p>
        )}
      </Panel>

      <Panel
        title="Configuration"
        subtitle={leaves ? `${shown} of ${total} keys` : undefined}
        flush
        actions={
          <label className={styles.search} htmlFor={searchId}>
            <span>Search</span>
            <input
              id={searchId}
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="key or value"
              aria-label="Search configuration keys and values"
            />
          </label>
        }
      >
        {leaves ? (
          groups.length ? (
            <Table
              label="Configuration"
              caption="Every scalar of the run's SystemConfig, grouped by section"
            >
              <thead>
                <tr>
                  <Th scope="col">Key</Th>
                  <Th scope="col">Value</Th>
                </tr>
              </thead>
              {groups.map((g) => (
                <tbody key={g.section}>
                  <tr className={styles.groupRow}>
                    <Th scope="colgroup" colSpan={2}>
                      {g.section}
                    </Th>
                  </tr>
                  {g.rows.map((r) => (
                    <tr key={r.key}>
                      <Th scope="row" className={styles.key}>
                        <Highlight text={r.leaf} query={deferred} />
                      </Th>
                      <Td className={styles.value}>
                        <code>
                          <Highlight text={r.value} query={deferred} />
                        </code>
                      </Td>
                    </tr>
                  ))}
                </tbody>
              ))}
            </Table>
          ) : (
            <EmptyState title={`No key or value contains “${deferred}”`}>
              <p>Keys are dotted paths such as portfolio.trade_aversion.</p>
            </EmptyState>
          )
        ) : cmp.isError ? (
          <ErrorState error={cmp.error} onRetry={() => void cmp.refetch()} />
        ) : (
          <Skeleton height="20rem" />
        )}
        <About
          label="About this table"
          handbook={{ page: 'glossary', section: 'SystemConfig' }}
          className={styles.inset}
        >
          Every behavioural knob lives in one frozen SystemConfig; its hash is the run's identity in
          the ledger. Lists are shown as compact JSON. The full YAML is in config.yaml under
          Artifacts.
        </About>
      </Panel>
    </DossierSection>
  );
}
