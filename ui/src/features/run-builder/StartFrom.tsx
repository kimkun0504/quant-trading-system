import { useId, useMemo, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { Link } from 'react-router';
import type { RunSummary } from '@/shared/api';
import { runDescriptor, useRuns } from '@/entities/run';
import { matchRuns } from '@/features/run-search';
import { dateTime, sharpe, shortHash } from '@/shared/lib/format';
import { Button, Panel, SegmentedControl } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import type { Base } from './lib/draft';
import styles from './StartFrom.module.css';

/** A searchable list of finished runs (ARIA combobox; Enter or a click picks one). */
function RunCombobox({
  runs,
  onPick,
}: {
  runs: readonly RunSummary[];
  onPick: (r: RunSummary) => void;
}) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();
  const matches = useMemo(() => matchRuns(runs, query, 8), [runs, query]);
  const idx = Math.min(active, Math.max(0, matches.length - 1));
  const choose = (r: RunSummary) => {
    setOpen(false);
    setQuery('');
    onPick(r);
  };
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(i + 1, matches.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === 'Enter') {
      const r = matches[idx];
      if (open && r) {
        e.preventDefault();
        choose(r);
      }
    } else if (e.key === 'Escape' && open) {
      e.preventDefault();
      setOpen(false);
    }
  };
  return (
    <div className={styles.combo}>
      <label className={styles.search}>
        <Icons.Search />
        <input
          type="search"
          role="combobox"
          aria-label="Find a run to start from"
          aria-expanded={open}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={open && matches[idx] ? `${listId}-${idx}` : undefined}
          placeholder="Search by label, id or config hash"
          autoComplete="off"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActive(0);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          onKeyDown={onKey}
        />
      </label>
      {open ? (
        <div id={listId} role="listbox" aria-label="Finished runs" className={styles.list}>
          {matches.length === 0 ? (
            <div className={styles.empty}>
              {runs.length === 0
                ? 'No finished run yet. Start from the defaults.'
                : `No finished run matches “${query}”.`}
            </div>
          ) : (
            matches.map((r, i) => (
              // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/interactive-supports-focus -- APG combobox: focus stays on the input, aria-activedescendant points here
              <div
                key={r.id}
                id={`${listId}-${i}`}
                role="option"
                aria-selected={i === idx}
                className={styles.option}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => choose(r)}
                onMouseEnter={() => setActive(i)}
              >
                <span className={styles.optionLabel}>{r.label}</span>
                <span className={styles.optionMeta}>
                  {r.headline ? `${sharpe(r.headline.sharpe_net, { signed: true })} net, ` : ''}
                  {runDescriptor(r)}, {dateTime(r.created_at)}
                </span>
              </div>
            ))
          )}
        </div>
      ) : null}
    </div>
  );
}

/**
 * Where the edit starts: the SystemConfig defaults, or a finished run (its config and
 * its recorded data source). Changing the base discards the edits, which were
 * differences from the old base; with edits on the bench it asks first.
 */
export function StartFrom({
  base,
  edits,
  onChange,
}: {
  base: Base;
  /** Edited keys across all variants. */
  edits: number;
  onChange: (runId: string | null) => void;
}) {
  const runs = useRuns();
  const done = useMemo(() => (runs.data ?? []).filter((r) => r.status === 'done'), [runs.data]);
  const [mode, setMode] = useState<'defaults' | 'run'>(base.runId ? 'run' : 'defaults');
  const [pending, setPending] = useState<{ runId: string | null; label: string } | null>(null);
  const request = (runId: string | null, label: string) => {
    if (runId === base.runId) return;
    if (edits > 0) setPending({ runId, label });
    else onChange(runId);
  };
  const summary = base.runId ? done.find((r) => r.id === base.runId) : undefined;

  return (
    <Panel title="Start from" className={styles.panel}>
      <div className={styles.row}>
        <SegmentedControl
          legend="Base configuration"
          value={mode}
          onChange={(m) => {
            setMode(m);
            if (m === 'defaults') request(null, 'the defaults');
          }}
          options={[
            { value: 'defaults', label: 'Defaults' },
            { value: 'run', label: 'An existing run' },
          ]}
        />
        {mode === 'run' ? <RunCombobox runs={done} onPick={(r) => request(r.id, r.label)} /> : null}
      </div>
      {pending ? (
        <div className={styles.confirm} role="alertdialog" aria-label="Discard edits?">
          <p>
            Starting from {pending.label} discards{' '}
            {edits === 1 ? 'the one edit' : `the ${edits} edits`} made here: they were differences
            from the current base.
          </p>
          <div className={styles.confirmActions}>
            <Button
              size="sm"
              variant="primary"
              onClick={() => {
                onChange(pending.runId);
                setPending(null);
              }}
            >
              Discard and switch
            </Button>
            <Button
              size="sm"
              onClick={() => {
                setPending(null);
                setMode(base.runId ? 'run' : 'defaults');
              }}
            >
              Keep editing
            </Button>
          </div>
        </div>
      ) : null}
      <p className={styles.base} data-testid="base-line">
        {base.runId ? (
          <>
            Base: <Link to={`/runs/${encodeURIComponent(base.runId)}`}>{base.label}</Link>
            {summary ? `, ${runDescriptor(summary).toLowerCase()}` : ''}, config{' '}
            <code title={base.configHash ?? ''}>{shortHash(base.configHash, 12)}</code>.{' '}
            {base.sourceRecorded
              ? 'Its config and data source are loaded below.'
              : 'This run did not record where its data came from: check the data settings before you queue it.'}
          </>
        ) : mode === 'run' ? (
          <>Pick a finished run; until then the base is the SystemConfig defaults.</>
        ) : (
          <>
            Base: the <code>SystemConfig</code> defaults on a synthetic market, like{' '}
            <code>qts demo</code>.
          </>
        )}
      </p>
    </Panel>
  );
}
