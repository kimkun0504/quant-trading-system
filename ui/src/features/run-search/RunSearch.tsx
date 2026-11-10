import { forwardRef, useId, useMemo, useState } from 'react';
import type { KeyboardEvent } from 'react';
import { useNavigate } from 'react-router';
import { useRuns } from '@/entities/run';
import type { RunSummary } from '@/shared/api';
import { matchRuns } from './match';
import { dateTime, sharpe } from '@/shared/lib/format';
import { Kbd } from '@/shared/ui';
import { Icons } from '@/shared/ui/icons';
import styles from './RunSearch.module.css';

/** Global run search: an ARIA combobox over the runs list; Enter opens the run. */
export const RunSearch = forwardRef<HTMLInputElement>(function RunSearch(_, ref) {
  const runs = useRuns();
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();
  const matches = useMemo(() => matchRuns(runs.data ?? [], query), [runs.data, query]);
  const activeIdx = Math.min(active, Math.max(0, matches.length - 1));

  const choose = (r: RunSummary) => {
    setOpen(false);
    setQuery('');
    void navigate(`/runs/${encodeURIComponent(r.id)}`);
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
      const r = matches[activeIdx];
      if (open && r) {
        e.preventDefault();
        choose(r);
      }
    } else if (e.key === 'Escape') {
      if (open) {
        e.preventDefault();
        setOpen(false);
      } else e.currentTarget.blur();
    }
  };

  const expanded = open && Boolean(runs.data);
  return (
    <div className={styles.wrap}>
      <label className={styles.field}>
        <Icons.Search />
        <input
          ref={ref}
          type="search"
          role="combobox"
          aria-label="Search runs"
          aria-expanded={expanded}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={
            expanded && matches[activeIdx] ? `${listId}-${activeIdx}` : undefined
          }
          placeholder="Search runs"
          autoComplete="off"
          className={styles.input}
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
        <Kbd>/</Kbd>
      </label>
      {expanded ? (
        <div id={listId} role="listbox" aria-label="Runs" className={styles.list}>
          {matches.length === 0 ? (
            <div className={styles.empty}>No run matches “{query}”.</div>
          ) : (
            matches.map((r, i) => (
              // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/interactive-supports-focus -- APG combobox: focus stays on the input, aria-activedescendant points here
              <div
                key={r.id}
                id={`${listId}-${i}`}
                role="option"
                aria-selected={i === activeIdx}
                className={styles.option}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => choose(r)}
                onMouseEnter={() => setActive(i)}
              >
                <span className={styles.label}>{r.label}</span>
                <span className={styles.meta}>
                  {r.headline ? `${sharpe(r.headline.sharpe_net, { signed: true })} net, ` : ''}
                  {dateTime(r.created_at)}
                </span>
              </div>
            ))
          )}
        </div>
      ) : null}
    </div>
  );
});
