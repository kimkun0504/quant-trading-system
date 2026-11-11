import type { ReactNode } from 'react';
import { Link } from 'react-router';
import { EmptyState, ErrorState, Page, Skeleton } from '@/shared/ui';
import { RunPicker } from './RunPicker';
import { useLabRuns } from './useLabRuns';
import type { LabRun } from './useLabRuns';
import styles from './LabPage.module.css';

/**
 * The frame every lab shares: a title, the run strip (the tray's runs with their series
 * swatches) and the picker when nothing is selected. `children` renders the lab for the
 * selected runs: one run is the deep view, several are small multiples.
 */
export function LabPage({
  title,
  docTitle,
  what,
  children,
}: {
  title: string;
  docTitle: string;
  /** One line on what the lab shows; also the picker's subtitle. */
  what: string;
  children: (runs: readonly LabRun[]) => ReactNode;
}) {
  const lab = useLabRuns();
  let body: ReactNode;
  if (lab.ids.length === 0) body = <RunPicker what={what} />;
  else if (lab.error)
    body = <ErrorState error={lab.error} onRetry={() => void lab.query.refetch()} />;
  else if (lab.isLoading) body = <Skeleton height="20rem" />;
  else if (lab.runs.length === 0)
    body = (
      <EmptyState title="None of the selected runs has results yet">
        <p>
          The tray holds {lab.missing.length} run{lab.missing.length === 1 ? '' : 's'} that{' '}
          {lab.missing.length === 1 ? 'is' : 'are'} still running, failed or unknown. Pick a
          finished run in <Link to="/">Experiments</Link>.
        </p>
      </EmptyState>
    );
  else
    body = (
      <>
        {lab.missing.length ? (
          <p className={styles.missing} role="status">
            {lab.missing.length} selected run{lab.missing.length === 1 ? ' is' : 's are'} not
            finished and {lab.missing.length === 1 ? 'is' : 'are'} left out.
          </p>
        ) : null}
        {children(lab.runs)}
      </>
    );
  return (
    <Page
      docTitle={docTitle}
      title={title}
      actions={lab.runs.length ? <RunStrip runs={lab.runs} /> : undefined}
    >
      {body}
    </Page>
  );
}

/** The selected runs with their swatches, as the key to every small multiple. */
export function RunStrip({ runs }: { runs: readonly LabRun[] }) {
  return (
    <ul className={styles.strip} aria-label="Runs in this lab">
      {runs.map((r) => (
        <li key={r.id} className={styles.stripItem}>
          <RunSwatch index={r.index} />
          <Link to={`/runs/${encodeURIComponent(r.id)}`}>{r.label}</Link>
        </li>
      ))}
    </ul>
  );
}

export function RunSwatch({ index }: { index: number }) {
  return (
    <span
      className={styles.swatch}
      style={{ background: `var(--color-series-${(index % 8) + 1})` }}
      aria-hidden="true"
    />
  );
}

/**
 * One panel per run, side by side (three across at most; one run fills the width).
 * `render` gets the run and whether it is the only one (the deep view can be taller).
 */
export function RunLanes({
  runs,
  render,
}: {
  runs: readonly LabRun[];
  render: (run: LabRun, solo: boolean) => ReactNode;
}) {
  const cols = Math.min(3, Math.max(1, runs.length));
  return (
    <div className={styles.lanes} style={{ '--cols': cols } as React.CSSProperties}>
      {runs.map((r) => (
        <section key={r.id} className={styles.lane} aria-label={r.label}>
          {runs.length > 1 ? (
            <h3 className={styles.laneTitle}>
              <RunSwatch index={r.index} />
              {r.label}
            </h3>
          ) : null}
          {render(r, runs.length === 1)}
        </section>
      ))}
    </div>
  );
}
