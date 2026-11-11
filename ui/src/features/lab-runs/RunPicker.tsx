import { useRuns, runDescriptor } from '@/entities/run';
import { MAX_SELECTED, toggleSelected, useSelection } from '@/features/selection';
import { date, sharpe } from '@/shared/lib/format';
import { ErrorState, Panel, Skeleton, Table, Td, Th } from '@/shared/ui';
import styles from './RunPicker.module.css';

/**
 * Shown by a lab when the tray is empty: every finished run with a checkbox. Ticking one
 * fills the tray, and the lab opens on it; tick several for small multiples.
 */
export function RunPicker({ what }: { what: string }) {
  const runs = useRuns();
  const ids = useSelection();
  const done = (runs.data ?? []).filter((r) => r.status === 'done');
  return (
    <Panel title="Pick the runs to study" subtitle={what}>
      <p className={styles.hint}>
        One run gives the deep view; two to {MAX_SELECTED} give small multiples side by side. The
        choice goes to the tray at the bottom, so every lab and Compare see the same runs.
      </p>
      {runs.isError ? (
        <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
      ) : runs.data ? (
        <Table label="Runs to pick from" caption={`${done.length} finished runs`}>
          <thead>
            <tr>
              <Th scope="col">
                <span className={styles.srOnly}>Select</span>
              </Th>
              <Th scope="col">Run</Th>
              <Th scope="col">Created</Th>
              <Th scope="col" numeric>
                Net Sharpe
              </Th>
            </tr>
          </thead>
          <tbody>
            {done.map((r) => {
              const checked = ids.includes(r.id);
              const full = !checked && ids.length >= MAX_SELECTED;
              return (
                <tr key={r.id}>
                  <Td>
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={full}
                      onChange={() => toggleSelected(r.id)}
                      aria-label={`Study ${r.label}`}
                    />
                  </Td>
                  <Td>
                    <span className={styles.label}>{r.label}</span>
                    <span className={styles.descriptor}>{runDescriptor(r)}</span>
                  </Td>
                  <Td>{date(r.created_at)}</Td>
                  <Td numeric>{sharpe(r.headline?.sharpe_net, { signed: true })}</Td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      ) : (
        <Skeleton height="12rem" />
      )}
    </Panel>
  );
}
